"""
Bifurcaciones de sistemas dinámicos de una dimensión
---------------------------------------------------------
Analiza bifurcaciones de sistemas autónomos 1D dy/dx = f(y; mu), con mu un
parámetro de control. El usuario ingresa la ecuación f(y; mu) y el script
detecta solo el (o los) punto(s) de bifurcación y clasifica automáticamente
de qué tipo es, entre los 3 casos clásicos (formas normales):
  - Silla-nodo (saddle-node):  f(y; mu) = mu - y**2
  - Transcrítica:               f(y; mu) = mu*y - y**2
  - Horquilla (pitchfork):      f(y; mu) = mu*y -+ y**3 (super/subcrítica)
o informa que la bifurcación no es ninguno de esos casos genéricos.

El script:
  - Pide f(y; mu) tal que dy/dx = f(y; mu) (sintaxis sympy, variables y y mu).
  - Busca los puntos de bifurcación: los (y0, mu0) donde f = 0 y f'(y) = 0
    simultáneamente (dos equilibrios que colisionan), simbólicamente con
    sympy.solve y, si no se puede en forma cerrada, numéricamente (rastrea
    cambios en la cantidad de equilibrios reales a lo largo de mu y refina
    con sympy.nsolve).
  - Clasifica cada punto de bifurcación con las condiciones de
    transversalidad de Sotomayor, evaluando ahí f_mu, f_yy, f_y,mu y f_yyy:
      f_mu != 0, f_yy != 0                          -> Silla-nodo
      f_mu  = 0, f_yy != 0, f_y,mu != 0              -> Transcrítica
      f_mu  = 0, f_yy  = 0, f_y,mu != 0, f_yyy != 0  -> Horquilla
        (supercrítica si f_y,mu * f_yyy < 0, subcrítica si > 0)
  - Calcula los puntos de equilibrio y_eq(mu) resolviendo f(y; mu) = 0 para
    y (simbólico), en función del parámetro mu.
  - Hace el estudio de estabilidad con el criterio de la primera derivada:
    calcula f'(y) y lo evalúa en cada rama y_eq(mu), muestreando su signo a
    lo largo del rango de mu para determinar en qué tramos cada rama es
    estable o inestable.
  - Grafica el diagrama de bifurcación (y_eq en función de mu): cada tramo
    estable en línea verde continua, cada tramo inestable en línea roja
    punteada, y el/los punto(s) de bifurcación marcados.
  - Grafica el diagrama de fase (línea de fase) para 3 valores de mu
    representativos -antes, en, y después de cada punto de bifurcación-,
    con el mismo criterio de la primera derivada y el mismo código de
    colores que sistemas_dinamicos.py (verde = estable, rojo = inestable,
    naranja = semiestable).

Requiere: sympy, numpy, matplotlib, tabulate
    pip install sympy numpy matplotlib tabulate
"""

import os
import itertools

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate

CARPETA_GRAFICOS = "graficos"
TOLERANCIA_CERO = 1e-7  # para las condiciones de Sotomayor (derivadas "= 0")

mu, y = sp.symbols('mu y', real=True)


# ----------------------------------------------------------------------
# Entrada de datos
# ----------------------------------------------------------------------

def pedir_funcion():
    while True:
        texto = input(
            "\nIngresá f(y; mu) tal que dy/dx = f(y; mu) (sintaxis sympy, "
            "variables y y mu; ej: mu - y**2, mu*y - y**2, mu*y - y**3): "
        ).strip()
        try:
            f_expr = sp.sympify(texto, locals={'y': y, 'mu': mu})
        except (sp.SympifyError, TypeError):
            print("  -> Expresión inválida, probá de nuevo.")
            continue

        libres = f_expr.free_symbols
        if libres - {y, mu}:
            otras = ", ".join(str(s) for s in libres - {y, mu})
            print(f"  -> La expresión no puede depender de {otras}: usá solo "
                  "las variables y (estado) y mu (parámetro).")
            continue
        if mu not in libres:
            print("  -> La expresión no depende de mu: no hay parámetro de "
                  "bifurcación, probá de nuevo.")
            continue
        if y not in libres:
            print("  -> La expresión no depende de y: no es una ecuación "
                  "diferencial en y, probá de nuevo.")
            continue
        return f_expr


def pedir_valor_opcional(mensaje, valor_por_defecto):
    texto = input(mensaje).strip()
    if texto == "":
        return valor_por_defecto
    try:
        return float(sp.N(sp.sympify(texto)))
    except (sp.SympifyError, TypeError, ValueError):
        print(f"  -> Valor inválido, se usa el valor por defecto ({valor_por_defecto}).")
        return valor_por_defecto


# ----------------------------------------------------------------------
# Búsqueda numérica de raíces (fallback para equilibrios a mu fijo)
# ----------------------------------------------------------------------

def _evaluar_seguro(f_num, v):
    try:
        resultado = float(f_num(v))
    except (ValueError, OverflowError, ZeroDivisionError, TypeError):
        return None
    return resultado if np.isfinite(resultado) else None


def buscar_raices(g, a, b, n=800):
    """Busca raíces de g(v) = 0 en [a, b] por cambio de signo + bisección.
    g puede devolver None donde no esté definida (se ignora esa muestra)."""
    malla = np.linspace(a, b, n)
    valores = [g(v) for v in malla]

    raices = []
    for i in range(n - 1):
        v0, v1 = valores[i], valores[i + 1]
        if v0 is None or v1 is None:
            continue
        if v0 == 0:
            raices.append(malla[i])
            continue
        if v0 * v1 < 0:
            izq, der = malla[i], malla[i + 1]
            for _ in range(60):
                medio = (izq + der) / 2
                v_izq, v_medio = g(izq), g(medio)
                if v_izq is None or v_medio is None:
                    break
                if v_izq * v_medio <= 0:
                    der = medio
                else:
                    izq = medio
            raices.append((izq + der) / 2)

    raices.sort()
    tolerancia = max((b - a) * 1e-4, 1e-9)
    filtradas = []
    for r in raices:
        if not filtradas or abs(r - filtradas[-1]) > tolerancia:
            filtradas.append(r)
    return filtradas


# ----------------------------------------------------------------------
# Detección y clasificación de los puntos de bifurcación
# ----------------------------------------------------------------------

def encontrar_puntos_bifurcacion(f_expr):
    """Busca puntos (y0, mu0) donde f = 0 y f'(y) = 0 a la vez (dos
    equilibrios que colisionan): son los candidatos a punto de
    bifurcación. Simbólico con sympy.solve; devuelve [] si no se puede
    resolver en forma cerrada."""
    if not f_expr.is_polynomial(y, mu):
        return []  # transcendente: sp.solve puede no terminar, se usa el fallback numérico

    df_dy = sp.diff(f_expr, y)
    try:
        soluciones = sp.solve([sp.Eq(f_expr, 0), sp.Eq(df_dy, 0)], [y, mu], dict=True)
    except NotImplementedError:
        soluciones = []

    puntos, vistos = [], set()
    for sol in soluciones:
        if y not in sol or mu not in sol:
            continue
        try:
            y0c = complex(sp.N(sol[y]))
            mu0c = complex(sp.N(sol[mu]))
        except (TypeError, ValueError):
            continue
        if abs(y0c.imag) > 1e-9 or abs(mu0c.imag) > 1e-9:
            continue
        clave = (round(y0c.real, 9), round(mu0c.real, 9))
        if clave in vistos:
            continue
        vistos.add(clave)
        puntos.append((sp.nsimplify(y0c.real, rational=False),
                        sp.nsimplify(mu0c.real, rational=False)))

    puntos.sort(key=lambda p: float(sp.N(p[1])))
    return puntos


def buscar_puntos_bifurcacion_numerico(f_expr, mu_min, mu_max, n=300):
    """Fallback numérico: recorre mu, cuenta cuántos equilibrios reales hay
    en cada paso y, donde ese número cambia (dos equilibrios se juntan o
    se separan), refina con sympy.nsolve el punto (y0, mu0) donde f = 0 y
    f'(y) = 0 simultáneamente."""
    df_dy = sp.diff(f_expr, y)
    mu_grid = np.linspace(mu_min, mu_max, n)

    conteos, raices_por_mu = [], []
    for m in mu_grid:
        raices, _ = calcular_equilibrios_en_mu(f_expr, m)
        raices_por_mu.append(raices)
        conteos.append(len(raices))

    puntos, vistos = [], set()
    for i in range(n - 1):
        if conteos[i] == conteos[i + 1]:
            continue
        m_seed = (mu_grid[i] + mu_grid[i + 1]) / 2
        candidatas = raices_por_mu[i] or raices_por_mu[i + 1] or [0.0]
        for y_seed in candidatas:
            try:
                sol = sp.nsolve([f_expr, df_dy], [y, mu], [y_seed, m_seed])
                y0, mu0 = float(sol[0]), float(sol[1])
            except Exception:
                continue
            if not (mu_min - 1e-6 <= mu0 <= mu_max + 1e-6):
                continue
            clave = (round(y0, 6), round(mu0, 6))
            if clave in vistos:
                continue
            vistos.add(clave)
            puntos.append((sp.Float(y0), sp.Float(mu0)))

    puntos.sort(key=lambda p: float(p[1]))
    return puntos


def clasificar_bifurcacion(f_expr, y0, mu0):
    """Clasifica el punto de bifurcación (y0, mu0) con las condiciones de
    transversalidad de Sotomayor, evaluando ahí las derivadas de f."""
    df_dyy = sp.diff(f_expr, y, 2)
    df_dyyy = sp.diff(f_expr, y, 3)
    df_dmu = sp.diff(f_expr, mu)
    df_dymu = sp.diff(f_expr, y, mu)

    subs = {y: y0, mu: mu0}
    a = complex(sp.N(df_dmu.subs(subs))).real   # f_mu
    b = complex(sp.N(df_dyy.subs(subs))).real   # f_yy
    c = complex(sp.N(df_dymu.subs(subs))).real  # f_y,mu
    d = complex(sp.N(df_dyyy.subs(subs))).real  # f_yyy
    tol = TOLERANCIA_CERO

    if abs(a) > tol and abs(b) > tol:
        return "Silla-nodo (saddle-node)"
    if abs(a) <= tol and abs(b) > tol and abs(c) > tol:
        return "Transcrítica"
    if abs(a) <= tol and abs(b) <= tol and abs(c) > tol and abs(d) > tol:
        return "Horquilla " + ("supercrítica" if c * d < 0 else "subcrítica")
    return "No clasificable con los criterios estándar (bifurcación degenerada o de orden superior)"


# ----------------------------------------------------------------------
# Ramas de equilibrio y_eq(mu) y estabilidad a lo largo de mu
# ----------------------------------------------------------------------

def calcular_ramas(f_expr):
    """Resuelve f(y; mu) = 0 para y: cada solución es una rama de
    equilibrio y_eq(mu), expresión simbólica en mu."""
    if not f_expr.is_polynomial(y):
        return []  # transcendente: sp.solve puede no terminar, no hay forma cerrada

    try:
        soluciones = sp.solve(sp.Eq(f_expr, 0), y)
    except NotImplementedError:
        soluciones = []

    ramas, vistas = [], set()
    for s in soluciones:
        s = sp.simplify(s)
        clave = sp.srepr(s)
        if clave not in vistas:
            vistas.add(clave)
            ramas.append(s)
    return ramas


def _evaluar_rama(func, valor):
    """Evalúa una rama (o su derivada) lambdificada en un valor de mu,
    descartando resultados complejos o no finitos (la rama no es real
    para ese mu, ej. sqrt(mu) con mu < 0)."""
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            resultado = complex(func(valor))
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    if abs(resultado.imag) > 1e-9:
        return None
    resultado = resultado.real
    return resultado if np.isfinite(resultado) else None


def clasificar_signo(valor):
    if valor is None:
        return None
    if valor < -1e-9:
        return "Estable"
    if valor > 1e-9:
        return "Inestable"
    return None  # f'(y_eq) = 0 exacto: punto de transición, no se colorea


def construir_tramos(estados):
    """Agrupa corridas contiguas de un mismo estado ('Estable'/'Inestable')
    en tramos de índices (i_inicio, i_fin, estado) sobre la grilla de mu."""
    tramos = []
    for estado, grupo in itertools.groupby(enumerate(estados), key=lambda par: par[1]):
        if estado not in ("Estable", "Inestable"):
            continue
        indices = [i for i, _ in grupo]
        tramos.append((indices[0], indices[-1], estado))
    return tramos


def analizar_ramas(f_expr, ramas, mu_min, mu_max, n=600):
    """Para cada rama y_eq(mu): evalúa f'(y_eq(mu)) (criterio de la primera
    derivada) a lo largo de la grilla de mu y arma los tramos estables/
    inestables. Devuelve (f'(y) general, lista de dicts con la info)."""
    df_expr = sp.diff(f_expr, y)
    mu_grid = np.linspace(mu_min, mu_max, n)

    info_ramas = []
    for rama in ramas:
        y_func = sp.lambdify(mu, rama, "numpy")
        d_expr = sp.simplify(df_expr.subs(y, rama))
        d_func = sp.lambdify(mu, d_expr, "numpy")

        y_vals = [_evaluar_rama(y_func, m) for m in mu_grid]
        d_vals = [_evaluar_rama(d_func, m) for m in mu_grid]
        estados = [clasificar_signo(d) if yv is not None else None
                   for yv, d in zip(y_vals, d_vals)]

        info_ramas.append({
            "expr": rama,
            "d_expr": d_expr,
            "mu_grid": mu_grid,
            "y_vals": y_vals,
            "tramos": construir_tramos(estados),
        })
    return df_expr, info_ramas


def mostrar_estabilidad_ramas(df_expr, info_ramas):
    print(f"\nDerivada: f'(y) = {df_expr}")
    print("\nEstudio de estabilidad por rama (criterio de la primera derivada):")
    filas = []
    for info in info_ramas:
        print(f"\n  Rama y_eq(mu) = {info['expr']}   ->   f'(y_eq) = {info['d_expr']}")
        if not info["tramos"]:
            print("    (sin equilibrio real en el rango, o f'(y_eq) = 0 en todo el rango)")
        for i0, i1, estado in info["tramos"]:
            m0, m1 = info["mu_grid"][i0], info["mu_grid"][i1]
            print(f"    {estado} para mu en [{m0:.4g}, {m1:.4g}]")
            filas.append([str(info["expr"]), estado, f"[{m0:.4g}, {m1:.4g}]"])
    if filas:
        print()
        print(tabulate(filas, headers=["y_eq(mu)", "Estado", "Rango de mu"], tablefmt="grid"))


def rango_y_equilibrios(info_ramas, margen_relativo=0.3, minimo=1.0):
    valores = [v for info in info_ramas for v in info["y_vals"] if v is not None]
    if not valores:
        return -minimo, minimo
    y_min, y_max = min(valores), max(valores)
    if y_min == y_max:
        y_min, y_max = y_min - minimo / 2, y_max + minimo / 2
    margen = max((y_max - y_min) * margen_relativo, minimo * 0.3)
    return y_min - margen, y_max + margen


# ----------------------------------------------------------------------
# Gráfico 1: diagrama de bifurcación (y_eq en función de mu)
# ----------------------------------------------------------------------

def graficar_diagrama_bifurcacion(info_ramas, mu_min, mu_max, nombre_tipo, puntos):
    plt.figure(figsize=(9, 7))

    for info in info_ramas:
        mu_grid, y_vals = info["mu_grid"], info["y_vals"]
        for i0, i1, estado in info["tramos"]:
            xs = mu_grid[i0:i1 + 1]
            ys = y_vals[i0:i1 + 1]
            color = "green" if estado == "Estable" else "red"
            estilo = "-" if estado == "Estable" else "--"
            plt.plot(xs, ys, color=color, linestyle=estilo, linewidth=2.3, zorder=2)

    for i, (y0, mu0) in enumerate(puntos, start=1):
        y0_num, mu0_num = float(y0), float(mu0)
        plt.axvline(mu0_num, color="gray", linestyle=":", linewidth=1, zorder=1)
        plt.plot(mu0_num, y0_num, marker="o", color="black", markersize=8, zorder=3)
        etiqueta = "Punto de\nbifurcación" if len(puntos) == 1 else f"Bifurcación {i}"
        plt.annotate(etiqueta, xy=(mu0_num, y0_num), xytext=(8, 8),
                     textcoords="offset points", fontsize=8)

    handles = [
        Line2D([0], [0], color="green", lw=2.3, label="Estable"),
        Line2D([0], [0], color="red", lw=2.3, linestyle="--", label="Inestable"),
    ]
    plt.legend(handles=handles, loc="best")
    plt.xlabel("mu")
    plt.ylabel("y_eq")
    plt.title(f"Diagrama de bifurcación: {nombre_tipo}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(mu_min, mu_max)
    plt.tight_layout()

    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    ruta = os.path.join(CARPETA_GRAFICOS, "bifurcaciones_diagrama.png")
    plt.savefig(ruta, dpi=150)
    print(f"\nGráfico guardado como '{ruta}'")


# ----------------------------------------------------------------------
# Equilibrios y estabilidad a mu fijo (para los diagramas de fase)
# ----------------------------------------------------------------------

def calcular_equilibrios_en_mu(f_expr, mu_val):
    """Para un valor numérico de mu, resuelve f(y; mu) = 0 (simbólico, con
    fallback numérico por bisección) y devuelve (equilibrios, f_num)."""
    f_mu = sp.simplify(f_expr.subs(mu, mu_val))
    f_num = sp.lambdify(y, f_mu, "numpy")
    if y not in f_mu.free_symbols:
        return [], f_num

    soluciones = []
    if f_mu.is_polynomial(y):
        try:
            soluciones = sp.solve(sp.Eq(f_mu, 0), y)
        except NotImplementedError:
            soluciones = []

    reales = []
    for s in soluciones:
        try:
            valor_complejo = complex(sp.N(s))
        except (TypeError, ValueError):
            continue
        if abs(valor_complejo.imag) < 1e-9:
            reales.append(valor_complejo.real)

    if not reales:
        reales = buscar_raices(lambda v: _evaluar_seguro(f_num, v), -25, 25, n=3000)

    # deduplicar valores muy cercanos (ej. raíz doble justo en el punto crítico)
    reales.sort()
    filtradas = []
    for r in reales:
        if not filtradas or abs(r - filtradas[-1]) > 1e-6:
            filtradas.append(r)
    return filtradas, f_num


def estudiar_estabilidad_en_mu(df_expr, mu_val, equilibrios):
    info = []
    for yeq in equilibrios:
        valor = complex(sp.N(df_expr.subs({mu: mu_val, y: yeq})))
        valor_num = valor.real
        if valor_num < -1e-9:
            estado = "Estable"
        elif valor_num > 1e-9:
            estado = "Inestable"
        else:
            estado = "Indeterminado"
        info.append({"y_eq_num": yeq, "derivada_valor_num": valor_num, "estado": estado})
    return info


def reclasificar_indeterminados(f_num, info_estabilidad, y_min, y_max):
    """Para los equilibrios donde f'(y_eq) = 0, el criterio de la primera
    derivada no decide: se mira el signo de f(y) a cada lado del punto."""
    delta = max((y_max - y_min) * 0.01, 1e-4)
    for inf in info_estabilidad:
        if inf["estado"] != "Indeterminado":
            continue
        izq = _evaluar_seguro(f_num, inf["y_eq_num"] - delta)
        der = _evaluar_seguro(f_num, inf["y_eq_num"] + delta)
        if izq is None or der is None:
            inf["estado"] = "Indeterminado"
        elif izq > 0 > der:
            inf["estado"] = "Estable"
        elif izq < 0 < der:
            inf["estado"] = "Inestable"
        else:
            inf["estado"] = "Semiestable"


def color_estado(estado):
    if estado == "Estable":
        return "green"
    if estado == "Inestable":
        return "red"
    return "orange"


# ----------------------------------------------------------------------
# Flechas de dirección de flujo (estilo campo vectorial 1D)
# ----------------------------------------------------------------------

def puntos_de_flechas(y_min, y_max, y_eqs, cantidad=14):
    """Centros donde dibujar flechas de dirección de flujo, espaciadas
    regularmente a lo largo de todo el rango [y_min, y_max], salteando
    los puntos de equilibrio."""
    paso = (y_max - y_min) / cantidad
    if paso <= 0:
        return []
    buffer = paso * 0.5
    puntos = []
    v = y_min + paso / 2
    while v < y_max:
        if all(abs(v - yeq) > buffer for yeq in y_eqs):
            puntos.append(v)
        v += paso
    return puntos


# ----------------------------------------------------------------------
# Gráfico 2: diagramas de fase (línea de fase) para 3 valores de mu
# ----------------------------------------------------------------------

def graficar_fase_subplot(ax, f_num, info_estabilidad, y_min, y_max, titulo):
    y_eqs = sorted(inf["y_eq_num"] for inf in info_estabilidad)

    ax.axvline(0, color="black", linewidth=1.2, zorder=1)

    cantidad_flechas = 14
    largo_flecha = (y_max - y_min) / cantidad_flechas * 0.65
    for centro in puntos_de_flechas(y_min, y_max, y_eqs, cantidad_flechas):
        signo = _evaluar_seguro(f_num, centro)
        if signo is None:
            continue
        y0, y1 = centro - largo_flecha / 2, centro + largo_flecha / 2
        if signo < 0:
            y0, y1 = y1, y0
        ax.annotate("", xy=(0, y1), xytext=(0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=2), zorder=2)

    for inf in info_estabilidad:
        color = color_estado(inf["estado"])
        ax.plot(0, inf["y_eq_num"], marker="o", markersize=14, markerfacecolor="white",
                 markeredgecolor=color, markeredgewidth=2.6, zorder=3)
        ax.annotate(f'y = {inf["y_eq_num"]:.4g}\n({inf["estado"]})',
                    xy=(0.12, inf["y_eq_num"]), va="center", fontsize=8)

    ax.set_xlim(-1, 1)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_ylabel("y")
    ax.set_title(titulo, fontsize=10)


def graficar_fases_tres_mu(f_expr, df_expr, mu0, delta, y_min, y_max, nombre_tipo, sufijo=""):
    mu_valores = [
        (mu0 - delta, "Antes de la bifurcación", "antes de la bifurcación"),
        (mu0, "En la bifurcación", "en la bifurcación"),
        (mu0 + delta, "Después de la bifurcación", "después de la bifurcación"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 6.5), sharey=True)
    for ax, (mu_val, titulo_corto, frase) in zip(axes, mu_valores):
        equilibrios, f_num = calcular_equilibrios_en_mu(f_expr, mu_val)
        info = estudiar_estabilidad_en_mu(df_expr, mu_val, equilibrios)
        if any(inf["estado"] == "Indeterminado" for inf in info):
            reclasificar_indeterminados(f_num, info, y_min, y_max)

        titulo = f"{titulo_corto}\nmu = {mu_val:g}"
        graficar_fase_subplot(ax, f_num, info, y_min, y_max, titulo)

        print(f"\nmu = {mu_val:g} ({frase}, mu0 = {mu0:g}):")
        if not info:
            print("    No hay puntos de equilibrio reales.")
        for inf in info:
            print(f"    y = {inf['y_eq_num']:.4g}  ->  {inf['estado']}")

    fig.suptitle(f"Diagramas de fase - {nombre_tipo} (mu0 = {mu0:g})", fontsize=12)
    plt.tight_layout()

    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    ruta = os.path.join(CARPETA_GRAFICOS, f"bifurcaciones_fases{sufijo}.png")
    plt.savefig(ruta, dpi=150)
    print(f"\nGráfico guardado como '{ruta}'")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=== Bifurcaciones de sistemas dinámicos de 1D (dy/dx = f(y; mu)) ===")

    f_expr = pedir_funcion()
    print(f"\nEcuación: dy/dx = {f_expr}")

    mu_min_usuario = pedir_valor_opcional(
        "\nRango de mu a explorar - mu mínimo (Enter = -5): ", -5.0)
    mu_max_usuario = pedir_valor_opcional(
        "Rango de mu a explorar - mu máximo (Enter = 5): ", 5.0)
    mu_min, mu_max = sorted((mu_min_usuario, mu_max_usuario))
    if mu_min == mu_max:
        mu_min, mu_max = mu_min - 1, mu_max + 1

    if not f_expr.is_polynomial(y, mu):
        print("\n(La ecuación no es polinómica: se usan métodos numéricos, no hay "
              "forma cerrada para los equilibrios ni para el punto de bifurcación.)")

    print("\nBuscando puntos de bifurcación (donde f = 0 y f'(y) = 0 a la vez)...")
    puntos = encontrar_puntos_bifurcacion(f_expr)
    if not puntos:
        print("  No se resolvió en forma cerrada; se busca numéricamente...")
        puntos = buscar_puntos_bifurcacion_numerico(f_expr, mu_min, mu_max)

    if not puntos:
        print("\nNo se encontraron puntos de bifurcación reales en el rango de mu dado.")
        return

    # el rango de mu tiene que incluir todos los puntos de bifurcación detectados
    for _, mu0 in puntos:
        mu_min, mu_max = min(mu_min, float(mu0)), max(mu_max, float(mu0))

    print(f"\nSe encontraron {len(puntos)} punto(s) de bifurcación:")
    clasificaciones = []
    for y0, mu0 in puntos:
        tipo = clasificar_bifurcacion(f_expr, y0, mu0)
        clasificaciones.append(tipo)
        print(f"  (y, mu) = ({y0}, {mu0})   ->   {tipo}")

    ramas = calcular_ramas(f_expr)
    print("\nPuntos de equilibrio en función de mu:")
    if not ramas:
        print("  (no se pudo resolver f(y; mu) = 0 para y en forma cerrada)")
    for rama in ramas:
        print(f"  y_eq(mu) = {rama}")

    df_expr, info_ramas = analizar_ramas(f_expr, ramas, mu_min, mu_max)
    mostrar_estabilidad_ramas(df_expr, info_ramas)

    nombre_resumen = ", ".join(dict.fromkeys(clasificaciones))  # únicos, en orden
    graficar_diagrama_bifurcacion(info_ramas, mu_min, mu_max, nombre_resumen, puntos)

    delta = pedir_valor_opcional(
        "\n¿Qué distancia de mu usar antes/después de cada punto de "
        "bifurcación para los diagramas de fase (Enter = 1)?: ", 1.0)
    y_min, y_max = rango_y_equilibrios(info_ramas)

    for idx, ((y0, mu0), tipo) in enumerate(zip(puntos, clasificaciones), start=1):
        sufijo = f"_{idx}" if len(puntos) > 1 else ""
        graficar_fases_tres_mu(f_expr, df_expr, float(mu0), delta, y_min, y_max, tipo, sufijo)

    plt.show()


if __name__ == "__main__":
    main()
