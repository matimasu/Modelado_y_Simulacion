"""
Sistemas Dinámicos de una dimensión (ecuaciones autónomas)
-------------------------------------------------------------
Analiza dy/dx = f(y), un sistema dinámico autónomo de 1 dimensión
(f depende solo de y, no de x).

El script:
  - Pide f(y) tal que dy/dx = f(y) (sintaxis sympy, sin prefijo, variable y).
  - Calcula los puntos de equilibrio resolviendo f(y) = 0 (simbólicamente
    con sympy; si no encuentra solución en forma cerrada, busca raíces
    numéricamente por cambio de signo + bisección).
  - Hace el estudio de estabilidad con el criterio de la primera derivada:
    muestra f'(y) resuelta y su valor en cada equilibrio.
      f'(y_eq) < 0 -> Estable      f'(y_eq) > 0 -> Inestable
    Si f'(y_eq) = 0, el criterio no decide: se reclasifica mirando el
    signo de f(y) a cada lado del punto (Estable / Inestable / Semiestable).
  - Grafica el diagrama de fase (línea de fase): varias flechas verticales
    espaciadas a lo largo de todo el eje (tipo campo vectorial 1D) que
    indican el sentido del flujo, y cada equilibrio marcado con un círculo
    vacío, verde si es estable y rojo si es inestable (naranja si es
    semiestable).
  - Grafica y' en función de y (curva f(y)), con las mismas flechas de
    dirección de flujo sobre el eje y' = 0 y los equilibrios marcados.
  - Resuelve la ecuación diferencial analíticamente con sympy.dsolve
    (solución general, o particular si se da una condición inicial) e
    imprime el resultado.
  - Grafica en el plano (x, y): las asíntotas horizontales en cada punto
    de equilibrio (mismo color que su estabilidad), las isoclinas
    y' = k para k múltiplo de 0.5 dentro del rango de f(y) graficado, y
    varias curvas solución (numéricas, RK4) que arrancan todas del mismo
    x0 (marcado con un punto) y avanzan hacia adelante, bien separadas
    entre sí, cada una de un color distinto con su condición inicial en
    la leyenda, más la solución particular (resaltada en negro) si se dio
    una condición inicial.

Requiere: sympy, numpy, matplotlib, tabulate
    pip install sympy numpy matplotlib tabulate
"""

import os
import math
import multiprocessing as mp

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

CARPETA_GRAFICOS = "graficos"
TIEMPO_LIMITE_DSOLVE = 12  # segundos; ver resolver_analiticamente

x, y = sp.symbols('x y')


# ----------------------------------------------------------------------
# Entrada de datos
# ----------------------------------------------------------------------

def pedir_funcion():
    while True:
        texto = input(
            "\nIngresá f(y) tal que dy/dx = f(y) (sistema autónomo, solo "
            "variable y; ej: y - y**2, y*(2-y), sin(y)): "
        ).strip()
        try:
            f_expr = sp.sympify(texto)
        except (sp.SympifyError, TypeError):
            print("  -> Expresión inválida, probá de nuevo.")
            continue

        libres = f_expr.free_symbols
        if libres - {y}:
            otras = ", ".join(str(s) for s in libres - {y})
            print(f"  -> La expresión no puede depender de {otras}: es un sistema "
                  "autónomo, usá solo la variable y.")
            continue

        f_num = sp.lambdify(y, f_expr, "numpy")
        return f_expr, f_num


def pedir_valor(mensaje):
    while True:
        texto = input(mensaje).strip()
        try:
            return float(sp.N(sp.sympify(texto)))
        except (sp.SympifyError, TypeError, ValueError):
            print("  -> Valor inválido, probá de nuevo (ej: 0, 2.5, pi/2).")


def pedir_valor_opcional(mensaje, valor_por_defecto):
    texto = input(mensaje).strip()
    if texto == "":
        return valor_por_defecto
    try:
        return float(sp.N(sp.sympify(texto)))
    except (sp.SympifyError, TypeError, ValueError):
        print(f"  -> Valor inválido, se usa el valor por defecto ({valor_por_defecto}).")
        return valor_por_defecto


def pedir_condicion_inicial():
    respuesta = input(
        "\n¿Ingresar una condición inicial para la solución particular? (s/n): "
    ).strip().lower()
    if respuesta != "s":
        return None, None
    x0 = pedir_valor("  x0: ")
    y0 = pedir_valor("  y0: ")
    return x0, y0


# ----------------------------------------------------------------------
# Búsqueda numérica de raíces (fallback para equilibrios e isoclinas)
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
# Equilibrios y estabilidad
# ----------------------------------------------------------------------

def calcular_equilibrios(f_expr, f_num):
    """Devuelve (equilibrios, es_aproximado). Intenta resolver f(y)=0 en
    forma cerrada con sympy; si no lo logra, busca raíces numéricamente."""
    if y not in f_expr.free_symbols:
        return [], False

    try:
        soluciones = sp.solve(sp.Eq(f_expr, 0), y)
    except NotImplementedError:
        soluciones = []

    reales = []
    for s in soluciones:
        try:
            valor_complejo = complex(sp.N(s))
        except (TypeError, ValueError):
            continue
        if abs(valor_complejo.imag) < 1e-9:
            reales.append(sp.simplify(s))

    if reales:
        reales.sort(key=lambda s: float(sp.N(s)))
        return reales, False

    raices_numericas = buscar_raices(lambda v: _evaluar_seguro(f_num, v), -25, 25, n=3000)
    return [sp.Float(r) for r in raices_numericas], bool(raices_numericas)


def estudiar_estabilidad(f_expr, equilibrios):
    df_expr = sp.diff(f_expr, y)
    df_simplificada = sp.simplify(df_expr)

    info = []
    for yeq in equilibrios:
        valor_simbolico = sp.simplify(df_simplificada.subs(y, yeq))
        valor_num = float(sp.N(sp.re(valor_simbolico)))

        if valor_num < -1e-9:
            estado = "Estable"
        elif valor_num > 1e-9:
            estado = "Inestable"
        else:
            estado = "Indeterminado"

        info.append({
            "y_eq": yeq,
            "y_eq_num": float(sp.N(yeq)),
            "derivada_valor": valor_simbolico,
            "derivada_valor_num": valor_num,
            "estado": estado,
        })
    return df_expr, df_simplificada, info


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


def mostrar_estabilidad(df_expr, df_simplificada, info_estabilidad):
    print(f"\nDerivada: f'(y) = {df_expr}")
    if df_expr != df_simplificada:
        print(f"f'(y) simplificada = {df_simplificada}")

    if not info_estabilidad:
        print("\nNo hay puntos de equilibrio para analizar.")
        return

    print("\nEstudio de estabilidad (criterio de la primera derivada):")
    filas = []
    for inf in info_estabilidad:
        print(f"  f'(y = {inf['y_eq']}) = {inf['derivada_valor']} = "
              f"{inf['derivada_valor_num']:.6g}  ->  {inf['estado']}")
        filas.append([str(inf["y_eq"]), str(inf["derivada_valor"]),
                      f"{inf['derivada_valor_num']:.6g}", inf["estado"]])
    print()
    print(tabulate(filas, headers=["y_eq", "f'(y_eq)", "valor", "Estado"], tablefmt="grid"))


# ----------------------------------------------------------------------
# Rango de graficación
# ----------------------------------------------------------------------

def ajustar_rango(y_min, y_max, info_estabilidad, margen_relativo=0.15):
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    extendido = False
    for inf in info_estabilidad:
        yeq = inf["y_eq_num"]
        if yeq < y_min:
            y_min, extendido = yeq, True
        if yeq > y_max:
            y_max, extendido = yeq, True

    margen = max((y_max - y_min) * margen_relativo, 0.5)
    y_min -= margen
    y_max += margen
    if extendido:
        print(f"\n(El rango de y se amplió para incluir todos los equilibrios: "
              f"[{y_min:.4g}, {y_max:.4g}])")
    return y_min, y_max


# ----------------------------------------------------------------------
# Flechas de dirección de flujo (estilo campo vectorial 1D)
# ----------------------------------------------------------------------

def puntos_de_flechas(y_min, y_max, y_eqs, cantidad=22):
    """Centros donde dibujar flechas de dirección de flujo, espaciadas
    regularmente a lo largo de todo el rango [y_min, y_max] (como un campo
    vectorial 1D), salteando los puntos de equilibrio."""
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
# Gráfico 1: diagrama de fase (línea de fase)
# ----------------------------------------------------------------------

def graficar_linea_fase(f_num, info_estabilidad, y_min, y_max):
    y_eqs = sorted(inf["y_eq_num"] for inf in info_estabilidad)

    fig, ax = plt.subplots(figsize=(3.5, 7))
    ax.axvline(0, color="black", linewidth=1.2, zorder=1)

    cantidad_flechas = 22
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
        ax.plot(0, inf["y_eq_num"], marker="o", markersize=15, markerfacecolor="white",
                 markeredgecolor=color, markeredgewidth=2.6, zorder=3)
        ax.annotate(f'y = {inf["y_eq_num"]:.4g}\n({inf["estado"]})',
                    xy=(0.12, inf["y_eq_num"]), va="center", fontsize=9)

    ax.set_xlim(-1, 1)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_ylabel("y")
    ax.set_title("Diagrama de fase\n(línea de fase)")
    plt.tight_layout()

    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    ruta = os.path.join(CARPETA_GRAFICOS, "sistemas_dinamicos_linea_fase.png")
    plt.savefig(ruta, dpi=150)
    print(f"\nGráfico guardado como '{ruta}'")


# ----------------------------------------------------------------------
# Gráfico 2: y' en función de y
# ----------------------------------------------------------------------

def graficar_y_vs_yprima(f_num, info_estabilidad, y_min, y_max):
    ys = np.linspace(y_min, y_max, 600)
    yprimas = f_num(ys) * np.ones_like(ys)

    plt.figure(figsize=(9, 6))
    plt.axhline(0, color="black", linewidth=1)
    plt.plot(ys, yprimas, color="steelblue", linewidth=2.2, label="y' = f(y)", zorder=2)

    y_eqs = sorted(inf["y_eq_num"] for inf in info_estabilidad)
    cantidad_flechas = 22
    largo_flecha = (y_max - y_min) / cantidad_flechas * 0.65
    for centro in puntos_de_flechas(y_min, y_max, y_eqs, cantidad_flechas):
        signo = _evaluar_seguro(f_num, centro)
        if signo is None:
            continue
        x0, x1 = centro - largo_flecha / 2, centro + largo_flecha / 2
        if signo < 0:
            x0, x1 = x1, x0
        plt.annotate("", xy=(x1, 0), xytext=(x0, 0),
                     arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=1.8, alpha=0.9), zorder=2)

    for inf in info_estabilidad:
        color = color_estado(inf["estado"])
        plt.plot(inf["y_eq_num"], 0, marker="o", markersize=13, markerfacecolor="white",
                  markeredgecolor=color, markeredgewidth=2.6, zorder=3)
        plt.annotate(inf["estado"], xy=(inf["y_eq_num"], 0), xytext=(0, 12),
                     textcoords="offset points", ha="center", fontsize=9, color=color)

    plt.xlabel("y")
    plt.ylabel("y'")
    plt.title("y' en función de y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    ruta = os.path.join(CARPETA_GRAFICOS, "sistemas_dinamicos_y_vs_yprima.png")
    plt.savefig(ruta, dpi=150)
    print(f"Gráfico guardado como '{ruta}'")


# ----------------------------------------------------------------------
# Resolución analítica
# ----------------------------------------------------------------------

def _resolver_dsolve_en_proceso(f_expr, x0, y0, queue):
    """Corre sp.dsolve en un proceso aparte. Para ciertas f(y) (ej. cúbicas)
    dsolve puede tardar minutos o fallar con errores internos de sympy
    (no solo NotImplementedError/ValueError); al aislarlo en un proceso se
    lo puede matar por tiempo desde afuera sin colgar el script."""
    try:
        yf = sp.Function('y')(x)
        ecuacion = sp.Eq(sp.diff(yf, x), f_expr.subs(y, yf))
        if x0 is not None and y0 is not None:
            solucion = sp.dsolve(ecuacion, yf, ics={yf.subs(x, x0): y0})
        else:
            solucion = sp.dsolve(ecuacion, yf)
        if isinstance(solucion, list):
            solucion = solucion[0]
        queue.put(solucion)
    except Exception:
        queue.put(None)


def resolver_analiticamente(f_expr, x0=None, y0=None):
    x0_exacto = sp.nsimplify(x0, rational=True) if x0 is not None else None
    y0_exacto = sp.nsimplify(y0, rational=True) if y0 is not None else None

    queue = mp.Queue()
    proceso = mp.Process(target=_resolver_dsolve_en_proceso,
                          args=(f_expr, x0_exacto, y0_exacto, queue))
    proceso.start()
    proceso.join(TIEMPO_LIMITE_DSOLVE)

    if proceso.is_alive():
        proceso.terminate()
        proceso.join()
        return None
    return queue.get() if not queue.empty() else None


def formatear_solucion(solucion):
    if solucion is None:
        return None
    if solucion.lhs == sp.Function('y')(x):
        return f"y(x) = {sp.simplify(solucion.rhs)}"
    return str(solucion)  # solución implícita: no se pudo despejar y(x)


# ----------------------------------------------------------------------
# Gráfico 3: solución, asíntotas e isoclinas
# ----------------------------------------------------------------------

def integrar_rk4(f_num, x0, y0, h, pasos, y_min, y_max, margen=3.0):
    """RK4 para dy/dx = f_num(y) (h puede ser negativo). Corta si y se
    escapa muy por fuera del rango de interés o si f_num no está definida."""
    xs, ys = [x0], [y0]
    xi, yi = x0, y0
    cota_inf = y_min - margen * (y_max - y_min)
    cota_sup = y_max + margen * (y_max - y_min)
    for _ in range(pasos):
        try:
            k1 = f_num(yi)
            k2 = f_num(yi + h / 2 * k1)
            k3 = f_num(yi + h / 2 * k2)
            k4 = f_num(yi + h * k3)
            yi_nuevo = yi + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            break
        if not np.isfinite(yi_nuevo) or not (cota_inf <= yi_nuevo <= cota_sup):
            break
        yi = yi_nuevo
        xi = xi + h
        xs.append(xi)
        ys.append(yi)
    return xs, ys


COLORES_CURVAS = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:cyan"]


def puntos_representativos(y_min, y_max, y_eqs, cantidad=5, paso=0.5):
    """Puntos de partida para curvas solución bien separadas y distinguibles
    entre sí (una por color, con su propia condición inicial en la leyenda),
    elegidos sobre la misma grilla "redonda" que las isoclinas (múltiplos de
    'paso', ej: 1.5, 2, 2.5, ...) en vez de valores decimales arbitrarios,
    evitando arrancar justo encima de un equilibrio."""
    inicio = math.ceil(y_min / paso) * paso
    tolerancia = paso * 0.25
    grilla = []
    v = inicio
    while v <= y_max + 1e-9:
        v_redondeado = round(v, 10)
        if not any(abs(v_redondeado - yeq) < tolerancia for yeq in y_eqs):
            grilla.append(v_redondeado)
        v += paso

    if len(grilla) <= cantidad:
        return grilla

    # de esa grilla, se eligen 'cantidad' valores lo más espaciados posible
    indices = sorted({round(i * (len(grilla) - 1) / (cantidad - 1)) for i in range(cantidad)})
    return [grilla[i] for i in indices]


def calcular_isoclinas(f_num, y_min, y_max, paso=0.5):
    """Isoclinas y' = k (rectas horizontales, ya que f depende solo de y),
    para cada k múltiplo de 'paso' dentro del rango de f(y) graficado."""
    evaluar = lambda v: _evaluar_seguro(f_num, v)
    muestras = [evaluar(v) for v in np.linspace(y_min, y_max, 400)]
    validas = [v for v in muestras if v is not None]
    if not validas:
        return []

    k_min = math.floor(min(validas) / paso) * paso
    k_max = math.ceil(max(validas) / paso) * paso

    isoclinas = []
    k = k_min
    while k <= k_max + 1e-9:
        def g(v, k=k):
            val = evaluar(v)
            return None if val is None else val - k
        for raiz in buscar_raices(g, y_min, y_max):
            isoclinas.append((k, raiz))
        k = round(k + paso, 10)
    return isoclinas


def graficar_solucion(f_num, info_estabilidad, y_min, y_max, x0, ancho_x, y0=None):
    xs_izq, xs_der = x0, x0 + ancho_x
    h = ancho_x / 500
    pasos = 500

    plt.figure(figsize=(10, 7))

    y_eqs = [inf["y_eq_num"] for inf in info_estabilidad]
    for i, yr in enumerate(puntos_representativos(y_min, y_max, y_eqs, cantidad=5)):
        color = COLORES_CURVAS[i % len(COLORES_CURVAS)]
        xs, ys = integrar_rk4(f_num, x0, yr, h, pasos, y_min, y_max)
        plt.plot(xs, ys, color=color, linewidth=1.8,
                  label=f"y({x0:g}) = {yr:.3g}", zorder=2)
        plt.plot(x0, yr, marker="o", color=color, markersize=5, zorder=3)

    if y0 is not None:
        xs, ys = integrar_rk4(f_num, x0, y0, h, pasos, y_min, y_max)
        plt.plot(xs, ys, color="black", linewidth=2.6,
                  label=f"Solución particular (y({x0:g})={y0:g})", zorder=4)
        plt.plot(x0, y0, marker="o", color="black", zorder=5)

    for inf in info_estabilidad:
        color = color_estado(inf["estado"])
        plt.axhline(inf["y_eq_num"], color=color, linestyle="--", linewidth=1.8,
                     label=f'Asíntota y={inf["y_eq_num"]:.4g} ({inf["estado"]})', zorder=3)

    isoclinas = sorted(calcular_isoclinas(f_num, y_min, y_max, paso=0.5), key=lambda t: t[1])
    gap_minimo_etiqueta = (y_max - y_min) * 0.02
    ultimo_y_etiqueta = None
    primera = True
    for k, yval in isoclinas:
        plt.axhline(yval, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
                     label="Isoclinas (y' = k)" if primera else None, zorder=1)
        # si f(y) varía mucho, puede haber muchas isoclinas muy juntas: se
        # etiqueta solo una de cada grupo cercano para que no se solapen
        if ultimo_y_etiqueta is None or abs(yval - ultimo_y_etiqueta) > gap_minimo_etiqueta:
            plt.text(xs_der, yval, f" k={k:g}", fontsize=7, color="gray", va="center")
            ultimo_y_etiqueta = yval
        primera = False

    plt.xlim(xs_izq, xs_der)
    plt.ylim(y_min, y_max)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solución, asíntotas e isoclinas")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    ruta = os.path.join(CARPETA_GRAFICOS, "sistemas_dinamicos_solucion.png")
    plt.savefig(ruta, dpi=150)
    print(f"Gráfico guardado como '{ruta}'")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=== Sistemas Dinámicos de una dimensión (dy/dx = f(y)) ===")

    f_expr, f_num = pedir_funcion()
    print(f"\nEcuación: dy/dx = {f_expr}")

    equilibrios, es_aproximado = calcular_equilibrios(f_expr, f_num)
    if not equilibrios:
        print("\nNo se encontraron puntos de equilibrio reales (f(y) = 0 no tiene solución real).")
    else:
        etiqueta = " (aproximados, hallados numéricamente)" if es_aproximado else ""
        print(f"\nPuntos de equilibrio{etiqueta}: {', '.join(str(e) for e in equilibrios)}")

    df_expr, df_simplificada, info_estabilidad = estudiar_estabilidad(f_expr, equilibrios)
    mostrar_estabilidad(df_expr, df_simplificada, info_estabilidad)

    y_min_usuario = pedir_valor("\nRango de y para los gráficos - y mínimo: ")
    y_max_usuario = pedir_valor("Rango de y para los gráficos - y máximo: ")
    y_min, y_max = ajustar_rango(y_min_usuario, y_max_usuario, info_estabilidad)

    if any(inf["estado"] == "Indeterminado" for inf in info_estabilidad):
        reclasificar_indeterminados(f_num, info_estabilidad, y_min, y_max)
        print("\nAlgunos equilibrios tenían f'(y_eq) = 0 (el criterio de la primera "
              "derivada no decide); se reclasificaron según el signo de f(y) a cada lado:")
        for inf in info_estabilidad:
            print(f"  y = {inf['y_eq']}  ->  {inf['estado']}")

    graficar_linea_fase(f_num, info_estabilidad, y_min, y_max)
    graficar_y_vs_yprima(f_num, info_estabilidad, y_min, y_max)

    solucion_general = resolver_analiticamente(f_expr)
    texto_general = formatear_solucion(solucion_general)
    if texto_general is not None:
        print(f"\nSolución general: {texto_general}")
    else:
        print("\nNo se pudo resolver la EDO analíticamente en forma cerrada.")

    x0_ic, y0_ic = pedir_condicion_inicial()
    if x0_ic is not None:
        solucion_particular = resolver_analiticamente(f_expr, x0_ic, y0_ic)
        texto_particular = formatear_solucion(solucion_particular)
        if texto_particular is not None:
            print(f"Solución particular (y({x0_ic:g}) = {y0_ic:g}): {texto_particular}")
        else:
            print("No se pudo resolver la solución particular en forma cerrada "
                  "(se integrará numéricamente para graficar).")

    ancho_x = pedir_valor_opcional(
        "\n¿Hasta qué distancia en x graficar la solución, hacia adelante desde x0 "
        "(Enter = 10)?: ", 10.0)
    x0_grafico = x0_ic if x0_ic is not None else 0.0

    graficar_solucion(f_num, info_estabilidad, y_min, y_max, x0_grafico, ancho_x, y0_ic)

    plt.show()


if __name__ == "__main__":
    main()
