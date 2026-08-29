"""
Métodos de integración numérica
--------------------------------
Menú con 7 métodos para aproximar una integral definida:
  1. Trapecio simple        5. Simpson 3/8 simple
  2. Trapecio compuesto      6. Simpson 3/8 compuesto
  3. Simpson 1/3 simple      7. Rectángulo medio (n rectángulos)
  4. Simpson 1/3 compuesto

Para los métodos SIMPLES, la cantidad de puntos queda fija por la fórmula
(2, 3 y 4 respectivamente) y no se pide n. Para los COMPUESTOS y el
rectángulo medio se pide n (par para Simpson 1/3, múltiplo de 3 para
Simpson 3/8, positivo para trapecio compuesto y rectángulo medio).
El rectángulo medio evalúa f en el punto medio de cada uno de los n
subintervalos, no en sus extremos.

El script:
  - Pide la función f(x) (sintaxis sympy, sin prefijo: sin(x), exp(x), pi, etc.)
  - Pide los extremos a, b y (si corresponde) n.
  - Calcula h, muestra la tabla de nodos (i, x_i, f(x_i)) y el resultado.
  - Calcula la cota del error de truncamiento con la fórmula clásica de
    cada método, estimando M = máx |f^(k)(x)| en [a,b] numéricamente.
  - Grafica f(x), marca las divisiones y dibuja los trapecios/parábolas/
    cúbicas con los que se aproxima cada panel.

Requiere: sympy, numpy, matplotlib, tabulate
    pip install sympy numpy matplotlib tabulate
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

x = sp.symbols('x')

METODOS = {
    1: "Trapecio simple",
    2: "Trapecio compuesto",
    3: "Simpson 1/3 simple",
    4: "Simpson 1/3 compuesto",
    5: "Simpson 3/8 simple",
    6: "Simpson 3/8 compuesto",
    7: "Rectángulo",
}
COMPUESTOS = {2, 4, 6}
REQUIERE_N = COMPUESTOS | {7}
N_FIJO_SIMPLE = {1: 1, 3: 2, 5: 3}
OPCIONES_MENU = {str(k) for k in METODOS}

VARIANTES_RECTANGULO = {1: "izquierda", 2: "punto medio", 3: "derecha"}


def mostrar_menu():
    print("=== Métodos de integración numérica ===\n")
    for k, v in METODOS.items():
        print(f"  {k}. {v}")
    while True:
        texto = input(f"\nElegí un método (1-{len(METODOS)}): ").strip()
        if texto in OPCIONES_MENU:
            return int(texto)
        print(f"  -> Opción inválida, ingresá un número del 1 al {len(METODOS)}.")


def pedir_variante_rectangulo():
    print("\n  1. Por izquierda")
    print("  2. Por punto medio")
    print("  3. Por derecha")
    while True:
        texto = input("Elegí la variante del rectángulo (1-3): ").strip()
        if texto in {"1", "2", "3"}:
            return VARIANTES_RECTANGULO[int(texto)]
        print("  -> Opción inválida, ingresá un número del 1 al 3.")


def nombre_metodo(metodo, variante):
    if metodo == 7:
        return f"Rectángulo ({variante})"
    return METODOS[metodo]


def pedir_funcion():
    while True:
        texto = input("\nIngresá f(x) (ej: x**2 - 2, sin(x), exp(x)/2): ").strip()
        try:
            f_expr = sp.sympify(texto)
            break
        except (sp.SympifyError, TypeError):
            print("  -> Expresión inválida, probá de nuevo.")
    f_num = sp.lambdify(x, f_expr, "numpy")
    return f_expr, f_num


def pedir_valor(mensaje):
    """Devuelve el valor exacto de sympy (racionaliza decimales, deja pi simbólico)."""
    while True:
        texto = input(mensaje).strip()
        try:
            return sp.nsimplify(sp.sympify(texto), rational=True)
        except (sp.SympifyError, TypeError, ValueError):
            print("  -> Valor inválido, probá de nuevo (ej: 0, 2.5, pi/2).")


def pedir_n(metodo):
    if metodo == 4:
        descripcion = "par (múltiplo de 2)"
        es_valido = lambda n: n >= 2 and n % 2 == 0
    elif metodo == 6:
        descripcion = "múltiplo de 3"
        es_valido = lambda n: n >= 3 and n % 3 == 0
    else:
        descripcion = "positivo"
        es_valido = lambda n: n >= 1

    while True:
        texto = input(f"Número de subintervalos n ({descripcion}): ").strip()
        try:
            n = int(texto)
        except ValueError:
            print("  -> Ingresá un número entero.")
            continue
        if not es_valido(n):
            print(f"  -> n debe ser {descripcion}.")
            continue
        return n


def calcular_ys(f_expr, f_num, xs):
    """Evalúa f en cada nodo. Si da una indeterminación (nan/inf, tipo 0/0),
    reemplaza el valor por el límite simbólico de f en ese punto."""
    with np.errstate(all="ignore"):
        ys = np.array(f_num(xs) * np.ones_like(xs), dtype=float)

    notas = [""] * len(xs)
    for i, xi in enumerate(xs):
        if not np.isfinite(ys[i]):
            limite = sp.limit(f_expr, x, float(xi))
            valor = sp.N(limite)
            if valor.is_real and valor.is_finite:
                ys[i] = float(valor)
                notas[i] = f"límite aplicado (f(x_i) era 0/0) -> {valor}"

    return ys, notas


def mostrar_tabla(puntos, ys, notas, etiqueta_x="x_i"):
    tabla = [[i, xi, yi, nota] for i, (xi, yi, nota) in enumerate(zip(puntos, ys, notas))]
    print("\nTabla de nodos:")
    print(tabulate(tabla, headers=["i", etiqueta_x, "f(x_i)", "Nota"], floatfmt=".8f", tablefmt="grid"))


def calcular_integral(metodo, ys, h):
    if metodo == 7:
        return h * sum(ys)
    elif metodo in (1, 2):
        return sum(h / 2 * (ys[k] + ys[k + 1]) for k in range(len(ys) - 1))
    elif metodo in (3, 4):
        return sum(h / 3 * (ys[k] + 4 * ys[k + 1] + ys[k + 2]) for k in range(0, len(ys) - 1, 2))
    else:
        return sum(3 * h / 8 * (ys[k] + 3 * ys[k + 1] + 3 * ys[k + 2] + ys[k + 3]) for k in range(0, len(ys) - 1, 3))


def maximo_abs_derivada(derivada_expr, a, b, muestras=20000):
    derivada_num = sp.lambdify(x, derivada_expr, "numpy")
    malla = np.linspace(a, b, muestras)
    with np.errstate(all="ignore"):
        valores = np.abs(derivada_num(malla) * np.ones_like(malla))
    return float(np.nanmax(valores))


def cota_error(metodo, f_expr, a, b, h, variante=None):
    """Cota del error de truncamiento con la fórmula clásica de cada método,
    reemplazando f^(k)(ξ) por M = máx |f^(k)(x)| en [a,b] (estimado numéricamente).
    Para el rectángulo por izquierda/derecha (orden 1, usa f') el error es de
    orden h; por punto medio (orden 2, usa f'') es de orden h^2."""
    if metodo in (1, 2):
        orden = 2
    elif metodo == 7:
        orden = 2 if variante == "punto medio" else 1
    else:
        orden = 4
    derivada = sp.diff(f_expr, x, orden)
    M = maximo_abs_derivada(derivada, a, b)

    if metodo == 1:
        formula, cota = "(b-a)^3/12 * M", (b - a) ** 3 / 12 * M
    elif metodo == 2:
        formula, cota = "(b-a)*h^2/12 * M", (b - a) * h ** 2 / 12 * M
    elif metodo == 7:
        if variante == "punto medio":
            formula, cota = "(b-a)*h^2/24 * M", (b - a) * h ** 2 / 24 * M
        else:
            formula, cota = "(b-a)*h/2 * M", (b - a) * h / 2 * M
    elif metodo == 3:
        formula, cota = "h^5/90 * M", h ** 5 / 90 * M
    elif metodo == 4:
        formula, cota = "(b-a)*h^4/180 * M", (b - a) * h ** 4 / 180 * M
    elif metodo == 5:
        formula, cota = "3*h^5/80 * M", 3 * h ** 5 / 80 * M
    else:
        formula, cota = "(b-a)*h^4/80 * M", (b - a) * h ** 4 / 80 * M

    return orden, derivada, M, formula, cota


def mostrar_error(orden, derivada, M, formula, cota):
    print("\n=== Cota del error de truncamiento ===")
    print(f"f^({orden})(x) = {sp.simplify(derivada)}")
    print(f"M = max |f^({orden})(x)| en [a,b] (estimado numericamente) ~= {M:.6g}")
    print(f"|E| <= {formula}  ~=  {cota:.6g}")


def graficar(f_num, xs, metodo, variante=None):
    malla = np.linspace(xs[0], xs[-1], 600)

    plt.figure(figsize=(10, 6))
    plt.plot(malla, f_num(malla), label="f(x)", color="steelblue", linewidth=2, zorder=1)

    for xi in xs:
        plt.axvline(xi, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    if metodo == 7:
        if variante == "izquierda":
            puntos = xs[:-1]
            etiqueta_puntos = "Puntos (izquierda)"
        elif variante == "derecha":
            puntos = xs[1:]
            etiqueta_puntos = "Puntos (derecha)"
        else:
            puntos = (xs[:-1] + xs[1:]) / 2
            etiqueta_puntos = "Puntos medios"

        alturas = f_num(puntos) * np.ones_like(puntos)
        plt.plot(puntos, alturas, "o", color="red", markersize=6, label=etiqueta_puntos, zorder=5)

        primero = True
        for k in range(len(xs) - 1):
            plt.plot([xs[k], xs[k + 1]], [alturas[k], alturas[k]], color="darkorange",
                      linewidth=1.5, zorder=3, label="Rectángulos" if primero else None)
            plt.plot([xs[k], xs[k]], [0, alturas[k]], color="darkorange", linewidth=1, zorder=3)
            plt.plot([xs[k + 1], xs[k + 1]], [0, alturas[k]], color="darkorange", linewidth=1, zorder=3)
            plt.fill_between([xs[k], xs[k + 1]], [alturas[k], alturas[k]], color="darkorange", alpha=0.15, zorder=2)
            primero = False
    else:
        ys = f_num(xs)
        plt.plot(xs, ys, "o", color="red", markersize=6, label="Nodos", zorder=5)

        paso = 1 if metodo in (1, 2) else (2 if metodo in (3, 4) else 3)
        etiqueta = "Trapecios" if paso == 1 else ("Parábolas (Simpson 1/3)" if paso == 2 else "Cúbicas (Simpson 3/8)")

        primero = True
        for k in range(0, len(xs) - 1, paso):
            panel_x = xs[k:k + paso + 1]
            panel_y = ys[k:k + paso + 1]
            fino = np.linspace(panel_x[0], panel_x[-1], 50)

            if paso == 1:
                aproximado = np.interp(fino, panel_x, panel_y)
            else:
                coef = np.polyfit(panel_x, panel_y, paso)
                aproximado = np.polyval(coef, fino)

            plt.plot(fino, aproximado, color="darkorange", linewidth=1.5, zorder=3,
                      label=etiqueta if primero else None)
            plt.fill_between(fino, aproximado, color="darkorange", alpha=0.15, zorder=2)
            primero = False

    plt.title(f"Aproximación de la integral - {nombre_metodo(metodo, variante)}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("integracion_resultado.png", dpi=150)
    print("\nGráfico guardado como 'integracion_resultado.png'")
    plt.show()


def main():
    metodo = mostrar_menu()
    variante = pedir_variante_rectangulo() if metodo == 7 else None

    f_expr, f_num = pedir_funcion()
    a_exacto = pedir_valor("Extremo inferior a: ")
    b_exacto = pedir_valor("Extremo superior b: ")

    if a_exacto == b_exacto:
        print("\nEl intervalo tiene longitud 0, no se puede integrar.")
        return

    invertir = a_exacto > b_exacto
    if invertir:
        a_exacto, b_exacto = b_exacto, a_exacto

    n = pedir_n(metodo) if metodo in REQUIERE_N else N_FIJO_SIMPLE[metodo]

    h_exacto = sp.nsimplify((b_exacto - a_exacto) / n, rational=True)
    a, b = float(a_exacto), float(b_exacto)
    h = float(h_exacto)

    xs = np.linspace(a, b, n + 1)
    if metodo == 7:
        if variante == "izquierda":
            puntos_eval = xs[:-1]
            etiqueta_x = "x_i (extremo izquierdo)"
        elif variante == "derecha":
            puntos_eval = xs[1:]
            etiqueta_x = "x_i (extremo derecho)"
        else:
            puntos_eval = (xs[:-1] + xs[1:]) / 2
            etiqueta_x = "x_i (punto medio)"
    else:
        puntos_eval = xs
        etiqueta_x = "x_i"
    ys, notas = calcular_ys(f_expr, f_num, puntos_eval)

    print(f"\nh = {h}  (fracción exacta: {h_exacto})")
    mostrar_tabla(puntos_eval, ys, notas, etiqueta_x)

    resultado = calcular_integral(metodo, ys, h)
    if invertir:
        resultado = -resultado
    print(f"\nResultado de la integral ({nombre_metodo(metodo, variante)}) ~= {resultado:.8f}")

    orden, derivada, M, formula, cota = cota_error(metodo, f_expr, a, b, h, variante)
    mostrar_error(orden, derivada, M, formula, cota)

    graficar(f_num, xs, metodo, variante)


if __name__ == "__main__":
    main()
