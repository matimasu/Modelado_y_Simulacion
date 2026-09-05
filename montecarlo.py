"""
Integración por Monte Carlo
----------------------------
Aproxima la integral de f sobre un dominio rectangular de n dimensiones
mediante el método de Monte Carlo (valor medio):

    I ~= V * (1/N) * sum_{i=1}^{N} f(P_i)

con P_i puntos aleatorios uniformes en el dominio [a1,b1] x ... x [an,bn]
y V el volumen (longitud, área, etc.) de ese dominio.

El script:
  - Pide f (sintaxis sympy, sin prefijo: sin(x1)*x2, x**2+y**2, etc.).
    La cantidad de variables libres detectadas en la expresión define la
    dimensión n (se ordenan alfabéticamente para pedir los extremos).
  - Pide alfa (para el intervalo de confianza), justo después de f.
    Si se deja vacío, se omite todo el cálculo del intervalo de confianza
    (z, margen e IC).
  - Pide los extremos [a_i, b_i] de cada variable y la cantidad N de
    puntos aleatorios (N lo decide el usuario; los puntos no se listan).
  - Genera los N puntos con random.uniform (semilla fija: random.seed(42)).
  - Calcula I, el desvío muestral, n y el error estándar; si se dio alfa,
    también z(alfa/2), z(alfa/2)*error_estandar y el intervalo de
    confianza I +- ese margen.
  - Grafica la función con su contenedor (el dominio rectangular) y los
    puntos arrojados. Disponible para 1 y 2 dimensiones.

Requiere: sympy, numpy, matplotlib
    pip install sympy numpy matplotlib
"""

import random
import math
from statistics import NormalDist

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def pedir_funcion():
    while True:
        texto = input("Ingresá f (ej: x**2, sin(x1)*x2, x**2 + y**2 + z**2): ").strip()
        try:
            f_expr = sp.sympify(texto)
        except (sp.SympifyError, TypeError):
            print("  -> Expresión inválida, probá de nuevo.")
            continue

        variables = sorted(f_expr.free_symbols, key=lambda s: s.name)
        if not variables:
            print("  -> La expresión no tiene variables, probá de nuevo.")
            continue

        f_num = sp.lambdify(variables, f_expr, "numpy")
        nombres = ", ".join(str(v) for v in variables)
        print(f"  -> f({nombres}) = {f_expr}   (dimensión: {len(variables)})")
        return f_expr, variables, f_num


def pedir_alfa():
    """Alfa para el intervalo de confianza. Vacío -> None (se omite el cálculo del IC)."""
    while True:
        texto = input("\nAlfa para el intervalo de confianza (Enter para omitir): ").strip()
        if texto == "":
            return None
        try:
            alfa = float(sp.N(sp.sympify(texto)))
        except (sp.SympifyError, TypeError, ValueError):
            print("  -> Valor inválido, probá de nuevo.")
            continue
        if 0 < alfa < 1:
            return alfa
        print("  -> Alfa debe estar entre 0 y 1.")


def pedir_valor(mensaje):
    while True:
        texto = input(mensaje).strip()
        try:
            return float(sp.N(sp.sympify(texto)))
        except (sp.SympifyError, TypeError, ValueError):
            print("  -> Valor inválido, probá de nuevo (ej: 0, 2.5, pi/2).")


def pedir_extremos(variables):
    extremos = []
    for v in variables:
        while True:
            a = pedir_valor(f"Extremo inferior de {v}: ")
            b = pedir_valor(f"Extremo superior de {v}: ")
            if a == b:
                print("  -> Los extremos no pueden ser iguales, probá de nuevo.")
                continue
            extremos.append((a, b) if a < b else (b, a))
            break
    return extremos


def pedir_n():
    while True:
        texto = input("\nCantidad de puntos aleatorios N: ").strip()
        try:
            n = int(texto)
        except ValueError:
            print("  -> Ingresá un número entero.")
            continue
        if n >= 2:
            return n
        print("  -> N debe ser al menos 2.")


def generar_puntos(extremos, n):
    random.seed(42)
    return [tuple(random.uniform(a, b) for a, b in extremos) for _ in range(n)]


def montecarlo(f_num, extremos, n):
    puntos = generar_puntos(extremos, n)
    valores = np.array([float(f_num(*p)) for p in puntos])

    volumen = 1.0
    for a, b in extremos:
        volumen *= (b - a)

    media = float(np.mean(valores))
    desvio = math.sqrt((1 / (n - 1)) * np.sum((valores - media) ** 2))
    I = volumen * media
    error_estandar = volumen * desvio / math.sqrt(n)

    return puntos, valores, volumen, media, I, desvio, error_estandar


def intervalo_confianza(I, error_estandar, alfa):
    z = NormalDist().inv_cdf(1 - alfa / 2)
    margen = z * error_estandar
    return z, margen, (I - margen, I + margen)


def mostrar_resultados(volumen, media, I, desvio, n, z=None, margen=None, ic=None):
    print(f"\nVolumen del dominio: {volumen:.6g}")
    print(f"Media de f(P_i) = {media:.8f}")
    print(f"I (aproximación de la integral) = {I:.8f}")
    print(f"Desvío muestral (V*s) = {volumen * desvio:.8f}")
    print(f"n = {n}")
    if z is not None:
        print(f"z(alfa/2) = {z:.6f}")
        print(f"Error estándar (z(alfa/2)*V*s/sqrt(n)) = {margen:.8f}")
        print(f"Intervalo de confianza: [{ic[0]:.8f}, {ic[1]:.8f}]")


def graficar(variables, extremos, puntos, valores, f_num):
    dim = len(variables)

    if dim == 1:
        a, b = extremos[0]
        malla = np.linspace(a, b, 400)
        curva = f_num(malla) * np.ones_like(malla)
        xs = [p[0] for p in puntos]

        plt.figure(figsize=(9, 6))
        plt.plot(malla, curva, color="steelblue", linewidth=2, label="f(x)", zorder=2)

        y_min = min(0.0, float(np.min(curva)), float(np.min(valores)))
        y_max = max(float(np.max(curva)), float(np.max(valores)))
        plt.gca().add_patch(plt.Rectangle((a, y_min), b - a, y_max - y_min, fill=False,
                                            edgecolor="gray", linestyle="--", linewidth=1.2,
                                            label="Contenedor", zorder=1))
        plt.scatter(xs, valores, color="darkorange", s=12, alpha=0.6, zorder=3, label="Puntos arrojados")

        plt.xlabel(str(variables[0]))
        plt.ylabel("f")

    elif dim == 2:
        (a1, b1), (a2, b2) = extremos
        xs = [p[0] for p in puntos]
        ys = [p[1] for p in puntos]

        plt.figure(figsize=(9, 7))
        plt.gca().add_patch(plt.Rectangle((a1, a2), b1 - a1, b2 - a2, fill=False,
                                            edgecolor="gray", linestyle="--", linewidth=1.2,
                                            label="Contenedor", zorder=1))
        disp = plt.scatter(xs, ys, c=valores, cmap="viridis", s=14, alpha=0.85, zorder=3,
                            label="Puntos arrojados")
        plt.colorbar(disp, label="f")

        plt.xlabel(str(variables[0]))
        plt.ylabel(str(variables[1]))

    else:
        print(f"\nEl gráfico solo está disponible para 1 o 2 dimensiones (esta función tiene {dim}).")
        return

    plt.title("Integración por Monte Carlo")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("montecarlo_resultado.png", dpi=150)
    print("\nGráfico guardado como 'montecarlo_resultado.png'")
    plt.show()


def main():
    f_expr, variables, f_num = pedir_funcion()
    alfa = pedir_alfa()
    extremos = pedir_extremos(variables)
    n = pedir_n()

    puntos, valores, volumen, media, I, desvio, error_estandar = montecarlo(f_num, extremos, n)

    if alfa is not None:
        z, margen, ic = intervalo_confianza(I, error_estandar, alfa)
        mostrar_resultados(volumen, media, I, desvio, n, z, margen, ic)
    else:
        mostrar_resultados(volumen, media, I, desvio, n)

    graficar(variables, extremos, puntos, valores, f_num)


if __name__ == "__main__":
    main()
