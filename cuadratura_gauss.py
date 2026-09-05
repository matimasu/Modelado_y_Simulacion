"""
Cuadratura de Gauss (Gauss-Legendre)
-------------------------------------
Aproxima ∫_a^b f(x) dx con la cuadratura de Gauss-Legendre de n puntos:

    ∫_a^b f(x) dx ~= (b-a)/2 * sum_{i=0}^{n-1} w_i * f(x_i)

con t_i, w_i los nodos y pesos estándar de Gauss-Legendre en [-1, 1]
(obtenidos con numpy.polynomial.legendre.leggauss) y

    x_i = (b-a)/2 * t_i + (a+b)/2

el nodo transformado al intervalo [a, b].

El script pide f(x) (sintaxis sympy, sin prefijo), los extremos a, b y la
cantidad de puntos n, muestra la tabla de nodos/pesos/f(x_i) y el
resultado final, todo con 10 decimales.

Requiere: sympy, numpy, tabulate
    pip install sympy numpy tabulate
"""

import sympy as sp
import numpy as np
from tabulate import tabulate

x = sp.symbols('x')


def pedir_funcion():
    while True:
        texto = input("Ingresá f(x) (ej: x**2 - 2, sin(x), exp(x)/2): ").strip()
        try:
            f_expr = sp.sympify(texto)
        except (sp.SympifyError, TypeError):
            print("  -> Expresión inválida, probá de nuevo.")
            continue
        f_num = sp.lambdify(x, f_expr, "numpy")
        return f_expr, f_num


def pedir_valor(mensaje):
    while True:
        texto = input(mensaje).strip()
        try:
            return float(sp.N(sp.sympify(texto)))
        except (sp.SympifyError, TypeError, ValueError):
            print("  -> Valor inválido, probá de nuevo (ej: 0, 2.5, pi/2).")


def pedir_n():
    while True:
        texto = input("Cantidad de puntos n: ").strip()
        try:
            n = int(texto)
        except ValueError:
            print("  -> Ingresá un número entero.")
            continue
        if n >= 1:
            return n
        print("  -> n debe ser al menos 1.")


def cuadratura_gauss(f_num, a, b, n):
    t, w = np.polynomial.legendre.leggauss(n)
    xs = (b - a) / 2 * t + (a + b) / 2
    ws = (b - a) / 2 * w
    ys = f_num(xs) * np.ones_like(xs)
    terminos = ws * ys
    I = float(np.sum(terminos))
    return xs, ws, ys, terminos, I


def mostrar_tabla(xs, ws, ys, terminos):
    tabla = [[i, xi, wi, yi, ti] for i, (xi, wi, yi, ti) in enumerate(zip(xs, ws, ys, terminos))]
    print("\nTabla de nodos y pesos:")
    print(tabulate(tabla, headers=["i", "x_i", "w_i", "f(x_i)", "w_i*f(x_i)"],
                    floatfmt=".10f", tablefmt="grid"))


def main():
    f_expr, f_num = pedir_funcion()
    a = pedir_valor("Extremo inferior a: ")
    b = pedir_valor("Extremo superior b: ")
    n = pedir_n()

    xs, ws, ys, terminos, I = cuadratura_gauss(f_num, a, b, n)
    mostrar_tabla(xs, ws, ys, terminos)

    print(f"\nResultado de la integral (Cuadratura de Gauss, n={n}) = {I:.10f}")


if __name__ == "__main__":
    main()
