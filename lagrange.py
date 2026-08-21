"""
Interpolación de Lagrange
-------------------------
Dado un conjunto de n nodos (x_i, y_i) ingresados por el usuario, el script:
  1. Pide los nodos de a pares "x,y" (línea vacía = fin de la carga).
  2. Muestra cada base de Lagrange L_i(x), factorizada y desarrollada,
     siempre en fracciones exactas (nunca en decimal).
  3. Desarrolla el polinomio de Lagrange P(x) = sum(y_i * L_i(x)) y lo
     reduce a su mínima expresión (términos agrupados por grado).
  4. Calcula el error local en cada nodo (P(x_i) - y_i) y el error
     global (suma y máximo de los errores locales absolutos), como
     verificación de que la interpolación pasa exactamente por los datos.

Requiere: sympy, matplotlib, numpy
    pip install sympy matplotlib numpy
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x = sp.symbols('x')


def parsear_fraccion(texto):
    """Convierte un string ('3', '-1/2', '2.5', etc.) en un sympy.Rational exacto."""
    return sp.nsimplify(sp.sympify(texto), rational=True)


def pedir_nodos():
    """Pide pares x,y hasta que se ingrese una línea vacía. Devuelve listas (xs, ys) de Rational."""
    xs, ys = [], []
    print("Ingresá los nodos como 'x,y' (podés usar fracciones, ej: 1/2,3). Línea vacía para terminar.\n")
    i = 0
    while True:
        linea = input(f"Nodo {i}: ").strip()
        if linea == "":
            break
        try:
            x_str, y_str = [p.strip() for p in linea.split(",")]
            xi = parsear_fraccion(x_str)
            yi = parsear_fraccion(y_str)
        except (ValueError, sp.SympifyError, TypeError):
            print("  -> Formato inválido. Usá 'x,y', por ejemplo: -1,4  o  1/2,3")
            continue

        if xi in xs:
            print(f"  -> El nodo x = {xi} ya fue ingresado, no puede repetirse.")
            continue

        xs.append(xi)
        ys.append(yi)
        i += 1

    return xs, ys


def bases_de_lagrange(xs):
    """Devuelve, para cada i, (L_i factorizada sin expandir, L_i expandida)."""
    n = len(xs)
    bases = []
    for i in range(n):
        numerador = sp.Integer(1)
        denominador = sp.Integer(1)
        for j in range(n):
            if j == i:
                continue
            numerador *= (x - xs[j])
            denominador *= (xs[i] - xs[j])
        factorizada = numerador / denominador
        expandida = sp.expand(factorizada)
        bases.append((factorizada, expandida))
    return bases


def formatear_polinomio(expr, var=x):
    """Convierte un polinomio en string sin ambigüedad: coeficientes fraccionarios
    entre paréntesis, ej: (-1/12)*x**3, para que no se confunda con x**(3/12)."""
    poli = sp.Poly(expr, var)
    terminos = []
    for (grado,), coef in poli.terms():
        coef = sp.nsimplify(coef, rational=True)
        signo = "-" if coef < 0 else "+"
        coef_abs = abs(coef)

        if coef_abs.q != 1:
            coef_str = f"({coef_abs})"
        else:
            coef_str = str(coef_abs)

        if grado == 0:
            termino = coef_str
        elif coef_abs == 1:
            termino = "x" if grado == 1 else f"x**{grado}"
        elif grado == 1:
            termino = f"{coef_str}*x"
        else:
            termino = f"{coef_str}*x**{grado}"

        terminos.append((signo, termino))

    if not terminos:
        return "0"

    resultado = terminos[0][1] if terminos[0][0] == "+" else f"-{terminos[0][1]}"
    for signo, termino in terminos[1:]:
        resultado += f" {signo} {termino}"
    return resultado


def mostrar_bases(bases):
    print("\n=== Bases de Lagrange ===")
    for i, (factorizada, expandida) in enumerate(bases):
        print(f"\nL_{i}(x) = {factorizada}")
        print(f"       = {formatear_polinomio(expandida)}   (desarrollada)")


def polinomio_de_lagrange(xs, ys, bases):
    """Arma P(x) = sum(y_i * L_i(x)) ya expandido y reducido a mínima expresión."""
    p = sp.Integer(0)
    for yi, (_, expandida) in zip(ys, bases):
        p += yi * expandida
    p = sp.expand(p)
    poli = sp.Poly(p, x)
    return poli.as_expr()


def mostrar_polinomio(p):
    print("\n=== Polinomio de Lagrange P(x) ===")
    print(f"P(x) = {formatear_polinomio(p)}")


def errores(xs, ys, p):
    """Error local en cada nodo: P(x_i) - y_i. Error global: suma y máximo de |error local|."""
    locales = []
    for xi, yi in zip(xs, ys):
        valor = sp.nsimplify(p.subs(x, xi), rational=True)
        locales.append(sp.nsimplify(valor - yi, rational=True))
    suma = sp.nsimplify(sum(abs(e) for e in locales), rational=True)
    maximo = max(locales, key=abs) if locales else sp.Integer(0)
    return locales, suma, maximo


def mostrar_errores(xs, locales, suma, maximo):
    print("\n=== Errores locales (P(x_i) - y_i) ===")
    for xi, e in zip(xs, locales):
        print(f"  x = {str(xi):>6}  ->  error local = {e}")

    print("\n=== Error global ===")
    print(f"  Suma de errores locales absolutos = {suma}")
    print(f"  Error local máximo (absoluto)     = {abs(maximo)}")


def graficar(p, xs, ys):
    """Grafica P(x) en un entorno de los nodos y marca los nodos ingresados."""
    p_num = sp.lambdify(x, p, "numpy")
    xs_f = [float(xi) for xi in xs]
    ys_f = [float(yi) for yi in ys]

    margen = max(1.0, (max(xs_f) - min(xs_f)) * 0.3)
    x_min = min(xs_f) - margen
    x_max = max(xs_f) + margen

    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = p_num(x_vals)

    plt.figure(figsize=(9, 6))
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.plot(x_vals, y_vals, label="P(x)", color="steelblue", linewidth=2)
    plt.plot(xs_f, ys_f, "o", color="red", markersize=8, label="Nodos", zorder=5)

    for xi, yi in zip(xs_f, ys_f):
        plt.annotate(f"({xi:g}, {yi:g})", (xi, yi), textcoords="offset points",
                     xytext=(6, 8), fontsize=9)

    plt.title("Polinomio de Lagrange")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("lagrange_resultado.png", dpi=150)
    print("\nGráfico guardado como 'lagrange_resultado.png'")
    plt.show()


def main():
    print("=== Interpolación por Polinomio de Lagrange ===\n")
    xs, ys = pedir_nodos()

    if len(xs) < 2:
        print("\nSe necesitan al menos 2 nodos para interpolar.")
        return

    bases = bases_de_lagrange(xs)
    mostrar_bases(bases)

    p = polinomio_de_lagrange(xs, ys, bases)
    mostrar_polinomio(p)

    locales, suma, maximo = errores(xs, ys, p)
    mostrar_errores(xs, locales, suma, maximo)

    graficar(p, xs, ys)


if __name__ == "__main__":
    main()
