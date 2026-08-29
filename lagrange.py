"""
Interpolación de Lagrange
-------------------------
Dado un conjunto de n nodos (x_i, y_i) ingresados por el usuario, el script:
  1. Pide los nodos de a pares "x,y" (línea vacía = fin de la carga).
  2. Muestra cada base de Lagrange L_i(x), factorizada y desarrollada,
     siempre en fracciones exactas (nunca en decimal).
  3. Desarrolla el polinomio de Lagrange P(x) = sum(y_i * L_i(x)) y lo
     reduce a su mínima expresión (términos agrupados por grado).
  4. Si se le da la función original f(x) y un punto x, calcula la cota
     de error de interpolación:
        |f(x) - P(x)| <= M/(n+1)! * |prod_{i=0}^{n} (x - x_i)|
     con M = máx |f^(n+1)(ξ)| en el intervalo que contiene los nodos y x.
     Si no se ingresa f(x) (línea vacía), se omite este cálculo.

Requiere: sympy, matplotlib, numpy
    pip install sympy matplotlib numpy
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x = sp.symbols('x')


def parsear_fraccion(texto):
    """Convierte un string ('3', '-1/2', '2.5', 'pi', 'pi/2', etc.) en un valor
    exacto de sympy: racionaliza los decimales pero deja pi simbólico (sin
    aproximar a decimal)."""
    return sp.nsimplify(sp.sympify(texto), rational=True)


def pedir_nodos():
    """Pide pares x,y hasta que se ingrese una línea vacía. Devuelve listas (xs, ys) de Rational."""
    xs, ys = [], []
    print("Ingresá los nodos como 'x,y' (podés usar fracciones o pi, ej: 1/2,3  o  pi/2,-1). Línea vacía para terminar.\n")
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
    (o con pi) entre paréntesis, ej: (-1/12)*x**3 o (pi/2)*x**2, para que no se
    confunda con x**(3/12)."""
    poli = sp.Poly(expr, var)
    terminos = []
    for (grado,), coef in poli.terms():
        coef = sp.nsimplify(coef, rational=True)
        signo = "-" if coef.could_extract_minus_sign() else "+"
        coef_abs = -coef if signo == "-" else coef

        if coef_abs.is_Integer:
            coef_str = str(coef_abs)
        else:
            coef_str = f"({coef_abs})"

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


def pedir_funcion_original():
    """Pide f(x) para poder acotar el error. Línea vacía -> se omite el cálculo."""
    texto = input("\nFunción original f(x) para la cota de error (opcional, Enter para omitir): ").strip()
    if texto == "":
        return None
    try:
        return sp.sympify(texto)
    except (sp.SympifyError, TypeError):
        print("  -> Expresión inválida, se omite el cálculo de la cota de error.")
        return None


def pedir_x_evaluacion():
    """Pide el punto x donde se quiere estimar la cota de error."""
    while True:
        texto = input("Valor de x en el que querés estimar la cota de error: ").strip()
        try:
            return parsear_fraccion(texto)
        except (ValueError, sp.SympifyError, TypeError):
            print("  -> Valor inválido, probá de nuevo (ej: 2, 1/2, pi/3).")


def maximo_productoria(xs, a, b):
    """Máximo de |g(t)| en [a, b], con g(t) = prod_{i=0}^{n} (t - x_i).
    Se halla igualando g'(t) = 0, tomando las raíces reales dentro de [a, b]
    y comparando |g| en esas raíces junto con los extremos del intervalo."""
    t = sp.symbols('t')
    g = sp.expand(sp.Mul(*[t - xi for xi in xs]))
    gp = sp.diff(g, t)

    candidatos = [a, b]
    coeficientes = sp.Poly(gp, t).all_coeffs()
    if len(coeficientes) > 1:
        raices = np.roots([float(c) for c in coeficientes])
        for r in raices:
            if abs(r.imag) < 1e-9 and a - 1e-9 <= r.real <= b + 1e-9:
                candidatos.append(float(r.real))

    g_num = sp.lambdify(t, g, "numpy")
    punto = max(candidatos, key=lambda c: abs(float(g_num(c))))
    valor = abs(float(g_num(punto)))
    return punto, valor


def cota_error(f_expr, xs, x_eval):
    """Cota de error de interpolación:
        |f(x) - P(x)| <= M/(n+1)! * max|prod_{i=0}^{n} (x - x_i)|
    con M = máx |f^(n+1)(ξ)| en el intervalo que contiene los nodos y x
    (estimado numéricamente por muestreo denso del intervalo), y el máximo
    de la productoria hallado igualando su derivada a 0 y comparando raíces."""
    orden = len(xs)  # n+1, con n = grado del polinomio = len(xs) - 1
    derivada = sp.simplify(sp.diff(f_expr, x, orden))

    extremos = [float(v) for v in xs + [x_eval]]
    a, b = min(extremos), max(extremos)

    derivada_num = sp.lambdify(x, derivada, "numpy")
    malla = np.linspace(a, b, 20000)
    valores = np.abs(derivada_num(malla) * np.ones_like(malla))
    M = float(np.max(valores))

    punto_producto, max_producto = maximo_productoria(xs, a, b)
    factorial = sp.factorial(orden)
    cota = M / float(factorial) * max_producto

    return {
        "orden": orden,
        "derivada": derivada,
        "intervalo": (a, b),
        "M": M,
        "punto_producto": punto_producto,
        "max_producto": max_producto,
        "factorial": factorial,
        "cota": cota,
    }


def mostrar_cota_error(x_eval, r):
    print("\n=== Cota de error de interpolación ===")
    print(f"Punto x solicitado: {x_eval}")
    print(f"Orden de la derivada: n+1 = {r['orden']}")
    print(f"f^({r['orden']})(x) = {r['derivada']}")
    print(f"Intervalo considerado (nodos y x): [{r['intervalo'][0]:.6g}, {r['intervalo'][1]:.6g}]")
    print(f"M = max |f^({r['orden']})(xi)| en el intervalo (estimado numericamente) ~= {r['M']:.6g}")
    print(f"max |productoria (x - x_i), i=0..n| en el intervalo, hallado en x ~= {r['punto_producto']:.6g}"
          f"  ->  ~= {r['max_producto']:.6g}")
    print(f"(n+1)! = {r['factorial']}")
    print(f"\n|f(x) - P(x)| <= M/(n+1)! * max|producto|  ~=  {r['M']:.6g}/{r['factorial']} * {r['max_producto']:.6g}"
          f"  ~=  {r['cota']:.6g}")


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

    f_expr = pedir_funcion_original()
    if f_expr is not None:
        x_eval = pedir_x_evaluacion()
        resultado = cota_error(f_expr, xs, x_eval)
        mostrar_cota_error(x_eval, resultado)

    graficar(p, xs, ys)


if __name__ == "__main__":
    main()
