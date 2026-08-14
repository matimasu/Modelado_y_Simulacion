"""
Método de Bisección para encontrar raíces de una función.

Criterios de parada:
    - Tolerancia mínima (en base 10, ej: 10^-3)
    - Número máximo de iteraciones

Incluye graficación del intervalo, la función y la raíz encontrada.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def biseccion(f, a, b, tol=1e-3, max_iter=100):
    """
    Encuentra una raíz de f(x) = 0 en el intervalo [a, b] usando bisección.

    Parámetros:
        f (callable): función a evaluar, f(x)
        a (float): extremo izquierdo del intervalo
        b (float): extremo derecho del intervalo
        tol (float): tolerancia mínima para el criterio de parada (ej: 1e-3)
        max_iter (int): número máximo de iteraciones permitidas

    Retorna:
        (raiz, iteraciones, historial)
    """
    fa = f(a)
    fb = f(b)

    if fa == 0:
        return a, 0, [(0, a, b, fa, fb, a, fa, 0.0)]
    if fb == 0:
        return b, 0, [(0, a, b, fa, fb, b, fb, 0.0)]

    if fa * fb > 0:
        raise ValueError(
            f"No se garantiza una raíz en [{a}, {b}]: f(a) y f(b) tienen el mismo signo."
        )

    historial = []
    iteracion = 0
    c = a
    c_anterior = None

    while iteracion < max_iter:
        c = (a + b) / 2
        fc = f(c)

        # Error: diferencia relativa entre la c actual y la anterior.
        # En la primera iteración no hay c anterior, se usa el semiancho del intervalo.
        if c_anterior is None:
            error = abs(b - a) / 2
        else:
            error = abs(c - c_anterior)

        historial.append((iteracion + 1, a, b, fa, fb, c, fc, error))

        # Criterio de parada por tolerancia
        if abs(fc) < tol or error < tol:
            return c, iteracion + 1, historial

        c_anterior = c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iteracion += 1

    print(f"Aviso: se alcanzó el número máximo de iteraciones ({max_iter}) sin cumplir la tolerancia.")
    return c, iteracion, historial


def mostrar_historial(historial):
    encabezado = (
        f"{'Iter':<6}{'a':<14}{'b':<14}{'f(a)':<14}{'f(b)':<14}"
        f"{'c=(a+b)/2':<14}{'f(c)':<14}{'Error':<14}"
    )
    print(encabezado)
    print("-" * len(encabezado))
    for it, a, b, fa, fb, c, fc, error in historial:
        print(
            f"{it:<6}{a:<14.6f}{b:<14.6f}{fa:<14.6f}{fb:<14.6f}"
            f"{c:<14.6f}{fc:<14.6f}{error:<14.6f}"
        )


def graficar_biseccion(f, a_ini, b_ini, raiz, historial, guardar=None):
    """
    Grafica f(x) en el intervalo original [a_ini, b_ini], marcando:
      - los extremos iniciales a y b
      - la raíz aproximada encontrada
      - la evolución de los puntos medios "c" evaluados en cada iteración

    Parámetros:
        f (callable): la función f(x)
        a_ini, b_ini (float): extremos originales del intervalo
        raiz (float): raíz aproximada devuelta por biseccion()
        historial (list): historial devuelto por biseccion()
        guardar (str|None): si se pasa una ruta, guarda la figura en ese archivo
    """
    # Un poco de margen alrededor del intervalo para que se vea mejor
    margen = 0.1 * (b_ini - a_ini) if b_ini != a_ini else 1.0
    x = np.linspace(a_ini - margen, b_ini + margen, 800)
    y = np.array([f(xi) for xi in x])

    fig, ax = plt.subplots(figsize=(9, 6))

    # Curva de la función
    ax.plot(x, y, label="f(x)", color="steelblue", linewidth=2)

    # Eje x (y = 0)
    ax.axhline(0, color="black", linewidth=0.8)

    # Intervalo inicial [a, b]
    ax.axvline(a_ini, color="gray", linestyle="--", linewidth=1)
    ax.axvline(b_ini, color="gray", linestyle="--", linewidth=1)
    ax.plot([a_ini, b_ini], [f(a_ini), f(b_ini)], "o", color="gray",
            label="Extremos del intervalo [a, b]")

    # Puntos medios "c" evaluados durante las iteraciones
    cs = [fila[5] for fila in historial]
    fcs = [fila[6] for fila in historial]
    ax.plot(cs, fcs, "x", color="orange", markersize=6, label="Puntos medios evaluados")

    # Raíz encontrada
    ax.plot(raiz, f(raiz), "*", color="red", markersize=16,
            label=f"Raíz ≈ {raiz:.6f}")
    ax.axvline(raiz, color="red", linestyle=":", linewidth=1)

    ax.set_title("Método de Bisección")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if guardar:
        plt.savefig(guardar, dpi=150)
        print(f"Gráfico guardado en: {guardar}")

    plt.show()


def pedir_datos():
    print("Método de bisección")
    print("Ingresá la función en términos de 'x' (podés usar funciones de math, ej: math.sin(x), math.exp(x))")
    expresion = input("f(x) = ")

    def f(x):
        return eval(expresion, {"x": x, "math": math})

    a = float(input("Extremo a: "))
    b = float(input("Extremo b: "))

    tol_exp = input("Exponente de la tolerancia (ej: -3 para 10^-3) [default -3]: ")
    tol_exp = float(tol_exp) if tol_exp.strip() != "" else -3
    tol = 10 ** tol_exp

    max_iter_str = input("Número máximo de iteraciones [default 100]: ")
    max_iter = int(max_iter_str) if max_iter_str.strip() != "" else 100

    return f, a, b, tol, max_iter


if __name__ == "__main__":
    f, a, b, tol, max_iter = pedir_datos()
    a_ini, b_ini = a, b  # guardamos el intervalo original para el gráfico

    try:
        raiz, iteraciones, historial = biseccion(f, a, b, tol, max_iter)
        mostrar_historial(historial)
        print(f"\nRaíz aproximada: {raiz:.6f}")
        print(f"Iteraciones realizadas: {iteraciones}")
        print(f"Tolerancia usada: {tol}")

        graficar_biseccion(f, a_ini, b_ini, raiz, historial)
    except ValueError as e:
        print(f"Error: {e}")