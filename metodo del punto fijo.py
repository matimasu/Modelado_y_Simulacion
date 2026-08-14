"""
Metodo del Punto Fijo (f(x) opcional, con tabla y grafico)
-------------------------------------------------------------
Se busca la raiz de x = g(x) iterando:
    x_(n+1) = g(x_n)
hasta que |x_n - x_(n-1)| < tolerancia, o se llegue al numero maximo
de iteraciones.

Se pide g(x), x0, tolerancia e iteraciones maximas. f(x) es OPCIONAL:
si no la conoces, dejala vacia (Enter) y el programa sigue sin ella.

Se muestra la tabla de iteraciones y la raiz aproximada, y se grafica
g(x) y la recta y=x (mas f(x), solo si fue ingresada).

Las funciones se escriben con sintaxis de Python, pudiendo usar el
modulo math, por ejemplo:
    f(x) = math.exp(x) - 2*x - 1
    g(x) = (math.exp(x) - 1) / 2

Requiere: numpy, pandas, matplotlib, tabulate
    pip install numpy pandas matplotlib tabulate
"""

import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def crear_funcion(expresion):
    """
    Convierte un string tipo "math.exp(x) - 2*x - 1" en una funcion
    Python evaluable, usando el modulo math como contexto.
    """
    contexto = {'math': math}

    def f(valor_x):
        contexto_local = dict(contexto)
        contexto_local['x'] = valor_x
        return eval(expresion, {"__builtins__": {}}, contexto_local)

    return f


def sustituir_x(expresion, valor_x):
    """
    Devuelve el string de la expresion con la variable 'x' reemplazada
    por su valor numerico (solo para mostrar en la tabla), respetando
    que 'x' no forme parte de otras palabras (ej: no toca 'math.exp').
    """
    return re.sub(r'\bx\b', f'({valor_x})', expresion)


def pedir_float(mensaje):
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print("  -> Ingresa un numero valido.")


def pedir_int(mensaje):
    while True:
        try:
            return int(input(mensaje))
        except ValueError:
            print("  -> Ingresa un numero entero valido.")


def pedir_parametros():
    print("=== Metodo del Punto Fijo ===")
    print("Escribi f(x) y g(x) con sintaxis de Python.")
    print("Podes usar el modulo 'math', ej: math.exp(x), math.sqrt(x), math.sin(x), math.log(x)")
    print("f(x) es opcional: si no la conoces, dejala vacia y apreta Enter.\n")

    f_str = input("f(x) (opcional, Enter para omitir) = ").strip()
    if f_str == "":
        f_str = None

    g_str = input("g(x) = ")
    x0 = pedir_float("x0 (valor inicial) = ")
    tol = pedir_float("Tolerancia (ej: 1e-6) = ")
    max_iter = pedir_int("Numero maximo de iteraciones = ")

    return f_str, g_str, x0, tol, max_iter


def punto_fijo(g_str, x0, tol=1e-6, max_iter=100):
    """
    Ejecuta el metodo del punto fijo (solo necesita g(x)).

    Retorna
    -------
    df   : pandas.DataFrame con las iteraciones
    raiz : aproximacion final de la raiz
    g    : funcion evaluable g(x)
    """
    g = crear_funcion(g_str)

    filas = []
    x_anterior = x0
    n = 0

    while True:
        n += 1
        g_sustituida = sustituir_x(g_str, x_anterior)
        x_actual = g(x_anterior)
        error = abs(x_actual - x_anterior)

        filas.append({
            'n': n,
            'x_(n-1)': x_anterior,
            'g(x_(n-1))': g_sustituida,
            'x_n': x_actual,
            'error': error
        })

        x_anterior = x_actual

        if error < tol or n >= max_iter:
            break

    df = pd.DataFrame(filas)
    raiz = x_anterior
    return df, raiz, g


def graficar(g, raiz, x0, df, f_str=None, rango_extra=2):
    """
    Grafica g(x) y la recta y = x (la recta identidad, que pasa por
    (1,1), (2,2), etc.), marcando la raiz encontrada. Si f_str no es
    None, tambien grafica f(x) y usa f(raiz) como altura del punto
    marcado; si es None, el punto se marca en (raiz, raiz), sobre la
    interseccion de g(x) con y=x (definicion geometrica de punto fijo).
    """
    valores_x = list(df['x_(n-1)']) + list(df['x_n']) + [x0, raiz]
    x_min = min(valores_x) - rango_extra
    x_max = max(valores_x) + rango_extra

    xs = np.linspace(x_min, x_max, 400)

    g_vals = np.array([g(v) for v in xs])

    fig, ax = plt.subplots(figsize=(8, 6))

    f = None
    if f_str:
        f = crear_funcion(f_str)
        f_vals = np.array([f(v) for v in xs])
        ax.plot(xs, f_vals, label='f(x)', color='tab:blue')

    ax.plot(xs, g_vals, label='g(x)', color='tab:green')
    ax.plot(xs, xs, label='y = x', color='gray', linestyle='--')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)

    altura_raiz = f(raiz) if f is not None else raiz
    ax.scatter([raiz], [altura_raiz], color='red', zorder=5,
               label=f'raiz aprox. x = {raiz:.6f}')

    ax.set_title('Metodo del Punto Fijo')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Se fija el eje Y con el mismo rango que el eje X para que la
    # recta y=x se vea correctamente como la diagonal a 45 grados
    # (evita que quede "aplastada" cuando g(x) o f(x) crecen mucho
    # mas rapido que x, como pasa con math.exp).
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect('equal', adjustable='box')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grafico_punto_fijo.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    f_str, g_str, x0, tol, max_iter = pedir_parametros()

    df, raiz, g = punto_fijo(g_str, x0, tol, max_iter)

    # Formatea los valores numericos de la tabla antes de mostrarla
    df_mostrar = df.copy()
    df_mostrar['x_(n-1)'] = df_mostrar['x_(n-1)'].map(lambda v: f'{v:.8f}')
    df_mostrar['x_n'] = df_mostrar['x_n'].map(lambda v: f'{v:.8f}')
    df_mostrar['error'] = df_mostrar['error'].map(lambda v: f'{v:.2e}')

    print(f"\nf(x) = {f_str if f_str else '(no ingresada)'}")
    print(f"g(x) = {g_str}")
    print(f"x0   = {x0}")
    print(f"tolerancia = {tol}, max_iter = {max_iter}\n")
    print(tabulate(df_mostrar, headers='keys', tablefmt='fancy_grid', showindex=False))
    print(f"\nRaiz aproximada: x = {raiz:.8f}")
    if f_str:
        f = crear_funcion(f_str)
        print(f"f(raiz) = {f(raiz):.2e}")

    graficar(g, raiz, x0, df, f_str=f_str)