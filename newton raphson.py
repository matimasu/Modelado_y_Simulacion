"""
Método de Newton-Raphson
-------------------------
Dado f(x) ingresada por el usuario, el script:
  1. Deriva f(x) simbólicamente con sympy y la muestra en pantalla.
  2. Pide semilla (x0), tolerancia y número máximo de iteraciones.
  3. Ejecuta el método e imprime una tabla con cada iteración.
  4. Grafica f(x), la raíz encontrada y la trayectoria de las
     aproximaciones (x0, x1, x2, ...).

Requiere: sympy, matplotlib, tabulate
    pip install sympy matplotlib tabulate
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def obtener_funcion():
    """Pide la función al usuario y devuelve la expresión simbólica y f, f' numéricas."""
    x = sp.symbols('x')
    while True:
        expr_str = input("Ingresá f(x) (ej: x**2 - 2, sin(x) - x/2, exp(x) - 3*x): ")
        try:
            f_expr = sp.sympify(expr_str)
            break
        except (sp.SympifyError, TypeError):
            print("  -> Expresión inválida, probá de nuevo (usá sintaxis tipo Python: **, *, sin, cos, exp, log, etc).")

    df_expr = sp.diff(f_expr, x)

    print("\nFunción ingresada:     f(x) =", sp.simplify(f_expr))
    print("Derivada calculada:    f'(x) =", sp.simplify(df_expr))
    print()

    f_num = sp.lambdify(x, f_expr, "numpy")
    df_num = sp.lambdify(x, df_expr, "numpy")
    return f_expr, df_expr, f_num, df_num


def pedir_parametros():
    x0 = float(input("Semilla inicial x0: "))
    tol = float(input("Tolerancia (ej: 1e-6): "))
    max_iter = int(input("Número máximo de iteraciones: "))
    return x0, tol, max_iter


def newton_raphson(f_num, df_num, x0, tol, max_iter):
    """Ejecuta Newton-Raphson y devuelve la lista de iteraciones (tabla) y la raíz final."""
    tabla = []
    x_actual = x0

    for i in range(0, max_iter + 1):
        f_val = f_num(x_actual)
        df_val = df_num(x_actual)

        if df_val == 0:
            print(f"\n⚠ f'(x) = 0 en x = {x_actual:.6f}. El método no puede continuar.")
            break

        x_siguiente = x_actual - f_val / df_val
        error_abs = abs(x_siguiente - x_actual)

        # Error relativo porcentual: |x_(n+1) - x_n| / |x_(n+1)| * 100
        if x_siguiente != 0:
            error_rel = abs((x_siguiente - x_actual) / x_siguiente) * 100
        else:
            error_rel = float("nan")

        tabla.append([i, x_actual, f_val, df_val, x_siguiente, error_abs, error_rel])

        if error_abs < tol:
            x_actual = x_siguiente
            break

        x_actual = x_siguiente

    return tabla, x_actual


def mostrar_tabla(tabla):
    headers = [
        "n", "x_n", "f(x_n)", "f'(x_n)", "x_(n+1)",
        "Error absoluto", "Error relativo porcentual"
    ]
    print("\nTabla de iteraciones:")
    print(tabulate(tabla, headers=headers, floatfmt=".8f", tablefmt="grid"))


def graficar(f_num, tabla, raiz, x0):
    xs_iter = [x0] + [fila[4] for fila in tabla]  # x0, x1, x2, ... incluyendo la raíz final
    ys_iter = [f_num(x) for x in xs_iter]

    margen = max(1.0, abs(raiz - x0) * 1.5)
    x_min = min(xs_iter) - margen
    x_max = max(xs_iter) + margen

    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = f_num(x_vals)

    plt.figure(figsize=(9, 6))
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.plot(x_vals, y_vals, label="f(x)", color="steelblue", linewidth=2)

    # Trayectoria de las iteraciones
    plt.plot(xs_iter, ys_iter, "o--", color="darkorange", label="Iteraciones", markersize=5)

    # Punto inicial y raíz final
    plt.plot(x0, f_num(x0), "s", color="green", markersize=8, label=f"x0 = {x0:.4f}")
    plt.plot(raiz, f_num(raiz), "*", color="red", markersize=14, label=f"Raíz ≈ {raiz:.6f}")

    plt.title("Método de Newton-Raphson")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("newton_raphson_resultado.png", dpi=150)
    print("\nGráfico guardado como 'newton_raphson_resultado.png'")
    plt.show()


def main():
    print("=== Método de Newton-Raphson ===\n")
    f_expr, df_expr, f_num, df_num = obtener_funcion()
    x0, tol, max_iter = pedir_parametros()

    tabla, raiz = newton_raphson(f_num, df_num, x0, tol, max_iter)

    if not tabla:
        print("No se realizó ninguna iteración.")
        return

    mostrar_tabla(tabla)

    print(f"\nResultado final: x ≈ {raiz:.8f}  (f(x) ≈ {f_num(raiz):.2e})")
    print(f"Iteraciones realizadas: {len(tabla)}")

    graficar(f_num, tabla, raiz, x0)


if __name__ == "__main__":
    main()