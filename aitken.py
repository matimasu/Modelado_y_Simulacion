"""
Método de Aitken (Delta cuadrado de Aitken)
--------------------------------------------
El usuario ingresa g(x) para la iteración de punto fijo:
    x_(n+1) = g(x_n)

El script:
  1. Muestra g(x) ingresada (parseada simbólicamente con sympy).
  2. Pide semilla (x0), tolerancia y número máximo de iteraciones.
  3. Genera la sucesión de punto fijo x0, x1, x2, ...
  4. Aplica la aceleración de Aitken sobre cada tripla (x_n, x_n+1, x_n+2):

         x_hat_n = x_n - (x_(n+1) - x_n)^2 / (x_(n+2) - 2*x_(n+1) + x_n)

  5. Imprime una tabla con todas las iteraciones (sucesión original
     y sucesión acelerada de Aitken).
  6. Grafica g(x), la recta y=x (para visualizar el punto fijo),
     la trayectoria de iteración tipo "telaraña" (cobweb) y los
     valores acelerados de Aitken.

Requiere: sympy, matplotlib, tabulate
    pip install sympy matplotlib tabulate
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def obtener_funcion():
    """Pide g(x) al usuario y devuelve la expresión simbólica y la función numérica."""
    x = sp.symbols('x')
    while True:
        expr_str = input("Ingresá g(x) para x = g(x) (ej: (x + 2/x)/2, cos(x), (x**2+2)/3): ")
        try:
            g_expr = sp.sympify(expr_str)
            break
        except (sp.SympifyError, TypeError):
            print("  -> Expresión inválida, probá de nuevo (usá sintaxis tipo Python: **, *, sin, cos, exp, log, etc).")

    print("\nFunción de iteración ingresada:   g(x) =", sp.simplify(g_expr))
    print()

    g_num = sp.lambdify(x, g_expr, "numpy")
    return g_expr, g_num


def pedir_parametros():
    x0 = float(input("Semilla inicial x0: "))
    tol = float(input("Tolerancia (ej: 1e-6): "))
    max_iter = int(input("Número máximo de iteraciones: "))
    return x0, tol, max_iter


def aitken(g_num, x0, tol, max_iter):
    """
    Genera la sucesión de punto fijo y aplica Aitken en cada tripla disponible.
    Devuelve la tabla de iteraciones y el mejor valor acelerado encontrado.
    """
    tabla = []
    xs = [x0]  # sucesión de punto fijo pura
    x_hat_final = x0

    for i in range(max_iter):
        x_n = xs[-1]
        x_n1 = g_num(x_n)
        xs.append(x_n1)

        x_n2 = g_num(x_n1)
        xs.append(x_n2)

        denom = x_n2 - 2 * x_n1 + x_n

        if denom == 0:
            # Evita división por cero: si ya convergió exactamente, cortamos.
            x_hat = x_n2
            error = abs(x_hat - x_hat_final)
            tabla.append([i + 1, x_n, x_n1, x_n2, "—", error])
            x_hat_final = x_hat
            break

        x_hat = x_n - (x_n1 - x_n) ** 2 / denom
        error = abs(x_hat - x_hat_final)

        tabla.append([i + 1, x_n, x_n1, x_n2, x_hat, error])

        x_hat_final = x_hat

        if error < tol:
            break

        # La siguiente iteración vuelve a arrancar desde el valor acelerado
        # (esto es lo que se conoce como "Aitken reiniciado" y mejora la velocidad)
        xs = [x_hat]

    return tabla, x_hat_final


def mostrar_tabla(tabla):
    headers = ["n", "x_n", "x_(n+1)=g(x_n)", "x_(n+2)=g(x_n+1)", "x_hat (Aitken)", "Error"]
    print("\nTabla de iteraciones:")
    print(tabulate(tabla, headers=headers, floatfmt=".8f", tablefmt="grid"))


def graficar(g_num, tabla, raiz, x0):
    # Puntos relevantes para definir el rango del gráfico
    puntos = [x0, raiz] + [fila[4] for fila in tabla if isinstance(fila[4], float)]
    margen = max(1.0, (max(puntos) - min(puntos)) * 1.5 if len(puntos) > 1 else 2.0)
    x_min = min(puntos) - margen
    x_max = max(puntos) + margen

    x_vals = np.linspace(x_min, x_max, 400)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_vals = g_num(x_vals)
    y_vals = np.where(np.isfinite(y_vals), y_vals, np.nan)

    # Recorta el eje Y a la zona relevante (evita que asíntotas rompan la escala)
    y_ref = [g_num(p) for p in puntos if np.isfinite(g_num(p))]
    if y_ref:
        y_margen = max(1.0, (max(y_ref) - min(y_ref)) * 2.0)
        y_min = min(y_ref + puntos) - y_margen
        y_max = max(y_ref + puntos) + y_margen

    plt.figure(figsize=(9, 6))
    plt.plot(x_vals, y_vals, label="g(x)", color="steelblue", linewidth=2)
    plt.plot(x_vals, x_vals, "--", color="gray", linewidth=1, label="y = x")

    # Trayectoria tipo "telaraña" (cobweb plot) usando la sucesión sin acelerar
    x_actual = x0
    cx, cy = [x_actual], [x_actual]
    for _ in range(min(len(tabla) * 2, 40)):
        x_sig = g_num(x_actual)
        cx += [x_actual, x_sig]
        cy += [x_sig, x_sig]
        x_actual = x_sig

    plt.plot(cx, cy, color="darkorange", linewidth=1, alpha=0.7, label="Iteración (telaraña)")

    # Punto inicial y raíz final (acelerada por Aitken)
    plt.plot(x0, g_num(x0), "s", color="green", markersize=8, label=f"x0 = {x0:.4f}")
    plt.plot(raiz, g_num(raiz), "*", color="red", markersize=14, label=f"Punto fijo ≈ {raiz:.6f}")

    plt.title("Método de Aitken (Δ²) sobre iteración de punto fijo")
    plt.xlabel("x")
    plt.ylabel("g(x)")
    if y_ref:
        plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("aitken_resultado.png", dpi=150)
    print("\nGráfico guardado como 'aitken_resultado.png'")
    plt.show()


def main():
    print("=== Método de Aitken (Delta cuadrado) ===\n")
    g_expr, g_num = obtener_funcion()
    x0, tol, max_iter = pedir_parametros()

    tabla, raiz = aitken(g_num, x0, tol, max_iter)

    if not tabla:
        print("No se realizó ninguna iteración.")
        return

    mostrar_tabla(tabla)

    print(f"\nResultado final (Aitken): x ≈ {raiz:.8f}")
    print(f"Iteraciones realizadas: {len(tabla)}")

    graficar(g_num, tabla, raiz, x0)


if __name__ == "__main__":
    main()