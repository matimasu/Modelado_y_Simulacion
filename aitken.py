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

  5. Imprime una tabla con numeración CONTINUA (sin repetir filas): cada
     valor calculado (por punto fijo o por Aitken) tiene su propio número
     de fila único, una columna indica si se calculó como "Punto Fijo" o
     como "Aitken", y todos los valores se muestran con 10 cifras
     decimales. El x̂ de Aitken de una ronda pasa a ser la semilla de la
     siguiente sin volver a imprimirse como fila aparte.
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

    En vez de reiniciar la numeración en cada ronda, cada valor calculado
    (ya sea por punto fijo x_(n+1)=g(x_n), o por la aceleración de Aitken)
    se numera de forma CONTINUA y única (1, 2, 3, 4, ...), sin repetir
    ningún valor: el x̂ de Aitken de una ronda pasa a ser la semilla de la
    siguiente sin volver a imprimirse como fila aparte.

    Devuelve:
        filas       : lista de [n, tipo, valor, error], donde tipo es
                      "Punto Fijo" o "Aitken", y error es la diferencia
                      absoluta contra el valor inmediatamente anterior.
        x_hat_final : el mejor valor acelerado encontrado.
    """
    filas = []
    n = 0
    x_semilla = x0
    valor_anterior = x0
    x_hat_final = x0

    for _ in range(max_iter):
        x_n = x_semilla

        x_n1 = g_num(x_n)
        n += 1
        error = abs(x_n1 - valor_anterior)
        filas.append([n, "Punto Fijo", x_n1, error])
        valor_anterior = x_n1

        x_n2 = g_num(x_n1)
        n += 1
        error = abs(x_n2 - valor_anterior)
        filas.append([n, "Punto Fijo", x_n2, error])
        valor_anterior = x_n2

        denom = x_n2 - 2 * x_n1 + x_n

        if denom == 0:
            # Evita división por cero: si ya convergió exactamente, cortamos.
            x_hat = x_n2
            n += 1
            error = abs(x_hat - valor_anterior)
            filas.append([n, "Aitken", x_hat, error])
            x_hat_final = x_hat
            break

        x_hat = x_n - (x_n1 - x_n) ** 2 / denom
        n += 1
        error = abs(x_hat - valor_anterior)
        filas.append([n, "Aitken", x_hat, error])
        valor_anterior = x_hat
        x_hat_final = x_hat

        if error < tol:
            break

        # La siguiente ronda arranca desde el valor acelerado (Aitken
        # reiniciado), sin volver a imprimirlo como fila nueva.
        x_semilla = x_hat

    return filas, x_hat_final


def mostrar_tabla(filas):
    """
    Muestra la tabla de iteraciones con numeración continua (sin repetir
    filas), una columna que indica si el valor se calculó por Punto Fijo
    o por Aitken, y todos los valores con 10 cifras decimales.
    """
    filas_fmt = [
        [n, tipo, f"{valor:.10f}", f"{error:.10f}"]
        for n, tipo, valor, error in filas
    ]
    headers = ["n", "Tipo", "Valor", "Error"]
    print("\nTabla de iteraciones:")
    print(tabulate(filas_fmt, headers=headers, tablefmt="grid", disable_numparse=True))


def graficar(g_num, filas, raiz, x0):
    # Puntos relevantes para definir el rango del gráfico (valores acelerados de Aitken)
    puntos = [x0, raiz] + [valor for _, tipo, valor, _ in filas if tipo == "Aitken"]
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
    rondas = sum(1 for _, tipo, _, _ in filas if tipo == "Aitken")
    x_actual = x0
    cx, cy = [x_actual], [x_actual]
    for _ in range(min(rondas * 2, 40)):
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

    filas, raiz = aitken(g_num, x0, tol, max_iter)

    if not filas:
        print("No se realizó ninguna iteración.")
        return

    mostrar_tabla(filas)

    rondas = sum(1 for _, tipo, _, _ in filas if tipo == "Aitken")
    print(f"\nResultado final (Aitken): x ≈ {raiz:.10f}")
    print(f"Rondas de Aitken realizadas: {rondas}  (total de filas calculadas: {len(filas)})")

    graficar(g_num, filas, raiz, x0)


if __name__ == "__main__":
    main()