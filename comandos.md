# Listado de comandos para escribir ecuaciones

Guía de sintaxis según qué módulo use cada script para interpretar `f(x)` o `g(x)`.

---

## 1. Con `math` (usado en `algoritmo de biseccion.py` y `metodo del punto fijo.py`)

| Operación | Sintaxis |
|---|---|
| Potencia | `x**2`, `x**3` |
| Raíz cuadrada | `math.sqrt(x)` |
| **Raíz cúbica** | `x**(1/3)` |
| **Raíz de cualquier índice n** | `x**(1/n)` → ej: raíz 5ta: `x**(1/5)` |
| Exponencial (eˣ) | `math.exp(x)` |
| Logaritmo natural | `math.log(x)` |
| Logaritmo base 10 | `math.log10(x)` |
| Logaritmo base b | `math.log(x, b)` |
| Seno / Coseno / Tangente | `math.sin(x)`, `math.cos(x)`, `math.tan(x)` |
| Arcoseno / Arcocoseno / Arcotangente | `math.asin(x)`, `math.acos(x)`, `math.atan(x)` |
| Valor absoluto | `abs(x)` |
| Número π | `math.pi` |
| Número e | `math.e` |
| Factorial | `math.factorial(x)` |

**Ejemplos:**
- `x**(1/3) - 2` → raíz cúbica de x, igualada a 2
- `math.exp(x) - 3*x`

⚠️ **Cuidado con `x**(1/n)` y valores negativos de x**: en Python, `(-8)**(1/3)` da un número complejo (o error), no `-2`. Si necesitás raíces de números negativos con índice impar, usá:
```python
math.copysign(abs(x)**(1/n), x)
```

---

## 2. Con `numpy` (si se agrega `np` al contexto de `eval()`)

| Operación | Sintaxis |
|---|---|
| Potencia | `x**2` |
| Raíz cuadrada | `np.sqrt(x)` |
| **Raíz cúbica** | `np.cbrt(x)` (soporta negativos correctamente) |
| **Raíz de cualquier índice n** | `np.power(x, 1/n)` o `x**(1/n)` |
| Exponencial | `np.exp(x)` |
| Logaritmo natural | `np.log(x)` |
| Logaritmo base 10 | `np.log10(x)` |
| Seno / Coseno / Tangente | `np.sin(x)`, `np.cos(x)`, `np.tan(x)` |
| Arcoseno / Arcocoseno / Arcotangente | `np.arcsin(x)`, `np.arccos(x)`, `np.arctan(x)` |
| Valor absoluto | `np.abs(x)` |
| Número π | `np.pi` |
| Número e | `np.e` |

**Ejemplos:**
- `np.power(x, 1/5) - 1` → raíz 5ta de x, igualada a 1
- `np.cbrt(x) + 2`

⚠️ Los scripts `algoritmo de biseccion.py` y `metodo del punto fijo.py` solo pasan `math` al contexto de `eval()`. Para poder escribir `np.` hay que agregar `"np": np` (importando `numpy as np`) al diccionario de contexto en la función `f()` / `crear_funcion()`.

---

## 3. Con `sympy` (usado en `newton raphson.py` y `aitken.py`, sin prefijo)

| Operación | Sintaxis |
|---|---|
| Potencia | `x**2` |
| Raíz cuadrada | `sqrt(x)` |
| **Raíz cúbica** | `cbrt(x)` o `x**(sp.Rational(1,3))` (al escribir la expresión, poner `x**(1/3)`) |
| **Raíz de cualquier índice n** | `root(x, n)` → ej: raíz 5ta: `root(x, 5)` |
| Exponencial | `exp(x)` |
| Logaritmo natural | `log(x)` |
| Logaritmo base b | `log(x, b)` |
| Seno / Coseno / Tangente | `sin(x)`, `cos(x)`, `tan(x)` |
| Arcoseno / Arcocoseno / Arcotangente | `asin(x)`, `acos(x)`, `atan(x)` |
| Valor absoluto | `Abs(x)` |
| Número π | `pi` |
| Número e | `E` |

**Ejemplos:**
- `root(x, 4) - 2` → raíz cuarta de x, igualada a 2
- `sin(x) - x/2`

✅ **Ventaja de `sympy`**: `root(x, n)` maneja bien raíces de índice impar con valores negativos, y además Newton-Raphson/Aitken derivan la expresión simbólicamente, así que no hay que preocuparse por errores numéricos de dominio como en `math`/`numpy`.

---

## Resumen: ¿qué usa cada script?

| Script | Módulo | Prefijo |
|---|---|---|
| `newton raphson.py` | sympy | sin prefijo (`sin(x)`, `root(x, n)`, `pi`) |
| `aitken.py` | sympy | sin prefijo |
| `algoritmo de biseccion.py` | math | `math.` (agregar `np` manualmente si se quiere) |
| `metodo del punto fijo.py` | math | `math.` (agregar `np` manualmente si se quiere) |

## Tabla rápida: raíz n-ésima de x en cada módulo

| Módulo | Sintaxis |
|---|---|
| `math` | `x**(1/n)` |
| `numpy` | `np.power(x, 1/n)` (o `np.cbrt(x)` si n=3) |
| `sympy` | `root(x, n)` |