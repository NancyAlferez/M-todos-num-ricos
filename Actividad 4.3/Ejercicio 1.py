import numpy as np
import matplotlib.pyplot as plt

def simpson_rule(f, a, b, n):
    """Aproxima la integral de f(x) en [a, b] usando la regla de Simpson."""
    if n % 2 == 1:
        raise ValueError("El número de subintervalos (n) debe ser par.")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    fx = f(x)
    
    integral = (h / 3) * (fx[0] + 2 * np.sum(fx[2:n:2]) + 4 * np.sum(fx[1:n:2]) + fx[n])
    
    return integral

#  Datos del problema
k = 200  # constante del resorte

# Función del trabajo: f(x) = kx
def funcion(x):
    return k * x

# Intervalo
a, b = 0.1, 0.3

# Valores de n solicitados
valores_n = [6, 10, 20, 30]

#  Solución analítica
# ∫ kx dx = (k/2)(b^2 - a^2)
analitica = (k / 2) * (b**2 - a**2)
print(f"Solución analítica: {analitica}")

#  Cálculo con Simpson
for n in valores_n:
    resultado = simpson_rule(funcion, a, b, n)
    error = abs(analitica - resultado)
    print(f"n = {n} -> Aproximación: {resultado} | Error: {error}")

#  Gráfica
x_vals = np.linspace(a, b, 100)
y_vals = funcion(x_vals)

plt.plot(x_vals, y_vals, label=r"$f(x) = kx$", color="blue")
plt.fill_between(x_vals, y_vals, alpha=0.3, color="cyan", label="Área (Trabajo)")
plt.scatter(np.linspace(a, b, 10), funcion(np.linspace(a, b, 10)), color="red", label="Puntos")
plt.xlabel("x")
plt.ylabel("Fuerza")
plt.legend()
plt.title("Trabajo realizado por un resorte (Regla de Simpson)")
plt.grid()

plt.savefig("resorte_simpson.png")
plt.show()
