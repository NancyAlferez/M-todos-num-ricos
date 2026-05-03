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
k = 0.5

# T(x) = 300 - 50x^2 → derivada
def funcion(x):
    return -100 * x

# Intervalo
a, b = 0, 2

# Valores de n
valores_n = [6, 10, 20, 30]

#  Solución analítica
Ta = 300 - 50 * (a**2)
Tb = 300 - 50 * (b**2)
analitica = k * (Tb - Ta)
print(f"Solución analítica: {analitica}")

#  Simpson
for n in valores_n:
    resultado = k * simpson_rule(funcion, a, b, n)
    error = abs(analitica - resultado)
    print(f"n = {n} -> Aproximación: {resultado} | Error: {error}")

#  Gráfica
x_vals = np.linspace(a, b, 100)
y_vals = funcion(x_vals)

plt.plot(x_vals, y_vals, label=r"$dT/dx = -100x$", color="blue")
plt.fill_between(x_vals, y_vals, alpha=0.3, color="cyan", label="Área")
plt.scatter(np.linspace(a, b, 10), funcion(np.linspace(a, b, 10)), color="red", label="Puntos")
plt.xlabel("x")
plt.ylabel("Gradiente de temperatura")
plt.legend()
plt.title("Flujo de calor (Regla de Simpson)")
plt.grid()

plt.savefig("calor_simpson.png")
plt.show()
