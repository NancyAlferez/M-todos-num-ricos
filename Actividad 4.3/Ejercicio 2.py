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

# 🔹 Datos del problema
C = 1e-6  # capacitancia

# Función de voltaje V(t)
def funcion(t):
    return 100 * np.exp(-2 * t)

# Intervalo de tiempo
a, b = 0, 5

# Valores de n
valores_n = [6, 10, 20, 30]

# 🔹 Solución analítica
# Q = C * ∫ V(t) dt = C * [ -50 e^{-2t} ] de 0 a 5
analitica = C * (50 * (1 - np.exp(-10)))
print(f"Solución analítica: {analitica}")

# 🔹 Cálculo con Simpson
for n in valores_n:
    resultado = C * simpson_rule(funcion, a, b, n)
    error = abs(analitica - resultado)
    print(f"n = {n} -> Aproximación: {resultado} | Error: {error}")

# 🔹 Gráfica
x_vals = np.linspace(a, b, 100)
y_vals = funcion(x_vals)

plt.plot(x_vals, y_vals, label=r"$V(t) = 100e^{-2t}$", color="blue")
plt.fill_between(x_vals, y_vals, alpha=0.3, color="cyan", label="Área (Voltaje integrado)")
plt.scatter(np.linspace(a, b, 10), funcion(np.linspace(a, b, 10)), color="red", label="Puntos")
plt.xlabel("t")
plt.ylabel("Voltaje")
plt.legend()
plt.title("Carga en un capacitor (Regla de Simpson)")
plt.grid()

plt.savefig("capacitor_simpson.png")
plt.show()
