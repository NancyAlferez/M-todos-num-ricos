import numpy as np
import matplotlib.pyplot as plt

# Definición de los puntos de interpolación
x_points = np.array([0.5, 1.0, 1.5, 2.0])
y_points = np.array([1.2, 2.3, 3.7, 5.2])

# Función de interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

#  Evaluar en x = 1.25
x_eval = 1.25
y_eval = lagrange_interpolation(x_eval, x_points, y_points)
print(f"La deformación en x = {x_eval} es: {y_eval}")

# Puntos para graficar la interpolación
x_values = np.linspace(min(x_points), max(x_points), 100)
y_values = [lagrange_interpolation(x, x_points, y_points) for x in x_values]

# Graficar los puntos y la interpolación
plt.figure(figsize=(6,4))
plt.plot(x_values, y_values, label="Interpolación de Lagrange", color="blue")
plt.scatter(x_points, y_points, color="red", label="Puntos dados")
plt.xlabel("x")
plt.ylabel("Deformación (mm)")
plt.title("Interpolación de Lagrange")
plt.legend()
plt.grid(True)
plt.savefig("lagrange_interpolacion.png")
plt.show()
