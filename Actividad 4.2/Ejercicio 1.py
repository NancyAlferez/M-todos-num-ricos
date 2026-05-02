import numpy as np
import matplotlib.pyplot as plt

# Definimos la función a integrar
def f(x):
    return x**2 + 3*x + 1   # ← solo cambiamos la función

# Implementación de la regla del trapecio
def trapezoidal_rule(a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    integral = (h / 2) * (y[0] + 2 * sum(y[1:n]) + y[n])
    return integral, x, y

# Parámetros de integración
a, b = 0, 2

# Valores de n
n_values = [10, 20, 50]

# Valor exacto
exacta = (2**3)/3 + (3*(2**2))/2 + 2
print(f"Valor exacto: {exacta:.6f}\n")

# Gráfica base
x_fine = np.linspace(a, b, 100)
y_fine = f(x_fine)

plt.figure(figsize=(8, 5))
plt.plot(x_fine, y_fine, 'r-', label=r'$f(x) = x^2+3x+1$', linewidth=2)

# Ciclo para n = 10,20,50
for n in n_values:
    integral_approx, x_vals, y_vals = trapezoidal_rule(a, b, n)
    error = abs(exacta - integral_approx)

    print(f"n = {n}")
    print(f"Integral aproximada: {integral_approx:.6f}")
    print(f"Error absoluto: {error:.6f}\n")

    # Solo graficamos n=10 para que no se sature
    if n == 10:
        plt.fill_between(x_vals, y_vals, alpha=0.3, color='blue', label="Trapecios (n=10)")
        plt.plot(x_vals, y_vals, 'bo-', label="Puntos de integración")

# Etiquetas y leyenda
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Aproximación de la integral con la regla del trapecio")
plt.legend()
plt.grid()

plt.savefig("trapecio.png")
plt.show()
