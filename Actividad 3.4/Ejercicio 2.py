import numpy as np
import matplotlib.pyplot as plt
import csv

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    # Valor inicial en 0
    x = np.zeros(n)
    x_prev = np.copy(x)
    errors = []
    
    for k in range(max_iter):
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_prev[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        abs_error = np.linalg.norm(x - x_prev, ord=np.inf)
        rel_error = abs_error / (np.linalg.norm(x, ord=np.inf) + 1e-10)
        quad_error = np.linalg.norm(x - x_prev) ** 2
        
        errors.append((k+1, abs_error, rel_error, quad_error))
        
        if abs_error < tol:
            break
        
        x_prev = np.copy(x)
    
    return x, errors

# Definir el sistema de ecuaciones de transferencia de calor
A = np.array([
    [20, -5, -3,  0,  0],
    [-4, 18, -2, -1,  0],
    [-3, -1, 22, -5,  0],
    [ 0, -2, -4, 25, -1],
    [ 0, -2, -4, 25, -1]
], dtype=float)

b = np.array([100, 120, 130, 140, 150], dtype=float)

# Llamar a la función de Gauss-Seidel
x_sol, errors = gauss_seidel(A, b)

# Guardar errores en CSV
with open("errors.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Iteración", "Error absoluto", "Error relativo", "Error cuadrático"])
    writer.writerows(errors)
    writer.writerow([])
    writer.writerow(["Solución aproximada"])
    for val in x_sol:
        writer.writerow([val])

# Graficar errores
iterations = [e[0] for e in errors]
abs_errors = [e[1] for e in errors]
rel_errors = [e[2] for e in errors]
quad_errors = [e[3] for e in errors]

plt.figure(figsize=(10, 5))
plt.plot(iterations, abs_errors, label="Error absoluto", marker='o')
plt.plot(iterations, rel_errors, label="Error relativo", marker='s')
plt.plot(iterations, quad_errors, label="Error cuadrático", marker='d')
plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Errores")
plt.title("Convergencia del método de Gauss-Seidel")
plt.legend()
plt.grid()
plt.savefig("convergencia_gauss_seidel.png")
plt.show()

# Mostrar la solución aproximada
print("Solución aproximada (T1, T2, T3, T4, T5):")
print(x_sol)
