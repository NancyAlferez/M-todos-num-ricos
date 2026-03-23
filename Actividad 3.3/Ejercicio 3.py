import numpy as np
import matplotlib.pyplot as plt

# Definir el sistema de ecuaciones del ejercicio
A = np.array([
    [12, -2,  1,  0,  0,  0,  0],
    [-3, 18, -4,  2,  0,  0,  0],
    [ 1, -2, 16, -1,  1,  0,  0],
    [ 0,  2, -1, 11, -3,  1,  0],
    [ 0,  0, -2,  4, 15, -2,  1],
    [ 0,  0,  0,  1, -3,  2, 13],
    [ 0,  0,  0,  0,  0,  0,  0]  # Ajuste si hubiera séptima ecuación explícita
], dtype=float)

b = np.array([20, 35, -5, 19, -12, 25, 0], dtype=float)

# Solución exacta para comparar errores
sol_exacta = np.linalg.solve(A[:6,:6], b[:6])  # Resolver solo las primeras 6 ecuaciones

# Criterio de paro
tolerancia = 1e-6
max_iter = 100

# Implementación del método de Jacobi
def jacobi(A, b, tol, max_iter):
    n = len(A)
    x = np.zeros(n)  # Aproximación inicial
    errores_abs = []
    errores_rel = []
    errores_cuad = []
    
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            if A[i, i] != 0:
                x_new[i] = (b[i] - suma) / A[i, i]
        
        # Calcular errores
        error_abs = np.linalg.norm(x_new[:6] - sol_exacta, ord=1)
        error_rel = error_abs / np.linalg.norm(sol_exacta, ord=1)
        error_cuad = np.linalg.norm(x_new[:6] - sol_exacta, ord=2)
        
        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)
        
        # Imprimir errores de la iteración
        print(f"Iteración {k+1}: Error absoluto = {error_abs:.6f}, Error relativo = {error_rel:.6f}, Error cuadrático = {error_cuad:.6f}")
        
        # Criterio de convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        
        x = x_new
    
    return x, errores_abs, errores_rel, errores_cuad, k+1

# Ejecutar el método de Jacobi
sol_aprox, errores_abs, errores_rel, errores_cuad, iteraciones = jacobi(A, b, tolerancia, max_iter)

# Graficar los errores
plt.figure(figsize=(8,6))
plt.plot(range(1, iteraciones+1), errores_abs, label="Error absoluto", marker='o')
plt.plot(range(1, iteraciones+1), errores_rel, label="Error relativo", marker='s')
plt.plot(range(1, iteraciones+1), errores_cuad, label="Error cuadrático", marker='d')
plt.xlabel("Iteraciones")
plt.ylabel("Error")
plt.yscale("log")
plt.title("Convergencia de los errores en el método de Jacobi")
plt.legend()
plt.grid()
plt.savefig("errores_jacobi.png")  # Guardar la figura en archivo PNG
plt.show()

# Mostrar la solución aproximada y exacta
print(f"Solución aproximada: {sol_aprox}")
print(f"Solución exacta: {sol_exacta}")
