import numpy as np
from cpu.laplace import resolve as cpu_resolve
from gpu.laplace import resolve as gpu_resolve

TOLERANCE = 1e-2

def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sinh(np.pi * y)

def test_laplace_gpu():
    a, b = 0.0, 1.0
    c, d = 0.0, 1.0 
    n_x = 51
    n_y = 51

    x = np.linspace(a, b, n_x)
    y = np.linspace(c, d, n_y)

    g_left = g_right = np.zeros(n_y)
    g_bottom = np.zeros(n_x)
    g_top = np.sin(np.pi * x) * np.sinh(np.pi)

    U = gpu_resolve(g_left, g_right, g_bottom, g_top, (a, b), (c, d))

    X, Y = np.meshgrid(x, y, indexing='ij')
    U_exact = exact_solution(X, Y)


    error_max = np.max(np.abs(U - U_exact))
    print(f"Error máximo: {error_max:.2e}")
    if error_max < TOLERANCE:
        print("✅ Test PASADO: la solución numérica se ajusta bien a la solución exacta.")
    else:
        print("❌ Test FALLIDO: error demasiado grande.")

def test_laplace_cpu():
    a, b = 0.0, 1.0
    c, d = 0.0, 1.0 
    n_x = 51
    n_y = 51

    x = np.linspace(a, b, n_x)
    y = np.linspace(c, d, n_y)

    g_left = g_right = np.zeros(n_y)
    g_bottom = np.zeros(n_x)
    g_top = np.sin(np.pi * x) * np.sinh(np.pi)

    U = cpu_resolve(g_left, g_right, g_bottom, g_top, (a, b), (c, d))

    X, Y = np.meshgrid(x, y, indexing='ij')
    U_exact = exact_solution(X, Y)


    error_max = np.max(np.abs(U - U_exact))
    print(f"Error máximo: {error_max:.2e}")
    if error_max < TOLERANCE:
        print("✅ Test PASADO: la solución numérica se ajusta bien a la solución exacta.")
    else:
        print("❌ Test FALLIDO: error demasiado grande.")

if __name__ == "__main__":
    print("Testing CPU implementation...")
    test_laplace_cpu()
    print("Testing GPU implementation...")
    test_laplace_gpu()
