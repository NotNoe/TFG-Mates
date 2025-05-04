import numpy as np
from cpu.heat2d import resolve as cpu_resolve
from gpu.heat2d import resolve as gpu_resolve

def exact_solution(x, y, t):
    return np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)

def test_2dheat_gpu():
    a, b = 0.0, 1.0
    c, d = 0.0, 1.0
    T = 0.1
    n_x = 50
    n_y = 51

    # Para asegurar estabilidad: lambda_x + lambda_y <= 0.5
    dx = (b - a) / (n_x - 1)
    dy = (d - c) / (n_y - 1)
    lam_x = 1 / dx**2
    lam_y = 1 / dy**2
    dt_max = 0.5 / (lam_x + lam_y)
    n_t = int(np.ceil(T / dt_max)) + 1

    x = np.linspace(a, b, n_x)
    y = np.linspace(c, d, n_y)
    t = np.linspace(0, T, n_t)

    # Condición inicial
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Condiciones de contorno homogéneas
    g_left = np.zeros((n_y, n_t))
    g_right = np.zeros((n_y, n_t))
    g_bottom = np.zeros((n_x, n_t))
    g_top = np.zeros((n_x, n_t))

    # Resolver numéricamente
    U = gpu_resolve(u0, g_left, g_right, g_bottom, g_top, (a, b), (c, d), T)

    # Construir solución exacta en la malla
    U_exact = np.zeros_like(U)
    for i in range(n_x):
        for j in range(n_y):
            U_exact[i, j, :] = exact_solution(x[i], y[j], t)

    error = np.abs(U - U_exact)
    error_max = np.max(error)
    print(f"Error máximo: {error_max:.2e}")
    if error_max < 1e-2:
        print("✅ Test PASADO")
    else:
        print("❌ Test FALLADO")

def test_2dheat_cpu():
    a, b = 0.0, 1.0
    c, d = 0.0, 1.0
    T = 0.1
    n_x = 50
    n_y = 51

    # Para asegurar estabilidad: lambda_x + lambda_y <= 0.5
    dx = (b - a) / (n_x - 1)
    dy = (d - c) / (n_y - 1)
    lam_x = 1 / dx**2
    lam_y = 1 / dy**2
    dt_max = 0.5 / (lam_x + lam_y)
    n_t = int(np.ceil(T / dt_max)) + 1

    x = np.linspace(a, b, n_x)
    y = np.linspace(c, d, n_y)
    t = np.linspace(0, T, n_t)

    # Condición inicial
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Condiciones de contorno homogéneas
    g_left = np.zeros((n_y, n_t))
    g_right = np.zeros((n_y, n_t))
    g_bottom = np.zeros((n_x, n_t))
    g_top = np.zeros((n_x, n_t))

    # Resolver numéricamente
    U = cpu_resolve(u0, g_left, g_right, g_bottom, g_top, (a, b), (c, d), T)

    # Construir solución exacta en la malla
    U_exact = np.zeros_like(U)
    for i in range(n_x):
        for j in range(n_y):
            U_exact[i, j, :] = exact_solution(x[i], y[j], t)

    error = np.abs(U - U_exact)
    error_max = np.max(error)
    print(f"Error máximo: {error_max:.2e}")
    if error_max < 1e-2:
        print("✅ Test PASADO")
    else:
        print("❌ Test FALLADO")

if __name__ == "__main__":
    print("Testing CPU implementation...")
    test_2dheat_cpu()
    print("Testing GPU implementation...")
    test_2dheat_gpu()
