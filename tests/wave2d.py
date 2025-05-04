import numpy as np
from cpu.wave2d import resolve as resolve_cpu
from gpu.wave2d import resolve as resolve_gpu

def f(x,y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def g(x,y):
    return np.zeros_like(x)

def exact_solution(x, y, t):
    return np.cos(np.sqrt(2) * np.pi * t) * np.sin(np.pi * x) * np.sin(np.pi * y)

def test_wave2d_gpu():
    a, b = 0.0, 1.0
    c, d = 0.0, 1.0
    T = 1.0
    n_x, n_y = 50, 51
    dx = (b - a) / (n_x - 1)
    dy = (d - c) / (n_y - 1)
    dt_max = 1.0 / np.sqrt(1.0 / dx**2 + 1.0 / dy**2)
    dt = 0.95 * dt_max  # márgen de seguridad
    n_t = int(np.ceil(T / dt)) + 1
    tol = 1e-2
    u_num = resolve_gpu(f, g, (a, b), (c, d), T, n_x, n_y, n_t)

    x = np.linspace(a, b, n_x)
    y = np.linspace(c, d, n_y)
    t = np.linspace(0, T, n_t)

    X, Y = np.meshgrid(x, y, indexing='ij')
    u_exact = np.zeros((n_x, n_y, n_t))

    for n in range(n_t):
        u_exact[:, :, n] = exact_solution(X, Y, t[n])

    error = np.abs(u_num - u_exact)
    error_max = np.max(error)

    print(f"Error máximo: {error_max:.2e}")
    if error_max < tol:
        print("✅ Test PASADO")
    else:
        print("❌ Test FALLADO")

def test_wave2d_cpu():
    a, b = 0.0, 1.0
    c, d = 0.0, 1.0
    T = 1.0
    n_x, n_y = 50, 51
    dx = (b - a) / (n_x - 1)
    dy = (d - c) / (n_y - 1)
    dt_max = 1.0 / np.sqrt(1.0 / dx**2 + 1.0 / dy**2)
    dt = 0.95 * dt_max  # márgen de seguridad
    n_t = int(np.ceil(T / dt)) + 1
    tol = 1e-2
    u_num = resolve_cpu(f, g, (a, b), (c, d), T, n_x, n_y, n_t)

    x = np.linspace(a, b, n_x)
    y = np.linspace(c, d, n_y)
    t = np.linspace(0, T, n_t)

    X, Y = np.meshgrid(x, y, indexing='ij')
    u_exact = np.zeros((n_x, n_y, n_t))

    for n in range(n_t):
        u_exact[:, :, n] = exact_solution(X, Y, t[n])

    error = np.abs(u_num - u_exact)
    error_max = np.max(error)

    print(f"Error máximo: {error_max:.2e}")
    if error_max < tol:
        print("✅ Test PASADO")
    else:
        print("❌ Test FALLADO")

if __name__ == "__main__":
    print("Testing CPU implementation...")
    test_wave2d_cpu()
    print("Testing GPU implementation...")
    test_wave2d_gpu()