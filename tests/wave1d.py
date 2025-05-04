import numpy as np
from cpu.wave1d import resolve as resolve_cpu
from gpu.wave1d import resolve as resolve_gpu

def f(x):
    return np.sin(np.pi * x)

def g(x):
    return np.zeros_like(x)

def exact_solution(x, t):
    return np.cos(np.pi * t) * np.sin(np.pi * x)

def test_wave1d_gpu():
    a, b = 0.0, 1.0
    T = 1.0
    n_x = 100
    n_t = 1000
    tol = 1e-2

    u_num = resolve_gpu(f, g, (a, b), T, n_x, n_t)

    x = np.linspace(a, b, n_x)
    t = np.linspace(0, T, n_t)

    u_exact = np.array([
        exact_solution(xi, t) for xi in x
    ])

    error = np.abs(u_num - u_exact)
    error_max = np.max(error)
    print(f"Error máximo: {error_max:.2e}")
    if error_max < tol:
        print("✅ Test PASADO")
    else:
        print("❌ Test FALLADO")

def test_wave1d_cpu():
    a, b = 0.0, 1.0
    T = 1.0
    n_x = 100
    n_t = 1000
    tol = 1e-2

    u_num = resolve_cpu(f, g, (a, b), T, n_x, n_t)

    x = np.linspace(a, b, n_x)
    t = np.linspace(0, T, n_t)

    u_exact = np.array([
        exact_solution(xi, t) for xi in x
    ])

    error = np.abs(u_num - u_exact)
    error_max = np.max(error)

    print(f"Error máximo: {error_max:.2e}")
    if error_max < tol:
        print("✅ Test PASADO")
    else:
        print("❌ Test FALLADO")

if __name__ == "__main__":
    print("Testing CPU implementation...")
    test_wave1d_cpu()
    print("Testing GPU implementation...")
    test_wave1d_gpu()