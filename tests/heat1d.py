import numpy as np
from cpu.heat1d import resolve as cpu_resolve
from gpu.heat1d import resolve as gpu_resolve

TOLERANCE = 1e-2

def exact_solution(x, t):
    # Solución exacta: u(x,t) = exp(-pi^2 t) * sin(pi x)
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

def test_1dheat_gpu():
    a,b = 0.0, 1.0
    T = 0.1
    n_x = 51
    n_t = 6251

    x = np.linspace(a, b, n_x)
    t = np.linspace(0, T, n_t)

    u0 = np.sin(np.pi * x)
    u0[0] = u0[-1] = 0.0
    g = np.zeros(n_t)
    h = np.zeros(n_t)

    U = gpu_resolve(u0, g, h, (a, b), T)

    U_exact = np.array([
        exact_solution(xi, t) for xi in x  # cada fila es u(xi, t_0...t_n)
    ])

    error_max = np.max(np.abs(U - U_exact))
    print(f"Error máximo: {error_max:.2e}")
    if error_max < TOLERANCE:
        print("✅ Test PASADO: la solución numérica se ajusta bien a la solución exacta.")
    else:
        print("❌ Test FALLIDO: error demasiado grande.")

def test_1dheat_cpu():
    a,b = 0.0, 1.0
    T = 0.1
    n_x = 51
    n_t = 6251

    x = np.linspace(a, b, n_x)
    t = np.linspace(0, T, n_t)

    u0 = np.sin(np.pi * x)
    u0[0] = u0[-1] = 0.0
    g = np.zeros(n_t)
    h = np.zeros(n_t)

    U = cpu_resolve(u0, g, h, (a, b), T)

    U_exact = np.array([
        exact_solution(xi, t) for xi in x  # cada fila es u(xi, t_0...t_n)
    ])

    error_max = np.max(np.abs(U - U_exact))
    print(f"Error máximo: {error_max:.2e}")
    if error_max < TOLERANCE:
        print("✅ Test PASADO: la solución numérica se ajusta bien a la solución exacta.")
    else:
        print("❌ Test FALLIDO: error demasiado grande.")

if __name__ == "__main__":
    print("Testing CPU implementation...")
    test_1dheat_cpu()
    print("Testing GPU implementation...")
    test_1dheat_gpu()
