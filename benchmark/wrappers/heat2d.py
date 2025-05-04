import time
import numpy as np
from gpu.heat2d import resolve as resolve_gpu
from cpu.heat2d import resolve as resolve_cpu
from typing import Literal, Tuple

def __exact_solution(x, y, t):
    # Solución exacta: u(x,t) = exp(-pi^2 t) * sin(pi x)
    return np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)
def test_heat2d_case(device: Literal["cpu", "gpu", "CPU", "GPU"], n_x: int, n_y: int, n_t: int) -> Tuple[np.array, dict]:
    #PVIC
    metrics = {}
    x = np.linspace(0, 1, n_x)
    y = np.linspace(0, 1, n_y)
    t = np.linspace(0, 0.1, n_t)
    X, Y = np.meshgrid(x, y)
    u0 = np.sin(np.pi*X) * np.sin(np.pi*Y)
    g_left = np.zeros((n_y, n_t))
    g_right = np.zeros((n_y, n_t))
    g_bottom = np.zeros((n_x, n_t))
    g_top = np.zeros((n_x, n_t))



    if device.lower() == "gpu":
        start = time.perf_counter()
        u = resolve_gpu(u0, g_left, g_right, g_bottom, g_top, (0, 1), (0, 1), 0.1)
        t_wall = time.perf_counter() - start
        metrics["t_wall"] = t_wall
        
    else:
        start = time.perf_counter()
        u = resolve_cpu(u0, g_left, g_right, g_bottom, g_top, (0, 1), (0, 1), 0.1)
        t_wall = time.perf_counter() - start
        metrics["t_wall"] = t_wall

    u_exact = np.zeros_like(u)
    for i in range(n_x):
        for j in range(n_y):
            u_exact[i, j, :] = __exact_solution(x[i], y[j], t)

    error_max = np.max(np.abs(u - u_exact))
    metrics["error"] = error_max

    return u, metrics

if __name__ == "__main__":
    # Test the function with example parameters
    params = {
        "n_x": 5,
        "n_y": 5,
        "n_t": 8
    }
    _, metrics = test_heat2d_case("cpu", **params)
    print("Metrics on CPU:", metrics)
    _, metrics = test_heat2d_case("gpu", **params)
    print("Metrics on GPU:", metrics)