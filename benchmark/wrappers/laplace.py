import time
import numpy as np
from gpu.laplace import resolve as resolve_gpu
from cpu.laplace import resolve as resolve_cpu
from typing import Literal, Tuple


def __exact_solution(x, y):
    return np.sin(np.pi * x) * np.sinh(np.pi * y)

def test_laplace_case(device: Literal["cpu", "gpu", "CPU", "GPU"], n_x: int, n_y: int) -> Tuple[np.array, dict]:
    #PVIC
    metrics = {}
    x = np.linspace(0, 1, n_x)
    y = np.linspace(0, 1, n_y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    g_left = np.zeros(n_y)
    g_right = np.zeros(n_y)
    g_bottom = np.zeros(n_x)
    g_top = np.sin(np.pi * x) * np.sinh(np.pi)

    if device.lower() == "gpu":
        start = time.perf_counter()
        u = resolve_gpu(g_left, g_right, g_bottom, g_top, (0, 1), (0, 1))
        t_wall = time.perf_counter() - start
        metrics["t_wall"] = t_wall
        
    else:
        start = time.perf_counter()
        u = resolve_cpu(g_left, g_right, g_bottom, g_top, (0, 1), (0, 1))
        t_wall = time.perf_counter() - start
        metrics["t_wall"] = t_wall

    u_exact = __exact_solution(X, Y)

    error_max = np.max(np.abs(u - u_exact))
    #Rounding
    metrics["error"] = error_max

    return u, metrics

if __name__ == "__main__":
    # Test the function with example parameters
    params = {
        "n_x": 51,
        "n_y": 51
    }
    _, metrics = test_laplace_case("cpu", **params)
    print("Metrics on CPU:", metrics)
    _, metrics = test_laplace_case("gpu", **params)
    print("Metrics on GPU:", metrics)