import time
import numpy as np
from gpu.wave2d import resolve as resolve_gpu
from cpu.wave2d import resolve as resolve_cpu
from typing import Literal, Tuple

def __f(x,y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def __g(x,y):
    return np.zeros_like(x)

def __exact_solution(x, y, t):
    return np.cos(np.sqrt(2) * np.pi * t) * np.sin(np.pi * x) * np.sin(np.pi * y)

def test_wave2d_case(device: Literal["cpu", "gpu", "CPU", "GPU"], n_x: int, n_y: int, n_t: int) -> Tuple[np.array, dict]:
    #PVIC
    metrics = {}
    x = np.linspace(0, 1, n_x)
    y = np.linspace(0, 1, n_y)
    t = np.linspace(0, 1, n_t)
    X, Y = np.meshgrid(x, y)

    u_exact = np.zeros((n_x, n_y, n_t))
    for n in range(n_t):
        u_exact[:, :, n] = __exact_solution(X, Y, t[n])

    if device.lower() == "gpu":
        start = time.perf_counter()
        u = resolve_gpu(__f,  __g, (0, 1), (0, 1), 1, n_x, n_y, n_t)
        t_wall = time.perf_counter() - start
        metrics["t_wall"] = t_wall

    else:
        start = time.perf_counter()
        u = resolve_cpu(__f,  __g, (0, 1), (0, 1), 1, n_x, n_y, n_t)
        t_wall = time.perf_counter() - start
        metrics["t_wall"] = t_wall

    u_exact = np.zeros((n_x, n_y, n_t))
    for n in range(n_t):
        u_exact[:, :, n] = __exact_solution(X, Y, t[n])

    error_max = np.max(np.abs(u - u_exact))
    #Rounding
    metrics["error"] = error_max

    return u, metrics

if __name__ == "__main__":
    # Test the function with example parameters
    params = {
        "n_x": 5,
        "n_y": 5,
        "n_t": 8
    }
    _, metrics = test_wave2d_case("cpu", **params)
    print("Metrics on CPU:", metrics)
    _, metrics = test_wave2d_case("gpu", **params)
    print("Metrics on GPU:", metrics)