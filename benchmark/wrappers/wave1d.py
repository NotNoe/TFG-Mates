import time
import numpy as np
from gpu.wave1d import resolve as resolve_gpu
from cpu.wave1d import resolve as resolve_cpu
from typing import Literal, Tuple

def __f(x):
    return np.sin(np.pi * x)

def __g(x):
    return np.zeros_like(x)

def __exact_solution(x, t):
    return np.cos(np.pi * t) * np.sin(np.pi * x)

def test_wave1d_case(device: Literal["cpu", "gpu", "CPU", "GPU"], n_x: int, n_t: int) -> Tuple[np.array, dict]:
    #PVIC
    metrics = {}
    x = np.linspace(0, 1, n_x)
    t = np.linspace(0, 1, n_t)


    if device.lower() == "gpu":
        start = time.perf_counter()
        u = resolve_gpu(__f, __g, (0, 1), 1, n_x, n_t)
        t_wall = time.perf_counter() - start
        metrics["t_wall"] = t_wall
        
    else:
        start = time.perf_counter()
        u = resolve_cpu(__f, __g, (0, 1), 1, n_x, n_t)
        t_wall = time.perf_counter() - start
        metrics["t_wall"] = t_wall



    u_exact = np.array([
        __exact_solution(xi, t) for xi in x
    ])

    error_max = np.max(np.abs(u - u_exact))
    #Rounding
    metrics["error"] = error_max

    return u, metrics

if __name__ == "__main__":
    # Test the function with example parameters
    params = {
        "n_x": 5,
        "n_t": 8
    }
    _, metrics = test_wave1d_case("cpu", **params)
    print("Metrics on CPU:", metrics)
    _, metrics = test_wave1d_case("gpu", **params)
    print("Metrics on GPU:", metrics)