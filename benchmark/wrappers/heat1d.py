import time
import numpy as np
from gpu.heat1d import resolve as resolve_gpu
from cpu.heat1d import resolve as resolve_cpu
from typing import Literal, Tuple

def __exact_solution(x, t):
    # Solución exacta: u(x,t) = exp(-pi^2 t) * sin(pi x)
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

def test_heat1d_case(device: Literal["cpu", "gpu", "CPU", "GPU"], n_x: int, n_t: int) -> Tuple[np.array, dict]:
    #PVIC
    metrics = {}
    x = np.linspace(0, 1, n_x)
    t = np.linspace(0, 0.1, n_t)
    u0 = np.sin(np.pi * x)
    g = np.zeros(n_t)
    h = np.zeros(n_t)



    if device.lower() == "gpu":
        start = time.perf_counter()
        u = resolve_gpu(u0, g, h, (0, 1), 0.1)
        t_wall = time.perf_counter() - start
        metrics["t_wall"] = t_wall
        
    else:
        start = time.perf_counter()
        u = resolve_cpu(u0, g, h, (0, 1), 0.1)
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
        "n_x": 128,
        "n_t": 3641
    }
    _, metrics = test_heat1d_case("cpu", **params)
    print("Metrics on CPU:", metrics)
    _, metrics = test_heat1d_case("gpu", **params)
    print("Metrics on GPU:", metrics)