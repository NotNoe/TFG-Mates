from typing import Callable, Tuple
import warnings
import numpy as np
import utils.generateMod as generateMod
import pycuda.driver as cuda
from math import ceil
from utils.timers import latency_timer, kernel_timer

def resolve(
    f: Callable[[float], float],
    g: Callable[[float], float],
    interval: Tuple[float, float],
    T: float,
    n_x: int,
    n_t: int
) -> np.ndarray:
    """
    Solve the 1D wave equation u_tt = u_xx using explicit finite differences
    on a uniform grid defined by an interval [a, b] and final time T.

    Parameters
    ----------
    f : Callable
        Value of function u in t=0.
    g : Callable
        Value of function u_t in t=0.
    interval : tuple (a, b)
        Spatial domain endpoints.
    interval : tuple (a, b)
        Spatial domain endpoints.
    T : float
        Final time (> 0).
    n_x : int
        Number of spatial nodes in [a,b] (> 2).
    n_t : int
        Number of time levels (> 1).

    Returns
    -------
    u : ndarray, shape (n_x, n_t)
        Solution matrix: columns=time levels (0 to n_t-1), rows=spatial nodes (0 to n_x-1).

    Notes
    -----
      dx = (b - a) / (n_x - 1)
      dt = T / (n_t - 1)
      lambda = dt / dx

    The update formula is:
      u[i, n+1] = 2*(1 - lambda^2) * u[i, n] +
                lambda^2 * (u[i+1, n] + u[i-1, n]) - U[i, n-1],
    for interior i = 1..n_x-2 and time n = 2..n_t-1.
    Stability requires lambda <= 1.
    """

    a, b = interval

    #Comprobaciones
    if b <= a:
        raise ValueError("Interval must satisfy a < b.")
    if T <= 0:
        raise ValueError("Final time T must be positive.")
    if n_x < 2 or n_t < 2:
        raise ValueError("n_x and n_t must be greater than 1.")
    
    #Definiciones
    dx = (b - a) / (n_x - 1)
    dt = T / (n_t - 1)
    lam = dt / dx
    lam2 = lam**2
    if lam > 1:
        raise warnings.warn(f"Stability condition violated: lambda = {lam:.4f} > 1; solution may diverge.", UserWarning)
    # padding
    pad = n_t - 1
    n_x_ext = n_x + 2 * pad
    a_ext = a - pad * dx
    x_ext = a_ext + dx * np.arange(n_x_ext)

    # Initial conditions
    u_ext = cuda.pagelocked_zeros((n_t, n_x_ext), np.float32)
    u_ext[0,:] = f(x_ext)
    u_ext[1,1:-1] = f(x_ext)[1:-1] + dt * g(x_ext)[1:-1] + 0.5* lam2 * (f(x_ext)[0:-2] - 2 * f(x_ext)[1:-1] + f(x_ext)[2:])

    #Streams
    mem_stream = cuda.Stream()
    kernel_stream = cuda.Stream()
    #Events for synchronization
    ev_ke = [cuda.Event() for _ in range(3)]
    ev_mem = [cuda.Event() for _ in range(3)]

    #Memcpy and mem_alloc (sync)
    u_gpu = [cuda.mem_alloc_like(u_ext[0]) for _ in range(3)]
    cuda.memcpy_htod(u_gpu[0], u_ext[0])
    cuda.memcpy_htod(u_gpu[1], u_ext[1])

    # Initialize CUDA kernel
    mod = generateMod.init(["wave1d.cu"])
    resolve_kernel = mod.get_function("resolve")
    BLOCK_SIZE = 256
    grid_size = ceil(n_x_ext / BLOCK_SIZE)

    for n in range(2, n_t):
        new = n % 3
        old = (n - 1) % 3
        older = (n - 2) % 3
        u_new = u_gpu[new]
        u_old = u_gpu[old]
        u_older = u_gpu[older]

        if n >= 3: #Hasta que el buffer no esté copiado no podemos escribir encima
            kernel_stream.wait_for_event(ev_mem[new])

        # Kernel launch
        resolve_kernel(
            u_old, u_older, u_new,
            np.int32(n),
            np.float32(lam2),
            np.int32(n_x_ext), #Para saber que indices se nos quedan fuera
            grid=(grid_size, 1, 1),
            block=(BLOCK_SIZE, 1, 1)
        )
        # Cuando termine el kernel, avisamos
        ev_ke[new].record(kernel_stream)

        #Copy the new result back to the host
        mem_stream.wait_for_event(ev_ke[old])
        cuda.memcpy_dtoh_async(u_ext[n ], u_new, stream=mem_stream)
        ev_mem[new].record(mem_stream)
    
    mem_stream.synchronize()
    u_gpu[0].free()
    u_gpu[1].free()
    u_gpu[2].free()

    return u_ext[:, pad:pad + n_x].copy().T