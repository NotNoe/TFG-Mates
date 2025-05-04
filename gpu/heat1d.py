from time import sleep
from typing import Tuple
import warnings
import numpy as np
import utils.generateMod as generateMod
import pycuda.driver as cuda
from math import ceil

def resolve(
    u0: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    interval: Tuple[float, float],
    T: float
) -> np.ndarray:
    """
    Solve the 1D heat equation u_t = u_xx using explicit finite differences
    on a uniform grid defined by an interval [a, b] and final time T on GPU.

    Parameters
    ----------
    u0 : array_like, shape (n_x,)
        Initial temperatures at spatial nodes (including boundaries at a and b).
    g : array_like, shape (n_t,)
        Boundary temperatures at the left boundary over time (including t=0).
    h : array_like, shape (n_t,)
        Boundary temperatures at the right boundary over time (including t=0).
    interval : tuple (a, b)
        Spatial domain endpoints.
    T : float
        Final time (> 0).

    Returns
    -------
    u : ndarray, shape (n_x, n_t)
        Solution matrix: columns=time levels (0 to n_t-1), rows=spatial nodes (0 to n_x-1).

    Notes
    -----
    Let n_x = u0.size, n_t = g.size. Then:
      dx = (b - a) / (n_x - 1)
      dt = T / (n_t - 1)
      lambda = dt / dx**2

    The update formula is:
      u[i, j] = (1 - 2*lambda) * u[i-1, j] +
                lambda * (u[i-1, j-1] + u[i-1, j+1]),
    for interior j = 1..n_x-2 and time i = 1..n_t-1.
    Stability requires lambda <= 0.5.
    """
        # Convert inputs
    u0 = np.asarray(u0, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)
    a, b = interval

    # Basic checks
    if u0.ndim != 1 or g.ndim != 1 or h.ndim != 1:
        raise ValueError("u0, g and h must be one-dimensional arrays.")
    if g.size != h.size:
        raise ValueError("g and h must have the same length (n_t).")
    if T <= 0:
        raise ValueError("Final time T must be positive.")
    if a >= b:
        raise ValueError("Interval must satisfy a < b.")
    if not np.isclose(u0[0], g[0]):
        raise ValueError("Boundary condition g[0] must equal u0[0].")
    if not np.isclose(u0[-1], h[0]):
        raise ValueError("Boundary condition h[0] must equal u0[-1].")

    # Dimensions and steps
    n_x = u0.size
    n_t = g.size
    dx = (b - a) / (n_x - 1)
    dt = T / (n_t - 1)
    lam = dt / dx**2

    if lam > 0.5:
        warnings.warn(
            f"Stability condition violated: lambda = {lam:.4f} > 0.5; solution may diverge.",
            UserWarning
        )
    u = cuda.pagelocked_empty((n_t, n_x), dtype=np.float32)
    u[0, :] = u0

    #Streams
    mem_stream = cuda.Stream()
    kernel_stream = cuda.Stream()
    #Events for synchronization
    ev_ke = [cuda.Event() for _ in range(2)]
    ev_mem = [cuda.Event() for _ in range(2)]

    #Memcpy and mem_alloc (sync)
    u_gpu = [cuda.mem_alloc_like(u[:,0]) for _ in range(2)]
    cuda.memcpy_htod(u_gpu[0], u[0])
    g_gpu = cuda.mem_alloc_like(g)
    h_gpu = cuda.mem_alloc_like(h)
    cuda.memcpy_htod(g_gpu, g)
    cuda.memcpy_htod(h_gpu, h)

    # Initialize CUDA kernel
    mod = generateMod.init(["heat1d.cu"])
    resolve_kernel = mod.get_function("resolve")
    BLOCK_SIZE = 256
    grid_size = ceil(n_x / BLOCK_SIZE)

    for n in range(1, n_t):
        new = n % 2
        old = (n - 1) % 2
        u_new = u_gpu[new]
        u_old = u_gpu[old]

        if n >= 2: #Hasta que el buffer no esté copiado no podemos escribir encima
            kernel_stream.wait_for_event(ev_mem[new])
        # Kernel launch
        resolve_kernel(
            u_old, u_new,
            g_gpu, h_gpu,
            np.float32(lam),
            np.int32(n),
            np.int32(n_x),
            block = (BLOCK_SIZE, 1, 1),
            grid = (grid_size, 1),
            stream = kernel_stream
        )
        # Cuando termimne el kernel, avisamos
        ev_ke[new].record(kernel_stream)

        # Copy the new result back to the host
        mem_stream.wait_for_event(ev_ke[new])
        cuda.memcpy_dtoh_async(u[n, :], u_new, stream=mem_stream)
        ev_mem[new].record(mem_stream)

    #Esperamos a que termine la memoria
    mem_stream.synchronize()
    u_gpu[0].free()
    u_gpu[1].free()
    return u.copy().T