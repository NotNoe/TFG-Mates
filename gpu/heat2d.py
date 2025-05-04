from math import ceil
import numpy as np
import warnings
from typing import Tuple
import utils.generateMod as generateMod
import pycuda.driver as cuda

def __check_data(
    u0: np.ndarray,
    g_left: np.ndarray,
    g_right: np.ndarray,
    g_bottom: np.ndarray,
    g_top: np.ndarray,
    interval_x: Tuple[float, float],
    interval_y: Tuple[float, float],
    T: float
):
    """
    Internal validation for 2D heat solver inputs, including corner consistency.
    """
    a, b = interval_x
    c, d = interval_y

    # Dimension checks
    if u0.ndim != 2:
        raise ValueError("u0 must be 2D array shape (n_x, n_y)." )
    n_x, n_y = u0.shape
    if g_left.shape != (n_y, g_left.shape[1]) or g_right.shape != g_left.shape:
        raise ValueError("g_left and g_right must have shape (n_y, n_t).")
    n_t = g_left.shape[1]
    if g_bottom.shape != (n_x, n_t) or g_top.shape != (n_x, n_t):
        raise ValueError("g_bottom and g_top must have shape (n_x, n_t).")
    if T <= 0:
        raise ValueError("Final time T must be positive.")
    if a >= b or c >= d:
        raise ValueError("Intervals must satisfy a < b in both directions.")

    # Boundary vs initial
    if not np.allclose(u0[0, :],    g_left[:, 0]):
        raise ValueError("Initial left boundary must match g_left[:,0].")
    if not np.allclose(u0[-1, :],   g_right[:, 0]):
        raise ValueError("Initial right boundary must match g_right[:,0].")
    if not np.allclose(u0[:, 0],    g_bottom[:, 0]):
        raise ValueError("Initial bottom boundary must match g_bottom[:,0].")
    if not np.allclose(u0[:, -1],   g_top[:, 0]):
        raise ValueError("Initial top boundary must match g_top[:,0].")

    # Corner consistency
    if not np.isclose(g_left[0, 0],    g_bottom[0, 0]) or \
       not np.isclose(g_left[-1, 0],   g_top[0, 0]) or \
       not np.isclose(g_right[0, -1],  g_bottom[-1, 0]) or \
       not np.isclose(g_right[-1, -1], g_top[-1, 0]):
        raise ValueError("Corner boundary values must agree at all four corners.")


def resolve(
    u0: np.ndarray,
    g_left: np.ndarray,
    g_right: np.ndarray,
    g_bottom: np.ndarray,
    g_top: np.ndarray,
    interval_x: Tuple[float, float],
    interval_y: Tuple[float, float],
    T: float
) -> np.ndarray:
    """
    Solve u_t = u_xx + u_yy on [a,b]×[c,d]×[0,T]
    via explicit finite differences on a uniform grid.

    Parameters
    ----------
    u0 : array_like, shape (n_x, n_y)
        Initial temperature at t=0 (incl. bordes).
    g_left, g_right : array_like, shape (n_y, n_t)
        Boundary temperatures at the left and right boundaries over time (incl. t=0).
    g_bottom, g_top : array_like, shape (n_x, n_t)
        Boundary temperatures at the bottom and top boundaries over time (incl. t=0).
    interval_x : (a, b)
    interval_y : (c, d)
    T : float
        Final time (> 0)

    Returns
    -------
    u : ndarray, shape (n_x, n_y, n_t)
        Solution Matrix (i,j,n) ≡ (x_i, y_j, t_n).

    Notes
    -----
    Let n_x = u0.shape[0], n_y = u0.shape[1] n_t = g.size. Then:
      dx = (b - a) / (n_x - 1)
      dy = (d - c) / (n_y - 1)
      dt = T / (n_t - 1)
      lambda_x = dt / dx**2
      lambda_y = dt / dy**2

    The update formula is:
      u[i, j, n+1] = (1 - 2lambda_x - 2lambda_y)) * u[i, j, n] +
        lambda_x * (u[i-1, j, n] + u[i+1, j, n]) +
        lambda_y * (u[i, j-1, n] + u[i, j+1, n])
    for interior i = 1..n_x-2, j = 1..n_y-2 and time i = 1..n_t-1.
    Stability requires lambda_x + lambda_y <= 0.5.
    """
    # Convert inputs
    u0       = np.asarray(u0, np.float32)
    g_left   = np.asarray(g_left, np.float32)
    g_right  = np.asarray(g_right, np.float32)
    g_bottom = np.asarray(g_bottom, np.float32)
    g_top    = np.asarray(g_top, np.float32)
    a, b = interval_x
    c, d = interval_y

    __check_data(
        u0, g_left, g_right, g_bottom, g_top,
        interval_x, interval_y, T
    )

    n_x, n_y = u0.shape
    n_t = g_left.shape[1]
    
    dx = (b - a) / (n_x - 1)
    dy = (d - c) / (n_y - 1)
    dt = T / (n_t - 1)
    lambda_x = dt / dx**2
    lambda_y = dt / dy**2
    if lambda_x + lambda_y > 0.5:
        warnings.warn(
            f"Stability condition violated: lambda_x + lambda_y = {lambda_x + lambda_y:.4f} > 0.5; solution may diverge.",
            UserWarning
        )

    # Initial conditions
    u = cuda.pagelocked_zeros((n_t, n_x, n_y), dtype=np.float32)
    u[0, :, :] = u0

    #Streams
    mem_stream = cuda.Stream()
    kernel_stream = cuda.Stream()
    #Events for synchronization
    ev_ke = [cuda.Event() for _ in range(2)]
    ev_mem = [cuda.Event() for _ in range(2)]
    
    #Malloc and memcpy (sync)
    u_gpu = [cuda.mem_alloc_like(u[:,:,0]) for _ in range(2)]
    g_left_gpu = cuda.mem_alloc(g_left.nbytes)
    g_right_gpu = cuda.mem_alloc(g_right.nbytes)
    g_bottom_gpu = cuda.mem_alloc(g_bottom.nbytes)
    g_top_gpu = cuda.mem_alloc(g_top.nbytes)
    cuda.memcpy_htod(u_gpu[0], u[0])       
    cuda.memcpy_htod(g_left_gpu, g_left)
    cuda.memcpy_htod(g_right_gpu, g_right)
    cuda.memcpy_htod(g_bottom_gpu, g_bottom)
    cuda.memcpy_htod(g_top_gpu, g_top)
        
    # Initialize CUDA kernel
    mod = generateMod.init(["heat2d.cu"])
    resolve_kernel = mod.get_function("resolve")
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    grid_size_x = ceil(n_x / BLOCK_SIZE_X)
    grid_size_y = ceil(n_y / BLOCK_SIZE_Y)

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
            g_left_gpu, g_right_gpu,
            g_bottom_gpu, g_top_gpu,
            np.int32(n),
            np.float32(lambda_x),
            np.float32(lambda_y),
            np.int32(n_x), #Para saber que indices se nos quedan fuera
            np.int32(n_y), #Para saber que indices se nos quedan fuera
            np.int32(n_t), #Para poder calcular los indices de la matriz aplanada
            grid=(grid_size_x, grid_size_y, 1),
            block=(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)
        )
        ev_ke[new].record(kernel_stream)

        # Copy the new result back to the host
        mem_stream.wait_for_event(ev_ke[new])
        cuda.memcpy_dtoh_async(u[n, :, :], u_new, stream=mem_stream)
        ev_mem[new].record(mem_stream)

    #Esperamos a que termine la memoria
    mem_stream.synchronize()
    u_gpu[0].free()
    u_gpu[1].free()
    g_left_gpu.free()
    g_right_gpu.free()
    g_bottom_gpu.free()
    g_top_gpu.free()

    return u.copy().T