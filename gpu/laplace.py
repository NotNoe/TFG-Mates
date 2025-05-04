from math import ceil
import numpy as np
import warnings
from typing import Tuple
from pycuda import driver as cuda
import utils.generateMod as generateMod
from utils.timers import latency_timer, kernel_timer

def __check_data(
    g_left: np.ndarray,
    g_right: np.ndarray,
    g_bottom: np.ndarray,
    g_top: np.ndarray,
    interval_x: Tuple[float, float],
    interval_y: Tuple[float, float],
):
    """
    Internal validation for 2D heat solver inputs, including corner consistency.
    """
    a, b = interval_x
    c, d = interval_y

    # Dimension checks
    if g_left.ndim != 1 or g_right.ndim != 1 or g_top.ndim != 1 or g_bottom.ndim != 1:
        raise ValueError("g_left, g_right, g_bottom, and g_top must be 1D arrays.")
    n_x = g_bottom.size
    n_y = g_left.size
    if g_right.size != n_y:
        raise ValueError("g_left and g_right must have the same size.")
    if g_bottom.size != n_x or g_top.size != n_x:
        raise ValueError("g_bottom and g_top must have the same size.")
    if n_x < 2 or n_y < 2:
        raise ValueError("g_left, g_right, g_bottom, and g_top must have size at least 2.")


    # Corner consistency
    if not np.isclose(g_left[0],    g_bottom[0]) or \
       not np.isclose(g_left[-1],   g_top[0]) or \
       not np.isclose(g_right[0],  g_bottom[-1]) or \
       not np.isclose(g_right[-1], g_top[-1]):
        raise ValueError("Corner boundary values must agree at all four corners.")


def resolve(
    g_left: np.ndarray,
    g_right: np.ndarray,
    g_bottom: np.ndarray,
    g_top: np.ndarray,
    interval_x: Tuple[float, float],
    interval_y: Tuple[float, float]
) -> np.ndarray:
    """
    Solve Laplace's equation u_xx + u_yy = 0 on [a,b]×[c,d]
    with Dirichlet boundary conditions via Jacobi iteration.

    Parameters
    ----------
    g_left : array_like, shape (n_y,)
        Boundary values on x = a, for each y_j (j=0..n_y-1).
    g_right : array_like, shape (n_y,)
        Boundary values on x = b, for each y_j.
    g_bottom : array_like, shape (n_x,)
        Boundary values on y = c, for each x_i (i=0..n_x-1).
    g_top : array_like, shape (n_x,)
        Boundary values on y = d, for each x_i.
    interval_x : tuple (a, b)
        Spatial domain in x; defines n_x = len(g_bottom) = len(g_top).
    interval_y : tuple (c, d)
        Spatial domain in y; defines n_y = len(g_left) = len(g_right).

    Returns
    -------
    u : ndarray, shape (n_x, n_y)
        Approximate solution on the grid (i, j) ≡ (x_i, y_j),
        matching the Dirichlet data on the boundary.

    Notes
    -----
    Let n_x, n_y = grid sizes including boundaries. Then
        dx = (b - a)/(n_x - 1),  dy = (d - c)/(n_y - 1)
    and set weights
        lambda_x = 1/2 * dy^2/(dx^2 + dy^2),
        lambda_y = 1/2 * dx^2/(dx^2 + dy^2).
    At each interior node (i=1..n_x-2, j=1..n_y-2), the Jacobi update is
        u[i,j] = lambda_x*(u[i-1,j] + u[i+1,j])
               + lambda_y*(u[i,j-1] + u[i,j+1]).
    Iteration starts with u=0 in interior, and continues until
    either the maximum change falls below tol = dx^2 + dy^2,
    or max_iter = n_x*n_y is reached.
    """

    # Convert inputs
    g_left   = np.asarray(g_left, np.float32)
    g_right  = np.asarray(g_right, np.float32)
    g_bottom = np.asarray(g_bottom, np.float32)
    g_top    = np.asarray(g_top, np.float32)
    a, b = interval_x
    c, d = interval_y

    __check_data(
        g_left, g_right, g_bottom, g_top,
        interval_x, interval_y
    )

    n_x = g_bottom.size
    n_y = g_left.size
    
    dx = (b - a) / (n_x - 1)
    dy = (d - c) / (n_y - 1)
    lambda_x = dy**2 / (2*(dx**2 + dy**2))
    lambda_y = dx**2 / (2*(dx**2 + dy**2))

    # Stop conditions
    h = max(dx, dy)
    tol = 0.01*h**2
    k_max = int(1/tol)

    # Boundary conditions
    u = cuda.pagelocked_zeros((n_x, n_y), dtype=np.float32)
    u[0, :] = g_left
    u[-1, :] = g_right
    u[:, 0] = g_bottom
    u[:, -1] = g_top

    diff_bits = [cuda.pagelocked_empty(1, dtype=np.uint32) for _ in range(2)]

    #Streams
    mem_stream = cuda.Stream()
    kernel_stream = cuda.Stream()
    #Events for synchronization
    ev_ke = [cuda.Event() for _ in range(2)]
    ev_mem = [cuda.Event() for _ in range(2)]

    #Memcpy and mem_alloc (sync)
    u_gpu = [cuda.mem_alloc_like(u) for _ in range(2)]
    cuda.memcpy_htod(u_gpu[0], u)
    cuda.memcpy_htod(u_gpu[1], u)
    diff_gpu = [cuda.mem_alloc(np.dtype(np.uint32).itemsize) for _ in range(2)] 




    #Initialize CUDA kernel
    mod = generateMod.init(["laplace.cu"])
    resolve_kernel = mod.get_function("resolve")
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    grid_size_x = ceil(n_x / BLOCK_SIZE_X)
    grid_size_y = ceil(n_y / BLOCK_SIZE_Y)

    for k in range(k_max):
        new = k % 2
        old = (k - 1) % 2
        u_new = u_gpu[new]
        u_old = u_gpu[old]
        #diff = 0
        cuda.memset_d32_async(diff_gpu[new], 0, 1, stream=kernel_stream)
        resolve_kernel(u_old, u_new, diff_gpu[new], np.int32(n_x),
                        np.int32(n_y), np.float32(lambda_x), np.float32(lambda_y),
                        block=(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1), grid=(grid_size_x, grid_size_y))
        ev_ke[new].record(kernel_stream)

        mem_stream.wait_for_event(ev_ke[new])
        cuda.memcpy_dtoh_async(diff_bits[new], diff_gpu[new], stream=mem_stream)
        ev_mem[new].record(mem_stream)

        # Mientra la GPU trabaja en esta iteracion, leemos el error de la anterior
        if k > 0:
            #Esperamos a tener el error de la ejecucion anterior
            ev_mem[old].synchronize()
            max_err = diff_bits[old].view(np.float32)[0]
            if max_err < tol:
                last_iter = k #Donde salio del bucle
                break
            elif k == k_max - 1:
                last_iter = k
                warnings.warn(
                    f"Maximum iterations reached: {k_max} iterations; solution may not converge.",
                    UserWarning
                )

    mem_stream.synchronize()
    kernel_stream.synchronize()
    u_final = u_gpu[last_iter % 2]
    cuda.memcpy_dtoh(u, u_final)
    u_gpu[0].free()
    u_gpu[1].free()
    diff_gpu[0].free()
    diff_gpu[1].free()
    return u