import numpy as np
import warnings
from typing import Tuple, Callable
from pycuda import driver as cuda
from utils import generateMod
from math import ceil

def __check_data(f, interval_x, interval_y, T, n_x, n_y, n_t):
    a, b = interval_x
    c, d = interval_y

    #Comprobaciones
    if b <= a:
        raise ValueError("Interval must satisfy a < b.")
    if d <= c:
        raise ValueError("Interval must satisfy c < d.")
    if T <= 0:
        raise ValueError("Final time T must be positive.")
    if n_x < 2 or n_y < 2 or n_t < 2:
        raise ValueError("n_x, n_y and n_t must be greater than 1.")
    x = np.linspace(a, b, n_x)
    y = np.linspace(c, d, n_y)
    if not np.allclose(f(a, y), 0):
        raise ValueError("f(a,·) must be 0.")
    if not np.allclose(f(b, y), 0):
        raise ValueError("f(b,·) must be 0.")
    if not np.allclose(f(x, c), 0):
        raise ValueError("f(·,c) must be 0.")
    if not np.allclose(f(x, d), 0):
        raise ValueError("f(·,d) must be 0.")


def resolve(
    f: Callable[[float, float], float],
    g: Callable[[float, float], float],
    interval_x: Tuple[float, float],
    interval_y: Tuple[float, float],
    T: float,
    n_x: int,
    n_y: int,
    n_t: int
) -> np.ndarray:
    """
    Solve the 2D wave equation
        u_tt = u_xx + u_yy
    on a rectangular domain [a,b]×[c,d] over t∈[0,T],
    using explicit finite differences and padding to mimic an infinite domain.

    Parameters
    ----------
    f : Callable[[float, float], float]
        Initial displacement u(x,y,0).
    g : Callable[[float, float], float]
        Initial velocity   u_t(x,y,0).
    interval_x : tuple (a, b)
        Spatial domain in the x-direction.
    interval_y : tuple (c, d)
        Spatial domain in the y-direction.
    T : float
        Final time (> 0).
    n_x : int
        Number of spatial nodes in x (>= 2).
    n_y : int
        Number of spatial nodes in y (>= 2).
    n_t : int
        Number of time levels (>= 2).

    Returns
    -------
    u : ndarray, shape (n_x, n_y, n_t)
        Solution array with indices (i, j, n) corresponding to
        x_i = a + i·dx, y_j = c + j·dy, t_n = n·dt.

    Notes
    -----
    Let
        dx = (b – a)/(n_x – 1),
        dy = (c – d)/(n_y – 1),
        dt = T       /(n_t – 1),
        λ_x = dt/dx,
        λ_y = dt/dy.
    Stability requires λ_x^2 + λ_y^2 ≤ 1.

    The update for interior nodes 1≤i≤n_x–2, 1≤j≤n_y–2, 1≤n≤n_t–2 is

        u[i,j,n+1] = 2·(1 – λ_x**2 – λ_y**2)·u[i,j,n]
                    + λ_x**2·(u[i-1,j,n] + u[i+1,j,n])
                    + λ_y**2·(u[i,j-1,n] + u[i,j+1,n])
                    – u[i,j,n–1].
    """
    #Check data
    __check_data(f, interval_x, interval_y, T, n_x, n_y, n_t)
    # Convert inputs
    
    a, b = interval_x
    c, d = interval_y
    
    dx = (b - a) / (n_x - 1)
    dy = (d - c) / (n_y - 1)
    dt = T / (n_t - 1)
    lam_x = dt / dx
    lam_y = dt / dy
    lam_x2 = lam_x**2
    lam_y2 = lam_y**2
    if lam_x2 + lam_y2 > 1:
        warnings.warn(
            f"Stability condition violated: lambda_x**2 + lambda_y**2 = {lam_x2 + lam_y2:.4f} > 1; solution may diverge.",
            UserWarning
        )

    x = np.linspace(a-dx, b+dx, n_x + 2)
    y = np.linspace(c-dx, d+dx, n_y + 2)
    # Initial conditions
    u = np.zeros((n_t, n_x, n_y), dtype=np.float32)
    u[0, :, :] = f(x[1:-1, None], y[None, 1:-1])
    u[1, :, :] = (f(x[1:-1, None], y[None, 1:-1]) + dt * g(x[1:-1, None], y[None, 1:-1]) 
                  + 0.5*lam_x2 * (f(x[2:, None], y[None, 1:-1]) - 2*f(x[1:-1, None], y[None, 1:-1]) + f(x[:-2, None], y[None, 1:-1])) +
                    0.5*lam_y2 * (f(x[1:-1, None], y[None, 2:]) - 2*f(x[1:-1, None], y[None, 1:-1]) + f(x[1:-1, None], y[None, :-2])))
    
    #Streams
    mem_stream = cuda.Stream()
    kernel_stream = cuda.Stream()
    #Events for synchronization
    ev_ke = [cuda.Event() for _ in range(3)]
    ev_mem = [cuda.Event() for _ in range(3)]

    #Memcpy and mem_alloc (sync)
    u_gpu = [cuda.mem_alloc_like(u[:,:,0]) for _ in range(3)]
    cuda.memcpy_htod(u_gpu[0], u[0])
    cuda.memcpy_htod(u_gpu[1], u[1])

    # Initialize CUDA kernel
    mod = generateMod.init(["wave2d.cu"])
    resolve_kernel = mod.get_function("resolve")
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    grid_size_x = ceil(n_x / BLOCK_SIZE_X)
    grid_size_y = ceil(n_y / BLOCK_SIZE_Y)

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
            np.float32(lam_x2),
            np.float32(lam_y2),
            np.int32(n_x),
            np.int32(n_y),
            block=(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1),
            grid=(grid_size_x, grid_size_y)
        )
        # Cuando termimne el kernel, avisamos
        ev_ke[new].record(kernel_stream)

        # Copy the new result back to the host
        mem_stream.wait_for_event(ev_ke[new])
        cuda.memcpy_dtoh_async(u[n, :, :], u_new, stream=mem_stream)
        ev_mem[new].record(mem_stream)

    # Esperamos a que termine la memoria
    mem_stream.synchronize()
    u_gpu[0].free()
    u_gpu[1].free()
    u_gpu[2].free()

    return u.T