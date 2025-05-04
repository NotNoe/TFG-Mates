import numpy as np
import warnings
from typing import Callable, Tuple

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
        warnings.warn(f"Stability condition violated: lambda = {lam:.4f} > 1; solution may diverge.", UserWarning)
    # padding
    pad = n_t - 1
    n_x_ext = n_x + 2 * pad
    a_ext = a - pad * dx
    x_ext = a_ext + dx * np.arange(n_x_ext)

    # Initial conditions
    u_ext = np.zeros((n_x_ext, n_t), float)
    u_ext[:,0] = f(x_ext)
    u_ext[1:-1,1] = f(x_ext)[1:-1] + dt * g(x_ext)[1:-1] + 0.5* lam2 * (f(x_ext)[0:-2] - 2 * f(x_ext)[1:-1] + f(x_ext)[2:])

    for n in range(1, n_t - 1):
        i_min = 1 + n
        i_max = n_x_ext - 1 - n
        for i in range(i_min, i_max):
            u_ext[i, n + 1] = 2*(1 - lam2) * u_ext[i, n] + lam2 * (u_ext[i - 1, n] + u_ext[i + 1, n]) - u_ext[i, n - 1]

    return u_ext[pad:pad + n_x, :]