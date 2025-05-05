import numpy as np
import warnings
from typing import Tuple

def resolve(
    u0: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    interval: Tuple[float, float],
    T: float
) -> np.ndarray:
    """
    Solve the 1D heat equation u_t = u_xx using explicit finite differences
    on a uniform grid defined by an interval [a, b] and final time T.

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
      u[i, n+1] = (1 - 2*lambda) * u[i, n] +
                lambda * (u[i+1, n] + u[i-1, n]),
    for interior i = 1..n_x-2 and time n = 1..n_t-1.
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

    # Initial conditions
    u = np.zeros((n_x, n_t), dtype=np.float32)
    u[:, 0] = u0

    # Boundary conditions
    u[0, :] = g
    u[-1, :] = h

    for n in range(0, n_t - 1):
        for i in range(1, n_x - 1):
            u[i, n + 1] = (1 - 2 * lam) * u[i, n] + lam * (u[i - 1, n] + u[i + 1, n])

    return u