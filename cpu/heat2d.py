import numpy as np
import warnings
from typing import Tuple

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
    u0       = np.asarray(u0, float)
    g_left   = np.asarray(g_left, float)
    g_right  = np.asarray(g_right, float)
    g_bottom = np.asarray(g_bottom, float)
    g_top    = np.asarray(g_top, float)
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
    u = np.zeros((n_x, n_y, n_t), dtype=float)
    u[:, :, 0] = u0

    # Boundary conditions
    u[0, :, :] = g_left
    u[-1, :, :] = g_right
    u[:, 0, :] = g_bottom
    u[:, -1, :] = g_top


    for n in range(0, n_t - 1):
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                u[i, j, n+1] = (
                    (1 -2*lambda_x - 2*lambda_y) * u[i,j,n] +
                    lambda_x * (u[i-1,j,n] + u[i+1,j,n]) +
                    lambda_y * (u[i,j-1,n] + u[i,j+1,n])
                )

    return u