import numpy as np
import warnings
from typing import Tuple

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
    g_left   = np.asarray(g_left, float)
    g_right  = np.asarray(g_right, float)
    g_bottom = np.asarray(g_bottom, float)
    g_top    = np.asarray(g_top, float)
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
    u = np.zeros((n_x, n_y), dtype=float)
    u[0, :] = g_left
    u[-1, :] = g_right
    u[:, 0] = g_bottom
    u[:, -1] = g_top

    #Jacobi iteration
    u_new = u.copy()
    for n in range(k_max):
        diff = 0
        #Update interior values
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                old = u[i, j]
                new = (
                    lambda_x * (u[i-1,j] + u[i+1,j]) +
                    lambda_y * (u[i,j-1] + u[i,j+1])
                )
                u_new[i, j] = new
                #Check improve while iterating
                d = abs(new - old)
                if d > diff:
                    diff = d
        #Check improve
        u, u_new = u_new, u
        #print(f"Iteration {n+1}: max diff = {diff:.2e}")
        if diff < tol:
            break
        elif n == k_max - 1:
            warnings.warn(
                f"Jacobi iteration did not converge after {k_max} iterations."
            )
    return u