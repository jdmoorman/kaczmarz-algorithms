import kaczmarz


def solve(
    self,
    A,
    b,
    x0=None,
    tol=1e-5,
    maxiter=float("inf"),
    callback=None,
    row_norms_squared=None,
    selection_strategy="Cyclic",
    **selection_strategy_kwargs
):
    """Solve a linear system of equations using the Kaczmarz algorithm.

    Parameters
    ----------
    A : (m, n) array
        The real or complex m-by-n matrix of the linear system.
    b : (m,) array
        Right hand side of the linear system.
    x0 : (n,) array, optional
        Starting guess for the solution.
    tol : float, optional
        Tolerance for convergence, `norm(residual) <= tol`.
    maxiter : int or float, optional
        Maximum number of iterations.
    callback : function, optional
        User-supplied function to call after each iteration.
        It is called as callback(xk), where xk is the current solution vector.
    selection_strategy : str, subclass of SelectionStrategy, or instance of SelectionStrategy, optional
        The selection strategy for choosing rows at each iteration.
        See `kaczmarz.selection` for options.
    selection_strategy_kwargs : dict
        Additional keyword arguments are passed to the selection strategy.

    Returns
    -------
    x : (n,) array
        The solution to the system `Ax = b`
    """
    iterates = kaczmarz.Iterates(
        A,
        b,
        x0,
        tol,
        maxiter,
        callback,
        row_norms_squared,
        selection_strategy,
        **selection_strategy_kwargs
    )
    for x in iterates:
        pass
    return x
