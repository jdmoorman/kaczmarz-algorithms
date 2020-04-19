"""Provide a class which implements the Kaczmarz/ART update."""
import numpy as np

from ._get_selection_strategy import get_selection_strategy
from .abc import SelectionStrategy


class Iterates:
    """Get the Iterates of the Kaczmarz algorithm.

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
    """

    def __init__(
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
        # TODO: Check what happens if we don't receive a seed.
        # TODO: Return the initial iterate during __iter__
        # TODO: Turn this class into a function.
        self._A = A
        if row_norms_squared is None:
            row_norms_squared = (A ** 2).sum(axis=1)
        self._row_norms_squared = row_norms_squared
        self._b = b.ravel()

        if x0 is None:
            n_cols = A.shape[1]
            x0 = np.zeros(n_cols)
        self._x0 = x0.ravel()
        self._tol = tol
        self._maxiter = maxiter
        self._callback = callback
        if not isinstance(selection_strategy, SelectionStrategy):
            selection_strategy = get_selection_strategy(
                selection_strategy,
                A=A,
                b=b,
                x0=x0,
                tol=tol,
                maxiter=maxiter,
                row_norms_squared=row_norms_squared,
                **selection_strategy_kwargs
            )
        self._selection_strategy = selection_strategy

    @property
    def ik(self):
        """The row used on the most recent iteration."""
        return self._ik

    def __next__(self):
        """Perform an iteration of the Kaczmarz algorithm.

        Returns
        -------
        xk : (n,) array
            The next iterate of the Kaczmarz algorithm.
        """
        if self._stopping_criterion():
            # TODO: If this is the first iteration, give a warning.
            raise StopIteration

        self._k += 1
        self._ik = self._selection_strategy.select_row_index(self._xk)
        self._xk = self._update_iterate(self._ik)

        if self._callback is not None:
            self._callback(self._xk.copy())

        return self._xk.copy()

    def __iter__(self):
        """Get an iterator of the Kaczmarz Iterates.

        Returns
        -------
        Iterates : iterator
            An iterator of (n,) arrays representing the Kaczmarz Iterates.
        """
        self._k = 0
        self._ik = -1
        self._xk = self._x0
        return self

    def _update_iterate(self, ik):
        """Apply the Kaczmarz update.

        Parameters
        ----------
        ik : int
            Row index to use for the update.

        Returns
        -------
        xk : (n,) array
            The next iterate
        """
        ai = self._A[ik]
        bi = self._b[ik]
        ai_norm_squared = self._row_norms_squared[ik]
        return self._xk + ((bi - ai @ self._xk) / ai_norm_squared) * ai

    def _stopping_criterion(self):
        """Check if iteration cap or desired accuracy have been reached.

        Returns
        -------
        stop : bool
            True if the iteration should be terminated.
        """
        if self._xk is None:
            return False

        if self._k >= self._maxiter:
            return True

        residual = self._b - self._A @ self._xk
        squared_residual_norm = (residual ** 2).sum()
        if squared_residual_norm < self._tol ** 2:
            return True
