"""Provide a class which implements the Kaczmarz/ART update."""
import numpy as np

from kaczmarz.selection import Cyclic as _DefaultSelectionStrategy


class iterates:
    def __init__(
        self,
        A,
        b,
        x0=None,
        tol=1e-5,
        maxiter=float("inf"),
        callback=None,
        selection_strategy=None,
        row_norms_squared=None,
    ):
        # TODO: Check what happens if we don't receive a seed.
        # TODO: Return the initial iterate during __iter__
        self._A = A
        if row_norms_squared is None:
            row_norms_squared = (A ** 2).sum(axis=1)
        self._row_norms_squared = row_norms_squared
        self._b = b.ravel()
        if selection_strategy is None:
            selection_strategy = _DefaultSelectionStrategy(A, b, x0)
        self._selection_strategy = selection_strategy

        if x0 is None:
            n_cols = A.shape[1]
            x0 = np.zeros(n_cols)
        self._x0 = x0.ravel()
        self._tol = tol
        self._maxiter = maxiter
        self._callback = callback
        self._k = -1
        self._xk = None

    def __next__(self):
        """Perform an iteration of the Kaczmarz algorithm."""
        if self.stopping_criterion():
            raise StopIteration

        self._xk = self.next_iterate()
        self._k += 1

        if self._callback is not None:
            self._callback(self._xk.copy())

        return self._xk.copy()

    def __iter__(self):
        """Start over, back at the initial guess."""
        self._k = -1
        self._xk = None
        return self

    def next_iterate(self):
        """Apply the Kaczmarz update."""
        if self._xk is None:
            return self._x0

        row_index = self._selection_strategy.next_row_index(self._xk)
        ai = self._A[row_index]
        bi = self._b[row_index]
        ai_norm_squared = self._row_norms_squared[row_index]
        return self._xk + ((bi - ai @ self._xk) / ai_norm_squared) * ai

    def stopping_criterion(self):
        """Check if iteration cap or desired accuracy have been reached.

        Returns
        -------
        bool
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
