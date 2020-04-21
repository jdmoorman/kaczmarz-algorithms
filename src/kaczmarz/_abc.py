from abc import ABC, abstractmethod

import numpy as np


class Base(ABC):
    """The Kaczmarz algorithm, without a selection strategy.

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
        if callback is None:

            def callback(xk):
                return None

        self._callback = callback
        self._k = -1
        self._ik = -1
        self._xk = None

    @property
    def ik(self):
        """int: The index of the row used on the most recent iteration."""
        return self._ik

    @property
    def xk(self):
        """(n,) array: The most recent iterate."""
        return self._xk.copy()

    def __next__(self):
        """Perform an iteration of the Kaczmarz algorithm.

        Returns
        -------
        xk : (n,) array
            The next iterate of the Kaczmarz algorithm.
        """
        if self._k == -1:
            self._k += 1
            self._xk = self._x0
            self._callback(self.xk)
            return self.xk

        if self._stopping_criterion(self._k, self._xk):
            # TODO: If this is the first iteration, give a warning.
            raise StopIteration

        self._k += 1
        self._ik = self.select_row_index(self._xk)
        self._xk = self.update_iterate(self._xk, self._ik)
        self._callback(self.xk)

        return self.xk

    def __iter__(self):
        """Perform an iteration of the Kaczmarz algorithm.

        Returns
        -------
        xk : (n,) array
            The next iterate of the Kaczmarz algorithm.
        """
        return self

    def update_iterate(self, xk, ik):
        """Apply the Kaczmarz update.

        Parameters
        ----------
        xk : (n,) array
            The current iterate of the Kaczmarz algorithm.
        ik : int
            Row index to use for the update.

        Returns
        -------
        xkp1 : (n,) array
            The next iterate
        """
        ai = self._A[ik]
        bi = self._b[ik]
        ai_norm_squared = self._row_norms_squared[ik]
        return xk + ((bi - ai @ xk) / ai_norm_squared) * ai

    def _stopping_criterion(self, k, xk):
        """Check if the iteration should terminate.

        Parameters
        ----------
        k : int
            The number of iterations that have passed.
        xk : (n,) array
            The current iterate of the Kaczmarz algorithm.

        Returns
        -------
        stop : bool
            True if the iteration should be terminated.
        """
        if self._k >= self._maxiter:
            return True

        residual = self._b - self._A @ self._xk
        squared_residual_norm = (residual ** 2).sum()
        if squared_residual_norm < self._tol ** 2:
            return True

        return False

    @classmethod
    def iterates(cls, *args, **kwargs):
        """Get an iterator of the Kaczmarz Iterates.

        TODO: Describe the invisible arguments.

        Returns
        -------
        iterates : iterable((n,) array)
            An iterable of the Kaczmarz iterates.
        """
        return cls(*args, **kwargs)

    @classmethod
    def solve(cls, *args, **kwargs):
        """Solve a linear system of equations using the Kaczmarz algorithm.

        TODO: Describe the invisible arguments.

        Returns
        -------
        x : (n,) array
            The solution to the system `Ax = b`
        """
        iterates = cls.iterates(*args, **kwargs)
        for x in iterates:
            pass
        return x

    @abstractmethod
    def select_row_index(self, xk):
        """Select a row to use for the next Kaczmarz update.

        Parameters
        ----------
        xk : (n,) array
            The current Kaczmarz iterate.

        Returns
        -------
        ik : int
            The index of the next row to use.
        """
