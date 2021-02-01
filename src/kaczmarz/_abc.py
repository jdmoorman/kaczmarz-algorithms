from abc import ABC, abstractmethod

import numpy as np

from ._normalize import normalize_system


class Base(ABC):
    """A base class for the Kaczmarz algorithm.

    This class cannot be instantiated directly.
    Subclasses should implement :meth:`kaczmarz.Base._select_row_index`.
    Subclasses will typically be constructed using
    :meth:`kaczmarz.Base.iterates` or :meth:`kaczmarz.Base.solve`.

    Parameters
    ----------
    A : (m, n) spmatrix or array_like
        The m-by-n matrix of the linear system.
    b : (m,) or (m, 1) array_like
        Right hand side of the linear system.
    x0 : (n,) or (n, 1) array_like, optional
        Starting guess for the solution.
    tol : float, optional
        Tolerance for convergence, ``norm(normalized_residual) <= tol``. Pass ``None`` to ignore the tolerance.
    maxiter : int or float, optional
        Maximum number of iterations. At least one of ``tol`` or ``maxiter`` must be passed.
    callback : function, optional
        User-supplied function to call after each iteration.
        It is called as ``callback(xk)``,
        where xk is the current solution vector.

    Notes
    -----
    There may be additional parameters not listed above
    depending on the selection strategy subclass.
    """

    def __init__(
        self,
        A,
        b,
        x0=None,
        tol=1e-5,
        maxiter=None,
        callback=None,
    ):
        self._A, self._b, self._row_norms = normalize_system(A, b)
        self._n_rows = len(self._b)

        if x0 is None:
            n_cols = self._A.shape[1]
            x0 = np.zeros(n_cols)
            self._iterate_shape = list(np.shape(b))  # [m,] or [m, 1]
            self._iterate_shape[0] = n_cols
        else:
            x0 = np.array(x0, dtype="float64")
            self._iterate_shape = x0.shape
        self._x0 = x0.ravel()

        if maxiter is None:
            if tol is not None:
                # TODO: Incorporate the initial error somehow, perhaps through the initial residual.
                # TODO: Explain where this comes from.
                # 1/min(self._A.shape) lower bounds the maximum singular value/sum of singular values.
                # For well matrices with condition number<10, 1/10*min(self._A.shape) lower bounds the minimum singular value.
                maxiter = 2 * np.log(tol) / np.log(1 - 1 / (10 * min(self._A.shape)))
            else:
                raise ValueError(
                    "At least one of ``tol`` or ``maxiter`` must be specified."
                )
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
        """int: The index of the row used on the most recent iteration.

        Takes the value -1 if a projection was not performed at iteration ``k``."""
        return self._ik

    @property
    def xk(self):
        """(n,) or (n, 1) array: The most recent iterate.

        The shape will match that of ``x0`` if provided, or ``b`` otherwise.
        """
        return self._xk.copy().reshape(*self._iterate_shape)

    @classmethod
    def iterates(cls, *base_args, **base_kwargs):
        """Get the Kaczmarz iterates.

        Note
        ----
        This method takes the same parameters as :class:`kaczmarz.Base`
        or the subclass from which it is called.
        For example, :meth:`kaczmarz.Cyclic.iterates`
        takes the same arguments as :class:`kaczmarz.Cyclic`.

        Parameters
        ----------
        base_args : tuple
            Positional arguments for :class:`kaczmarz.Base` constructor
            or the subclass in use.
        base_kwargs : dict
            Keyword arguments for :class:`kaczmarz.Base` constructor
            or the subclass in use.

        Returns
        -------
        iterates : iterable((n,) or (n, 1) array)
            An iterable of the Kaczmarz iterates.
            The shapes will match that of ``x0`` if provided,
            or ``b`` otherwise.
        """
        return cls(*base_args, **base_kwargs)

    @classmethod
    def solve(cls, *base_args, **base_kwargs):
        """Solve a linear system of equations using the Kaczmarz algorithm.

        Note
        ----
        This method takes the same parameters as :class:`kaczmarz.Base`
        or the subclass from which it is called.
        For example, :meth:`kaczmarz.Cyclic.solve`
        takes the same arguments as :class:`kaczmarz.Cyclic`.

        Parameters
        ----------
        base_args : tuple
            Positional arguments for :class:`kaczmarz.Base` constructor
            or the subclass in use.
        base_kwargs : dict
            Keyword arguments for :class:`kaczmarz.Base` constructor
            or the subclass in use.

        Returns
        -------
        x : (n,) or (n, 1) array
            The solution to the system ``A @ x = b``.
            The shape will match that of ``x0`` if provided,
            or ``b`` otherwise.
        """
        iterates = cls.iterates(*base_args, **base_kwargs)
        for x in iterates:
            pass
        return x

    def __next__(self):
        """Perform an iteration of the Kaczmarz algorithm.

        Returns
        -------
        xk : (n,) or (n, 1) array
            The next iterate of the Kaczmarz algorithm.
            The shape will match that of ``x0`` if provided,
            or ``b`` otherwise.
        """
        if self._k == -1:
            self._k += 1
            self._xk = self._x0
            self._callback(self.xk)
            return self.xk

        if self._stopping_criterion(self._k, self._xk):
            raise StopIteration

        self._k += 1
        self._ik = self._select_row_index(self._xk)
        if self._ik != -1:
            self._xk = self._update_iterate(self._xk, self._ik)

        self._callback(self.xk)

        return self.xk

    def __iter__(self):
        """Iterator for iterates of the Kaczmarz algorithm."""
        return self

    @abstractmethod
    def _select_row_index(self, xk):
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

    def _update_iterate(self, xk, ik):
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
        return xk + (bi - ai @ xk) * ai

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

        if self._tol is not None:
            residual = self._b - self._A @ xk
            residual_norm = np.linalg.norm(residual)

            if residual_norm < self._tol:
                return True

        return False
