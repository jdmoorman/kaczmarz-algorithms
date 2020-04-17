from abc import ABC, abstractmethod


class SelectionStrategy(ABC):
    """An abstract base class for Kaczmarz selection strategies.

    Parameters
    ----------
    A : 2darray (m, n)
        A matrix whose rows should be selected from.
    b : 1darray (m,)
        The right-hand-side vector of a system of equations.
    x0 : 1darray (n,)
        The initial guess of the Kaczmarz algorithm.
    tol : float
        The residual norm tolerance used as a convergence criterion.
    maxiter : int
        The maximum number of iterations the Kaczmarz algorithm will perform.
    row_norms_squared : 1darray (m,)
        The squared norm of each row of `A`.

    Attributes
    ----------
    ik : int
        The most recent
    """

    def __init__(self, A, b, x0, tol, maxiter, row_norms_squared):
        self._A = A
        self._b = b
        self._x0 = x0
        self._tol = tol
        self._maxiter = maxiter
        self._row_norms_squared = row_norms_squared
        self.ik

    @abstractmethod
    def select_row_index(self, xk):
        """Select a row to use for the next Kaczmarz update.

        Parameters
        ----------
        xk : ndarray
            The current Kaczmarz iterate.

        Returns
        -------
        int
            The index of the next row to use.
        """
        return 0  # This is not a very good selection strategy.

    @classmethod
    def __subclasshook__(cls, C):
        if cls is SelectionStrategy:
            if any("_select_row_index" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented
