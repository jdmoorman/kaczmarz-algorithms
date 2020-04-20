from abc import ABC, abstractmethod


class SelectionStrategy(ABC):
    """An abstract base class for Kaczmarz selection strategies.

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
    kwargs : dict
    """

    def __init__(self, A, b, x0, tol, maxiter, **kwargs):
        pass

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
        return 0

    @classmethod
    def __subclasshook__(cls, C):
        if cls is SelectionStrategy:
            if any("select_row_index" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented
