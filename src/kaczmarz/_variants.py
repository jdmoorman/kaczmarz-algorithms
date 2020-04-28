"""A module providing selection strategies for the Kaczmarz algorithm."""

import numpy as np

import kaczmarz


class Cyclic(kaczmarz.Base):
    """Cycle through the equations of the system in order, repeatedly.

    Note
    ----
    This class inherits the parameters, methods, and properties of
    :class:`kaczmarz.Base`.

    Parameters
    ----------
    base_args : tuple
        Positional arguments for :class:`kaczmarz.Base` constructor.
    base_kwargs : dict
        Keyword arguments for :class:`kaczmarz.Base` constructor.

    References
    ----------
    1. S. Kaczmarz.
       "Angenäherte Auflösung von Systemen linearer Gleichungen."
       *Bulletin International de l’Académie Polonaise
       des Sciences et des Lettres.
       Classe des Sciences Mathématiques et Naturelles.
       Série A, Sciences Mathématiques*, 35, 335–357, 1937
    """

    def __init__(self, *base_args, **base_kwargs):
        super().__init__(*base_args, **base_kwargs)
        self.n_rows = self._A.shape[0]
        self.row_index = -1

    def _select_row_index(self, xk):
        self.row_index = (1 + self.row_index) % self.n_rows
        return self.row_index


class MaxDistance(kaczmarz.Base):
    """Choose the equation which leads to the most progress.

    This selection strategy is also known as `Motzkin's method`.

    Note
    ----
    This class inherits the parameters, methods, and properties of
    :class:`kaczmarz.Base`.

    References
    ----------
    1. T. S. Motzkin and I. J. Schoenberg.
       "The relaxation method for linear inequalities."
       *Canadian Journal of Mathematics*, 6:393–404, 1954.
    """

    def __init__(self, *base_args, **base_kwargs):
        super().__init__(*base_args, **base_kwargs)

    def _select_row_index(self, xk):
        residual = self._b - self._A @ self._xk
        return np.argmax(np.abs(residual))
