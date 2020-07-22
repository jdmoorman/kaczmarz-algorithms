"""A module providing selection strategies for the Kaczmarz algorithm."""

from collections import deque
from abc import ABC, abstractmethod

import numpy as np

import kaczmarz


class Cyclic(kaczmarz.Base):
    """Cycle through the equations of the system in order, repeatedly.

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
        self._row_index = -1

    def _select_row_index(self, xk):
        self._row_index = (1 + self._row_index) % self._n_rows
        return self._row_index


class MaxDistance(kaczmarz.Base):
    """Choose equations which leads to the most progress.

    This selection strategy is also known as `Motzkin's method`.

    References
    ----------
    1. T. S. Motzkin and I. J. Schoenberg.
       "The relaxation method for linear inequalities."
       *Canadian Journal of Mathematics*, 6:393–404, 1954.
    """

    def _select_row_index(self, xk):
        # TODO: use auxiliary update for the residual.
        residual = self._b - self._A @ self._xk
        return np.argmax(np.abs(residual))


class Random(kaczmarz.Base):
    """Sample equations according to a `fixed` probability distribution.

    Parameters
    ----------
    p : (m,) array_like, optional
        Sampling probability for each equation. Uniform by default.
    """

    def __init__(self, *base_args, p=None, **base_kwargs):
        super().__init__(*base_args, **base_kwargs)
        self._p = p  # p=None corresponds to uniform.

    def _select_row_index(self, xk):
        return np.random.choice(self._n_rows, p=self._p)


class SVRandom(Random):
    """Sample equations with probability proportional to the squared row norms.

    References
    ----------
    1. T. Strohmer and R. Vershynin,
       "A Randomized Kaczmarz Algorithm with Exponential Convergence."
       Journal of Fourier Analysis and Applications 15, 262 2009.
    """

    def __init__(self, *base_args, **base_kwargs):
        super().__init__(*base_args, **base_kwargs)
        squared_row_norms = self._row_norms ** 2
        self._p = squared_row_norms / squared_row_norms.sum()


class UniformRandom(Random):
    """Sample equations uniformly at random."""

    # Nothing to do since uniform sampling is the default behavior of Random.


class ThresholdedBase(Random, ABC):
    @abstractmethod
    def _threshold(self):
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

    def _distance(self, xk, ik):
        """
        Computes distance from vector xk to the index ik hyperplane.
        """
        return np.abs(self._b[ik] - self._A[ik] @ xk)

    def _select_row_index(self, xk):
        """
        Sample a row by index.  Returns the index if the row satisfies the threshold
        condition.  Otherwise returns -1.
        """
        ik = super()._select_row_index(xk)

        distance = self._distance(xk, ik)
        threshold = self._threshold(xk)

        if distance < threshold or np.isclose(distance, threshold):
            return ik

        return -1  # No projection please


class Quantile(ThresholdedBase):
    def __init__(self, *args, quantile=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._quantile = quantile

    def _distances(self, xk):
        return np.abs(self._b - self._A @ xk)

    def _threshold(self, xk):
        distances = self._distances(xk)

        if len(distances) == 0:
            return 0

        return np.quantile(distances, self._quantile)


class SampledQuantile(Quantile):
    def __init__(self, *args, n_samples=None, **kwargs):
        super().__init__(*args, **kwargs)
        if n_samples is None:
            n_samples = self._n_rows
        self._n_samples = n_samples

    def _distances(self, xk):
        idxs = np.random.choice(self._n_rows, self._n_samples, replace=False)
        return np.abs(self._b[idxs] - self._A[idxs] @ xk)


class WindowedQuantile(Quantile):
    def __init__(self, *args, window_size=100, **kwargs):
        super().__init__(*args, **kwargs)
        self._window_size = window_size
        self._window = deque([])

    def _update_window(self, xk, ik):
        self._window.append(self._distance(xk, ik))
        if len(self._window) > self._window_size:
            self._window.popleft()

    def _select_row_index(self, xk):
        ik = super()._select_row_index(xk)

        # is this really the right place to update the window?
        self._update_window(xk, self._most_recent_index)
        return ik

    def _distances(self, xk):  # NOTE: xk is only here so that _distances is overridden
        return self._window
