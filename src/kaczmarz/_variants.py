"""A module providing selection strategies for the Kaczmarz algorithm."""

from collections import deque

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


class Quantile(Random):
    """Reject equations whose normalized residual is above a quantile.

    This algorithm is intended for use in solving corrupted systems of equations.
    That is, systems where a subset of the equations are consistent,
    while a minority of the equations are not.
    Such systems are almost always overdetermined.

    Parameters
    ----------
    quantile : float, optional
        Quantile of normalized residual above which to reject.

    References
    ----------
    1. There will be a reference soon. Keep an eye out for that.
    """

    def __init__(self, *args, quantile=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._quantile = quantile

    def _distance(self, xk, ik):
        return np.abs(self._b[ik] - self._A[ik] @ xk)

    def _threshold_distances(self, xk):
        return np.abs(self._b - self._A @ xk)

    def _threshold(self, xk):
        distances = self._threshold_distances(xk)

        return np.quantile(distances, self._quantile)

    def _select_row_index(self, xk):
        ik = super()._select_row_index(xk)

        distance = self._distance(xk, ik)
        threshold = self._threshold(xk)

        if distance < threshold or np.isclose(distance, threshold):
            return ik

        return -1  # No projection please


class SampledQuantile(Quantile):
    """Reject equations whose normalized residual is above a quantile of a random subset of residual entries.

    Parameters
    ----------
    n_samples: int, optional
        Number of normalized residual samples used to compute the threshold quantile.

    References
    ----------
    1. There will be a reference soon. Keep an eye out for that.
    """

    def __init__(self, *args, n_samples=None, **kwargs):
        super().__init__(*args, **kwargs)
        if n_samples is None:
            n_samples = self._n_rows
        self._n_samples = n_samples

    def _threshold_distances(self, xk):
        idxs = np.random.choice(self._n_rows, self._n_samples, replace=False)
        return np.abs(self._b[idxs] - self._A[idxs] @ xk)


class WindowedQuantile(Quantile):
    """Reject equations whose normalized residual is above a quantile of the most recent normalized residual values.

    Parameters
    ----------
    window_size : int, optional
        Number of recent normalized residual values used to compute the threshold quantile.

    Note
    ----
    ``WindowedQuantile`` also accepts the parameters of ``Quantile``.

    References
    ----------
    1. There will be a reference soon. Keep an eye out for that.
    """

    def __init__(self, *args, window_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        if window_size is None:
            window_size = self._n_rows
        self._window = deque([], maxlen=window_size)

    def _distance(self, xk, ik):
        distance = super()._distance(xk, ik)
        self._window.append(distance)
        return distance

    def _threshold_distances(self, xk):
        return self._window
