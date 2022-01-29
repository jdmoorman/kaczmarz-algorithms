"""A module providing selection strategies for the Kaczmarz algorithm."""

from collections import deque

import numpy as np
from scipy import sparse

import kaczmarz

from ._utils import scale_cols, scale_rows, square


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

    def __init__(self, *base_args, order=None, **base_kwargs):
        super().__init__(*base_args, **base_kwargs)
        self._row_index = -1
        if order is None:
            order = range(self._n_rows)
        self._order = order

    def _select_row_index(self, xk):
        self._row_index = (1 + self._row_index) % self._n_rows
        return self._order[self._row_index]


class MaxDistanceLookahead(kaczmarz.Base):
    """Choose equations which lead to the most progress after a 2 step lookahead."""

    def __init__(self, *base_args, **base_kwargs):
        super().__init__(*base_args, **base_kwargs)
        self._next_i = None
        self._gramian = self._A @ self._A.T
        self._gramian2 = square(self._gramian)

    def _select_row_index(self, xk):
        if self._next_i is not None:
            temp = self._next_i
            self._next_i = None
            return temp

        residual = self._b - self._A @ xk
        residual_2 = np.square(residual)
        cost_mat = np.array(
            residual_2[:, None]
            + residual_2[None, :]
            - 2 * scale_rows(scale_cols(self._gramian, residual), residual)
            + scale_rows(self._gramian2, residual_2)
        )
        best_cost = np.max(cost_mat)

        sort_idxs = np.argsort(residual_2)[::-1]
        best_i = sort_idxs[np.any(cost_mat[sort_idxs, :] == best_cost, axis=1)][0]
        self._next_i = np.argwhere(cost_mat[best_i] == best_cost)[0][0]
        return best_i


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
        residual = self._b - self._A @ xk
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


class RandomOrthoGraph(kaczmarz.Base):
    """Try to only sample equations which are not already satisfied.

    Use the orthogonality graph defined in [1] to decide which rows should
    be considered "selectable" at each iteration.

    Parameters
    ----------
    p : (m,) array_like, optional
        Sampling probability for each equation. Uniform by default.
        These probabilities will be re-normalized based on the selectable rows
        at each iteration.

    References
    ----------
    1. Nutini, Julie, et al.
       "Convergence rates for greedy Kaczmarz algorithms,
       and faster randomized Kaczmarz rules using the orthogonality graph."
       arXiv preprint arXiv:1612.07838 2016.
    """

    def __init__(self, *args, p=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._gramian = self._A @ self._A.T

        # Map each row index i to indexes of rows that are NOT orthogonal to it.
        self._i_to_neighbors = {}
        for i in range(self._n_rows):
            self._i_to_neighbors[i] = self._gramian[[i], :].nonzero()[1]
        if p is None:
            p = np.ones((self._n_rows,))
        self._p = p

        self._selectable = self._A @ self._x0 - self._b != 0

    def _update_selectable(self, ik):
        self._selectable[self._i_to_neighbors[ik]] = True
        self._selectable[ik] = False

    def _select_row_index(self, xk):
        p = self._p.copy()
        p[~self._selectable] = 0
        p /= p.sum()
        ik = np.random.choice(self._n_rows, p=p)
        self._update_selectable(ik)
        return ik

    @property
    def selectable(self):
        """(s,) array(bool): Selectable rows at the current iteration."""
        return self._selectable.copy()


class Nonrepetitive(Random):
    """Do not sample the most recently projected row.

    References
    ----------
    1. Yaniv, Yotam, et al.
       "Selectable Set Randomized Kaczmarz."
       arXiv preprint arXiv:2110.04703 2021.
    """

    def _select_row_index(self, xk):
        i = super()._select_row_index(xk)

        # This loops infinitely if there is only one row.
        while i == self._ik:
            i = super()._select_row_index(xk)

        return i


class RelaxedGreedy(kaczmarz.Base):
    """Only sample equations that lead to a sufficiently large update.
    Parameters
    ----------
    theta : float, optional
        Parameter in the range [0,1]
    References
    ----------
    1. Zhong-Zhi Bai, Wen-Ting Wu,
    On relaxed greedy randomized Kaczmarz methods for solving large sparse linear systems,
    Applied Mathematics Letters, Volume 83, 2018, Pages 21-26,
    """

    def __init__(self, *args, theta=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if theta < 0 or theta > 1:
            raise Exception("Theta value outside parameter range [0, 1]")
        self._theta = theta
        self._row_norms_sq = self._row_norms ** 2
        self._fro_sq = np.sum(self._row_norms_sq)

    # Bai and Wu's algorithm
    def _select_row_index(self, xk):
        residual_sq = (self._b - self._A @ xk) ** 2
        residual_unnormalized_sq = self._row_norms_sq * residual_sq
        res_norm_sq = residual_unnormalized_sq.sum()
        epsilon = (
            self._theta / res_norm_sq * residual_sq + (1 - self._theta) / self._fro_sq
        )

        index_bool = residual_unnormalized_sq >= epsilon * res_norm_sq * (
            self._row_norms_sq
        )
        if ~np.any(index_bool):
            raise Exception("Index set empty")

        prob = residual_unnormalized_sq
        prob[~index_bool] = 0
        prob /= prob.sum()
        return np.random.choice(self._n_rows, p=prob)


class ParallelOrthoUpdate(RandomOrthoGraph):
    """Perform multiple updates in parallel, using only rows which are mutually orthogonal

    Parameters
    ----------
    q : int, optional
        Maximum number of updates to do in parallel.
    """

    def __init__(self, *args, q=None, **kwargs):
        super().__init__(*args, **kwargs)

        if q is None:
            q = self._n_rows
        self._q = q

    def _update_iterate(self, xk, tauk):
        """Do a sum of the usual updates."""
        # TODO: We should implement averaged kaczmarz as a mixin or something.
        xkp1 = xk
        for i in tauk:
            xkp1 = super()._update_iterate(xkp1, i)
        return xkp1

    def _select_row_index(self, xk):
        """Select a group of mutually orthogonal rows to project onto."""
        curr_selectable = self._selectable.copy()  # Equations that are not satisfied.
        tauk = []
        curr_p = self._p.copy()
        while len(tauk) != self._q and np.any(curr_selectable):
            curr_p[~curr_selectable] = 0  # Don't want to sample unselectable entries
            curr_p /= curr_p.sum()  # Renormalize probabilities

            i = np.random.choice(self._n_rows, p=curr_p)
            tauk.append(i)

            # Remove rows from selectable set that are not orthogonal to i
            curr_selectable[self._i_to_neighbors[i]] = False

        for i in tauk:
            self._update_selectable(i)

        return tauk
