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
        self._i_to_neighbors = {
            i: np.argwhere(self._gramian[i, :]).flatten() for i in range(self._n_rows)
        }

        # Initially, any row whose equation is not satisfied is selectable.
        self._selectable = np.argwhere(self._A @ self._x0 - self._b).flatten()
        if p is None:
            p = np.ones((self._n_rows,))
        self._p = p

    def _update_selectable(self, ik):
        # Every time a row is selected, all of its neighbors become selectable, and itself becomes unselectable.
        newly_selectable = self._i_to_neighbors[ik]
        selectable_with_ik = np.union1d(self._selectable, newly_selectable)
        self._selectable = np.setdiff1d(selectable_with_ik, [ik], assume_unique=True)

    def _select_row_index(self, xk):
        unnormalized_p = self._p[self._selectable]
        p = unnormalized_p / unnormalized_p.sum()
        ik = np.random.choice(self._selectable, p=p)
        self._update_selectable(ik)
        return ik

    @property
    def selectable(self):
        """(s,) array: Selectable row indexes at the current iteration."""
        return self._selectable.copy()

class ParallelOrthoUpdate(kaczmarz.Base):
    """Perform multiple updates in parallel, using only rows which are mutually orthogonal

    Parameters
    ----------

    References
    ----------
    1. Nutini, Julie, et al.
       "Convergence rates for greedy Kaczmarz algorithms,
       and faster randomized Kaczmarz rules using the orthogonality graph."
       arXiv preprint arXiv:1612.07838 2016.
    """

    def __init__(self, *args, q=None, p=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._gramian = self._A @ self._A.T

        # Map each row index i to indexes of rows that are NOT orthogonal to it.
        self._i_to_neighbors = {}
        for i in range(self._n_rows):
            self._i_to_neighbors[i] = self._gramian[[i], :].nonzero()[1]

        if q is None:
            q = self._n_rows
        self._q = q
        if p is None:
            p = np.ones((self._n_rows,))
        self._p = p
        self._clique_sizes = []

    def _update_iterate(self, xk, ik_list):
        xkp1 = xk
        self._clique_sizes.append(len(ik_list))
        for ik in ik_list:
            ai = self._A[ik]
            bi = self._b[ik]
            xkp1 += (bi - ai @ xk) * ai
        return xkp1

    def _select_row_index(self, xk):
        ik_list = []
        selectable = np.ones(self._n_rows, dtype=np.bool)
        curr_p = self._p.copy()
        while len(ik_list) != self._q:
            ik = np.random.choice(self._n_rows, p=curr_p)
            if not selectable[ik]:
                raise Exception(
                    "Probability removal should prevent this from happening"
                )
            ik_list.append(ik)
            # Remove all rows from selectable set that are not orthogonal to ik
            selectable[self._i_to_neighbors[ik]] = False
            if np.any(selectable):
                new_p = np.zeros(self._n_rows)
                new_p[selectable] = curr_p[selectable]
                new_p /= np.sum(new_p)  # Renormalize probabilities
                curr_p = new_p
            else:
                break

        return ik_list
