"""A module providing selection strategies for the Kaczmarz algorithm."""

import kaczmarz


class Cyclic(kaczmarz.Base):
    """Cycle through the row indexes of a matrix in order, repeatedly.

    Parameters
    ----------
    A : (m, n) array
        A matrix whose rows should be cycled through.
    args : tuple
        Unneeded arguments, likely provided automatically by kaczmarz.Iterates.
    kwargs : dict
        Unneeded arguments, likely provided automatically by kaczmarz.Iterates.
    """

    def __init__(self, A, *args, **kwargs):
        super().__init__(A, *args, **kwargs)
        self.n_rows = A.shape[0]
        self.row_index = -1

    def _select_row_index(self, xk):
        """Get the next row index, in cyclic order.

        Parameters
        ----------
        xk : (n,) array
            The current Kaczmarz iterate. Not used for this selection strategy.

        Returns
        -------
        ik : int
            The next row index, in cyclic order.
        """
        self.row_index = (1 + self.row_index) % self.n_rows
        return self.row_index
