"""A module providing selection strategies for the Kaczmarz algorithm."""

from .abc import SelectionStrategy


class Cyclic(SelectionStrategy):
    """Cycle through the row indexes of a matrix in order, repeatedly.

    Parameters
    ----------
    A : 2darray
        A matrix whose rows should be cycled through.
    args : tuple
        Unneeded arguments, likely provided automatically by kaczmarz.iterates.
    kwargs : dict
        Unneeded arguments, likely provided automatically by kaczmarz.iterates.
    """

    def __init__(self, A, *args, **kwargs):
        self.n_rows = A.shape[0]
        self.row_index = -1

    def select_row_index(self, xk):
        """Get the next row index, in cyclic order.

        Parameters
        ----------
        xk : ndarray
            The current Kaczmarz iterate. Not used for this selection strategy.

        Returns
        -------
        int
            The next row index, in cyclic order.
        """
        self.row_index = (1 + self.row_index) % self.n_rows
        return self.row_index


strategies = {
    attr_name: value
    for attr_name, value in vars().items()
    if not attr_name.startswith("_") and attr_name != "SelectionStrategy"
}
