class Cyclic:
    """Cycle through the row indexes of a matrix in order, repeatedly.

    Parameters
    ----------
    A : 2darray
        A matrix whose rows should be cycled through.
    """

    def __init__(self, A, *args, **kwargs):
        self.n_rows = A.shape[0]
        self.row_index = -1

    def next_row_index(self, *args, **kwargs):
        """Get the next row index, in cyclic order.

        Returns
        -------
        int
            The next row index.
        """
        self.row_index = (1 + self.row_index) % self.n_rows
        return self.row_index
