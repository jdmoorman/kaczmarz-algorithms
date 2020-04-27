"""A module providing selection strategies for the Kaczmarz algorithm."""

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
    1. Kaczmarz, S.: Angenäherte Auflösung von Systemen linearer Gleichungen.
       *Bulletin International de l’Académie Polonaise
       des Sciences et des Lettres.
       Classe des Sciences Mathématiques et Naturelles.
       Série A, Sciences Mathématiques* 35, 335–357 (1937)
    """

    def __init__(self, *base_args, **base_kwargs):
        super().__init__(*base_args, **base_kwargs)
        self.n_rows = self._A.shape[0]
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
