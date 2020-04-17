from abc import ABC, abstractmethod


class SelectionStrategy(ABC):
    """An abstract base class for Kaczmarz selection strategies."""

    @abstractmethod
    def select_row_index(self, xk):
        """Select a row to use for the next Kaczmarz update.

        Parameters
        ----------
        xk : ndarray
            The current Kaczmarz iterate.

        Returns
        -------
        int
            The index of the next row to use.
        """

    @classmethod
    def __subclasshook__(cls, C):
        if cls is SelectionStrategy:
            if any("select_row_index" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented
