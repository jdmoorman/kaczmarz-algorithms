import numpy as np

from kaczmarz.selection import Cyclic


def test_cyclic():
    A = np.empty((2, 3))
    x = np.empty(3)
    cyclic = Cyclic(A)
    assert 0 == cyclic.next_row_index(x)
    assert 1 == cyclic.next_row_index(x)
    assert 0 == cyclic.next_row_index(x)
    assert 1 == cyclic.next_row_index(x)
