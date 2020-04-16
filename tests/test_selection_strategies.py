import numpy as np
import pytest

from kaczmarz import Iterates
from kaczmarz.selection import Cyclic


def test_cyclic():
    A = np.empty((2, 3))
    x = np.empty(3)
    cyclic = Cyclic(A)
    assert 0 == cyclic.next_row_index(x)
    assert 1 == cyclic.next_row_index(x)
    assert 0 == cyclic.next_row_index(x)
    assert 1 == cyclic.next_row_index(x)

    A = np.eye(3)
    b = np.ones(3)
    x0 = np.zeros(3)
    iteration = Iterates(A, b, x0, selection_strategy=Cyclic(A))
    iterator = iter(iteration)
    assert [0, 0, 0] == list(next(iterator))
    assert [1, 0, 0] == list(next(iterator))
    assert [1, 1, 0] == list(next(iterator))
    assert [1, 1, 1] == list(next(iterator))
    with pytest.raises(StopIteration):
        next(iterator)
