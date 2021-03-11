import numpy as np
import pytest

import kaczmarz


def test_row_indexes(eye23, ones2):
    x = np.zeros(3)
    cyclic = kaczmarz.Cyclic(eye23, ones2)
    assert 0 == cyclic._select_row_index(x)
    assert 1 == cyclic._select_row_index(x)
    assert 0 == cyclic._select_row_index(x)
    assert 1 == cyclic._select_row_index(x)

    x0 = np.zeros(3)
    iterates = kaczmarz.Cyclic.iterates(eye23, ones2, x0)
    iterator = iter(iterates)
    assert [0, 0, 0] == list(next(iterator))
    assert [1, 0, 0] == list(next(iterator))
    assert [1, 1, 0] == list(next(iterator))
    with pytest.raises(StopIteration):
        next(iterator)


def test_ordered_cyclic(eye23, ones2):
    x = np.zeros(3)
    cyclic = kaczmarz.Cyclic(eye23, ones2, order=[1, 0])
    assert 1 == cyclic._select_row_index(x)
    assert 0 == cyclic._select_row_index(x)
    assert 1 == cyclic._select_row_index(x)
    assert 0 == cyclic._select_row_index(x)

    x0 = np.zeros(3)
    iterates = kaczmarz.Cyclic.iterates(eye23, ones2, x0, order=[1, 0])
    iterator = iter(iterates)
    assert [0, 0, 0] == list(next(iterator))
    assert [0, 1, 0] == list(next(iterator))
    assert [1, 1, 0] == list(next(iterator))
    with pytest.raises(StopIteration):
        next(iterator)
