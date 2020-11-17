import numpy as np
import pytest

import kaczmarz


def test_selectable_set(eye33, ones3):

    x0 = np.zeros(3)
    solver = kaczmarz.RandomOrthoGraph(eye33, ones3)
    iterator = iter(solver)
    assert 3 == len(solver.selectable)
    next(iterator)
    assert 2 == len(solver.selectable)
    next(iterator)
    assert 1 == len(solver.selectable)
    next(iterator)
    assert 0 == len(solver.selectable)

    with pytest.raises(StopIteration):
        next(iterator)
