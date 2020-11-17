import numpy as np
import pytest

import kaczmarz


def test_selectable_set(eye33, ones3):

    x0 = np.zeros(3)
    solver = kaczmarz.RandomOrthoGraph(eye33, ones3, x0)

    # Length is 3 for two iterations because first iteration yields x0 without performing an update.
    assert 3 == len(solver.selectable)
    next(solver)
    assert 3 == len(solver.selectable)
    next(solver)
    assert 2 == len(solver.selectable)
    next(solver)
    assert 1 == len(solver.selectable)
    next(solver)
    assert 0 == len(solver.selectable)

    with pytest.raises(StopIteration):
        next(solver)
