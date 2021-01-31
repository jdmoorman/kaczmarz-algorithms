import numpy as np
import pytest

import kaczmarz


def test_selectable_set(eye33, ones3):

    x0 = np.zeros(3)
    solver = kaczmarz.ParallelOrthoUpdate(eye33, ones3, x0, q=1)

    # Check that only one row is selected
    assert solver.ik == -1
    next(solver)
    assert solver.ik == -1
    next(solver)
    assert 1 == len(solver.ik)
    next(solver)
    assert 1 == len(solver.ik)
    next(solver)
    assert 1 == len(solver.ik)

    with pytest.raises(StopIteration):
        next(solver)
