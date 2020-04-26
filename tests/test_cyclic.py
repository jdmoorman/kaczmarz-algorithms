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


def test_solve_identity(eye33, ones3):
    x = kaczmarz.Cyclic.solve(eye33, ones3)
    assert [1, 1, 1] == list(x)


def test_solve_non_orthogonal_matrix():
    Alol = [
        [1, 0, 0],
        [2, 1, 0],
        [1, 2, 1],
        [0, 1, 2],
        [0, 0, 1],
    ]
    A = np.array(Alol)
    row_norms = np.linalg.norm(A, axis=1)
    x_exact = np.ones(3)
    b = A @ x_exact
    tol = 1e-5
    x = kaczmarz.Cyclic.solve(A, b, tol=tol)
    assert np.linalg.norm(A @ (x - x_exact) / row_norms) < tol
