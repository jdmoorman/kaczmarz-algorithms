import numpy as np

import kaczmarz


def test_identity():
    A = np.eye(3)
    b = np.ones(3)
    x = kaczmarz.solve(A, b)
    assert [1, 1, 1] == list(x)


def test_non_orthogonal_matrix():
    A = np.array([[1, 0, 0], [2, 1, 0], [1, 2, 1], [0, 1, 2], [0, 0, 1]])
    x_exact = np.ones(3)
    b = A @ x_exact
    tol = 1e-5
    x = kaczmarz.solve(A, b, tol=tol)
    assert np.linalg.norm(x - x_exact) < tol
