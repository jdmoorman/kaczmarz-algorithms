import numpy as np
import pytest
import scipy.sparse as sp

import kaczmarz


def orthogonal_rows():
    examples = []

    A = [
        [1, 0, 0],
        [0, 1, 0],
    ]
    x = [1, 2, 0]
    examples.append((A, x))
    A = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    x = [1, 2, 3]
    examples.append((A, x))
    A = [
        [1, 1, 0],
        [-1, 1, 0],
    ]
    x = [1, 2, 0]
    examples.append((A, x))
    A = [
        [1, 1, 0],
        [-1, 1, 0],
        [0, 0, 1],
    ]
    x = [1, 2, 3]
    examples.append((A, x))

    return examples


def underdetermined():
    examples = []

    A = [
        [0, 0, 1, 1],
    ]
    x = [0, 0, 1, 1]
    examples.append((A, x))
    A = [
        [0, 0, 0, 2],
    ]
    x = [0, 0, 0, 0.5]
    examples.append((A, x))
    A = [
        [0, 0, 0, 2],
        [1, 0, 0, 1],
    ]
    x = [1, 0, 0, 0.5]
    examples.append((A, x))
    A = [
        [2, 1, 2, 1],
        [1, 2, 1, 2],
        [2, 1, 2, 1],
    ]
    x = [0.5, 0.25, 0.5, 0.25]
    examples.append((A, x))

    return examples


def strategies():
    strategy_classes = []
    for name in dir(kaczmarz):
        if name.startswith("_"):
            continue
        attr = getattr(kaczmarz, name)
        if attr == kaczmarz.Base:
            continue
        if not issubclass(attr, kaczmarz.Base):
            continue
        strategy_classes.append(attr)

    return strategy_classes


@pytest.mark.parametrize("A,x_exact", orthogonal_rows() + underdetermined())
@pytest.mark.parametrize("Strategy", strategies())
def test_solve(A, x_exact, Strategy, allclose):
    """Check that solver works on list-of-lists, np.ndarray, and csr_matrix."""
    row_norms = np.linalg.norm(A, axis=1)
    tol = 1e-5
    Anp = np.array(A)
    Asp = sp.csr_matrix(A)
    b = Anp @ x_exact
    x_approx = Strategy.solve(A, b, tol=tol)
    assert np.linalg.norm(Anp @ (x_approx - x_exact) / row_norms) < tol
    assert allclose(x_approx, x_exact, rtol=10 * tol)
    x_approx = Strategy.solve(Anp, b, tol=tol)
    assert np.linalg.norm(Anp @ (x_approx - x_exact) / row_norms) < tol
    assert allclose(x_approx, x_exact, rtol=10 * tol)
    x_approx = Strategy.solve(Asp, b, tol=tol)
    assert np.linalg.norm(Anp @ (x_approx - x_exact) / row_norms) < tol
    assert allclose(x_approx, x_exact, rtol=10 * tol)


#     x = kaczmarz.Cyclic.solve(eye33, ones3)
#     assert [1, 1, 1] == list(x)
#
#
# def test_solve_non_orthogonal_matrix():
#     Alol = [
#     [1, 0, 0],
#     [2, 1, 0],
#     [1, 2, 1],
#     [0, 1, 2],
#     [0, 0, 1],
#     ]
#     A = np.array(Alol)
#     row_norms = np.linalg.norm(A, axis=1)
#     x_exact = np.ones(3)
#     b = A @ x_exact
#     tol = 1e-5
#     x = kaczmarz.Cyclic.solve(A, b, tol=tol)
#     assert np.linalg.norm(A @ (x - x_exact) / row_norms) < tol