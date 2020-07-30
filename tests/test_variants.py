from inspect import signature

import numpy as np
import pytest
import scipy.sparse as sp

import kaczmarz

np.random.seed(0)

strategies = []
for name in dir(kaczmarz):
    if name.startswith("_"):
        continue
    attr = getattr(kaczmarz, name)
    if attr == kaczmarz.Base:
        continue
    if not issubclass(attr, kaczmarz.Base):
        continue
    strategies.append(attr)


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


def overdetermined():
    examples = []
    A = [
        [2],
        [1],
        [2],
    ]
    x = [1]
    examples.append((A, x))
    A = [
        [2, 1],
        [1, 2],
        [2, 3],
    ]
    x = [2, 3]
    examples.append((A, x))
    return examples


systems = orthogonal_rows() + underdetermined() + overdetermined()


@pytest.mark.parametrize("A,x_exact", systems)
@pytest.mark.parametrize("Strategy", strategies)
def test_solve(A, x_exact, Strategy, allclose):
    """Solvers should accept list-of-lists, np.ndarray, and csr_matrix."""
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


@pytest.mark.parametrize("A,x_exact", systems)
@pytest.mark.parametrize("Strategy", strategies)
def test_iterates_converge_monotonically(A, x_exact, Strategy):
    """Check that solver works on list-of-lists, np.ndarray, and csr_matrix."""
    Anp = np.array(A)
    b = Anp @ x_exact
    errors = [np.linalg.norm(xk - x_exact) for xk in Strategy.iterates(Anp, b)]
    assert errors[1:] <= errors[:-1]


@pytest.mark.parametrize("A,x_exact", systems)
@pytest.mark.parametrize("Strategy", strategies)
def test_with_nonuniform_probabilities(A, x_exact, Strategy, allclose):
    """Solvers should accept list-of-lists, np.ndarray, and csr_matrix."""
    if "p" not in signature(Strategy).parameters:
        return

    A = np.array(A)
    row_norms = np.linalg.norm(A, axis=1)
    squared_row_norms = row_norms ** 2
    p = squared_row_norms / squared_row_norms.sum()
    tol = 1e-5
    b = A @ x_exact
    x_approx = Strategy.solve(A, b, tol=tol, p=p)
    assert np.linalg.norm(A @ (x_approx - x_exact) / row_norms) < tol
    assert allclose(x_approx, x_exact, rtol=10 * tol)
