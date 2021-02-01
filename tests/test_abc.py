import numpy as np
import pytest

import kaczmarz


@pytest.fixture()
def DummyStrategy():
    class _DummyStrategy(kaczmarz.Base):
        def _select_row_index(self, xk):
            return 0

    return _DummyStrategy


@pytest.fixture()
def NonStrategy():
    class _NonStrategy(kaczmarz.Base):
        pass

    return _NonStrategy


def terminates_after_n_iterations(iterates, n):
    iterator = iter(iterates)
    for _ in range(n + 1):
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


def test_undefined_abstract_method(eye23, ones2, DummyStrategy, NonStrategy):
    """Forgetting to implement the abstract method ``select_row_index`` should result in a TypeError on instantiation."""
    with pytest.raises(TypeError):
        NonStrategy()

    DummyStrategy(eye23, ones2)


@pytest.mark.timeout(1)
def test_inconsistent_system_terminates(eye23, ones2, DummyStrategy, NonStrategy):
    """Make sure inconsistent systems do not run forever."""
    A = np.array([[1], [2]])
    b = np.array([1, 1])
    DummyStrategy.solve(A, b)


def test_single_row_matrix(DummyStrategy, allclose):
    A = np.array([[0, 0, 1, 1]])
    b = np.array([1])
    iterator = iter(DummyStrategy.iterates(A, b))
    next(iterator)
    x_exact = next(iterator)
    assert allclose([0, 0, 0.5, 0.5], x_exact)
    with pytest.raises(StopIteration):
        next(iterator)


def test_iterate_shape(eye23, ones2, DummyStrategy):
    """Row selected at each iteration should be accessable through the .ik attribute."""
    x0 = np.array([0, 0, 0])
    iterator = iter(DummyStrategy(eye23, ones2, x0))
    assert x0.shape == next(iterator).shape
    assert x0.shape == next(iterator).shape
    x0 = np.array([[0], [0], [0]])
    iterator = iter(DummyStrategy(eye23, ones2, x0))
    assert x0.shape == next(iterator).shape
    assert x0.shape == next(iterator).shape
    iterator = iter(DummyStrategy(eye23, ones2.reshape(-1)))
    assert (3,) == next(iterator).shape
    assert (3,) == next(iterator).shape
    iterator = iter(DummyStrategy(eye23, ones2.reshape(-1, 1)))
    assert (3, 1) == next(iterator).shape
    assert (3, 1) == next(iterator).shape


def test_initial_guess(eye23, ones2, DummyStrategy):
    # Does the default initial iterate have the right shape?
    iterates = DummyStrategy.iterates(eye23, ones2)
    assert (3,) == next(iter(iterates)).shape

    # Does the supplied initial iterate get used correctly?
    x0 = np.array([1, 2, 3])
    iterates = DummyStrategy.iterates(eye23, ones2, x0)
    assert list(x0) == list(next(iter(iterates)))


def test_ik(eye23, ones2, zeros3, DummyStrategy):
    """Row selected at each iteration should be accessable through the .ik attribute."""
    iterates = DummyStrategy(eye23, ones2, zeros3)
    iterator = iter(iterates)
    next(iterator)
    assert -1 == iterates.ik
    next(iterator)
    assert 0 == iterates.ik
    next(iterator)
    assert 0 == iterates.ik


def test_maxiter(eye23, ones2, zeros3, DummyStrategy):
    """Passing ``maxiter=n`` should cause the algorithm to terminate after n iterations."""

    # [0, 0, 0] is not the exact solution.
    args = [eye23, ones2, zeros3]

    iterates = DummyStrategy.iterates(*args, maxiter=0)
    terminates_after_n_iterations(iterates, 0)

    iterates = DummyStrategy.iterates(*args, maxiter=1)
    terminates_after_n_iterations(iterates, 1)

    for maxiter in range(1, 5):
        iterates = DummyStrategy.iterates(*args, maxiter=maxiter, tol=None)
        terminates_after_n_iterations(iterates, maxiter)

    with pytest.raises(ValueError):
        iterates = DummyStrategy.iterates(*args, maxiter=None, tol=None)


def test_solve(eye23, ones2, zeros3, DummyStrategy):
    # [0, 0, 0] is not the exact solution.

    x = DummyStrategy.solve(eye23, ones2, zeros3, maxiter=0)
    assert [0, 0, 0] == list(x)

    x = DummyStrategy.solve(eye23, ones2, zeros3, maxiter=1)
    assert [1, 0, 0] == list(x)


def test_tolerance(eye23, ones2, DummyStrategy):
    x_exact = np.array([1, 1, 0])

    # If we start at the answer, we're done.
    iterates = DummyStrategy.iterates(eye23, ones2, x_exact)
    terminates_after_n_iterations(iterates, 0)

    # Initial residual has norm 1.
    x0 = np.array([1, 0, 0])
    iterates = DummyStrategy.iterates(eye23, ones2, x0, tol=1.01)
    terminates_after_n_iterations(iterates, 0)


def test_callback(eye23, ones2, zeros3, DummyStrategy):
    """Callback function should be called after each iteration."""
    actual_iterates = []

    def callback(xk):
        actual_iterates.append(list(xk))

    iterator = iter(DummyStrategy.iterates(eye23, ones2, zeros3, callback=callback))
    next(iterator)
    assert actual_iterates == [[0, 0, 0]]
    next(iterator)
    assert actual_iterates == [[0, 0, 0], [1, 0, 0]]


def test_sparse(speye23, ones2, zeros3, DummyStrategy):
    iterator = iter(DummyStrategy.iterates(speye23, ones2, zeros3))

    assert [0, 0, 0] == list(next(iterator))
    assert [1, 0, 0] == list(next(iterator))


def test_array_like(eye23, ones2, zeros3, DummyStrategy):
    iterator = iter(
        DummyStrategy.iterates(eye23.tolist(), ones2.tolist(), zeros3.tolist())
    )

    assert [0, 0, 0] == list(next(iterator))
    assert [1, 0, 0] == list(next(iterator))


def test_iterates_are_copies(speye23, ones2, zeros3, DummyStrategy):
    """Check that modifying the iterate inplace does not affect the underlying iteration."""
    iterator = iter(DummyStrategy.iterates(speye23, ones2, zeros3))

    xk = next(iterator)
    assert [0, 0, 0] == list(xk)
    xk[:] = np.inf
    xk = next(iterator)
    assert [1, 0, 0] == list(xk)
    xk[:] = np.inf
    xk = next(iterator)
    assert [1, 0, 0] == list(xk)
