import numpy as np
import pytest

import kaczmarz


@pytest.fixture()
def DummyStrategy():
    class _DummyStrategy(kaczmarz.Base):
        def select_row_index(self, xk):
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
    """Forgetting to implement the abstract method `select_row_index` should result in a TypeError on instantiation."""
    with pytest.raises(TypeError):
        NonStrategy()

    DummyStrategy(eye23, ones2)


def test_initial_guess(eye23, ones2, DummyStrategy):
    # Does the default initial iterate have the right shape?
    iterates = DummyStrategy.iterates(eye23, ones2)
    assert (3,) == next(iter(iterates)).shape

    # Does the supplied initial iterate get used correctly?
    x0 = np.array([1, 2, 3])
    iterates = DummyStrategy.iterates(eye23, ones2, x0)
    assert list(x0) == list(next(iter(iterates)))


def test_maxiter(eye23, ones2, DummyStrategy):
    """Passing `maxiter=1` should cause the algorithm to terminate after one iteration."""

    # This is not the exact solution.
    x0 = np.array([0, 0, 0])

    iterates = DummyStrategy.iterates(eye23, ones2, x0, maxiter=0)
    terminates_after_n_iterations(iterates, 0)

    iterates = DummyStrategy.iterates(eye23, ones2, x0, maxiter=1)
    terminates_after_n_iterations(iterates, 1)


def test_tolerance(eye23, ones2, DummyStrategy):
    x_exact = np.array([1, 1, 0])

    # If we start at the answer, we're done.
    iterates = DummyStrategy.iterates(eye23, ones2, x_exact)
    terminates_after_n_iterations(iterates, 0)

    # Initial residual has norm 1.
    x0 = np.array([1, 0, 0])
    iterates = DummyStrategy.iterates(eye23, ones2, x0, tol=1.01)
    terminates_after_n_iterations(iterates, 0)


def test_row_norms_squared(eye23, ones2, DummyStrategy):
    """Passing row norms 2x too large causes steps to be half as big."""
    x0 = np.array([0, 0, 0])
    fake_row_norms_squared = np.array([2, 2])  # They should be [1, 1]
    iterates = DummyStrategy.iterates(
        eye23, ones2, x0, row_norms_squared=fake_row_norms_squared,
    )
    iterator = iter(iterates)
    assert [0, 0, 0] == list(next(iterator))
    assert [0.5, 0, 0] == list(next(iterator))  # Correct iterate would be [1, 0, 0]


def test_callback(eye23, ones2, DummyStrategy):
    """Callback function should be called after each iteration."""
    x0 = np.array([0, 0, 0])
    actual_iterates = []

    def callback(xk):
        actual_iterates.append(list(xk))

    iterator = iter(DummyStrategy.iterates(eye23, ones2, x0, callback=callback))
    next(iterator)
    assert actual_iterates == [[0, 0, 0]]
    next(iterator)
    assert actual_iterates == [[0, 0, 0], [1, 0, 0]]
