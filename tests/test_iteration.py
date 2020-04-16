import numpy as np
import pytest

from kaczmarz import Iteration
from kaczmarz.selection import Cyclic


@pytest.fixture()
def A():
    # fmt: off
    return np.array([[1, 0, 0],
                     [0, 1, 0]])
    # fmt: on


@pytest.fixture()
def b():
    return np.array([1, 1])


@pytest.fixture()
def x_exact():
    return np.array([1, 1, 0])


def terminates_in_one_iteration(iteration):
    iterator = iter(iteration)
    next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


def test_initial_guess(A, b):
    # Does the default initial iterate have the right shape?
    iteration = Iteration(A, b)
    assert (3,) == next(iter(iteration)).shape

    # Does the supplied initial iterate get used correctly?
    x0 = np.array([1, 2, 3])
    iteration = Iteration(A, b, x0)
    assert list(x0) == list(next(iter(iteration)))


def test_maxiter(A, b):
    # This is not the exact solution.
    x0 = np.array([0, 0, 0])

    iteration = Iteration(A, b, x0, maxiter=0)
    terminates_in_one_iteration(iteration)


def test_tolerance(A, b, x_exact):
    # If we start at the answer, we're done.
    iteration = Iteration(A, b, x_exact)
    terminates_in_one_iteration(iteration)

    # Initial residual has norm 1.
    x0 = np.array([1, 0, 0])
    iteration = Iteration(A, b, x0, tol=1.01)
    terminates_in_one_iteration(iteration)


def test_row_norms_squared(A, b):
    """Passing row norms 2x too large causes steps to be half as big."""
    x0 = np.array([0, 0, 0])
    fake_row_norms_squared = np.array([2, 2])  # They should be [1, 1]
    iteration = Iteration(
        A, b, x0, row_norms_squared=fake_row_norms_squared, selection_strategy=Cyclic(A)
    )
    iterator = iter(iteration)
    assert [0, 0, 0] == list(next(iterator))
    assert [0.5, 0, 0] == list(next(iterator))  # Correct iterate would be [1, 0, 0]
    assert [0.5, 0.5, 0] == list(next(iterator))  # Correct iterate would be [1, 1, 0]


def test_callback(A, b):
    """Callback function should be called after each iteration."""
    x0 = np.array([0, 0, 0])
    actual_iterates = []

    def callback(xk):
        actual_iterates.append(list(xk))

    iterator = iter(Iteration(A, b, x0, callback=callback))
    next(iterator)
    assert actual_iterates == [[0, 0, 0]]
    next(iterator)
    assert actual_iterates == [[0, 0, 0], [1, 0, 0]]
    next(iterator)
    assert actual_iterates == [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
