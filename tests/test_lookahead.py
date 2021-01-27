import pytest

import kaczmarz


def test_simple_cases():
    A = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    b = [1, 2, 3]
    x0 = [0, 0, 0]
    iterator = iter(kaczmarz.MaxDistanceLookahead.iterates(A, b, x0))
    assert [0, 0, 0] == list(next(iterator))
    assert [0, 0, 3] == list(next(iterator))
    assert [0, 2, 3] == list(next(iterator))
    assert [1, 2, 3] == list(next(iterator))
    with pytest.raises(StopIteration):
        next(iterator)

    A = [
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 4],
    ]
    b = [1, 1, 1]
    x0 = [0, 0, 0]
    iterator = iter(kaczmarz.MaxDistanceLookahead.iterates(A, b, x0))
    assert [0, 0, 0] == list(next(iterator))
    assert [1, 0, 0] == list(next(iterator))
    assert [1, 0.5, 0] == list(next(iterator))
    assert [1, 0.5, 0.25] == list(next(iterator))
    with pytest.raises(StopIteration):
        next(iterator)

    A = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    b = [3, 4, 3]
    x0 = [1, 1, 1]
    iterates = kaczmarz.MaxDistanceLookahead.iterates(A, b, x0)
    iterator = iter(iterates)
    next(iterator)
    next(iterator)
    assert 1 == iterates.ik


def test_lookahead_case():
    A = [
        [1, 0],
        [0, 1],
        [1, -1],
    ]
    b = [0, 0, 0]
    x0 = [1, -0.5]
    iterates = kaczmarz.MaxDistanceLookahead.iterates(A, b, x0)
    iterator = iter(iterates)
    assert [1, -0.5] == list(next(iterator))
    assert [0, -0.5] == list(next(iterator))
    assert [0, 0] == list(next(iterator))
