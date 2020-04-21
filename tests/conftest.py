"""
Dummy conftest.py for clapsolver.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
https://pytest.org/latest/plugins.html

https://stackoverflow.com/questions/34466027/in-pytest-what-is-the-use-of-conftest-py-files
"""

import numpy as np
import pytest

import kaczmarz


@pytest.fixture()
def DummyStrategy():
    class _DummyStrategy(kaczmarz.Base):
        def select_row_index(self, xk):
            return 1

    return _DummyStrategy


@pytest.fixture()
def NonStrategy():
    class _NonStrategy(kaczmarz.Base):
        pass

    return _NonStrategy


@pytest.fixture()
def A():
    rows = [
        [1, 0, 0],
        [0, 1, 0],
    ]
    return np.array(rows)


@pytest.fixture()
def b():
    return np.array([1, 1])


@pytest.fixture()
def x_exact():
    return np.array([1, 1, 0])
