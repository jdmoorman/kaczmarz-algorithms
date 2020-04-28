"""
Dummy conftest.py for clapsolver.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
https://pytest.org/latest/plugins.html

https://stackoverflow.com/questions/34466027/in-pytest-what-is-the-use-of-conftest-py-files
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def eye33():
    return np.eye(3)


@pytest.fixture
def eye23():
    return np.eye(2, 3)


@pytest.fixture
def speye23():
    return sp.csr_matrix(np.eye(2, 3))


@pytest.fixture
def zeros3():
    return np.zeros(3)


@pytest.fixture
def ones3():
    return np.ones(3)


@pytest.fixture()
def ones2():
    return np.ones(2)
