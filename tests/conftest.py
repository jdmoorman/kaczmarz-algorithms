"""
Dummy conftest.py for clapsolver.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
https://pytest.org/latest/plugins.html

https://stackoverflow.com/questions/34466027/in-pytest-what-is-the-use-of-conftest-py-files
"""

import pytest


@pytest.fixture()
def LegitStrategy():
    class _LegitStrategy:
        def select_row_index(self, xk):
            return 1

    return _LegitStrategy


@pytest.fixture()
def NonStrategy():
    class _NonStrategy:
        pass

    return _NonStrategy
