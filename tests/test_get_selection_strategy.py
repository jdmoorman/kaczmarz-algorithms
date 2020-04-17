import numpy as np
import pytest

import kaczmarz
from kaczmarz._get_selection_strategy import (
    get_selection_strategy,
    selection_strategy_class_from_str,
)
from kaczmarz.exceptions import SelectionStrategyNotFoundError


def test_class_from_str():
    for strategy_name, StrategyClass in kaczmarz.selection.strategies.items():
        assert selection_strategy_class_from_str(strategy_name) == StrategyClass

    with pytest.raises(SelectionStrategyNotFoundError):
        selection_strategy_class_from_str("NotARealStrategy")

    with pytest.raises(SelectionStrategyNotFoundError):
        # Abstract base class should not pass the test.
        selection_strategy_class_from_str("SelectionStrategy")


def test_all_strategies_are_gettable():
    A = np.eye(3)
    b = np.ones(3)
    for strategy_name, StrategyClass in kaczmarz.selection.strategies.items():
        kaczmarz.iterates(A, b, selection_strategy=strategy_name)
        kaczmarz.iterates(A, b, selection_strategy=StrategyClass)


def test_non_strategy_raises_type_error(NonStrategy):
    with pytest.raises(TypeError):
        get_selection_strategy(NonStrategy)
