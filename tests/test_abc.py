import pytest

from kaczmarz import selection
from kaczmarz.abc import SelectionStrategy


def test_issubclass(LegitStrategy, NonStrategy):
    """A class should be considered a valid selection strategy iff it implements `select_row_index`."""

    assert issubclass(LegitStrategy, SelectionStrategy)
    assert not issubclass(NonStrategy, SelectionStrategy)


def test_isinstance(LegitStrategy, NonStrategy):
    """An object should be considered a selection strategy iff it implements `select_row_index`."""

    legit_strategy = LegitStrategy()
    non_strategy = NonStrategy()

    assert isinstance(legit_strategy, SelectionStrategy)
    assert not isinstance(non_strategy, SelectionStrategy)


def test_actual_selection_strategies_are_subclasses_of_abc():
    for strategy_name, StrategyClass in selection.strategies.items():
        assert issubclass(StrategyClass, SelectionStrategy)


def test_undefined_abstract_method():
    """Forgetting to implement the abstract method `select_row_index` should result in a TypeError on instantiation."""

    class IncompleteStrategy(SelectionStrategy):
        pass

    with pytest.raises(TypeError):
        IncompleteStrategy()
