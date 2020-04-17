from . import selection
from .abc import SelectionStrategy
from .exceptions import SelectionStrategyNotFoundError


def _selection_strategy_class_from_str(selection_strategy_str):
    try:
        return getattr(selection, selection_strategy_str)
    except AttributeError:
        selection_strategies = [
            item for item in dir(selection) if not item.startswith("_")
        ]
        raise SelectionStrategyNotFoundError(
            "{} is not a valid selection strategy. Try one of {}.".format(
                selection_strategy_str, selection_strategies
            )
        )


def get_selection_strategy(
    selection_strategy, *selection_strategy_args, **selection_strategy_kwargs
):
    """Instantiate the specified selection strategy.

    Parameters
    ----------
    selection_strategy : str or subclass(SelectionStrategy)
        The desired selection strategy
    selection_strategy_args : tuple
        Positional arguments for the selection strategy constructor.
    selection_strategy_kwargs : tuple
        Keyword arguments for the selection strategy constructor.
    """

    if not isinstance(selection_strategy, str) and not isinstance(
        selection_strategy, SelectionStrategy
    ):
        raise TypeError(
            "selection_strategy must be a string or a subclass of SelectionStrategy."
        )

    if isinstance(selection_strategy, str):
        selection_strategy = _selection_strategy_class_from_str(selection_strategy)

    return selection_strategy(*selection_strategy_args, **selection_strategy_kwargs)
