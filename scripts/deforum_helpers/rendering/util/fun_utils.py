import collections.abc
from functools import reduce
from itertools import chain
from typing import Callable, TypeVar


def flat_map(func, iterable):
    """Applies a function to each element in an iterable and flattens the results."""
    mapped_iterable = map(func, iterable)
    if any(isinstance(item, collections.abc.Iterable) for item in mapped_iterable):
        return chain.from_iterable(mapped_iterable)
    else:
        return mapped_iterable


T = TypeVar('T')


def tube(*funcs: Callable[[T], T], is_do_process=lambda: True) -> Callable[[T], T]:
    """Tubes a value through a sequence of functions with a predicate `is_do_process` for skipping."""
    return lambda value: reduce(lambda x, f: f(x) if is_do_process() else x, funcs, value)
