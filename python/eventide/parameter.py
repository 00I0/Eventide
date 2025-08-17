"""
Parameter‐range helpers and Latin‐Hypercube sampler factory.
"""
from __future__ import annotations

# noinspection PyUnresolvedReferences
from ._eventide import Parameter as _Parameter


class Parameter:
    """
    Represents a parameter with a name, minimum value, and maximum value.

    This class provides a convenient interface for managing parameters with
    defined bounds. It encapsulates functionality for retrieving the parameter's
    name, minimum value, and maximum value.

    Args:
        name: The name of the parameter must be one of 'R0', 'k', 'r', 'alpha', 'theta'.
        min: The minimum value of the parameter.
        max: The maximum value of the parameter.
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, name: str, min: float, max: float):
        if name not in ('R0', 'k', 'r', 'alpha', 'theta'):
            raise ValueError(f'name must be one of R0, k, r, alpha, theta, not {name}')
        self._cpp_parameter = _Parameter(name, min, max)

    @property
    def name(self) -> str:
        """str: The name of the parameter."""
        return self._cpp_parameter.name()

    @property
    def min(self) -> float:
        """float: The minimum value of the parameter."""
        return self._cpp_parameter.min()

    @property
    def max(self) -> float:
        """float: The maximum value of the parameter."""
        return self._cpp_parameter.max()

    @property
    def cpp_parameter(self):
        """_Parameter: The underlying C++ Parameter object."""
        return self._cpp_parameter
