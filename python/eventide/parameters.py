from __future__ import annotations

from typing import Tuple, Dict, List

from .parameter import Parameter
from .sampler import LatinHypercubeSampler


class Parameters:
    """
    Encapsulates named parameter ranges and validation expressions.

    Args:
        **kw: Keyword arguments mapping each of 'R0', 'k', 'r', 'alpha', 'theta' to a (min, max) tuple.

    Example:
        >>> pars = Parameters(R0=(0.1,1.0), k=(0.5,10), r=(0.01,0.99), alpha=(0.1,5), theta=(0.1,5))
        >>> pars.require('R0 * r < 1').require('alpha * theta < 10')
    """
    _ORDER = ('R0', 'k', 'r', 'alpha', 'theta')

    def __init__(self, **kw: Tuple[float, float]):
        """
        Args:
            **kw: Keyword arguments mapping each of 'R0', 'k', 'r', 'alpha', 'theta' to a (min, max) tuple.

        Raises:
            ValueError: If any required parameter is missing.
        """
        missing = [n for n in self._ORDER if n not in kw]
        if missing: raise ValueError(f'missing parameters: {missing}')

        self._params: Dict[str, Parameter] = {k: Parameter(k, *kw[k]) for k in self._ORDER}
        self._ranges: Dict[str, Tuple[float, float]] = {k: kw[k] for k in self._ORDER}
        self._validators: List[str] = []

    def require(self, expr: str) -> 'Parameters':
        """
        Add a validation boolean expression on parameters.

        Args:
            expr: A Python‐style boolean expression involving R0, k, r,alpha,theta.

        Returns:
            self (to allow chaining).
        """
        self._validators.append(expr)
        return self

    def create_latin_hypercube_sampler(self, *, scramble: bool = True) -> LatinHypercubeSampler:
        """
        Build a LatinHypercubeSampler over these ranges.

        Args:
            scramble: Whether to shuffle each dimension.

        Returns:
            LatinHypercubeSampler instance.
        """
        return LatinHypercubeSampler(list(self._params.values()), scramble)

    @property
    def validators(self) -> List[str]:
        """List[str]: The added validation expressions."""
        return self._validators

    def __iter__(self):
        """Iterate over Parameter objects in canonical order."""
        yield from self._params.values()

    def __repr__(self):
        return (
                'Parameters(' +
                ', '.join(f'{k}={v.min}…{v.max}' for k, v in self._params.items()) +
                f', validators={len(self._validators)})'
        )

    # noinspection PyPep8Naming
    @property
    def R0_range(self) -> Tuple[float, float]:
        """Tuple[float,float]: Range for R0."""
        return self._ranges['R0']

    @property
    def r_range(self) -> Tuple[float, float]:
        """Tuple[float,float]: Range for r."""
        return self._ranges['r']

    @property
    def k_range(self) -> Tuple[float, float]:
        """Tuple[float,float]: Range for k."""
        return self._ranges['k']

    @property
    def alpha_range(self) -> Tuple[float, float]:
        """Tuple[float,float]: Range for alpha."""
        return self._ranges['alpha']

    @property
    def theta_range(self):
        return self._ranges['theta']
