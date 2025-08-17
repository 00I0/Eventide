"""
Python‐side Sampler wrappers over C++ Sampler interface.
"""

import abc
from typing import List, Tuple

import numpy as np

# noinspection PyUnresolvedReferences
from ._eventide import (
    LatinHypercubeSampler as _LatinHypercubeSampler,
    PreselectedSampler as _PreselectedSampler,
)
from .collectors import DrawCollector
from .parameter import Parameter


class Sampler(abc.ABC):
    """Base class for samplers."""

    @abc.abstractmethod
    def _get_cpp_sampler(self):
        """Return the underlying C++ sampler instance."""
        raise NotImplemented


class LatinHypercubeSampler(Sampler):
    """
    Latin‐Hypercube sampler over 5 Parameter ranges.

    Args:
        parameters: List of 5 Parameter objects.
        scramble: Whether to randomly shuffle strata per dimension.
    """

    def __init__(self, parameters: List[Parameter], scramble: bool = True):
        self.__parameters = [parameter.cpp_parameter for parameter in parameters]
        self.__scramble = scramble

        self.__sampler = None

    def _get_cpp_sampler(self):
        """
        Instantiate (or return cached) C++ LatinHypercubeSampler.
        """
        if self.__sampler: return self.__sampler

        self.__sampler = _LatinHypercubeSampler(self.__parameters, self.__scramble)
        return self.__sampler


class PreselectedSampler(Sampler):
    """
    Sampler that draws repeatedly from a pre‐chosen list of parameter tuples.

    Args:
        draws: List of (R0, k, r, alpha, theta) tuples or a populated DrawCollector.
        max_trials: Max retry attempts per draw.
    """

    def __init__(self, draws: List[Tuple[float, float, float, float, float]] | DrawCollector, max_trials: int):
        if isinstance(draws, DrawCollector):
            draws = np.asarray(draws)
        self.__draws = draws
        self.__max_trials = max_trials

        self.__sampler = None

    def _get_cpp_sampler(self):
        """
        Instantiate (or return cached) C++ PreselectedSampler.
        """
        if self.__sampler: return self.__sampler

        self.__sampler = _PreselectedSampler(self.__draws, self.__max_trials)
        return self.__sampler
