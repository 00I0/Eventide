"""
Python criteria wrappers for C++ Criterion implementations.
"""

import abc
from datetime import datetime

# noinspection PyUnresolvedReferences
from ._eventide import (
    IntervalCriterion as _IntervalCriterion,
    OffspringCriterion as _OffspringCriterion
)


class Criterion(abc.ABC):
    """
    Base class for trajectory acceptance/rejection criteria.
    """

    @abc.abstractmethod
    def _get_cpp_criterion(self, simulation_start_date: datetime):
        """
        Create the underlying C++ criterion.

        Args:
            simulation_start_date: datetime start of simulation.

        Returns:
            C++ Criterion instance.
        """
        raise NotImplemented


class IndexOffspringCriterion(Criterion):
    """
    Require the index case to have between min_offsprings and max_offsprings children.

    Args:
        min_offsprings: minimum allowed children of root.
        max_offsprings: maximum allowed children of root.
    """

    def __init__(self, min_offsprings: int, max_offsprings: int):
        self.__min_offsprings = min_offsprings
        self.__max_offsprings = max_offsprings

    def _get_cpp_criterion(self, simulation_start_date):
        """
        Convert to C++ IndexOffspringCriterion.

        Args:
            simulation_start_date: datetime start of simulation.
        """
        return _OffspringCriterion(self.__min_offsprings, self.__max_offsprings)


class IntervalCriterion(Criterion):
    """
    Require the total cases in [start_date, end_date] to lie within [min_cases, max_cases].

    Args:
        start_date: window start datetime.
        end_date: window end datetime.
        min_cases: inclusive lower bound.
        max_cases: inclusive upper bound.
    """

    def __init__(self, start_date: datetime, end_date: datetime, min_cases: int, max_cases: int):
        self.__start_date = start_date
        self.__end_date = end_date
        self.__min_cases = min_cases
        self.__max_cases = max_cases

    def _get_cpp_criterion(self, simulation_start_date: datetime):
        """
        Convert to C++ IntervalCriterion.

        Args:
            simulation_start_date: datetime start of simulation.
        """
        return _IntervalCriterion(
            (self.__start_date - simulation_start_date).days,
            (self.__end_date - simulation_start_date).days,
            self.__min_cases,
            self.__max_cases
        )
