import abc
from datetime import datetime

from ._eventide import IntervalCriterion as _IntervalCriterion, OffspringCriterion as _OffspringCriterion


class Criterion(abc.ABC):
    @abc.abstractmethod
    def _get_cpp_criterion(self, simulation_start_date: datetime): raise NotImplemented


class IndexOffspringCriterion(Criterion):
    def __init__(self, min_offsprings: int, max_offsprings: int):
        self.__min_offsprings = min_offsprings
        self.__max_offsprings = max_offsprings

    def _get_cpp_criterion(self, simulation_start_date):
        return _OffspringCriterion(self.__min_offsprings, self.__max_offsprings)


class IntervalCriterion(Criterion):
    def __init__(self, start_date: datetime, end_date: datetime, min_cases: int, max_cases: int):
        self.__start_date = start_date
        self.__end_date = end_date
        self.__min_cases = min_cases
        self.__max_cases = max_cases

    def _get_cpp_criterion(self, simulation_start_date: datetime):
        return _IntervalCriterion(
            (self.__start_date - simulation_start_date).days,
            (self.__end_date - simulation_start_date).days,
            self.__min_cases,
            self.__max_cases
        )
