from __future__ import annotations

import abc
from datetime import datetime
from typing import Tuple, Union

import numpy as np

from ._eventide import _Hist1D, _Hist2D, _Expr, _TimeMatrixCollector


class Collector(abc.ABC):
    @abc.abstractmethod
    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime): raise NotImplemented


class Hist1D(Collector):

    def __init__(self, f: str, *, bins: int = 200, range: Tuple[float, float], name: str | None = None):
        self.__f = f
        self.__bins = bins
        self.__range = range
        self.__name = name or f
        self._range = range

        self.__collector = None

    def numpy(self) -> np.ndarray:
        if not self.__collector: raise ValueError('Must create simulation first')
        return np.asarray(self.__collector.histogram(), dtype=np.int64)

    @property
    def range(self):
        return self.__range

    @property
    def name(self):
        return self.__name

    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime):
        if self.__collector: return self.__collector

        lo, hi = self.__range
        self.__collector = _Hist1D(_Expr(self.__f), self.__bins, lo, hi)
        return self.__collector

    def __repr__(self):
        return f"Hist1D({self.__name}, bins={self.__bins})"


class Hist2D(Collector):
    def __init__(self, f: Union[Tuple[str, str]], *, bins: int = 50,
                 range: Tuple[Tuple[float, float], Tuple[float, float]], name: str | None = None):
        self.__f = f
        self.__bins = bins
        self.__range = range

        fx, fy = f
        self.__name = name or f'{fx}~{fy}'
        self.__collector = None

    def numpy(self) -> np.ndarray:
        if not self.__collector: raise ValueError('Must create simulation first')
        return np.asarray(self.__collector.histogram(), dtype=np.int64)

    @property
    def range(self):
        return self.__range

    @property
    def name(self):
        return self.__name

    @property
    def var_names(self):
        fx, fy = self.__f
        return fx, fy

    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime):
        if self.__collector: return self.__collector

        fx, fy = self.__f
        (lox, hix), (loy, hiy) = self.__range
        self.__collector = _Hist2D(_Expr(fx), _Expr(fy), self.__bins, lox, hix, loy, hiy)
        return self.__collector

    def __repr__(self):
        return f'Hist2D({self.__name}, shape={self.__bins}×{self.__bins})'


class TimeMatrix(Collector):
    def __init__(self, cutoff_date: datetime):
        self.__cutoff_date = cutoff_date
        self.__collector = None

    def numpy(self):
        return np.asarray(self.__collector.matrix(), dtype=np.int64)

    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime):
        if self.__collector: return self.__collector

        self.__collector = _TimeMatrixCollector(simulation_length, (self.__cutoff_date - simulation_start_date).days)
        return self.__collector

    def cutoff_day(self, simulation_start_date: datetime):
        return (self.__cutoff_date - simulation_start_date).days

    def __repr__(self):
        return f'TimeMatrix({self.__cutoff_date})'
