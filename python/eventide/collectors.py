from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from ._eventide import _Hist1D, _Hist2D, _Expr, _TimeMatrixCollector


class Hist1D:

    def __init__(self, f: str, *, bins: int = 200, range: Tuple[float, float] | None = None,
                 name: str | None = None):
        self._range = range

        if range is None:
            raise ValueError('range is None')
        lo, hi = range
        self._col = _Hist1D(_Expr(f), bins, lo, hi)
        self.name = name or f

    def numpy(self) -> np.ndarray:
        return np.asarray(self._col.histogram(), dtype=np.int64)

    @property
    def _col(self):
        return self.__collector

    @property
    def range(self):
        return self._range

    @_col.setter
    def _col(self, col):
        self.__collector = col

    def __repr__(self):
        return f"Hist1D({self.name}, bins={len(self._col.bins())})"


class Hist2D:
    def __init__(self, f: Union[Tuple[str, str]], *, bins: int = 50,
                 range: Tuple[Tuple[float, float], Tuple[float, float]] | None = None, name: str | None = None):
        self._range = range
        fx, fy = f
        if range is None:
            raise ValueError('range is None')
        (lox, hix), (loy, hiy) = range
        self.__collector = _Hist2D(_Expr(fx), _Expr(fy), bins, lox, hix, loy, hiy)
        self.name = name or f"{fx}~{fy}"

    def numpy(self) -> np.ndarray:
        return np.asarray(self._col.histogram(), dtype=np.int64)

    @property
    def range(self):
        return self.range

    @property
    def _col(self):
        return self.__collector

    def __repr__(self):
        mat = self._col.bins()
        return f'Hist2D({self.name}, shape={len(mat)}×{len(mat[0])})'


class TimeMatrix:
    def __init__(self, T: int, cutoff_day: int):
        self.__T = T
        self.__cutoff_day = cutoff_day
        self.__collector = _TimeMatrixCollector(T, cutoff_day)

    @property
    def _col(self):
        return self.__collector

    def numpy(self):
        return np.asarray(self._col.matrix(), dtype=np.int64)

    def __repr__(self):
        return f'TimeMatrix({self.__T}, {self.__cutoff_day})'
