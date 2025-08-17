"""
Pure‑Python collector facades that wrap C++ DataCollector classes.
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Tuple, Any

import numpy as np

# noinspection PyUnresolvedReferences
from ._eventide import (
    CompiledExpression,
    Hist1D as _Hist1D,
    Hist2D as _Hist2D,
    TimeMatrixCollector as _TimeMatrixCollector,
    DrawCollector as _DrawCollector,
    ActiveSetSizeCollector as _ActiveSetSizeCollector,
    InfectionTimeCollector as _InfectionTimeCollector
)


class Collector(abc.ABC):
    """
    Base class for Python‐side collectors.

    Subclasses must implement _get_cpp_collector and __array__.
    """

    @abc.abstractmethod
    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime):
        """Return the underlying C++ collector instance."""
        raise NotImplemented

    @abc.abstractmethod
    def __array__(self, dtype=None) -> np.ndarray:
        """Return the collected data as a NumPy array."""
        raise NotImplemented


class Histogram(Collector, abc.ABC):
    """
    Abstract base for histogram collectors.

    Properties:
        range: Histogram axis range.
        name: Histogram label.
    """

    @property
    @abc.abstractmethod
    def range(self) -> Any:
        """Axis range(s) for the histogram (type varies)."""
        raise None

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Label for the histogram."""
        raise NotImplemented


class Hist1D(Histogram):
    """
    One‐dimensional histogram of a compiled expression over each **accepted** trajectory.

    Binning is over the final `expression(draw)` value, with `bins` equally‐spaced between `lo` and `hi`.

    Args:
        expr: Expression mapping Draw → float.
        bins: Number of bins.
        range: (lo, hi) value range.
        name: Optional label.
    """

    def __init__(self, expr: str, *, bins: int = 200, range: Tuple[float, float], name: str | None = None):
        self.__expr = expr
        self.__bins = bins
        self.__range = range
        self.__name = name or expr

        self.__collector = None

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Return the histogram counts as a NumPy array.

        Raises:
            ValueError: if simulation has not been run yet.
        """
        if not self.__collector: raise ValueError('Must create simulation first')
        return np.asarray(self.__collector.histogram(), dtype=np.int64)

    @property
    def range(self) -> Tuple[float, float]:
        """(lo, hi) range for the histogram."""
        return self.__range

    @property
    def name(self) -> str:
        """Label for the histogram."""
        return self.__name

    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime):
        """
        Instantiate (or return cached) C++ Hist1D collector.

        Args:
            simulation_length: T_run days.
            simulation_start_date: datetime start of simulation.
        """
        if self.__collector: return self.__collector

        lo, hi = self.__range
        self.__collector = _Hist1D(CompiledExpression(self.__expr), self.__bins, lo, hi)
        return self.__collector

    def __repr__(self):
        return f"Hist1D({self.__name}, bins={self.__bins})"


class Hist2D(Histogram):
    """
    Two‐dimensional histogram of two compiled expressions over each **accepted** trajectory.

    The result is a `bins×bins` matrix of counts for (X,Y) pairs lying in
    corresponding 2D bins evenly spaced over the provided ranges.

    Args:
        expression_tuple: Tuple of two expression strings (expr_x, expr_y).
        bins: Number of bins per dimension.
        range: ((lo_x, hi_x), (lo_y, hi_y)).
        name: Optional label.
    """

    def __init__(
            self,
            expression_tuple: Tuple[str, str],
            *,
            bins: int = 50,
            range: Tuple[Tuple[float, float], Tuple[float, float]],
            name: str | None = None
    ):
        self.__expression_tuple = expression_tuple
        self.__bins = bins
        self.__range = range

        expr_x, expr_y = expression_tuple
        self.__name = name or f'{expr_x}~{expr_y}'
        self.__collector = None

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Return the 2D histogram counts as a NumPy array.

        Raises:
            ValueError: if simulation has not been run yet.
        """
        if not self.__collector: raise ValueError('Must create simulation first')
        return np.asarray(self.__collector.histogram(), dtype=np.int64)

    @property
    def range(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """((lo_x,hi_x),(lo_y,hi_y)) axes ranges."""
        return self.__range

    @property
    def name(self) -> str:
        """Label for the histogram."""
        return self.__name

    @property
    def var_names(self) -> Tuple[str, str]:
        """Variable names (expr_x, expr_y)."""
        fx, fy = self.__expression_tuple
        return fx, fy

    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime):
        """
        Instantiate (or return cached) C++ Hist2D collector.

        Args:
            simulation_length: T_run days.
            simulation_start_date: datetime start of simulation.
        """
        if self.__collector: return self.__collector

        fx, fy = self.__expression_tuple
        (lox, hix), (loy, hiy) = self.__range
        self.__collector = _Hist2D(CompiledExpression(fx), CompiledExpression(fy), self.__bins, lox, hix, loy, hiy)
        return self.__collector

    def __repr__(self):
        return f'Hist2D({self.__name}, shape={self.__bins}×{self.__bins})'


class TimeMatrix(Collector):
    """
    Time‑matrix collector for tracking first post‑cutoff events for each **accepted** trajectory.

    Maintains a (Simulation Length + 2)×(Simulation Length + 2) matrix `M` where
      - `i = floor(final infection time)`
      - `j = floor(first infection after cutoffDay)`
    and `M[i][j]` counts trajectories that end at time `i` and first exceed
    the cutoff on day `j`.  Useful for computing “completion” probabilities
    across time.

    Args:
        cutoff_date: Python datetime of cutoff.
    """

    def __init__(self, cutoff_date: datetime):
        self.__cutoff_date = cutoff_date
        self.__collector = None

    def __array__(self, dtype=None) -> np.ndarray:
        """Return the time‑matrix as a NumPy 2D array."""

        return np.asarray(self.__collector.matrix(), dtype=np.int64)

    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime):
        """
        Instantiate (or return cached) C++ TimeMatrixCollector.

        Args:
            simulation_length: T_run days.
            simulation_start_date: datetime start of simulation.
        """
        if self.__collector: return self.__collector

        self.__collector = _TimeMatrixCollector(simulation_length, (self.__cutoff_date - simulation_start_date).days)
        return self.__collector

    def cutoff_day(self, simulation_start_date: datetime) -> int:
        """Cutoff day index relative to simulation start."""
        return (self.__cutoff_date - simulation_start_date).days

    @property
    def cutoff_date(self):
        """Cutoff date."""
        return self.__cutoff_date

    def __repr__(self):
        return f'TimeMatrix({self.__cutoff_date})'


class DrawCollector(Collector):
    """
    Stores the full 5‐parameter Draw of each **accepted** trajectory.

    After simulation, `np.asarray(draw_collector)` yields a (N_accepted × 5) array in the order [R0, k, r, alpha, theta].
    """

    def __init__(self):
        self.__collector = None

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Return an array of shape (N_accepted,5) of accepted draws.
        """
        return np.asarray(self.__collector.draws())

    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime):
        """
        Instantiate (or return cached) C++ DrawCollector.

        Args:
            simulation_length: T_run days.
            simulation_start_date: datetime start of simulation.
        """
        if self.__collector: return self.__collector

        self.__collector = _DrawCollector()
        return self.__collector

    def __repr__(self):
        return f'DrawCollector()'


class ActiveSetSizeCollector(Collector):
    """
    Measures the active infection set size at one fixed time per **accepted** trajectory.

    If an infection crosses the `collection_time`, increments an internal counter; at the end, that counter is recorded.

    Args:
        collection_date: datetime at which to sample the active set size.
    """

    def __init__(self, collection_date: datetime):
        self.__collector = None
        self.__collection_date = collection_date

    def __array__(self, dtype=None) -> np.ndarray:
        """Return array of active set sizes at the collection time."""

        return np.asarray(self.__collector.active_set_sizes())

    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime):
        """
        Instantiate (or return cached) C++ ActiveSetSizeCollector.

        Args:
            simulation_length: T_run days.
            simulation_start_date: datetime start of simulation.
        """
        if self.__collector: return self.__collector

        self.__collector = _ActiveSetSizeCollector(
            (self.__collection_date - simulation_start_date).total_seconds() / (24 * 3600)
        )
        return self.__collector

    @property
    def collection_date(self):
        return self.__collection_date

    def __repr__(self):
        return f'ActiveSetSizeCollector(collection_date = {self.collection_date})'


class InfectionTimeCollector(Collector):
    """
    Captures the full sequence of infection times in each **accepted** trajectory.
    """

    def __init__(self):
        self.__collector = None

    def __array__(self, dtype=None) -> np.ndarray:
        raise ValueError('Infection times are not stored as a NumPy array.')

    @property
    def infection_times(self) -> list[list[float]]:
        """Return a list of infection times for each accepted trajectory."""
        return self.__collector.infection_times()

    def _get_cpp_collector(self, simulation_length: int, simulation_start_date: datetime):
        """
        Instantiate (or return cached) C++ ActiveSetSizeCollector.

        Args:
            simulation_length: T_run days.
            simulation_start_date: datetime start of simulation.
        """
        if self.__collector: return self.__collector

        self.__collector = _InfectionTimeCollector()
        return self.__collector

    def __repr__(self):
        return 'InfectionTimeCollector'
