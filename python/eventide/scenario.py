"""
Python‐side Scenario and ParameterChangePoint wrappers.
"""

from datetime import datetime

# noinspection PyUnresolvedReferences
from ._eventide import (
    Scenario as _Scenario,
    ParameterChangePoint as _ParameterChangePoint,
    CompiledExpression
)


class ParameterChangePoint:
    """
    Schedule a parameter change or restoration at a given time.

    Args:
        parameter_name: One of "R0","k","r","alpha","theta".
        change_date: datetime of the change.
        expr: Optional expression string; if None, restores original draw.
    """

    def __init__(self, parameter_name: str, change_date: datetime, expr: str | None = None):
        self.__parameter_name = parameter_name
        self.__change_date = change_date
        self.__expr = expr

    def _get_cpp_parameter_change_point(self, simulation_start_date: datetime):
        """
        Convert to C++ ParameterChangePoint.

        Args:
            simulation_start_date: datetime start of simulation.
        """
        change_time = (self.__change_date - simulation_start_date).days

        if self.__expr is None:
            return _ParameterChangePoint(change_time, self.__parameter_name)

        return _ParameterChangePoint(change_time, self.__parameter_name, CompiledExpression(self.__expr))


class Scenario:
    """
    A sequence of ParameterChangePoints to apply during simulation.

    Args:
        change_points: list of ParameterChangePoint.
    """

    def __init__(self, change_points: list[ParameterChangePoint]):
        self.__change_points = change_points

    def _get_cpp_scenario(self, simulation_start_date: datetime):
        """
        Convert to C++ Scenario.

        Args:
            simulation_start_date: datetime start of simulation.
        """
        cpp_change_points = [c._get_cpp_parameter_change_point(simulation_start_date) for c in self.__change_points]
        return _Scenario(cpp_change_points)
