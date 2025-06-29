from datetime import datetime

from ._eventide import Scenario as _Scenario, ParameterChangePoint as _ParameterChangePoint, _Expr


class ParameterChangePoint:
    def __init__(self, parameter_name: str, change_date: datetime, expr: str | None = None):
        # new_value = None -> restore.
        self.__parameter_name = parameter_name
        self.__change_date = change_date
        self.__expr = expr

    def _get_cpp_parameter_change_point(self, simulation_start_date: datetime):
        change_time = (self.__change_date - simulation_start_date).days

        if self.__expr is None:
            return _ParameterChangePoint(change_time, self.__parameter_name)

        return _ParameterChangePoint(change_time, self.__parameter_name, _Expr(self.__expr))


class Scenario:
    def __init__(self, change_points: list[ParameterChangePoint]):
        self.__change_points = change_points

    def _get_cpp_scenario(self, simulation_start_date: datetime):
        cpp_change_points = [c._get_cpp_parameter_change_point(simulation_start_date) for c in self.__change_points]
        return _Scenario(cpp_change_points)
