from dataclasses import dataclass
from datetime import datetime, timedelta

# noinspection PyUnresolvedReferences
from ._eventide import (
    Parameter,
    LatinHypercubeSampler,
    _Expr,
    _Hist1D,
    _Hist2D,
    Simulator as _PySimulator,
    _TimeMatrixCollector
)

"""
eventide – friendly Python facade for the C++ branching-process core.
"""

# ---- pure-python helpers -------------------------------------------------
from .parameter import Parameters
from .collectors import Hist1D, Hist2D, TimeMatrix, Collector, ActiveSetSizeCollector, InfectionTimeCollector
from .criterion import IntervalCriterion, IndexOffspringCriterion, Criterion
from .scenario import Scenario, ParameterChangePoint

__all__ = [
    "Parameter", "ParameterChangePoint", "Scenario",
    "IndexOffspringCriterion", "IntervalCriterion", "LatinHypercubeSampler",
    "Parameters", "Hist1D", "Hist2D", "Simulator", "TimeMatrix", "ActiveSetSizeCollector", "InfectionTimeCollector"
]


@dataclass(frozen=True, slots=True)
class Simulator:
    parameters: Parameters
    sampler: LatinHypercubeSampler
    start_date: datetime
    scenario: Scenario
    criteria: list[Criterion]
    collectors: list[Collector]
    num_trajectories: int
    chunk_size: int
    T_run: int
    max_cases: int
    max_workers: int

    def run(self) -> None:
        validators = self.parameters.validators() or []
        body = " and ".join(f"({expr})" for expr in validators) if validators else "true"

        cpp_collectors = [c._get_cpp_collector(self.T_run, self.start_date) for c in self.collectors]
        cpp_criteria = [c._get_cpp_criterion(self.start_date) for c in self.criteria]

        _PySimulator(
            self.sampler,
            self.scenario._get_cpp_scenario(self.start_date),
            cpp_criteria,
            cpp_collectors,
            _Expr(body),
            self.num_trajectories,
            self.chunk_size,
            self.T_run,
            self.max_cases,
            self.max_workers
        ).run()

    @property
    def end_date(self) -> datetime:
        return self.start_date + timedelta(days=self.T_run)
