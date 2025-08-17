"""
eventide – a Python facade for the C++ branching-process core.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

# noinspection PyUnresolvedReferences
from ._eventide import PySimulator, CompiledExpression
# ---- pure-python helpers -------------------------------------------------
from .collectors import Histogram, Hist1D, Hist2D, TimeMatrix, Collector, ActiveSetSizeCollector, \
    InfectionTimeCollector, DrawCollector
from .criterion import IntervalCriterion, IndexOffspringCriterion, Criterion
from .parameter import Parameter
from .parameters import Parameters
from .sampler import Sampler, PreselectedSampler, LatinHypercubeSampler
from .scenario import Scenario, ParameterChangePoint

__all__ = [
    'Parameter', 'ParameterChangePoint', 'Scenario',
    'IndexOffspringCriterion', 'IntervalCriterion',
    'LatinHypercubeSampler', 'PreselectedSampler',
    'Parameters',
    'Histogram', 'Hist1D', 'Hist2D', 'Simulator', 'TimeMatrix', 'ActiveSetSizeCollector', 'InfectionTimeCollector',
    'DrawCollector'
]


@dataclass(frozen=True, slots=True)
class Simulator:
    """
    High‐level simulator binding.

    Wraps the C++ Simulator and merges results back into Python collector objects.

    Attributes:
        parameters: Parameter ranges.
        sampler: Parameter sampler (Latin‑Hypercube or Preselected).
        start_date: Simulation start date.
        scenario: Scenario of time‑varying parameter change points.
        criteria: List of acceptance/rejection criteria.
        collectors: List of data collectors to fill.
        num_trajectories: Total trajectories to attempt.
        chunk_size: Number of draws per internal batch.
        T_run: Simulation horizon length (days).
        max_cases: Per‑trajectory cap on new cases.
        max_workers: Number of parallel worker threads.
    """
    parameters: Parameters
    sampler: Sampler
    start_date: datetime
    scenario: Scenario
    criteria: list[Criterion]
    collectors: list[Collector]
    num_trajectories: int
    chunk_size: int
    T_run: int
    max_cases: int
    max_workers: int

    def __post_init__(self) -> None:
        """
        Validate inter‐dependent arguments.
        
        Raises:
            ValueError: If using PreselectedSampler with chunk_size != 1.
        """
        if isinstance(self.sampler, PreselectedSampler) and self.chunk_size != 1:
            raise ValueError('Chunk_size must be 1 when using PreselectedSampler.')

    def run(self) -> None:
        """
        Execute the simulation.

        Collectors passed into this object will be populated with results
        when this method returns.
        """
        validators = self.parameters.validators or []
        body = ' and '.join(f'({expr})' for expr in validators) if validators else 'true'

        cpp_collectors = [c._get_cpp_collector(self.T_run, self.start_date) for c in self.collectors]
        cpp_criteria = [c._get_cpp_criterion(self.start_date) for c in self.criteria]

        PySimulator(
            self.sampler._get_cpp_sampler(),
            self.scenario._get_cpp_scenario(self.start_date),
            cpp_criteria,
            cpp_collectors,
            CompiledExpression(body),
            self.num_trajectories,
            self.chunk_size,
            self.T_run,
            self.max_cases,
            self.max_workers
        ).run()

    @property
    def end_date(self) -> datetime:
        """
        datetime: The end date of the simulation (start_date + T_run days).
        """
        return self.start_date + timedelta(days=self.T_run)
