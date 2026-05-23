from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ._eventide import CompiledExpression, PyAlternatingSimulator, Species
from .collectors import Collector
from .criterion import Criterion
from .parameters import Parameters
from .sampler import PreselectedSampler, Sampler
from .scenario import Scenario


@dataclass(frozen=True, slots=True)
class SpeciesConfiguration:
    parameters: Parameters
    sampler: Sampler
    scenario: Scenario
    criteria: list[Criterion]
    collectors: list[Collector]

    def _get_cpp_collectors(self, simulation_length: float, simulation_start_date: datetime):
        return [collector._get_cpp_collector(simulation_length, simulation_start_date) for collector in self.collectors]

    def _get_cpp_criteria(self, simulation_start_date: datetime):
        return [criterion._get_cpp_criterion(simulation_start_date) for criterion in self.criteria]


@dataclass(frozen=True, slots=True)
class AlternatingSimulator:
    host: SpeciesConfiguration
    vector: SpeciesConfiguration
    start_date: datetime
    num_trajectories: int
    chunk_size: int
    T_run: float
    max_cases: int
    max_workers: int
    root_species: Species = Species.HOST
    min_required: Optional[int] = None
    accepted: Optional[int] = None
    processed: Optional[int] = None

    def __post_init__(self) -> None:
        for label, sampler in (('host', self.host.sampler), ('vector', self.vector.sampler)):
            if isinstance(sampler, PreselectedSampler) and sampler.max_trials > 1 and self.chunk_size != 1:
                raise ValueError(
                    f'Chunk_size must be 1 when using {label} PreselectedSampler with max_trials > 1.'
                )

        shared_collectors = {id(collector) for collector in self.host.collectors}
        shared_collectors.intersection_update(id(collector) for collector in self.vector.collectors)
        if shared_collectors:
            raise ValueError('Host and vector collectors must be separate collector instances.')

    def run(self) -> None:
        py_sim = PyAlternatingSimulator(
            self.host.sampler._get_cpp_sampler(),
            self.vector.sampler._get_cpp_sampler(),
            self.host.scenario._get_cpp_scenario(self.start_date),
            self.vector.scenario._get_cpp_scenario(self.start_date),
            self.host._get_cpp_criteria(self.start_date),
            self.vector._get_cpp_criteria(self.start_date),
            self.host._get_cpp_collectors(self.T_run, self.start_date),
            self.vector._get_cpp_collectors(self.T_run, self.start_date),
            CompiledExpression(self.host.parameters.validator_expression),
            CompiledExpression(self.vector.parameters.validator_expression),
            self.num_trajectories,
            self.min_required if self.min_required else self.num_trajectories,
            self.chunk_size,
            self.T_run,
            self.max_cases,
            self.max_workers,
            self.root_species,
        )
        py_sim.run()
        object.__setattr__(self, 'accepted', int(py_sim.accepted))
        object.__setattr__(self, 'processed', int(py_sim.processed))

    @property
    def end_date(self) -> datetime:
        return self.start_date + timedelta(days=self.T_run)
