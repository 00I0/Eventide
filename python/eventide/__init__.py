# noinspection PyUnresolvedReferences
from ._eventide import (
    Parameter,
    ParameterChangePoint,
    Scenario,
    OffspringCriterion,
    IntervalCriterion,
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
from .collectors import Hist1D, Hist2D, TimeMatrix

__all__ = [
    # low-level
    "Parameter", "ParameterChangePoint", "Scenario",
    "OffspringCriterion", "IntervalCriterion", "LatinHypercubeSampler",
    # helpers
    "Parameters", "Hist1D", "Hist2D", "Simulator", "TimeMatrix"
]


# Thin alias that hides _Expr / _Hist* from end-users
def Simulator(*, sampler, scenario, criteria, collectors, num_trajectories: int, chunk_size: int, T_run: int,
              max_cases: int, max_workers: int, cutoff_day: int, validators: list = None):
    if validators is None: validators = []

    if validators:
        body = " && ".join(f"({expr})" for expr in validators)
    else:
        body = "true"

    cpp_collectors = []
    for c in collectors:
        if hasattr(c, "_col"):
            cpp_collectors.append(c._col)
        else:
            cpp_collectors.append(c)

    return _PySimulator(sampler, scenario, criteria, cpp_collectors, _Expr(body), num_trajectories,
                        chunk_size, T_run,
                        max_cases, max_workers, cutoff_day)
