from __future__ import annotations

from typing import Dict, List, Tuple

from ._eventide import Parameter as _Param, LatinHypercubeSampler


class Parameters:
    _ORDER = ("R0", "k", "r", "alpha", "theta")

    def __init__(self, **kw: Tuple[float, float]):
        missing = [n for n in self._ORDER if n not in kw]
        if missing: raise ValueError(f"missing parameters: {missing}")

        self._params: Dict[str, _Param] = {k: _Param(k, *kw[k]) for k in self._ORDER}
        self._validators: List[str] = []

    def require(self, f: str) -> "Parameters":
        self._validators.append(f)
        return self

    def create_latin_hypercube_sampler(self, *, scramble: bool = True) -> LatinHypercubeSampler:
        return LatinHypercubeSampler(list(self._params.values()), scramble)

    def validators(self):
        return self._validators

    def __iter__(self):
        yield from self._params.values()

    def __repr__(self):
        return (
                'Parameters(' +
                ', '.join(f'{k}={v.min}…{v.max}' for k, v in self._params.items()) +
                f', validators={len(self._validators)})'
        )
