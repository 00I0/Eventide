"""Snapshot probabilities for the host/vector Gamma–Poisson process.

Host silence permits unobserved vector infections. For each existing host,
future vectors are thinned by their probability of causing a host infection.
Only one further vector generation is needed: a later host already violates
silence. Existing vectors contribute their remaining direct host offspring.

Parameters belong to the parent at its infection time, as in the simulator;
the root uses R0 and all other individuals use R0*r. Index bounds constrain
direct vector offspring of the host root. Quiet-period reweighting uses a
ratio of posterior means, evaluated in log space. Finite-horizon integrals
use Gauss–Legendre quadrature in Gamma survival-probability coordinates.
"""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammainc, gammaincc, gammainccinv, gammaln, logsumexp, xlogy


PARAMETER_NAMES = ("R0", "k", "r", "alpha", "theta")


@dataclass(frozen=True)
class ParameterChange:
    day: float
    parameter: str
    factor: float

    def __post_init__(self):
        if not np.isfinite(self.day) or self.day <= 0:
            raise ValueError("Control days must be finite and positive.")
        if self.parameter not in PARAMETER_NAMES or not np.isfinite(self.factor) or self.factor < 0:
            raise ValueError("Controls require a known parameter and a finite nonnegative factor.")


def parameters_at(draw, times, changes=()):
    """Return parent-specific [R0, k, r, alpha, theta] arrays.

    Changes apply strictly after their boundary, matching the C++ segment
    convention. Changes to a parent's offspring parameters are not retroactive.
    """
    values = np.broadcast_to(np.asarray(draw, dtype=float), np.shape(times) + (5,)).copy()
    for change in sorted(changes, key=lambda item: item.day):
        values[..., PARAMETER_NAMES.index(change.parameter)] *= np.where(
            np.asarray(times) > change.day, change.factor, 1.0,
        )
    return values


def _validate_parameters(values):
    if np.any(~np.isfinite(values)) or np.any(values[..., [0, 2]] < 0):
        raise ValueError("Reproduction parameters must be finite and nonnegative.")
    if np.any(values[..., [1, 3, 4]] <= 0):
        raise ValueError("k, alpha and theta must be finite and positive.")


def _history(host_pairs, vector_pairs, cutoff):
    pairs_by_species = []
    for pairs in (host_pairs, vector_pairs):
        pairs = np.asarray(pairs, dtype=float)
        if pairs.size == 0:
            pairs = np.empty((0, 2))
        if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(~np.isfinite(pairs)) or np.any(pairs < 0):
            raise ValueError("Trees must contain finite nonnegative parent-child pairs.")
        pairs = pairs[pairs[:, 1] <= cutoff]
        pairs_by_species.append(pairs)
    host_pairs, vector_pairs = pairs_by_species
    if np.count_nonzero(np.all(host_pairs == 0, axis=1)) != 1 or np.any(vector_pairs[:, 1] == 0):
        raise ValueError("Each history must contain exactly one host root at zero.")
    times = [pairs[:, 1] for pairs in pairs_by_species]
    if any(np.unique(species_times).size != len(species_times) for species_times in times):
        raise ValueError("Infection times must uniquely identify individuals within each species.")
    counts = []
    for species_index in (0, 1):
        edges = pairs_by_species[1 - species_index]
        edges = edges[edges[:, 1] > 0]
        if np.any(edges[:, 1] < edges[:, 0]) or np.any(~np.isin(edges[:, 0], times[species_index])):
            raise ValueError("Each child must follow a parent of the opposite species in this history.")
        parent_times, offspring = np.unique(edges[:, 0], return_counts=True)
        by_parent = dict(zip(parent_times, offspring))
        counts.append(np.array([by_parent.get(time, 0) for time in times[species_index]]))
    return times, counts


def _host_risk(times, params, vector_draw, vector_changes, cutoff, horizons, nodes, weights):
    ages = cutoff - times
    risks = np.zeros((len(times), len(horizons)))
    boundaries = sorted({cutoff, *(change.day for change in vector_changes if change.day > cutoff)})
    boundaries.append(np.inf)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        vector_params = parameters_at(vector_draw, np.nextafter(start, np.inf), vector_changes)
        _validate_parameters(vector_params)
        reproduction = vector_params[0] * vector_params[2]
        if reproduction == 0:
            continue
        end = cutoff + horizons
        upper = np.minimum(stop, end)
        active = upper > start
        if not np.any(active):
            continue
        lower_tail = gammaincc(params[:, 3], (start - times) / params[:, 4])
        upper_tail = gammaincc(params[:, 3, None], (upper[active][None, :] - times[:, None]) / params[:, 4, None])
        mass = np.maximum(lower_tail[:, None] - upper_tail, 0.0)
        active_horizons = horizons[active]
        integrated = np.zeros_like(mass)
        infinite = np.isinf(active_horizons)
        integrated[:, infinite] = mass[:, infinite] * (
            -np.expm1(-vector_params[1] * np.log1p(reproduction / vector_params[1]))
        )
        finite = ~infinite
        if np.any(finite):
            probabilities = upper_tail[:, finite, None] + mass[:, finite, None] * nodes
            delays = params[:, 4, None, None] * gammainccinv(params[:, 3, None, None], probabilities)
            remaining = np.maximum(
                end[active][finite][None, :, None] - times[:, None, None] - delays, 0.0,
            )
            host_mass = gammainc(vector_params[3], remaining / vector_params[4])
            risk = -np.expm1(-vector_params[1] * np.log1p(reproduction * host_mass / vector_params[1]))
            integrated[:, finite] = mass[:, finite] * np.sum(risk * weights, axis=-1)
        risks[:, active] += integrated
    tail = gammaincc(params[:, 3], ages / params[:, 4])
    return np.clip(risks, 0, tail[:, None])


def _parent_components(times, counts, params, risks, cutoff, bounds=None):
    tail = gammaincc(params[:, 3], (cutoff - times) / params[:, 4])
    reproduction = params[:, 0] * params[:, 2]
    reproduction = np.where(times == 0, params[:, 0], reproduction)
    if np.any((reproduction == 0) & (counts > 0)):
        raise ValueError("A zero-reproduction parent has observed offspring.")
    shapes = params[:, 1] + counts
    inverse_rates = reproduction / (params[:, 1] + reproduction * (1 - tail))
    log_components = -shapes[:, None] * np.log1p(risks * inverse_rates[:, None])
    if bounds is not None:
        root = int(np.flatnonzero(times == 0)[0])
        lower, upper = bounds
        minimum = max(0, lower - int(counts[root]))
        maximum = upper - int(counts[root])
        if maximum < minimum:
            raise ValueError("History violates the index-offspring bounds.")
        future = np.arange(minimum, maximum + 1)
        probability = tail[root] * inverse_rates[root]
        log_weights = (
            gammaln(shapes[root] + future) - gammaln(shapes[root]) - gammaln(future + 1)
            - shapes[root] * np.log1p(probability)
            + xlogy(future, probability / (1 + probability))
        )
        normalizer = logsumexp(log_weights)
        if not np.isfinite(normalizer):
            raise ValueError("Index-offspring restriction has zero numerical probability.")
        safe_probability = np.clip(1 - risks[root] / tail[root], 0, 1) if tail[root] > 0 else np.ones(risks.shape[1])
        log_components[root] = logsumexp(
            (log_weights - normalizer)[:, None] + xlogy(future[:, None], safe_probability), axis=0,
        )
    return np.sum(log_components, axis=0)


def conditional_extinction_probability(
        host_pairs, vector_pairs, host_draws, vector_draws, horizons, cutoff,
        *, index_offspring_bounds=None, host_changes=(), vector_changes=(), quadrature_order=48, workers=1,
):
    """Return host cessation and all-transmission cessation given host silence.

    Both curves condition on no HOST infections during (cutoff, cutoff+T].
    ``host`` means no later host infections ever; ``all`` means no infections
    of either species after cutoff. Neither event means eventual extinction
    after allowing further infections. An impossible conditioning event gives
    NaN. Histories must be complete through cutoff but can include future
    edges; those future edges are ignored, never used to declare extinction.
    Omitted or None index bounds leave the root offspring unrestricted.
    """
    horizons = np.asarray(horizons, dtype=float)
    if horizons.ndim != 1 or np.any(np.isnan(horizons)) or np.any(horizons < 0):
        raise ValueError("Horizons must be a nonnegative one-dimensional array.")
    if not np.isfinite(cutoff) or cutoff < 0 or quadrature_order < 8 or workers < 1:
        raise ValueError("Use a finite nonnegative cutoff, quadrature_order >= 8 and workers >= 1.")
    trajectories = len(host_pairs)
    host_draws, vector_draws = np.asarray(host_draws, dtype=float), np.asarray(vector_draws, dtype=float)
    if len(vector_pairs) != trajectories or host_draws.shape != (trajectories, 5) or vector_draws.shape != (trajectories, 5):
        raise ValueError("Host/vector trees and parameter draws must align exactly.")
    _validate_parameters(host_draws)
    _validate_parameters(vector_draws)
    if index_offspring_bounds is not None:
        lower, upper = index_offspring_bounds
        if lower < 0 or upper < lower or int(lower) != lower or int(upper) != upper:
            raise ValueError("Index bounds must be ordered nonnegative integers.")
        index_offspring_bounds = int(lower), int(upper)
    nodes, weights = leggauss(quadrature_order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    evaluation_horizons = np.append(horizons, np.inf)
    log_host = np.empty((trajectories, len(evaluation_horizons)))
    log_all = np.empty(trajectories)
    def evaluate_draw(draw_index):
        times, counts = _history(host_pairs[draw_index], vector_pairs[draw_index], cutoff)
        host_params = parameters_at(host_draws[draw_index], times[0], host_changes)
        vector_params = parameters_at(vector_draws[draw_index], times[1], vector_changes)
        _validate_parameters(host_params)
        _validate_parameters(vector_params)
        host_tail = gammaincc(host_params[:, 3], (cutoff - times[0]) / host_params[:, 4])
        vector_tail = gammaincc(vector_params[:, 3], (cutoff - times[1]) / vector_params[:, 4])
        host_risk = _host_risk(
            times[0], host_params, vector_draws[draw_index], vector_changes,
            cutoff, evaluation_horizons, nodes, weights,
        )
        vector_risk = np.maximum(vector_tail[:, None] - gammaincc(
            vector_params[:, 3, None],
            (cutoff - times[1][:, None] + evaluation_horizons) / vector_params[:, 4, None],
        ), 0)
        host_components = _parent_components(
            times[0], counts[0], host_params, host_risk, cutoff, index_offspring_bounds,
        ) + _parent_components(times[1], counts[1], vector_params, vector_risk, cutoff)
        all_components = _parent_components(
            times[0], counts[0], host_params, host_tail[:, None], cutoff, index_offspring_bounds,
        )[0] + _parent_components(times[1], counts[1], vector_params, vector_tail[:, None], cutoff)[0]
        return host_components, all_components

    if workers == 1:
        for result_index, components in enumerate(map(evaluate_draw, range(trajectories))):
            log_host[result_index], log_all[result_index] = components
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result_index, components in enumerate(executor.map(evaluate_draw, range(trajectories))):
                log_host[result_index], log_all[result_index] = components
    denominator = logsumexp(log_host[:, :-1], axis=0)
    supported = np.isfinite(denominator)
    results = {}
    for name, numerator in (("host", log_host[:, -1]), ("all", log_all)):
        curve = np.full(horizons.shape, np.nan)
        curve[supported] = np.exp(logsumexp(numerator) - denominator[supported])
        results[name] = np.clip(curve, 0, 1)
    return results
