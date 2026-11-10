"""Conditional extinction under a Gamma–Poisson branching model.

Unlike the legacy estimator in on_the_fly.py, this calculator retains the
index-offspring restriction used during ABC acceptance. Histories and all
acceptance windows must stop at t_star; observations are assumed complete
within that history. Predictions condition on no new infections after t_star.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.special import gammaincc, gammaln, logsumexp, xlogy


def _log_cond_components(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star,
        *, index_offspring_bounds: Optional[Tuple[int, int]] = (2, 5),
):
    trajectories = len(infection_times)
    parameters = [np.asarray(values, dtype=float) for values in (R0s, rs, ks, alphas, thetas)]
    if len(stopped_pairs) != trajectories or any(values.shape != (trajectories,) for values in parameters):
        raise ValueError("Histories, transmission trees, and parameter draws must align exactly.")
    horizons = np.asarray(T_grid, dtype=float)
    if horizons.ndim != 1 or np.any(np.isnan(horizons)) or np.any(horizons < 0):
        raise ValueError("T_grid must be a one-dimensional array of nonnegative horizons.")
    if not np.isfinite(t_star) or t_star < 0:
        raise ValueError("t_star must be finite and nonnegative.")
    if any(np.any(~np.isfinite(values)) for values in parameters):
        raise ValueError("Parameter draws must be finite.")
    R0s, rs, ks, alphas, thetas = parameters
    if any(np.any(values <= 0) for values in (R0s, ks, alphas, thetas)) or np.any(rs < 0):
        raise ValueError("R0, k, alpha, and theta must be positive; r must be nonnegative.")
    if index_offspring_bounds is not None:
        lower, upper = index_offspring_bounds
        if lower < 0 or upper < lower or int(lower) != lower or int(upper) != upper:
            raise ValueError("Index-offspring bounds must be ordered nonnegative integers.")

    log_infinite = np.zeros(trajectories)
    log_quiet = np.zeros((trajectories, horizons.size))
    for draw_index in range(trajectories):
        times = np.asarray(infection_times[draw_index], dtype=float)
        if times.ndim != 1 or np.any(~np.isfinite(times)) or np.any(times < 0):
            raise ValueError("Infection times must be finite and nonnegative.")
        parents = times[times <= t_star]
        if np.count_nonzero(parents == 0.0) != 1 or np.unique(parents).size != parents.size:
            raise ValueError("Each history must contain one index at zero and distinct parent times.")
        pairs = np.asarray(stopped_pairs[draw_index], dtype=float)
        if pairs.size == 0:
            pairs = np.empty((0, 2))
        if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(~np.isfinite(pairs)):
            raise ValueError("Transmission trees must contain finite parent-child pairs.")
        pre_pairs = pairs[(pairs[:, 1] <= t_star) & (pairs[:, 1] > pairs[:, 0])]
        parent_times, child_counts = np.unique(pre_pairs[:, 0], return_counts=True)
        counts = dict(zip(parent_times, child_counts))
        for parent in parents:
            offspring_before = int(counts.get(parent, 0))
            reproduction = R0s[draw_index] * (1.0 if parent == 0.0 else rs[draw_index])
            if reproduction == 0.0:
                if offspring_before:
                    raise ValueError("A zero-reproduction parent cannot have observed offspring.")
                continue
            age = t_star - parent
            shape = ks[draw_index] + offspring_before
            tail = gammaincc(alphas[draw_index], age / thetas[draw_index])
            rate = ks[draw_index] / reproduction + (1.0 - tail)
            remaining_tail = gammaincc(alphas[draw_index], (age + horizons) / thetas[draw_index])
            quiet_mass = np.maximum(tail - remaining_tail, 0.0)
            if parent != 0.0 or index_offspring_bounds is None:
                log_infinite[draw_index] -= shape * np.log1p(tail / rate)
                log_quiet[draw_index] -= shape * np.log1p(quiet_mass / rate)
                continue

            minimum_future = max(0, int(lower) - offspring_before)
            maximum_future = int(upper) - offspring_before
            if maximum_future < minimum_future:
                raise ValueError("History violates the index-offspring restriction.")
            future_counts = np.arange(minimum_future, maximum_future + 1)
            log_weights = (
                gammaln(shape + future_counts) - gammaln(shape) - gammaln(future_counts + 1)
                - shape * np.log1p(tail / rate)
                + xlogy(future_counts, tail / (rate + tail))
            )
            normalizer = logsumexp(log_weights)
            if not np.isfinite(normalizer):
                raise ValueError("Index-offspring restriction has zero numerical probability for this history.")
            log_weights -= normalizer
            log_infinite[draw_index] += log_weights[0] if minimum_future == 0 else -np.inf
            survival = remaining_tail / tail if tail > 0 else np.ones_like(horizons)
            log_quiet[draw_index] += logsumexp(
                log_weights[:, None] + xlogy(future_counts[:, None], np.clip(survival, 0.0, 1.0)), axis=0,
            )
    return log_infinite, log_quiet


def rb_cond_components_post(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star,
        *, index_offspring_bounds: Optional[Tuple[int, int]] = (2, 5),
):
    """Return per-draw extinction and silence components using exact Gamma tails."""
    log_infinite, log_quiet = _log_cond_components(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star,
        index_offspring_bounds=index_offspring_bounds,
    )
    return np.exp(log_infinite), np.exp(log_quiet)


def conditional_extinction_probability(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star,
        *, index_offspring_bounds: Optional[Tuple[int, int]] = (2, 5),
):
    """Return E[P(no future infections | history)] / E[P(silence through T | history)].

    The ratio of posterior means accounts for reweighting by the quiet period.
    Undefined conditioning events, including exhausted mandatory-offspring
    support at an infinite horizon, return NaN rather than false certainty.
    """
    log_infinite, log_quiet = _log_cond_components(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star,
        index_offspring_bounds=index_offspring_bounds,
    )
    if not len(log_infinite):
        return np.full(np.asarray(T_grid).shape, np.nan)
    numerator = logsumexp(log_infinite)
    denominator = logsumexp(log_quiet, axis=0)
    probabilities = np.full(denominator.shape, np.nan)
    supported = np.isfinite(denominator)
    probabilities[supported] = np.exp(numerator - denominator[supported])
    return np.clip(probabilities, 0.0, 1.0)
