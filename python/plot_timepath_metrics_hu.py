from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from python.eventide import (
    ActiveSetSizeCollector,
    DrawCollector,
    IndexOffspringCriterion,
    InfectionTimeCollector,
    Parameters,
    Scenario,
    Simulator,
)
from python.on_the_fly import (
    rb_cond_components_post,
    rb_draws_cond_from_components,
    rb_draws_uncond_full_to_grid,
    rao_blackwell_uncond_over_post_full,
)
from python.optimize_acceptance_windows import build_acceptance_inequalities
from python.robustness_appendix_analysis import (
    BASELINE_PRIORS,
    BASELINE_REQUIREMENTS,
    OBS_POINTS,
    SNAPSHOT_BUILDER_KWARGS_BY_M,
)

COLORS = {
    "re": "#0072B2",
    "growth": "#D55E00",
    "alpha_theta": "#009E73",
    "p0": "#CC79A7",
    "secondary": "#888888",
    'primary': '#0072B2',  # Blue
}


@dataclass
class TimepathSnapshotResult:
    m: int
    t_star: float
    T_grid: np.ndarray
    p_uncond_mean: np.ndarray
    p_cond_mean: np.ndarray
    p_uncond_draws: np.ndarray
    p_cond_draws: np.ndarray
    draws_array: np.ndarray
    infection_times_2d: List[np.ndarray]
    n_obs: int
    next_T: Optional[float]
    stopped_pairs: List[Sequence[Tuple[float, float]]] | None
    accepted_count: int
    processed_count: int


def set_hungarian_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _get_first_post_infection_times(res: Any) -> np.ndarray:
    t_star = float(res.t_star)
    first_posts = []
    for traj in res.infection_times_2d:
        ts = np.sort(np.asarray(traj, dtype=float))
        future_infs = ts[ts > t_star]
        if future_infs.size > 0:
            first_posts.append(float(future_infs[0] - t_star))
        else:
            first_posts.append(np.inf)
    return np.asarray(first_posts, dtype=float)


def _retain_mask(first_post: np.ndarray, T: float, step_index: int) -> np.ndarray:
    # Retention logic: at T=0 keep everything, then keep only trajectories
    # with no post-snapshot infection yet.
    if step_index == 0:
        return np.ones(first_post.shape[0], dtype=bool)
    return first_post > T


def _extract_metric_values(draws: np.ndarray) -> Dict[str, np.ndarray]:
    R0 = draws[:, 0]
    k = draws[:, 1]
    r = draws[:, 2]
    alpha = draws[:, 3]
    theta = draws[:, 4]

    Re = R0 * r
    alpha_theta = alpha * theta
    p0 = (k / (k + Re + 1e-12)) ** k
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        growth = (np.power(Re, 1.0 / alpha) - 1.0) / theta

    return {
        "Re": Re,
        "growth": growth,
        "alpha_theta": alpha_theta,
        "p0": p0,
    }


def _weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    """Compute a weighted quantile for 1D arrays.

    This uses the left-continuous inverse CDF of the weighted empirical
    distribution.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or values.shape[0] != weights.shape[0]:
        raise ValueError("values and weights must be 1D arrays of the same length")
    if values.size == 0:
        return float("nan")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")

    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return float("nan")

    v = values[good]
    w = weights[good]
    order = np.argsort(v)
    v = v[order]
    w = w[order]

    cum = np.cumsum(w)
    total = float(cum[-1])
    if total <= 0.0:
        return float("nan")
    target = q * total
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def _gaussian_distance_weights(distance: np.ndarray, sigma: float) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        w = np.exp(-0.5 * (distance / sigma) ** 2)
    w[~np.isfinite(w)] = 0.0
    return w


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return float("nan")
    v = values[good]
    w = weights[good]
    s = float(np.sum(w))
    if s <= 0.0:
        return float("nan")
    return float(np.sum(v * w) / s)


def _weighted_tail_mean_band(values: np.ndarray, weights: np.ndarray, p_central: float) -> Tuple[float, float, float]:
    """Return (mu, lower, upper) using tail-means with shrinkage.

    This mirrors the *idea* of the tail-mean bands used in plot_TDK: instead of
    using quantiles directly, we compute weighted means of the lower/upper tails
    and shrink them toward the overall mean.

    This tends to look smoother than median/quantile summaries when the survivor
    set changes discretely over time.
    """
    p_central = float(p_central)
    if not (0.0 < p_central < 1.0):
        return float("nan"), float("nan"), float("nan")

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return float("nan"), float("nan"), float("nan")

    v = values[good]
    w = weights[good]
    total = float(np.sum(w))
    if total <= 0.0:
        return float("nan"), float("nan"), float("nan")

    mu = float(np.sum(v * w) / total)
    if p_central >= 0.999:
        return mu, mu, mu

    tail = (1.0 - p_central) / 2.0
    tail_mass = tail * total

    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)

    lower_mask = cw <= tail_mass
    if not np.any(lower_mask):
        lower_mask[0] = True
    upper_mask = cw >= (total - tail_mass)
    if not np.any(upper_mask):
        upper_mask[-1] = True

    w_lo = float(np.sum(w[lower_mask]))
    w_hi = float(np.sum(w[upper_mask]))
    lower_raw = float(np.sum(v[lower_mask] * w[lower_mask]) / max(w_lo, 1e-15))
    upper_raw = float(np.sum(v[upper_mask] * w[upper_mask]) / max(w_hi, 1e-15))

    beta = p_central ** 0.7
    lower = mu + beta * (lower_raw - mu)
    upper = mu + beta * (upper_raw - mu)
    return mu, float(lower), float(upper)


def _sort_results_chronologically(results: Sequence[Any]) -> List[Any]:
    indexed = list(enumerate(results))
    indexed.sort(key=lambda x: (float(getattr(x[1], "t_star", 0.0)), x[0]))
    return [res for _, res in indexed]


def _filter_results_by_axis_start_date(
        results: Sequence[Any],
        start_date: datetime,
        axis_start_date: Optional[datetime],
) -> List[Any]:
    if axis_start_date is None:
        return list(results)
    return [
        res for res in results
        if start_date + timedelta(days=float(res.t_star)) >= axis_start_date
    ]


def _get_segment_end_days(
        ordered_results: Sequence[Any],
        final_horizon_days: float,
) -> Tuple[List[float], List[float]]:
    snapshot_days = [float(res.t_star) for res in ordered_results]
    segment_end_days: List[float] = []
    jump_days: List[float] = []

    for idx, start_day in enumerate(snapshot_days):
        if idx + 1 < len(snapshot_days):
            end_day = snapshot_days[idx + 1]
            jump_days.append(end_day)
        else:
            end_day = start_day + final_horizon_days
        segment_end_days.append(end_day)

    return segment_end_days, jump_days


def _format_hungarian_date(x: float, pos: int) -> str:
    dt = mdates.num2date(x)
    month_names = {
        1: "jan.",
        2: "febr.",
        3: "márc.",
        4: "ápr.",
        5: "máj.",
        6: "jún.",
        7: "júl.",
        8: "aug.",
        9: "szept.",
        10: "okt.",
        11: "nov.",
        12: "dec.",
    }
    return f"{month_names[dt.month]} {dt.day:02d}"


def _apply_common_date_axis(
        axes: Sequence[plt.Axes],
        start_date: datetime,
        final_day: float,
        axis_start_date: Optional[datetime] = None,
) -> None:
    x_min = axis_start_date if axis_start_date is not None else start_date
    x_max = start_date + timedelta(days=float(final_day))
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(_format_hungarian_date))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.tick_params(axis="x", labelrotation=30)
        # ax.set_xlabel("Dátum")
        ax.set_xlim(x_min, x_max)


def _derive_secondary_save_path(save_path: Optional[Path], suffix: str) -> Optional[Path]:
    if save_path is None:
        return None
    save_path = Path(save_path)
    return save_path.with_name(f"{save_path.stem}_{suffix}{save_path.suffix}")


def plot_timepath_re_growth_alpha_theta_p0_hu(
        results: Sequence[Any],
        start_date: datetime,
        *,
        step: float = 0.1,
        final_horizon_days: float = 20.0,
        perc_bands: Sequence[float] = (0.95, 0.8, 0.5),
        figsize: Tuple[float, float] = (10.8, 6.6),
        ylims: Optional[Mapping[str, Tuple[float, float]]] = None,
        axis_start_date: Optional[datetime] = None,
        save_path: Optional[Path] = None,
) -> None:
    """
    Plot a 2x2 time-series figure for:
    - Effective reproduction number (Re)
    - Euler-Lotka growth rate
    - alpha*theta
    - Probability of 0 new infections (p0)

    At each time point, the posterior is filtered by the
    "still no new post-snapshot infection" condition.
    """
    set_hungarian_style()
    if not results:
        return

    for p in perc_bands:
        if not 0.0 < p < 1.0:
            raise ValueError("Band levels must be in the (0, 1) interval.")

    ordered_results = _filter_results_by_axis_start_date(
        _sort_results_chronologically(results),
        start_date,
        axis_start_date,
    )
    if not ordered_results:
        return
    segment_end_days, jump_days = _get_segment_end_days(ordered_results, final_horizon_days)

    metric_names = ("Re", "growth", "alpha_theta", "p0")
    metrics = {
        name: {"mid": [], "bands": {p: {"lo": [], "hi": []} for p in perc_bands}}
        for name in metric_names
    }
    all_x: List[np.ndarray] = []
    jump_locs: List[datetime] = []

    for res, end_day in zip(ordered_results, segment_end_days):
        start_day = float(res.t_star)
        T_end = end_day - start_day

        if T_end <= 0:
            continue

        n_steps = max(1, int(np.floor(T_end / step)) + 1)
        T_eval = np.linspace(0.0, T_end, n_steps)

        draws = np.asarray(res.draws_array, dtype=float)
        first_post = _get_first_post_infection_times(res)

        n_draws = min(len(draws), len(first_post))
        if n_draws == 0:
            continue
        draws = draws[:n_draws]
        first_post = first_post[:n_draws]

        values_by_metric = _extract_metric_values(draws)

        segment_stats = {
            name: {"mid": [], "bands": {p: {"lo": [], "hi": []} for p in perc_bands}}
            for name in metric_names
        }

        for j, T in enumerate(T_eval):
            keep = _retain_mask(first_post, float(T), j)

            for name in metric_names:
                vals = values_by_metric[name][keep]
                vals = vals[np.isfinite(vals)]

                if vals.size == 0:
                    segment_stats[name]["mid"].append(np.nan)
                    for p in perc_bands:
                        segment_stats[name]["bands"][p]["lo"].append(np.nan)
                        segment_stats[name]["bands"][p]["hi"].append(np.nan)
                    continue

                segment_stats[name]["mid"].append(float(np.median(vals)))
                for p in perc_bands:
                    alpha_tail = (1.0 - p) / 2.0
                    lo_pct = float(np.percentile(vals, alpha_tail * 100.0))
                    hi_pct = float(np.percentile(vals, (1.0 - alpha_tail) * 100.0))
                    segment_stats[name]["bands"][p]["lo"].append(lo_pct)
                    segment_stats[name]["bands"][p]["hi"].append(hi_pct)

        all_x.append(start_day + T_eval)
        for name in metric_names:
            metrics[name]["mid"].append(np.asarray(segment_stats[name]["mid"], dtype=float))
            for p in perc_bands:
                metrics[name]["bands"][p]["lo"].append(
                    np.asarray(segment_stats[name]["bands"][p]["lo"], dtype=float)
                )
                metrics[name]["bands"][p]["hi"].append(
                    np.asarray(segment_stats[name]["bands"][p]["hi"], dtype=float)
                )

    for jump_day in jump_days:
        jump_locs.append(start_date + timedelta(days=jump_day))

    if not all_x:
        return

    X_num = np.concatenate(all_x)
    X_dates = [start_date + timedelta(days=float(x)) for x in X_num]
    X_dates_arr = np.asarray(X_dates, dtype=object)
    final_day = segment_end_days[-1]

    sorted_bands = sorted(perc_bands, reverse=True)
    label_idx = max(0, len(X_dates) - 20)

    def _nearest_finite_idx(mask: np.ndarray, target_idx: int) -> Optional[int]:
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            return None
        return int(idxs[np.argmin(np.abs(idxs - target_idx))])

    panel_specs = [
        ("Re", r"Effektív reprodukciós szám $rR_0$", COLORS["re"], 1.0),
        ("growth", r"Euler-Lotka növekedési ráta $g$", COLORS["growth"], 0.0),
        ("alpha_theta", r"Generációk közti idő $\alpha\theta$", COLORS["alpha_theta"], None),
        ("p0", r"0 új fertőzés valószínűsége $p_0$", COLORS["p0"], None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes_flat = np.ravel(axes)

    for ax, (name, ylabel, color, ref_value) in zip(axes_flat, panel_specs):
        mid = np.concatenate(metrics[name]["mid"]) if metrics[name]["mid"] else np.array([])

        if ref_value is not None:
            ax.axhline(ref_value, color="#eeeeee", linestyle=":", linewidth=0.8, zorder=8, alpha=0.8)

        finite_band_values: List[float] = []
        for i, p in enumerate(sorted_bands):
            lo = np.concatenate(metrics[name]["bands"][p]["lo"])
            hi = np.concatenate(metrics[name]["bands"][p]["hi"])
            good = np.isfinite(lo) & np.isfinite(hi)
            if not np.any(good):
                continue

            finite_band_values.extend([float(np.nanmin(lo[good])), float(np.nanmax(hi[good]))])
            alpha_vis = 0.14 + 0.07 * i
            ax.fill_between(X_dates, lo, hi, color=color, alpha=alpha_vis, linewidth=0)

            band_label_idx = _nearest_finite_idx(good, label_idx)
            if band_label_idx is not None:
                label_y = hi[band_label_idx]
                span = max(float(np.nanmax(hi[good]) - np.nanmin(lo[good])), 1e-9)
                # ax.text(
                #     X_dates[band_label_idx],
                #     label_y + 0.015 * span,
                #     f"{int(p * 100)}% KI",
                #     color=color,
                #     alpha=min(1.0, 1.1 * alpha_vis),
                #     fontsize=7,
                #     va="bottom",
                #     ha="center",
                #     zorder=100,
                # )

        good_mid = np.isfinite(mid)
        ax.plot(X_dates_arr[good_mid], mid[good_mid], color=color, lw=1.5, zorder=10)

        if np.any(good_mid):
            mid_label_idx = _nearest_finite_idx(good_mid, label_idx)
            if mid_label_idx is not None:
                median_y = mid[mid_label_idx]
                span = max(float(np.nanmax(mid[good_mid]) - np.nanmin(mid[good_mid])), 1e-9)
                # ax.text(
                #     X_dates[mid_label_idx],
                #     median_y + 0.02 * span,
                #     "medián",
                #     color=color,
                #     fontsize=7,
                #     va="bottom",
                #     ha="center",
                # )

        for j_day in jump_locs:
            ax.axvline(j_day, color=COLORS["secondary"], linestyle="-", lw=0.8, zorder=5)

        ax.set_title("")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color="#eeeeee", linestyle="-", which="major", zorder=0, alpha=0.5, lw=0.8)
        ax.xaxis.grid(False)

        if name == "growth":
            ax.set_ylim(-0.2, 0.1)
        elif ylims and name in ylims:
            ax.set_ylim(*ylims[name])
        elif finite_band_values:
            lo = min(finite_band_values)
            hi = max(finite_band_values)
            if ref_value is not None:
                lo = min(lo, ref_value)
                hi = max(hi, ref_value)
            pad = 0.08 * max(hi - lo, 1e-9)
            ax.set_ylim(lo - pad, hi + pad)

    _apply_common_date_axis(axes_flat, start_date, final_day, axis_start_date=axis_start_date)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_timepath_weighted_re_growth_alpha_theta_p0_hu(
        results: Sequence[Any],
        start_date: datetime,
        *,
        step: float = 0.1,
        final_horizon_days: float = 20.0,
        sigma: float = 2.0,
        perc_bands: Sequence[float] = (0.95, 0.8, 0.5),
        figsize: Tuple[float, float] = (10.8, 6.6),
        ylims: Optional[Mapping[str, Tuple[float, float]]] = None,
        axis_start_date: Optional[datetime] = None,
        save_path: Optional[Path] = None,
) -> None:
    """Weighted 2x2 time-series figure for Re, growth, alpha*theta, p0.

    This is analogous to :func:`plot_timepath_re_growth_alpha_theta_p0_hu`, but
    at each time-since-snapshot T it:

    1) keeps the same surviving trajectories (no post-snapshot infection by T)
       i.e. first_post > T
    2) assigns each surviving trajectory a *distance-to-T* weight using a
       Gaussian kernel with width ``sigma`` (in days):

           w_i(T) = exp(-0.5 * ((first_post_i - T)/sigma)^2)

       (computed on finite first_post only; if all survivors are right-censored
       then it falls back to uniform weights).

    The plotted curves are weighted means, with tail-mean shrinkage bands.
    """
    set_hungarian_style()
    if not results:
        return

    for p in perc_bands:
        if not 0.0 < float(p) < 1.0:
            raise ValueError("Band levels must be in the (0, 1) interval.")
    if float(sigma) <= 0.0:
        raise ValueError("sigma must be positive")

    ordered_results = _filter_results_by_axis_start_date(
        _sort_results_chronologically(results),
        start_date,
        axis_start_date,
    )
    if not ordered_results:
        return

    segment_end_days, jump_days = _get_segment_end_days(ordered_results, final_horizon_days)

    metric_names = ("Re", "growth", "alpha_theta", "p0")
    metrics = {
        name: {"mid": [], "bands": {p: {"lo": [], "hi": []} for p in perc_bands}}
        for name in metric_names
    }
    all_x: List[np.ndarray] = []
    jump_locs: List[datetime] = []

    for res, end_day in zip(ordered_results, segment_end_days):
        start_day = float(res.t_star)
        T_end = end_day - start_day
        if T_end <= 0:
            continue

        n_steps = max(1, int(np.floor(T_end / step)) + 1)
        T_eval = np.linspace(0.0, T_end, n_steps)

        draws = np.asarray(res.draws_array, dtype=float)
        first_post = _get_first_post_infection_times(res)
        n_draws = min(len(draws), len(first_post))
        if n_draws == 0:
            continue
        draws = draws[:n_draws]
        first_post = first_post[:n_draws]

        values_by_metric = _extract_metric_values(draws)
        finite_first_post = np.isfinite(first_post)

        segment_stats = {
            name: {"mid": [], "bands": {p: {"lo": [], "hi": []} for p in perc_bands}}
            for name in metric_names
        }

        for j, T in enumerate(T_eval):
            keep = _retain_mask(first_post, float(T), j)

            # Use distance weights based on first_post - T for survivors with finite first_post.
            keep_finite = keep & finite_first_post
            if np.any(keep_finite):
                dist = first_post[keep_finite] - float(T)
                w_finite = _gaussian_distance_weights(dist, sigma=sigma)
            else:
                w_finite = np.asarray([], dtype=float)

            for name in metric_names:
                if np.any(keep_finite):
                    vals = values_by_metric[name][keep_finite]
                    w = w_finite
                else:
                    # If all survivors are censored (first_post=inf), fall back to uniform weights.
                    vals = values_by_metric[name][keep]
                    w = np.ones(vals.shape[0], dtype=float)

                good = np.isfinite(vals)
                if not np.any(good):
                    segment_stats[name]["mid"].append(np.nan)
                    for p in perc_bands:
                        segment_stats[name]["bands"][p]["lo"].append(np.nan)
                        segment_stats[name]["bands"][p]["hi"].append(np.nan)
                    continue

                vals = vals[good]
                w = w[good]
                if vals.size == 0:
                    segment_stats[name]["mid"].append(np.nan)
                    for p in perc_bands:
                        segment_stats[name]["bands"][p]["lo"].append(np.nan)
                        segment_stats[name]["bands"][p]["hi"].append(np.nan)
                    continue
                if float(np.sum(w)) <= 0.0:
                    w = np.ones_like(w)

                mu = _weighted_mean(vals, w)
                segment_stats[name]["mid"].append(mu)
                for p in perc_bands:
                    _, lo_b, hi_b = _weighted_tail_mean_band(vals, w, float(p))
                    segment_stats[name]["bands"][p]["lo"].append(lo_b)
                    segment_stats[name]["bands"][p]["hi"].append(hi_b)

        all_x.append(start_day + T_eval)
        for name in metric_names:
            metrics[name]["mid"].append(np.asarray(segment_stats[name]["mid"], dtype=float))
            for p in perc_bands:
                metrics[name]["bands"][p]["lo"].append(
                    np.asarray(segment_stats[name]["bands"][p]["lo"], dtype=float)
                )
                metrics[name]["bands"][p]["hi"].append(
                    np.asarray(segment_stats[name]["bands"][p]["hi"], dtype=float)
                )

    for jump_day in jump_days:
        jump_locs.append(start_date + timedelta(days=jump_day))

    if not all_x:
        return

    X_num = np.concatenate(all_x)
    X_dates = [start_date + timedelta(days=float(x)) for x in X_num]
    X_dates_arr = np.asarray(X_dates, dtype=object)
    final_day = segment_end_days[-1]

    sorted_bands = sorted(perc_bands, reverse=True)

    panel_specs = [
        ("Re", r"Effektív reprodukciós szám $rR_0$ (súlyozott átlag)", COLORS["re"], 1.0),
        ("growth", r"Euler-Lotka növekedési ráta $g$ (súlyozott átlag)", COLORS["growth"], 0.0),
        ("alpha_theta", r"Generációk közti idő $\alpha\theta$ (súlyozott átlag)", COLORS["alpha_theta"], None),
        ("p0", r"0 új fertőzés valószínűsége $p_0$ (súlyozott átlag)", COLORS["p0"], None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes_flat = np.ravel(axes)

    for ax, (name, ylabel, color, ref_value) in zip(axes_flat, panel_specs):
        mid = np.concatenate(metrics[name]["mid"]) if metrics[name]["mid"] else np.array([])

        if ref_value is not None:
            ax.axhline(ref_value, color="#eeeeee", linestyle=":", linewidth=0.8, zorder=8, alpha=0.8)

        finite_band_values: List[float] = []
        for i, p in enumerate(sorted_bands):
            lo = np.concatenate(metrics[name]["bands"][p]["lo"])
            hi = np.concatenate(metrics[name]["bands"][p]["hi"])
            good = np.isfinite(lo) & np.isfinite(hi)
            if not np.any(good):
                continue

            finite_band_values.extend([float(np.nanmin(lo[good])), float(np.nanmax(hi[good]))])
            alpha_vis = 0.14 + 0.07 * i
            ax.fill_between(X_dates, lo, hi, color=color, alpha=alpha_vis, linewidth=0)

        good_mid = np.isfinite(mid)
        ax.plot(X_dates_arr[good_mid], mid[good_mid], color=color, lw=1.5, zorder=10)

        for j_day in jump_locs:
            ax.axvline(j_day, color=COLORS["secondary"], linestyle="-", lw=0.8, zorder=5)

        ax.set_title("")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color="#eeeeee", linestyle="-", which="major", zorder=0, alpha=0.5, lw=0.8)
        ax.xaxis.grid(False)

        if name == "growth":
            ax.set_ylim(-0.2, 0.1)
        elif ylims and name in ylims:
            ax.set_ylim(*ylims[name])
        elif finite_band_values:
            lo_y = min(finite_band_values)
            hi_y = max(finite_band_values)
            if ref_value is not None:
                lo_y = min(lo_y, ref_value)
                hi_y = max(hi_y, ref_value)
            pad = 0.08 * max(hi_y - lo_y, 1e-9)
            ax.set_ylim(lo_y - pad, hi_y + pad)

    _apply_common_date_axis(axes_flat, start_date, final_day, axis_start_date=axis_start_date)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_timepath_acceptance_ratio_hu(
        results: Sequence[Any],
        start_date: datetime,
        *,
        step: float = 0.1,
        final_horizon_days: float = 20.0,
        figsize: Tuple[float, float] = (7.0, 3.8),
        axis_start_date: Optional[datetime] = None,
        save_path: Optional[Path] = None,
) -> None:
    """
    Plot the overall acceptance ratio through time, using the simulator's
    processed trajectory count as the segment-specific denominator.
    """
    set_hungarian_style()
    if not results:
        return

    ordered_results = _filter_results_by_axis_start_date(
        _sort_results_chronologically(results),
        start_date,
        axis_start_date,
    )
    if not ordered_results:
        return
    segment_end_days, jump_days = _get_segment_end_days(ordered_results, final_horizon_days)

    all_x: List[np.ndarray] = []
    all_ratio: List[np.ndarray] = []
    jump_locs = [start_date + timedelta(days=jump_day) for jump_day in jump_days]

    for res, end_day in zip(ordered_results, segment_end_days):
        start_day = float(res.t_star)
        T_end = end_day - start_day
        if T_end <= 0:
            continue

        n_steps = max(1, int(np.floor(T_end / step)) + 1)
        T_eval = np.linspace(0.0, T_end, n_steps)

        first_post = _get_first_post_infection_times(res)
        if first_post.size == 0:
            continue

        denom = float(getattr(res, "processed_count", 0))
        seg_ratio = []
        for j, T in enumerate(T_eval):
            keep = _retain_mask(first_post, float(T), j)
            if denom <= 0.0:
                seg_ratio.append(np.nan)
            else:
                seg_ratio.append(float(np.count_nonzero(keep)) / denom)

        all_x.append(start_day + T_eval)
        all_ratio.append(np.asarray(seg_ratio, dtype=float))

    if not all_x:
        return

    final_day = segment_end_days[-1]
    x_num = np.concatenate(all_x)
    x_dates = np.asarray([start_date + timedelta(days=float(x)) for x in x_num], dtype=object)
    ratios = np.concatenate(all_ratio)
    good = np.isfinite(ratios) & (ratios > 0.0)
    if not np.any(good):
        return

    pos_ratios = ratios[good]
    y_min = max(float(np.min(pos_ratios)) * 0.95, 1e-12)
    y_max = float(np.max(pos_ratios)) * 1.05

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_yscale("log")
    ax.plot(x_dates[good], ratios[good], color=COLORS["primary"], lw=1.6, zorder=10)
    ax.fill_between(
        x_dates[good],
        y_min,
        ratios[good],
        where=ratios[good] >= y_min,
        color=COLORS["primary"],
        alpha=0.15,
        linewidth=0,
    )

    for jump_day in jump_locs:
        ax.axvline(jump_day, color=COLORS["secondary"], linestyle="-", lw=0.8, zorder=5)

    ax.set_ylabel("Elfogadási arány")
    ax.set_ylim(y_min, y_max)
    ax.yaxis.grid(True, color="#eeeeee", linestyle="-", which="major", zorder=0, alpha=0.5, lw=0.8)
    ax.xaxis.grid(False)

    _apply_common_date_axis([ax], start_date, final_day, axis_start_date=axis_start_date)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_rb_online_right_pane_hu(
        results: Sequence[Any],
        start_date: datetime,
        *,
        ylim: Tuple[float, float] = (0.0, 1.05),
        figsize: Tuple[float, float] = (5.04, 3.1248),
        final_horizon_days: float = 20.0,
        axis_start_date: Optional[datetime] = None,
        save_path: Optional[Path] = None,
) -> None:
    """
    Hungarian single-pane plot for the conditional probability curves
    (adapted from plot_FMD.plot_rb_online_right_pane).
    """
    set_hungarian_style()
    if not results:
        return

    ordered_results = _filter_results_by_axis_start_date(
        _sort_results_chronologically(results),
        start_date,
        axis_start_date,
    )
    if not ordered_results:
        return

    cmap = plt.get_cmap("Blues")
    color_indices = np.linspace(0.4, 1.0, len(ordered_results))
    colors = [cmap(i) for i in color_indices]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    max_days = max(
        float(res.t_star) + (float(res.next_T) if res.next_T is not None else final_horizon_days)
        for res in ordered_results
    )

    for i, res in enumerate(ordered_results):
        y_vals = np.asarray(res.p_cond_mean, dtype=float)
        snapshot_day = float(res.t_star)
        x_vals = np.asarray(snapshot_day + res.T_grid, dtype=float)

        if y_vals.size == 0 or x_vals.size == 0:
            continue

        if res.next_T is not None:
            x_event = snapshot_day + float(res.next_T)
            y_event = float(np.interp(res.next_T, res.T_grid, y_vals))
            date_event = start_date + timedelta(days=x_event)

            # Solid curve up to segment end.
            x_pre = np.append(x_vals[x_vals < x_event], x_event)
            y_pre = np.append(y_vals[x_vals < x_event], y_event)
            dates_pre = [start_date + timedelta(days=float(x)) for x in x_pre]
            ax.plot(dates_pre, y_pre, color=colors[i], lw=1.8, linestyle="-", alpha=0.9, zorder=10 - i)

            # Dashed continuation after the segment end.
            x_post = np.concatenate(([x_event], x_vals[x_vals > x_event]))
            y_post = np.concatenate(([y_event], y_vals[x_vals > x_event]))
            if x_post.size >= 2:
                dates_post = [start_date + timedelta(days=float(x)) for x in x_post]
                ax.plot(
                    dates_post,
                    y_post,
                    color=colors[i],
                    lw=1.2,
                    linestyle=(0, (2, 0.5)),
                    dash_capstyle="butt",
                    alpha=1.0,
                    zorder=10 - i,
                )

            ax.scatter([date_event], [y_event], s=23, facecolors="white", edgecolors=colors[i], lw=1.5, zorder=20)
        else:
            dates = [start_date + timedelta(days=float(x)) for x in x_vals]
            ax.plot(dates, y_vals, color=colors[i], lw=1.8, alpha=0.9, zorder=10 - i)

    legend_lines = [plt.Line2D([0], [0], color=c, lw=2) for c in colors]
    legend_labels = [f"$N={res.n_obs}$" for res in ordered_results]
    ax.legend(
        legend_lines,
        legend_labels,
        loc="lower right",
        frameon=False,
        fontsize=8,
        title="Megfigyelt esetszám",
        title_fontsize=8,
    )

    ax.set_ylabel("A járvány lecsengésének valószínűsége")
    ax.set_ylim(ylim)
    ax.yaxis.grid(True, color="#eeeeee", linestyle="-", zorder=0)
    ax.xaxis.grid(False)

    _apply_common_date_axis([ax], start_date, max_days, axis_start_date=axis_start_date)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def _build_baseline_parameters() -> Parameters:
    pars = Parameters(
        R0=BASELINE_PRIORS["R0"],
        k=BASELINE_PRIORS["k"],
        r=BASELINE_PRIORS["r"],
        alpha=BASELINE_PRIORS["alpha"],
        theta=BASELINE_PRIORS["theta"],
    )
    for req in BASELINE_REQUIREMENTS:
        pars = pars.require(req)
    return pars


def _cast_accept_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    if "kmax" in out:
        out["kmax"] = int(round(float(out["kmax"])))
    for name in ("include_gap_windows", "include_union_windows", "include_global_total"):
        if name in out:
            out[name] = bool(round(float(out[name])))
    if "max_unions_to_keep" in out:
        out["max_unions_to_keep"] = int(round(float(out["max_unions_to_keep"])))
    if "mode" in out:
        out["mode"] = str(out["mode"])
    return out


def _build_criteria_for_prefix(
        obs_points: Sequence[Tuple[datetime, int]],
        m: int,
        **builder_kwargs: Any,
) -> Tuple[List[Any], datetime]:
    pts = obs_points[:m]
    sim_start = min(t for t, _ in pts)
    if m == 1:
        return [IndexOffspringCriterion(2, 5)], sim_start
    criteria = [IndexOffspringCriterion(2, 5)] + build_acceptance_inequalities(
        obs_points=pts,
        simulation_start=sim_start,
        **_cast_accept_kwargs(builder_kwargs),
    )
    return criteria, sim_start


def run_snapshot_with_counts(
        m: int,
        obs_points: Sequence[Tuple[datetime, int]],
        pars: Parameters,
        builder_kwargs: Mapping[str, Any],
        *,
        num_trajectories: int = 800_000,
        chunk_size: int = 100_000,
        T_run: int = 70,
        max_cases: int = 1000,
        max_workers: int = 8,
        T_grid: np.ndarray = np.arange(0.0, 70.0 + 1e-9, 1.0),
        h: float = 0.2,
        H_pad: float = 10.0,
        min_required: Optional[int] = None,
) -> TimepathSnapshotResult:
    criteria, sim_start = _build_criteria_for_prefix(obs_points, m, **builder_kwargs)

    collectors = [
        draws := DrawCollector(),
        active_set := ActiveSetSizeCollector(obs_points[m - 1][0]),
        infection_times := InfectionTimeCollector(),
    ]
    sim = Simulator(
        parameters=pars,
        sampler=pars.create_latin_hypercube_sampler(),
        start_date=sim_start,
        scenario=Scenario([]),
        criteria=criteria,
        collectors=collectors,
        num_trajectories=num_trajectories,
        chunk_size=chunk_size,
        T_run=T_run,
        max_cases=max_cases,
        max_workers=max_workers,
        min_required=min_required,
    )
    now = time.time()
    sim.run()
    print("Run", m, "took", time.time() - now, "seconds")

    draws_array = np.asarray(draws, dtype=float)
    if draws_array.ndim == 1 and draws_array.size:
        draws_array = draws_array.reshape(1, -1)
    infection_times_2d = list(infection_times.infection_times)
    stopped_pairs = active_set.active_sets
    t_star = (active_set.collection_date - sim.start_date).days
    accepted_count = int(sim.accepted or 0)
    processed_count = int(sim.processed or 0)
    print("Run", m, "accepted", len(infection_times_2d))

    if draws_array.size == 0:
        p_uncond_mean = np.full(T_grid.shape, np.nan)
        p_cond_mean = np.full(T_grid.shape, np.nan)
        p_uncond_draws = np.empty((0, T_grid.size))
        p_cond_draws = np.empty((0, T_grid.size))
    else:
        R0s, ks, rs, alphas, thetas = draws_array.T
        T_fine, p_uncond_mean_fine, g_uncond_fine = rao_blackwell_uncond_over_post_full(
            infection_times_2d,
            stopped_pairs,
            R0s,
            rs,
            ks,
            alphas,
            thetas,
            T_max=float(T_grid[-1]),
            t_star=t_star,
            h=h,
            H_pad=H_pad,
        )
        p_uncond_mean = np.interp(T_grid, T_fine, p_uncond_mean_fine)
        p_uncond_draws = rb_draws_uncond_full_to_grid(T_fine, g_uncond_fine, T_grid)

        g_cond_inf, g_cond_quiet = rb_cond_components_post(
            infection_times_2d,
            stopped_pairs,
            R0s,
            rs,
            ks,
            alphas,
            thetas,
            T_grid,
            t_star,
            h=h,
        )
        p_cond_mean = (
            g_cond_inf.mean() / g_cond_quiet.mean(axis=0)
            if g_cond_quiet.size
            else np.full_like(T_grid, np.nan)
        )
        p_cond_draws = (
            rb_draws_cond_from_components(g_cond_inf, g_cond_quiet)
            if g_cond_quiet.size
            else np.empty((0, T_grid.size))
        )

    next_T = None
    if m < len(obs_points):
        delta = (obs_points[m][0] - obs_points[m - 1][0]).total_seconds() / 86400.0
        next_T = float(delta)

    n_obs = int(sum(y for _, y in obs_points[:m]))
    return TimepathSnapshotResult(
        m=m,
        t_star=float(t_star),
        T_grid=T_grid,
        p_uncond_mean=p_uncond_mean,
        p_cond_mean=p_cond_mean,
        p_uncond_draws=p_uncond_draws,
        p_cond_draws=p_cond_draws,
        draws_array=draws_array,
        infection_times_2d=infection_times_2d,
        n_obs=n_obs,
        next_T=next_T,
        stopped_pairs=stopped_pairs,
        accepted_count=accepted_count,
        processed_count=processed_count,
    )


def run_all_snapshots_with_counts(
        obs_points: Sequence[Tuple[datetime, int]],
        pars: Parameters,
        builder_kwargs_by_m: Mapping[int, Mapping[str, Any]],
        snapshots: Sequence[int],
        *,
        num_trajectories: int = 1_000_000,
        chunk_size: int = 100_000,
        T_run: int = 70,
        max_cases: int = 1000,
        max_workers: int = 12,
        T_grid: np.ndarray = np.arange(0.0, 70.0 + 1e-9, 1.0),
        h: float = 0.2,
        H_pad: float = 10.0,
        min_required: Optional[int] = None,
) -> List[TimepathSnapshotResult]:
    results: List[TimepathSnapshotResult] = []
    for m in snapshots:
        result = run_snapshot_with_counts(
            m=m,
            obs_points=obs_points,
            pars=pars,
            builder_kwargs=builder_kwargs_by_m.get(m, {}),
            num_trajectories=num_trajectories,
            chunk_size=chunk_size,
            T_run=T_run,
            max_cases=max_cases,
            max_workers=max_workers,
            T_grid=T_grid,
            h=h,
            H_pad=H_pad,
            min_required=min_required,
        )
        results.append(result)
    return results


def _get_sorted_obs_points() -> List[Tuple[datetime, int]]:
    return sorted(OBS_POINTS, key=lambda item: item[0])


def run_baseline_timepath_results() -> List[TimepathSnapshotResult]:
    pars = _build_baseline_parameters()
    obs_points = _get_sorted_obs_points()
    # snapshots = tuple(range(3, len(obs_points) + 1))
    snapshots = tuple(range(1, len(obs_points) + 1))
    results = run_all_snapshots_with_counts(
        obs_points=obs_points,
        pars=pars,
        builder_kwargs_by_m={
            m: dict(SNAPSHOT_BUILDER_KWARGS_BY_M[m])
            for m in snapshots
            if m in SNAPSHOT_BUILDER_KWARGS_BY_M
        },
        snapshots=snapshots,
        num_trajectories=100_000_000_000,
        chunk_size=500,
        T_run=(obs_points[-1][0] - obs_points[0][0]).days + 40,
        max_cases=4000,
        max_workers=13,
        T_grid=np.arange(0.0, 80.0 + 1e-9, 0.25),
        h=0.1,
        H_pad=10.0,
        min_required=100_000,
    )
    return list(results)


def run_baseline_driver(
        *,
        save_path: Optional[Path] = None,
        step: float = 0.1,
        final_horizon_days: float = 30.0,
        perc_bands: Sequence[float] = (0.95, 0.8, 0.5),
) -> None:
    results = run_baseline_timepath_results()
    obs_dates = [dt for dt, _ in _get_sorted_obs_points()]
    start_date = obs_dates[0]

    acceptance_save_path = _derive_secondary_save_path(save_path, "acceptance_ratio")
    rb_right_pane_save_path = _derive_secondary_save_path(save_path, "rb_right_pane")

    plot_timepath_re_growth_alpha_theta_p0_hu(
        results=results,
        start_date=start_date,
        step=step,
        final_horizon_days=final_horizon_days,
        perc_bands=perc_bands,
        axis_start_date=obs_dates[1],
        # save_path=save_path,
    )
    plot_timepath_acceptance_ratio_hu(
        results=results,
        start_date=start_date,
        step=step,
        final_horizon_days=final_horizon_days,
        axis_start_date=obs_dates[0],
        # save_path=acceptance_save_path,
    )
    plot_rb_online_right_pane_hu(
        results=results,
        start_date=start_date,
        final_horizon_days=final_horizon_days,
        axis_start_date=obs_dates[2],
        # save_path=rb_right_pane_save_path,
    )

    plot_timepath_weighted_re_growth_alpha_theta_p0_hu(
        results=results,
        start_date=start_date,
        step=step,
        final_horizon_days=final_horizon_days,
        sigma=2.0,  # smoothing width in days
        perc_bands=perc_bands,
    )


if __name__ == "__main__":
    run_baseline_driver()
