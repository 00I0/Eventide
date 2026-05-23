from __future__ import annotations

"""Time-path (multi-snapshot) metric plots for the Epuyén hantavirus observations.

This script mirrors the structure of :mod:`python.plot_timepath_metrics_hu`, but:

- uses the Epuyén (Argentina) 2018–2019 host-side observation series
- uses English labels throughout
- optionally applies a time-varying *control* scenario starting on a given date

The plots show how posterior summaries evolve within each snapshot as we
condition on "no post-snapshot infections yet" for an increasing quiet period.
"""

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
    ParameterChangePoint,
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

COLORS = {
    "re": "#0072B2",
    "growth": "#D55E00",
    "alpha_theta": "#009E73",
    "p0": "#CC79A7",
    "secondary": "#888888",
    "primary": "#0072B2",
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


def _empirical_fadeout_curves(
        infection_times_2d: Sequence[Sequence[float] | np.ndarray],
        *,
        t_star: float,
        T_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scenario-aware fade-out probability curves from simulated paths.

    The Simulator already incorporates the time-varying Scenario (controls starting
    on Jan 1, 2019). The RB utilities in :mod:`python.on_the_fly` assume
    time-homogeneous parameters; for a time-varying r(t) we instead compute the
    fade-out curves directly from accepted simulated infection time paths.

    Definitions (using times measured in days since simulation start):
    - Let t_star be the snapshot time.
    - For each accepted trajectory, define:
        * last_post = max(t - t_star) over infection times t (clipped at 0)
        * first_post = min(t - t_star) over infection times t > t_star (or +inf)

    Returned curves on T_grid:
    - p_uncond_mean(T) = P(last_post <= T)
      (probability the outbreak has ended by t_star + T, within the simulated horizon)
    - p_cond_mean(T) = P(no post-snapshot infections at all | first_post > T)
      (probability the outbreak has ended given no infections for T days)

    Also returns per-trajectory matrices for p_uncond and p_cond (the latter is
    a 0/1 indicator repeated across T, useful for diagnostics).
    """

    T_grid = np.asarray(T_grid, dtype=float)
    t_star = float(t_star)
    nT = int(T_grid.size)

    if not infection_times_2d:
        empty = np.empty((0, nT), dtype=float)
        nan = np.full(nT, np.nan, dtype=float)
        return nan, nan, empty, empty

    # Convert to arrays and compute last/first post-snapshot infection offsets.
    last_post_list: List[float] = []
    first_post_list: List[float] = []
    extinct_now_list: List[bool] = []

    for traj in infection_times_2d:
        ts = np.sort(np.asarray(traj, dtype=float))
        if ts.size == 0:
            last_post_list.append(0.0)
            first_post_list.append(np.inf)
            extinct_now_list.append(True)
            continue

        post = ts[ts > t_star]
        if post.size:
            first_post = float(post[0] - t_star)
            last_post = float(post[-1] - t_star)
            extinct_now = False
        else:
            first_post = np.inf
            last_post = 0.0
            extinct_now = True

        first_post_list.append(first_post)
        last_post_list.append(max(0.0, last_post))
        extinct_now_list.append(extinct_now)

    first_post_arr = np.asarray(first_post_list, dtype=float)
    last_post_arr = np.asarray(last_post_list, dtype=float)
    extinct_now = np.asarray(extinct_now_list, dtype=bool)

    # Unconditional: ended by T.
    p_uncond_draws = (last_post_arr[:, None] <= T_grid[None, :]).astype(float)
    p_uncond_mean = p_uncond_draws.mean(axis=0) if p_uncond_draws.size else np.full(nT, np.nan)

    # Conditional: ended given quiet for T days (i.e., first_post > T).
    p_cond_mean = np.full(nT, np.nan, dtype=float)
    for j, T in enumerate(T_grid):
        quiet = first_post_arr > float(T)
        denom = int(np.count_nonzero(quiet))
        if denom <= 0:
            p_cond_mean[j] = np.nan
        else:
            p_cond_mean[j] = float(np.count_nonzero(extinct_now & quiet)) / float(denom)

    # A simple per-trajectory "conditional" diagnostic curve: 1 if extinct immediately after t_star.
    p_cond_draws = np.repeat(extinct_now.astype(float)[:, None], nT, axis=1)
    return p_uncond_mean, p_cond_mean, p_uncond_draws, p_cond_draws


def set_english_style() -> None:
    """A journal-friendly matplotlib style (English version)."""

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


def _format_english_date(x: float, pos: int) -> str:
    dt = mdates.num2date(x)
    months = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    return f"{months[dt.month]} {dt.day:02d}"


def _apply_common_date_axis(
        axes: Sequence[plt.Axes],
        start_date: datetime,
        final_day: float,
        axis_start_date: Optional[datetime] = None,
) -> None:
    x_min = axis_start_date if axis_start_date is not None else start_date
    x_max = start_date + timedelta(days=float(final_day))
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(_format_english_date))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_xlim(x_min, x_max)


def _derive_secondary_save_path(save_path: Optional[Path], suffix: str) -> Optional[Path]:
    if save_path is None:
        return None
    save_path = Path(save_path)
    return save_path.with_name(f"{save_path.stem}_{suffix}{save_path.suffix}")


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
        res
        for res in results
        if start_date + timedelta(days=float(getattr(res, "t_star", 0.0))) >= axis_start_date
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


def _get_first_post_infection_times(res: Any) -> np.ndarray:
    t_star = float(res.t_star)
    first_posts = []
    for traj in res.infection_times_2d:
        ts = np.sort(np.asarray(traj, dtype=float))
        future = ts[ts > t_star]
        first_posts.append(float(future[0] - t_star) if future.size else np.inf)
    return np.asarray(first_posts, dtype=float)


def _retain_mask(first_post: np.ndarray, T: float, step_index: int) -> np.ndarray:
    # At T=0 we keep all accepted trajectories; thereafter keep only trajectories
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
    p0_R0 = (k / (k + R0 + 1e-12)) ** k

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        growth = (np.power(Re, 1.0 / alpha) - 1.0) / theta

    return {
        "R0": R0,
        "Re": Re,
        "growth": growth,
        "alpha_theta": alpha_theta,
        "p0": p0,
        "p0_R0": p0_R0,
    }


def _build_parameters(
        *,
        priors: Mapping[str, Tuple[float, float]],
        requirements: Sequence[str],
) -> Parameters:
    pars = Parameters(
        R0=priors["R0"],
        k=priors["k"],
        r=priors["r"],
        alpha=priors["alpha"],
        theta=priors["theta"],
    )
    for req in requirements:
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
    # Match optimize_acceptance_windows.py: always include IndexOffspringCriterion(3,7)
    # and use the acceptance-window inequalities built from snapshot-specific kwargs.
    criteria = [IndexOffspringCriterion(3, 7)] + build_acceptance_inequalities(
        obs_points=pts,
        simulation_start=sim_start,
        **_cast_accept_kwargs(builder_kwargs),
    )
    return criteria, sim_start


def _scenario_controls_starting(
        *,
        simulation_start: datetime,
        control_start: datetime,
) -> Scenario:
    """Scenario where r=1.0 until control_start, then restore the drawn r."""

    cps: List[ParameterChangePoint] = []
    if control_start > simulation_start:
        cps.append(ParameterChangePoint("r", simulation_start, "1.0"))
        cps.append(ParameterChangePoint("r", control_start))
    return Scenario(cps)


def run_snapshot_with_counts(
        *,
        m: int,
        obs_points: Sequence[Tuple[datetime, int]],
        pars: Parameters,
        builder_kwargs: Mapping[str, Any],
        scenario: Scenario,
        control_start: Optional[datetime] = None,
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
        scenario=scenario,
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
    t_star = float((active_set.collection_date - sim.start_date).days)
    accepted_count = int(sim.accepted or 0)
    processed_count = int(sim.processed or 0)

    # ------------------------------------------------------------
    # Rao–Blackwellized fade-out / extinction probability curves
    # ------------------------------------------------------------
    #
    # We use the RB formulas from python.on_the_fly, but pass `control_day` so
    # the effective reproduction number used for each parent depends on that
    # parent's infection time:
    #   - parents infected before control_day: Reff = R0
    #   - parents infected on/after control_day: Reff = r*R0
    #
    # NOTE: `control_day` is measured in days since simulation start.
    control_day: Optional[float] = None
    if control_start is not None:
        control_day = float((control_start - sim.start_date).days)

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
            control_day=control_day,
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
            control_day=control_day,
        )

        if g_cond_quiet.size:
            p_cond_mean = g_cond_inf.mean() / g_cond_quiet.mean(axis=0)
            p_cond_draws = rb_draws_cond_from_components(g_cond_inf, g_cond_quiet)
        else:
            p_cond_mean = np.full_like(T_grid, np.nan)
            p_cond_draws = np.empty((0, T_grid.size))

    if m < len(obs_points):
        delta = (obs_points[m][0] - obs_points[m - 1][0]).total_seconds() / 86400.0
        next_T = float(delta)
    else:
        next_T = None

    n_obs = int(sum(y for _, y in obs_points[:m]))

    return TimepathSnapshotResult(
        m=m,
        t_star=t_star,
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
        *,
        obs_points: Sequence[Tuple[datetime, int]],
        pars: Parameters,
        builder_kwargs_by_m: Mapping[int, Mapping[str, Any]],
        snapshots: Sequence[int],
        scenario: Scenario,
        control_start: Optional[datetime] = None,
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
            scenario=scenario,
            control_start=control_start,
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


def plot_timepath_re_growth_alpha_theta_p0_en(
        *,
        results: Sequence[Any],
        start_date: datetime,
        step: float = 0.1,
        final_horizon_days: float = 20.0,
        perc_bands: Sequence[float] = (0.95, 0.8, 0.5),
        figsize: Tuple[float, float] = (10.8, 9.2),
        ylims: Optional[Mapping[str, Tuple[float, float]]] = None,
        axis_start_date: Optional[datetime] = None,
        save_path: Optional[Path] = None,
) -> None:
    r"""Plot a 3×2 time-series figure for selected parameter-derived metrics.

    Panels:
    - $R_0$
    - $rR_0$ (effective reproduction number)
    - Euler–Lotka growth rate $g$
    - $\alpha\theta$ (generation time)
    - $\left(\frac{k}{k+rR_0}\right)^k$
    - $\left(\frac{k}{k+R_0}\right)^k$
    """

    set_english_style()
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

    metric_names = ("R0", "Re", "growth", "alpha_theta", "p0", "p0_R0")
    metrics = {name: {"mid": [], "bands": {p: {"lo": [], "hi": []} for p in perc_bands}} for name in metric_names}
    all_x: List[np.ndarray] = []

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

        segment_stats = {name: {"mid": [], "bands": {p: {"lo": [], "hi": []} for p in perc_bands}} for name in
                         metric_names}

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
                    alpha_tail = (1.0 - float(p)) / 2.0
                    lo_pct = float(np.percentile(vals, alpha_tail * 100.0))
                    hi_pct = float(np.percentile(vals, (1.0 - alpha_tail) * 100.0))
                    segment_stats[name]["bands"][p]["lo"].append(lo_pct)
                    segment_stats[name]["bands"][p]["hi"].append(hi_pct)

        all_x.append(start_day + T_eval)
        for name in metric_names:
            metrics[name]["mid"].append(np.asarray(segment_stats[name]["mid"], dtype=float))
            for p in perc_bands:
                metrics[name]["bands"][p]["lo"].append(np.asarray(segment_stats[name]["bands"][p]["lo"], dtype=float))
                metrics[name]["bands"][p]["hi"].append(np.asarray(segment_stats[name]["bands"][p]["hi"], dtype=float))

    if not all_x:
        return

    jump_locs = [start_date + timedelta(days=float(d)) for d in jump_days]
    X_num = np.concatenate(all_x)
    X_dates = [start_date + timedelta(days=float(x)) for x in X_num]
    X_dates_arr = np.asarray(X_dates, dtype=object)
    final_day = segment_end_days[-1]

    sorted_bands = sorted(perc_bands, reverse=True)

    panel_specs = [
        ("R0", r"Basic reproduction number $R_0$", COLORS["primary"], None),
        ("Re", r"Effective reproduction number $rR_0$", COLORS["re"], 1.0),
        ("growth", r"Euler–Lotka growth rate $g$", COLORS["growth"], 0.0),
        ("alpha_theta", r"Generation time $\alpha\theta$", COLORS["alpha_theta"], None),
        ("p0", r"$\left(\frac{k}{k+rR_0}\right)^k$", COLORS["p0"], None),
        ("p0_R0", r"$\left(\frac{k}{k+R_0}\right)^k$", COLORS["p0"], None),
    ]

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    axes_flat = np.ravel(axes)

    for ax, (name, ylabel, color, ref_value) in zip(axes_flat, panel_specs):
        mid = np.concatenate(metrics[name]["mid"]) if metrics[name]["mid"] else np.array([])

        if ref_value is not None:
            ax.axhline(ref_value, color="#eeeeee", linestyle=":", linewidth=0.8, zorder=8, alpha=0.8)

        finite_band_values: List[float] = []
        for i, p in enumerate(sorted_bands):
            lo = np.concatenate(metrics[name]["bands"][p]["lo"]) if metrics[name]["bands"][p]["lo"] else np.array([])
            hi = np.concatenate(metrics[name]["bands"][p]["hi"]) if metrics[name]["bands"][p]["hi"] else np.array([])
            if lo.size == 0 or hi.size == 0:
                continue

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

        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color="#eeeeee", linestyle="-", which="major", zorder=0, alpha=0.5, lw=0.8)
        ax.xaxis.grid(False)

        if name == "growth":
            ax.set_ylim(-0.2, 0.2)
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


def plot_timepath_acceptance_ratio_en(
        *,
        results: Sequence[Any],
        start_date: datetime,
        step: float = 0.1,
        final_horizon_days: float = 20.0,
        figsize: Tuple[float, float] = (7.0, 3.8),
        axis_start_date: Optional[datetime] = None,
        save_path: Optional[Path] = None,
) -> None:
    """Plot the overall acceptance ratio through time (log y-axis)."""

    set_english_style()
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
    jump_locs = [start_date + timedelta(days=float(jump_day)) for jump_day in jump_days]

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
            seg_ratio.append(np.nan if denom <= 0.0 else float(np.count_nonzero(keep)) / denom)

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

    ax.set_ylabel("Acceptance ratio")
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


def plot_rb_online_right_pane_en(
        *,
        results: Sequence[Any],
        start_date: datetime,
        ylim: Tuple[float, float] = (0.0, 1.05),
        figsize: Tuple[float, float] = (5.04, 3.1248),
        final_horizon_days: float = 20.0,
        end_date: Optional[datetime] = None,
        axis_start_date: Optional[datetime] = None,
        save_path: Optional[Path] = None,
) -> None:
    """Single-pane plot for the conditional fade-out probability curves."""

    set_english_style()
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

    if end_date is not None:
        max_days = (end_date - start_date).total_seconds() / 86400.0
    else:
        max_days = max(
            float(res.t_star) + (float(res.next_T) if res.next_T is not None else final_horizon_days) for res in
            ordered_results
        )

    for i, res in enumerate(ordered_results):
        y_vals = np.asarray(res.p_cond_mean, dtype=float)
        snapshot_day = float(res.t_star)
        x_vals = np.asarray(snapshot_day + res.T_grid, dtype=float)

        if y_vals.size == 0 or x_vals.size == 0:
            continue

        visible = x_vals <= max_days
        if not np.any(visible):
            continue

        x_visible = x_vals[visible]
        y_visible = y_vals[visible]

        if res.next_T is not None:
            x_event = snapshot_day + float(res.next_T)
            x_pre = x_visible[x_visible < x_event]
            y_pre = y_visible[x_visible < x_event]

            if x_event <= max_days:
                y_event = float(np.interp(res.next_T, res.T_grid, y_vals))
                date_event = start_date + timedelta(days=x_event)
                x_pre = np.append(x_pre, x_event)
                y_pre = np.append(y_pre, y_event)

            if x_pre.size >= 2:
                dates_pre = [start_date + timedelta(days=float(x)) for x in x_pre]
                ax.plot(dates_pre, y_pre, color=colors[i], lw=1.8, linestyle="-", alpha=0.9, zorder=10 - i)

            if x_event <= max_days:
                x_post = x_visible[x_visible > x_event]
                y_post = y_visible[x_visible > x_event]
                x_post = np.concatenate(([x_event], x_post))
                y_post = np.concatenate(([y_event], y_post))
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
            if x_visible.size >= 2:
                dates = [start_date + timedelta(days=float(x)) for x in x_visible]
                ax.plot(dates, y_visible, color=colors[i], lw=1.8, alpha=0.9, zorder=10 - i)

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

    ax.set_ylabel("A járvány végének valószínűsége")
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


def main() -> None:
    # ============================================================
    # Observations (Epuyén, Argentina; host-side incidence)
    # ============================================================
    host_observations_epuyen: List[Tuple[datetime, int]] = [
        (datetime(2018, 11, 2), 1),
        (datetime(2018, 11, 22), 2),
        (datetime(2018, 11, 28), 3),
        (datetime(2018, 12, 5), 4),
        (datetime(2018, 12, 11), 5),
        (datetime(2018, 12, 18), 6),
        (datetime(2018, 12, 26), 5),
        (datetime(2019, 1, 3), 4),
        (datetime(2019, 1, 10), 2),
        (datetime(2019, 1, 18), 1),
        (datetime(2019, 1, 25), 1),
    ]
    host_observations_epuyen1: List[Tuple[datetime, int]] = [
        (datetime(2026, 4, 6), 1),  # Case 1 (Index Case)
        (datetime(2026, 4, 24), 2),  # Case 2 and Case 3
        (datetime(2026, 4, 27), 1),  # Case 6 (Crew/Guide)
        (datetime(2026, 4, 28), 2),  # Case 4 and Case 8
        (datetime(2026, 4, 30), 1),  # Case 5 (Ship Doctor)
        (datetime(2026, 5, 1), 1),  # Case 7 (Disembarked early, onset in Switzerland)
        (datetime(2026, 5, 11), 1),  # Case 10
        (datetime(2026, 5, 12), 1),  # Case 9
        (datetime(2026, 5, 14), 1),  # Case 11
        (datetime(2026, 5, 17), 1),  # Case 12
    ]

    obs_points = sorted(host_observations_epuyen, key=lambda x: x[0])
    start_date = obs_points[0][0]

    # ============================================================
    # Priors and acceptance criteria (edit as needed)
    # ============================================================
    priors: Dict[str, Tuple[float, float]] = {
        "R0": (0.0, 7.5),
        "k": (0.2, 40.0),
        "r": (0.01, 0.99),
        "alpha": (0.01, 30.0),
        "theta": (0.01, 20.0),
    }

    requirements: List[str] = [
        "R0 * r < 3",
        "alpha * theta > 15",
        "alpha * theta < 60",
        "(k / (k + R0)) ^ k > 0.25",
        "(k / (k + R0 * r)) ^ k > 0.35",
        "((R0 * r) ^ (1 / alpha) - 1) / theta < 0.05",
    ]

    # Snapshot-specific acceptance-window construction kwargs (from the previous optimizer output).
    # These are passed into build_acceptance_inequalities for each prefix m.
    builder_kwargs_by_m: Dict[int, Dict[str, Any]] = {
        1: {
            "sigma_days": 3.876986677334013,
            "beta": 0.8536521235132486,
            "neighbor_weight": 0.640374898332219,
            "grid_step_days": 0.10599357413613672,
            "min_seg_days": 0.7285707803290452,
            "kmax": 2,
            "baseline_p": 0.1843105102607828,
            "alpha": 0.39988138399181644,
            "h_max": 0.4231284739255994,
            "eps_share": 0.09178407307432239,
            "include_gap_windows": True,
            "include_union_windows": False,
            "max_unions_to_keep": 0,
            "gap_scale": 0.8016084430512683,
            "include_global_total": False,
        },
        2: {
            "sigma_days": 1.8445791914323304,
            "beta": 0.8816601601104272,
            "neighbor_weight": 1.2102294660275816,
            "grid_step_days": 0.43546831487101323,
            "min_seg_days": 3.1987225675922053,
            "kmax": 6,
            "baseline_p": 0.28376740693924385,
            "alpha": 0.09799519411618896,
            "h_max": 0.8185162430576272,
            "eps_share": 0.06512535606974085,
            "include_gap_windows": True,
            "include_union_windows": True,
            "max_unions_to_keep": 0,
            "gap_scale": 0.10186318097771685,
            "include_global_total": True,
        },
        3: {
            "sigma_days": 7.81309491615046,
            "beta": 0.9578135534988547,
            "neighbor_weight": 1.8279768796531257,
            "grid_step_days": 0.16046667759248917,
            "min_seg_days": 4.966058815762482,
            "kmax": 2,
            "baseline_p": 0.06865373864095785,
            "alpha": 0.010554750277145204,
            "h_max": 0.1367819342609044,
            "eps_share": 0.07117480127990024,
            "include_gap_windows": False,
            "include_union_windows": True,
            "max_unions_to_keep": 6,
            "gap_scale": 0.8705002095780249,
            "include_global_total": False,
        },
        4: {
            "sigma_days": 7.230142046451505,
            "beta": 0.5000307517236018,
            "neighbor_weight": 2.034541151376431,
            "grid_step_days": 0.31096993492248015,
            "min_seg_days": 4.678181054040574,
            "kmax": 2,
            "baseline_p": 0.1763631488167721,
            "alpha": 0.04982099599959151,
            "h_max": 0.10022453373717369,
            "eps_share": 0.0005548428848050443,
            "include_gap_windows": False,
            "include_union_windows": True,
            "max_unions_to_keep": 3,
            "gap_scale": 0.8919467963913615,
            "include_global_total": True,
        },
        5: {
            "sigma_days": 5.896779157840167,
            "beta": 0.8894045810357124,
            "neighbor_weight": 2.8366104994965755,
            "grid_step_days": 0.7448871585883066,
            "min_seg_days": 1.609828560076077,
            "kmax": 3,
            "baseline_p": 0.048728197753616934,
            "alpha": 0.09354187205294318,
            "h_max": 0.28584425154799786,
            "eps_share": 0.03407536361980045,
            "include_gap_windows": True,
            "include_union_windows": True,
            "max_unions_to_keep": 3,
            "gap_scale": 0.11185380746783324,
            "include_global_total": False,
        },
        6: {
            "sigma_days": 6.106501326327924,
            "beta": 0.5256891135165026,
            "neighbor_weight": 1.3672741944106894,
            "grid_step_days": 0.7274105187796782,
            "min_seg_days": 1.14535435555878,
            "kmax": 3,
            "baseline_p": 0.06682068388674232,
            "alpha": 0.013870087712047747,
            "h_max": 0.249889783048155,
            "eps_share": 0.032635876707872234,
            "include_gap_windows": True,
            "include_union_windows": True,
            "max_unions_to_keep": 6,
            "gap_scale": 0.8892829872491416,
            "include_global_total": True,
        },
        7: {
            "sigma_days": 7.116888769886009,
            "beta": 0.7289981655377629,
            "neighbor_weight": 0.4603085602769692,
            "grid_step_days": 0.3689880093118396,
            "min_seg_days": 4.7130775585359865,
            "kmax": 2,
            "baseline_p": 0.17018271443712157,
            "alpha": 0.10615833482828443,
            "h_max": 0.5023157446544768,
            "eps_share": 0.0886387104281635,
            "include_gap_windows": True,
            "include_union_windows": True,
            "max_unions_to_keep": 5,
            "gap_scale": 0.7515885712089204,
            "include_global_total": True,
        },
        8: {
            "sigma_days": 7.618148264912112,
            "beta": 0.6639606520497713,
            "neighbor_weight": 0.7983152483309041,
            "grid_step_days": 0.30276960264247754,
            "min_seg_days": 4.030559854773017,
            "kmax": 2,
            "baseline_p": 0.2859517471439936,
            "alpha": 0.06965407744112657,
            "h_max": 0.552419107409047,
            "eps_share": 0.022263962523991456,
            "include_gap_windows": True,
            "include_union_windows": True,
            "max_unions_to_keep": 3,
            "gap_scale": 0.3256759640980279,
            "include_global_total": False,
        },
        9: {
            "sigma_days": 4.830514211689886,
            "beta": 0.8498009781272136,
            "neighbor_weight": 0.2341760492678853,
            "grid_step_days": 0.47352688279069455,
            "min_seg_days": 2.1015806529196013,
            "kmax": 3,
            "baseline_p": 0.04847992329718288,
            "alpha": 0.3069766208772683,
            "h_max": 0.9704615608663967,
            "eps_share": 0.000396162152668443,
            "include_gap_windows": True,
            "include_union_windows": True,
            "max_unions_to_keep": 4,
            "gap_scale": 0.6344008608356021,
            "include_global_total": True,
        },
        10: {
            "sigma_days": 6.353369405193468,
            "beta": 0.7434956147057368,
            "neighbor_weight": 0.1281014391336382,
            "grid_step_days": 0.32655880245142344,
            "min_seg_days": 4.985489866806982,
            "kmax": 4,
            "baseline_p": 0.06579324087694745,
            "alpha": 0.29314687776231696,
            "h_max": 0.1532076599122092,
            "eps_share": 0.08214825792562591,
            "include_gap_windows": False,
            "include_union_windows": False,
            "max_unions_to_keep": 3,
            "gap_scale": 0.12278422433810095,
            "include_global_total": True,
        },
        11: {
            "sigma_days": 7.422983570150037,
            "beta": 0.989262527201425,
            "neighbor_weight": 2.14274118060158,
            "grid_step_days": 0.12442459103366947,
            "min_seg_days": 0.5000857976034738,
            "kmax": 4,
            "baseline_p": 0.024567889698469254,
            "alpha": 0.3979365013171724,
            "h_max": 0.8513333723779887,
            "eps_share": 0.0682294608729182,
            "include_gap_windows": False,
            "include_union_windows": False,
            "max_unions_to_keep": 5,
            "gap_scale": 0.8999821695409332,
            "include_global_total": True,
        },
    }
    builder_kwargs_by_m1: Dict[int, Dict[str, Any]] = {
        1: {
            "sigma_days": 1.1253077611883389,
            "beta": 0.9269668561471294,
            "neighbor_weight": 1.4025293253011095,
            "grid_step_days": 0.7009467884711393,
            "min_seg_days": 4.908118420398451,
            "kmax": 7,
            "baseline_p": 0.1839835231600347,
            "alpha": 0.033285974325294315,
            "h_max": 0.42878229034936355,
            "eps_share": 0.015041145162612917,
            "include_gap_windows": True,
            "include_union_windows": False,
            "max_unions_to_keep": 1,
            "gap_scale": 0.19042846636521432,
            "include_global_total": False,
        },
        2: {
            "sigma_days": 6.081300565314073,
            "beta": 0.974942697570574,
            "neighbor_weight": 0.9466095135488098,
            "grid_step_days": 0.28600913096519653,
            "min_seg_days": 3.1742404903073096,
            "kmax": 3,
            "baseline_p": 0.26647151466594277,
            "alpha": 0.07405030570102389,
            "h_max": 0.9475362071960769,
            "eps_share": 0.037857011907687935,
            "include_gap_windows": True,
            "include_union_windows": False,
            "max_unions_to_keep": 1,
            "gap_scale": 0.11569130163797606,
            "include_global_total": True,
        },
        3: {
            "sigma_days": 0.7653403148896346,
            "beta": 0.5987485961653242,
            "neighbor_weight": 1.6500754790932635,
            "grid_step_days": 0.3650895193940761,
            "min_seg_days": 1.4109653642254854,
            "kmax": 5,
            "baseline_p": 0.014789432422911065,
            "alpha": 0.39840034061498847,
            "h_max": 0.1308972657711816,
            "eps_share": 0.039999923460640276,
            "include_gap_windows": True,
            "include_union_windows": True,
            "max_unions_to_keep": 4,
            "gap_scale": 0.8082759755680968,
            "include_global_total": False,
        },
        4: {
            "sigma_days": 7.797135826075775,
            "beta": 0.5771000378758345,
            "neighbor_weight": 0.5951076064929567,
            "grid_step_days": 0.10286027476969772,
            "min_seg_days": 4.695535042493828,
            "kmax": 3,
            "baseline_p": 0.010344940919398766,
            "alpha": 0.29430663371538485,
            "h_max": 0.12129504877838976,
            "eps_share": 0.013240765166649696,
            "include_gap_windows": True,
            "include_union_windows": True,
            "max_unions_to_keep": 1,
            "gap_scale": 0.2933330295753821,
            "include_global_total": False,
        },
        5: {
            "sigma_days": 2.997890886490486,
            "beta": 0.500857012796471,
            "neighbor_weight": 0.11172096580404678,
            "grid_step_days": 0.10008039098466114,
            "min_seg_days": 4.771015548674616,
            "kmax": 7,
            "baseline_p": 0.016478174548807012,
            "alpha": 0.36847233840090193,
            "h_max": 0.10032616559324287,
            "eps_share": 0.09063341233668298,
            "include_gap_windows": False,
            "include_union_windows": False,
            "max_unions_to_keep": 6,
            "gap_scale": 0.2651847132294505,
            "include_global_total": True,
        },
        6: {
            "sigma_days": 7.991074504598734,
            "beta": 0.653809222878009,
            "neighbor_weight": 0.10930557297885897,
            "grid_step_days": 0.1211312366026706,
            "min_seg_days": 0.5118707082604929,
            "kmax": 8,
            "baseline_p": 0.1343024808675321,
            "alpha": 0.36999651398652245,
            "h_max": 0.3445638188588871,
            "eps_share": 0.06668312808060688,
            "include_gap_windows": False,
            "include_union_windows": True,
            "max_unions_to_keep": 1,
            "gap_scale": 0.39381732090998234,
            "include_global_total": False,
        },
        7: {
            "sigma_days": 5.873296919450529,
            "beta": 0.5824845115470392,
            "neighbor_weight": 0.291796990168927,
            "grid_step_days": 0.12331531726891505,
            "min_seg_days": 0.5823519413501396,
            "kmax": 6,
            "baseline_p": 0.12562279663847944,
            "alpha": 0.23691382509886752,
            "h_max": 0.1025574741102243,
            "eps_share": 0.015025082676865954,
            "include_gap_windows": True,
            "include_union_windows": True,
            "max_unions_to_keep": 1,
            "gap_scale": 0.10426792241931677,
            "include_global_total": True,
        },
        8: {
            "sigma_days": 4.030865507521659,
            "beta": 0.9307912574927261,
            "neighbor_weight": 1.3376802408030994,
            "grid_step_days": 0.3287166256320232,
            "min_seg_days": 2.4266296495384783,
            "kmax": 7,
            "baseline_p": 0.010261854691918907,
            "alpha": 0.37510170167668483,
            "h_max": 0.2182158786222464,
            "eps_share": 0.06200349156077785,
            "include_gap_windows": False,
            "include_union_windows": True,
            "max_unions_to_keep": 4,
            "gap_scale": 0.43705266298039136,
            "include_global_total": True,
        },
        9: {
            "sigma_days": 2.2224233697676192,
            "beta": 0.8563752728749588,
            "neighbor_weight": 2.1403570387301323,
            "grid_step_days": 0.6463033889253157,
            "min_seg_days": 0.6700350548491563,
            "kmax": 7,
            "baseline_p": 0.01710636813601208,
            "alpha": 0.3025610165973527,
            "h_max": 0.6279154110638231,
            "eps_share": 0.09950652720696691,
            "include_gap_windows": False,
            "include_union_windows": True,
            "max_unions_to_keep": 6,
            "gap_scale": 0.8389161575264481,
            "include_global_total": False,
        },
        10: {
            "sigma_days": 2.2224233697676192,
            "beta": 0.8563752728749588,
            "neighbor_weight": 2.1403570387301323,
            "grid_step_days": 0.6463033889253157,
            "min_seg_days": 0.6700350548491563,
            "kmax": 7,
            "baseline_p": 0.01710636813601208,
            "alpha": 0.3025610165973527,
            "h_max": 0.6279154110638231,
            "eps_share": 0.09950652720696691,
            "include_gap_windows": False,
            "include_union_windows": True,
            "max_unions_to_keep": 6,
            "gap_scale": 0.8389161575264481,
            "include_global_total": False,
        },
    }

    # ============================================================
    # Scenario: controls apply from day 1, so there is no parameter change point
    # ============================================================
    control_start = datetime(2019, 1, 1)
    scenario = _scenario_controls_starting(simulation_start=start_date, control_start=control_start)
    # scenario = Scenario([])

    # ============================================================
    # Run snapshots
    # ============================================================
    pars = _build_parameters(priors=priors, requirements=requirements)
    snapshots = tuple(range(1, len(obs_points) + 1))

    # Sanity: ensure we have kwargs for every snapshot we plan to run.
    missing = sorted(set(snapshots) - set(builder_kwargs_by_m.keys()))
    if missing:
        raise ValueError(f"Missing builder kwargs for snapshots: {missing}")

    results = run_all_snapshots_with_counts(
        obs_points=obs_points,
        pars=pars,
        builder_kwargs_by_m=builder_kwargs_by_m,
        snapshots=snapshots,
        scenario=scenario,
        control_start=control_start,
        num_trajectories=1_000_000_000_000_000,
        chunk_size=50_000,
        T_run=(obs_points[-1][0] - obs_points[0][0]).days + 40,
        max_cases=4000,
        max_workers=13,
        T_grid=np.arange(0.0, 80.0 + 1e-9, 0.25),
        h=0.1,
        H_pad=10.0,
        min_required=1_000,
    )

    # ============================================================
    # Plots
    # ============================================================
    plot_rb_online_right_pane_en(
        results=results,
        start_date=start_date,
        final_horizon_days=50.0,
        axis_start_date=datetime(2018, 11, 28),
    )
    #
    plot_timepath_re_growth_alpha_theta_p0_en(
        results=results,
        start_date=start_date,
        step=0.25,
        final_horizon_days=30.0,
        perc_bands=(0.95, 0.8, 0.5),
    )

    plot_timepath_acceptance_ratio_en(
        results=results,
        start_date=start_date,
        step=0.25,
        final_horizon_days=30.0,
    )


if __name__ == "__main__":
    main()
