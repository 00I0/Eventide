from __future__ import annotations

"""English-language counterpart of `plot_TDK.py`.

This module is a close replicate of `python/plot_TDK.py`, but with all plot labels
and user-facing text translated to English. The underlying simulation and plotting
logic is intentionally kept the same.

Notes
-----
- The heavy lifting (simulation and plotting) is shared with the rest of the
  repository. This script focuses on wiring up one *final snapshot* run.
- For reproducibility/editability, the priors, acceptance requirements, and the
  observation series are defined in :func:`main`.
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

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
from python.plot_FMD import (
    COLORS,
    _cum_matrix_from_times,
    _format_tick,
    _is_too_close,
    _kde_smart,
    set_journal_style,
)
from python.optimize_acceptance_windows import build_acceptance_inequalities
from python.robustness_appendix_analysis import (
    CHUNK_SIZE,
    H,
    H_PAD,
    MAX_CASES,
    MAX_WORKERS,
    NUM_TRAJECTORIES,
    T_GRID,
    T_RUN,
)

MIN_REQUIRED = 3_000


def _cast_accept_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce acceptance-window kwargs to the types expected by the builder."""
    out = dict(kwargs)
    if "kmax" in out:
        out["kmax"] = int(round(float(out["kmax"])))
    if "max_unions_to_keep" in out:
        out["max_unions_to_keep"] = int(round(float(out["max_unions_to_keep"])))
    for name in ("include_gap_windows", "include_union_windows", "include_global_total"):
        if name in out:
            out[name] = bool(out[name])
    return out


@dataclass(slots=True)
class SnapshotResultLite:
    """Minimal snapshot result container for plotting in this script."""

    m: int
    t_star: float
    draws_array: np.ndarray  # (n_accept, 5) [R0,k,r,alpha,theta]
    infection_times_2d: List[np.ndarray]
    stopped_pairs: List[Sequence[Tuple[float, float]]] | None


def _scenario_controls_starting(
        *,
        simulation_start: datetime,
        control_start: datetime,
) -> Scenario:
    """Return a Scenario where r=1.0 until control_start, then restore drawn r."""
    cps: List[ParameterChangePoint] = []
    # Ensure r=1.0 initially (no control)
    cps.append(ParameterChangePoint("r", simulation_start, "1.0"))
    # Restore the sampled r at the control start date (controls start)
    if control_start > simulation_start:
        cps.append(ParameterChangePoint("r", control_start))
    return Scenario(cps)


def _run_single_snapshot(
        *,
        obs_points: Sequence[Tuple[datetime, int]],
        pars: Parameters,
        builder_kwargs: Mapping[str, Any],
        scenario: Scenario,
        num_trajectories: int,
        chunk_size: int,
        T_run: int,
        max_cases: int,
        max_workers: int,
        min_required: Optional[int],
) -> SnapshotResultLite:
    """Run one final snapshot (m=len(obs_points)) with an explicit Scenario."""

    obs_sorted = _get_sorted_obs_points(obs_points)
    m = len(obs_sorted)
    if m < 3:
        raise ValueError("This workflow expects at least 3 observation points.")

    pts = obs_sorted[:m]
    sim_start = min(t for t, _ in pts)

    # Criteria are the same as in presentationplots.run_snapshot
    # Match optimize_acceptance_windows.py / plot_timepath_metrics_hanta.py
    crit = [IndexOffspringCriterion(2, 7)] + build_acceptance_inequalities(
        obs_points=pts,
        simulation_start=sim_start,
        **_cast_accept_kwargs(builder_kwargs),
    )

    collectors = [
        draws := DrawCollector(),
        active_set := ActiveSetSizeCollector(obs_sorted[m - 1][0]),  # t_star = t_m
        itimes := InfectionTimeCollector(),
    ]

    sampler = pars.create_latin_hypercube_sampler()
    sim = Simulator(
        parameters=pars,
        sampler=sampler,
        start_date=min(t for t, _ in obs_sorted),
        scenario=scenario,
        criteria=crit,
        collectors=collectors,
        num_trajectories=num_trajectories,
        chunk_size=chunk_size,
        T_run=T_run,
        max_cases=max_cases,
        max_workers=max_workers,
        min_required=min_required,
    )
    sim.run()

    infection_times_2d = [np.sort(np.asarray(t, dtype=float)) for t in itimes.infection_times]
    t_star = float((active_set.collection_date - sim.start_date).days)

    return SnapshotResultLite(
        m=m,
        t_star=t_star,
        draws_array=np.asarray(draws),
        infection_times_2d=infection_times_2d,
        stopped_pairs=active_set.active_sets,
    )


def _format_english_date(x: float, pos: int) -> str:
    dt = mdates.num2date(x)
    month_names = {
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
    return f"{month_names[dt.month]} {dt.day:02d}"


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


def _get_sorted_obs_points(obs_points: Sequence[Tuple[datetime, int]]) -> List[Tuple[datetime, int]]:
    return sorted(list(obs_points), key=lambda x: x[0])


def run_baseline_final_snapshot_results(
        *,
        obs_points: Sequence[Tuple[datetime, int]],
        pars: Parameters,
        builder_kwargs: Mapping[str, Any],
        control_start_date: datetime,
        min_required: int = MIN_REQUIRED,
) -> List[Any]:
    obs_sorted = _get_sorted_obs_points(obs_points)
    scenario = _scenario_controls_starting(
        simulation_start=obs_sorted[0][0],
        control_start=control_start_date,
    )
    res = _run_single_snapshot(
        obs_points=obs_sorted,
        pars=pars,
        builder_kwargs=builder_kwargs,
        scenario=Scenario([]),
        num_trajectories=1_000_000_000_000_000,
        chunk_size=50_000,
        T_run=(obs_sorted[-1][0] - obs_sorted[0][0]).days + 10,
        max_cases=4000,
        max_workers=13,
        min_required=min_required,
    )
    return [res]


def _extract_first_gen_times(
        stopped_pairs_for_traj: Sequence[Tuple[float, float]],
        eps: float = 0.01,
) -> np.ndarray:
    arr = np.asarray(stopped_pairs_for_traj, dtype=float).reshape(-1, 2)
    if arr.size == 0:
        return np.empty(0, dtype=float)
    mask = (arr[:, 0] < eps) & (arr[:, 1] > eps)
    children = arr[mask, 1]
    return np.sort(children)


def _clone_parameters_with_fixed_r(pars: Parameters, r_value: float = 1.0) -> Parameters:
    p = Parameters(
        R0=pars.R0_range,
        k=pars.k_range,
        r=(r_value, r_value),
        alpha=pars.alpha_range,
        theta=pars.theta_range,
    )
    for expr in pars.validators:
        p.require(expr)
    return p


def _run_uncontrolled_pool(
        *,
        pars: Parameters,
        start_date: datetime,
        num_trajectories: int,
        chunk_size: int,
        T_run: int,
        max_cases: int,
        max_workers: int,
        min_required: Optional[int] = None,
) -> List[np.ndarray]:
    no_ctrl = _clone_parameters_with_fixed_r(pars, r_value=1.0)
    collectors = [itimes := InfectionTimeCollector()]

    sim = Simulator(
        parameters=no_ctrl,
        sampler=no_ctrl.create_latin_hypercube_sampler(),
        start_date=start_date,
        scenario=Scenario([]),
        criteria=[],
        collectors=collectors,
        num_trajectories=num_trajectories,
        chunk_size=chunk_size,
        T_run=T_run,
        max_cases=max_cases,
        max_workers=max_workers,
        min_required=min_required,
    )
    sim.run()
    return [np.sort(np.asarray(t, dtype=float)) for t in itimes.infection_times]


def _assemble_first_gen_uncontrolled(
        controlled_stopped_pairs: List[Sequence[Tuple[float, float]]],
        pool: List[np.ndarray],
        rng: np.random.Generator | None = None,
) -> List[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    pool_size = len(pool)
    result: List[np.ndarray] = []

    for sp in controlled_stopped_pairs:
        first_gen = _extract_first_gen_times(sp)
        parts: List[np.ndarray] = [np.array([0.0])]
        for t_child in first_gen:
            idx = rng.integers(0, pool_size)
            shifted = pool[idx] + float(t_child)
            parts.append(shifted)
        merged = np.sort(np.concatenate(parts))
        result.append(merged)

    return result


def plot_cumulative_controlled_vs_first_gen_en(
        controlled_results: Sequence[Any],
        uncontrolled_first_gen_2d: Sequence[Sequence[float]],
        start_date: datetime,
        *,
        obs_points: Optional[List[Tuple[datetime, int]]] = None,
        resolution_days: float = 0.25,
        perc_bands: Sequence[float] = (0.8, 0.5),
        u_horizon_days: float = 69.0,
        figsize: Tuple[float, float] = (5.04, 3.1248),
        y_limit_from_controlled_only: bool = True,
) -> None:
    set_journal_style()
    if not controlled_results:
        return

    controlled = controlled_results[-1]
    grid_numeric = np.arange(0, u_horizon_days + 1e-9, resolution_days)
    grid_dates = [start_date + timedelta(days=float(x)) for x in grid_numeric]
    horizon_end = start_date + timedelta(days=float(u_horizon_days))

    controlled_matrix = _cum_matrix_from_times(controlled.infection_times_2d, grid_numeric)
    controlled_mean = controlled_matrix.mean(axis=0)
    unc_matrix = _cum_matrix_from_times(uncontrolled_first_gen_2d, grid_numeric)
    unc_mean = unc_matrix.mean(axis=0)

    def _tail_mean_band(matrix: np.ndarray, p_central: float) -> Tuple[np.ndarray, np.ndarray]:
        mu = matrix.mean(axis=0)
        if p_central >= 0.999:
            return mu, mu
        n = matrix.shape[0]
        tail = (1.0 - p_central) / 2.0
        k = max(1, int(np.ceil(n * tail)))
        sorted_cols = np.sort(matrix, axis=0)
        lower_raw = sorted_cols[:k, :].mean(axis=0)
        upper_raw = sorted_cols[-k:, :].mean(axis=0)
        beta = p_central ** 0.7
        lower = mu + beta * (lower_raw - mu)
        upper = mu + beta * (upper_raw - mu)
        return lower, upper

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylim(bottom=1, top=40)
    controlled_top_candidates = [float(np.nanmax(controlled_mean))] if controlled_mean.size else [0.0]

    for i, p in enumerate(sorted(perc_bands, reverse=True)):
        lower, upper = _tail_mean_band(controlled_matrix, p)
        alpha_vis = 0.1 + 0.05 * i
        ax.fill_between(grid_dates, lower, upper, color=COLORS["primary"], alpha=alpha_vis, lw=0)
        controlled_top_candidates.append(float(np.nanmax(upper)))

    for i, p in enumerate(sorted(perc_bands, reverse=True)):
        lower, upper = _tail_mean_band(unc_matrix, p)
        alpha_vis = 0.1 + 0.05 * i
        ax.fill_between(grid_dates, lower, upper, color="#D62728", alpha=alpha_vis, lw=0, zorder=4)

    ax.plot(grid_dates, controlled_mean, color=COLORS["primary"], lw=1.5, zorder=5)
    ax.plot(grid_dates, unc_mean, color="#D62728", lw=1.5, zorder=6)

    def _label_along_curve(
            y_curve: np.ndarray,
            label: str,
            color: str,
            idx_anchor: int,
            x_shift_days: float = -1.0,
            fontsize: float = 8.0,
    ) -> None:
        if len(grid_dates) < 3:
            return
        i1 = int(np.clip(idx_anchor, 1, len(grid_dates) - 1))
        i0 = max(0, i1 - 4)
        if i0 == i1:
            i0 = max(0, i1 - 1)

        p0 = ax.transData.transform((mdates.date2num(grid_dates[i0]), float(y_curve[i0])))
        p1 = ax.transData.transform((mdates.date2num(grid_dates[i1]), float(y_curve[i1])))
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        angle = float(np.degrees(np.arctan2(dy, dx))) if abs(dx) + abs(dy) > 0 else 0.0

        y_span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-9)
        y_offset = 0.012 * y_span
        ax.text(
            grid_dates[i1] + timedelta(days=x_shift_days),
            float(y_curve[i1]) + y_offset,
            label,
            color=color,
            fontsize=fontsize,
            rotation=angle,
            rotation_mode="anchor",
            va="bottom",
            ha="center",
            zorder=7,
        )

    if len(grid_dates) > 10:
        _label_along_curve(
            controlled_mean,
            "Controlled",
            COLORS["primary"],
            idx_anchor=len(grid_dates) - 3.3,
            x_shift_days=-2.2,
            fontsize=8,
        )

    y_top_curr = ax.get_ylim()[1]
    visible = np.where(unc_mean <= y_top_curr - 4)[0]
    if visible.size >= 2:
        idx = int(visible[-6]) if visible.size >= 6 else int(visible[-1])
        _label_along_curve(
            unc_mean,
            "Uncontrolled",
            "#D62728",
            idx_anchor=idx,
            x_shift_days=-1.2,
            fontsize=8,
        )

    if obs_points:
        obs_sorted = sorted(obs_points, key=lambda x: x[0])
        dates, incs = zip(*obs_sorted)
        cums = np.cumsum(incs)
        ax.scatter(dates, cums, s=23, facecolors="white", edgecolors=COLORS["obs"], lw=1.0, zorder=10)
        # ax.text(dates[1], cums[1] + 1, "Observed", color=COLORS["obs"], fontsize=8, va="bottom", ha="center")

    ax.yaxis.grid(True, color="#eeeeee", linestyle="-", which="major", zorder=0, alpha=0.5, lw=0.8)
    ax.xaxis.grid(False)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_english_date))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_ylabel("Cumulative infections")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if y_limit_from_controlled_only:
        ax.set_ylim(1, 40)
        _ = max(controlled_top_candidates)
    ax.set_xlim(start_date, horizon_end)
    ax.margins(x=0.0)

    plt.tight_layout()
    plt.show()


def run_and_plot_cumulative_first_gen_hanta(
        *,
        obs_points: Sequence[Tuple[datetime, int]],
        controlled_results: Optional[Sequence[Any]] = None,
        pars: Parameters,
        builder_kwargs: Mapping[str, Any],
        control_start_date: datetime,
        num_trajectories: int = NUM_TRAJECTORIES,
        chunk_size: int = CHUNK_SIZE,
        T_run: int = T_RUN,
        max_cases: int = MAX_CASES,
        max_workers: int = MAX_WORKERS,
        T_grid: np.ndarray = T_GRID,
        h: float = H,
        H_pad: float = H_PAD,
        min_required: Optional[int] = MIN_REQUIRED,
        u_horizon_days: float = 69.0,
        pool_min_required: Optional[int] = None,
) -> None:
    obs_sorted = _get_sorted_obs_points(obs_points)

    if controlled_results is None:
        scenario = _scenario_controls_starting(
            simulation_start=obs_sorted[0][0],
            control_start=control_start_date,
        )
        controlled_results = [
            _run_single_snapshot(
                obs_points=obs_sorted,
                pars=pars,
                builder_kwargs=builder_kwargs,
                scenario=scenario,
                num_trajectories=10_000_000_000,
                chunk_size=500,
                T_run=(obs_sorted[-1][0] - obs_sorted[0][0]).days + 40,
                max_cases=4000,
                max_workers=10,
                min_required=MIN_REQUIRED,
            )
        ]

    last = list(controlled_results)[-1]
    stopped_pairs = last.stopped_pairs
    n_controlled = len(last.infection_times_2d)
    all_first_gen = [_extract_first_gen_times(sp) for sp in stopped_pairs]
    total_first_gen = sum(len(fg) for fg in all_first_gen)
    avg_first_gen = total_first_gen / max(1, n_controlled)
    print(
        f"Controlled trajectories: {n_controlled}, "
        f"total first-generation children: {total_first_gen}, "
        f"avg per trajectory: {avg_first_gen:.2f}"
    )

    effective_pool_min = pool_min_required or max(min_required or 1000, int(1.5 * total_first_gen))
    start_date = obs_sorted[0][0]
    print(f"Running uncontrolled pool (min_required={effective_pool_min}) ...")
    pool = _run_uncontrolled_pool(
        pars=pars,
        start_date=start_date,
        num_trajectories=num_trajectories,
        chunk_size=chunk_size,
        T_run=T_run,
        max_cases=max_cases,
        max_workers=max_workers,
        min_required=effective_pool_min,
    )
    print(f"Uncontrolled pool size: {len(pool)}")

    unc_first_gen_2d = _assemble_first_gen_uncontrolled(controlled_stopped_pairs=stopped_pairs, pool=pool)
    plot_cumulative_controlled_vs_first_gen_en(
        controlled_results=controlled_results,
        uncontrolled_first_gen_2d=unc_first_gen_2d,
        start_date=start_date,
        obs_points=obs_sorted,
        u_horizon_days=u_horizon_days,
    )


def plot_last_posterior(
        results: Sequence[Any],
        figsize: Tuple[float, float] = (11.25, 4.3),
        pdf_estimator: str = "scipy",
) -> None:
    set_journal_style()
    if not results:
        return
    res = results[-1]

    arr = np.asarray(res.draws_array)
    R0, k, r, alpha, theta = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    Re = R0 * r
    alpha_theta = alpha * theta
    p0Re = (k / (k + Re + 1e-12)) ** k
    p0 = (k / (k + R0 + 1e-12)) ** k

    vars_config = [
        (R0, r"$R$", (0.0, 7.5), "truncate"),
        (r, r"$r$", (0.0, 1), "truncate"),
        (alpha, r"$\alpha$", (0, 30.0), "truncate"),
        (theta, r"$\theta$", (0, 20.0), "truncate"),
        (k, r"$k$", (0, 30.0), "truncate"),
        (Re, r"$rR$", (0, 2.5), "truncate"),
        (Re / k, r"$\frac{rR}{k}$", (0, 2), "truncate"),
        # (Re / k, r"$\frac{r R}{k}$", (0, 0.4), "truncate"),
        (alpha_theta, r"$\alpha\theta$", (13, 40.0), "truncate"),
        (np.sqrt(alpha) * theta, r"$\sqrt{\alpha}\theta$", (0, 25.0), "truncate"),
        (p0Re, r"$\left(\frac{k}{k+rR}\right)^k$", (0.2, 1.0), "truncate"),
        (p0, r"$\left(\frac{k}{k+R}\right)^k$", (0.15, 0.8), "truncate"),
        (
            ((Re ** (1 / alpha) - 1) / theta),
            r"$\frac{\left(r R\right)^{\frac{1}{\alpha}} - 1}{\theta}$",
            (-0.09, 0.09),
            "truncate",
        ),
    ]

    fig, axes = plt.subplots(2, 6, figsize=figsize)
    axes = axes.flatten()

    for i, (data, title, bounds, method) in enumerate(vars_config):
        ax = axes[i]
        lo_b, hi_b = bounds
        grid = np.linspace(lo_b, hi_b, 600)
        pdf = _kde_smart(data, grid, bounds, method, estimator=pdf_estimator)
        med = np.median(data)
        q_lo = np.percentile(data, 2.5)
        q_hi = np.percentile(data, 97.5)

        ax.plot(grid, pdf, color=COLORS["primary"], lw=1.5, zorder=10)
        mask = (grid >= q_lo) & (grid <= q_hi)
        ax.fill_between(grid, 0, pdf, where=mask, color=COLORS["primary"], alpha=0.35, linewidth=0)
        med_height = np.interp(med, grid, pdf)
        ax.vlines(med, 0, med_height, color="#777", linestyle="--", lw=1, zorder=5)

        tick_locs = [lo_b, med, hi_b]
        tick_labels = [
            _format_tick(lo_b, is_edge=True),
            _format_tick(med, is_edge=False),
            _format_tick(hi_b, is_edge=True),
        ]
        ax.set_xlim(lo_b, hi_b)
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)

        trans = ax.get_xaxis_transform()
        tick_vals = [lo_b, med, hi_b]
        span = hi_b - lo_b

        if not _is_too_close(q_lo, tick_vals, span) and lo_b < q_lo < hi_b:
            ax.text(
                q_lo,
                -0.02,
                f"{q_lo:.2f}",
                color=COLORS["primary"],
                fontsize=7,
                ha="center",
                va="top",
                transform=trans,
                alpha=0.6,
            )
            ax.plot([q_lo, q_lo], [0, 0.02], color=COLORS["primary"], lw=1, transform=trans, clip_on=False, alpha=0.6)
        if not _is_too_close(q_hi, tick_vals, span) and lo_b < q_hi < hi_b:
            ax.text(
                q_hi,
                -0.02,
                f"{q_hi:.2f}",
                color=COLORS["primary"],
                fontsize=7,
                ha="center",
                va="top",
                transform=trans,
                alpha=0.6,
            )
            ax.plot([q_hi, q_hi], [0, 0.02], color=COLORS["primary"], lw=1, transform=trans, clip_on=False, alpha=0.6)

        ax.set_title(title, pad=6, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    plt.show()


def plot_last_posterior_conditioned_quiet_for_T(
        results: Sequence[Any],
        T: float,
        *,
        figsize: Tuple[float, float] = (11.25, 4.3),
        pdf_estimator: str = "scipy",
        t_star: Optional[float] = None,
) -> None:
    """Plot the last-snapshot posterior, conditioned on *no new infections* for T days.

    This is a filtered variant of :func:`plot_last_posterior`.

    We keep only trajectories for which there are no infection times in the interval
    (t_star, t_star + T]. Equivalently, if the first post-snapshot infection happens
    at time first_post (days after t_star), we keep draws with first_post > T.

    Parameters
    ----------
    results:
        Snapshot results; the last element is used.
    T:
        Quiet-period length (days) after the snapshot start (last observation).
    t_star:
        Snapshot start time in the same units as `infection_times_2d` (days since
        simulation start). If omitted, will use `results[-1].t_star`.
    """

    set_journal_style()
    if not results:
        return
    if T < 0:
        raise ValueError("T must be non-negative (days).")

    res = results[-1]
    if t_star is None:
        if not hasattr(res, "t_star"):
            raise ValueError("t_star was not provided and results[-1] has no attribute 't_star'.")
        t_star = float(getattr(res, "t_star"))
    else:
        t_star = float(t_star)

    draws = np.asarray(res.draws_array)
    infection_times_2d = list(getattr(res, "infection_times_2d", []))
    if draws.size == 0 or len(infection_times_2d) == 0:
        return

    # Align lengths defensively (some pipelines may have slight mismatches).
    n = min(draws.shape[0], len(infection_times_2d))
    draws = draws[:n]
    infection_times_2d = infection_times_2d[:n]

    first_post = []
    for traj in infection_times_2d:
        ts = np.sort(np.asarray(traj, dtype=float))
        future = ts[ts > t_star]
        first_post.append(float(future[0] - t_star) if future.size else np.inf)
    first_post_arr = np.asarray(first_post, dtype=float)
    keep = first_post_arr > float(T)

    draws_f = draws[keep]
    if draws_f.size == 0:
        return

    R0, k, r, alpha, theta = draws_f[:, 0], draws_f[:, 1], draws_f[:, 2], draws_f[:, 3], draws_f[:, 4]
    Re = R0 * r
    alpha_theta = alpha * theta
    p0 = (k / (k + Re + 1e-12)) ** k

    vars_config = [
        (R0, r"$R$", (0.0, 7.5), "truncate"),
        (r, r"$r$", (0.2, 1), "truncate"),
        (alpha, r"$\alpha$", (0, 20.0), "truncate"),
        (theta, r"$\theta$", (0, 20.0), "truncate"),
        (k, r"$k$", (0, 20.0), "truncate"),
        (Re, r"$rR$", (0, 3.0), "truncate"),
        (R0 / k, r"$\frac{R}{k}$", (0, 1.2), "truncate"),
        (Re / k, r"$\frac{r R}{k}$", (0, 0.4), "truncate"),
        (alpha_theta, r"$\alpha\theta$", (1.0, 28.0), "truncate"),
        (np.sqrt(alpha) * theta, r"$\sqrt{\alpha}\theta$", (0, 22), "truncate"),
        (p0, r"$\left(\frac{k}{k+rR}\right)^k$", (0.05, 0.95), "truncate"),
        (
            ((Re ** (1 / alpha) - 1) / theta),
            r"$\frac{\left(r R\right)^{\frac{1}{\alpha}} - 1}{\theta}$",
            (-0.15, 0.085),
            "truncate",
        ),
    ]

    fig, axes = plt.subplots(2, 6, figsize=figsize)
    axes = axes.flatten()

    for i, (data, title, bounds, method) in enumerate(vars_config):
        ax = axes[i]
        lo_b, hi_b = bounds
        grid = np.linspace(lo_b, hi_b, 600)
        pdf = _kde_smart(data, grid, bounds, method, estimator=pdf_estimator)
        med = np.median(data)
        q_lo = np.percentile(data, 2.5)
        q_hi = np.percentile(data, 97.5)

        ax.plot(grid, pdf, color=COLORS["primary"], lw=1.5, zorder=10)
        mask = (grid >= q_lo) & (grid <= q_hi)
        ax.fill_between(grid, 0, pdf, where=mask, color=COLORS["primary"], alpha=0.35, linewidth=0)
        med_height = np.interp(med, grid, pdf)
        ax.vlines(med, 0, med_height, color="#777", linestyle="--", lw=1, zorder=5)

        tick_locs = [lo_b, med, hi_b]
        tick_labels = [
            _format_tick(lo_b, is_edge=True),
            _format_tick(med, is_edge=False),
            _format_tick(hi_b, is_edge=True),
        ]
        ax.set_xlim(lo_b, hi_b)
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)

        trans = ax.get_xaxis_transform()
        tick_vals = [lo_b, med, hi_b]
        span = hi_b - lo_b

        if not _is_too_close(q_lo, tick_vals, span) and lo_b < q_lo < hi_b:
            ax.text(
                q_lo,
                -0.02,
                f"{q_lo:.2f}",
                color=COLORS["primary"],
                fontsize=7,
                ha="center",
                va="top",
                transform=trans,
                alpha=0.6,
            )
            ax.plot([q_lo, q_lo], [0, 0.02], color=COLORS["primary"], lw=1, transform=trans, clip_on=False, alpha=0.6)
        if not _is_too_close(q_hi, tick_vals, span) and lo_b < q_hi < hi_b:
            ax.text(
                q_hi,
                -0.02,
                f"{q_hi:.2f}",
                color=COLORS["primary"],
                fontsize=7,
                ha="center",
                va="top",
                transform=trans,
                alpha=0.6,
            )
            ax.plot([q_hi, q_hi], [0, 0.02], color=COLORS["primary"], lw=1, transform=trans, clip_on=False, alpha=0.6)

        ax.set_title(title, pad=6, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    plt.show()


def plot_cumulative_infections_datetime(
        results: Sequence[Any],
        start_date: datetime,
        *,
        obs_points: Optional[List[Tuple[datetime, int]]] = None,
        resolution_days: float = 0.25,
        perc_bands: Sequence[float] = (0.95, 0.8, 0.5),
        u_horizon_days: float = 55.0,
        figsize: Tuple[float, float] = (5.04, 3.1248),
) -> None:
    set_journal_style()
    if not results:
        return
    res = results[-1]

    def _tail_mean_band(matrix: np.ndarray, p_central: float) -> Tuple[np.ndarray, np.ndarray]:
        mu = matrix.mean(axis=0)
        if p_central >= 0.999:
            return mu, mu
        n = matrix.shape[0]
        t = (1.0 - p_central) / 2.0
        k = max(1, int(np.ceil(n * t)))
        sorted_cols = np.sort(matrix, axis=0)
        lower_raw = sorted_cols[:k, :].mean(axis=0)
        upper_raw = sorted_cols[-k:, :].mean(axis=0)
        beta = p_central ** 0.7
        lower = mu + beta * (lower_raw - mu)
        upper = mu + beta * (upper_raw - mu)
        return lower, upper

    grid_numeric = np.arange(0, u_horizon_days + 1e-9, resolution_days)
    grid_dates = [start_date + timedelta(days=float(x)) for x in grid_numeric]
    cum_matrix = _cum_matrix_from_times(res.infection_times_2d, grid_numeric)
    mean_curve = cum_matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    sorted_bands = sorted(perc_bands, reverse=True)
    for i, p in enumerate(sorted_bands):
        lower, upper = _tail_mean_band(cum_matrix, p)
        alpha_vis = 0.15 + (0.07 * i)
        ax.fill_between(grid_dates, lower, upper, color=COLORS["primary"], alpha=alpha_vis, lw=0)

    ax.plot(grid_dates, mean_curve, color=COLORS["primary"], lw=1.5, zorder=5)
    if len(grid_dates) > 0:
        ax.text(
            grid_dates[-10],
            mean_curve[-10] + 0.25,
            "Szimulált",
            color=COLORS["primary"],
            fontsize=8,
            va="bottom",
            ha="center",
            fontweight="normal",
        )

    if obs_points:
        obs_sorted = sorted(obs_points, key=lambda x: x[0])
        dates, incs = zip(*obs_sorted)
        cums = np.cumsum(incs)
        ax.scatter(dates, cums, s=23, facecolors="white", edgecolors=COLORS["obs"], lw=1.0, zorder=10)
        ax.text(
            dates[-2],
            cums[-2] + 0.5,
            "Megfigyelt",
            color=COLORS["obs"],
            fontsize=8,
            va="bottom",
            ha="center",
            fontweight="normal",
        )

    ax.yaxis.grid(True, color="#eeeeee", linestyle="-", which="major", zorder=0, alpha=0.5, lw=0.8)
    ax.xaxis.grid(False)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_english_date))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_ylabel("Kumulált fertőzések")
    ax.set_ylim(bottom=1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(start_date, start_date + timedelta(days=u_horizon_days))

    plt.tight_layout()
    plt.show()


def main() -> None:
    # ============================================================
    # Observations (Epuyén, Argentina; host-side incidence)
    # ============================================================
    host_observations_epuyen: List[Tuple[datetime, int]] = [
        (datetime(2018, 11, 2), 1),  # The index case (spillover)
        (datetime(2018, 11, 22), 2),  # First generation from the party
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

    # ============================================================
    # Priors and acceptance criteria
    # ============================================================

    baseline_priors: Dict[str, Tuple[float, float]] = {
        "R0": (0.0, 7.5),
        "k": (0.2, 30.0),
        "r": (0.01, 0.99),
        "alpha": (0.01, 30.0),
        "theta": (0.01, 20.0),
    }

    baseline_requirements: List[str] = [
        "R0 * r < 3",
        "alpha * theta > 15",
        "alpha * theta < 60",
        "(k / (k + R0)) ^ k > 0.25",
        "(k / (k + R0 * r)) ^ k > 0.35",
        "((R0 * r) ^ (1 / alpha) - 1) / theta < 0.05",
    ]

    # Acceptance-window construction kwargs for snapshot 11 (from your optimizer output).
    baseline_builder_kwargs: Dict[str, Any] = {
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
    }

    baseline_builder_kwargs1: Dict[str, Any] = {
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
    }

    # Run only a single (final) snapshot.
    obs_sorted = _get_sorted_obs_points(host_observations_epuyen)
    pars = _build_parameters(priors=baseline_priors, requirements=baseline_requirements)
    control_start_date = datetime(2026, 4, 6)
    results = run_baseline_final_snapshot_results(
        obs_points=obs_sorted,
        pars=pars,
        builder_kwargs=baseline_builder_kwargs,
        control_start_date=control_start_date,
        min_required=MIN_REQUIRED,
    )
    start_date = obs_sorted[0][0]

    plot_last_posterior(results, pdf_estimator="scipy")
    plot_cumulative_infections_datetime(results, start_date=start_date, obs_points=obs_sorted,
                                        u_horizon_days=(obs_sorted[-1][0] - obs_sorted[0][0]).days + 5)
    # run_and_plot_cumulative_first_gen_hanta(
    #     obs_points=obs_sorted,
    #     controlled_results=results,
    #     pars=pars,
    #     builder_kwargs=baseline_builder_kwargs,
    #     control_start_date=control_start_date,
    #     u_horizon_days=70.0,
    # )

    # plot_last_posterior_conditioned_quiet_for_T(results, T=10.0)


if __name__ == "__main__":
    main()
