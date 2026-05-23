from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lmfit import Parameters as LMParams, minimize

from python.eventide import (
    DrawCollector,
    InfectionTimeCollector,
    IndexOffspringCriterion,
    Parameters,
    Scenario,
    Simulator,
)
from python.optimize_acceptance_windows import (
    EvalLogger,
    REPO_ROOT,
    _compute_dynamic_t_run,
    build_acceptance_inequalities,
    l2_distance,
    mean_cumulative_from_infections,
    observed_cumulative_at,
)
from python.plots.misc import SnapshotResult
from python.plots.plot_cumulative import plot_cumulative_infections_last_numeric
from python.plots.plot_posteriors import plot_posterior_grid_single, _support_interval
from python.plots.style import set_style

PARAMETER_ORDER = ("R0", "k", "r", "alpha", "theta")
DEFAULT_DE_KWARGS: Dict[str, Any] = {
    "strategy": "best1bin",
    "max_nfev": 500,
    "popsize": 10,
    "workers": 1,
    "polish": False,
}
iter_counter = 0


@dataclass(frozen=True, slots=True)
class SyntheticTruth:
    values: Dict[str, float]
    total_individuals: int
    start_date: datetime
    obs_points: List[Tuple[datetime, int]]
    infection_times: np.ndarray


@dataclass(frozen=True, slots=True)
class EarlyOptimizationStop(RuntimeError):
    best_kwargs: Dict[str, Any]
    objective: float
    accepted: int


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


def default_acceptance_kwargs() -> Dict[str, Any]:
    return {
        "sigma_days": 1.0,
        "beta": 0.75,
        "neighbor_weight": 0.8,
        "grid_step_days": 0.25,
        "min_seg_days": 1.0,
        "kmax": 5,
        "baseline_p": 0.10,
        "alpha": 0.10,
        "h_max": 0.50,
        "eps_share": 1e-3,
        "include_gap_windows": True,
        "include_union_windows": True,
        "max_unions_to_keep": 3,
        "gap_scale": 0.40,
        "include_global_total": True,
    }


def _cast_accept_kwargs(raw: Mapping[str, Any]) -> Dict[str, Any]:
    d = dict(raw)
    d["sigma_days"] = float(d["sigma_days"])
    d["beta"] = float(d["beta"])
    d["neighbor_weight"] = float(d["neighbor_weight"])
    d["grid_step_days"] = float(d["grid_step_days"])
    d["min_seg_days"] = float(d["min_seg_days"])
    d["kmax"] = int(round(float(d["kmax"])))
    d["baseline_p"] = float(d["baseline_p"])
    d["alpha"] = float(d["alpha"])
    d["h_max"] = float(d["h_max"])
    d["eps_share"] = float(d["eps_share"])
    d["include_gap_windows"] = bool(d["include_gap_windows"])
    d["include_union_windows"] = bool(d["include_union_windows"])
    d["max_unions_to_keep"] = int(round(float(d["max_unions_to_keep"])))
    d["gap_scale"] = float(d["gap_scale"])
    d["include_global_total"] = bool(d["include_global_total"])
    return d


def _kwargs_from_params(pars: LMParams) -> Dict[str, Any]:
    out = {name: float(pars[name].value) for name in pars.keys()}
    out["kmax"] = int(round(out["kmax"]))
    out["max_unions_to_keep"] = int(round(out["max_unions_to_keep"]))
    out["include_gap_windows"] = bool(out["include_gap_windows"] > 0.5)
    out["include_union_windows"] = bool(out["include_union_windows"] > 0.5)
    out["include_global_total"] = bool(out["include_global_total"] > 0.5)
    return out


def _create_acceptance_lmparams(seed_kwargs: Mapping[str, Any]) -> LMParams:
    seed = _cast_accept_kwargs(seed_kwargs)
    params = LMParams()
    params.add("sigma_days", value=float(seed["sigma_days"]), min=0.25, max=8.0)
    params.add("beta", value=float(seed["beta"]), min=0.50, max=0.99)
    params.add("neighbor_weight", value=float(seed["neighbor_weight"]), min=0.10, max=3.0)
    params.add("grid_step_days", value=float(seed["grid_step_days"]), min=0.10, max=0.75)
    params.add("min_seg_days", value=float(seed["min_seg_days"]), min=0.5, max=5.0)
    params.add("kmax", value=float(seed["kmax"]), min=2.0, max=8.0)
    params.add("baseline_p", value=float(seed["baseline_p"]), min=0.01, max=0.30)
    params.add("alpha", value=float(seed["alpha"]), min=0.01, max=0.40)
    params.add("h_max", value=float(seed["h_max"]), min=0.10, max=1.00)
    params.add("eps_share", value=float(seed["eps_share"]), min=1e-6, max=0.10)
    params.add("include_gap_windows", value=1.0 if seed["include_gap_windows"] else 0.0, min=0.0, max=1.0)
    params.add("include_union_windows", value=1.0 if seed["include_union_windows"] else 0.0, min=0.0, max=1.0)
    params.add("max_unions_to_keep", value=float(seed["max_unions_to_keep"]), min=0.0, max=6.0)
    params.add("gap_scale", value=float(seed["gap_scale"]), min=0.10, max=0.90)
    params.add("include_global_total", value=1.0 if seed["include_global_total"] else 0.0, min=0.0, max=1.0)
    return params


def _infection_times_to_obs_points(
        infection_times: Sequence[float],
        *,
        start_date: datetime,
) -> List[Tuple[datetime, int]]:
    by_day: Dict[datetime, int] = {}
    for t in np.sort(np.asarray(infection_times, dtype=float)):
        date = start_date + timedelta(days=int(np.floor(float(t))))
        by_day[date] = by_day.get(date, 0) + 1
    return sorted(by_day.items(), key=lambda x: x[0])


def _simulate_synthetic_truth(
        *,
        true_params: Mapping[str, float],
        total_individuals: int,
        start_date: datetime,
        t_run_days: float,
        max_cases: int,
        max_workers: int,
) -> SyntheticTruth:
    pars = Parameters(
        R0=(true_params["R0"], true_params["R0"]),
        k=(true_params["k"], true_params["k"]),
        r=(true_params["r"], true_params["r"]),
        alpha=(true_params["alpha"], true_params["alpha"]),
        theta=(true_params["theta"], true_params["theta"]),
    )
    collector = InfectionTimeCollector()
    sim = Simulator(
        parameters=pars,
        sampler=pars.create_latin_hypercube_sampler(scramble=False),
        start_date=start_date,
        scenario=Scenario([]),
        criteria=[],
        collectors=[collector],
        num_trajectories=1,
        chunk_size=1,
        T_run=float(t_run_days),
        max_cases=int(max_cases),
        max_workers=max_workers,
        min_required=1,
    )
    sim.run()

    if not collector.infection_times:
        raise RuntimeError("Synthetic trajectory generation produced no trajectory.")

    infection_times = np.sort(np.asarray(collector.infection_times[0], dtype=float))
    if infection_times.size < total_individuals:
        raise ValueError(
            f"Synthetic trajectory produced only {infection_times.size} individuals, "
            f"but {total_individuals} were requested. Increase T_run/max_cases or change the true parameters."
        )

    infection_times = infection_times[:total_individuals]
    obs_points = _infection_times_to_obs_points(infection_times, start_date=start_date)
    return SyntheticTruth(
        values={name: float(true_params[name]) for name in PARAMETER_ORDER},
        total_individuals=int(total_individuals),
        start_date=start_date,
        obs_points=obs_points,
        infection_times=infection_times,
    )


def run_once_and_score(
        sim_start: datetime,
        obs_points: Sequence[Tuple[datetime, int]],
        accept_kwargs: Mapping[str, Any],
        *,
        parameters: Parameters,
        scenario: Scenario,
        index_offspring_bounds: Tuple[int, int],
        min_required: int,
        num_trajectories: int,
        chunk_size: int,
        max_workers: int,
        max_cases: int,
        include_full_artifacts: bool = False,
        verbose: bool = True,
) -> Tuple[float, int, dict]:
    criteria = build_acceptance_inequalities(
        obs_points=obs_points,
        simulation_start=sim_start,
        **_cast_accept_kwargs(accept_kwargs),
    )
    criteria = criteria + [IndexOffspringCriterion(*index_offspring_bounds)]

    t_run_dynamic = _compute_dynamic_t_run(
        sim_start=sim_start,
        obs_points=obs_points,
        criteria=criteria,
        safety_extra_days=7,
    )

    infection_times = InfectionTimeCollector()
    sim = Simulator(
        parameters=parameters,
        sampler=parameters.create_latin_hypercube_sampler(scramble=False),
        start_date=sim_start,
        scenario=scenario,
        criteria=criteria,
        collectors=[infection_times],
        num_trajectories=num_trajectories,
        chunk_size=chunk_size,
        T_run=t_run_dynamic,
        max_cases=max_cases,
        max_workers=max_workers,
        min_required=min_required,
    )
    sim.run()

    times_eval = sorted({t for t, _ in obs_points})
    n_obs = observed_cumulative_at(times_eval, obs_points)
    n_bar = mean_cumulative_from_infections(times_eval, infection_times.infection_times, sim.start_date)
    objective = l2_distance(n_bar, n_obs)

    try:
        accepted_count = int(getattr(sim, "accepted", len(infection_times.infection_times)))
    except Exception:
        accepted_count = len(infection_times.infection_times)

    if verbose:
        global iter_counter
        iter_counter += 1
        print(f"{iter_counter: 5d} | N_obs={len(obs_points)} | Obj(L2)={objective:10.6f} | Acc={accepted_count}")

    artifacts = {
        "criteria": criteria,
        "sim": sim if include_full_artifacts else None,
        "infection_times": infection_times if include_full_artifacts else None,
        "accepted_count": accepted_count,
        "N_obs": n_obs,
        "N_bar": n_bar,
        "T_run": t_run_dynamic,
    }
    return objective, accepted_count, artifacts


def optimize_acceptance_kwargs_synthetic(
        *,
        obs_points: Sequence[Tuple[datetime, int]],
        parameters: Parameters,
        scenario: Scenario,
        initial_kwargs: Mapping[str, Any],
        index_offspring_bounds: Tuple[int, int],
        optimization_min_required: int,
        optimization_num_trajectories: int,
        penalty_min_accepted: int,
        max_cases: int,
        chunk_size: int,
        max_workers: int,
        objective_tolerance: float,
        log_path: str,
        de_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sim_start = min(t for t, _ in obs_points)
    params = _create_acceptance_lmparams(initial_kwargs)
    opt_kwargs = dict(DEFAULT_DE_KWARGS)
    if de_kwargs:
        opt_kwargs.update(dict(de_kwargs))
    opt_kwargs["polish"] = False

    logger = EvalLogger(log_path, write_every=1)

    def residuals(lmpars: LMParams) -> np.ndarray:
        trial = _kwargs_from_params(lmpars)
        raw_objective, accepted, _ = run_once_and_score(
            sim_start,
            obs_points,
            trial,
            parameters=parameters,
            scenario=scenario,
            index_offspring_bounds=index_offspring_bounds,
            min_required=optimization_min_required,
            num_trajectories=optimization_num_trajectories,
            chunk_size=chunk_size,
            max_workers=max_workers,
            max_cases=max_cases,
            include_full_artifacts=False,
            verbose=True,
        )
        penalized = raw_objective
        if accepted < penalty_min_accepted:
            shortfall = penalty_min_accepted - accepted
            penalized += float(shortfall * shortfall)
        print(
            f"      raw={raw_objective:10.6f} penalized={penalized:10.6f} "
            f"accepted={accepted:6d} kwargs={trial}"
        )
        logger.log({"objective": penalized, "raw_objective": raw_objective, "accepted": accepted, **trial})
        if raw_objective <= objective_tolerance and accepted >= penalty_min_accepted:
            raise EarlyOptimizationStop(
                best_kwargs=_cast_accept_kwargs(trial),
                objective=float(raw_objective),
                accepted=int(accepted),
            )
        return np.array([penalized], dtype=float)

    global iter_counter
    iter_counter = 0

    try:
        result = minimize(
            residuals,
            params,
            method="differential_evolution",
            **opt_kwargs,
        )
        best_kwargs = _cast_accept_kwargs(_kwargs_from_params(result.params))
        stop_info = {
            "stopped_early": False,
            "objective": float(getattr(result, "chisqr", np.nan)),
            "accepted": None,
        }
    except EarlyOptimizationStop as exc:
        best_kwargs = dict(exc.best_kwargs)
        stop_info = {
            "stopped_early": True,
            "objective": float(exc.objective),
            "accepted": int(exc.accepted),
        }
    finally:
        logger.flush()

    _, accepted_count, artifacts = run_once_and_score(
        sim_start,
        obs_points,
        best_kwargs,
        parameters=parameters,
        scenario=scenario,
        index_offspring_bounds=index_offspring_bounds,
        min_required=optimization_min_required,
        num_trajectories=optimization_num_trajectories,
        chunk_size=chunk_size,
        max_workers=max_workers,
        max_cases=max_cases,
        include_full_artifacts=False,
        verbose=False,
    )
    artifacts["stop_info"] = stop_info
    artifacts["accepted_count"] = accepted_count
    return best_kwargs, artifacts


def run_final_abc_snapshot(
        *,
        obs_points: Sequence[Tuple[datetime, int]],
        parameters: Parameters,
        accept_kwargs: Mapping[str, Any],
        scenario: Scenario,
        index_offspring_bounds: Tuple[int, int],
        min_required: int,
        num_trajectories: int,
        chunk_size: int,
        max_workers: int,
        max_cases: int,
) -> SnapshotResult:
    sim_start = min(t for t, _ in obs_points)
    criteria = build_acceptance_inequalities(
        obs_points=obs_points,
        simulation_start=sim_start,
        **_cast_accept_kwargs(accept_kwargs),
    )
    criteria = criteria + [IndexOffspringCriterion(*index_offspring_bounds)]

    t_run_dynamic = _compute_dynamic_t_run(
        sim_start=sim_start,
        obs_points=obs_points,
        criteria=criteria,
        safety_extra_days=7,
    )

    draws_collector = DrawCollector()
    infection_times = InfectionTimeCollector()
    sim = Simulator(
        parameters=parameters,
        sampler=parameters.create_latin_hypercube_sampler(scramble=False),
        start_date=sim_start,
        scenario=scenario,
        criteria=criteria,
        collectors=[draws_collector, infection_times],
        num_trajectories=num_trajectories,
        chunk_size=chunk_size,
        T_run=t_run_dynamic,
        max_cases=max_cases,
        max_workers=max_workers,
        min_required=min_required,
    )
    sim.run()

    return SnapshotResult(
        m=len(obs_points),
        t_star=float((obs_points[-1][0] - sim_start).days),
        T_grid=np.empty(0, dtype=float),
        p_uncond_mean=np.empty(0, dtype=float),
        p_cond_mean=np.empty(0, dtype=float),
        p_uncond_draws=np.empty((0, 0), dtype=float),
        p_cond_draws=np.empty((0, 0), dtype=float),
        draws_array=np.asarray(draws_collector, dtype=float),
        infection_times_2d=[np.sort(np.asarray(t, dtype=float)) for t in infection_times.infection_times],
        n_obs=int(sum(c for _, c in obs_points)),
        next_T=None,
        stopped_pairs=None,
    )


def _derived_metrics_from_draws(draws: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.asarray(draws, dtype=float)
    R0 = arr[:, 0]
    k = arr[:, 1]
    r = arr[:, 2]
    alpha = arr[:, 3]
    theta = arr[:, 4]
    Re = r * R0
    alpha_theta = alpha * theta
    p0_Re = (k / (k + Re)) ** k
    return {
        "R0": R0,
        "k": k,
        "r": r,
        "alpha": alpha,
        "theta": theta,
        "Re": Re,
        "alpha_theta": alpha_theta,
        "p0_Re": p0_Re,
    }


def _truth_metric_values(true_params: Mapping[str, float]) -> Dict[str, float]:
    R0 = float(true_params["R0"])
    k = float(true_params["k"])
    r = float(true_params["r"])
    alpha = float(true_params["alpha"])
    theta = float(true_params["theta"])
    Re = r * R0
    alpha_theta = alpha * theta
    p0_Re = (k / (k + Re)) ** k
    return {
        "R0": R0,
        "k": k,
        "r": r,
        "alpha": alpha,
        "theta": theta,
        "Re": Re,
        "alpha_theta": alpha_theta,
        "p0_Re": p0_Re,
    }


def _posterior_comparison_table(
        *,
        snapshot: SnapshotResult,
        true_params: Mapping[str, float],
) -> pd.DataFrame:
    posterior = _derived_metrics_from_draws(snapshot.draws_array)
    truth = _truth_metric_values(true_params)
    rows: List[Dict[str, Any]] = []
    for key, x in posterior.items():
        vals = np.asarray(x, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        truth_val = float(truth[key])
        mean = float(np.mean(vals))
        median = float(np.median(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        var = float(np.var(vals, ddof=1)) if vals.size > 1 else 0.0
        cv = float(sd / mean) if abs(mean) > 1e-12 else np.nan
        q025, q25, q75, q975 = np.percentile(vals, [2.5, 25.0, 75.0, 97.5])
        hdi_lo, hdi_hi = _support_interval(vals, mass=0.95)
        rows.append({
            "metric": key,
            "artificial_value": truth_val,
            "mean": mean,
            "median": median,
            "std": sd,
            "var": var,
            "cv": cv,
            "min": float(np.min(vals)),
            "q02.5": float(q025),
            "q25": float(q25),
            "q75": float(q75),
            "q97.5": float(q975),
            "max": float(np.max(vals)),
            "hdi95_lo": float(hdi_lo),
            "hdi95_hi": float(hdi_hi),
            "mean_bias": mean - truth_val,
            "median_bias": median - truth_val,
            "mean_rel_err_pct": 100.0 * (mean - truth_val) / truth_val if abs(truth_val) > 1e-12 else np.nan,
            "median_rel_err_pct": 100.0 * (median - truth_val) / truth_val if abs(truth_val) > 1e-12 else np.nan,
            "inside_hdi95": bool(hdi_lo <= truth_val <= hdi_hi),
            "inside_q95": bool(q025 <= truth_val <= q975),
            "n_accept": int(vals.size),
        })
    return pd.DataFrame(rows)


def _print_run_summary(
        *,
        truth: SyntheticTruth,
        best_kwargs: Mapping[str, Any],
        optimization_artifacts: Mapping[str, Any],
        final_snapshot: SnapshotResult,
        comparison_table: pd.DataFrame,
) -> None:
    print("\nSynthetic truth")
    print(f"  Total individuals used: {truth.total_individuals}")
    print(f"  Observation points:      {len(truth.obs_points)}")
    print(f"  Date range:              {truth.obs_points[0][0].date()} -> {truth.obs_points[-1][0].date()}")
    print("  Parameters:")
    for name in PARAMETER_ORDER:
        print(f"    {name:6s} = {truth.values[name]:.6f}")

    print("\nBest acceptance kwargs")
    for key, value in best_kwargs.items():
        print(f"  {key:20s}: {value}")

    stop_info = optimization_artifacts.get("stop_info", {})
    print("\nOptimization summary")
    print(f"  Accepted in scored best run: {optimization_artifacts.get('accepted_count')}")
    print(f"  Dynamic T_run:               {optimization_artifacts.get('T_run')}")
    print(f"  Early stop:                  {stop_info.get('stopped_early')}")
    print(f"  Stop objective:              {stop_info.get('objective')}")
    print(f"  Stop accepted:               {stop_info.get('accepted')}")

    print("\nFinal ABC posterior")
    print(f"  Accepted posterior draws: {final_snapshot.draws_array.shape[0]}")
    print(f"  Observation total:        {final_snapshot.n_obs}")

    print("\nPosterior vs artificial values")
    with pd.option_context("display.max_columns", None, "display.width", 240, "display.float_format", "{:,.6f}".format):
        print(comparison_table.to_string(index=False))


def main() -> None:
    set_style(column="double", base_font=10, use_tex=False, show_grid=False, dpi=320)

    # ------------------------------------------------------------------
    # Reproducibility and synthetic truth
    # ------------------------------------------------------------------
    master_seed = 20260520
    np.random.seed(master_seed)

    start_date = datetime(2026, 1, 1)
    artificial_total_individuals = 35
    true_params = {
        "R0": 2.2,
        "k": 0.9,
        "r": 0.35,
        "alpha": 5.5,
        "theta": 1.4,
    }
    synthetic_t_run_days = 120.0
    synthetic_max_cases = 500

    # ------------------------------------------------------------------
    # Prior ranges and parameter restrictions for inference
    # ------------------------------------------------------------------
    priors = {
        "R0": (0.25, 6.0),
        "k": (0.2, 5.0),
        "r": (0.01, 0.99),
        "alpha": (0.4, 12.0),
        "theta": (0.2, 4.0),
    }
    requirements = [
        "R0 * r < 2.5",
        "3.0 < alpha * theta",
        "alpha * theta < 12.0",
        "(k / (k + R0 * r)) ^ k > 0.10",
    ]

    # ------------------------------------------------------------------
    # ABC / optimization controls
    # ------------------------------------------------------------------
    initial_acceptance_kwargs = default_acceptance_kwargs()
    index_offspring_bounds = (0, artificial_total_individuals)
    optimization_num_trajectories = 2_000_000
    optimization_min_required = 500
    optimization_penalty_min_accepted = 80
    optimization_chunk_size = 10_000
    optimization_max_workers = 13
    optimization_max_cases = 500
    objective_tolerance = 0.01 * artificial_total_individuals
    de_kwargs = {
        "strategy": "best1bin",
        "max_nfev": 500,
        "popsize": 10,
        "workers": 1,
        "polish": False,
    }

    final_num_trajectories = 200_000_000
    final_min_required = 10_000
    final_chunk_size = 10_000
    final_max_workers = 13
    final_max_cases = 500

    # ------------------------------------------------------------------
    # Build synthetic data
    # ------------------------------------------------------------------
    truth = _simulate_synthetic_truth(
        true_params=true_params,
        total_individuals=artificial_total_individuals,
        start_date=start_date,
        t_run_days=synthetic_t_run_days,
        max_cases=synthetic_max_cases,
        max_workers=1,
    )
    obs_points = truth.obs_points

    print("Synthetic observation points")
    for t, c in obs_points:
        print(f"  {t.date()}  {c}")

    # ------------------------------------------------------------------
    # Inference setup
    # ------------------------------------------------------------------
    parameters = _build_parameters(priors=priors, requirements=requirements)
    scenario = Scenario([])

    best_kwargs, optimization_artifacts = optimize_acceptance_kwargs_synthetic(
        obs_points=obs_points,
        parameters=parameters,
        scenario=scenario,
        initial_kwargs=initial_acceptance_kwargs,
        index_offspring_bounds=index_offspring_bounds,
        optimization_min_required=optimization_min_required,
        optimization_num_trajectories=optimization_num_trajectories,
        penalty_min_accepted=optimization_penalty_min_accepted,
        max_cases=optimization_max_cases,
        chunk_size=optimization_chunk_size,
        max_workers=optimization_max_workers,
        objective_tolerance=objective_tolerance,
        log_path=str(REPO_ROOT / "optimization_logs" / "synthetic_acceptance_search.csv"),
        de_kwargs=de_kwargs,
    )

    final_snapshot = run_final_abc_snapshot(
        obs_points=obs_points,
        parameters=parameters,
        accept_kwargs=best_kwargs,
        scenario=scenario,
        index_offspring_bounds=index_offspring_bounds,
        min_required=final_min_required,
        num_trajectories=final_num_trajectories,
        chunk_size=final_chunk_size,
        max_workers=final_max_workers,
        max_cases=final_max_cases,
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    obs_points_days = [
        (((t - start_date).total_seconds() / 86400.0), c)
        for t, c in obs_points
    ]
    plot_cumulative_infections_last_numeric(
        [final_snapshot],
        resolution=0.25,
        perc_bands=(0.95, 0.8, 0.5),
        obs_points_days=obs_points_days,
    )
    plot_posterior_grid_single(final_snapshot, mass=0.95, bw_adjust=1.0, n_grid=600)

    # ------------------------------------------------------------------
    # Detailed comparison table
    # ------------------------------------------------------------------
    comparison_table = _posterior_comparison_table(
        snapshot=final_snapshot,
        true_params=true_params,
    )
    _print_run_summary(
        truth=truth,
        best_kwargs=best_kwargs,
        optimization_artifacts=optimization_artifacts,
        final_snapshot=final_snapshot,
        comparison_table=comparison_table,
    )


if __name__ == "__main__":
    main()
