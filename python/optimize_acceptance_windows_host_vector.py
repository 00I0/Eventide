from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from lmfit import Parameters as LMParams, minimize

from python.eventide import (
    AlternatingSimulator,
    IndexOffspringCriterion,
    InfectionTimeCollector,
    Parameters,
    Scenario,
    Species,
    SpeciesConfiguration,
)
from python.optimize_acceptance_windows import (
    EvalLogger,
    OptimizerConfig,
    REPO_ROOT,
    SnapshotConfig,
    _compute_dynamic_t_run,
    build_acceptance_inequalities,
    l2_distance,
    mean_cumulative_from_infections,
    observed_cumulative_at,
)

OPTIMIZATION_MIN_ACCEPTED = 1000
OPTIMIZATION_NUM_TRAJECTORIES = 1_000_000_000
OPTIMIZATION_CHUNK_SIZE = 50_000
OPTIMIZATION_MAX_WORKERS = 13
OPTIMIZATION_MAX_NFEV = 500
FINAL_MIN_REQUIRED = 1_000
FINAL_NUM_TRAJECTORIES = 1_000_000_000
FINAL_CHUNK_SIZE = 50_000
FINAL_MAX_WORKERS = 13

DEFAULT_DE_KWARGS: Dict[str, Any] = {
    "strategy": "best1bin",
    "max_nfev": OPTIMIZATION_MAX_NFEV,
    "popsize": 10,
    "workers": 1,
    "polish": False,
}

iter_counter = 0


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


def _cast_accept_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
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


def cumulative_to_incidence(obs_points_cumulative: Sequence[Tuple[datetime, int]]) -> list[tuple[datetime, int]]:
    obs = sorted(obs_points_cumulative, key=lambda x: x[0])
    if not obs:
        return []

    incidence: list[tuple[datetime, int]] = []
    prev = 0
    for t, cum in obs:
        inc = int(cum) - prev
        if inc < 0:
            raise ValueError("Cumulative host observations must be nondecreasing.")
        if inc > 0:
            incidence.append((t, inc))
        prev = int(cum)
    return incidence


def run_once_and_score_alternating_host_observed(
        sim_start: datetime,
        host_obs_points: Sequence[Tuple[datetime, int]],
        accept_kwargs: Dict[str, Any],
        *,
        host_parameters: Parameters,
        vector_parameters: Parameters,
        host_scenario: Scenario,
        vector_scenario: Scenario,
        min_required: int,
        num_trajectories: int,
        chunk_size: int,
        max_workers: int,
        host_base_criteria: Optional[Sequence[Any]] = None,
        vector_base_criteria: Optional[Sequence[Any]] = None,
        root_species: Species = Species.HOST,
        max_cases: int = 500,
        include_full_artifacts: bool = False,
        verbose: bool = True,
) -> Tuple[float, int, dict]:
    host_window_criteria = build_acceptance_inequalities(
        obs_points=host_obs_points,
        simulation_start=sim_start,
        **_cast_accept_kwargs(accept_kwargs),
    )
    host_criteria = list(host_window_criteria)
    if host_base_criteria:
        host_criteria.extend(host_base_criteria)
    vector_criteria = list(vector_base_criteria or [])

    t_run_dynamic = _compute_dynamic_t_run(
        sim_start=sim_start,
        obs_points=host_obs_points,
        criteria=host_criteria,
        safety_extra_days=7,
    )

    host_infection_times = InfectionTimeCollector()
    vector_infection_times = InfectionTimeCollector()

    host_config = SpeciesConfiguration(
        parameters=host_parameters,
        sampler=host_parameters.create_latin_hypercube_sampler(),
        scenario=host_scenario,
        criteria=host_criteria,
        collectors=[host_infection_times],
    )
    vector_config = SpeciesConfiguration(
        parameters=vector_parameters,
        sampler=vector_parameters.create_latin_hypercube_sampler(),
        scenario=vector_scenario,
        criteria=vector_criteria,
        collectors=[vector_infection_times],
    )

    sim = AlternatingSimulator(
        host=host_config,
        vector=vector_config,
        start_date=sim_start,
        num_trajectories=num_trajectories,
        min_required=min_required,
        chunk_size=chunk_size,
        T_run=t_run_dynamic,
        max_cases=max_cases,
        max_workers=max_workers,
        root_species=root_species,
    )
    sim.run()

    times_eval = sorted({t for t, _ in host_obs_points})
    n_obs = observed_cumulative_at(times_eval, host_obs_points)
    n_bar = mean_cumulative_from_infections(times_eval, host_infection_times.infection_times, sim.start_date)
    objective = l2_distance(n_bar, n_obs)

    try:
        accepted_count = int(getattr(sim, "accepted", len(host_infection_times.infection_times)))
    except Exception:
        accepted_count = len(host_infection_times.infection_times)

    if verbose:
        global iter_counter
        iter_counter += 1
        print(f"{iter_counter: 5d} | N_obs={len(host_obs_points)} | Obj(L2)={objective:10.6f} | Acc={accepted_count}")

    artifacts = {
        "host_window_criteria": host_window_criteria,
        "host_criteria": host_criteria,
        "vector_criteria": vector_criteria,
        "accepted_count": accepted_count,
        "N_obs": n_obs,
        "N_bar": n_bar,
        "T_run": t_run_dynamic,
        "sim": sim if include_full_artifacts else None,
        "host_infection_times": host_infection_times if include_full_artifacts else None,
        "vector_infection_times": vector_infection_times if include_full_artifacts else None,
    }
    return objective, accepted_count, artifacts


def optimize_acceptance_kwargs_alternating_host_observed(
        *,
        host_obs_points: Sequence[Tuple[datetime, int]],
        initial_kwargs: Optional[Dict[str, Any]],
        host_parameters: Parameters,
        vector_parameters: Parameters,
        host_scenario: Scenario,
        vector_scenario: Scenario,
        host_base_criteria: Optional[Sequence[Any]] = None,
        vector_base_criteria: Optional[Sequence[Any]] = None,
        root_species: Species = Species.HOST,
        max_cases: int = 4_000,
        optimization_min_accepted: int = OPTIMIZATION_MIN_ACCEPTED,
        optimization_num_trajectories: int = OPTIMIZATION_NUM_TRAJECTORIES,
        optimization_chunk_size: int = OPTIMIZATION_CHUNK_SIZE,
        optimization_max_workers: int = OPTIMIZATION_MAX_WORKERS,
        penalty_min_accepted: int = 80,
        final_min_required: int = FINAL_MIN_REQUIRED,
        final_num_trajectories: int = FINAL_NUM_TRAJECTORIES,
        final_chunk_size: int = FINAL_CHUNK_SIZE,
        final_max_workers: int = FINAL_MAX_WORKERS,
        de_kwargs: Optional[Dict[str, Any]] = None,
        log_path: Optional[str] = None,
        store_full_artifacts: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not host_obs_points:
        raise ValueError("host_obs_points must not be empty.")

    sim_start = min(t for t, _ in host_obs_points)
    seed = _cast_accept_kwargs(initial_kwargs or default_acceptance_kwargs())

    params = LMParams()
    params.add("sigma_days", value=float(seed["sigma_days"]), min=0.25, max=4.0)
    params.add("beta", value=float(seed["beta"]), min=0.50, max=0.99)
    params.add("neighbor_weight", value=float(seed["neighbor_weight"]), min=0.10, max=2.0)
    params.add("grid_step_days", value=float(seed["grid_step_days"]), min=0.10, max=0.75)
    params.add("min_seg_days", value=float(seed["min_seg_days"]), min=0.5, max=5.0)
    params.add("kmax", value=float(seed["kmax"]), min=2.0, max=8.0)
    params.add("baseline_p", value=float(seed["baseline_p"]), min=0.01, max=0.30)
    params.add("alpha", value=float(seed["alpha"]), min=0.01, max=0.40)
    params.add("h_max", value=float(seed["h_max"]), min=0.10, max=1.00)
    params.add("eps_share", value=float(seed["eps_share"]), min=1e-6, max=0.10)
    params.add("include_gap_windows", value=1.0 if seed["include_gap_windows"] else 0.0, min=0.0, max=1.0)
    params.add("include_union_windows", value=1.0 if seed["include_union_windows"] else 0.0, min=0.0, max=1.0)
    params.add("max_unions_to_keep", value=float(seed["max_unions_to_keep"]), min=0.0, max=5.0)
    params.add("gap_scale", value=float(seed["gap_scale"]), min=0.10, max=0.90)
    params.add("include_global_total", value=1.0 if seed["include_global_total"] else 0.0, min=0.0, max=1.0)

    logger = EvalLogger(
        log_path or str(REPO_ROOT / "optimization_logs" / "host_vector_host_only.csv"),
        write_every=5,
    )
    eval_counter = 0

    def _kwargs_from_params(pars: LMParams) -> Dict[str, Any]:
        out = {name: float(pars[name].value) for name in pars.keys()}
        out["kmax"] = int(round(out["kmax"]))
        out["max_unions_to_keep"] = int(round(out["max_unions_to_keep"]))
        out["include_gap_windows"] = bool(out["include_gap_windows"] > 0.5)
        out["include_union_windows"] = bool(out["include_union_windows"] > 0.5)
        out["include_global_total"] = bool(out["include_global_total"] > 0.5)
        return out

    def residuals(lmpars: LMParams) -> np.ndarray:
        nonlocal eval_counter
        eval_counter += 1
        trial = _kwargs_from_params(lmpars)
        objective, accepted, _ = run_once_and_score_alternating_host_observed(
            sim_start,
            host_obs_points,
            trial,
            host_parameters=host_parameters,
            vector_parameters=vector_parameters,
            host_scenario=host_scenario,
            vector_scenario=vector_scenario,
            host_base_criteria=host_base_criteria,
            vector_base_criteria=vector_base_criteria,
            root_species=root_species,
            max_cases=max_cases,
            min_required=optimization_min_accepted,
            num_trajectories=optimization_num_trajectories,
            chunk_size=optimization_chunk_size,
            max_workers=optimization_max_workers,
            include_full_artifacts=False,
            verbose=False,
        )
        if accepted < penalty_min_accepted:
            shortfall = penalty_min_accepted - accepted
            objective += float(shortfall * shortfall)
        print(f"opt_eval={eval_counter:4d} obj={objective:10.4f} accepted={accepted:6d}")
        logger.log({"objective": objective, "accepted": accepted, **trial})
        return np.array([objective], dtype=float)

    opt_kwargs = dict(DEFAULT_DE_KWARGS)
    if de_kwargs:
        opt_kwargs.update(de_kwargs)
    opt_kwargs["polish"] = False

    result = minimize(
        residuals,
        params,
        method="differential_evolution",
        **opt_kwargs,
    )
    logger.flush()

    best_kwargs = _cast_accept_kwargs(_kwargs_from_params(result.params))
    print("Optimized acceptance kwargs:", best_kwargs)

    _, _, artifacts = run_once_and_score_alternating_host_observed(
        sim_start,
        host_obs_points,
        best_kwargs,
        host_parameters=host_parameters,
        vector_parameters=vector_parameters,
        host_scenario=host_scenario,
        vector_scenario=vector_scenario,
        host_base_criteria=host_base_criteria,
        vector_base_criteria=vector_base_criteria,
        root_species=root_species,
        max_cases=max_cases,
        min_required=final_min_required,
        num_trajectories=final_num_trajectories,
        chunk_size=final_chunk_size,
        max_workers=final_max_workers,
        include_full_artifacts=store_full_artifacts,
        verbose=False,
    )

    return best_kwargs, artifacts


def optimize_snapshots_alternating_host_observed(
        *,
        sim_start: datetime,
        all_host_obs_points: Sequence[Tuple[datetime, int]],
        host_parameters: Parameters,
        vector_parameters: Parameters,
        host_scenario: Scenario,
        vector_scenario: Scenario,
        snapshot_config: SnapshotConfig,
        host_base_criteria: Optional[Sequence[Any]] = None,
        vector_base_criteria: Optional[Sequence[Any]] = None,
        root_species: Species = Species.HOST,
        max_cases: int = 4_000,
        base_log_path: str = "opt_results_host_vector",
        initial_kwargs: Optional[Dict[str, Any]] = None,
        opt_min_required: int = OPTIMIZATION_MIN_ACCEPTED,
        opt_penalty_min_accepted: int = 80,
        opt_num_trajectories: int = OPTIMIZATION_NUM_TRAJECTORIES,
        opt_chunk_size: int = OPTIMIZATION_CHUNK_SIZE,
        opt_max_workers: int = OPTIMIZATION_MAX_WORKERS,
        final_min_required: int = FINAL_MIN_REQUIRED,
        final_num_trajectories: int = FINAL_NUM_TRAJECTORIES,
        final_chunk_size: int = FINAL_CHUNK_SIZE,
        final_max_workers: int = FINAL_MAX_WORKERS,
        store_full_snapshot_artifacts: bool = False,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    if not all_host_obs_points:
        raise ValueError("all_host_obs_points must not be empty.")

    current_best_kwargs = _cast_accept_kwargs(initial_kwargs or default_acceptance_kwargs())
    final_artifacts: Dict[int, Dict[str, Any]] = {}

    def create_params(init_values: Optional[Dict[str, Any]] = None) -> LMParams:
        seed = _cast_accept_kwargs(init_values or default_acceptance_kwargs())
        p = LMParams()
        p.add("sigma_days", value=float(seed["sigma_days"]), min=0.25, max=4.0)
        p.add("beta", value=float(seed["beta"]), min=0.50, max=0.99)
        p.add("neighbor_weight", value=float(seed["neighbor_weight"]), min=0.10, max=2.0)
        p.add("grid_step_days", value=float(seed["grid_step_days"]), min=0.10, max=0.75)
        p.add("min_seg_days", value=float(seed["min_seg_days"]), min=0.5, max=5.0)
        p.add("kmax", value=float(seed["kmax"]), min=2.0, max=8.0)
        p.add("baseline_p", value=float(seed["baseline_p"]), min=0.01, max=0.30)
        p.add("alpha", value=float(seed["alpha"]), min=0.01, max=0.40)
        p.add("h_max", value=float(seed["h_max"]), min=0.10, max=1.00)
        p.add("eps_share", value=float(seed["eps_share"]), min=1e-6, max=0.10)
        p.add("include_gap_windows", value=1.0 if seed["include_gap_windows"] else 0.0, min=0.0, max=1.0)
        p.add("include_union_windows", value=1.0 if seed["include_union_windows"] else 0.0, min=0.0, max=1.0)
        p.add("max_unions_to_keep", value=float(seed["max_unions_to_keep"]), min=0.0, max=5.0)
        p.add("gap_scale", value=float(seed["gap_scale"]), min=0.10, max=0.90)
        p.add("include_global_total", value=1.0 if seed["include_global_total"] else 0.0, min=0.0, max=1.0)
        return p

    def kwargs_from_params(pars: LMParams) -> Dict[str, Any]:
        out = {name: float(pars[name].value) for name in pars.keys()}
        out["kmax"] = int(round(out["kmax"]))
        out["max_unions_to_keep"] = int(round(out["max_unions_to_keep"]))
        out["include_gap_windows"] = bool(out["include_gap_windows"] > 0.5)
        out["include_union_windows"] = bool(out["include_union_windows"] > 0.5)
        out["include_global_total"] = bool(out["include_global_total"] > 0.5)
        return out

    for snap_idx, n_obs in enumerate(snapshot_config.counts):
        if n_obs > len(all_host_obs_points):
            print(
                f"Warning: Requested snapshot size {n_obs} exceeds available data ({len(all_host_obs_points)}). Skipping.")
            continue

        current_obs = list(all_host_obs_points[:n_obs])
        print(f"\n{'=' * 60}")
        print(f"SNAPSHOT {snap_idx + 1}/{len(snapshot_config.counts)}: Using first {n_obs} observation points")
        print(f"{'=' * 60}")

        logger = EvalLogger(f"{base_log_path}_snapshot_{n_obs}.csv", write_every=5)

        def residuals(pars: LMParams) -> np.ndarray:
            kwargs = kwargs_from_params(pars)
            obj, accepted, _ = run_once_and_score_alternating_host_observed(
                sim_start,
                current_obs,
                kwargs,
                host_parameters=host_parameters,
                vector_parameters=vector_parameters,
                host_scenario=host_scenario,
                vector_scenario=vector_scenario,
                host_base_criteria=host_base_criteria,
                vector_base_criteria=vector_base_criteria,
                root_species=root_species,
                max_cases=max_cases,
                min_required=opt_min_required,
                num_trajectories=opt_num_trajectories,
                chunk_size=opt_chunk_size,
                max_workers=opt_max_workers,
                include_full_artifacts=False,
                verbose=True,
            )
            row = {"objective": obj, "accepted": accepted, **kwargs}
            logger.log(row)

            if accepted < opt_penalty_min_accepted:
                shortfall = opt_penalty_min_accepted - accepted
                obj += float(shortfall * shortfall)
            return np.array([obj], dtype=float)

        global iter_counter
        iter_counter = 0

        print(f"\n--- Global Optimization ({snapshot_config.global_opt.method}) ---")
        params_global = create_params(current_best_kwargs)
        global_kwargs = snapshot_config.global_opt.kwargs.copy()
        if snapshot_config.global_opt.method == "differential_evolution":
            global_kwargs["polish"] = False

        result_global = minimize(
            residuals,
            params_global,
            method=snapshot_config.global_opt.method,
            **global_kwargs,
        )
        logger.flush()

        current_best_kwargs = _cast_accept_kwargs(kwargs_from_params(result_global.params))
        print(f"Global phase finished. Best Obj: {result_global.chisqr:.6f}")
        print("Best kwargs:", current_best_kwargs)

        _, _, artifacts = run_once_and_score_alternating_host_observed(
            sim_start,
            current_obs,
            current_best_kwargs,
            host_parameters=host_parameters,
            vector_parameters=vector_parameters,
            host_scenario=host_scenario,
            vector_scenario=vector_scenario,
            host_base_criteria=host_base_criteria,
            vector_base_criteria=vector_base_criteria,
            root_species=root_species,
            max_cases=max_cases,
            min_required=final_min_required,
            num_trajectories=final_num_trajectories,
            chunk_size=final_chunk_size,
            max_workers=final_max_workers,
            include_full_artifacts=store_full_snapshot_artifacts,
            verbose=False,
        )
        artifacts["best_kwargs"] = dict(current_best_kwargs)
        final_artifacts[n_obs] = artifacts

    return current_best_kwargs, final_artifacts


def _example_host_parameters() -> Parameters:
    # Dengue Outbreak
    return (
        Parameters(
            R0=(1.0, 5.0),
            k=(0.2, 20.0),
            r=(0.0, 1.0),
            alpha=(0.2, 15.0),
            theta=(0.2, 15.0),
        )
        .require("2.0 <= alpha * theta")
        .require("alpha * theta <= 10.0")
    )

    # Chikungunya Outbreak
    return (
        Parameters(
            R0=(1.0, 6.0),
            k=(1.0, 20.0),
            r=(1.0, 1.0),
            alpha=(0.0, 6.0),
            theta=(0.3, 4.0),
        )
        .require("2.0 <= alpha * theta")
        .require("alpha * theta <= 6.0")
    )


def _example_vector_parameters() -> Parameters:
    # Dengue Outbreak
    return (
        Parameters(
            R0=(0.2, 3),
            k=(0.2, 20.0),
            r=(0.0, 1.0),
            alpha=(0.2, 15.0),
            theta=(0.2, 15.0),
        )
        .require("3.0 <= alpha * theta")
        .require("alpha * theta <= 18.0")
    )

    # Chikungunya Outbreak
    return (
        Parameters(
            R0=(0.3, 2.0),
            k=(1.0, 20.0),
            r=(0.0, 1.0),
            alpha=(1.0, 4.0),
            theta=(0.5, 10.0),
        )
        .require("5.0 <= alpha * theta")
        .require("alpha * theta <= 14.0")
    )


def _example_host_scenario(start_date: datetime) -> Scenario:
    return Scenario([
        # ParameterChangePoint("r", start_date + timedelta(days=8), "0.65 * r"),
    ])


def _example_vector_scenario(start_date: datetime) -> Scenario:
    return Scenario([
        # ParameterChangePoint("R0", start_date + timedelta(days=8), "0.80 * R0"),
    ])


def main() -> None:
    host_obs = [
        (datetime(2009, 7, 28), 1),
        (datetime(2009, 8, 4), 2),
        (datetime(2009, 8, 10), 4),
        (datetime(2009, 8, 15), 5),
        (datetime(2009, 8, 22), 6),
        (datetime(2009, 8, 29), 4),
        (datetime(2009, 9, 5), 3),
        (datetime(2009, 9, 12), 2),
    ]

    host_obs_points = sorted(host_obs, key=lambda x: x[0])
    start_date = host_obs_points[0][0]

    config = SnapshotConfig(
        # counts=list(range(2, len(host_obs_points) + 1)),
        counts=list(range(len(host_obs_points), len(host_obs_points) + 1)),
        # counts=(6,),
        global_opt=OptimizerConfig(
            method="differential_evolution",
            kwargs={
                "strategy": "best1bin",
                "max_nfev": 500,
                "popsize": 10,
                "workers": 1,
                "polish": False,
            },
        ),
        local_opt=None,
    )

    best_kwargs, all_artifacts = optimize_snapshots_alternating_host_observed(
        sim_start=start_date,
        all_host_obs_points=host_obs_points,
        initial_kwargs=default_acceptance_kwargs(),
        host_parameters=_example_host_parameters(),
        vector_parameters=_example_vector_parameters(),
        host_scenario=_example_host_scenario(start_date),
        vector_scenario=_example_vector_scenario(start_date),
        snapshot_config=config,
        host_base_criteria=[IndexOffspringCriterion(1, 5)],
        vector_base_criteria=[],
        base_log_path=str(REPO_ROOT / "optimization_logs" / "host_vector_host_dengue_florida_control"),
    )

    print("\nFinal optimized host-only acceptance kwargs")
    for key, value in best_kwargs.items():
        print(f"{key:25s}: {value}")
    last_snapshot = max(all_artifacts)
    print(f"Accepted in final run: {all_artifacts[last_snapshot]['accepted_count']}")
    print(f"Dynamic T_run: {all_artifacts[last_snapshot]['T_run']}")


if __name__ == "__main__":
    main()
