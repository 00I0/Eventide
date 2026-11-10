"""Plot saved alternating acceptance searches without repeating optimization.

Run from any directory with --list to inspect available scenario metadata.
The JSON configuration records observations, priors and controls separately
from optimization CSVs, which contain only acceptance-window settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammainc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.alternating_extinction_probability import (
    ParameterChange, conditional_extinction_probability, parameters_at,
)
from python.eventide import (
    ActiveSetSizeCollector, AlternatingSimulator, DrawCollector,
    IndexOffspringCriterion, ParameterChangePoint, Parameters, Scenario, SpeciesConfiguration,
)
from python.eventide.criterion import Criterion
from python.optimize_acceptance_windows import build_acceptance_inequalities
from python.plots.sae_alternating import plot_latest_cumulative, plot_latest_posteriors, plot_online_extinction


def load_best_settings(path, minimum_accepted=80):
    """Select the lowest finite objective among sufficiently accepted trials."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    eligible = []
    for row in rows:
        try:
            objective, accepted = float(row["objective"]), int(float(row["accepted"]))
        except (KeyError, ValueError, TypeError):
            continue
        if np.isfinite(objective) and accepted >= minimum_accepted:
            eligible.append((objective, row))
    if not eligible:
        raise ValueError(f"{path}: no finite trial with at least {minimum_accepted} accepted histories.")
    objective, row = min(eligible, key=lambda item: item[0])
    integer_fields = {"kmax", "max_unions_to_keep"}
    boolean_fields = {"include_gap_windows", "include_union_windows", "include_global_total"}
    settings = {}
    for key, value in row.items():
        if key in ("objective", "accepted"):
            continue
        if key in boolean_fields:
            if value.strip().lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"Invalid boolean {key}={value!r} in {path}")
            settings[key] = value.strip().lower() in ("true", "1")
        elif key in integer_fields:
            settings[key] = int(round(float(value)))
        else:
            settings[key] = float(value)
    return settings, {"objective": objective, "accepted": int(float(row["accepted"]))}


def _changes(spec):
    return tuple(ParameterChange(**change) for change in spec.get("changes", []))


def _parameters(spec):
    parameters = Parameters(**{key: tuple(value) for key, value in spec["ranges"].items()})
    for requirement in spec.get("requirements", []):
        parameters.require(requirement)
    return parameters


def _scenario(changes, start):
    return Scenario([
        ParameterChangePoint(change.parameter, start + timedelta(days=change.day),
                             f"{change.factor:.17g} * {change.parameter}")
        for change in changes
    ])


def _pack_pairs(histories):
    arrays = [np.asarray(history, dtype=float).reshape(-1, 2) for history in histories]
    offsets = np.concatenate(([0], np.cumsum([len(history) for history in arrays])))
    return np.concatenate(arrays) if arrays else np.empty((0, 2)), offsets


def _unpack_pairs(values, offsets):
    return [values[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]


def run_snapshot(spec, observations, snapshot, log_path, args):
    bounds = spec.get("index_offspring_bounds")
    start = observations[0][0]
    subset = observations[:snapshot]
    cutoff_date = subset[-1][0] + timedelta(days=1)
    cutoff = (cutoff_date - start).total_seconds() / 86400.0
    settings, selected = load_best_settings(log_path, args.minimum_log_accepted)
    fingerprint = {
        "version": 1, "scenario": {**spec, "index_offspring_bounds": bounds},
        "snapshot": snapshot, "settings": settings,
        "accepted": args.accepted, "attempts": args.attempts, "max_cases": args.max_cases,
        "extension": hashlib.sha256(
            Path(str(sys.modules["python.eventide._eventide"].__file__)).read_bytes()).hexdigest(),
        "builder": hashlib.sha256((REPO_ROOT / "python/optimize_acceptance_windows.py").read_bytes()).hexdigest(),
        "wrappers": {name: hashlib.sha256((REPO_ROOT / "python/eventide" / name).read_bytes()).hexdigest()
                     for name in ("criterion.py", "scenario.py", "alternating.py")},
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
    cache_path = args.output / "cache" / f"{spec['key']}_snapshot_{snapshot}_{digest}.npz"
    loaded_cache = cache_path.exists() and not args.refresh
    result: dict[str, Any]
    if loaded_cache:
        with np.load(cache_path, allow_pickle=False) as saved:
            result = {"host_draws": saved["host_draws"], "vector_draws": saved["vector_draws"],
                      "host_pairs": _unpack_pairs(saved["host_pairs"], saved["host_offsets"]),
                      "vector_pairs": _unpack_pairs(saved["vector_pairs"], saved["vector_offsets"]),
                      "processed": int(saved["processed"]), "seconds": float(saved["seconds"])}
        print(f"  snapshot {snapshot}: cached {len(result['host_draws'])} histories", flush=True)
    else:
        criteria: list[Criterion] = list(build_acceptance_inequalities(
            subset, simulation_start=start, observation_end=cutoff_date,
            index_case_count=1, include_terminal_window=True, **settings,
        ))
        if bounds is not None:
            criteria.append(IndexOffspringCriterion(*bounds))
        host_pairs, vector_pairs = ActiveSetSizeCollector(cutoff_date), ActiveSetSizeCollector(cutoff_date)
        host_draws, vector_draws = DrawCollector(), DrawCollector()
        host_parameters, vector_parameters = _parameters(spec["host"]), _parameters(spec["vector"])
        simulation = AlternatingSimulator(
            host=SpeciesConfiguration(host_parameters, host_parameters.create_latin_hypercube_sampler(),
                                      _scenario(_changes(spec["host"]), start), criteria, [host_pairs, host_draws]),
            vector=SpeciesConfiguration(vector_parameters, vector_parameters.create_latin_hypercube_sampler(),
                                        _scenario(_changes(spec["vector"]), start), [], [vector_pairs, vector_draws]),
            start_date=start, num_trajectories=args.attempts, min_required=args.accepted,
            chunk_size=args.chunk_size, T_run=cutoff, max_cases=args.max_cases, max_workers=args.workers,
        )
        print(f"  snapshot {snapshot}: simulating through day {cutoff:g}", flush=True)
        began = time.perf_counter()
        simulation.run()
        result = {"host_draws": np.asarray(host_draws), "vector_draws": np.asarray(vector_draws),
                  "host_pairs": host_pairs.active_sets, "vector_pairs": vector_pairs.active_sets,
                  "processed": simulation.processed, "seconds": time.perf_counter() - began}
        accepted = len(result["host_draws"])
        if accepted == 0:
            raise RuntimeError(
                f"{spec['key']} snapshot {snapshot}: no histories accepted in {simulation.processed} attempts.")
        if any(len(host) + len(vector) >= args.max_cases
               for host, vector in zip(result["host_pairs"], result["vector_pairs"])):
            raise RuntimeError("A saved history hit max_cases and may be incomplete. Increase --max-cases.")
        if accepted < args.accepted:
            print(f"  WARNING: only {accepted}/{args.accepted} histories; increase --attempts for the final figure.",
                  flush=True)
        packed_host, host_offsets = _pack_pairs(result["host_pairs"])
        packed_vector, vector_offsets = _pack_pairs(result["vector_pairs"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, host_draws=result["host_draws"], vector_draws=result["vector_draws"],
                            host_pairs=packed_host, host_offsets=host_offsets,
                            vector_pairs=packed_vector, vector_offsets=vector_offsets,
                            processed=result["processed"], seconds=result["seconds"])
        print(f"  accepted {accepted} / {simulation.processed} in {result['seconds']:.2f}s", flush=True)
    result.update(snapshot=snapshot, cutoff=cutoff, cutoff_date=cutoff_date, selected=selected,
                  settings=settings, log_path=str(log_path), cache_path=str(cache_path))
    if len(result["host_draws"]) < args.accepted and loaded_cache:
        print(
            f"  WARNING: {len(result['host_draws'])}/{args.accepted} histories available; increase --attempts for the final figure.",
            flush=True)
    began = time.perf_counter()
    result["curves"] = conditional_extinction_probability(
        result["host_pairs"], result["vector_pairs"], result["host_draws"], result["vector_draws"],
        args.horizons, cutoff, index_offspring_bounds=bounds,
        host_changes=_changes(spec["host"]), vector_changes=_changes(spec["vector"]),
        quadrature_order=args.quadrature_order,
        workers=args.workers,
    )
    result["analysis_seconds"] = time.perf_counter() - began
    print(f"  probabilities calculated in {result['analysis_seconds']:.2f}s", flush=True)
    return result


def cycle_timing_density(host_params, vector_params, grid):
    """Posterior mixture of the sum of two Gamma intervals, using FFTs."""
    edges = np.arange(len(grid) + 1) * (grid[1] - grid[0])
    host_mass = np.diff(gammainc(host_params[:, 3, None], edges / host_params[:, 4, None]), axis=1)
    vector_mass = np.diff(gammainc(vector_params[:, 3, None], edges / vector_params[:, 4, None]), axis=1)
    mass = fftconvolve(host_mass, vector_mass, mode="full", axes=1)
    density = np.mean(mass, axis=0) / (grid[1] - grid[0])
    return np.interp(grid, edges[1] + np.arange(len(density)) * edges[1], density, left=0)


def plot_scenario(spec, observations, results, args):
    effective_parameters = []
    for result in results:
        host = parameters_at(result["host_draws"], np.full(len(result["host_draws"]), result["cutoff"]),
                             _changes(spec["host"]))
        vector = parameters_at(result["vector_draws"], np.full(len(result["vector_draws"]), result["cutoff"]),
                               _changes(spec["vector"]))
        effective_parameters.append((host, vector))
    summary: dict[str, Any] = {
        "scenario": {**spec, "index_offspring_bounds": spec.get("index_offspring_bounds")}, "snapshots": [],
    }
    for result, (host, vector) in zip(results, effective_parameters):
        reproduction = np.sqrt(host[:, 0] * host[:, 2] * vector[:, 0] * vector[:, 2])
        summary["snapshots"].append({
            "snapshot": result["snapshot"], "cutoff": result["cutoff"], "accepted": len(host),
            "processed": result["processed"], "simulation_seconds": result["seconds"],
            "analysis_seconds": result["analysis_seconds"],
            "selected_trial": result["selected"], "acceptance_settings": result["settings"],
            "log_path": result["log_path"], "cache_path": result["cache_path"],
            "cycle_reproduction_quantiles": np.quantile(reproduction, [.025, .5, .975]).tolist(),
            "mean_host_to_host_interval_quantiles": np.quantile(host[:, 3] * host[:, 4] + vector[:, 3] * vector[:, 4],
                                                                [.025, .5, .975]).tolist(),
            "horizons": args.horizons.tolist(),
            "host_cessation": [float(value) if np.isfinite(value) else None for value in result["curves"]["host"]],
            "all_transmission_cessation": [float(value) if np.isfinite(value) else None for value in
                                           result["curves"]["all"]],
        })
    latest = results[-1]
    host, vector = effective_parameters[-1]
    reproduction = np.sqrt(host[:, 0] * host[:, 2] * vector[:, 0] * vector[:, 2])
    timing_limit = float(np.quantile(
        host[:, 3] * host[:, 4] + vector[:, 3] * vector[:, 4]
        + 6 * np.sqrt(host[:, 3] * host[:, 4] ** 2 + vector[:, 3] * vector[:, 4] ** 2), .995,
    ))
    timing_grid = np.linspace(0, timing_limit, 768)
    posterior = plot_latest_posteriors(reproduction, float(np.max(reproduction)), timing_grid,
                                       cycle_timing_density(host, vector, timing_grid), language=args.language)
    cumulative = plot_latest_cumulative(latest["host_pairs"], observations[:latest["snapshot"]],
                                        observations[0][0], latest["cutoff"], language=args.language)
    extinction = plot_online_extinction(
        [{**result, "horizons": args.horizons} for result in results], observations,
        final_horizon=args.final_horizon, language=args.language,
    )
    for figure, name in ((posterior, "posterior"), (cumulative, "cumulative"), (extinction, "extinction")):
        for extension in ("pdf", "png"):
            figure.savefig(args.output / f"{spec['key']}_{name}.{extension}", dpi=300)
        plt.close(figure)
    (args.output / f"{spec['key']}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("alternating_scenarios.json"))
    parser.add_argument("--logs", type=Path, default=REPO_ROOT / "optimization_logs/alternating_quick",
                        help="Saved optimizer CSV directory (default: optimization_logs/alternating_quick).")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "plots/alternating_optimized",
                        help="Figure, summary and history-cache directory (default: plots/alternating_optimized).")
    parser.add_argument("--scenario", action="append", help="Scenario key; repeat to select several.")
    parser.add_argument("--snapshots", nargs="+", type=int, help="Only these observation counts.")
    parser.add_argument("--list", action="store_true", help="List log groups and configuration readiness.")
    parser.add_argument("--accepted", type=int, default=2000)
    parser.add_argument("--attempts", type=int, default=100_000_000_000)
    parser.add_argument("--workers", type=int, default=min(13, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-cases", type=int, default=400)
    parser.add_argument("--minimum-log-accepted", type=int, default=80)
    parser.add_argument("--horizon", type=float, default=90)
    parser.add_argument("--step", type=float, default=1)
    parser.add_argument("--quadrature-order", type=int, default=48)
    parser.add_argument("--final-horizon", type=float, default=40,
                        help="Days displayed after the final snapshot in the SAE calendar-date extinction plot.")
    parser.add_argument("--language", choices=("hu", "en"), default="hu",
                        help="Language for all figures; hu matches plot_sae's Hungarian wording.")
    parser.add_argument("--refresh", action="store_true", help="Rerun simulations instead of using cached histories.")
    args = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0 for value in (
            args.accepted, args.attempts, args.workers, args.chunk_size, args.max_cases, args.horizon, args.step,
    )):
        parser.error("Simulation sizes, horizon and step must be finite and positive.")
    if args.quadrature_order < 8 or args.minimum_log_accepted < 0:
        parser.error("Use --quadrature-order >= 8 and --minimum-log-accepted >= 0.")
    if not np.isfinite(args.final_horizon) or not 0 < args.final_horizon <= args.horizon:
        parser.error("Use 0 < --final-horizon <= --horizon.")
    args.horizons = np.unique(np.append(np.arange(0, args.horizon, args.step), args.horizon))
    specifications = json.loads(args.config.read_text())["scenarios"]
    keys = {spec["key"] for spec in specifications}
    if args.scenario and not set(args.scenario) <= keys:
        parser.error(f"Unknown scenario keys: {sorted(set(args.scenario) - keys)}")
    matched_logs = set()
    failures = []
    args.output.mkdir(parents=True, exist_ok=True)
    for spec in specifications:
        logs = sorted(Path(args.logs).glob(spec["log_pattern"]), key=lambda candidate_path: candidate_path.name)
        matched_logs.update(logs)
        missing = [key for key in ("observations", "host", "vector") if spec.get(key) is None]
        if args.list:
            print(f"{spec['key']}: {len(logs)} logs; " + (f"missing {', '.join(missing)}" if missing else "ready"))
            if spec.get("notes"):
                print(f"  {spec['notes']}")
            continue
        if args.scenario and spec["key"] not in args.scenario:
            continue
        if not logs:
            message = f"{spec['key']}: no optimized results in {args.logs}"
            if args.scenario:
                failures.append(message)
            else:
                print(f"SKIP: {message}")
            continue
        if missing:
            failures.append(f"{spec['key']}: missing {', '.join(missing)}; complete {args.config}")
            continue
        observations = [(datetime.fromisoformat(date), int(count)) for date, count in spec["observations"]]
        if not observations or observations != sorted(observations) or any(count <= 0 for _, count in observations):
            raise ValueError(f"{spec['key']}: observations must be positive daily incidence, ordered by date.")
        if len({date for date, _ in observations}) != len(observations) or any(
                date.time() != datetime.min.time() for date, _ in observations):
            raise ValueError("Use one midnight timestamp per observation day; aggregate repeated dates first.")
        if "changes" not in spec["host"] or "changes" not in spec["vector"]:
            raise ValueError(
                f"{spec['key']}: explicitly specify changes for each species (empty lists mean no controls).")
        print(f"\n{spec['title']}", flush=True)
        if spec.get("notes"):
            print(spec["notes"], flush=True)
        results = []
        for log_path in logs:
            match = re.search(r"_snapshot_(\d+)\.csv$", log_path.name)
            snapshot = int(match.group(1)) if match else len(observations)
            if args.snapshots and snapshot not in args.snapshots:
                continue
            if snapshot > len(observations):
                failures.append(f"{log_path.name}: snapshot exceeds available observations")
                continue
            try:
                results.append(run_snapshot(spec, observations, snapshot, log_path, args))
            except (ValueError, RuntimeError) as error:
                failures.append(f"{spec['key']} snapshot {snapshot}: {error}")
        if results:
            plot_scenario(spec, observations, sorted(results, key=lambda item: item["snapshot"]), args)
        else:
            failures.append(f"{spec['key']}: no snapshots plotted")
    for path in sorted(set(args.logs.glob("host_vector*.csv")) - matched_logs):
        print(f"Unconfigured log: {path.name}")
    for failure in failures:
        print(f"INCOMPLETE: {failure}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
