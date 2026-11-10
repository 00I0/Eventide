"""Quick, resumable CMA-ES acceptance-window search for all alternating examples.

The stopping target is relative L2 error, ||mean cumulative - observed||_2 /
||observed||_2, not absolute MSE or an absolute number of cases. Default: 5%.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import inspect
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.eventide import AlternatingSimulator, IndexOffspringCriterion, InfectionTimeCollector, SpeciesConfiguration
from python.optimize_acceptance_windows import build_acceptance_inequalities, mean_cumulative_from_infections
from python.optimize_acceptance_windows_host_vector import default_acceptance_kwargs, _cast_accept_kwargs
from python.plot_alternating import _changes, _parameters, _scenario, load_best_settings

PARAM_NAMES = tuple(default_acceptance_kwargs())
PARAM_BOUNDS = np.array([
    [.25, 6], [.50, .99], [.10, 3.5], [.10, .75], [.50, 5], [2, 8],
    [.01, .35], [.01, .40], [.10, 1], [1e-5, .10], [0, 1], [0, 1], [0, 5], [.10, .90], [0, 1],
])
BOOLEAN_FIELDS = {"include_gap_windows", "include_union_windows", "include_global_total"}
DEFAULT_RELATIVE_TOLERANCE = .05


def encode_state(value):
    if isinstance(value, np.ndarray):
        return {"array": base64.b64encode(value.tobytes()).decode(), "dtype": value.dtype.str,
                "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: encode_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_state(item) for item in value]
    return value


def decode_state(value):
    if isinstance(value, dict):
        if "array" in value:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise ValueError("Object arrays are not permitted in optimizer checkpoints.")
            return np.frombuffer(base64.b64decode(value["array"]), dtype=dtype).reshape(value["shape"]).copy()
        return {key: decode_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_state(item) for item in value]
    return value


def decode_vector(vector):
    values = PARAM_BOUNDS[:, 0] + np.clip(vector, 0, 1) * np.ptp(PARAM_BOUNDS, axis=1)
    raw = dict(zip(PARAM_NAMES, values))
    for name in BOOLEAN_FIELDS:
        raw[name] = bool(raw[name] >= .5)
    return _cast_accept_kwargs(raw)


def encode_params(params):
    values = np.array([params[name] for name in PARAM_NAMES], dtype=float)
    return np.clip((values - PARAM_BOUNDS[:, 0]) / np.ptp(PARAM_BOUNDS, axis=1), 0, 1)


def relative_l2_error(predicted, observed):
    predicted, observed = np.asarray(predicted, dtype=float), np.asarray(observed, dtype=float)
    if predicted.shape != observed.shape or observed.size == 0 or not np.all(np.isfinite(observed)):
        raise ValueError("Cumulative curves must be nonempty, finite and aligned.")
    scale = np.linalg.norm(observed)
    if scale <= 0 or not np.all(np.isfinite(predicted)):
        raise ValueError("Relative error requires positive observed norm and finite predictions.")
    return float(np.linalg.norm(predicted - observed) / scale)


def select_snapshots(count, mode="smart"):
    if mode == "final":
        return [count]
    if mode == "all" or count <= 6:
        return list(range(1, count + 1))
    return sorted({int(value) for value in np.linspace(1, count, min(6, count), dtype=int)} | {count})


def load_observations(spec):
    observations = [(datetime.fromisoformat(date), int(count)) for date, count in spec["observations"]]
    dates = [date for date, _ in observations]
    if (not observations or dates != sorted(set(dates))
            or any(date.time() != datetime.min.time() or count <= 0 for date, count in observations)):
        raise ValueError(f"{spec['key']}: require ordered, unique daily dates and positive incidence.")
    return observations


def build_snapshot_criteria(spec, observations, params):
    cutoff_date = observations[-1][0] + timedelta(days=1)
    windows = build_acceptance_inequalities(
        observations, simulation_start=observations[0][0], observation_end=cutoff_date,
        index_case_count=1, include_terminal_window=True, **params,
    )
    criteria = list(windows)
    if spec.get("index_offspring_bounds") is not None:
        criteria.append(IndexOffspringCriterion(*spec["index_offspring_bounds"]))
    return windows, criteria, cutoff_date


@dataclass
class EvalResult:
    objective: float
    relative_error: float | None
    accepted_count: int
    processed: int
    meets_minimum: bool
    params: dict
    capped: bool = False


class FastAlternatingOptimizer:
    def __init__(self, spec, observations, *, min_accepted=100, max_trajectories=200_000_000,
                 chunk_size=10_000, workers=13, max_cases=300, relative_tolerance=DEFAULT_RELATIVE_TOLERANCE):
        self.spec = spec
        self.observations = observations
        self.start = observations[0][0]
        self.min_accepted = min_accepted
        self.max_trajectories = max_trajectories
        self.chunk_size = chunk_size
        self.workers = workers
        self.max_cases = max_cases
        self.relative_tolerance = relative_tolerance
        self.host_parameters = _parameters(spec["host"])
        self.vector_parameters = _parameters(spec["vector"])
        self.host_scenario = _scenario(_changes(spec["host"]), self.start)
        self.vector_scenario = _scenario(_changes(spec["vector"]), self.start)
        self.times_eval = [date + timedelta(days=1) for date, _ in observations]
        self.observed = np.cumsum([count for _, count in observations])
        self.memo = {}
        self.best = None
        self.evaluations = 0
        self.simulations = 0

    def evaluate(self, params):
        params = _cast_accept_kwargs(params)
        windows, criteria, cutoff_date = build_snapshot_criteria(self.spec, self.observations, params)
        signature = tuple((window.start_date, window.end_date, window.min_cases, window.max_cases)
                          for window in windows)
        self.evaluations += 1
        if signature in self.memo:
            return replace(self.memo[signature], params=params)
        host_times, vector_times = InfectionTimeCollector(), InfectionTimeCollector()
        simulation = AlternatingSimulator(
            host=SpeciesConfiguration(self.host_parameters, self.host_parameters.create_latin_hypercube_sampler(),
                                      self.host_scenario, criteria, [host_times]),
            vector=SpeciesConfiguration(self.vector_parameters, self.vector_parameters.create_latin_hypercube_sampler(),
                                        self.vector_scenario, [], [vector_times]),
            start_date=self.start, T_run=(cutoff_date - self.start).total_seconds() / 86400,
            num_trajectories=self.max_trajectories, min_required=self.min_accepted,
            chunk_size=self.chunk_size, max_cases=self.max_cases, max_workers=self.workers,
        )
        simulation.run()
        self.simulations += 1
        accepted = len(host_times.infection_times)
        capped = any(len(host) + len(vector) >= self.max_cases
                     for host, vector in zip(host_times.infection_times, vector_times.infection_times))
        feasible = accepted >= self.min_accepted and not capped
        error = None
        if feasible:
            predicted = mean_cumulative_from_infections(self.times_eval, host_times.infection_times, self.start)
            error = relative_l2_error(predicted, self.observed)
        objective = error if feasible else 1e4 + max(0, self.min_accepted - accepted) / self.min_accepted + int(capped)
        result = EvalResult(objective, error, accepted, simulation.processed, feasible, params, capped)
        self.memo[signature] = result
        if feasible and (self.best is None or result.objective < self.best.objective):
            self.best = result
        return result

    def reached_target(self, result):
        return result.meets_minimum and result.relative_error <= self.relative_tolerance

    def optimize(self, *, generations=30, population=12, seed=42, warm_seed=None, on_evaluation=None,
                 resume_state=None, on_checkpoint=None):
        try:
            from cmaes import CMA
        except ImportError as error:
            raise RuntimeError("The quick optimizer requires cmaes in the project interpreter.") from error
        params = warm_seed if warm_seed is not None else default_acceptance_kwargs()
        from importlib.metadata import version
        cma_version = version("cmaes")

        def evaluate(params):
            result = self.evaluate(params)
            if on_evaluation:
                on_evaluation(result)
            return result

        optimizer = None
        solutions = []

        def checkpoint(last):
            if on_checkpoint:
                on_checkpoint({
                    "cma_version": cma_version, "initial": asdict(initial),
                    "best": asdict(self.best) if self.best is not None else None,
                    "last": asdict(last), "evaluations": self.evaluations, "simulations": self.simulations,
                    "optimizer": encode_state(optimizer.__getstate__()) if optimizer is not None else None,
                    "rng": encode_state(optimizer._rng.get_state()) if optimizer is not None else None,
                    "solutions": [[vector.tolist(), objective] for vector, objective in solutions],
                })

        if resume_state is not None:
            if resume_state["cma_version"] != cma_version:
                raise ValueError("CMA-ES checkpoint version differs from the installed cmaes version.")
            initial = EvalResult(**resume_state["initial"])
            self.best = EvalResult(**resume_state["best"]) if resume_state["best"] is not None else None
            self.evaluations = resume_state["evaluations"]
            self.simulations = resume_state["simulations"]
            last = EvalResult(**resume_state["last"])
            if resume_state["optimizer"] is not None:
                optimizer = CMA.__new__(CMA)
                optimizer.__setstate__(decode_state(resume_state["optimizer"]))
                optimizer._rng.set_state(tuple(decode_state(resume_state["rng"])))
                solutions = [(np.asarray(vector), objective) for vector, objective in resume_state["solutions"]]
            if self.best is not None and self.reached_target(self.best):
                return self.best
        else:
            initial = evaluate(params)
            last = initial
            checkpoint(last)
            if self.reached_target(initial):
                return initial
        if optimizer is None:
            optimizer = CMA(mean=encode_params(initial.params), sigma=.2,
                            bounds=np.tile([0., 1.], (len(PARAM_NAMES), 1)), population_size=population, seed=seed)
        while optimizer.generation < generations:
            while len(solutions) < population:
                vector = optimizer.ask()
                result = evaluate(decode_vector(vector))
                last = result
                solutions.append((vector, result.objective))
                checkpoint(last)
                if self.reached_target(result):
                    return result
            optimizer.tell(solutions)
            solutions = []
            checkpoint(last)
            if optimizer.should_stop():
                break
        return self.best or initial


def context_fingerprint(spec, snapshot, args):
    from python.eventide import _eventide
    paths = [Path(_eventide.__file__), REPO_ROOT / "python/optimize_acceptance_windows.py"]
    paths.extend(REPO_ROOT / "python/eventide" / name for name in ("criterion.py", "scenario.py", "alternating.py"))
    model = {key: spec[key] for key in ("host", "vector")}
    model.update(observations=spec["observations"][:snapshot], index_offspring_bounds=spec.get("index_offspring_bounds"))
    functions = (build_snapshot_criteria, relative_l2_error, FastAlternatingOptimizer.evaluate,
                 _changes, _parameters, _scenario, _cast_accept_kwargs)
    context = {"version": 2, "model": model, "max_cases": args.max_cases,
               "functions": [inspect.getsource(function) for function in functions],
               "sources": {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in paths}}
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("alternating_scenarios.json"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "optimization_logs/alternating_quick")
    parser.add_argument("--scenario", action="append", help="Repeat to select cases; default: every complete case.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show the batch plan without simulations.")
    parser.add_argument("--snapshot-mode", choices=("smart", "all", "final"), default="smart")
    parser.add_argument("--all-snapshots", action="store_true")
    parser.add_argument("--snapshots", nargs="+", type=int)
    parser.add_argument("--relative-tolerance", "--target-tol", type=float, default=DEFAULT_RELATIVE_TOLERANCE,
                        help="Relative L2 stopping target as a fraction; default .05 (5%%), never absolute MSE.")
    parser.add_argument("--min-accepted", type=int, default=200)
    parser.add_argument("--num-trajectories", type=int, default=200_000_000)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=min(13, os.cpu_count() or 1))
    parser.add_argument("--max-cases", type=int, default=300)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42, help="CMA-ES seed; simulator randomness is independent.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Start fresh instead of resuming saved results (default: false).")
    parser.add_argument("--continue-unfinished", action="store_true", default=False,
                        help="Continue saved searches that missed the target; default: skip saved results.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not np.isfinite(args.relative_tolerance) or not 0 < args.relative_tolerance < 1:
        parser.error("--relative-tolerance must be a fraction between 0 and 1; .05 means 5%.")
    if any(getattr(args, name) <= 0 for name in (
            "min_accepted", "num_trajectories", "chunk_size", "workers", "max_cases")):
        parser.error("Simulation budgets, worker count and case cap must be positive.")
    if args.generations < 0 or args.population < 2 or args.seed < 0:
        parser.error("Require nonnegative generations/seed and population >= 2.")
    specs = json.loads(args.config.read_text())["scenarios"]
    unknown = set(args.scenario or []) - {spec["key"] for spec in specs}
    if unknown:
        parser.error(f"Unknown scenarios: {', '.join(sorted(unknown))}")
    print(f"Relative L2 target: {args.relative_tolerance:.1%}; minimum accepted: {args.min_accepted}")
    failed = []
    for spec in specs:
        if args.scenario and spec["key"] not in args.scenario:
            continue
        missing = [name for name in ("observations", "host", "vector") if spec.get(name) is None]
        if missing:
            print(f"SKIP {spec['key']}: missing {', '.join(missing)}")
            if args.scenario:
                failed.append(spec["key"])
            continue
        observations = load_observations(spec)
        snapshots = sorted(set(args.snapshots)) if args.snapshots else select_snapshots(
            len(observations), "all" if args.all_snapshots else args.snapshot_mode,
        )
        if any(snapshot < 1 or snapshot > len(observations) for snapshot in snapshots):
            parser.error(f"{spec['key']}: snapshots must be within 1..{len(observations)}.")
        print(f"{spec['key']}: {sum(count for _, count in observations)} cases; snapshots {snapshots}; "
              f"index bounds={spec.get('index_offspring_bounds')}")
        if args.list or args.dry_run:
            continue
        args.output.mkdir(parents=True, exist_ok=True)
        cache_path = args.output / f"{spec['key']}_optimal.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {"scenario": spec, "snapshots": {}}
        cache["scenario"] = spec
        warm_seed = None
        with tqdm(total=len(snapshots), desc=spec["key"], unit="snap", dynamic_ncols=True, leave=True) as progress:
            for snapshot in snapshots:
                fingerprint = context_fingerprint(spec, snapshot, args)
                cached = cache["snapshots"].get(str(snapshot), {})
                valid = cached.get("fingerprint") == fingerprint and cached.get("meets_minimum")
                log_name = spec["log_pattern"].replace("*", str(snapshot))
                if "*" not in spec["log_pattern"]:
                    log_name = f"host_vector_host_{spec['key']}_snapshot_{snapshot}.csv"
                log_path = args.output / log_name
                checkpoint_path = args.output / f"{spec['key']}_snapshot_{snapshot}.checkpoint.json"
                search_settings = {"min_accepted": args.min_accepted, "num_trajectories": args.num_trajectories,
                                   "population": args.population, "seed": args.seed}
                saved_search = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
                resume_state = None
                if (not args.force and saved_search.get("fingerprint") == fingerprint
                        and saved_search.get("settings") == search_settings):
                    from importlib.metadata import version
                    if saved_search["state"]["cma_version"] == version("cmaes"):
                        resume_state = saved_search["state"]
                if (cached.get("params") and not args.force
                        and (not args.continue_unfinished or cached.get("fingerprint") == fingerprint)):
                    warm_seed = cached["params"]
                    meets_target = (cached.get("meets_minimum") and cached["accepted_count"] >= args.min_accepted
                                    and cached["relative_error"] <= args.relative_tolerance)
                    if meets_target or not args.continue_unfinished:
                        if not log_path.exists():
                            with log_path.open("w", newline="") as stream:
                                writer = csv.DictWriter(stream, fieldnames=("objective", "accepted", *PARAM_NAMES))
                                writer.writeheader()
                                writer.writerow(
                                    {"objective": cached["objective"],
                                     "accepted": cached["accepted_count"] if cached["meets_minimum"] else 0,
                                     **cached["params"]})
                        error_label = (f"rel={cached['relative_error']:.2%}"
                                       if cached.get("relative_error") is not None else "infeasible")
                        status = "cached" if meets_target else "saved; target not met, skipped"
                        if cached.get("fingerprint") != fingerprint:
                            status += "; older cache reused"
                        progress.set_postfix_str(f"m={snapshot}/{len(observations)} [{status}] "
                                                 f"{error_label} acc={cached['accepted_count']}")
                        progress.update(1)
                        continue
                if not args.force and resume_state is None:
                    seed_error = float("inf")
                    if cached.get("params"):
                        warm_seed = cached["params"]
                        if cached.get("meets_minimum") and cached["accepted_count"] >= args.min_accepted:
                            seed_error = cached["relative_error"]
                    if log_path.exists():
                        try:
                            log_seed, selected = load_best_settings(log_path, args.min_accepted)
                            if selected["objective"] < seed_error:
                                warm_seed = log_seed
                        except ValueError:
                            pass
                action = "resuming checkpoint" if resume_state is not None else (
                    "revalidating saved settings" if cached or log_path.exists() else "starting CMA-ES"
                )
                progress.set_postfix_str(f"m={snapshot}/{len(observations)} {action}...")
                append_log = not args.force and (valid or resume_state is not None) and log_path.exists()
                if log_path.exists() and not append_log:
                    log_path.rename(log_path.with_name(f"{log_path.name}.previous-{time.time_ns()}"))
                if args.force:
                    checkpoint_path.unlink(missing_ok=True)
                began = time.perf_counter()
                optimizer = FastAlternatingOptimizer(
                    spec, observations[:snapshot], min_accepted=args.min_accepted,
                    max_trajectories=args.num_trajectories,
                    chunk_size=args.chunk_size, workers=args.workers, max_cases=args.max_cases,
                    relative_tolerance=args.relative_tolerance,
                )
                with log_path.open("a" if append_log else "w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=("objective", "accepted", *PARAM_NAMES))
                    if not append_log:
                        writer.writeheader()

                    def record(result):
                        writer.writerow({"objective": result.objective,
                                         "accepted": result.accepted_count if result.meets_minimum else 0,
                                         **result.params})
                        stream.flush()
                        best = optimizer.best or result
                        error_label = f"rel={best.relative_error:.2%}" if best.meets_minimum else "seeking_viable"
                        generation = 0 if optimizer.evaluations == 1 else (
                                                                                  optimizer.evaluations - 2) // args.population + 1
                        progress.set_postfix_str(
                            f"m={snapshot}/{len(observations)} gen={generation}/{args.generations} "
                            f"{error_label} acc={best.accepted_count} evals={optimizer.evaluations}",
                        )

                    def checkpoint(state):
                        save_json(checkpoint_path, {"version": 1, "fingerprint": fingerprint,
                                                   "settings": search_settings, "state": state})
                        best = state["best"] or state["last"]
                        cache["snapshots"][str(snapshot)] = {
                            **best, "fingerprint": fingerprint, "relative_tolerance": args.relative_tolerance,
                            "target_met": best["meets_minimum"] and best["relative_error"] <= args.relative_tolerance,
                            "error_definition": "norm(mean_cumulative-observed)/norm(observed)",
                            "evaluations": state["evaluations"], "simulations": state["simulations"],
                            "seconds": time.perf_counter() - began, "complete": False,
                        }
                        save_json(cache_path, cache)

                    result = optimizer.optimize(generations=args.generations, population=args.population,
                                                seed=args.seed,
                                                warm_seed=warm_seed, on_evaluation=record,
                                                resume_state=resume_state, on_checkpoint=checkpoint)
                success = optimizer.reached_target(result)
                cache["snapshots"][str(snapshot)] = {
                    **asdict(result), "fingerprint": fingerprint, "relative_tolerance": args.relative_tolerance,
                    "target_met": success, "error_definition": "norm(mean_cumulative-observed)/norm(observed)",
                    "evaluations": optimizer.evaluations, "simulations": optimizer.simulations,
                    "seconds": time.perf_counter() - began,
                    "complete": True,
                }
                save_json(cache_path, cache)
                if result.meets_minimum:
                    warm_seed = result.params
                if not success:
                    failed.append(f"{spec['key']} snapshot {snapshot}")
                    progress.set_postfix_str(f"m={snapshot}/{len(observations)} target not reached "
                                             f"acc={result.accepted_count} evals={optimizer.evaluations}")
                progress.update(1)
    if failed:
        print("Target not reached (increase search/simulation budgets): " + "; ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
