# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import floor, ceil, log, sqrt
from pathlib import Path
from typing import List, Tuple, Sequence, Optional, Dict, Any

import numpy as np
import pandas as pd
from lmfit import Parameters as LMParams, minimize
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

# --- USER IMPORTS -----------------------------------------------------------
# Ensure imports work regardless of current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Assumes python.eventide is available in your environment
try:
    from python.eventide import (
        IntervalCriterion,
        InfectionTimeCollector,
        Simulator,
        IndexOffspringCriterion,
        Scenario,
        Parameters
    )
except ImportError:
    print("Warning: python.eventide not found. Ensure it is in your PYTHONPATH.")
    sys.exit(1)


# --- CONFIGURATION CLASSES --------------------------------------------------

@dataclass
class OptimizerConfig:
    """Configuration for a specific optimization step."""
    method: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SnapshotConfig:
    """Defines the schedule of optimization."""
    # List of integers: how many observation points to include in each stage
    # e.g., [2, 5, 10] means optimize on first 2 points, then first 5, then 10.
    counts: List[int]

    # Configuration for the Global step (run first per snapshot)
    global_opt: OptimizerConfig

    # Configuration for the Local step (run second per snapshot)
    local_opt: OptimizerConfig


# --- LIGHT CSV LOGGER -------------------------------------------------------
class EvalLogger:
    def __init__(self, path: str, write_every: int = 10):
        self.path = path
        self.write_every = max(1, int(write_every))
        self._rows: List[Dict] = []
        self._count = 0
        self._header_written = os.path.exists(path) and os.path.getsize(path) > 0

    def log(self, row: dict):
        self._rows.append(row)
        self._count += 1
        if self._count % self.write_every == 0:
            self.flush()

    def flush(self):
        if not self._rows:
            return
        df = pd.DataFrame(self._rows)
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        df.to_csv(self.path, mode='a', index=False, header=not self._header_written)
        self._header_written = True
        self._rows.clear()


# --- DUAL MODE ACCEPTANCE WINDOW BUILDER ------------------------------------
def build_acceptance_inequalities(
        obs_points: Sequence[Tuple[datetime, int]],
        *,
        simulation_start: datetime,

        # Cluster–merging params
        sigma_days: float = 2.0,
        beta: float = 0.25,
        neighbor_weight: float = 0.8,

        # Interpolant–segmentation params
        grid_step_days: float = 0.25,
        min_seg_days: float = 1.0,
        kmax: int = 4,

        # Bands
        baseline_p: float = 0.10,
        alpha: float = 0.10,
        h_max: float = 0.50,
        eps_share: float = 1e-6,

        # Cross–window additions
        include_gap_windows: bool = True,
        include_union_windows: bool = False,
        max_unions_to_keep: int = 0,
        gap_scale: float = 0.4,

        # Global total
        include_global_total: bool = False,

        # Finalization
        snap_to_days: bool = False,
) -> List[IntervalCriterion]:
    """
    Generates acceptance criteria using both Cluster-Merging and Interpolant-Segmentation
    strategies, intersecting their bounds to find the strictest feasible constraints.
    """

    # ---- 1. Data Preparation ----
    obs = sorted([(t, int(c)) for (t, c) in obs_points if int(c) > 0], key=lambda x: x[0])
    if not obs:
        return []

    times = [t for t, _ in obs]
    counts = [c for _, c in obs]
    A, B = times[0], times[-1]
    n_total = int(sum(counts))

    def day_floor(t: datetime) -> datetime:
        return datetime(t.year, t.month, t.day)

    def day_ceil(t: datetime) -> datetime:
        d0 = day_floor(t)
        return d0 if t == d0 else d0 + timedelta(days=1)

    d0, d1 = day_floor(A), day_ceil(B)
    day_edges = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    per_day_map = {}
    for t, c in obs:
        key = (t.year, t.month, t.day)
        per_day_map[key] = per_day_map.get(key, 0.0) + float(c)

    def get_observed_mass(a: datetime, b: datetime) -> float:
        total = 0.0
        for i in range(len(day_edges) - 1):
            D0, D1 = day_edges[i], day_edges[i + 1]
            if D1 <= a: continue
            if D0 >= b: break
            key = (D0.year, D0.month, D0.day)
            val = per_day_map.get(key, 0.0)
            if val == 0: continue
            overlap = max(0.0, (min(b, D1) - max(a, D0)).total_seconds())
            if overlap > 0:
                total += (overlap / (D1 - D0).total_seconds()) * val
        return total

    z = float(norm.ppf(1 - alpha / 2.0))

    def calculate_band(a: datetime, b: datetime) -> Tuple[int, int]:
        mass = get_observed_mass(a, b)
        pi_hat = (mass / n_total) if n_total > 0 else 0.0
        if n_total <= 0:
            w = 0.0
        else:
            denom = 1.0 + (z * z) / n_total
            term_sq = (pi_hat * (1 - pi_hat)) / n_total + (z * z) / (4 * n_total * n_total)
            w = (z * sqrt(max(0, term_sq))) / denom

        safe_pi = max(pi_hat, eps_share)
        h = min(h_max, w / safe_pi)
        p = baseline_p + h
        L = max(0, int(floor((1 - p) * mass)))
        U = int(ceil((1 + p) * mass))
        return L, U

    # ---- 2. Strategy 1: Cluster Merging ----
    def discover_cluster_windows() -> List[Tuple[datetime, datetime]]:
        z_beta = float(norm.ppf(0.5 + beta / 2.0))
        base_half = z_beta * float(sigma_days)

        def gap_sec(i: int, j: int) -> Optional[float]:
            if i < 0 or j >= len(times): return None
            return (times[j] - times[i]).total_seconds()

        raw = []
        for i, (t, _) in enumerate(obs):
            gL = gap_sec(i - 1, i)
            gR = gap_sec(i, i + 1)
            gL_d = (gL / 86400.0) if gL is not None else None
            gR_d = (gR / 86400.0) if gR is not None else None

            if gL_d is None and gR_d is None:
                local_term = 0.0
            elif gL_d is None or gR_d is None:
                local_term = neighbor_weight * (gL_d or gR_d) / 2.0
            else:
                local_term = neighbor_weight * (gL_d + gR_d) / 4.0

            half_days = max(base_half, local_term)
            raw.append((t - timedelta(days=half_days), t + timedelta(days=half_days)))

        raw.sort(key=lambda x: x[0])
        if not raw: return []

        merged = []
        curr_s, curr_e = raw[0]
        for next_s, next_e in raw[1:]:
            if next_s <= curr_e:
                curr_e = max(curr_e, next_e)
            else:
                merged.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e
        merged.append((curr_s, curr_e))
        return merged

    # ---- 3. Strategy 2: Interpolant Segmentation ----
    def discover_segment_windows() -> List[Tuple[datetime, datetime]]:
        if len(obs) < 2: return []

        def to_days(t: datetime) -> float:
            return (t - A).total_seconds() / 86400.0

        x_obs = np.array([to_days(t) for t in times], dtype=float)
        cum_counts = np.cumsum(counts).astype(float)

        try:
            f = PchipInterpolator(x_obs, cum_counts, axis=0)
        except:
            return []

        step = float(grid_step_days)
        x_min, x_max = float(x_obs.min()), float(x_obs.max())
        x_dense = np.arange(x_min, x_max + 1e-9, step)
        f_dense = f(x_dense)
        n_dense = len(x_dense)

        min_pts = max(2, int(round(min_seg_days / step)))
        if n_dense < min_pts: return []

        memo_cost = {}

        def get_cost(i, j):
            if (i, j) not in memo_cost:
                segment_len_idx = j - i
                y_start, y_end = f_dense[i], f_dense[j]
                slope = (y_end - y_start) / max(1, segment_len_idx)
                indices = np.arange(0, segment_len_idx + 1)
                y_lin = y_start + indices * slope
                y_actual = f_dense[i: j + 1]
                memo_cost[(i, j)] = float(np.sum((y_actual - y_lin) ** 2))
            return memo_cost[(i, j)]

        best_windows = []
        best_score = float('inf')

        for K in range(1, kmax + 1):
            dp = np.full((K + 1, n_dense), 1e100)
            parent = np.full((K + 1, n_dense), -1, dtype=int)
            for j in range(min_pts, n_dense):
                dp[1, j] = get_cost(0, j)
                parent[1, j] = 0
            for k in range(2, K + 1):
                for j in range(k * min_pts, n_dense):
                    valid_i = range((k - 1) * min_pts, j - min_pts + 1)
                    if not valid_i: continue
                    costs = [dp[k - 1, i] + get_cost(i, j) for i in valid_i]
                    min_c = min(costs)
                    dp[k, j] = min_c
                    parent[k, j] = valid_i[costs.index(min_c)]

            sse = dp[K, n_dense - 1]
            if sse >= 1e99: continue
            mse = max(sse / n_dense, 1e-12)
            penalty = 2 * K * log(n_dense)
            score = n_dense * log(mse) + penalty

            if score < best_score:
                best_score = score
                indices = [n_dense - 1]
                curr = n_dense - 1
                for k_curr in range(K, 1, -1):
                    prev = parent[k_curr, curr]
                    indices.append(prev)
                    curr = prev
                indices.append(0)
                indices.reverse()
                w_list = []
                for ii in range(len(indices) - 1):
                    t_s = A + timedelta(days=float(x_dense[indices[ii]]))
                    t_e = A + timedelta(days=float(x_dense[indices[ii + 1]]))
                    w_list.append((t_s, t_e))
                best_windows = w_list

        return best_windows

    # ---- 4. Collection & Merging Phase ----
    candidate_bounds: Dict[Tuple[datetime, datetime], List[int]] = {}

    def clamp_window(a: datetime, b: datetime) -> Optional[Tuple[datetime, datetime]]:
        s = max(a, simulation_start)
        e = min(b, B)
        if snap_to_days:
            s, e = day_floor(s), day_ceil(e)
        return (s, e) if s < e else None

    def add_candidate(start: datetime, end: datetime, l_cand: int, u_cand: int):
        clamped = clamp_window(start, end)
        if not clamped: return
        s, e = clamped
        if (s, e) not in candidate_bounds:
            candidate_bounds[(s, e)] = [l_cand, u_cand]
        else:
            current_L, current_U = candidate_bounds[(s, e)]
            new_L = max(current_L, l_cand)
            new_U = min(current_U, u_cand)
            if new_L > new_U:  # Fallback to union if strict intersection fails
                new_L = min(current_L, l_cand)
                new_U = max(current_U, u_cand)
            candidate_bounds[(s, e)] = [new_L, new_U]

    windows_cluster = discover_cluster_windows()
    windows_segment = discover_segment_windows()

    all_base_windows = list(set(windows_cluster + windows_segment))
    all_base_windows.sort(key=lambda x: x[0])

    for (s, e) in all_base_windows:
        L, U = calculate_band(s, e)
        add_candidate(s, e, L, U)

    if include_gap_windows and len(windows_cluster) >= 2:
        lengths = [(b - a).total_seconds() for (a, b) in windows_cluster]
        median_sec = float(np.median(lengths)) if lengths else 86400.0
        for i in range(len(windows_cluster) - 1):
            b1 = windows_cluster[i][1]
            a2 = windows_cluster[i + 1][0]
            gap_sec = (a2 - b1).total_seconds()
            if gap_sec <= 0: continue
            center = b1 + timedelta(seconds=gap_sec / 2)
            half_days = min(gap_scale * (gap_sec / 86400.0), 0.5 * (median_sec / 86400.0))
            if half_days <= 1e-3: continue
            g_start, g_end = center - timedelta(days=half_days), center + timedelta(days=half_days)
            Lg, Ug = calculate_band(g_start, g_end)
            add_candidate(g_start, g_end, Lg, Ug)

    sorted_unique_bases = sorted([k for k in candidate_bounds.keys()], key=lambda x: x[0])
    if include_union_windows and len(sorted_unique_bases) >= 2 and max_unions_to_keep > 0:
        base_constraints = [candidate_bounds[k] for k in sorted_unique_bases]
        union_candidates = []
        K_wins = len(sorted_unique_bases)
        for i in range(K_wins):
            s_union = sorted_unique_bases[i][0]
            sum_L, sum_U = 0, 0
            for j in range(i, min(i + 5, K_wins)):
                window = sorted_unique_bases[j]
                if j > i:
                    prev_end = sorted_unique_bases[j - 1][1]
                    if (window[0] - prev_end).total_seconds() > 86400: break
                e_union = window[1]
                L_part, U_part = base_constraints[j]
                sum_L += L_part
                sum_U += U_part
                L_true, U_true = calculate_band(s_union, e_union)
                eff_L, eff_U = max(L_true, sum_L), min(U_true, sum_U)
                if eff_L > eff_U: continue
                gain = (sum_U - sum_L) - (eff_U - eff_L)
                if gain > 0:
                    union_candidates.append({'range': (s_union, e_union), 'L': eff_L, 'U': eff_U, 'gain': gain})
        union_candidates.sort(key=lambda x: -x['gain'])
        for cand in union_candidates[:max_unions_to_keep]:
            r = cand['range']
            add_candidate(r[0], r[1], cand['L'], cand['U'])

    if include_global_total:
        L_tot, U_tot = calculate_band(A, B)
        add_candidate(A, B, L_tot, U_tot)

    final_list = []
    sorted_keys = sorted(candidate_bounds.keys(), key=lambda x: (x[0], x[1]))
    for (s, e) in sorted_keys:
        L, U = candidate_bounds[(s, e)]
        final_list.append(IntervalCriterion(s, e, L, U))
    return final_list


# --- SIMULATION HELPERS -----------------------------------------------------

def observed_cumulative_at(times_eval: Sequence[datetime],
                           obs_points: Sequence[Tuple[datetime, int]]) -> np.ndarray:
    obs = sorted(obs_points, key=lambda x: x[0])
    cum = 0
    out = []
    j = 0
    for t in times_eval:
        while j < len(obs) and obs[j][0] <= t:
            cum += int(obs[j][1])
            j += 1
        out.append(cum)
    return np.asarray(out, dtype=float)


def mean_cumulative_from_infections(times_eval: Sequence[datetime],
                                    infection_times_2d: Sequence[Sequence[float]],
                                    start_date: datetime) -> np.ndarray:
    if not infection_times_2d:
        return np.zeros(len(times_eval), dtype=float)
    t_eval_days = np.array([(t - start_date).total_seconds() / 86400.0 for t in times_eval], dtype=float)
    sums = np.zeros_like(t_eval_days, dtype=float)
    for traj in infection_times_2d:
        ts = np.sort(np.asarray(traj, dtype=float))
        counts = np.searchsorted(ts, t_eval_days, side="right")
        sums += counts
    return sums / max(1, len(infection_times_2d))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))


iter_counter = 0


def _ceil_days_between(a: datetime, b: datetime) -> int:
    """Ceil((b-a) in days), floored at 0."""
    return max(0, int(ceil((b - a).total_seconds() / 86400.0)))


def _compute_dynamic_t_run(
        sim_start: datetime,
        obs_points: Sequence[Tuple[datetime, int]],
        criteria: Sequence[Any],
        safety_extra_days: int = 7,
) -> int:
    """
    Compute a simulation horizon that covers the data/criteria without large overrun:
      max(observation span, largest criterion end offset) + max observation gap + safety.
    """
    if not obs_points:
        return max(1, int(safety_extra_days))

    obs_sorted = sorted(obs_points, key=lambda x: x[0])
    first_obs = obs_sorted[0][0]
    last_obs = obs_sorted[-1][0]

    obs_span_days = _ceil_days_between(first_obs, last_obs)

    max_obs_gap_days = 0
    if len(obs_sorted) >= 2:
        for i in range(len(obs_sorted) - 1):
            gap_days = _ceil_days_between(obs_sorted[i][0], obs_sorted[i + 1][0])
            if gap_days > max_obs_gap_days:
                max_obs_gap_days = gap_days

    max_criterion_end_days = 0
    for crit in criteria:
        end_dt = getattr(crit, "end_date", None)
        if end_dt is None:
            continue
        end_days = _ceil_days_between(sim_start, end_dt)
        if end_days > max_criterion_end_days:
            max_criterion_end_days = end_days

    base_days = max(obs_span_days, max_criterion_end_days)
    t_run = base_days + max_obs_gap_days + int(safety_extra_days)
    return max(1, int(t_run))


def run_once_and_score(
        sim_start: datetime,
        obs_points: Sequence[Tuple[datetime, int]],
        accept_kwargs: dict,
        *,
        parameters,
        scenario,
        min_required: int,
        num_trajectories: int,
        chunk_size: int,
        max_workers: int,
        include_full_artifacts: bool = False,
        verbose: bool = True,
) -> Tuple[float, int, dict]:
    """Runs a single simulation and calculates L2 score vs observations."""

    # 1. Build Criteria
    criteria = build_acceptance_inequalities(
        obs_points=obs_points,
        simulation_start=sim_start,
        **accept_kwargs,
    )

    t_run_dynamic = _compute_dynamic_t_run(
        sim_start=sim_start,
        obs_points=obs_points,
        criteria=criteria,
        safety_extra_days=7,
    )

    # 2. Run Simulator
    infection_times = InfectionTimeCollector()

    # NOTE: Adjust Simulator arguments if your version of python.eventide differs
    sim = Simulator(
        parameters=parameters,
        sampler=parameters.create_latin_hypercube_sampler(),
        start_date=sim_start,
        scenario=scenario,
        criteria=criteria + [IndexOffspringCriterion(2, 5)],
        collectors=[infection_times],
        T_run=t_run_dynamic,
        max_cases=15,
        max_workers=max_workers,
        num_trajectories=num_trajectories,
        chunk_size=chunk_size,
        min_required=min_required
    )
    sim.run()

    # 3. Score
    times_eval = sorted({t for t, _ in obs_points})
    N_obs = observed_cumulative_at(times_eval, obs_points)
    N_bar = mean_cumulative_from_infections(times_eval, infection_times.infection_times, sim.start_date)
    obj = l2_distance(N_bar, N_obs)

    try:
        accepted_count = int(getattr(sim, "accepted", len(infection_times.infection_times)))
    except Exception:
        accepted_count = len(infection_times.infection_times)

    if verbose:
        global iter_counter
        iter_counter += 1
        print(f"{iter_counter: 5d} | N_obs={len(obs_points)} | Obj(L2)={obj:10.6f} | Acc={accepted_count}")

    artifacts = dict(
        criteria=criteria,
        sim=(sim if include_full_artifacts else None),
        infection_times=(infection_times if include_full_artifacts else None),
        accepted_count=accepted_count,
        N_obs=N_obs,
        N_bar=N_bar,
        T_run=t_run_dynamic,
    )
    return obj, accepted_count, artifacts


# --- CORE OPTIMIZATION LOGIC ------------------------------------------------

def optimize_snapshots(
        sim_start: datetime,
        all_obs_points: Sequence[Tuple[datetime, int]],
        *,
        base_parameters,
        base_scenario,
        snapshot_config: SnapshotConfig,
        base_log_path: str = "opt_results",
        opt_min_required: int = 1000,
        opt_num_trajectories: int = 1_000_000_000,
        opt_chunk_size: int = 50_000,
        opt_max_workers: int = 12,
        final_min_required: int = 1000,
        final_num_trajectories: int = 1_000_000_000,
        final_chunk_size: int = 50_000,
        final_max_workers: int = 13,
        store_full_snapshot_artifacts: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Main Loop:
    Iterates through snapshots. For each snapshot:
    1. Runs Global Optimizer (seeded with previous best).
    2. Runs Local Optimizer (starting from Global result).
    """

    current_best_params: Optional[Dict[str, float]] = None
    final_artifacts = {}

    # Prepare parameter definitions
    def create_params(init_values: Optional[Dict[str, float]] = None) -> LMParams:
        p = LMParams()
        # Cluster
        p.add("sigma_days", value=1.0, min=0.25, max=4.0)
        p.add("beta", value=0.75, min=0.50, max=0.99)
        p.add("neighbor_weight", value=0.8, min=0.1, max=2.0)
        # Segment
        p.add("grid_step_days", value=0.25, min=0.10, max=0.75)
        p.add("min_seg_days", value=1.0, min=0.5, max=5.0)
        p.add("kmax", value=5.0, min=2.0, max=8.0)
        # Bands
        p.add("baseline_p", value=0.10, min=0.01, max=0.30)
        p.add("alpha", value=0.10, min=0.01, max=0.40)
        p.add("h_max", value=0.50, min=0.10, max=1.00)
        p.add("eps_share", value=1e-3, min=1e-6, max=0.10)
        # Cross-window
        p.add("include_gap_windows", value=1.0, min=0.0, max=1.0)
        p.add("include_union_windows", value=1.0, min=0.0, max=1.0)
        p.add("max_unions_to_keep", value=3.0, min=0.0, max=5.0)
        p.add("gap_scale", value=0.40, min=0.10, max=0.90)
        # Global
        p.add("include_global_total", value=1.0, min=0.0, max=1.0)

        # Apply seeds if available
        if init_values:
            for k, v in init_values.items():
                if k in p:
                    p[k].value = float(v)
        return p

    def convert_params_to_kwargs(pars) -> dict:
        d = {k: float(pars[k].value) for k in pars.keys()}
        d["kmax"] = int(round(d["kmax"]))
        d["max_unions_to_keep"] = int(round(d["max_unions_to_keep"]))
        d["include_gap_windows"] = bool(d["include_gap_windows"] > 0.5)
        d["include_union_windows"] = bool(d["include_union_windows"] > 0.5)
        d["include_global_total"] = bool(d["include_global_total"] > 0.5)
        return d

    # Loop over snapshots
    for snap_idx, n_obs in enumerate(snapshot_config.counts):
        if n_obs > len(all_obs_points):
            print(f"Warning: Requested snapshot size {n_obs} exceeds available data ({len(all_obs_points)}). Skipping.")
            continue

        current_obs = all_obs_points[:n_obs]
        print(f"\n{'=' * 60}")
        print(f"SNAPSHOT {snap_idx + 1}/{len(snapshot_config.counts)}: Using first {n_obs} observation points")
        print(f"{'=' * 60}")

        # Logger for this snapshot
        logger = EvalLogger(f"{base_log_path}_snapshot_{n_obs}.csv", write_every=5)

        # --- Define Residual Function for this Snapshot ---
        def residuals(pars):
            kwargs = convert_params_to_kwargs(pars)
            obj, accepted, _ = run_once_and_score(
                sim_start, current_obs, kwargs,
                parameters=base_parameters, scenario=base_scenario,
                min_required=opt_min_required,
                num_trajectories=opt_num_trajectories,
                chunk_size=opt_chunk_size,
                max_workers=opt_max_workers,
                include_full_artifacts=False,
                verbose=True
            )
            # Log
            row = {"objective": obj, "accepted": accepted, **kwargs}
            logger.log(row)

            # Soft additive penalty for low acceptance (matches on_the_fly.py behavior)
            min_accept = 80
            if accepted < min_accept:
                short = (min_accept - accepted)
                obj += (short ** 2)
            return np.array([obj])

        global iter_counter

        # --- PHASE 1: GLOBAL OPTIMIZATION ---
        print(f"\n--- Phase 1: Global Optimization ({snapshot_config.global_opt.method}) ---")
        iter_counter = 0

        # Initialize params (seeded with previous best if exists)
        params_global = create_params(current_best_params)

        # Enforce polish=False for global
        global_kwargs = snapshot_config.global_opt.kwargs.copy()
        if snapshot_config.global_opt.method == 'differential_evolution':
            global_kwargs['polish'] = False

        result_global = minimize(
            residuals,
            params_global,
            method=snapshot_config.global_opt.method,
            **global_kwargs
        )
        logger.flush()

        # Extract best from global
        best_global_vals = {k: float(result_global.params[k].value) for k in result_global.params.keys()}
        print(f"Global Phase finished. Best Obj: {result_global.chisqr:.6f}")

        # --- PHASE 2: LOCAL OPTIMIZATION ---
        print(f"\n--- Phase 2: Local Optimization ({snapshot_config.local_opt.method}) ---")
        iter_counter = 0

        # Initialize params EXACTLY at global optimum
        params_local = create_params(best_global_vals)

        result_local = minimize(
            residuals,
            params_local,
            method=snapshot_config.local_opt.method,
            **snapshot_config.local_opt.kwargs
        )
        logger.flush()

        # Update best params for next snapshot
        current_best_params = {k: float(result_local.params[k].value) for k in result_local.params.keys()}
        print(f"Local Phase finished. Best Obj: {result_local.chisqr:.6f}")

        # Store artifacts
        final_kwargs = convert_params_to_kwargs(result_local.params)
        _, _, artifacts = run_once_and_score(
            sim_start, current_obs, final_kwargs,
            parameters=base_parameters, scenario=base_scenario,
            min_required=final_min_required,
            num_trajectories=final_num_trajectories,
            chunk_size=final_chunk_size,
            max_workers=final_max_workers,
            include_full_artifacts=store_full_snapshot_artifacts,
            verbose=False
        )
        final_artifacts[n_obs] = artifacts

    return convert_params_to_kwargs(create_params(current_best_params)), final_artifacts


# --- MAIN EXECUTION ---------------------------------------------------------
def main():
    # 1. Setup Simulation Environment (User defined)
    scenario = Scenario([])

    # Define your Epidemic Parameters
    pars = (
        Parameters(
            R0=(0.25, 15),
            k=(0.2, 10),
            r=(0.01, 0.99),
            alpha=(0.01, 20),
            theta=(0.01, 40)
        ).require('R0 * r < 3')
        .require('1 < alpha * theta').require('alpha * theta < 28')
        .require('sqrt(alpha) * theta < 21')
        .require('R0 / k < 1.2').require('R0 * r / k < 0.4')
        .require('(k / (k + R0 * r)) ^ k > 0.05').require('(k / (k + R0 * r)) ^ k < 0.95')
        .require('((R0 * r) ^ (1 / alpha) - 1) / theta < 0.10')
    )

    # 2. Define Observations
    obs_points = [
        (datetime(2025, 3, 3), 1),
        (datetime(2025, 3, 20), 3),
        (datetime(2025, 3, 24), 1),
        (datetime(2025, 3, 25), 1),
        (datetime(2025, 3, 30), 1),
        (datetime(2025, 4, 1), 2),
        (datetime(2025, 4, 4), 1),
        (datetime(2025, 4, 17), 1),
    ]

    sim_start = obs_points[0][0]

    # 3. Configure Optimizers & Snapshots (EASY SELECTION HERE)

    config = SnapshotConfig(
        # Which prefixes of data to optimize on?
        counts=list(range(1, len(obs_points) + 1)),

        # Global Optimizer Settings
        # 'differential_evolution', 'shgo', 'dual_annealing', 'basinhopping'
        global_opt=OptimizerConfig(
            method='differential_evolution',
            kwargs={
                'strategy': 'best1bin',  # Standard robust strategy
                'max_nfev': 10_000,  # Cap the cost
                'popsize': 10,  # Smaller population = faster (10 is usually enough for 10-12 params)
                'workers': 1,  # KEEP AS 1. Let the simulation handle parallelism.
                'polish': False  # We do this manually in the next step
            }
        ),

        # Local Optimizer Settings
        # 'nelder', 'powell', 'lbfgsb', 'cobyla'
        local_opt=OptimizerConfig(
            method='nelder',
            kwargs={
                'max_nfev': 2000,
                'options': {'fatol': 0.01, 'xatol': 0.01}
            }
        )
    )

    # 4. Run Logic
    best_params, all_artifacts = optimize_snapshots(
        sim_start=sim_start,
        all_obs_points=obs_points,
        base_parameters=pars,
        base_scenario=scenario,
        snapshot_config=config,
        base_log_path=str(REPO_ROOT / "optimization_logs" / "run_99")
    )

    print("\n" + "=" * 60)
    print("FINAL OPTIMIZED PARAMETERS")
    print("=" * 60)
    for k, v in best_params.items():
        print(f"{k:25s}: {v}")


if __name__ == "__main__":
    main()
