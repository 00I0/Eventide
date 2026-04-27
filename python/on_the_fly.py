from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
# NEW: lightweight optimizer deps (no CSV logger used)
from lmfit import minimize, Parameters as LMParams
from scipy.special import gamma as gamma_func, gammainc
from scipy.stats import gaussian_kde

from python.eventide import Simulator, Scenario, DrawCollector, ActiveSetSizeCollector, InfectionTimeCollector, \
    IndexOffspringCriterion, Parameters
from python.optimize_acceptance_windows import build_acceptance_inequalities


# ------------------------------------------------------------
# Utils: gamma PDF/CDF grid, Volterra solver for H(u)
# ------------------------------------------------------------

def gamma_pdf(x, a, th):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    m = x > 0
    out[m] = (x[m] ** (a - 1) * np.exp(-x[m] / th)) / (gamma_func(a) * th ** a)
    return out


def gamma_cdf_grid(a: float, th: float, X_max: float, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0.0, X_max + 1e-12, h, dtype=float)
    f = gamma_pdf(t, a, th)
    # Exact CDF values avoid the severe alpha < 1 bias from coarse PDF quadrature.
    F = gammainc(a, np.maximum(t / th, 0.0))
    return t, f, F


def compute_H_grid(R: float, k: float, a: float, th: float, U_max: float = 80.0, h: float = 0.2) -> np.ndarray:
    """
    Solve H(u) on u∈[0, U_max] with step h for:
        H(u) = [ β / (β + 1 - (f * H)(u)) ]^k,  where β = k/R,
    with f the Gamma(a, th) pdf. H(0) = (β/(β+1))^k.
    """
    beta = k / max(R, 1e-12)
    N = int(np.round(U_max / h))
    t = np.arange(0, (N + 1) * h, h)
    f = gamma_pdf(t, a, th)
    H = np.empty(N + 1, dtype=float)
    H[0] = (beta / (beta + 1.0)) ** k
    for n in range(1, N + 1):
        conv = h * float(np.dot(f[1:n + 1], H[n - 1::-1]))
        denom = beta + 1.0 - conv
        denom = max(denom, 1e-300)
        H[n] = (beta / denom) ** k
    return H


def _interp_grid(arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation on unit-spaced indices [0..N]."""
    arr = np.asarray(arr, dtype=float)
    N = arr.size - 1
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, float(N))
    i = np.floor(x).astype(int)
    w = x - i
    ip1 = np.minimum(i + 1, N)
    return (1.0 - w) * arr[i] + w * arr[ip1]


# ------------------------------------------------------------
# RB building blocks (posterior-collapsed)
# ------------------------------------------------------------

def _align_common_prefix(*arrays):
    lens = [len(a) for a in arrays]
    M = int(min(lens))
    return [a[:M] for a in arrays] + [M]


def _count_pre_children_for_parents_fast(stopped_pairs_m, parents, t_star):
    """Counts pre-cutoff first-gen children per parent (using exact time equality)."""
    arr = np.array(stopped_pairs_m, dtype=float).reshape(-1, 2)
    if arr.size == 0 or parents.size == 0:
        return np.zeros(parents.size, dtype=int)
    mask = (arr[:, 0] <= t_star) & (arr[:, 1] <= t_star) & (arr[:, 1] > arr[:, 0])
    P = arr[mask, 0]
    if P.size == 0:
        return np.zeros(parents.size, dtype=int)
    u, cnt = np.unique(P, return_counts=True)
    count_map = {float(ui): int(ci) for ui, ci in zip(u, cnt)}
    return np.array([count_map.get(float(t), 0) for t in parents], dtype=int)


def rb_cond_components_post(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=0.2
):
    """
    Posterior-collapsed conditional components (Theorem RB-cond).
    """
    infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, M = _align_common_prefix(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas
    )
    T_grid = np.asarray(T_grid, float)
    T_max = float(np.max(T_grid))
    X_max = float(t_star + T_max)
    at_keys = {(round(float(a), 6), round(float(th), 6), h, round(X_max, 6))
               for a, th in zip(alphas, thetas)}
    at_cache = {key: gamma_cdf_grid(key[0], key[1], X_max, h) for key in at_keys}
    g_inf = np.ones(M, dtype=float)
    g_quiet = np.ones((M, T_grid.size), dtype=float)
    for m in range(M):
        times = np.sort(np.asarray(infection_times[m], float))
        parents = times[times <= t_star]
        if parents.size == 0:
            continue
        R0, r, k, a, th = map(float, (R0s[m], rs[m], ks[m], alphas[m], thetas[m]))
        _, _, F_g = at_cache[(round(a, 6), round(th, 6), h, round(X_max, 6))]
        Δ = t_star - parents
        xΔ = Δ / h
        FΔ = _interp_grid(F_g, xΔ)
        μpre = FΔ
        μinf = 1.0 - FΔ
        is_index = np.isclose(parents, 0.0)
        Reff_i = np.where(is_index, R0, R0 * r)
        β_i = k / np.maximum(Reff_i, 1e-12)
        n_pre = _count_pre_children_for_parents_fast(stopped_pairs[m], parents, t_star)
        k_star = k + n_pre
        β_star = β_i + μpre
        log_g_inf = np.sum(k_star * (np.log(β_star) - np.log(β_star + μinf)))
        g_inf[m] = float(np.exp(log_g_inf))
        xT = T_grid / h
        log_g_T = np.zeros_like(T_grid, float)
        for i in range(parents.size):
            FΔT = _interp_grid(F_g, xΔ[i] + xT)
            μT = FΔT - FΔ[i]
            log_g_T += k_star[i] * (np.log(β_star[i]) - np.log(β_star[i] + μT))
        g_quiet[m, :] = np.exp(log_g_T)
    return g_inf, g_quiet


def rao_blackwell_uncond_over_post_full(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_max, t_star, h=0.2, H_pad=10.0
):
    """
    Posterior-collapsed RB unconditional; returns T_fine, mean curve, and per-draw matrix.
    """
    infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, M = _align_common_prefix(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas
    )
    T_fine = np.arange(0.0, T_max + 1e-12, h, dtype=float)
    NT = T_fine.size
    X_max = float(t_star + T_max)
    at_keys = {(round(float(a), 6), round(float(th), 6), h, round(X_max, 6)) for a, th in zip(alphas, thetas)}
    at_cache = {k: gamma_cdf_grid(k[0], k[1], X_max, h) for k in at_keys}
    H_cache: Dict[Tuple[float, ...], np.ndarray] = {}
    for m in range(M):
        R_post = float(R0s[m] * rs[m])
        keyH = (round(R_post, 6), round(float(ks[m]), 6), round(float(alphas[m]), 6),
                round(float(thetas[m]), 6), h, round(T_max + H_pad, 6))
        if keyH not in H_cache:
            H_cache[keyH] = compute_H_grid(R_post, ks[m], alphas[m], thetas[m],
                                           U_max=T_max + H_pad, h=h)
    p_mean = np.zeros(NT, dtype=float)
    g_uncond = np.ones((M, NT), dtype=float)
    for m in range(M):
        times = np.sort(np.asarray(infection_times[m], float))
        parents = times[times <= t_star]
        if parents.size == 0:
            p_mean += 1.0
            g_uncond[m, :] = 1.0
            continue
        R0, r, k, a, th = map(float, (R0s[m], rs[m], ks[m], alphas[m], thetas[m]))
        keyAT = (round(a, 6), round(th, 6), h, round(X_max, 6))
        _, f_g, F_g = at_cache[keyAT]
        keyH = (round(R0 * r, 6), round(k, 6), round(a, 6), round(th, 6), h, round(T_max + H_pad, 6))
        H = H_cache[keyH]
        gH = 1.0 - H
        gH[gH < 0] = 0.0
        Δ = t_star - parents
        xΔ = Δ / h
        F_Δ = _interp_grid(F_g, xΔ)
        μpre = F_Δ
        is_index = np.isclose(parents, 0.0)
        Reff_i = np.where(is_index, R0, R0 * r)
        β_i = k / np.maximum(Reff_i, 1e-12)
        n_pre = _count_pre_children_for_parents_fast(stopped_pairs[m], parents, t_star)
        k_star = k + n_pre
        β_star = β_i + μpre
        log_prod = np.zeros(NT, dtype=float)
        j_grid = np.arange(1, NT + 1, dtype=float)
        for i in range(parents.size):
            f_slice = gamma_pdf(Δ[i] + h * j_grid, a, th)
            conv = np.convolve(f_slice, gH, mode='full')[:NT]
            I_vals = np.zeros(NT, dtype=float)
            I_vals[1:] = h * conv[:NT - 1]
            F_ΔT = _interp_grid(F_g, xΔ[i] + np.arange(NT, dtype=float))
            tail = 1.0 - F_ΔT
            ψ = I_vals + tail
            log_prod += k_star[i] * (np.log(β_star[i]) - np.log(β_star[i] + ψ))
        contrib = np.exp(log_prod)
        g_uncond[m, :] = contrib
        p_mean += contrib
    p_mean /= max(M, 1)
    return T_fine, p_mean, g_uncond


def rb_draws_uncond_full_to_grid(T_fine: np.ndarray, g_uncond: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
    """Interpolate RB unconditional per-draw matrix to T_grid."""
    M = g_uncond.shape[0]
    return np.vstack([np.interp(T_grid, T_fine, g_uncond[m]) for m in range(M)])


def rb_draws_cond_from_components(g_inf: np.ndarray, g_quiet: np.ndarray) -> np.ndarray:
    """Per-draw conditional RB curves from components."""
    return g_inf[:, None] / g_quiet


# ------------------------------------------------------------
# Snapshots: run ABC per prefix, compute RB curves
# ------------------------------------------------------------

@dataclass
class SnapshotResult:
    m: int
    t_star: float
    T_grid: np.ndarray
    p_uncond_mean: np.ndarray
    p_cond_mean: np.ndarray
    p_uncond_draws: np.ndarray  # (M_accept, nT)
    p_cond_draws: np.ndarray  # (M_accept, nT)
    draws_array: np.ndarray  # (M_accept, 5) [R0,k,r,alpha,theta]
    n_obs: int
    next_T: Optional[float] = None


def _cast_accept_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    """Cast DE-optimized hyperparams to the right types for build_acceptance_inequalities."""
    out = dict(d)
    if "kmax" in out: out["kmax"] = int(round(float(out["kmax"])))
    for b in ("include_gap_windows", "include_union_windows"):
        if b in out: out[b] = bool(round(float(out[b])))
    if "max_unions_to_keep" in out: out["max_unions_to_keep"] = int(round(float(out["max_unions_to_keep"])))
    if "mode" in out: out["mode"] = str(out["mode"])
    return out


def _build_criteria_for_prefix(obs_points, m, **kwargs):
    pts = obs_points[:m]
    sim_start = min(t for t, _ in pts)
    crit = [IndexOffspringCriterion(2, 5)] + build_acceptance_inequalities(
        obs_points=pts,
        simulation_start=sim_start,
        **_cast_accept_kwargs(kwargs)
    )
    return crit, sim_start


def run_snapshot(m: int,
                 obs_points,
                 pars,
                 builder_kwargs: Dict[str, Any],
                 num_trajectories: int = 800_000,
                 chunk_size: int = 100_000,
                 T_run: int = 70,
                 max_cases: int = 1000,
                 max_workers: int = 8,
                 T_grid: np.ndarray = np.arange(0, 70 + 1e-9, 1.0),
                 h: float = 0.2,
                 H_pad: float = 10.0) -> SnapshotResult:
    """
    Build acceptance from the first m observations, run ABC, compute RB curves on T_grid.
    Horizons are relative to that snapshot's t_star (last observed infection time).
    """
    crit, sim_start = _build_criteria_for_prefix(obs_points, m, **builder_kwargs)
    sampler = pars.create_latin_hypercube_sampler()

    collectors = [
        draws := DrawCollector(),
        active_set := ActiveSetSizeCollector(obs_points[m - 1][0]),  # t_star^{(m)} = t_m
        itimes := InfectionTimeCollector(),
    ]
    sim = Simulator(
        parameters=pars,
        sampler=sampler,
        start_date=min(t for t, _ in obs_points),
        scenario=Scenario([]),
        criteria=crit,
        collectors=collectors,
        num_trajectories=num_trajectories,
        chunk_size=chunk_size,
        T_run=T_run,
        max_cases=max_cases,
        max_workers=max_workers,
    )
    sim.run()

    infection_times = itimes.infection_times
    stopped_pairs = active_set.active_sets
    R0s, ks, rs, alphas, thetas = np.asarray(draws).T
    t_star = (active_set.collection_date - sim.start_date).days

    # RB unconditional
    Tf, pU_mean_exact, gU_draws_fine = rao_blackwell_uncond_over_post_full(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_max=float(T_grid[-1]), t_star=t_star, h=h, H_pad=H_pad
    )
    p_uncond_mean = np.interp(T_grid, Tf, pU_mean_exact)
    p_uncond_draws = rb_draws_uncond_full_to_grid(Tf, gU_draws_fine, T_grid)

    # RB conditional
    gC_inf, gC_quiet = rb_cond_components_post(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=h
    )
    p_cond_mean = (gC_inf.mean() / gC_quiet.mean(axis=0)) if gC_quiet.size else np.full_like(T_grid, np.nan)
    p_cond_draws = rb_draws_cond_from_components(gC_inf, gC_quiet) if gC_quiet.size else np.empty((0, T_grid.size))

    print('Run', m, 'accepted', len(infection_times))

    if m < len(obs_points):
        delta = (obs_points[m][0] - obs_points[m - 1][0]).total_seconds() / 86400.0
        next_T = float(delta)
    else:
        next_T = None

    n_obs = int(sum(y for _, y in obs_points[:m]))

    return SnapshotResult(
        m=m,
        t_star=t_star,
        T_grid=T_grid,
        p_uncond_mean=p_uncond_mean,
        p_cond_mean=p_cond_mean,
        p_uncond_draws=p_uncond_draws,
        p_cond_draws=p_cond_draws,
        draws_array=np.asarray(draws),
        next_T=next_T,
        n_obs=n_obs,
    )


def run_all_snapshots_per_m(obs_points,
                            pars,
                            builder_kwargs_by_m: Dict[int, Dict[str, Any]],
                            snapshots: Sequence[int],
                            *,
                            num_trajectories: int = 10_000_000,
                            chunk_size: int = 100_000,
                            T_run: int = 70,
                            max_cases: int = 1000,
                            max_workers: int = 13,
                            T_grid: np.ndarray = np.arange(0, 70 + 1e-9, 1.0),
                            h: float = 0.2,
                            H_pad: float = 10.0) -> List[SnapshotResult]:
    results = []
    for m in (m for m in snapshots if m >= 3):
        kw = builder_kwargs_by_m.get(m, {})
        results.append(run_snapshot(
            m, obs_points, pars, kw,
            num_trajectories=num_trajectories,
            chunk_size=chunk_size,
            T_run=T_run,
            max_cases=max_cases,
            max_workers=max_workers,
            T_grid=T_grid, h=h, H_pad=H_pad
        ))
    return results


# ------------------------------------------------------------
# Cumulative-curve scoring + per-snapshot optimizer (NEW)
# ------------------------------------------------------------

def observed_cumulative_at(times_eval: Sequence[datetime],
                           obs_points: Sequence[Tuple[datetime, int]]) -> np.ndarray:
    """Stepwise N_obs evaluated at specific datetimes."""
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
    """
    Mean cumulative across accepted trajectories at each evaluation time.
    infection_times_2d: per-trajectory infection times in DAYS since start_date (floats).
    """
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
    """Mean-squared distance (scale-invariant)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))


def run_once_and_score(sim_start: datetime,
                       obs_points: Sequence[Tuple[datetime, int]],
                       accept_kwargs: dict,
                       *,
                       parameters: Parameters,
                       scenario: Scenario,
                       min_accept: int = 80,  # <= NEW: snapshot penalty threshold
                       num_eval_points: int = 8,
                       T_run: int = 45,
                       max_cases: int = 100,
                       max_workers: int = 12,
                       num_trajectories: int = 1_000_000,  # <= NEW: optimization budget
                       chunk_size: int = 100_000,
                       verbose: bool = False) -> Tuple[float, int, dict]:
    """
    Build acceptance inequalities, run once, return (objective ℓ², accepted_count, artifacts).
    """
    criteria = build_acceptance_inequalities(
        obs_points=obs_points,
        simulation_start=sim_start,
        **_cast_accept_kwargs(accept_kwargs),
    )

    infection_times = InfectionTimeCollector()
    sim = Simulator(
        parameters=parameters,
        sampler=parameters.create_latin_hypercube_sampler(),
        start_date=sim_start,
        scenario=scenario,
        criteria=criteria + [IndexOffspringCriterion(2, 5)],
        collectors=[infection_times],
        T_run=T_run, max_cases=max_cases, max_workers=max_workers,
        num_trajectories=num_trajectories, chunk_size=chunk_size,
    )
    sim.run()

    A = min(t for t, _ in obs_points)
    B = max(t for t, _ in obs_points)
    times_eval = [A + i * (B - A) / (num_eval_points - 1) for i in range(num_eval_points)]
    N_obs = observed_cumulative_at(times_eval, obs_points)
    N_bar = mean_cumulative_from_infections(times_eval, infection_times.infection_times, sim.start_date)
    obj = l2_distance(N_bar, N_obs)

    accepted = len(infection_times.infection_times)
    if accepted < min_accept:
        short = (min_accept - accepted)
        obj += (short ** 2)

    if verbose:
        print(f"accepted={accepted:7d}  objective(L2)={obj:10.6f}")

    artifacts = dict(criteria=criteria, sim=sim, infection_times=infection_times,
                     times_eval=times_eval, N_obs=N_obs, N_bar=N_bar)
    return obj, accepted, artifacts


def optimize_window_hparams_for_snapshot(sim_start: datetime,
                                         obs_points: Sequence[Tuple[datetime, int]],
                                         *,
                                         base_parameters: Parameters,
                                         base_scenario: Scenario,
                                         mode: str = "cluster") -> Dict[str, Any]:
    """
    Differential-evolution over window parameters to minimize L2(mean-cum, obs) for ONE snapshot.
    Returns best_param_dict.
    """
    p = LMParams()
    # cluster knobs
    p.add("sigma_days", value=1.0, min=0.25, max=4.0, vary=(mode == "cluster"))
    p.add("beta", value=0.75, min=0.50, max=0.98, vary=(mode == "cluster"))
    p.add("neighbor_weight", value=0.8, min=0.3, max=2.5, vary=(mode == "cluster"))

    # segment knobs
    p.add("grid_step_days", value=0.25, min=0.15, max=0.75, vary=(mode == "segment"))
    p.add("min_seg_days", value=1.0, min=0.5, max=4.0, vary=(mode == "segment"))
    p.add("kmax", value=4.0, min=2.0, max=8.0, vary=(mode == "segment"))

    # bands
    p.add("baseline_p", value=0.10, min=0.05, max=0.20)
    p.add("alpha", value=0.20, min=0.05, max=0.45)
    p.add("h_max", value=0.30, min=0.00, max=0.40)
    p.add("eps_share", value=1e-3, min=1e-6, max=0.10)

    # cross-window
    p.add("include_gap_windows", value=1.0, min=0.0, max=1.0)
    p.add("include_union_windows", value=1.0, min=0.0, max=1.0)
    p.add("max_unions_to_keep", value=3.0, min=0.0, max=6.0)
    p.add("gap_scale", value=0.40, min=0.15, max=0.90)

    def residuals(pars):
        d = {k: float(pars[k].value) for k in pars.keys()}
        d["mode"] = mode
        d["kmax"] = int(round(d["kmax"]))
        d["include_gap_windows"] = bool(round(d["include_gap_windows"]))
        d["include_union_windows"] = bool(round(d["include_union_windows"]))
        d["max_unions_to_keep"] = int(round(d["max_unions_to_keep"]))
        obj, accepted, _ = run_once_and_score(
            sim_start, obs_points, d,
            parameters=base_parameters, scenario=base_scenario,
            min_accept=80, num_eval_points=len(obs_points),
            num_trajectories=1_000_000, chunk_size=25_000,
            max_workers=13, verbose=False,
        )
        return np.array([obj], dtype=float)

    result = minimize(residuals, p, method="differential_evolution",
                      max_nfev=20000, popsize=20)
    best = {k: float(result.params[k].value) for k in result.params.keys()}
    best.update({
        "mode": mode,
        "kmax": int(round(best["kmax"])),
        "include_gap_windows": bool(round(best["include_gap_windows"])),
        "include_union_windows": bool(round(best["include_union_windows"])),
        "max_unions_to_keep": int(round(best["max_unions_to_keep"])),
    })
    return best


def optimize_windows_for_each_snapshot(obs_points: Sequence[Tuple[datetime, int]],
                                       snapshots: Sequence[int],
                                       *,
                                       parameters: Parameters,
                                       scenario: Scenario,
                                       mode: str = "cluster") -> Dict[int, Dict[str, Any]]:
    """Loop snapshots m -> optimize kwargs on obs_points[:m]."""
    sim_start = min(t for t, _ in obs_points)
    best_by_m: Dict[int, Dict[str, Any]] = {}
    for m in snapshots:
        obs_m = obs_points[:m]
        print(f"\n=== Optimizing window hyperparams for snapshot m={m} "
              f"(N_obs={int(sum(y for _, y in obs_m))}, len(obs_m)={len(obs_m)}) ===")
        best = optimize_window_hparams_for_snapshot(
            sim_start, obs_m,
            base_parameters=parameters, base_scenario=scenario,
            mode=mode
        )
        best_by_m[m] = best
        print("Best params for m=", m, ":", best)
    return best_by_m


# ------------------------------------------------------------
# Plot 1: Two-pane RB curves (Unconditional | Conditional)
# ------------------------------------------------------------

def plot_rb_online_two_pane(results: List[SnapshotResult],
                            fname: str = "rb_online_progress.pgf",
                            dpi: int = 300,
                            dot_size: float = 36.0):
    """
    Two-pane plot of RB probabilities.
    For each snapshot curve, add a circle at T = time-to-next-observation (days after that snapshot’s t_*).
    """

    def _add_dot(ax, line_obj, T_grid, y_curve, T_next):
        if T_next is None:
            return
        if not (T_grid[0] <= T_next <= T_grid[-1]):
            return
        y = float(np.interp(T_next, T_grid, y_curve))
        col = line_obj.get_color()
        ax.scatter([T_next], [y], s=dot_size, facecolors=col, edgecolors=col, linewidths=1,
                   zorder=(line_obj.get_zorder() + 1), clip_on=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), sharex=True, sharey=True, dpi=dpi)

    # Left: Unconditional
    ax = axes[0]
    for res in results:
        (line_uncond,) = ax.plot(res.T_grid, res.p_uncond_mean, lw=1.8, label=fr"$N_{{\mathrm{{obs}}}}={res.n_obs:2d}$")
        _add_dot(ax, line_uncond, res.T_grid, res.p_uncond_mean, res.next_T)
    ax.set_title(r"Unconditional")
    ax.set_xlabel(r"$T$ days since $t_\star$")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_xlim(results[0].T_grid[0], results[0].T_grid[-1])
    ax.legend(frameon=False, loc="lower right")

    # Right: Conditional
    ax = axes[1]
    for res in results:
        (line_cond,) = ax.plot(res.T_grid, res.p_cond_mean, lw=1.8, label=fr"$N_{{\mathrm{{obs}}}}={res.n_obs:2d}$")
        _add_dot(ax, line_cond, res.T_grid, res.p_cond_mean, res.next_T)
    ax.set_title(r"Conditional")
    ax.set_xlabel(r"$T$ days since $t_\star$")
    ax.set_ylim(0, 1)
    ax.set_xlim(results[0].T_grid[0], results[0].T_grid[-1])
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    plt.show()
    # fig.savefig(fname)


# ------------------------------------------------------------
# HDI helpers and posterior 2x4 grid
# ------------------------------------------------------------

def _support_interval(x: np.ndarray, mass: float = 0.95) -> Tuple[float, float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan, np.nan
    xs = np.sort(x)
    n = xs.size
    if n == 1: return float(xs[0]), float(xs[0])
    m = int(np.floor(mass * n))
    m = max(1, min(m, n - 1))
    widths = xs[m:] - xs[:n - m]
    j = int(np.argmin(widths))
    return float(xs[j]), float(xs[j + m])


def _collect_vars(res) -> Dict[str, np.ndarray]:
    arr = res.draws_array
    R0 = arr[:, 0]
    k = arr[:, 1]
    r = arr[:, 2]
    alpha = arr[:, 3]
    theta = arr[:, 4]
    Re = r * R0
    return dict(
        R0=R0,
        r=r,
        alpha=alpha,
        theta=theta,
        k=k,
        Re=Re,
        alpha_theta=(alpha * theta)[(3 < alpha * theta) & (alpha * theta < 20)],
        p0_Re=(k / (k + Re)) ** k
    )


def _fd_bins(all_vals: np.ndarray, lo: float, hi: float, min_bins=30, max_bins=120) -> Tuple[np.ndarray, int]:
    x = all_vals[(all_vals >= lo) & (all_vals <= hi)]
    n = x.size
    if n < 2:
        return np.linspace(lo, hi, min_bins + 1), min_bins
    iqr = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    if iqr <= 0:
        return np.linspace(lo, hi, min_bins + 1), min_bins
    bw = 2 * iqr / np.cbrt(n)
    nb = int(np.clip(np.ceil((hi - lo) / bw), min_bins, max_bins))
    return np.linspace(lo, hi, nb + 1), nb


def plot_posterior_kde_boxes(results: List,
                             mass: float = 0.95,
                             quantile_clip: Tuple[float, float] = (0.005, 0.995),
                             fname: str = "posterior_kde_boxes.pgf",
                             dpi: int = 300,
                             n_grid: int = 600,
                             row_spacing: float = 1.0,
                             ridge_height_frac: float = 0.85,
                             bandwidth: str | float = "scott"):
    """
    2x4 grid (R0, r, alpha, theta, k, rR0, alpha*theta, (k/(k+rR0))^k).
    KDE ridgelines per snapshot; colored 95% HDI band; flipped around x-axis.
    Y-ticks are N_obs for each snapshot.
    """
    var_specs = [
        ("R0", r"$R_0$"),
        ("r", r"$r$"),
        ("alpha", r"$\alpha$"),
        ("theta", r"$\theta$"),
        ("k", r"$k$"),
        ("Re", r"$rR_0$"),
        ("alpha_theta", r"$\alpha\theta$"),
        ("p0_Re", r"$\left(\frac{k}{k+rR_0}\right)^k$"),
    ]

    snapshots = results
    S = len(snapshots)
    sv_all = [_collect_vars(res) for res in snapshots]

    # Per-variable x-lims & grids shared across snapshots
    clip_lo, clip_hi = quantile_clip
    grids: Dict[str, np.ndarray] = {}
    xlims: Dict[str, Tuple[float, float]] = {}
    max_pdf: Dict[str, float] = {}
    for key, _ in var_specs:
        cat = np.concatenate([sv[key] for sv in sv_all if sv[key].size])
        if cat.size == 0:
            xlims[key] = (0.0, 1.0)
            grids[key] = np.linspace(0.0, 1.0, n_grid)
            max_pdf[key] = 1.0
            continue
        qlo, qhi = np.quantile(cat, [clip_lo, clip_hi])
        if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi == qlo:
            qlo, qhi = float(np.min(cat)), float(np.max(cat))
        span = qhi - qlo
        pad = 0.04 * span if span > 0 else 1.0
        lo, hi = qlo - pad, qhi + pad
        xlims[key] = (lo, hi)
        grids[key] = np.linspace(lo, hi, n_grid)

        mmax = 0.0
        for sv in sv_all:
            arr = sv[key]
            if arr.size < 2:
                continue
            kde = gaussian_kde(arr, bw_method=None if isinstance(bandwidth, float) else bandwidth)
            if isinstance(bandwidth, (int, float)):
                kde.set_bandwidth(bw_method=bandwidth)
            mmax = max(mmax, float(np.max(kde(grids[key]))))
        max_pdf[key] = mmax if mmax > 0 else 1.0

    # --- Create figure ---
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.8), dpi=dpi, sharex=False, sharey=False)
    axes = np.array(axes).reshape(2, 4)
    for j, (_, lab) in enumerate(var_specs[:4]): axes[0, j].set_title(lab)
    for j, (_, lab) in enumerate(var_specs[4:], start=0): axes[1, j].set_title(lab)

    def draw_var(ax, key: str):
        lo, hi = xlims[key]
        x = grids[key]
        y_min = - (ridge_height_frac * row_spacing + 0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(y_min, S - 0.5)
        ax.set_yticks(np.arange(S))
        ax.set_yticklabels([fr"$N_{{\mathrm{{obs}}}}={res.n_obs: 2d}$" for res in snapshots])
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.28)
        ax.tick_params(axis='y', length=0)

        scale = ridge_height_frac * row_spacing / max_pdf[key]

        for i_row, (res, sv) in enumerate(zip(snapshots, sv_all)):
            arr = sv[key]
            if arr.size < 2:
                continue

            kde = gaussian_kde(arr, bw_method=None if isinstance(bandwidth, float) else bandwidth)
            if isinstance(bandwidth, (int, float)):
                kde.set_bandwidth(bw_method=bandwidth)
            pdf = kde(x)
            ridge = pdf * scale

            base_val = float(i_row)
            base = np.full_like(x, base_val, dtype=float)
            ycurve = base - ridge  # flipped

            # FULL area in light gray
            ax.fill_between(x, base, ycurve, color='0.87',
                            edgecolor='white', linewidth=0.25, zorder=1)

            # HDI precise clipping
            lo_hdi, hi_hdi = _support_interval(arr, mass=mass)
            if not (np.isfinite(lo_hdi) and np.isfinite(hi_hdi) and hi_hdi > lo_hdi):
                ax.plot(x, ycurve, color='0.55', lw=0.9, alpha=0.95, zorder=3)
                continue

            pdf_edges = kde(np.array([lo_hdi, hi_hdi]))
            y_edges = base_val - pdf_edges * scale
            mask_inside = (x > lo_hdi) & (x < hi_hdi)
            x_ci = np.concatenate(([lo_hdi], x[mask_inside], [hi_hdi]))
            y_ci = np.concatenate(([y_edges[0]], ycurve[mask_inside], [y_edges[1]]))
            base_ci = np.full_like(x_ci, base_val, dtype=float)

            col = f"C{(i_row) % 10}"

            ax.fill_between(x_ci, base_ci, y_ci, color=col, edgecolor='none',
                            alpha=1.0, zorder=2)
            ax.plot(x_ci, y_ci, color=col, lw=1.0, alpha=1.0, zorder=4,
                    solid_capstyle='butt', solid_joinstyle='round')

            # Outside outline in darker gray
            left_mask = (x < lo_hdi)
            right_mask = (x > hi_hdi)
            if np.any(left_mask):
                ax.plot(x[left_mask], ycurve[left_mask], color='0.45', lw=0.9, alpha=1.0, zorder=3,
                        solid_capstyle='butt', solid_joinstyle='round')
            if np.any(right_mask):
                ax.plot(x[right_mask], ycurve[right_mask], color='0.45', lw=0.9, alpha=1.0, zorder=3,
                        solid_capstyle='butt', solid_joinstyle='round')

            # Median line (subtle)
            med = float(np.median(arr))
            med = float(np.clip(med, lo, hi))
            pdf_med = float(kde([med])[0])
            y_med = base_val - pdf_med * scale
            ax.plot([med, med], [base_val, y_med], color='black', lw=2, alpha=0.33, zorder=5,
                    solid_capstyle='butt')

    for j, (key, _) in enumerate(var_specs[:4]): draw_var(axes[0, j], key)
    for j, (key, _) in enumerate(var_specs[4:]): draw_var(axes[1, j], key)

    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    # fig.savefig(fname)
    plt.show()


# ------------------------------------------------------------
# Example glue (EDIT to your needs)
# ------------------------------------------------------------
if __name__ == "__main__":
    # ---- Inputs you already have (paste/tweak as needed) ----
    from datetime import datetime

    # Parameters object and prior constraints (example from your post)
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

    # Snapshots to overlay (prefix indices in 1..J)
    snapshots = (3, 4, 5, 6, 7)

    # Common horizon grid (relative to each snapshot's own t_star)
    T_grid = np.arange(0, 70 + 1e-9, 1.0)

    # ---- Optimize window hyperparams PER SNAPSHOT (num_trajectories=1e6; penalty if accepted < 80) ----
    scenario = Scenario([])
    mode_for_opt = "cluster"  # or "segment"
    best_kwargs_by_m = optimize_windows_for_each_snapshot(
        obs_points=obs_points,
        snapshots=snapshots,
        parameters=pars,
        scenario=scenario,
        mode=mode_for_opt
    )
    exit()
    #
    best_kwargs_by_m = {m: dict(sigma_days=3.5138784768847935, beta=0.5448705790647912,
                                neighbor_weight=0.3259905700578138, grid_step_days=0.29607190994514637,
                                min_seg_days=1.1388690204854988, kmax=4, baseline_p=0.07466034737192961,
                                alpha=0.3887850260822494, h_max=0.16418840893328812,
                                eps_share=0.0999993345398409, include_gap_windows=False,
                                include_union_windows=True, max_unions_to_keep=6,
                                gap_scale=0.3312202106199904, mode='cluster') for m in snapshots}

    best_kwargs_by_m = {
        2: {'sigma_days': 2.2568097755997485, 'beta': 0.7998708321056884, 'neighbor_weight': 1.1806520464866863,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.11245573995075447,
            'alpha': 0.32385439495197954, 'h_max': 0.12871399732163935, 'eps_share': 0.09924765304804734,
            'include_gap_windows': True, 'include_union_windows': True, 'max_unions_to_keep': 6,
            'gap_scale': 0.7560703267371328, 'mode': 'cluster'},
        3: {'sigma_days': 0.5653056052984651, 'beta': 0.6529871512752563, 'neighbor_weight': 0.387089675346202,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.08749158534707935,
            'alpha': 0.14394029922091184, 'h_max': 0.0279233396513767, 'eps_share': 0.09833774700801128,
            'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 4,
            'gap_scale': 0.15707848510892658, 'mode': 'cluster'},
        4: {'sigma_days': 3.7430822454871073, 'beta': 0.6186183122500877, 'neighbor_weight': 0.7761022110914765,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.06204652329785984,
            'alpha': 0.2536563022079606, 'h_max': 0.38087866375897683, 'eps_share': 0.06508600138947582,
            'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 3,
            'gap_scale': 0.291816139290914, 'mode': 'cluster'},
        5: {'sigma_days': 2.7144407238423396, 'beta': 0.5661796416762115, 'neighbor_weight': 0.3821037348416254,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.05992387448283569,
            'alpha': 0.13045585214541777, 'h_max': 0.29078303850618525, 'eps_share': 0.09932394410370898,
            'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 2,
            'gap_scale': 0.5494360431590082, 'mode': 'cluster'},
        6: {'sigma_days': 3.06015118575848, 'beta': 0.5053936723256046, 'neighbor_weight': 0.8892458447406966,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.05864907499636429,
            'alpha': 0.3767759340509563, 'h_max': 0.007239389104548999, 'eps_share': 0.06261011827627241,
            'include_gap_windows': True, 'include_union_windows': True, 'max_unions_to_keep': 3,
            'gap_scale': 0.25039868818075706, 'mode': 'cluster'},
        7: {'sigma_days': 1.87448405791103, 'beta': 0.8122752531114021, 'neighbor_weight': 0.36814473528396524,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.11350999644046003,
            'alpha': 0.44260858510448914, 'h_max': 0.1860463899250351, 'eps_share': 0.0988640778045239,
            'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 6,
            'gap_scale': 0.15009926426544729, 'mode': 'cluster'}
    }

    # ---- Final heavy runs with the per-snapshot optimum (num_trajectories=1e7) ----
    results = run_all_snapshots_per_m(
        obs_points=obs_points,
        pars=pars,
        builder_kwargs_by_m=best_kwargs_by_m,
        snapshots=snapshots,
        num_trajectories=10_000_000,
        chunk_size=100_000,
        T_run=70,
        max_cases=1000,
        max_workers=13,
        T_grid=T_grid,
        h=0.2,
        H_pad=10.0
    )

    # ---- Figures ----

    # mpl.use('pgf')
    # mpl.rcParams.update({
    #     'text.usetex': True, 'pgf.rcfonts': False, 'pgf.texsystem': 'pdflatex',
    #     'pgf.preamble': r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{siunitx}\usepackage[T1]{fontenc}\usepackage{lmodern}',
    #     'font.family': 'serif', 'font.size': 11,
    #     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    #     'axes.grid': True, 'axes.grid.axis': 'x', 'grid.linewidth': 0.4, 'grid.alpha': 0.28,
    #     'axes.spines.top': False, 'axes.spines.right': False,
    #     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'axes.unicode_minus': False,
    # })

    plot_rb_online_two_pane(results, fname="../img/rb_on_the_fly.pgf", dpi=300)
    plot_posterior_kde_boxes(results[-1:], mass=0.95, fname="../img/posterior_on_the_fly.pgf", dpi=300)
