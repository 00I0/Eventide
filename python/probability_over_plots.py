import time
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_func, gammainc

from python.eventide import Parameters, Simulator, Scenario, DrawCollector, ActiveSetSizeCollector, \
    InfectionTimeCollector, IndexOffspringCriterion
from python.optimize_acceptance_windows import build_acceptance_inequalities

pars = (Parameters(
    R0=(0.25, 15),
    k=(0.2, 10),
    r=(0.01, 0.99),
    alpha=(0.01, 20),
    theta=(0.01, 20)
).require('R0 * r < 3')
        .require('3 < alpha * theta').require('alpha * theta < 20')
        .require('1/sqrt(alpha) >= 0.1').require('1/sqrt(alpha) <= 0.9')
        .require('1 <= sqrt(alpha) * theta').require('sqrt(alpha) * theta <= 15'))

obs_points = [
    (datetime(2025, 3, 6), 1),
    (datetime(2025, 3, 21), 3),
    (datetime(2025, 3, 25), 1),
    (datetime(2025, 3, 26), 1),
    (datetime(2025, 3, 30), 1),
    (datetime(2025, 4, 2), 2),
    (datetime(2025, 4, 17), 1),
]
sim_start = min(t for t, _ in obs_points)
sim = Simulator(
    parameters=pars,
    sampler=pars.create_latin_hypercube_sampler(),
    start_date=sim_start,
    scenario=Scenario([
        # ParameterChangePoint('R0', datetime(2025, 4, 14), '0.5 * R0'),  # rábapordány
        # ParameterChangePoint('R0', datetime(2025, 4, 17))
    ]),
    criteria=[IndexOffspringCriterion(2, 5)] + build_acceptance_inequalities(
        obs_points=obs_points,
        simulation_start=sim_start,
        sigma_days=3.5138784768847935,
        beta=0.5448705790647912,
        neighbor_weight=0.3259905700578138,
        grid_step_days=0.29607190994514637,
        min_seg_days=1.1388690204854988,
        kmax=4,
        baseline_p=0.07466034737192961,
        alpha=0.3887850260822494,
        h_max=0.16418840893328812,
        eps_share=0.0999993345398409,
        include_gap_windows=False,
        include_union_windows=True,
        max_unions_to_keep=6,
        gap_scale=0.3312202106199904,
        mode='cluster'
    )
    # [
    #     IntervalCriterion(datetime(2025, 3, 3), datetime(2025, 4, 4), 8, 10),
    #     IntervalCriterion(datetime(2025, 4, 1), datetime(2025, 4, 4), 1, 10),
    #     # IntervalCriterion(datetime(2025, 4, 4), datetime(2025, 4, 14), 0, 0),
    #     # IntervalCriterion(datetime(2025, 4, 14), datetime(2025, 4, 17), 1, 1),
    #     # IndexOffspringCriterion(2, 5)
    # ]
    ,
    collectors=[
        draws := DrawCollector(),
        active_set_size := ActiveSetSizeCollector(datetime(2025, 4, 17)),
        infection_times_collector := InfectionTimeCollector(),
    ],
    num_trajectories=200_000_000,
    chunk_size=100_000,
    T_run=70,
    max_cases=1000,
    max_workers=13,
    min_required=20000
)

now = time.time()
sim.run()
print((datetime(2025, 4, 17) - datetime(2025, 3, 6)).days)
print('Runtime:', time.time() - now)
print('accepted:', len(np.asarray(draws)))

infection_times = infection_times_collector.infection_times
stopped_pairs = active_set_size.active_sets
R0s, ks, rs, alphas, thetas = np.asarray(draws).T

print(f'infection times ({len(infection_times)}):',
      *[', '.join(f'{t: 2.2f}' for t in sorted(times)) for times in infection_times[:10]],
      sep='\n\t')

print()
print(f'stopped times ({len(stopped_pairs)}):',
      *[', '.join(f'({p: 2.2f}, {c: 2.2f})' for p, c in sorted(times, key=lambda t: t[0])) for times in
        stopped_pairs[:10]],
      sep='\n\t')

print()
print(f'R0s {R0s.shape}:\t', ', '.join(f'{x: 2.2f}' for x in R0s[:10]))
print(f'ks {ks.shape}:\t\t', ', '.join(f'{x: 2.2f}' for x in ks[:10]))
print(f'rs {rs.shape}:\t\t', ', '.join(f'{x: 2.2f}' for x in rs[:10]))
print(f'alphas {alphas.shape}:\t', ', '.join(f'{x: 2.2f}' for x in alphas[:10]))
print(f'thetas {thetas.shape}:\t', ', '.join(f'{x: 2.2f}' for x in thetas[:10]))
#


T_max = sim.T_run
T_grid = np.arange(0.0, T_max + 1e-9, 1.0)
h = 0.2  # small step for Volterra recursion (smoother H)
U_max = T_max  # horizon for H(u) grid
M = len(stopped_pairs)
t_star = (active_set_size.collection_date - sim.start_date).days


# ---------- helpers (reuse yours, just ensure h=0.2) ----------
def gamma_pdf(x, a, th):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    m = x > 0
    out[m] = (x[m] ** (a - 1) * np.exp(-x[m] / th)) / (gamma_func(a) * th ** a)
    return out


def gamma_cdf_grid(a, th, X_max, h):
    t = np.arange(0.0, X_max + 1e-12, h, dtype=float)
    f = gamma_pdf(t, a, th)
    F = gammainc(a, np.maximum(t / th, 0.0))
    return t, f, F


def _interp_grid(arr, x):
    """Linear interpolation on unit-spaced indices [0..N]."""
    arr = np.asarray(arr, dtype=float)
    N = arr.size - 1
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, float(N))
    i = np.floor(x).astype(int)
    w = x - i
    ip1 = np.minimum(i + 1, N)
    return (1.0 - w) * arr[i] + w * arr[ip1]


def compute_H_grid(R, k, a, th, U_max=80.0, h=0.2):
    """
    Solve H(u) on u∈[0, U_max] with step h for the Volterra recursion:
      H(u) = [ β / (β + 1 - (f * H)(u)) ]^k,  β = k/R
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


def H_eval_vec(H, h, u):
    u = np.asarray(u, dtype=float)
    N = len(H) - 1
    x = u / h
    out = np.empty_like(x, dtype=float)
    mask_low = (x <= 0)
    mask_high = (x >= N)
    mask_mid = ~(mask_low | mask_high)
    out[mask_low] = H[0]
    out[mask_high] = H[-1]
    if np.any(mask_mid):
        xm = x[mask_mid]
        i = np.floor(xm).astype(int)
        w = xm - i
        out[mask_mid] = (1.0 - w) * H[i] + w * H[i + 1]
    return out


# =========================================================
# Baseline empirical + analytic (seed-based) curves
# =========================================================
def empirical_uncond_over(infection_times, T_grid, t_star):
    last_time = np.array([
        np.max(np.asarray(times, dtype=float)) if len(times) > 0 else -np.inf
        for times in infection_times
    ], dtype=float)
    p = np.array([np.mean(last_time <= t_star + T) for T in T_grid], dtype=float)
    M = len(infection_times)
    se = np.sqrt(np.clip(p * (1 - p) / max(M, 1), 0.0, 1.0))
    return p, se


def analytic_uncond_over(stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=0.2, U_max=80.0):
    M = len(stopped_pairs)
    # gather seeds per trajectory
    seeds_by = []
    for m in range(M):
        arr = np.array(stopped_pairs[m], dtype=float).reshape(-1, 2)
        if arr.size == 0:
            seeds_by.append(np.empty(0));
            continue
        mask = (arr[:, 0] <= t_star) & (arr[:, 1] > t_star)
        seeds_by.append(np.sort(arr[mask, 1]))
    # cache H by unique param keys
    key_to_indices = {}
    for m in range(M):
        key = (round(float(R0s[m] * rs[m]), 6),
               round(float(ks[m]), 6),
               round(float(alphas[m]), 6),
               round(float(thetas[m]), 6),
               h)
        key_to_indices.setdefault(key, []).append(m)
    H_grid_by_key = {k: compute_H_grid(k[0], k[1], k[2], k[3], U_max=U_max, h=h)
                     for k in key_to_indices.keys()}
    horizons = t_star + T_grid
    p = np.zeros(T_grid.size, dtype=float)
    for kkey, idxs in key_to_indices.items():
        H = H_grid_by_key[kkey]
        for m in idxs:
            seeds = seeds_by[m]
            if seeds.size == 0:
                p += 1.0;
                continue
            max_s = float(seeds[-1])
            mask = horizons >= max_s
            if not np.any(mask): continue
            u = horizons[mask][None, :] - seeds[:, None]
            vals = H_eval_vec(H, h, u)
            prod = np.prod(vals, axis=0)
            p[mask] += prod
    p /= M
    return p


def empirical_cond_over(infection_times, T_grid, t_star):
    M = len(infection_times)
    first_post = np.full(M, np.inf, dtype=float)
    has_post = np.zeros(M, dtype=bool)
    for m, times in enumerate(infection_times):
        ts = np.sort(np.asarray(times, dtype=float))
        j = np.searchsorted(ts, t_star, side='right')
        if j < ts.size:
            first_post[m] = ts[j];
            has_post[m] = True
    B_mask = ~has_post
    p = np.empty(T_grid.size, dtype=float)
    se = np.empty(T_grid.size, dtype=float)
    for j, T in enumerate(T_grid):
        cutoff = t_star + T
        A_mask = (first_post > cutoff)
        denom = np.count_nonzero(A_mask)
        numer = np.count_nonzero(A_mask & B_mask)
        if denom > 0:
            pj = numer / denom
            p[j] = pj
            se[j] = np.sqrt(np.clip(pj * (1 - pj) / denom, 0.0, 1.0))
        else:
            p[j] = np.nan;
            se[j] = np.nan
    return p, se


def analytic_cond_over(stopped_pairs, T_grid, t_star):
    M = len(stopped_pairs)
    any_seed = np.zeros(M, dtype=bool)
    first_seed = np.full(M, np.inf, dtype=float)
    for m, pairs in enumerate(stopped_pairs):
        arr = np.array(pairs, dtype=float).reshape(-1, 2)
        if arr.size == 0: continue
        mask = (arr[:, 0] <= t_star) & (arr[:, 1] > t_star)
        if np.any(mask):
            any_seed[m] = True
            first_seed[m] = float(np.min(arr[mask, 1]))
    B_mask = ~any_seed
    p = np.empty(T_grid.size, dtype=float)
    for j, T in enumerate(T_grid):
        cutoff = t_star + T
        A_mask = (first_seed > cutoff)
        denom = np.count_nonzero(A_mask)
        numer = np.count_nonzero(A_mask & B_mask)
        p[j] = numer / denom if denom > 0 else np.nan
    return p


# =========================================================
# RB components (validated constructions)
# =========================================================
def _align_common_prefix(*arrays):
    lens = [len(a) for a in arrays]
    M = int(min(lens))
    if len(set(lens)) > 1:
        print(f"[RB] Warning: mismatched lengths {tuple(lens)} -> trimming to M={M}")
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


# ---------- PRIOR (kept for reference/sensitivity; NOT used in plot wrapper) ----------
def rb_cond_components_prior(
        infection_times, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=0.2
):
    """
    Prior-collapsed conditional components:
      g_inf[m]      = ∏_i (β_i / (β_i + μ_i(∞)))^k
      g_quiet[m,t]  = ∏_i (β_i / (β_i + μ_i(T)))^k
    where β_i = k / R_eff(t_i), μ_i(T) = F(Δ_i+T)-F(Δ_i).
    """
    infection_times, R0s, rs, ks, alphas, thetas, M = _align_common_prefix(
        infection_times, R0s, rs, ks, alphas, thetas
    )
    T_grid = np.asarray(T_grid, float)
    T_max = float(np.max(T_grid));
    X_max = float(t_star + T_max)
    at_keys = {(round(float(a), 6), round(float(th), 6), h, round(X_max, 6))
               for a, th in zip(alphas, thetas)}
    at_cache = {key: gamma_cdf_grid(key[0], key[1], X_max, h) for key in at_keys}
    g_inf = np.ones(M, dtype=float)
    g_quiet = np.ones((M, T_grid.size), dtype=float)
    for m in range(M):
        times = np.sort(np.asarray(infection_times[m], float))
        parents = times[times <= t_star]
        if parents.size == 0: continue
        R0, r, k, a, th = map(float, (R0s[m], rs[m], ks[m], alphas[m], thetas[m]))
        _, _, F_g = at_cache[(round(a, 6), round(th, 6), h, round(X_max, 6))]
        Δ = t_star - parents
        xΔ = Δ / h
        FΔ = _interp_grid(F_g, xΔ)
        μinf = 1.0 - FΔ
        Reff_i = np.where(np.isclose(parents, 0.0), R0, R0 * r)
        β_i = k / np.maximum(Reff_i, 1e-12)
        # numerator
        log_g_inf = k * np.sum(np.log(β_i) - np.log(β_i + μinf))
        g_inf[m] = float(np.exp(log_g_inf))
        # denominator over T
        xT = T_grid / h
        log_g_T = np.zeros_like(T_grid, float)
        for i in range(parents.size):
            FΔT = _interp_grid(F_g, xΔ[i] + xT)
            μT = FΔT - FΔ[i]
            log_g_T += k * (np.log(β_i[i]) - np.log(β_i[i] + μT))
        g_quiet[m, :] = np.exp(log_g_T)
    return g_inf, g_quiet


# ---------- NEW: POSTERIOR for CONDITIONAL (use this) ----------
def rb_cond_components_post(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=0.2
):
    """
    Posterior-collapsed conditional components:
      For parent i with Δ_i=t* - t_i, pre-cutoff exposure μ_i^pre = F(Δ_i),
      and pre-cutoff first-gen count n_i^pre,
      update Gamma(k, rate=β_i) -> Gamma(k+n_i^pre, rate=β_i + μ_i^pre).
      Then
        g_inf[m]     = ∏_i [(β_i+μ_i^pre)/(β_i+μ_i^pre + μ_i(∞))]^{k+n_i^pre},
        g_quiet[m,t] = ∏_i [(β_i+μ_i^pre)/(β_i+μ_i^pre + μ_i(T))]^{k+n_i^pre},
      where μ_i(T)=F(Δ_i+T)-F(Δ_i),  μ_i(∞)=1-F(Δ_i),  β_i = k / R_eff(t_i).
    """
    infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, M = _align_common_prefix(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas
    )
    T_grid = np.asarray(T_grid, float)
    T_max = float(np.max(T_grid));
    X_max = float(t_star + T_max)
    # cache gamma CDFs
    at_keys = {(round(float(a), 6), round(float(th), 6), h, round(X_max, 6))
               for a, th in zip(alphas, thetas)}
    at_cache = {key: gamma_cdf_grid(key[0], key[1], X_max, h) for key in at_keys}
    g_inf = np.ones(M, dtype=float)
    g_quiet = np.ones((M, T_grid.size), dtype=float)
    for m in range(M):
        times = np.sort(np.asarray(infection_times[m], float))
        parents = times[times <= t_star]
        if parents.size == 0:  # contributes 1 to both numerator & denominator
            continue
        R0, r, k, a, th = map(float, (R0s[m], rs[m], ks[m], alphas[m], thetas[m]))
        _, _, F_g = at_cache[(round(a, 6), round(th, 6), h, round(X_max, 6))]
        Δ = t_star - parents
        xΔ = Δ / h
        FΔ = _interp_grid(F_g, xΔ)
        μpre = FΔ
        μinf = 1.0 - FΔ
        # per-parent updates
        is_index = np.isclose(parents, 0.0)
        Reff_i = np.where(is_index, R0, R0 * r)
        β_i = k / np.maximum(Reff_i, 1e-12)
        n_pre = _count_pre_children_for_parents_fast(stopped_pairs[m], parents, t_star)
        k_star = k + n_pre
        β_star = β_i + μpre
        # numerator
        log_g_inf = np.sum(k_star * (np.log(β_star) - np.log(β_star + μinf)))
        g_inf[m] = float(np.exp(log_g_inf))
        # denominator as a function of T
        xT = T_grid / h
        log_g_T = np.zeros_like(T_grid, float)
        for i in range(parents.size):
            FΔT = _interp_grid(F_g, xΔ[i] + xT)
            μT = FΔT - FΔ[i]
            log_g_T += k_star[i] * (np.log(β_star[i]) - np.log(β_star[i] + μT))
        g_quiet[m, :] = np.exp(log_g_T)
    return g_inf, g_quiet


# =========================================================
# RB UNCONDITIONAL (posterior) — mean + per-draw contributions
# =========================================================
def rao_blackwell_uncond_over_post_full(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_max, t_star, h=0.2, H_pad=10.0
):
    """
    Posterior-collapsed RB UNCONDITIONAL (mean) + per-draw contributions (g_uncond).
    """
    infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, M = _align_common_prefix(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas
    )
    T_fine = np.arange(0.0, T_max + 1e-12, h, dtype=float)
    NT = T_fine.size
    X_max = float(t_star + T_max)
    # caches
    at_keys = {(round(float(a), 6), round(float(th), 6), h, round(X_max, 6)) for a, th in zip(alphas, thetas)}
    at_cache = {k: gamma_cdf_grid(k[0], k[1], X_max, h) for k in at_keys}
    H_cache = {}
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
            p_mean += 1.0;
            g_uncond[m, :] = 1.0;
            continue
        R0, r, k, a, th = map(float, (R0s[m], rs[m], ks[m], alphas[m], thetas[m]))
        keyAT = (round(a, 6), round(th, 6), h, round(X_max, 6))
        _, f_g, F_g = at_cache[keyAT]
        keyH = (round(R0 * r, 6), round(k, 6), round(a, 6), round(th, 6), h, round(T_max + H_pad, 6))
        H = H_cache[keyH]
        gH = 1.0 - H;
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
        J_max = NT
        j_grid = np.arange(1, J_max + 1, dtype=float)
        for i in range(parents.size):
            # I(T) via discrete conv
            f_slice = gamma_pdf(Δ[i] + h * j_grid, a, th)
            conv = np.convolve(f_slice, gH, mode='full')[:NT]
            I_vals = np.zeros(NT, dtype=float)
            I_vals[1:] = h * conv[:NT - 1]
            # tail
            F_ΔT = _interp_grid(F_g, xΔ[i] + np.arange(NT, dtype=float))
            tail = 1.0 - F_ΔT
            ψ = I_vals + tail
            log_prod += k_star[i] * (np.log(β_star[i]) - np.log(β_star[i] + ψ))
        contrib = np.exp(log_prod)
        g_uncond[m, :] = contrib
        p_mean += contrib
    p_mean /= max(M, 1)
    return T_fine, p_mean, g_uncond


# =========================================================
# RB per-draw matrices on plot grid
# =========================================================
def rb_draws_uncond_full_to_grid(T_fine, g_uncond, T_grid):
    """Interpolate posterior-collapsed RB unconditional draws to T_grid."""
    M = g_uncond.shape[0]
    out = np.vstack([np.interp(T_grid, T_fine, g_uncond[m]) for m in range(M)])
    return out


def rb_draws_cond_from_components(g_inf, g_quiet):
    """Per-draw conditional RB curves from components."""
    return g_inf[:, None] / g_quiet


# =========================================================
# Probability-based bands (HPD + simultaneous)
# =========================================================
def pointwise_hpd_from_draws(draws, level=0.95):
    draws = np.asarray(draws, float)
    M, nT = draws.shape
    q = max(1, int(np.floor(level * M)))
    lo = np.empty(nT, float);
    hi = np.empty(nT, float)
    for j in range(nT):
        col = np.sort(draws[:, j])
        widths = col[q - 1:] - col[:M - q + 1]
        idx = int(np.argmin(widths))
        lo[j] = col[idx];
        hi[j] = col[idx + q - 1]
    return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)


def simultaneous_band_from_draws(draws, level=0.95, eps=1e-12, center='mean'):
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError(f"`draws` must be 2D (M, nT); got shape {draws.shape}")
    M, nT = draws.shape
    if isinstance(center, str) and center == 'mean':
        mu = draws.mean(axis=0)
    else:
        mu = np.asarray(center, dtype=float)
        if mu.shape != (nT,):
            raise ValueError(f"`center` must have shape (nT,) = {(nT,)}, got {mu.shape}")
    sd = draws.std(axis=0, ddof=1) if M > 1 else np.zeros(nT, dtype=float)
    sd = np.maximum(sd, eps)
    tvals = np.max(np.abs((draws - mu) / sd), axis=1)
    alpha = 1.0 - float(level)
    c = float(np.quantile(tvals, 1.0 - alpha))
    lo = np.clip(mu - c * sd, 0.0, 1.0)
    hi = np.clip(mu + c * sd, 0.0, 1.0)
    inside = np.all((draws >= lo) & (draws <= hi), axis=1)
    realized = float(np.mean(inside))
    return mu, (lo, hi), c, realized


# =========================================================
# Plotting wrapper: posterior RB for BOTH targets
# =========================================================
def plot_all_with_probability_bands(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_grid, t_star, h=0.2, H_pad=10.0,
        show_pointwise_hpd=False, level_hpd=0.95,
        show_simultaneous=False, level_sim=0.95,
        dpi=260
):
    # Baselines
    pU_emp, _ = empirical_uncond_over(infection_times, T_grid, t_star)
    pU_ana = analytic_uncond_over(stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=h, U_max=T_grid[-1])
    pC_emp, _ = empirical_cond_over(infection_times, T_grid, t_star)
    pC_ana = analytic_cond_over(stopped_pairs, T_grid, t_star)

    # RB UNCONDITIONAL — posterior-collapsed
    T_fine, pU_rb_mean_exact, gU_uncond_full = rao_blackwell_uncond_over_post_full(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_max=float(T_grid[-1]), t_star=t_star, h=h, H_pad=H_pad
    )
    pU_rb_mean = np.interp(T_grid, T_fine, pU_rb_mean_exact)
    pU_draws = rb_draws_uncond_full_to_grid(T_fine, gU_uncond_full, T_grid)

    # RB CONDITIONAL — posterior-collapsed (NEW)
    gC_inf_post, gC_quiet_post = rb_cond_components_post(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=h
    )
    pC_rb_mean = gC_inf_post.mean() / gC_quiet_post.mean(axis=0)
    pC_draws = rb_draws_cond_from_components(gC_inf_post, gC_quiet_post)

    # Bands (from RB per-draws)
    if show_pointwise_hpd:
        U_lo_hpd, U_hi_hpd = pointwise_hpd_from_draws(pU_draws, level=level_hpd)
        C_lo_hpd, C_hi_hpd = pointwise_hpd_from_draws(pC_draws, level=level_hpd)
    else:
        U_lo_hpd = U_hi_hpd = C_lo_hpd = C_hi_hpd = None

    if show_simultaneous:
        _, (U_lo_sim, U_hi_sim), U_c, U_real = simultaneous_band_from_draws(
            pU_draws, level=level_sim, center=pU_rb_mean
        )
        _, (C_lo_sim, C_hi_sim), C_c, C_real = simultaneous_band_from_draws(
            pC_draws, level=level_sim, center=pC_rb_mean
        )
    else:
        U_lo_sim = U_hi_sim = U_c = U_real = None
        C_lo_sim = C_hi_sim = C_c = C_real = None

    # --- Plot ---
    COL_EMP = '#4C72B0'
    COL_ANA = '#E69F00'
    COL_RB = '#333333'
    COL_SIM = '#E69F00'
    COL_HPD = '#66c2a5'

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), sharex=True, sharey=True, dpi=dpi)

    # (a) Unconditional
    ax = axes[0]
    ax.plot(T_grid, pU_emp, color=COL_EMP, lw=1.6, ls=(0, (2.6, 1.4, 1, 2.7, 1, 1.4)), zorder=4,
            dash_capstyle='butt',
            solid_capstyle='butt',
            label=r"$\widehat p_{\mathrm{uncond}}^{(\mathrm{E})}(T)$")
    ax.plot(T_grid, pU_ana, color=COL_ANA, lw=1.6,
            label=r"$ p_{\mathrm{uncond}}^{(\mathrm{SF})}(T)$ ",
            zorder=3, dash_capstyle='butt', solid_capstyle='butt')

    if show_simultaneous and U_lo_sim is not None:
        ax.fill_between(T_grid, U_lo_sim, U_hi_sim, color=COL_SIM, alpha=0.18, linewidth=0,
                        label=f"RB Simul. {int(100 * level_sim)}%")
    if show_pointwise_hpd and U_lo_hpd is not None:
        ax.fill_between(T_grid, U_lo_hpd, U_hi_hpd, color=COL_HPD, alpha=0.25, linewidth=0,
                        label=f"RB HPD {int(100 * level_hpd)}%")
    ax.plot(T_grid, pU_rb_mean, color=COL_RB, lw=1.6, ls='-', zorder=2,
            label=r"$ p_{\mathrm{uncond}}^{(\mathrm{RB})}(T)$", dash_capstyle='butt', solid_capstyle='butt')
    ax.set_title(r"Unconditional")
    ax.set_xlabel(r"$T$ days since $t_\star$")
    ax.set_ylabel("Probability")
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")

    # (b) Conditional
    ax = axes[1]
    ax.plot(T_grid, pC_emp, color=COL_EMP, lw=1.6, ls=(0, (2.6, 1.4, 1, 2.7, 1, 1.4)), zorder=4,
            dash_capstyle='butt',
            solid_capstyle='butt',
            label=r"$\widehat p_{\mathrm{cond}}^{(\mathrm{E})}(T)$")
    ax.plot(T_grid, pC_ana, color=COL_ANA, lw=1.6, dash_capstyle='butt',
            solid_capstyle='butt',
            label=r"$ p_{\mathrm{cond}}^{(\mathrm{SF})}(T)$",
            zorder=3)

    if show_simultaneous and C_lo_sim is not None:
        ax.fill_between(T_grid, C_lo_sim, C_hi_sim, color=COL_SIM, alpha=0.18, linewidth=0,
                        label=f"RB Simul. {int(100 * level_sim)}%")
    if show_pointwise_hpd and C_lo_hpd is not None:
        ax.fill_between(T_grid, C_lo_hpd, C_hi_hpd, color=COL_HPD, alpha=0.25, linewidth=0,
                        label=f"RB HPD {int(100 * level_hpd)}%")
    ax.plot(T_grid, pC_rb_mean, color=COL_RB, lw=2.2, ls='solid', dash_capstyle='butt', solid_capstyle='butt',
            label=r"$ p_{\mathrm{cond}}^{(\mathrm{RB})}(T)$",
            zorder=2)
    ax.set_title(r"Conditional")
    ax.set_xlabel(r"$T$ days since $t_\star$")
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    # fig.savefig(f"../img/rb_probability_over_20k.pgf")
    plt.show()

    return dict(
        uncond=dict(emp=pU_emp, ana=pU_ana, rb_mean=pU_rb_mean, draws=pU_draws,
                    pointwise_hpd=(U_lo_hpd, U_hi_hpd) if show_pointwise_hpd else None,
                    simultaneous=(U_lo_sim, U_hi_sim, U_c, U_real) if show_simultaneous else None),
        cond=dict(emp=pC_emp, ana=pC_ana, rb_mean=pC_rb_mean, draws=pC_draws,
                  pointwise_hpd=(C_lo_hpd, C_hi_hpd) if show_pointwise_hpd else None,
                  simultaneous=(C_lo_sim, C_hi_sim, C_c, C_real) if show_simultaneous else None)
    )


def plot_all_with_probability_bands2(
        infection_times,
        stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_grid, t_star, h=0.2, dpi=300
):
    N = 250
    pU_ana = analytic_uncond_over(stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=h, U_max=T_grid[-1])
    pU_ana2 = analytic_uncond_over(stopped_pairs[:N], R0s[:N], rs[:N], ks[:N], alphas[:N], thetas[:N],
                                   T_grid, t_star, h)
    T_fine, pU_rb_mean_exact, gU_uncond_full = rao_blackwell_uncond_over_post_full(
        infection_times[:N], stopped_pairs[:N], R0s[:N], rs[:N], ks[:N], alphas[:N], thetas[:N],
        T_max=float(T_grid[-1]), t_star=t_star, h=h, H_pad=10.0
    )
    pU_rb = np.interp(T_grid, T_fine, pU_rb_mean_exact)

    pC_ana = analytic_cond_over(stopped_pairs, T_grid, t_star)
    pC_ana2 = analytic_cond_over(stopped_pairs[:N], T_grid, t_star)
    gC_inf_post, gC_quiet_post = rb_cond_components_post(
        infection_times[:N], stopped_pairs[:N], R0s[:N], rs[:N], ks[:N], alphas[:N], thetas[:N],
        T_grid, t_star, h=h
    )
    pC_rb = gC_inf_post.mean() / gC_quiet_post.mean(axis=0)

    COL_ANA = '#333333'
    COL_ANA_small = '#E69F00'
    COL_RB = '#6F9CEB'

    l = [
        (
            [(pC_ana, COL_ANA, r"$ p_{\mathrm{c}}^{\mathrm{SF}}(T)$ $N=20\'000$")],
            [(pU_ana, COL_ANA, r"$ p_{\mathrm{u}}^{\mathrm{SF}}(T)$ $N=20\'000$")]
        ),
        (
            [(pC_ana, COL_ANA, r"$ p_{\mathrm{c}}^{\mathrm{SF}}(T)$ $N=20\'000$"),
             (pC_ana2, COL_ANA_small, r"$ p_{\mathrm{c}}^{\mathrm{SF}}(T)$ $N=500$")],
            [(pU_ana, COL_ANA, r"$ p_{\mathrm{u}}^{\mathrm{SF}}(T)$ $N=20\'000$"),
             (pU_ana2, COL_ANA_small, r"$ p_{\mathrm{u}}^{\mathrm{SF}}(T)$ $N=500$")]
        ),
        (
            [(pC_rb, COL_RB, r"$ p_{\mathrm{c}}^{\mathrm{RB}}(T)$ $N=500$")],
            [(pU_rb, COL_RB, r"$ p_{\mathrm{u}}^{\mathrm{RB}}(T)$ $N=500$")]
        ),
        (
            [(pC_rb, COL_RB, r"$ p_{\mathrm{c}}^{\mathrm{RB}}(T)$ $N=500$"),
             (pC_ana2, COL_ANA_small, r"$ p_{\mathrm{c}}^{\mathrm{SF}}(T)$ $N=500$")],
            [(pU_rb, COL_RB, r"$ p_{\mathrm{u}}^{\mathrm{RB}}(T)$ $N=500$"),
             (pU_ana2, COL_ANA_small, r"$ p_{\mathrm{u}}^{\mathrm{SF}}(T)$ $N=500$")]
        ),
    ]

    for pus, pcs in l:
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), sharex=True, sharey=True, dpi=dpi, )
        for ax in axes:
            ax.set_xlabel(r"$T$ days since $t_\star$")
            ax.set_ylabel("Probability")
            ax.set_xlim(T_grid[0], T_grid[-1])
            ax.set_ylim(0.0, 1.0)

        ax = axes[1]
        ax.set_title(r"Conditional")
        for pu, col, label in pus: ax.plot(T_grid, pu, color=col, lw=2.2, label=label)
        ax.legend(frameon=False, loc="lower right")

        ax = axes[0]
        ax.set_title(r"Unconditional")
        for pc, col, label in pcs:   ax.plot(T_grid, pc, color=col, lw=2.2, label=label)
        ax.legend(frameon=False, loc="lower right")

        plt.tight_layout()
        plt.show()


def plot_all_with_probability_bands3(
        stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_grid, t_star, h=0.2,
        dpi=260
):
    # Baselines
    pU_ana = analytic_uncond_over(stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=h, U_max=T_grid[-1])
    pC_ana = analytic_cond_over(stopped_pairs, T_grid, t_star)

    # RB UNCONDITIONAL — posterior-collapsed

    COL_EMP = '#4C72B0'
    COL_ANA = '#E69F00'
    COL_RB = '#333333'
    COL_SIM = '#E69F00'
    COL_HPD = '#66c2a5'

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), sharex=True, sharey=True, dpi=dpi)

    # (a) Unconditional
    ax = axes[0]
    # ax.plot(T_grid, pU_emp, color=COL_EMP, lw=1.6, ls=(0, (2.6, 1.4, 1, 2.7, 1, 1.4)), zorder=4,
    #         dash_capstyle='butt',
    #         solid_capstyle='butt',
    #         label=r"$\widehat p_{\mathrm{uncond}}^{\mathrm{(E)}}(T)$")
    ax.plot(T_grid, pU_ana, color=COL_RB, lw=2.2,
            label=r"$ p_{\mathrm{uncond}}^{\mathrm{SF}}(T)$ $N=20\'000$",
            zorder=3, dash_capstyle='butt', solid_capstyle='butt')

    ax.set_title(r"Unconditional")
    ax.set_xlabel(r"$T$ days since $t_\star$")
    ax.set_ylabel("Probability")
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")

    # (b) Conditional
    ax = axes[1]

    ax.plot(T_grid, pC_ana, color=COL_RB, lw=2.2, dash_capstyle='butt',
            solid_capstyle='butt',
            label=r"$ p_{\mathrm{cond}}^{\mathrm{SF}}(T)$ $N=20\'000$",
            zorder=3)

    ax.set_title(r"Conditional")
    ax.set_xlabel(r"$T$ days since $t_\star$")
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    # fig.savefig(f"../img/rb_probability_over_20k.pgf")
    plt.show()


# ---------- Plot 1: two-panel figure ----------

if False:
    mpl.use('pgf')
    mpl.rcParams.update({
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.texsystem': 'pdflatex',
        'pgf.preamble': r'''
            \usepackage{amsmath}
            \usepackage{amsfonts}
            \usepackage{siunitx}
            \usepackage[T1]{fontenc}
            \usepackage{lmodern}
        ''',

        # Fonts
        'font.family': 'serif',  # or 'sans-serif' to match your journal
        'font.size': 12,
        # 'axes.titlesize': 8,
        # 'axes.labelsize': 8,
        # 'legend.fontsize': 7,
        # 'xtick.labelsize': 7,
        # 'ytick.labelsize': 7,

        # Lines & ticks (subtle but crisp)
        'lines.linewidth': 1.0,
        'lines.markersize': 3.0,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,

        # Grid (light, y-only by default)
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3,

        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Save tight
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.unicode_minus': False,

        'axes.prop_cycle': mpl.cycler(color=['#4C72B0', '#333333', '#E69F00']),
    })


def apply_presentation_style(
        *,
        base_font=14,
        title_size=18,
        label_size=16,
        tick_size=12,
        line_width=2.4,
        marker_size=7,
        grid_alpha=0.28,
        fig_dpi=140,
):
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex": False,
        "font.size": base_font,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "lines.linewidth": line_width,
        "lines.markersize": marker_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": grid_alpha,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": fig_dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def apply_journal_style(
        *,
        column: str = "single",  # "single" or "double"
        base_font: int = 12,  # main font size (pt)
        title_size: Optional[int] = None,
        label_size: Optional[int] = None,
        tick_size: Optional[int] = None,
        line_width: float = 1.0,
        marker_size: float = 4.0,
        use_mathtext_latex: bool = False,
        show_grid: bool = False,
        grid_alpha: float = 0.10,
        grid_linewidth: float = 0.5,
        color_cycle: Optional[list] = None,
        serif_family: str = "serif",
):
    """
    Apply a clean, publication-style rcParams set.
    - column: pick "single" (~8.6 cm) or "double" (~17.8 cm) for suggested default figure sizes.
    - use_mathtext_latex: if True, sets mpl to use 'text.usetex' (requires LaTeX installed).
      Default False to avoid external dependency; Matplotlib mathtext looks fine for most journals.
    """
    if title_size is None:
        title_size = base_font + 2
    if label_size is None:
        label_size = base_font + 1
    if tick_size is None:
        tick_size = max(8, base_font - 1)

    # colorblind-friendly default (Paul Tol / ColorBrewer inspired)
    if color_cycle is None:
        color_cycle = [
            "#0072B2",  # blue
            "#D55E00",  # orange
            "#009E73",  # green
            "#CC79A7",  # pink
            "#F0E442",  # yellow
            "#56B4E9",  # light blue
            "#E69F00",  # orange 2
            "#000000",  # black
        ]

    import matplotlib as mpl
    # set LaTeX usage
    mpl.rcParams["text.usetex"] = bool(use_mathtext_latex)
    # If not using full LaTeX, prefer Computer Modern mathtext for consistent math font
    mpl.rcParams["mathtext.fontset"] = "cm" if not use_mathtext_latex else "dejavusans"

    # fonts: prefer serif for journals
    mpl.rcParams["font.family"] = serif_family
    mpl.rcParams["font.size"] = base_font
    mpl.rcParams["axes.titlesize"] = title_size
    mpl.rcParams["axes.labelsize"] = label_size
    mpl.rcParams["xtick.labelsize"] = tick_size
    mpl.rcParams["ytick.labelsize"] = tick_size
    mpl.rcParams["legend.fontsize"] = max(8, base_font - 1)
    mpl.rcParams["lines.linewidth"] = line_width
    mpl.rcParams["lines.markersize"] = marker_size

    # axes / ticks / spines: full box and ticks inward (common in publications)
    mpl.rcParams["axes.spines.top"] = True
    mpl.rcParams["axes.spines.right"] = True
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.spines.bottom"] = True
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.major.size"] = 4.0
    mpl.rcParams["ytick.major.size"] = 4.0

    # grid: off by default (journals prefer clean axes); keep tiny grid option
    mpl.rcParams["axes.grid"] = bool(show_grid)
    mpl.rcParams["grid.alpha"] = grid_alpha
    mpl.rcParams["grid.linewidth"] = grid_linewidth

    # legend
    mpl.rcParams["legend.frameon"] = False  # journals often prefer no heavy box; set True if required
    mpl.rcParams["legend.framealpha"] = 0.9

    # savefig / vector output helpers (embed fonts)
    mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts (good for embedding in PDFs)
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.02

    # color cycle
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color_cycle)

    # dpi: leave default (user passes when saving); but many journals accept vector PDF so DPI less critical
    # minor aesthetics
    mpl.rcParams["axes.titleweight"] = "normal"
    mpl.rcParams["axes.labelweight"] = "normal"


# apply_journal_style()
apply_presentation_style()

plot_all_with_probability_bands2(infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=0.2,
                                 dpi=260)
# results = plot_all_with_probability_bands(
#     infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas,
#     T_grid, t_star,
#     h=0.2, H_pad=10.0,
#
#     # show_simultaneous=True, level_sim=0.3,
#     dpi=300
# )

exit()

pars = (Parameters(
    R0=(0.25, 15),
    k=(0.2, 10),
    r=(0.01, 0.99),
    alpha=(0.01, 20),
    theta=(0.01, 20)
).require('R0 * r < 3')
        .require('3 < alpha * theta').require('alpha * theta < 20')
        .require('1/sqrt(alpha) >= 0.1').require('1/sqrt(alpha) <= 0.9')
        .require('1 <= sqrt(alpha) * theta').require('sqrt(alpha) * theta <= 15'))

obs_points = [
    (datetime(2025, 3, 6), 1),
    (datetime(2025, 3, 21), 3),
    (datetime(2025, 3, 25), 1),
    (datetime(2025, 3, 26), 1),
    (datetime(2025, 3, 30), 1),
    (datetime(2025, 4, 2), 2),
    (datetime(2025, 4, 17), 1),
]
sim_start = min(t for t, _ in obs_points)
sim = Simulator(
    parameters=pars,
    sampler=pars.create_latin_hypercube_sampler(),
    start_date=sim_start,
    scenario=Scenario([
        # ParameterChangePoint('R0', datetime(2025, 4, 14), '0.5 * R0'),  # rábapordány
        # ParameterChangePoint('R0', datetime(2025, 4, 17))
    ]),
    criteria=[IndexOffspringCriterion(2, 5)] + build_acceptance_inequalities(
        obs_points=obs_points,
        simulation_start=sim_start,
        sigma_days=3.5138784768847935,
        beta=0.5448705790647912,
        neighbor_weight=0.3259905700578138,
        grid_step_days=0.29607190994514637,
        min_seg_days=1.1388690204854988,
        kmax=4,
        baseline_p=0.07466034737192961,
        alpha=0.3887850260822494,
        h_max=0.16418840893328812,
        eps_share=0.0999993345398409,
        include_gap_windows=False,
        include_union_windows=True,
        max_unions_to_keep=6,
        gap_scale=0.3312202106199904,
        mode='cluster'
    )
    # [
    #     IntervalCriterion(datetime(2025, 3, 3), datetime(2025, 4, 4), 8, 10),
    #     IntervalCriterion(datetime(2025, 4, 1), datetime(2025, 4, 4), 1, 10),
    #     # IntervalCriterion(datetime(2025, 4, 4), datetime(2025, 4, 14), 0, 0),
    #     # IntervalCriterion(datetime(2025, 4, 14), datetime(2025, 4, 17), 1, 1),
    #     # IndexOffspringCriterion(2, 5)
    # ]
    ,
    collectors=[
        draws := DrawCollector(),
        active_set_size := ActiveSetSizeCollector(datetime(2025, 4, 17)),
        infection_times_collector := InfectionTimeCollector(),
    ],
    num_trajectories=100_000_000,
    chunk_size=100_000,
    T_run=70,
    max_cases=1000,
    max_workers=13,
)

now = time.time()
sim.run()
print((datetime(2025, 4, 17) - datetime(2025, 3, 6)).days)
print('Runtime:', time.time() - now)
print('accepted:', len(np.asarray(draws)))

infection_times2 = infection_times_collector.infection_times
stopped_pairs2 = active_set_size.active_sets
R0s2, ks2, rs2, alphas2, thetas2 = np.asarray(draws).T

T_max = sim.T_run
T_grid = np.arange(0.0, T_max + 1e-9, 1.0)
h = 0.2  # small step for Volterra recursion (smoother H)
U_max = T_max  # horizon for H(u) grid
M = len(stopped_pairs)
t_star = (active_set_size.collection_date - sim.start_date).days

plot_all_with_probability_bands2(
    stopped_pairs2, R0s2, rs2, ks2, alphas2, thetas2,
    stopped_pairs, R0s, rs, ks, alphas, thetas,
    T_grid, t_star,

)

plot_all_with_probability_bands3(
    stopped_pairs2, R0s2, rs2, ks2, alphas2, thetas2, T_grid, t_star,
)
exit()
# --------

R_eff = R0s * rs
df = pd.DataFrame({
    "R": R_eff,
    "k": ks,
    "alpha": alphas,
    "theta": thetas,
    "meanGI": alphas * thetas
})

# --- Select representative parameter draws by quantiles of (R, k) ---
R_targets = np.quantile(df["R"], [0.10, 0.50, 0.90])
k_targets = np.quantile(df["k"], [0.25, 0.75])

selected_idx = []
for Rt in R_targets:
    for kt in k_targets:
        dist = (df["R"] - Rt) ** 2 + (df["k"] - kt) ** 2
        # Bias tie-breaking toward median meanGI to avoid pathological tails
        cand = dist.idxmin()
        selected_idx.append(cand)

selected_idx = sorted(set(int(i) for i in selected_idx))  # unique, sorted


# --- Helper: BGW ever-extinction probability q (NB offspring) ---
def bgw_ext_prob(R, k, tol=1e-12, itmax=2000):
    # pgf G(s) = (beta / (beta + 1 - s))^k, beta = k/R
    if R <= 1.0:
        return 1.0
    beta = k / R
    q = 1.0  # monotone iteration from 1
    for _ in range(itmax):
        q_new = (beta / (beta + 1.0 - q)) ** k
        if abs(q_new - q) < tol:
            return float(q_new)
        q = q_new
    return float(q)


# --- Compute H(u) grids for selected draws ---
U_plot = T_max  # same horizon as before, in days
h_H = 0.2  # fine step
uu = np.arange(0, U_plot + 1e-9, h_H)

curves = []
for idx in selected_idx:
    R, k, a, th = float(df.at[idx, "R"]), float(df.at[idx, "k"]), float(df.at[idx, "alpha"]), float(df.at[idx, "theta"])
    H = compute_H_grid(R, k, a, th, U_max=U_plot, h=h_H)
    q = bgw_ext_prob(R, k)
    curves.append({
        "idx": idx,
        "R": R, "k": k, "alpha": a, "theta": th, "meanGI": a * th,
        "H": H, "q": q
    })

# --- Plot with your PGF/TeX style (assumes rcParams already set) ---
fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.4))

# Hand-picked color list (print-friendly, consistent with your palette)
palette = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
ls_for_k = {"low": "-", "high": "--"}

# Decide which k is "low/high" relative to sample quantiles for linestyles
k_q25, k_q75 = k_targets[0], k_targets[1]

for j, c in enumerate(curves):
    col = palette[j % len(palette)]
    # linestyle by k (optional subtle encoding)
    ls = ls_for_k["low"] if c["k"] <= k_q25 else (ls_for_k["high"] if c["k"] >= k_q75 else "-.")
    ax.plot(uu, c["H"], color=col, lw=1.4, ls='-',  # ls=ls,
            label=rf"$R_{{\mathrm{{eff}}}}={c['R']:.2f},\ k={c['k']:.2f},\ "
                  rf"\alpha={c['alpha']:5.2f},\ \theta={c['theta']:5.2f}$"
            )
    # Mark H(0)
    H0 = (c["k"] / (c["k"] + c["R"])) ** c["k"]
    ax.plot([0], [H0], marker='o', ms=4.0, color=col)
    # Asymptote q (thin)
    ax.hlines(c["q"], 0, U_plot, color=col, lw=1, linestyles=':', alpha=0.7)

ax.set_xlabel(r"$u$ (days since seed)")
ax.set_ylabel(r"$H(u)$")
ax.set_xlim(0, U_plot)
ax.set_ylim(-0.02, 1.02)
ax.set_title(r"Single-seed finite-horizon survival $H(u)$")
ax.legend(frameon=False, loc="lower right", ncol=1, fontsize=10)

plt.tight_layout()
plt.show()
# plt.savefig("../img/single_seed_H.pgf")
