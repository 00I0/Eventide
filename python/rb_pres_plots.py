# =========================
# Presentation style + 3 plots (SF vs RB)
# =========================
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_func, gammainc

from python.eventide import DrawCollector, ActiveSetSizeCollector, InfectionTimeCollector, Scenario, Simulator, \
    IndexOffspringCriterion, Parameters
from python.optimize_acceptance_windows import build_acceptance_inequalities


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


def _choose_indices(M: int, n: int, seed: int | None = 123) -> np.ndarray:
    n_eff = min(int(n), int(M))
    if n_eff <= 0:
        return np.zeros(0, dtype=int)
    rng = np.random.default_rng(seed)
    return np.arange(M, dtype=int) if n_eff == M else rng.choice(M, size=n_eff, replace=False)


def _sf_curves_for_subset(
        idxs: np.ndarray,
        *,
        stopped_pairs,
        R0s, rs, ks, alphas, thetas,
        T_grid, t_star,
        h: float = 0.2
):
    """Analytical SF using ONLY the subset (filters seeds parent<=t*<child inside analytic_*)."""
    sp_sub = [stopped_pairs[i] for i in idxs]
    pU = analytic_uncond_over(sp_sub, R0s[idxs], rs[idxs], ks[idxs], alphas[idxs], thetas[idxs],
                              T_grid, t_star, h=h, U_max=float(T_grid[-1]))
    pC = analytic_cond_over(sp_sub, T_grid, t_star)
    return np.clip(pU, 0.0, 1.0), np.clip(pC, 0.0, 1.0)


def _rb_curves_mean_for_subset(
        idxs: np.ndarray,
        *,
        infection_times, stopped_pairs,
        R0s, rs, ks, alphas, thetas,
        T_grid, t_star,
        h: float = 0.2,
        H_pad: float = 10.0
):
    """
    Compute RB per-draw matrices once, then average on the given subset.
    (Exact same RB implementation as your validated code.)
    """
    # Unconditional RB per-draw matrix on T_grid
    T_fine, pU_rb_mean_exact, gU_uncond_full = rao_blackwell_uncond_over_post_full(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_max=float(T_grid[-1]), t_star=t_star, h=h, H_pad=H_pad
    )
    pU_draws = rb_draws_uncond_full_to_grid(T_fine, gU_uncond_full, T_grid)

    # Conditional RB per-draw matrix on T_grid
    gC_inf, gC_quiet = rb_cond_components_post(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=h
    )
    pC_draws = rb_draws_cond_from_components(gC_inf, gC_quiet)

    # Subset means
    pU_rb = np.nanmean(pU_draws[idxs, :], axis=0)
    pC_rb = np.nanmean(pC_draws[idxs, :], axis=0)
    return np.clip(pU_rb, 0.0, 1.0), np.clip(pC_rb, 0.0, 1.0)


def _plot_two_panel(
        *,
        T_grid,
        pU_sf_main=None,  # solid
        pC_sf_main=None,
        pU_sf_small=None,  # dashed
        pC_sf_small=None,
        pU_rb=None,  # solid
        pC_rb=None,
        title_suffix="",
        ylim=(0, 1)
):
    apply_presentation_style()

    COL_SF = "#1f77b4"  # same hue for both SF lines
    COL_RB = "#333333"

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharex=True, sharey=True)

    # ---------- Unconditional ----------
    ax = axes[0]
    ax.set_title(f"Unconditional{title_suffix}")
    ax.set_xlabel(r"$T$ days since $t_\star$")
    ax.set_ylabel("Probability")

    handles, labels = [], []

    if pU_rb is not None:
        ln, = ax.plot(T_grid, pU_rb, color=COL_RB, lw=2.2,
                      label=r"$p_{\mathrm{uncond}}^{\mathrm{RB}}$")
        handles.append(ln);
        labels.append(ln.get_label())

    if pU_sf_main is not None:
        ln, = ax.plot(T_grid, pU_sf_main, color=COL_SF, lw=2.2,
                      label=r"$p_{\mathrm{uncond}}^{\mathrm{SF}}$")
        handles.append(ln);
        labels.append(ln.get_label())

    if pU_sf_small is not None:
        ln, = ax.plot(T_grid, pU_sf_small, color=COL_SF, lw=2.2, ls="--", dashes=(6, 4),
                      label=r"$p_{\mathrm{uncond}}^{\mathrm{SF}}$")
        handles.append(ln);
        labels.append(ln.get_label())

    # de-dup legend labels
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l);
            H.append(h);
            L.append(l)
    if L: ax.legend(H, L, loc="lower right", frameon=False)

    ax.set_xlim(float(T_grid[0]), float(T_grid[-1]))
    ax.set_ylim(*ylim)

    # ---------- Conditional ----------
    ax = axes[1]
    ax.set_title(f"Conditional{title_suffix}")
    ax.set_xlabel(r"$T$ days since $t_\star$")

    handles, labels = [], []

    if pC_rb is not None:
        ln, = ax.plot(T_grid, pC_rb, color=COL_RB, lw=2.2,
                      label=r"$p_{\mathrm{cond}}^{\mathrm{RB}}$")
        handles.append(ln);
        labels.append(ln.get_label())

    if pC_sf_main is not None:
        ln, = ax.plot(T_grid, pC_sf_main, color=COL_SF, lw=2.2,
                      label=r"$p_{\mathrm{cond}}^{\mathrm{SF}}$")
        handles.append(ln);
        labels.append(ln.get_label())

    if pC_sf_small is not None:
        ln, = ax.plot(T_grid, pC_sf_small, color=COL_SF, lw=2.2, ls="--", dashes=(6, 4),
                      label=r"$p_{\mathrm{cond}}^{\mathrm{SF}}$")
        handles.append(ln);
        labels.append(ln.get_label())

    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l);
            H.append(h);
            L.append(l)
    if L: ax.legend(H, L, loc="lower right", frameon=False)

    ax.set_xlim(float(T_grid[0]), float(T_grid[-1]))
    ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()


def plot_sf_rb_three_figures(
        *,
        infection_times, stopped_pairs,
        R0s, rs, ks, alphas, thetas,
        T_grid, t_star,
        n_high: int = 20_000,
        n_low: int = 600,
        seed: int | None = 123,
        h: float = 0.2,
        H_pad: float = 10.0,
        ylim=(0, 1)
):
    """
    Build three presentation-style figures for the *last snapshot*:
      1) SF only (N=20k).
      2) SF(20k) + SF(600 dashed).
      3) SF(600 dashed) + RB(600).
    """
    M = len(stopped_pairs)
    idx_hi = _choose_indices(M, n_high, seed=seed)
    idx_lo = _choose_indices(M, n_low, seed=seed)

    # SF curves
    pU_sf_hi, pC_sf_hi = _sf_curves_for_subset(
        idx_hi,
        stopped_pairs=stopped_pairs,
        R0s=R0s, rs=rs, ks=ks, alphas=alphas, thetas=thetas,
        T_grid=T_grid, t_star=t_star, h=h
    )
    pU_sf_lo, pC_sf_lo = _sf_curves_for_subset(
        idx_lo,
        stopped_pairs=stopped_pairs,
        R0s=R0s, rs=rs, ks=ks, alphas=alphas, thetas=thetas,
        T_grid=T_grid, t_star=t_star, h=h
    )

    # RB curves (use the same 600 subset)
    pU_rb_lo, pC_rb_lo = _rb_curves_mean_for_subset(
        idx_lo,
        infection_times=infection_times, stopped_pairs=stopped_pairs,
        R0s=R0s, rs=rs, ks=ks, alphas=alphas, thetas=thetas,
        T_grid=T_grid, t_star=t_star, h=h, H_pad=H_pad
    )

    # --- Fig 1: SF only (20k) ---
    _plot_two_panel(
        T_grid=T_grid,
        pU_sf_main=pU_sf_hi, pC_sf_main=pC_sf_hi,
        title_suffix=f"  (SF only; N={len(idx_hi)})",
        ylim=ylim
    )

    # --- Fig 2: SF(20k) + SF(600 dashed) ---
    _plot_two_panel(
        T_grid=T_grid,
        pU_sf_main=pU_sf_hi, pC_sf_main=pC_sf_hi,
        pU_sf_small=pU_sf_lo, pC_sf_small=pC_sf_lo,
        title_suffix=f"  (SF: 20{chr(8239)}000 solid + 600 dashed)",
        ylim=ylim
    )

    # --- Fig 3: SF(600 dashed) + RB(600) ---
    _plot_two_panel(
        T_grid=T_grid,
        pU_sf_small=pU_sf_lo, pC_sf_small=pC_sf_lo,
        pU_rb=pU_rb_lo, pC_rb=pC_rb_lo,
        title_suffix=f"  (SF dashed + RB solid; N={len(idx_lo)})",
        ylim=ylim
    )


# =========================
# CALL IT (after your simulation & arrays are populated)
# =========================

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

    ,
    collectors=[
        draws := DrawCollector(),
        active_set_size := ActiveSetSizeCollector(datetime(2025, 4, 17)),
        infection_times_collector := InfectionTimeCollector(),
    ],
    num_trajectories=1_35_000_000,
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

infection_times = infection_times_collector.infection_times
stopped_pairs = active_set_size.active_sets
R0s, ks, rs, alphas, thetas = np.asarray(draws).T

T_max = sim.T_run
T_grid = np.arange(0.0, T_max + 1e-9, 1.0)
h = 0.2  # small step for Volterra recursion (smoother H)
U_max = T_max  # horizon for H(u) grid
M = len(stopped_pairs)
t_star = (active_set_size.collection_date - sim.start_date).days

apply_presentation_style()
plot_sf_rb_three_figures(
    infection_times=infection_times,
    stopped_pairs=stopped_pairs,
    R0s=R0s, rs=rs, ks=ks, alphas=alphas, thetas=thetas,
    T_grid=T_grid, t_star=t_star,
    n_high=20_000,  # 1) SF only
    n_low=600,  # 2) SF dashed, 3) SF dashed + RB
    seed=123,
    h=0.2,
    H_pad=10.0,
    ylim=(0, 1)
)
