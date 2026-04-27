from typing import Dict, Tuple, List

import numpy as np
from scipy.special import gamma as gamma_func, gammainc


def empirical_uncond_over(infection_times, T_grid, t_star):
    last_time = np.array([
        np.max(np.asarray(times, dtype=float)) if len(times) > 0 else -np.inf
        for times in infection_times
    ], dtype=float)
    p = np.array([np.mean(last_time <= t_star + T) for T in T_grid], dtype=float)
    M = len(infection_times)
    se = np.sqrt(np.clip(p * (1 - p) / max(M, 1), 0.0, 1.0))
    return p, se


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
    arr = np.asarray(arr, dtype=float)
    N = arr.size - 1
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, float(N))
    i = np.floor(x).astype(int)
    w = x - i
    ip1 = np.minimum(i + 1, N)
    return (1.0 - w) * arr[i] + w * arr[ip1]


def compute_H_grid(R, k, a, th, U_max=80.0, h=0.2):
    beta = k / max(R, 1e-12)
    N = int(np.round(U_max / h))
    t = np.arange(0, (N + 1) * h, h)
    f = gamma_pdf(t, a, th)
    H = np.empty(N + 1, dtype=float)
    H[0] = (beta / (beta + 1.0)) ** k
    for n in range(1, N + 1):
        conv = h * float(np.dot(f[1:n + 1], H[n - 1::-1]))
        denom = max(beta + 1.0 - conv, 1e-300)
        H[n] = (beta / denom) ** k  # !!! potential overflow if denom tiny
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


def analytic_uncond_over(stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=0.2, U_max=80.0):
    M = len(stopped_pairs)
    # per-trajectory seeds
    seeds_by = []
    for m in range(M):
        arr = np.array(stopped_pairs[m], dtype=float).reshape(-1, 2)
        if arr.size == 0:
            seeds_by.append(np.empty(0))
            continue
        mask = (arr[:, 0] <= t_star) & (arr[:, 1] > t_star)
        seeds_by.append(np.sort(arr[mask, 1]))

    # cache H per unique parameter combo
    key_to_indices: Dict[Tuple[float, float, float, float, float], List[int]] = {}
    for m in range(M):
        key = (
            round(float(R0s[m] * rs[m]), 6),
            round(float(ks[m]), 6),
            round(float(alphas[m]), 6),
            round(float(thetas[m]), 6),
            h,
        )
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
                p += 1.0
                continue
            max_s = float(seeds[-1])
            mask = horizons >= max_s
            if not np.any(mask):
                continue
            u = horizons[mask][None, :] - seeds[:, None]
            prod = np.prod(H_eval_vec(H, h, u), axis=0)
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
            first_post[m] = ts[j]
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
            p[j] = np.nan
            se[j] = np.nan
    return p, se


def analytic_cond_over(stopped_pairs, T_grid, t_star):
    M = len(stopped_pairs)
    any_seed = np.zeros(M, dtype=bool)
    first_seed = np.full(M, np.inf, dtype=float)
    for m, pairs in enumerate(stopped_pairs):
        arr = np.array(pairs, dtype=float).reshape(-1, 2)
        if arr.size == 0:
            continue
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


def _align_common_prefix(*arrays):
    lens = [len(a) for a in arrays]
    M = int(min(lens))
    if len(set(lens)) > 1:
        print(f"[RB] Warning: mismatched lengths {tuple(lens)} -> trimming to M={M}")
    return [a[:M] for a in arrays] + [M]


def _count_pre_children_for_parents_fast(stopped_pairs_m, parents, t_star):
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
    infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, M = _align_common_prefix(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas
    )
    T_fine = np.arange(0.0, T_max + 1e-12, h, dtype=float)
    NT = T_fine.size
    X_max = float(t_star + T_max)

    at_keys = {(round(float(a), 6), round(float(th), 6), h, round(X_max, 6)) for a, th in zip(alphas, thetas)}
    at_cache = {k: gamma_cdf_grid(k[0], k[1], X_max, h) for k in at_keys}

    H_cache: Dict[Tuple[float, float, float, float, float, float], np.ndarray] = {}
    for m in range(M):
        R_post = float(R0s[m] * rs[m])
        keyH = (
            round(R_post, 6), round(float(ks[m]), 6),
            round(float(alphas[m]), 6), round(float(thetas[m]), 6),
            h, round(T_max + H_pad, 6)
        )
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
        J_max = NT
        j_grid = np.arange(1, J_max + 1, dtype=float)
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
