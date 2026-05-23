# ===== Presentation-oriented plotting + time-path analysis =====
# Drop this near your plotting section. No pgf, no saving, unified style.

from __future__ import annotations

import time
from dataclasses import dataclass
from math import log, ceil, floor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Rectangle, Patch
from scipy.interpolate import PchipInterpolator
from scipy.special import gamma as gamma_func, gammainc
from scipy.stats import norm  # for truncated Gaussian KDE

from python.on_the_fly import rao_blackwell_uncond_over_post_full, rb_draws_uncond_full_to_grid, \
    rb_cond_components_post, rb_draws_cond_from_components


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


# ----------------------------------------------------------------------
# Style (edit once, all figures inherit)
# ----------------------------------------------------------------------
def apply_presentation_style(
        *,
        base_font=14,
        title_size=18,
        label_size=16,
        tick_size=12,
        line_width=2.4,
        marker_size=7,
        grid_alpha=0.28,
        fig_dpi=300,
):
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex": False,
        # "font.family": "cambria",
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

    color_cycle = [
        '#85BAA1',  # blue
        "#7B9292",  # orange
        "#6E6C82",  # green
        "#6B6690",  # pink
        "#7080BC",  # yellow
        "#6F9CEB",  # light blue
    ]

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color_cycle)


def cm2inch(x: float) -> float:
    return float(x) / 2.54


def apply_journal_style(
        *,
        column: str = "single",  # "single" or "double"
        base_font: int = 10,  # main font size (pt)
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


def default_figsize_for_column(column: str = "single", *, aspect: Tuple[float, float] = (4.0, 3.0)) -> Tuple[
    float, float]:
    """
    Return a default figsize in inches suited to the selected column width.
    - single -> ~8.6 cm width (common single-column width)
    - double -> ~17.8 cm width (common double-column width)
    aspect = (width_ratio, height_ratio) can be used to pick a different aspect.
    """
    if column == "single":
        width_cm = 8.6
    elif column == "double":
        width_cm = 17.8
    else:
        raise ValueError("column must be 'single' or 'double'")
    width_in = cm2inch(width_cm)
    # pick aspect ratio: by default use landscape-ish; allow override via aspect param
    w_ratio, h_ratio = aspect
    height_in = width_in * (h_ratio / w_ratio)
    return (width_in, height_in)


# ----------------------------------------------------------------------
# SnapshotResult: add infection_times_2d so we can time-condition draws
# ----------------------------------------------------------------------
@dataclass()
class SnapshotResult:
    m: Optional[int] = None
    t_star: Optional[float] = None
    T_grid: Optional[np.ndarray] = None
    horizon_days_from_start: Optional[float] = None
    acceptance_horizon_days_from_start: Optional[float] = None
    p_uncond_mean: Optional[np.ndarray] = None
    p_cond_mean: Optional[np.ndarray] = None
    p_uncond_draws: Optional[np.ndarray] = None  # (M_accept, nT)
    p_cond_draws: Optional[np.ndarray] = None  # (M_accept, nT)
    draws_array: Optional[np.ndarray] = None  # (M_accept, 5) [R0,k,r,alpha,theta]
    infection_times_2d: Optional[List[np.ndarray]] = None  # per-trajectory infection times (days since start_date)
    n_obs: Optional[int] = None
    next_T: Optional[float] = None
    stopped_pairs: List[Sequence[Tuple[float, float]]] | None = None


# ----------------------------------------------------------------------
# === MOD: run_snapshot now stores infection_times_2d in SnapshotResult ===
# Keep your internal ABC/RB logic as-is; only the return is extended.
# ----------------------------------------------------------------------
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
                 H_pad: float = 10.0,
                 min_required: Optional[int] = None) -> SnapshotResult:
    """
    Build acceptance from the first m observations, run ABC, compute RB curves on T_grid.
    Horizons are relative to that snapshot's t_star (last observed infection time).
    """
    # --- your existing helpers ---
    from python.eventide import Simulator, Scenario, DrawCollector, ActiveSetSizeCollector, InfectionTimeCollector
    from python.eventide import IndexOffspringCriterion
    from python.optimize_acceptance_windows import build_acceptance_inequalities

    def _cast_accept_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
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

    # === build acceptance + run sim ===
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
        min_required=min_required
    )
    now = time.time()
    sim.run()
    print('Run', m, 'took', time.time() - now, 'seconds')

    infection_times_2d = list(itimes.infection_times)  # <<< keep per-trajectory infection times

    stopped_pairs = active_set.active_sets
    R0s, ks, rs, alphas, thetas = np.asarray(draws).T
    t_star = (active_set.collection_date - sim.start_date).days

    # --- your existing RB computations (unchanged) ---
    # (Assumes gamma_pdf, rb_* utilities already defined in your file.)

    Tf, pU_mean_exact, gU_draws_fine = rao_blackwell_uncond_over_post_full(
        infection_times_2d, stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_max=float(T_grid[-1]), t_star=t_star, h=h, H_pad=H_pad
    )
    p_uncond_mean = np.interp(T_grid, Tf, pU_mean_exact)
    p_uncond_draws = rb_draws_uncond_full_to_grid(Tf, gU_draws_fine, T_grid)

    gC_inf, gC_quiet = rb_cond_components_post(
        infection_times_2d, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=h
    )
    p_cond_mean = (gC_inf.mean() / gC_quiet.mean(axis=0)) if gC_quiet.size else np.full_like(T_grid, np.nan)
    p_cond_draws = rb_draws_cond_from_components(gC_inf, gC_quiet) if gC_quiet.size else np.empty((0, T_grid.size))

    print('Run', m, 'accepted', len(infection_times_2d))

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
        infection_times_2d=infection_times_2d,  # <<< NEW
        next_T=next_T,
        n_obs=n_obs,
        stopped_pairs=stopped_pairs
    )


# ----------------------------------------------------------------------
# Shifted online probabilities (curves start only when previous dot appears)
# ----------------------------------------------------------------------
def _segment_offsets(results: Sequence[SnapshotResult]) -> List[float]:
    """Offset (in days) for each snapshot so curve m starts after the previous dot."""
    offs = []
    acc = 0.0
    for r in results:
        offs.append(acc)
        if r.next_T is not None:
            acc += float(r.next_T)
    return offs


def plot_rb_online_two_pane_shifted(results: Sequence[SnapshotResult],
                                    *,
                                    show_next_dot=True,
                                    ylim=(0, 1)):
    """
    Multi-snapshot online probabilities, but on a single absolute axis:
    each curve is shifted right by the sum of previous next_T's and thus
    only "starts" when the previous dot has happened.
    """
    if not results:
        return

    offsets = _segment_offsets(results)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=False, sharey=True)
    titles = ["Unconditional", "Conditional"]
    series = [("p_uncond_mean", 0), ("p_cond_mean", 1)]

    for name, col in series:
        ax = axes[col]
        ax.set_title(titles[col])
        ax.set_xlabel("Elapsed days since first $t_\\star$")
        if col == 0:
            ax.set_ylabel("Probability")
        ax.set_ylim(*ylim)

        for r, x0 in zip(results, offsets):
            y = getattr(r, name)
            x = x0 + r.T_grid
            (line,) = ax.plot(x, y, label=r'$N_\text{obs}=' + str(r.n_obs) + '$')
            if show_next_dot and (r.next_T is not None) and (r.T_grid[0] <= r.next_T <= r.T_grid[-1]):
                xd = x0 + r.next_T
                yd = float(np.interp(r.next_T, r.T_grid, y))
                colr = line.get_color()
                ax.scatter([xd], [yd], s=53, facecolors=colr, edgecolors=colr, linewidths=1.2,
                           zorder=line.get_zorder() + 1)
        if col == 1:
            ax.legend(loc="lower right", frameon=False)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    plt.show()


# ----------------------------------------------------------------------
# Posterior 2x4 for LAST snapshot only, with boundary-corrected KDE
# ----------------------------------------------------------------------
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


def _scott_bw(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    n = max(1, x.size)
    if n <= 1:
        return 1.0
    sigma = np.std(x, ddof=1) if n > 1 else 1.0
    return sigma * n ** (-1 / 5)


def _kde_truncated_gaussian(x: np.ndarray, grid: np.ndarray, lo: float, hi: float, bw: float) -> np.ndarray:
    """
    Truncated (renormalized) Gaussian KDE on [lo, hi], prevents artificial thin tails at hard cutoffs.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    grid = np.asarray(grid, float)
    if x.size == 0:
        return np.zeros_like(grid)
    h = max(1e-12, float(bw))

    X = x[:, None]  # (n,1)
    G = grid[None, :]  # (1,m)
    K = norm.pdf((G - X) / h) / h  # (n,m)

    # per-sample mass inside [lo, hi]
    Ci = norm.cdf((hi - x) / h) - norm.cdf((lo - x) / h)  # (n,)
    Ci = np.maximum(Ci, 1e-15)
    weights = 1.0 / Ci
    f = (K * weights[:, None]).mean(axis=0)

    f[(grid < lo) | (grid > hi)] = 0.0
    # final numeric renorm
    mask = (grid >= lo) & (grid <= hi)
    area = np.trace(f[mask], grid[mask])
    if area > 0:
        f /= area
    return f


def _collect_vars(res: SnapshotResult) -> Dict[str, np.ndarray]:
    arr = np.asarray(res.draws_array)
    R0 = arr[:, 0]
    k = arr[:, 1]
    r = arr[:, 2]
    a = arr[:, 3]
    th = arr[:, 4]
    Re = r * R0
    alpha_theta = a * th
    p0_Re = (k / (k + Re)) ** k

    def good(v):
        v = np.asarray(v)
        return v[np.isfinite(v)]

    return dict(R0=good(R0), r=good(r), alpha=good(a), theta=good(th),
                k=good(k), Re=good(Re), alpha_theta=good(alpha_theta), p0_Re=good(p0_Re))


def _default_supports() -> Dict[str, Tuple[float, float]]:
    # Based on your declared constraints
    R0_rng = (0.25, 15.0)
    r_rng = (0.01, 0.99)
    a_rng = (0.01, 20.0)
    th_rng = (0.01, 20.0)
    k_rng = (0.2, 10.0)
    Re_rng = (0.0, 3.0)  # R0*r < 3
    at_rng = (3.0, 20.0)  # 3 < alpha*theta < 20
    p0_rng = (0.0, 1.0)
    return dict(R0=R0_rng, r=r_rng, alpha=a_rng, theta=th_rng,
                k=k_rng, Re=Re_rng, alpha_theta=at_rng, p0_Re=p0_rng)


def plot_posterior_grid_single(res: SnapshotResult,
                               *,
                               mass=0.95,
                               bw_adjust=1.0,
                               n_grid=600):
    """
    2x4 panels for the LAST snapshot only.
    - no N_obs labels
    - boundary-corrected KDE (truncated Gaussian) to avoid fake thin tails at hard cutoffs
    """
    sv = _collect_vars(res)
    supports = _default_supports()

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

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.5))
    axes = axes.reshape(2, 4)

    for idx, (key, title) in enumerate(var_specs):
        ax = axes.flat[idx]
        x = sv.get(key, np.array([]))
        lo, hi = supports[key]
        x = x[(x >= lo) & (x <= hi)]
        ax.set_title(title)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.grid(True, axis="x")

        if x.size < 2:
            continue

        h = bw_adjust * _scott_bw(x)
        grid = np.linspace(lo, hi, n_grid)
        pdf = _kde_truncated_gaussian(x, grid, lo, hi, h)
        hdi_lo, hdi_hi = _support_interval(x, mass=mass)
        hdi_lo = max(hdi_lo, lo)
        hdi_hi = min(hdi_hi, hi)

        mask = (grid >= hdi_lo) & (grid <= hdi_hi)
        ax.plot(grid, pdf, lw=1.6, alpha=0.95)
        ax.fill_between(grid[mask], 0.0, pdf[mask], alpha=0.35)

        med = float(np.median(x))
        med = min(max(med, lo), hi)
        j = int(np.clip(np.searchsorted(grid, med), 0, len(grid) - 1))
        ax.plot([med, med], [0.0, pdf[j]], lw=2.2, alpha=0.6)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Time-path of r*R0: decreases while "quiet", jumps at each observed case
# ----------------------------------------------------------------------
def _first_post_after_tstar_per_draw(res: SnapshotResult) -> np.ndarray:
    ts = float(res.t_star)
    out = np.empty(len(res.infection_times_2d), dtype=float)
    for i, traj in enumerate(res.infection_times_2d):
        t = np.asarray(traj, float)
        rel = t - ts
        rel = rel[rel > 0.0]
        out[i] = rel[0] if rel.size else np.inf
    return out


def _summarize(vals: np.ndarray, band: tuple[float, float], agg="median") -> tuple[float, float, float]:
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    qlo, qhi = np.quantile(vals, band)
    if agg == "mean":
        mid = float(np.mean(vals))
    elif callable(agg):
        mid = float(agg(vals))
    else:
        mid = float(np.median(vals))
    return mid, float(qlo), float(qhi)


def plot_timepath_Re(
        results: Sequence[SnapshotResult],
        *,
        step: float = 0.25,
        band: tuple[float, float] = (0.2, 0.8),  # central 60%
        summary: str | callable = "median",
        draw_verticals: bool = True,
        clip_Re_max: float = 3.0,  # prior: R0*r < 3
        ensure_upward_jumps: bool = False,  # optional safety if you want hard monotone up-jumps
):
    if not results:
        return

    offsets = _segment_offsets(results)

    # Precompute per-snapshot
    payload = []
    for res in results:
        R0 = res.draws_array[:, 0]
        r = res.draws_array[:, 2]
        Re = R0 * r
        first_post = _first_post_after_tstar_per_draw(res)
        payload.append((res, Re, first_post))

    all_x, all_mid, all_lo, all_hi = [], [], [], []
    jump_locs = []
    prev_end_mid = None

    tiny = 1e-9

    for (res, Re, first_post), x0 in zip(payload, offsets):
        # Segment length: [0, next_T) (right-open). If none, show up to horizon.
        if res.next_T is not None:
            T_end = float(res.next_T)
            open_end = True
        else:
            T_end = 20
            open_end = False
        if T_end <= 0:
            continue

        # Build evaluation grid. If right-open, stop just before T_end.
        if open_end:
            n_steps = max(1, int(np.floor((T_end - tiny) / step)) + 1)
            T_eval = np.linspace(0.0, max(0.0, T_end - tiny), n_steps)
        else:
            T_eval = np.arange(0.0, T_end + tiny, step)

        mids, los, his = [], [], []

        for j, T in enumerate(T_eval):
            if j == 0:
                # Unconditional reset exactly at the event: use ALL draws for this snapshot
                vals = Re
            else:
                # Quiet up to T: first post-snapshot infection strictly after T
                vals = Re[first_post > T]

            m, lo, hi = _summarize(vals, band, agg=summary)

            # Optional enforcement: do not allow a down-jump at the boundary
            if ensure_upward_jumps and j == 0 and prev_end_mid is not None and np.isfinite(
                    prev_end_mid) and np.isfinite(m):
                if m < prev_end_mid:
                    m = prev_end_mid

            # clip for safety
            if np.isfinite(m):  m = float(np.clip(m, 0.0, clip_Re_max))
            if np.isfinite(lo): lo = float(np.clip(lo, 0.0, clip_Re_max))
            if np.isfinite(hi): hi = float(np.clip(hi, 0.0, clip_Re_max))

            mids.append(m);
            los.append(lo);
            his.append(hi)

        x = x0 + T_eval
        all_x.append(x)
        all_mid.append(np.asarray(mids))
        all_lo.append(np.asarray(los))
        all_hi.append(np.asarray(his))

        # Remember end-of-segment median (to compare with next segment's start)
        if len(mids):
            prev_end_mid = mids[-1]

        if res.next_T is not None:
            jump_locs.append(x0 + float(res.next_T))

    # Concatenate
    X = np.concatenate(all_x) if all_x else np.array([])
    MID = np.concatenate(all_mid) if all_mid else np.array([])
    LO = np.concatenate(all_lo) if all_lo else np.array([])
    HI = np.concatenate(all_hi) if all_hi else np.array([])

    # Single figure / single axes
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    ax.set_title(r"Effective reproduction number given no unobserved infections")
    ax.set_xlabel(r"Elapsed days since first $t_\star$")
    ax.set_ylabel(r"$rR_0$")

    # Draw
    if X.size:
        # mask to finite regions only (avoid plotting all-NaN spans)
        good = np.isfinite(MID)
        # Plot band once (label only once)
        (ln,) = ax.plot(X[good], MID[good])
        color = ln.get_color()
        ax.fill_between(X[good], LO[good], HI[good], alpha=0.25, color=color,
                        label=f"central {int((band[1] - band[0]) * 100)}% band")
        ax.plot(X[good], MID[good], color=color, label="median")

    # Jumps
    if draw_verticals:
        for xj in jump_locs:
            ax.axvline(xj, linestyle="--", linewidth=1.2, alpha=0.6, color='#E69F00')

    # Legend (deduplicate)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uniq_h.append(h)
            uniq_l.append(l)

    ax.set_ylim(0, clip_Re_max)
    # x-lim to content
    if X.size:
        ax.set_xlim(0.0, float(np.nanmax(X)) + step)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Bulk driver: run all requested snapshots with per-snapshot kwargs
# ----------------------------------------------------------------------
def run_all_snapshots_per_m(obs_points,
                            pars,
                            builder_kwargs_by_m: Dict[int, Dict[str, Any]],
                            snapshots: Sequence[int],
                            *,
                            num_trajectories: int = 1_000_000,
                            chunk_size: int = 100_000,
                            T_run: int = 70,
                            max_cases: int = 1000,
                            max_workers: int = 12,
                            T_grid: np.ndarray = np.arange(0, 70 + 1e-9, 1.0),
                            h: float = 0.2,
                            H_pad: float = 10.0,
                            min_required: Optional[int] = None) -> List[SnapshotResult]:
    results = []
    for m in (m for m in snapshots if m >= 3):
        kw = builder_kwargs_by_m.get(m, {})
        res = run_snapshot(
            m=m,
            obs_points=obs_points,
            pars=pars,
            builder_kwargs=kw,
            num_trajectories=num_trajectories,
            chunk_size=chunk_size,
            T_run=T_run,
            max_cases=max_cases,
            max_workers=max_workers,
            T_grid=T_grid,
            h=h,
            H_pad=H_pad,
            min_required=min_required,
        )
        results.append(res)
    return results


# ----------------------------------------------------------------------
# Back-compat alias: keep the original name but use the shifted version
# ----------------------------------------------------------------------
def plot_rb_online_two_pane(results: Sequence[SnapshotResult], **kwargs):
    """
    Compatibility wrapper: original name, new 'shifted' behavior so
    curves only begin when the previous dot has occurred.
    """
    return plot_rb_online_two_pane_shifted(results, **kwargs)


# ----------------------------------------------------------------------
# Optional helper: plot only the LAST snapshot's posteriors
# ----------------------------------------------------------------------
def plot_last_posterior(results: Sequence[SnapshotResult], **kwargs):
    if not results:
        return
    return plot_posterior_grid_single(results[-1], **kwargs)


# ================= Cumulative infections: last snapshot, numeric axis =================

def _cum_matrix_from_times(infection_times_2d: Sequence[Sequence[float]],
                           grid_days: np.ndarray) -> np.ndarray:
    """Matrix shape (N_trajectories, len(grid_days)) of cumulative counts."""
    if not infection_times_2d:
        return np.zeros((0, grid_days.size), dtype=float)
    rows = []
    for traj in infection_times_2d:
        t = np.sort(np.asarray(traj, float))
        rows.append(np.searchsorted(t, grid_days, side="right"))
    return np.vstack(rows).astype(float)


def _tail_mean_band(cum_matrix: np.ndarray, p_central: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tail-mean envelope, shrunk toward the mean for small central masses.
    p_central in (0,1]. Returns (lower, upper).
    """
    if cum_matrix.size == 0:
        return np.zeros(0), np.zeros(0)
    if not (0.0 < p_central <= 1.0):
        raise ValueError(f"perc_bands values must be in (0, 1], got {p_central}")
    mu = cum_matrix.mean(axis=0)
    if p_central == 1.0:
        return mu.copy(), mu.copy()

    N = cum_matrix.shape[0]
    t = (1.0 - p_central) / 2.0
    k = max(1, int(np.ceil(N * t)))

    sorted_cols = np.sort(cum_matrix, axis=0)
    lower_raw = sorted_cols[:k, :].mean(axis=0)
    upper_raw = sorted_cols[-k:, :].mean(axis=0)

    beta = p_central ** 0.7  # gentle shrink
    lower = mu + beta * (lower_raw - mu)
    upper = mu + beta * (upper_raw - mu)
    return lower, upper


def plot_cumulative_infections_last_numeric(
        results: Sequence["SnapshotResult"],
        *,
        resolution: float = 0.25,  # days
        perc_bands: Sequence[float] = (0.95, 0.5, 0.2),  # central masses
        cmap: str | Sequence[str] = "PuBu",
        scale: str = "linear",  # "linear" or "log"
        show_mean: bool = True,
        mean_style: Optional[Dict[str, Any]] = None,
        show_median: bool = False,
        median_style: Optional[Dict[str, Any]] = None,
        obs_points_days: Optional[Sequence[Tuple[float, int]]] = None,  # optional incident counts as (day, count)
        figsize: Tuple[float, float] = (12.5, 5.8),
        dpi: int = 300,
):
    """
    Cumulative infections for the LAST snapshot only (numeric x-axis: days since start).

    - Uses all accepted trajectories' infection times in that snapshot.
    - Draws tail-mean bands for given central masses (widest band drawn first).
    - Optional overlays: mean/median.
    - Optional observed incident counts via (day_since_start, count); plotted cumulatively.

    Example:
        plot_cumulative_infections_last_numeric(results, obs_points_days=[(0.0,1),(15.0,3), ...])
    """
    if not results:
        return
    res = results[-1]

    # horizon: cover both RB window and any infections that may extend beyond
    last_inf = max((float(tr[-1]) for tr in res.infection_times_2d if len(tr) > 0), default=0.0)
    rb_horizon = float(res.t_star + (res.T_grid[-1] if res.T_grid.size else 0.0))
    total_days = 55

    grid_days = np.arange(0.0, total_days + 1e-9, float(resolution))
    cum_matrix = _cum_matrix_from_times(res.infection_times_2d, grid_days)

    # colors for bands (draw widest first, under the rest)
    bands = sorted(perc_bands, reverse=True)
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
        positions = np.linspace(0.25, 0.85, num=len(bands)) if len(bands) > 1 else [0.6]
        band_colors = [cmap_obj(p) for p in positions]
    else:
        base = [mcolors.to_rgba(c) for c in cmap]
        if len(base) == 0:
            raise ValueError("Custom color list must contain at least one color.")
        band_colors = (base * (len(bands) // len(base) + 1))[:len(bands)]

    # figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if scale.lower() == "log":
        ax.set_yscale("log")

    ax.set_xlabel("Days since start")
    ax.set_ylabel("Cumulative infections")

    # bands
    computed = []
    for p in bands:
        lo, hi = _tail_mean_band(cum_matrix, p)
        computed.append((lo, hi, p))
    for (lo, hi, p), color in zip(computed, band_colors):
        if lo.size:
            ax.fill_between(grid_days, lo, hi, color=color, alpha=0.35,
                            label=f"Simulation {int(round(p * 100))}% band")

    # overlays
    if show_mean and cum_matrix.size:
        mean_style = mean_style or {"color": "black", "lw": 2.0}
        ax.plot(grid_days, cum_matrix.mean(axis=0), label="Simulation mean", **mean_style)

    if show_median and cum_matrix.size:
        median_style = median_style or {"color": "gray", "lw": 2.0, "ls": "--"}
        ax.plot(grid_days, np.median(cum_matrix, axis=0), label="median", **median_style)

    # optional observed points (incident -> cumulative), numeric days
    if obs_points_days:
        obs_sorted = sorted(obs_points_days, key=lambda x: float(x[0]))
        xs, ys_inc = zip(*obs_sorted)
        ys_cum = np.cumsum(np.asarray(ys_inc, dtype=float))
        ax.scatter(xs, ys_cum, marker="o", s=62, linewidths=1.3, zorder=10,
                   edgecolors="white", color="tab:red", label="Observations")

    # legend de-dupe
    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            H.append(h)
            L.append(l)
    if L:
        ax.legend(H, L, loc="upper left", frameon=False)

    # x-lims to content
    ax.set_xlim(0.0, 55 if grid_days.size else 1.0)
    ax.set_ylim(0.0, 26.5)
    plt.tight_layout()
    plt.show()
    return fig


# ==========================

# ---- helpers shared only by this plot ----
@dataclass(frozen=True)
class _WindowBand:
    start: datetime
    end: datetime
    L: int
    U: int


def _days_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0


def _staircase_cumulative_at(obs_points: Sequence[Tuple[datetime, int]], t: datetime) -> int:
    return int(sum(c for (ti, c) in obs_points if ti <= t))


def _proportional_observed_mass(
        obs_points: Sequence[Tuple[datetime, int]],
        span_start: datetime,
        span_end: datetime,
        a: datetime,
        b: datetime,
) -> float:
    """Observed cases in (a,b], with proportional split if a/b slice a day."""
    if not (a < b):
        return 0.0
    d0 = datetime(span_start.year, span_start.month, span_start.day)
    d1 = datetime(span_end.year, span_end.month, span_end.day) + timedelta(days=1)
    num_days = (d1 - d0).days
    edges = [d0 + timedelta(days=i) for i in range(num_days + 1)]

    by_day = {}
    for t, c in obs_points:
        if t < span_start or t > span_end:
            continue
        key = (t.year, t.month, t.day)
        by_day[key] = by_day.get(key, 0) + int(c)

    total = 0.0
    for i in range(num_days):
        D0, D1 = edges[i], edges[i + 1]
        key = (D0.year, D0.month, D0.day)
        c = float(by_day.get(key, 0))
        overlap = max(0.0, (min(b, D1) - max(a, D0)).total_seconds())
        if overlap > 0:
            total += (overlap / (D1 - D0).total_seconds()) * c
    return total


def _wilson_halfwidth(pi_hat: float, n: int, z: float) -> float:
    if n <= 0:
        return 0.0
    return (z * np.sqrt((pi_hat * (1 - pi_hat)) / n + (z * z) / (4 * n * n))) / (1 + (z * z) / n)


def _band_from_target_and_share(
        M_hat: float,
        Y_hat: float,
        n_total: int,
        baseline_p: float,
        alpha: float,
        h_max: float,
        eps_share: float,
) -> Tuple[int, int, float, float]:
    z = float(norm.ppf(1 - alpha / 2.0))
    pi_hat = (Y_hat / n_total) if n_total > 0 else 0.0
    w = _wilson_halfwidth(pi_hat, n_total, z)
    denom = max(pi_hat, eps_share) if pi_hat > 0 else eps_share
    h = min(h_max, w / denom)
    p = baseline_p + h
    L = max(0, int(floor((1 - p) * M_hat)))
    U = int(ceil((1 + p) * M_hat))
    return L, U, p, h


def _build_cluster_windows(
        obs_points: Sequence[Tuple[datetime, int]],
        sigma_days: float,
        beta: float,
        neighbor_weight: float,
) -> Tuple[List[Tuple[datetime, datetime]], List[Tuple[datetime, datetime]]]:
    """Returns (raw_local_windows, merged_cluster_windows) as lists of (start, end)."""
    obs = sorted([(t, int(c)) for (t, c) in obs_points if int(c) > 0], key=lambda x: x[0])
    times = [t for t, _ in obs]
    z_beta = float(norm.ppf(0.5 + beta / 2.0))
    base_half = z_beta * float(sigma_days)

    raw: List[Tuple[datetime, datetime]] = []

    def gap_days(i: int, j: int) -> Optional[float]:
        if i < 0 or j >= len(times):
            return None
        return _days_between(times[i], times[j])

    for i, (t, _) in enumerate(obs):
        gL = gap_days(i - 1, i)
        gR = gap_days(i, i + 1)
        if gL is None and gR is None:
            local_term = 0.0
        elif gL is None or gR is None:
            local_term = neighbor_weight * (gL or gR) / 2.0
        else:
            local_term = neighbor_weight * (gL + gR) / 4.0
        half = max(base_half, local_term)
        raw.append((t - timedelta(days=half), t + timedelta(days=half)))

    raw_sorted = sorted(raw, key=lambda ab: (ab[0], ab[1]))
    merged: List[Tuple[datetime, datetime]] = []
    s, e = raw_sorted[0]
    for s2, e2 in raw_sorted[1:]:
        if s2 <= e:  # overlap/touch
            e = max(e, e2)
        else:
            merged.append((s, e))
            s, e = s2, e2
    merged.append((s, e))
    return raw_sorted, merged


def _segment_interpolant(
        obs_points: Sequence[Tuple[datetime, int]],
        grid_step_days: float,
        min_seg_days: float,
        kmax: int,
) -> Tuple[List[datetime], PchipInterpolator, np.ndarray, np.ndarray]:
    """Monotone interpolant of cumulative data; segment via DP polyline with BIC-like score."""
    obs = sorted([(t, int(c)) for (t, c) in obs_points if int(c) > 0], key=lambda x: x[0])
    times = [t for t, _ in obs]
    counts = [c for _, c in obs]
    A = times[0]
    x_obs = np.array([_days_between(A, t) for t in times], dtype=float)
    cum = np.cumsum(counts).astype(float)
    f = PchipInterpolator(x_obs, cum, axis=0)

    step = float(grid_step_days)
    x_dense = np.arange(float(x_obs.min()), float(x_obs.max()) + 1e-9, step)
    f_dense = f(x_dense)

    min_len_pts = max(2, int(round(min_seg_days / step)))

    def seg_cost_sse(i: int, j: int) -> float:
        xi, xj = x_dense[i], x_dense[j]
        yi, yj = f_dense[i], f_dense[j]
        h = max(xj - xi, 1e-12)
        t = (x_dense[i:j + 1] - xi) / h
        y_lin = yi + (yj - yi) * t
        r = f_dense[i:j + 1] - y_lin
        return float(np.sum(r * r))

    def dp_polyline(K: int) -> Tuple[List[Tuple[int, int]], float]:
        n = len(x_dense)
        INF = 1e100
        dp = np.full((K, n), INF)
        prev = np.full((K, n), -1, dtype=int)

        for j in range(min_len_pts, n):
            dp[0, j] = seg_cost_sse(0, j)
            prev[0, j] = 0

        for k in range(1, K):
            for j in range(min_len_pts * (k + 1), n):
                best, arg = INF, -1
                i_min = k * min_len_pts
                for i in range(i_min, j - min_len_pts + 1):
                    val = dp[k - 1, i] + seg_cost_sse(i, j)
                    if val < best:
                        best, arg = val, i
                dp[k, j] = best
                prev[k, j] = arg

        segs: List[Tuple[int, int]] = []
        k = K - 1
        j = n - 1
        while k >= 0:
            i = prev[k, j]
            if i < 0:
                break
            segs.append((i, j))
            j = i
            k -= 1
        segs.append((0, j))
        segs.sort()
        return segs, float(dp[K - 1, n - 1])

    # choose K by BIC-like score
    best = None
    n = len(x_dense)
    for K in range(1, kmax + 1):
        segs, sse = dp_polyline(K)
        peff = 2 * K
        score = n * log(max(sse / max(n, 1), 1e-12)) + peff * log(n)
        if best is None or score < best[0]:
            best = (score, segs)

    segs_opt = best[1]
    idx_bounds = [segs_opt[0][0]] + [j for (_, j) in segs_opt]
    tau_days = [float(x_dense[idx]) for idx in idx_bounds]
    A_dt = obs[0][0]
    tau_dt = [A_dt + timedelta(days=d) for d in tau_days]
    return tau_dt, f, x_dense, f_dense


def plot_acceptance_construction_side_by_side(
        obs_points: Sequence[Tuple[datetime, int]],
        *,
        # Cluster controls
        sigma_days: float = 1.8,
        beta: float = 0.75,
        neighbor_weight: float = 0.6,
        # Interpolant controls
        grid_step_days: float = 0.2,
        min_seg_days: float = 1.0,
        kmax: int = 3,
        # Bands (common)
        baseline_p: float = 0.10,
        alpha: float = 0.10,
        h_max: float = 0.30,
        eps_share: float = 1e-6,
        # Quiet windows
        include_quiet_windows: bool = True,
        gap_scale: float = 0.4,
        # Optional: close short gaps after merging (days). 0 → off.
        close_gaps_days: float = 0.0,
        # Visual
        paren_inset_frac: float = 0.06,
        figsize: Tuple[float, float] = (12.5, 4.8),
        dpi: int = 300,
):
    """
    Presentation-ready side-by-side construction figure:
      Left: cluster-merging windows (+ quiet gaps, Wilson bands, raw local parentheses).
      Right: interpolant-segmentation windows (+ Wilson bands on segments).
    - X axis is numeric: days since first observation (consistent with your other plots).
    """

    obs = sorted([(t, int(c)) for (t, c) in obs_points if int(c) > 0], key=lambda x: x[0])
    if not obs:
        raise ValueError("obs_points must have at least one positive-count entry.")
    times = [t for t, _ in obs]
    counts = [c for _, c in obs]
    A, B = times[0], times[-1]
    n_total = int(sum(counts))

    # --------- cluster route ----------
    raw_local, clusters = _build_cluster_windows(obs, sigma_days, beta, neighbor_weight)
    base_windows_cluster = clusters[:]

    # closing short gaps if requested
    if close_gaps_days > 0 and len(base_windows_cluster) >= 2:
        merged2 = []
        s, e = base_windows_cluster[0]
        for s2, e2 in base_windows_cluster[1:]:
            gap = _days_between(e, s2)
            if gap <= close_gaps_days:
                e = max(e, e2)
            else:
                merged2.append((s, e))
                s, e = s2, e2
        merged2.append((s, e))
        base_windows_cluster = merged2

    cluster_bands: List[_WindowBand] = []
    for (a, b) in base_windows_cluster:
        M_hat = _proportional_observed_mass(obs, A, B, a, b)
        L, U, _, _ = _band_from_target_and_share(M_hat, M_hat, n_total, baseline_p, alpha, h_max, eps_share)
        cluster_bands.append(_WindowBand(a, b, L, U))

    # quiet windows inside gaps
    quiet_cluster: List[Tuple[datetime, datetime]] = []
    if include_quiet_windows and len(base_windows_cluster) >= 2:
        base_lengths = np.array([_days_between(a, b) for (a, b) in base_windows_cluster], dtype=float)
        med_len = float(np.median(base_lengths)) if len(base_lengths) > 0 else 1.0
        for (a1, b1), (a2, b2) in zip(base_windows_cluster, base_windows_cluster[1:]):
            if a2 <= b1:
                continue
            gap_len = _days_between(b1, a2)
            center = b1 + (a2 - b1) / 2
            half_days = max(0.0, min(gap_scale * gap_len, 0.5 * med_len))
            if half_days > 0:
                qa, qb = center - timedelta(days=half_days), center + timedelta(days=half_days)
                qa = max(qa, A)
                qb = min(qb, B)
                if qa < qb:
                    quiet_cluster.append((qa, qb))

    # color dots by merged cluster id
    def cluster_idx_for_time(t):
        for k, (a, b) in enumerate(base_windows_cluster):
            if a <= t <= b:
                return k
        return -1

    # --------- segmentation route ----------
    tau_dt, f_interp, _, _ = _segment_interpolant(obs, grid_step_days, min_seg_days, kmax)
    base_windows_seg = [(tau_dt[i], tau_dt[i + 1]) for i in range(len(tau_dt) - 1)]

    def Nref(t: datetime) -> float:
        return float(f_interp(_days_between(A, t)))

    seg_bands: List[_WindowBand] = []
    for (a, b) in base_windows_seg:
        M_hat = Nref(b) - Nref(a)
        if (b <= a) or (abs(M_hat) < 1e-9):
            continue
        Y_hat = _proportional_observed_mass(obs, A, B, a, b)
        L, U, _, _ = _band_from_target_and_share(M_hat, Y_hat, n_total, baseline_p, alpha, h_max, eps_share)
        seg_bands.append(_WindowBand(a, b, L, U))

    # --------- numeric X (days since first obs) ----------
    def xday(t: datetime) -> float:
        return _days_between(A, t)

    # --------- figure ----------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharey=True)

    COL_BAND = "#4C72B0"  # acceptance band
    COL_POINTS = "#333333"  # observed points
    COL_PAREN = "#6b6b6b"  # raw local window parentheses
    COL_QUIET = "#E69F00"  # quiet windows
    COL_INTERP = "#2E7D32"  # interpolant/polyline

    # LEFT: Cluster–merging
    axL.set_title("Cluster–merging")
    for a, b in quiet_cluster:
        axL.axvspan(xday(a), xday(b), color=COL_QUIET, alpha=0.28, zorder=0.5)

    t_pts = [t for t, _ in obs]
    cum_pts = [_staircase_cumulative_at(obs, t) for t in t_pts]
    colors = [plt.get_cmap("tab10")(cluster_idx_for_time(t) % 10) for t in t_pts]
    axL.scatter([xday(t) for t in t_pts], cum_pts, s=53, c=colors, zorder=2.6,
                edgecolor="white", linewidths=1.2, label="observed cumulative (points)")

    # raw-local parentheses around each dot
    def draw_parentheses(ax, tdt, y, a_left, b_right):
        width_days = max(xday(b_right) - xday(a_left), 1e-9)
        inset = paren_inset_frac * width_days
        a_in = xday(a_left) + inset
        b_in = xday(b_right) - inset
        ax.hlines(y, a_in, xday(tdt), colors=COL_PAREN, linestyles="-", linewidth=0.9, alpha=0.25, zorder=2.3)
        ax.hlines(y, xday(tdt), b_in, colors=COL_PAREN, linestyles="-", linewidth=0.9, alpha=0.25, zorder=2.3)
        ax.text(a_in, y, "(", ha="center", va="center", fontsize=12, color=COL_PAREN, alpha=0.9, zorder=2.4)
        ax.text(b_in, y, ")", ha="center", va="center", fontsize=12, color=COL_PAREN, alpha=0.9, zorder=2.4)

    for (t, _), (a_raw, b_raw) in zip(obs, raw_local):
        y = _staircase_cumulative_at(obs, t)
        draw_parentheses(axL, t, y, a_raw, b_raw)

    def centered_band_rect_num(ax, a, b, center_y, L, U, k_index=None):
        height = max(U - L, 1e-9)
        y0 = center_y - height / 2.0
        ax.add_patch(Rectangle(
            (xday(a), y0),
            xday(b) - xday(a),
            height,
            facecolor=COL_BAND, edgecolor='none', alpha=0.35, linewidth=0.9, zorder=1.5
        ))
        xr = xday(b) - 0.02 * max(xday(b) - xday(a), 1e-6)
        if k_index is None:
            ax.text(xr, y0 + 0.02 * height, f"L={L}", ha="right", va="bottom", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0))
            ax.text(xr, y0 + height + 0.06 * height, f"U={U}", ha="right", va="top", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0))
        else:
            ax.text(xr, y0 + 0.02 * height, f"L{k_index}={L}", ha="right", va="bottom", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0))
            ax.text(xr, y0 + height + 0.06 * height, f"U{k_index}={U}", ha="right", va="top", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0))

    # bands centered at mean of observed cum points inside each cluster
    for k, w in enumerate(cluster_bands, start=1):
        vals = [_staircase_cumulative_at(obs, t) for t in times if (w.start < t <= w.end)]
        if len(vals) == 0:
            center_y = (_staircase_cumulative_at(obs, w.start) +
                        0.5 * _proportional_observed_mass(obs, A, B, w.start, w.end))
        else:
            center_y = float(np.mean(vals))
        centered_band_rect_num(axL, w.start, w.end, center_y, w.L, w.U, k_index=k)

    axL.set_xlabel("Days since first observation")
    axL.set_ylabel("Cumulative cases")
    axL.legend(handles=[
        Patch(facecolor=COL_QUIET, edgecolor='none', label="quiet window", alpha=0.28),
        Patch(facecolor=COL_BAND, edgecolor='none', label="acceptance band [L,U]", alpha=0.35),
        # Line2D([], [], color=COL_PAREN, lw=1.0, label="raw local window (parentheses)"),
        # Line2D([], [], marker="o", lw=0, markersize=6, markerfacecolor=COL_POINTS,
        #        markeredgecolor="white", label="observed cumulative (points)"),
    ], frameon=False, loc="upper left")

    # RIGHT: Interpolant–segmentation
    axR.set_title("Interpolant–segmentation")
    x_plot = np.linspace(0.0, _days_between(A, B), 400)
    axR.plot(x_plot, [float(f_interp(d)) for d in x_plot],
             color=COL_INTERP, lw=2.0, zorder=2.0, label="interpolant")

    for (a, b) in base_windows_seg:
        if b <= a:
            continue
        xa, xb = xday(a), xday(b)
        ya, yb = float(f_interp(xa)), float(f_interp(xb))
        axR.plot([xa, xb], [ya, yb], color="#BB4430", lw=2.0, alpha=0.9, zorder=2.1)

    # bands centered at median(interpolant) on each segment
    for k, w in enumerate(seg_bands, start=1):
        x0, x1 = xday(w.start), xday(w.end)
        xs = np.linspace(x0, x1, 201)
        ys = f_interp(xs)
        center_y = float(np.median(ys))
        centered_band_rect_num(axR, w.start, w.end, center_y, w.L, w.U, k_index=k)

    axR.set_xlabel("Days since first observation")
    axR.legend(frameon=False, loc="upper left")

    # shared y-lims padding
    for ax in (axL, axR):
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(0.0, ymin), ymax * 1.05)

    plt.tight_layout()
    plt.show()


# =========== Probability-over (Analytic vs RB) — last snapshot, 3 versions ===========
# Uses the exact SF/RB math as validated elsewhere. No empirical curves shown.

import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Optional


# RB machinery comes from your validated module (already imported above):
#   rao_blackwell_uncond_over_post_full, rb_draws_uncond_full_to_grid,
#   rb_cond_components_post, rb_draws_cond_from_components
#
# SF analytics (analytic_uncond_over / analytic_cond_over) are defined earlier
# in this file exactly as in your reference; we call those here.

def _choose_indices(M: int, n: int, seed: int | None = 123) -> np.ndarray:
    n_eff = min(int(n), int(M))
    if n_eff <= 0:
        return np.zeros(0, dtype=int)
    rng = np.random.default_rng(seed)
    return np.arange(M, dtype=int) if n_eff == M else rng.choice(M, size=n_eff, replace=False)


def _infer_stopped_pairs_from_infections(
        infection_times_2d: Sequence[Sequence[float]],
        t_star: float
) -> list[list[tuple[float, float]]]:
    """
    Fallback reconstruction of crossing seeds (parent <= t* < child) from infection times.
    """
    out: list[list[tuple[float, float]]] = []
    ts_star = float(t_star)
    for traj in infection_times_2d:
        ts = np.sort(np.asarray(traj, float))
        pairs: list[tuple[float, float]] = []
        if ts.size >= 2:
            post = np.where(ts > ts_star)[0]
            for j in post:
                if j == 0:
                    continue
                if ts[j - 1] <= ts_star:
                    pairs.append((float(ts[j - 1]), float(ts[j])))
        out.append(pairs)
    return out


def _analytic_curves_from_subset_stopped(
        res: "SnapshotResult",
        idxs: np.ndarray,
        *,
        h: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Seed-factorized analytics for ONLY the subset (parent<=t*<child filtering, as in the original).
    Uses res.stopped_pairs if present; otherwise infers them from infection_times (warns once).
    """
    # Parameters subset
    R0s = res.draws_array[idxs, 0]
    ks = res.draws_array[idxs, 1]
    rs = res.draws_array[idxs, 2]
    alps = res.draws_array[idxs, 3]
    ths = res.draws_array[idxs, 4]

    # stopped_pairs subset (or fallback inference)
    if getattr(res, "stopped_pairs", None) is not None:
        sp_sub = [res.stopped_pairs[i] for i in idxs]
    else:
        warnings.warn("stopped_pairs not found on SnapshotResult; inferring crossing seeds from infection_times.",
                      RuntimeWarning)
        inferred = _infer_stopped_pairs_from_infections(res.infection_times_2d, res.t_star)
        sp_sub = [inferred[i] for i in idxs]

    # Analytic SF curves (EXACT reference implementation; functions defined earlier)
    pU_sf = analytic_uncond_over(
        sp_sub, R0s, rs, ks, alps, ths,
        res.T_grid, res.t_star, h=h, U_max=float(res.T_grid[-1])
    )
    pC_sf = analytic_cond_over(sp_sub, res.T_grid, res.t_star)

    return np.clip(pU_sf, 0.0, 1.0), np.clip(pC_sf, 0.0, 1.0)


def _rb_curves_from_subset(res: "SnapshotResult", idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    RB mean curves from the subset using the stored per-draw matrices (exact same RB logic as reference).
    """
    if res.p_uncond_draws.size == 0 or res.p_cond_draws.size == 0:
        raise ValueError("SnapshotResult must contain RB per-draw matrices (p_uncond_draws / p_cond_draws).")
    pU_rb = np.nanmean(res.p_uncond_draws[idxs, :], axis=0)
    pC_rb = np.nanmean(res.p_cond_draws[idxs, :], axis=0)
    return np.clip(pU_rb, 0.0, 1.0), np.clip(pC_rb, 0.0, 1.0)


def _plot_prob_over_two_panel_multi(
        T_grid: np.ndarray,
        *,
        # SF curves
        pU_sf_main: Optional[np.ndarray] = None,  # solid (e.g., N=20k)
        pC_sf_main: Optional[np.ndarray] = None,
        pU_sf_small: Optional[np.ndarray] = None,  # dashed (e.g., N=600)
        pC_sf_small: Optional[np.ndarray] = None,
        # RB curves (solid)
        pU_rb: Optional[np.ndarray] = None,
        pC_rb: Optional[np.ndarray] = None,
        title_suffix: str = '',
        ylim: tuple[float, float] = (0.0, 1.0),
):
    """
    Draw unconditional | conditional panels with:
      - SF main (solid), SF small (dashed), RB (solid).
    Legend labels (latex), exactly as requested:
      p^RB_uncond, p^RB_cond, p^SF_uncond, p^SF_cond.
    """
    COL_SF = "#1f77b4"  # SF color for both solid & dashed
    COL_RB = "#333333"  # RB color

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharex=True, sharey=True)

    # ---------------- Unconditional ----------------
    ax = axes[0]
    ax.set_title(f"Unconditional{title_suffix}")
    ax.set_xlabel(r"$T$ days since $t_\star$")
    ax.set_ylabel("Probability")

    handles, labels = [], []

    if pU_rb is not None:
        h_rb, = ax.plot(T_grid, pU_rb, lw=2.2, color=COL_RB,
                        label=r"$p_{\mathrm{uncond}}^{\mathrm{RB}}$")
        handles.append(h_rb);
        labels.append(h_rb.get_label())

    if pU_sf_main is not None:
        h_sf_main, = ax.plot(T_grid, pU_sf_main, lw=2.2, color=COL_SF,
                             label=r"$p_{\mathrm{uncond}}^{\mathrm{SF}}$")
        handles.append(h_sf_main);
        labels.append(h_sf_main.get_label())

    if pU_sf_small is not None:
        h_sf_small, = ax.plot(T_grid, pU_sf_small, lw=2.2, color=COL_SF,
                              ls="--", dashes=(6, 4),
                              label=r"$p_{\mathrm{uncond}}^{\mathrm{SF}}$")
        handles.append(h_sf_small);
        labels.append(h_sf_small.get_label())

    # de-dup legend entries while preserving order
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l);
            H.append(h);
            L.append(l)

    ax.set_xlim(float(T_grid[0]), float(T_grid[-1]))
    ax.set_ylim(*ylim)
    if L: ax.legend(H, L, frameon=False, loc="lower right")

    # ---------------- Conditional ----------------
    ax = axes[1]
    ax.set_title(f"Conditional{title_suffix}")
    ax.set_xlabel(r"$T$ days since $t_\star$")

    handles, labels = [], []

    if pC_rb is not None:
        h_rb, = ax.plot(T_grid, pC_rb, lw=2.2, color=COL_RB,
                        label=r"$p_{\mathrm{cond}}^{\mathrm{RB}}$")
        handles.append(h_rb);
        labels.append(h_rb.get_label())

    if pC_sf_main is not None:
        h_sf_main, = ax.plot(T_grid, pC_sf_main, lw=2.2, color=COL_SF,
                             label=r"$p_{\mathrm{cond}}^{\mathrm{SF}}$")
        handles.append(h_sf_main);
        labels.append(h_sf_main.get_label())

    if pC_sf_small is not None:
        h_sf_small, = ax.plot(T_grid, pC_sf_small, lw=2.2, color=COL_SF,
                              ls="--", dashes=(6, 4),
                              label=r"$p_{\mathrm{cond}}^{\mathrm{SF}}$")
        handles.append(h_sf_small);
        labels.append(h_sf_small.get_label())

    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l);
            H.append(h);
            L.append(l)

    ax.set_xlim(float(T_grid[0]), float(T_grid[-1]))
    ax.set_ylim(*ylim)
    if L: ax.legend(H, L, frameon=False, loc="lower right")

    plt.tight_layout()
    plt.show()


def plot_probability_over_last_versions_rb_vs_analytic(
        results: Sequence["SnapshotResult"],
        *,
        n_high: int = 20_000,  # Fig 1 & 2: SF(20k)
        n_low: int = 600,  # Fig 2 & 3: SF(600) and RB(600)
        seed: int | None = 123,
        h: float = 0.2,
):
    """
    Make three figures for the LAST snapshot:
      1) SF only (N=20k).
      2) SF (N=20k) + SF (N=600, dashed).
      3) SF (N=600, dashed) + RB (N=600).
    Legends show only: p^RB_uncond, p^RB_cond, p^SF_uncond, p^SF_cond.
    """
    if not results:
        raise ValueError("Empty results.")
    res = results[-1]
    M = res.draws_array.shape[0]

    # Choose subsets
    idx_hi = _choose_indices(M, n_high, seed=seed)
    idx_lo = _choose_indices(M, n_low, seed=seed)

    # SF (seed-factorized) curves
    pU_sf_hi, pC_sf_hi = _analytic_curves_from_subset_stopped(res, idx_hi, h=h)
    pU_sf_lo, pC_sf_lo = _analytic_curves_from_subset_stopped(res, idx_lo, h=h)

    # RB curves (use the SAME subset size as the dashed SF: 600)
    pU_rb_lo, pC_rb_lo = _rb_curves_from_subset(res, idx_lo)

    _plot_prob_over_two_panel_multi(
        res.T_grid,
        pU_sf_main=pU_sf_hi, pC_sf_main=pC_sf_hi,
    )

    _plot_prob_over_two_panel_multi(
        res.T_grid,
        pU_sf_main=pU_sf_hi, pC_sf_main=pC_sf_hi,
        pU_sf_small=pU_sf_lo, pC_sf_small=pC_sf_lo,
    )

    _plot_prob_over_two_panel_multi(
        res.T_grid,
        pU_sf_small=pU_sf_lo, pC_sf_small=pC_sf_lo,
        pU_rb=pU_rb_lo, pC_rb=pC_rb_lo,
    )


def plot_cumulative_timepath_clean(
        results: Sequence[SnapshotResult],
        *,
        n_T_per_snapshot: int = 6,  # how many conditioning times T to show per snapshot (incl. T=0)
        pick_T_mode: str = "linspace",  # "linspace" or "quantiles" (quantiles uses first_post distribution)
        central_p_uncond: float = 0.95,  # shaded band for unconditional curve
        resolution: float = 0.5,  # grid spacing (days) for cumulative curves
        u_horizon: float = 55.0,  # days after t_star to show
        show_uncond_only_band: bool = True,  # draw band only for T==0 (avoid clutter)
        panels: bool = False,  # if True, create one subplot per snapshot
        figsize: Tuple[float, float] = (12.8, 5.2),
):
    """
    Cleaner time-path for mean cumulative infections.
    - Shows a limited number of conditional curves per snapshot (n_T_per_snapshot).
    - Emphasizes unconditional (T==0) curve + optional band.
    - Optionally create one panel per snapshot.
    """
    import matplotlib.pyplot as plt
    if not results:
        return

    offsets = _segment_offsets(results)
    # precompute first_post per-draw
    payload = []
    for res in results:
        first_post = _first_post_after_tstar_per_draw(res)
        payload.append((res, first_post))

    # u grid
    u_grid = np.arange(0.0, float(u_horizon) + 1e-12, float(resolution))

    n_snap = len(results)
    # colors
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_snap)]

    # panels vs single axes
    if panels:
        cols = min(3, n_snap)
        rows = (n_snap + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows / 1.6), squeeze=False)
        axes = axes.flat
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax] * n_snap  # reuse the same axis

    all_x_max = 0.0
    jump_locs = []

    for i, ((res, first_post), x0) in enumerate(zip(payload, offsets)):
        ax_i = axes[i] if panels else axes[0]
        col = colors[i]

        # choose T_eval samples
        if res.next_T is not None:
            Tmax = float(res.next_T)
        else:
            Tmax = min(float(res.T_grid[-1]) if res.T_grid.size else u_horizon, u_horizon)

        if Tmax <= 0:
            continue

        # sample times: include 0 and Tmax-epsilon and some interior points
        if n_T_per_snapshot <= 1:
            T_samples = np.array([0.0])
        else:
            if pick_T_mode == "linspace":
                T_samples = np.linspace(0.0, max(1e-8, Tmax), n_T_per_snapshot)
            elif pick_T_mode == "quantiles":
                # use quantiles of first_post (only finite) to pick Ts that are representative
                finite_fp = first_post[np.isfinite(first_post)]
                if finite_fp.size >= (n_T_per_snapshot - 1):
                    qs = np.linspace(0.0, 1.0, n_T_per_snapshot)
                    T_samples = np.quantile(finite_fp, qs)
                    T_samples = np.clip(T_samples, 0.0, Tmax)
                    T_samples[0] = 0.0
                else:
                    T_samples = np.linspace(0.0, max(1e-8, Tmax), n_T_per_snapshot)
            else:
                T_samples = np.linspace(0.0, max(1e-8, Tmax), n_T_per_snapshot)

        # ensure unique and sorted
        T_samples = np.unique(np.clip(T_samples, 0.0, Tmax))

        # Plot each selected T (but draw band only for T==0 unless requested)
        for j, T in enumerate(T_samples):
            if j == 0:
                sel_mask = np.ones(len(res.infection_times_2d), dtype=bool)
            else:
                sel_mask = first_post > T

            # build chosen trajectories relative to t_star
            chosen_trajs = []
            for sel, traj in zip(sel_mask, res.infection_times_2d):
                if not sel:
                    continue
                if len(traj) == 0:
                    chosen_trajs.append(np.empty(0))
                    continue
                arr = np.asarray(traj, dtype=float) - float(res.t_star)
                arr = arr[arr >= 0.0]
                chosen_trajs.append(arr)

            if len(chosen_trajs) == 0:
                continue

            cum_matrix = _cum_matrix_from_times(chosen_trajs, u_grid)

            if cum_matrix.size == 0:
                continue

            mid = cum_matrix.mean(axis=0)
            # shading only for T==0 (unconditional) by default
            if show_uncond_only_band and j == 0:
                lo, hi = _tail_mean_band(cum_matrix, central_p_uncond)
            else:
                lo, hi = None, None

            x_curve = x0 + T + u_grid

            # style: unconditional emphasized
            if j == 0:
                ax_i.plot(x_curve, mid, lw=2.6, color=col, label=f"m={res.m}")
                if lo is not None:
                    ax_i.fill_between(x_curve, lo, hi, color=col, alpha=0.20)
                # marker at start
                ax_i.scatter([x0], [float(mid[0])], s=60, facecolors='white', edgecolors=col, linewidths=1.6, zorder=4)
            else:
                alpha_line = 0.5 * (1.0 - float(j) / max(1.0, len(T_samples)))
                ax_i.plot(x_curve, mid, lw=1.1, color=col, alpha=alpha_line)

            all_x_max = max(all_x_max,
                            float(np.nanmax(x_curve[np.isfinite(mid)])) if np.any(np.isfinite(mid)) else all_x_max)

        if res.next_T is not None:
            jump_locs.append(x0 + float(res.next_T))

        # formatting per-axis
        if panels:
            ax_i.set_title(f"Snapshot m={res.m}")
            ax_i.set_xlabel("Elapsed days since first $t_\\star$")
            ax_i.set_ylabel("Mean cumulative infections" if i % max(1, int(len(axes) / len(axes))) == 0 else "")
            ax_i.set_xlim(0.0, max(all_x_max + 1.0, 1.0))
            ymin, ymax = ax_i.get_ylim()
            ax_i.set_ylim(max(0.0, ymin), ymax * 1.05)
            ax_i.grid(True)
            if i == 0:
                ax_i.legend(frameon=False, loc="upper left")
    # shared decorations if single axis
    if not panels:
        ax = axes[0]
        for xj in jump_locs:
            ax.axvline(xj, linestyle="--", linewidth=1.2, alpha=0.6, color='#E69F00')
        ax.set_xlim(0.0, max(all_x_max + 1.0, 1.0))
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(0.0, ymin), ymax * 1.05)
        ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_cumulative_timepath_interactive(
        results: Sequence[SnapshotResult],
        *,
        n_T_per_snapshot: int = 8,
        resolution: float = 0.5,
        u_horizon: float = 55.0,
        central_p_uncond: float = 0.95,
        pick_T_mode: str = "linspace",
        title: str = "Mean cumulative infections — interactive time-path",
):
    """
    Interactive Plotly figure: one trace per snapshot per sampled T (toggleable).
    Requires `plotly`.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        raise ImportError("plotly is required for the interactive version. Install via `pip install plotly`") from e

    if not results:
        return
    offsets = _segment_offsets(results)
    payload = [(res, _first_post_after_tstar_per_draw(res)) for res in results]
    u_grid = np.arange(0.0, float(u_horizon) + 1e-12, float(resolution))

    fig = go.Figure()
    for i, ((res, first_post), x0) in enumerate(zip(payload, offsets)):
        # pick T samples
        if res.next_T is not None:
            Tmax = float(res.next_T)
        else:
            Tmax = min(float(res.T_grid[-1]) if res.T_grid.size else u_horizon, u_horizon)
        if Tmax <= 0:
            continue

        if n_T_per_snapshot <= 1:
            T_samples = np.array([0.0])
        else:
            if pick_T_mode == "linspace":
                T_samples = np.linspace(0.0, max(1e-8, Tmax), n_T_per_snapshot)
            else:
                finite_fp = first_post[np.isfinite(first_post)]
                if finite_fp.size >= (n_T_per_snapshot - 1):
                    qs = np.linspace(0.0, 1.0, n_T_per_snapshot)
                    T_samples = np.quantile(finite_fp, qs)
                    T_samples = np.clip(T_samples, 0.0, Tmax)
                    T_samples[0] = 0.0
                else:
                    T_samples = np.linspace(0.0, max(1e-8, Tmax), n_T_per_snapshot)
        T_samples = np.unique(np.clip(T_samples, 0.0, Tmax))

        for j, T in enumerate(T_samples):
            if j == 0:
                sel_mask = np.ones(len(res.infection_times_2d), dtype=bool)
            else:
                sel_mask = first_post > T

            chosen_trajs = []
            for sel, traj in zip(sel_mask, res.infection_times_2d):
                if not sel:
                    continue
                if len(traj) == 0:
                    chosen_trajs.append(np.empty(0))
                    continue
                arr = np.asarray(traj, dtype=float) - float(res.t_star)
                arr = arr[arr >= 0.0]
                chosen_trajs.append(arr)
            if len(chosen_trajs) == 0:
                continue
            cum_matrix = _cum_matrix_from_times(chosen_trajs, u_grid)
            if cum_matrix.size == 0:
                continue
            mid = cum_matrix.mean(axis=0)
            x_curve = x0 + T + u_grid
            name = f"m={res.m}  T={T:.2f}"
            visible = True if j == 0 else "legendonly"
            fig.add_trace(go.Scatter(x=x_curve, y=mid, mode="lines", name=name, visible=visible,
                                     hovertemplate="day=%{x:.2f}<br>mean cum=%{y:.2f}"))
            # band only for unconditional
            if j == 0:
                lo, hi = _tail_mean_band(cum_matrix, central_p_uncond)
                fig.add_trace(go.Scatter(x=np.concatenate([x_curve, x_curve[::-1]]),
                                         y=np.concatenate([hi, lo[::-1]]),
                                         fill="toself", fillcolor='rgba(0,0,0,0.1)',
                                         line=dict(color='rgba(255,255,255,0)'),
                                         hoverinfo="skip", name=f"m={res.m} band", visible=True))
    fig.update_layout(title=title, xaxis_title="Elapsed days since first $t_\\star$",
                      yaxis_title="Mean cumulative infections", template="simple_white")
    fig.show()
    return fig


# ----------------------------------------------------------------------
# =============================== DRIVER ===============================
# Example main: builds parameters & obs, runs snapshots, plots figures.
# Adjust num_trajectories/max_workers to suit your environment.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime, timedelta
    from python.eventide import Parameters, Scenario

    # ---- Priors & constraints (from your earlier example) ----
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

    # ---- Observations (example) ----
    obs_points = [
        (datetime(2025, 3, 6), 1),
        (datetime(2025, 3, 21), 3),
        (datetime(2025, 3, 25), 1),
        (datetime(2025, 3, 26), 1),
        (datetime(2025, 3, 30), 1),
        (datetime(2025, 3, 31), 1),
        (datetime(2025, 4, 2), 2),
        (datetime(2025, 4, 17), 1),
    ]
    # obs_points = [
    #     (datetime(2025, 3, 6), 1),
    #     (datetime(2025, 3, 21), 3),
    #     (datetime(2025, 3, 25), 1),
    #     (datetime(2025, 3, 26), 1),
    #     (datetime(2025, 3, 30), 1),
    #     (datetime(2025, 4, 2), 2),
    #     (datetime(2025, 4, 17), 1),
    # ]

    # ---- Which prefixes to run/overlay ----
    snapshots = (3, 4, 5, 7, 8)
    snapshots = (8,)
    # snapshots = (7,)

    # ---- Common horizon grid (relative per snapshot; we shift in the plot) ----
    T_grid = np.arange(0, 70 + 1e-9, 1.0)

    # ---- Per-snapshot acceptance-window parameters (no optimization step needed) ----
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
            'gap_scale': 0.15009926426544729, },

        8: {'sigma_days': 3.7781097376173833, 'beta': 0.5068644011368587, 'neighbor_weight': 0.11150380609174124,
            'grid_step_days': 0.28283975232293723, 'min_seg_days': 3.1876070745495713, 'kmax': 8,
            'baseline_p': 0.012782592311887171,
            'alpha': 0.39918253112411994, 'h_max': 0.11775991884158551, 'eps_share': 0.016036133279788863,
            'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 4,
            'gap_scale': 0.21347989807730383, 'include_global_total': True},

        # 8: {'sigma_days': 3.982802469310849, 'beta': 0.5620160669654297, 'neighbor_weight': 0.14374831069682925,
        #     'grid_step_days': 0.29535300036388434, 'min_seg_days': 4.829205983803487, 'kmax': 7,
        #     'baseline_p': 0.04699967547277244,
        #     'alpha': 0.39922441681969195, 'h_max': 0.10930906122596419, 'eps_share': 0.0011455595753183461,
        #     'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 1,
        #     'gap_scale': 0.25437829016143054, 'include_global_total': True},
    }

    # ---- Heavy run parameters (tune for your machine) ----
    scenario = Scenario([])
    results = run_all_snapshots_per_m(
        obs_points=obs_points,
        pars=pars,
        builder_kwargs_by_m={i: best_kwargs_by_m[7] for i in snapshots},
        snapshots=snapshots,
        num_trajectories=100_000_000_000,
        chunk_size=100_000,
        T_run=60,
        max_cases=2000,
        max_workers=13,
        T_grid=T_grid,
        h=0.2,
        H_pad=10.0,
        min_required=10_000
    )
    # apply_presentation_style()
    apply_journal_style()

    exit()

    plot_cumulative_infections_last_numeric(
        results,
        resolution=0.25,
        perc_bands=(0.95, 0.5, 0.2),
        cmap="PuBu",
        scale="linear",
        show_mean=True,
        show_median=False,
        obs_points_days=[((day - datetime(2025, 3, 6)).days, cases) for (day, cases) in obs_points]
    )

    plot_acceptance_construction_side_by_side(
        obs_points,
        sigma_days=1.8,
        beta=0.75,
        neighbor_weight=0.6,
        grid_step_days=0.2,
        min_seg_days=1.0,
        kmax=3,
        baseline_p=0.10,
        alpha=0.10,
        h_max=0.30,
        eps_share=1e-6,
        include_quiet_windows=True,
        gap_scale=0.4,
        close_gaps_days=0.0,
    )

    # plot_last_posterior(results, mass=0.95, bw_adjust=1.0, n_grid=600)
    # plot_timepath_Re(results, step=0.25, band=(0.025, 0.975), summary="median", draw_verticals=True)

    # exit()

    plot_cumulative_infections_last_numeric(
        results,
        resolution=0.25,
        perc_bands=(),
        cmap="PuBu",
        scale="linear",
        show_mean=False,
        show_median=False,
        # Optional observed incidents as numeric days since start:
        obs_points_days=[((day - datetime(2025, 3, 6)).days, cases) for (day, cases) in obs_points]
    )
    plot_cumulative_infections_last_numeric(
        results,
        resolution=0.25,
        perc_bands=(),
        cmap="PuBu",
        scale="linear",
        show_mean=True,
        show_median=False,
        # Optional observed incidents as numeric days since start:
        obs_points_days=[((day - datetime(2025, 3, 6)).days, cases) for (day, cases) in obs_points]
    )
    plot_cumulative_infections_last_numeric(
        results,
        resolution=0.25,
        perc_bands=(0.95, 0.5, 0.2),
        cmap="PuBu",
        scale="linear",
        show_mean=True,
        show_median=False,
        # Optional observed incidents as numeric days since start:
        obs_points_days=[((day - datetime(2025, 3, 6)).days, cases) for (day, cases) in obs_points]
    )

    exit()

    # ---- Figures ----

    # A) Online RB probabilities (shifted on a common time axis)
    plot_rb_online_two_pane(results[:1], show_next_dot=True, ylim=(0, 1))
    plot_rb_online_two_pane(results[:2], show_next_dot=True, ylim=(0, 1))
    plot_rb_online_two_pane(results[:3], show_next_dot=True, ylim=(0, 1))
    plot_rb_online_two_pane(results[:4], show_next_dot=True, ylim=(0, 1))
    plot_rb_online_two_pane(results[:5], show_next_dot=True, ylim=(0, 1))
    plot_rb_online_two_pane(results[:6], show_next_dot=True, ylim=(0, 1))
    exit()

    # C) Time-path of r*R0 between events, with central band
    plot_timepath_Re(results, step=0.25, band=(0.025, 0.975), summary="median", draw_verticals=True)

    # plot_cumulative_timepath_clean(results, n_T_per_snapshot=6, pick_T_mode="linspace",
    #                                central_p_uncond=0.95, resolution=0.5, panels=False)

    # Static, panels (one subplot per snapshot)
    # plot_cumulative_timepath_clean(results, n_T_per_snapshot=5, panels=True)

    # Interactive (requires plotly)
    # plot_cumulative_timepath_interactive(results, n_T_per_snapshot=10, resolution=0.5)

    # plot_last_posterior(results, mass=0.95, bw_adjust=1.0, n_grid=600)

    plot_probability_over_last_versions_rb_vs_analytic(
        results[-1:],
        n_high=20_000,  # 1) Analytic only
        n_low=600,  # 2) Analytic only; 3) Analytic + RB
        seed=123,
        h=0.2,
    )
    exit()
    plot_cumulative_infections_last_numeric(
        results,
        resolution=0.25,
        perc_bands=(0.95, 0.5, 0.2),
        cmap="PuBu",
        scale="linear",
        show_mean=True,
        show_median=False,
        # Optional observed incidents as numeric days since start:
        obs_points_days=[((day - datetime(2025, 3, 6)).days, cases) for (day, cases) in obs_points]
    )

    plot_acceptance_construction_side_by_side(
        obs_points,
        sigma_days=1.8,
        beta=0.75,
        neighbor_weight=0.6,
        grid_step_days=0.2,
        min_seg_days=1.0,
        kmax=3,
        baseline_p=0.10,
        alpha=0.10,
        h_max=0.30,
        eps_share=1e-6,
        include_quiet_windows=True,
        gap_scale=0.4,
        close_gaps_days=0.0,
    )
