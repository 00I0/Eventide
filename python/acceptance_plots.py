from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil, floor, log
from typing import List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm


# -------------------------- Data structures --------------------------------
@dataclass(frozen=True)
class WindowBand:
    start: datetime
    end: datetime
    L: int
    U: int


# -------------------------- Time helpers -----------------------------------
def days_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0


def proportional_observed_mass(
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

    # tally by day
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


def staircase_cumulative_at(obs_points: Sequence[Tuple[datetime, int]], t: datetime) -> int:
    """Sum of observed counts at times <= t (no fractional split; used as stepwise cum)."""
    return int(sum(c for (ti, c) in obs_points if ti <= t))


# -------------------------- Wilson bands -----------------------------------
def wilson_halfwidth(pi_hat: float, n: int, z: float) -> float:
    if n <= 0:
        return 0.0
    return (z * np.sqrt((pi_hat * (1 - pi_hat)) / n + (z * z) / (4 * n * n))) / (1 + (z * z) / n)


def band_from_target_and_share(
        M_hat: float,
        Y_hat: float,
        n_total: int,
        baseline_p: float,
        alpha: float,
        h_max: float,
        eps_share: float,
) -> Tuple[int, int, float, float]:
    """
    Compute (L, U, p, h) from target mass M_hat and observed mass Y_hat.
    """
    z = float(norm.ppf(1 - alpha / 2.0))
    pi_hat = (Y_hat / n_total) if n_total > 0 else 0.0
    w = wilson_halfwidth(pi_hat, n_total, z)
    # translate Wilson halfwidth on proportion into a relative add-on on the mass
    denom = max(pi_hat, eps_share) if pi_hat > 0 else eps_share
    h = min(h_max, w / denom)
    p = baseline_p + h
    L = max(0, int(floor((1 - p) * M_hat)))
    U = int(ceil((1 + p) * M_hat))
    return L, U, p, h


# -------------------------- Cluster–merging windows ------------------------
def build_cluster_windows(
        obs_points: Sequence[Tuple[datetime, int]],
        sigma_days: float,
        beta: float,
        neighbor_weight: float,
) -> Tuple[List[Tuple[datetime, datetime]], List[Tuple[datetime, datetime]]]:
    """
    Returns (raw_local_windows, merged_cluster_windows) as lists of (start, end).
    """
    obs = sorted([(t, int(c)) for (t, c) in obs_points if int(c) > 0], key=lambda x: x[0])
    times = [t for t, _ in obs]
    z_beta = float(norm.ppf(0.5 + beta / 2.0))
    base_half = z_beta * float(sigma_days)

    raw: List[Tuple[datetime, datetime]] = []

    def gap_days(i: int, j: int) -> Optional[float]:
        if i < 0 or j >= len(times):
            return None
        return days_between(times[i], times[j])

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

    # merge overlaps/touches
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


# -------------------------- Interpolant segmentation -----------------------
def segment_interpolant(
        obs_points: Sequence[Tuple[datetime, int]],
        grid_step_days: float,
        min_seg_days: float,
        kmax: int,
) -> Tuple[List[datetime], PchipInterpolator, np.ndarray, np.ndarray]:
    """
    Fit a monotone interpolant to the cumulative and segment it into K chords
    chosen by a simple information criterion (BIC-like).
    Returns (breakpoints_in_datetime, interpolant f, x_dense_days, f_dense)
    """
    obs = sorted([(t, int(c)) for (t, c) in obs_points if int(c) > 0], key=lambda x: x[0])
    times = [t for t, _ in obs]
    counts = [c for _, c in obs]
    A = times[0]
    x_obs = np.array([days_between(A, t) for t in times], dtype=float)
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
    tau_dt = [A + timedelta(days=d) for d in tau_days]
    return tau_dt, f, x_dense, f_dense


# -------------------------- Plot helpers -----------------------------------
def plot_construction_side_by_side(
        obs_points: Sequence[Tuple[datetime, int]],
        *,
        # Cluster controls
        sigma_days: float = 1.0,
        beta: float = 0.75,
        neighbor_weight: float = 0.8,
        # Interpolant controls
        grid_step_days: float = 0.25,
        min_seg_days: float = 1.0,
        kmax: int = 6,
        # Bands (common)
        baseline_p: float = 0.10,
        alpha: float = 0.10,
        h_max: float = 0.50,
        eps_share: float = 1e-6,
        # Quiet windows
        include_quiet_windows: bool = True,
        gap_scale: float = 0.4,
        # Optional post-merge short-gap closing (days). 0 → off.
        close_gaps_days: float = 0.0,
        # Visual
        paren_inset_frac: float = 0.06,  # tiny inward nudge for parentheses
        figsize=(12, 4.6),
        savepath: str = "construction_side_by_side.pgf",
):
    # Pull first color from your prop_cycle to keep palette consistent
    cycle_cols = mpl.rcParams['axes.prop_cycle'].by_key().get('color', ['#4C72B0'])
    COL_BAND = '#4C72B0'
    COL_INTERP = cycle_cols[0]
    COL_POINTS = cycle_cols[1] if len(cycle_cols) > 1 else '#333333'
    COL_PAREN = '#6b6b6b'
    COL_QUIET = '#E69F00'

    # ---------- data ----------
    obs = sorted([(t, int(c)) for (t, c) in obs_points if int(c) > 0], key=lambda x: x[0])
    if not obs:
        raise ValueError("obs_points must have at least one positive-count entry.")
    times = [t for t, _ in obs]
    counts = [c for _, c in obs]
    A, B = times[0], times[-1]
    n_total = int(sum(counts))

    # ---------- local helpers ----------
    def format_time_axis(ax):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    def N_obs_at(tdt: datetime) -> int:
        return staircase_cumulative_at(obs, tdt)

    def label_box(ax, xdt, y, text, valign):
        ax.text(
            xdt, y, text,
            ha="right", va=valign, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, lw=0, ),
            zorder=5,
        )

    def centered_band_rect(ax, a, b, center_y, L, U, k_index=None):
        """Draw band [L,U] centered at center_y. With TeX L_k/U_k callouts."""
        height = max(U - L, 1e-9)
        y0 = center_y - height / 2.0
        ax.add_patch(Rectangle(
            (mdates.date2num(a), y0),
            (b - a).total_seconds() / 86400.0,
            height,
            facecolor=COL_BAND, edgecolor='none', alpha=0.5, linewidth=0.9,
            zorder=1.5,
        ))
        # callouts at lower-right / upper-right corners (LaTeX)
        xr = b - timedelta(days=0.02 * max((b - a).total_seconds() / 86400.0, 1e-6))
        if k_index is None:
            label_box(ax, xr, y0 + 0.02 * height, rf"$L={L}$", "bottom")
            label_box(ax, xr, y0 + height + 0.07 * height, rf"$U={U}$", "top")
        else:
            label_box(ax, xr, y0 + 0.02 * height, rf"$L_{k_index}={L}$", "bottom")
            label_box(ax, xr, y0 + height + 0.07 * height, rf"$U_{k_index}={U}$", "top")

    def draw_parentheses(ax, tdt, y, a_left, b_right, color="#333333"):
        """Show raw local window as '( dot )', with a tiny inward nudge from edges."""
        width_days = max((b_right - a_left).total_seconds() / 86400.0, 1e-9)
        inset = timedelta(days=paren_inset_frac * width_days)
        a_in = a_left + inset
        b_in = b_right - inset
        # faint guides
        ax.hlines(y, a_in, tdt, colors=color, linestyles="-", linewidth=0.7, alpha=0.25, zorder=2.3)
        ax.hlines(y, tdt, b_in, colors=color, linestyles="-", linewidth=0.7, alpha=0.25, zorder=2.3)
        # draw literal parentheses (NOT r"\(" / r"\)")
        ax.text(a_in, y, "(", ha="center", va="center", fontsize=10, color=color, alpha=0.9, zorder=2.4)
        ax.text(b_in, y, ")", ha="center", va="center", fontsize=10, color=color, alpha=0.9, zorder=2.4)

    # ---------- cluster route ----------
    raw_local, clusters = build_cluster_windows(obs, sigma_days, beta, neighbor_weight)
    base_windows_cluster = clusters[:]

    # Optional: close small gaps after merging (morphological closing)
    if close_gaps_days > 0 and len(base_windows_cluster) >= 2:
        merged2 = []
        s, e = base_windows_cluster[0]
        for s2, e2 in base_windows_cluster[1:]:
            gap = (s2 - e).total_seconds() / 86400.0
            if gap <= close_gaps_days:
                e = max(e, e2)  # bridge the short gap
            else:
                merged2.append((s, e))
                s, e = s2, e2
        merged2.append((s, e))
        base_windows_cluster = merged2

    # Bands for clusters
    cluster_bands: List[WindowBand] = []
    for (a, b) in base_windows_cluster:
        M_hat = proportional_observed_mass(obs, A, B, a, b)  # observed mass in (a,b]
        L, U, _, _ = band_from_target_and_share(M_hat, M_hat, n_total, baseline_p, alpha, h_max, eps_share)
        cluster_bands.append(WindowBand(a, b, L, U))

    # Quiet windows inside gaps (optional; clipped)
    quiet_cluster: List[Tuple[datetime, datetime]] = []
    if include_quiet_windows and len(base_windows_cluster) >= 2:
        base_lengths = np.array([days_between(a, b) for (a, b) in base_windows_cluster], dtype=float)
        med_len = float(np.median(base_lengths)) if len(base_lengths) > 0 else 1.0
        for (a1, b1), (a2, b2) in zip(base_windows_cluster, base_windows_cluster[1:]):
            if a2 <= b1:  # no gap
                continue
            gap_len = days_between(b1, a2)
            center = b1 + (a2 - b1) / 2
            half_days = max(0.0, min(gap_scale * gap_len, 0.5 * med_len))
            if half_days > 0:
                qa, qb = center - timedelta(days=half_days), center + timedelta(days=half_days)
                qa = max(qa, A)
                qb = min(qb, B)
                if qa < qb:
                    quiet_cluster.append((qa, qb))

    # Color dots by merged cluster id
    palette = plt.get_cmap("tab10")

    def cluster_idx_for_time(t):
        for k, (a, b) in enumerate(base_windows_cluster):
            if a <= t <= b:
                return k
        return -1

    # ---------- segmentation route ----------
    tau_dt, f_interp, _, _ = segment_interpolant(obs, grid_step_days, min_seg_days, kmax)
    base_windows_seg = [(tau_dt[i], tau_dt[i + 1]) for i in range(len(tau_dt) - 1)]

    def Nref(t: datetime) -> float:
        return float(f_interp(days_between(A, t)))

    seg_bands: List[WindowBand] = []
    for (a, b) in base_windows_seg:
        M_hat = Nref(b) - Nref(a)
        if (b <= a) or (abs(M_hat) < 1e-9):
            continue
        Y_hat = proportional_observed_mass(obs, A, B, a, b)
        L, U, _, _ = band_from_target_and_share(M_hat, Y_hat, n_total,
                                                baseline_p, alpha, h_max, eps_share)
        seg_bands.append(WindowBand(a, b, L, U))

    # ---------- figure ----------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize, sharey=True, dpi=300)

    # LEFT: Cluster–merging
    axL.set_title(r"Cluster–merging")
    for a, b in quiet_cluster:
        axL.axvspan(a, b, color=COL_QUIET, zorder=0.5, alpha=0.5)

    t_pts = [t for t, _ in obs]
    cum_pts = [N_obs_at(t) for t in t_pts]
    colors = [palette(cluster_idx_for_time(t) % 10) for t in t_pts]
    axL.scatter(t_pts, cum_pts, s=16, c=colors, zorder=2.6,
                edgecolor="white", linewidth=0.35, label=r"observed cumulative (points)")

    # raw-local parentheses around each dot
    for (t, _), (a_raw, b_raw) in zip(obs, raw_local):
        y = N_obs_at(t)
        draw_parentheses(axL, t, y, a_raw, b_raw)

    # bands centered at mean of observed cum points inside each cluster
    for k, w in enumerate(cluster_bands, start=1):
        vals = [N_obs_at(t) for t in times if (w.start < t <= w.end)]
        center_y = (N_obs_at(w.start) + 0.5 * proportional_observed_mass(obs, A, B, w.start, w.end)) if len(
            vals) == 0 else float(np.mean(vals))
        centered_band_rect(axL, w.start, w.end, center_y, w.L, w.U, k_index=k)

    format_time_axis(axL)
    axL.set_xlabel(r"time")
    axL.set_ylabel(r"cumulative cases")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axL.legend(handles=[
        Patch(facecolor=COL_QUIET, edgecolor='none', label=r"quiet window", alpha=0.5),
        Patch(facecolor=COL_BAND, edgecolor='none', label=r"acceptance band $[L,U]$", alpha=0.4),
        Line2D([], [], color=COL_PAREN, lw=0.9, label=r"raw local window (parentheses)"),
        Line2D([], [], marker="o", lw=0, markersize=4, markerfacecolor=COL_POINTS,
               markeredgecolor="white", label=r"observed cumulative (points)"),
    ], frameon=False, loc="upper left")

    # RIGHT: Interpolant–segmentation
    axR.set_title(r"Interpolant–segmentation")
    # interpolant + polyline
    x_plot = np.linspace(0.0, days_between(A, B), 400)
    axR.plot([A + timedelta(days=float(d)) for d in x_plot],
             [float(f_interp(d)) for d in x_plot],
             color=COL_INTERP, lw=1.0, zorder=2.0, label=r"interpolant")
    for (a, b) in base_windows_seg:
        if b <= a:  continue
        xa, xb = days_between(A, a), days_between(A, b)
        ya, yb = float(f_interp(xa)), float(f_interp(xb))
        axR.plot([a, b], [ya, yb], color=COL_POINTS, lw=1.0, zorder=2.1, label=None)

    # points (for fit context)
    axR.plot(t_pts, cum_pts, "o", ms=3.0, color='#BB4430', zorder=3.0, alpha=0.9, label=r"observed cumulative (points)")

    # bands centered at median(interpolant) on each segment
    for k, w in enumerate(seg_bands, start=1):
        x0, x1 = days_between(A, w.start), days_between(A, w.end)
        xs = np.linspace(x0, x1, 201)
        ys = f_interp(xs)
        center_y = float(np.median(ys))
        centered_band_rect(axR, w.start, w.end, center_y, w.L, w.U, k_index=k)

    format_time_axis(axR)
    axR.set_xlabel(r"time")
    axR.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    plt.show()


# -------------------------- Example / Run ----------------------------------
def main():
    # Example data (edit freely)
    obs_points = [
        (datetime(2025, 3, 6), 1),
        (datetime(2025, 3, 21), 3),
        (datetime(2025, 3, 25), 1),
        (datetime(2025, 3, 26), 1),
        (datetime(2025, 3, 30), 1),
        (datetime(2025, 4, 2), 2),
        (datetime(2025, 4, 17), 1),
    ]

    plot_construction_side_by_side(
        obs_points,
        # Cluster params
        sigma_days=1.8,
        beta=0.75,
        neighbor_weight=0.6,
        # Interpolant params
        grid_step_days=0.2,
        min_seg_days=1.0,
        kmax=3,
        # Banding (baseline + Wilson)
        baseline_p=0.10,
        alpha=0.10,  # 90% conf → smaller z → tighter than 95
        h_max=0.30,
        eps_share=1e-6,
        # Quiet windows
        include_quiet_windows=True,
        gap_scale=0.4,
        # Output
        savepath="../img/construction_side_by_side.pgf",
    )


if __name__ == "__main__":
    main()
