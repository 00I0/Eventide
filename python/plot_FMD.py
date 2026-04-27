from datetime import datetime, timedelta
from typing import Sequence, Tuple, Any, Literal, Optional, List

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde, norm
from sklearn.neighbors import KernelDensity

from python.eventide import Parameters, Scenario
from python.presentationplots import run_all_snapshots_per_m


# ==============================================================================
# 1. JOURNAL STYLE CONFIGURATION
# ==============================================================================

def set_journal_style():
    """
    Configures Matplotlib for D1 Applied Math Journal standards.
    """
    plt.rcParams.update({
        # Typography
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',  # Computer Modern for LaTeX-like math
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,

        # Layout & Spines
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': True,

        # Ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'ytick.left': True,

        # Colors (Colorblind friendly + High Contrast)
        'axes.prop_cycle': plt.cycler(color=['#0072B2', '#D55E00', '#009E73', '#CC79A7']),

        # Output
        'figure.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


# Palette Constants
COLORS = {
    'primary': '#0072B2',  # Blue
    'secondary': '#b0b0b0',
    'band': '#0072B2',  # Band color
    'obs': '#000000',  # Black
    'grid': '#E0E0E0'
}


# ==============================================================================
# 2. NUMERICAL HELPERS
# ==============================================================================
def _kde_scipy(x_data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Standard Gaussian KDE using Scipy."""
    kde = gaussian_kde(x_data)
    return kde(grid)


def _kde_sklearn(x_data: np.ndarray, grid: np.ndarray, kernel: str = 'epanechnikov') -> np.ndarray:
    """
    KDE using Scikit-Learn.
    Allows for different kernels (epanechnikov, exponential, etc).
    Implements Silverman's Rule manually for bandwidth selection.
    """

    # 1. Bandwidth Selection (Silverman's Rule approximation)
    # Sklearn doesn't do this auto-magically like Scipy
    n = len(x_data)
    std = np.std(x_data)
    q75, q25 = np.percentile(x_data, [75, 25])
    iqr = q75 - q25
    # Use the smaller of std or IQR/1.34 to be robust to outliers
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    if sigma == 0: sigma = 1.0  # Fallback

    # Silverman's factor roughly
    bw = 0.9 * sigma * n ** (-0.2)

    # Fit model
    kde = KernelDensity(kernel=kernel, bandwidth=bw)
    kde.fit(x_data[:, None])

    # Score returns log-density
    log_pdf = kde.score_samples(grid[:, None])
    return np.exp(log_pdf)


def _kde_smart(
        x: np.ndarray,
        grid: np.ndarray,
        bounds: Tuple[float, float],
        method: Literal['reflect', 'truncate'],
        estimator: Literal['scipy', 'sklearn_epanechnikov', 'sklearn_gaussian'] = 'scipy'
):
    """
    Selects the correct KDE method per variable and estimator backend.

    Estimators:
    - 'scipy': Standard Gaussian KDE (Smooth, infinite support).
    - 'sklearn_epanechnikov': Parabolic kernel (Optimal MSE, finite support, sharper peaks).
    - 'sklearn_gaussian': Same as scipy but via sklearn engine.
    """
    x = x[np.isfinite(x)]
    lo, hi = bounds

    # --- 1. Prepare Data (Handle Reflection) ---
    if method == 'reflect':
        # Reflect data across boundaries
        reflect_min = 2 * lo - x
        reflect_max = 2 * hi - x

        # Filter reflections (optimization)
        bw_proxy = (np.max(x) - np.min(x)) * 0.4
        reflect_min = reflect_min[reflect_min > lo - bw_proxy]
        reflect_max = reflect_max[reflect_max < hi + bw_proxy]

        x_proc = np.concatenate([x, reflect_min, reflect_max])
    else:
        x_proc = x

    # --- 2. Run Estimator ---
    if estimator == 'scipy':
        pdf = _kde_scipy(x_proc, grid)
    elif estimator.startswith('sklearn'):
        kernel_name = estimator.split('_')[1]  # e.g. 'epanechnikov'
        pdf = _kde_sklearn(x_proc, grid, kernel=kernel_name)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # --- 3. Post-Process (Normalize) ---
    # Re-normalize over the visible grid because reflection or truncation
    # might have shifted the integral sum.
    integral = np.trapezoid(pdf, grid)
    if integral > 0:
        pdf /= integral

    return pdf


def _format_tick(val, is_edge=False):
    """Integers for edges, 2 decimals for median."""
    if abs(val) < 1e-9: return "0"
    if abs(val - round(val)) < 1e-6: return f"{int(round(val))}"
    return f"{val:.2f}".rstrip('0').rstrip('.')


def _get_hdi(x, mass=0.95):
    """Highest Density Interval via quantiles."""
    x = x[np.isfinite(x)]
    if len(x) == 0: return np.nan, np.nan
    alpha = 1 - mass
    return np.quantile(x, alpha / 2), np.quantile(x, 1 - alpha / 2)


def _cum_matrix_from_times(infection_times, grid_points_numeric):
    rows = []
    for traj in infection_times:
        t = np.sort(np.asarray(traj, dtype=float))
        rows.append(np.searchsorted(t, grid_points_numeric, side="right"))
    if not rows: return np.zeros((0, len(grid_points_numeric)))
    return np.vstack(rows).astype(float)


def _is_too_close(val, others, span, threshold=0.08):
    """Checks if 'val' is within 'threshold' percent of the 'span' from any value in 'others'."""
    for o in others:
        if abs(val - o) / span < threshold:
            return True
    return False


def _get_first_post_infection_times(res: Any) -> np.ndarray:
    """
    For each trajectory in the snapshot, find the time of the first infection
    that occurred strictly AFTER the snapshot's t_star.
    """
    t_star = float(res.t_star)
    first_posts = []

    for traj in res.infection_times_2d:
        ts = np.sort(np.asarray(traj, dtype=float))
        future_infs = ts[ts > t_star]

        if len(future_infs) > 0:
            first_posts.append(future_infs[0] - t_star)
        else:
            first_posts.append(np.inf)

    return np.array(first_posts)


# ==============================================================================
# 3. DATE-AWARE PLOTTING FUNCTIONS
# ==============================================================================

def plot_timepath_Re(
        results: Sequence[Any],
        start_date: datetime,
        *,
        step: float = 0.1,
        final_horizon_days: float = 20.0,
        perc_bands: Sequence[float] = (0.95, 0.8, 0.5),  # User specified bands
        figsize: Tuple[float, float] = (5.04, 3.1248)
):
    # Ensure style is active
    set_journal_style()

    # --- Internal Helpers ---
    def _get_offsets(res_list) -> list:
        offs = []
        acc = 0.0
        for r in res_list:
            offs.append(acc)
            if r.next_T is not None:
                acc += float(r.next_T)
        return offs

    def _get_first_post_inf(res_obj) -> np.ndarray:
        t_star = float(res_obj.t_star)
        out = np.empty(len(res_obj.infection_times_2d), dtype=float)
        for i, traj in enumerate(res_obj.infection_times_2d):
            t = np.asarray(traj, float)
            rel = t - t_star
            rel = rel[rel > 0.0]
            out[i] = rel[0] if rel.size else np.inf
        return out

    # --- 1. Prepare Data Containers ---
    offsets = _get_offsets(results)

    # We need a dictionary to store lists for each requested band
    bands_data = {p: {'lo': [], 'hi': []} for p in perc_bands}

    all_x = []
    all_mid = []
    jump_locs = []

    # --- 2. Iterate Snapshots ---
    for i, (res, x0) in enumerate(zip(results, offsets)):

        if res.next_T is not None:
            T_end = float(res.next_T)
            is_final = False
        else:
            T_end = final_horizon_days
            is_final = True

        if T_end <= 0: continue

        n_steps = max(1, int(np.floor(T_end / step)) + 1)
        T_eval = np.linspace(0.0, T_end, n_steps)

        draws = np.asarray(res.draws_array)
        Re_base = draws[:, 0] * draws[:, 2]
        first_post = _get_first_post_inf(res)

        # Temp storage for this segment
        seg_mids = []
        seg_bands = {p: {'lo': [], 'hi': []} for p in perc_bands}

        for j, T in enumerate(T_eval):
            if j == 0:
                vals = Re_base
            else:
                vals = Re_base[first_post > T]

            # Calculate Median
            seg_mids.append(np.median(vals))

            # Calculate requested bands
            for p in perc_bands:
                alpha = (1.0 - p) / 2.0
                lo = np.percentile(vals, alpha * 100)
                hi = np.percentile(vals, (1.0 - alpha) * 100)
                seg_bands[p]['lo'].append(lo)
                seg_bands[p]['hi'].append(hi)

        # Store Segment Data
        all_x.append(x0 + T_eval)
        all_mid.append(np.array(seg_mids))

        for p in perc_bands:
            bands_data[p]['lo'].append(np.array(seg_bands[p]['lo']))
            bands_data[p]['hi'].append(np.array(seg_bands[p]['hi']))

        if not is_final:
            jump_locs.append(x0 + T_end)

    # --- 3. Concatenate and Convert to Dates ---
    X_num = np.concatenate(all_x)
    MID = np.concatenate(all_mid)
    X_dates = [start_date + timedelta(days=float(x)) for x in X_num]

    # Get the final date and values for anchoring the legend text
    final_median_y = MID[-1]

    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    # Reference line at 1.0
    ax.axhline(1.0, color='#eee', linestyle=':', linewidth=0.8, zorder=8, alpha=0.5)
    # Label for Reference line

    # Draw Bands (Sort so widest is drawn first/underneath)
    sorted_bands = sorted(perc_bands, reverse=True)

    for i, p in enumerate(sorted_bands):
        # Concatenate this specific band's data
        LO = np.concatenate(bands_data[p]['lo'])
        HI = np.concatenate(bands_data[p]['hi'])

        alpha = 0.15 + (0.07 * i)

        ax.fill_between(X_dates, LO, HI, color=COLORS['primary'], alpha=alpha, linewidth=0)

        ax.text(X_dates[-20], HI[-1] + 0.01, f" {int(p * 100)}% CI",
                color=COLORS['primary'], alpha=1.1 * alpha, fontsize=7, va='bottom', ha='center', zorder=100, )

    # Draw Median Line
    ax.plot(X_dates, MID, color=COLORS['primary'], lw=1.5, zorder=10)

    # --- NEW: Median Labeling ---
    ax.text(X_dates[-20], final_median_y + 0.01, "median",
            color=COLORS['primary'], fontsize=7, va='bottom', ha='center', fontweight='normal')

    # Draw Vertical Event Lines (Subtle Style)
    for j_day in jump_locs:
        j_date = start_date + timedelta(days=j_day)
        ax.axvline(j_date, color=COLORS['secondary'], linestyle='-', lw=0.8, zorder=5)

    # --- 5. Formatting ---

    # Explicit "Month Day" format
    date_fmt = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(date_fmt)

    # Ensure locator doesn't overcrowd labels
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))

    ax.set_ylabel(r"Effective reproduction number $rR_0$")
    ax.set_ylim(0, 3.0)

    final_day = offsets[-1] + (float(results[-1].next_T) if results[-1].next_T else final_horizon_days)
    # Extended limit by +3 days to make room for the new text labels
    ax.set_xlim(start_date, start_date + timedelta(days=final_day))

    plt.tight_layout()
    plt.show()


def plot_timepath_Re_and_euler_lotka_growth_rate(
        results: Sequence[Any],
        start_date: datetime,
        *,
        step: float = 0.1,
        final_horizon_days: float = 20.0,
        perc_bands: Sequence[float] = (0.95, 0.8, 0.5),
        figsize: Tuple[float, float] = (5.04, 5.2),
        re_ylim: Tuple[float, float] = (0.0, 3.0),
        growth_ylim: Optional[Tuple[float, float]] = None,
):
    """
    Date-aware time paths for the effective reproduction number and Euler-Lotka
    growth-rate proxy, conditioning each horizon on no new post-snapshot infection.
    """
    set_journal_style()
    if not results:
        return

    for p in perc_bands:
        if not 0.0 < p < 1.0:
            raise ValueError("All percentile bands must be central masses in (0, 1).")

    offsets = []
    acc = 0.0
    for res in results:
        offsets.append(acc)
        if res.next_T is not None:
            acc += float(res.next_T)

    metrics = {
        "Re": {
            "mid": [],
            "bands": {p: {"lo": [], "hi": []} for p in perc_bands},
        },
        "growth": {
            "mid": [],
            "bands": {p: {"lo": [], "hi": []} for p in perc_bands},
        },
    }
    all_x = []
    jump_locs = []

    for res, x0 in zip(results, offsets):
        if res.next_T is not None:
            T_end = float(res.next_T)
            is_final = False
        else:
            T_end = final_horizon_days
            is_final = True

        if T_end <= 0:
            continue

        n_steps = max(1, int(np.floor(T_end / step)) + 1)
        T_eval = np.linspace(0.0, T_end, n_steps)

        draws = np.asarray(res.draws_array, dtype=float)
        first_post = _get_first_post_infection_times(res)

        n_draws = min(len(draws), len(first_post))
        if n_draws == 0:
            continue
        draws = draws[:n_draws]
        first_post = first_post[:n_draws]

        R0 = draws[:, 0]
        r = draws[:, 2]
        alpha = draws[:, 3]
        theta = draws[:, 4]

        Re_base = R0 * r
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            growth_base = (np.power(Re_base, 1.0 / alpha) - 1.0) / theta

        values_by_metric = {
            "Re": Re_base,
            "growth": growth_base,
        }

        segment_stats = {
            name: {
                "mid": [],
                "bands": {p: {"lo": [], "hi": []} for p in perc_bands},
            }
            for name in values_by_metric
        }

        for j, T in enumerate(T_eval):
            keep = np.ones(n_draws, dtype=bool) if j == 0 else first_post > T

            for name, base_values in values_by_metric.items():
                vals = base_values[keep]
                vals = vals[np.isfinite(vals)]

                if vals.size == 0:
                    segment_stats[name]["mid"].append(np.nan)
                    for p in perc_bands:
                        segment_stats[name]["bands"][p]["lo"].append(np.nan)
                        segment_stats[name]["bands"][p]["hi"].append(np.nan)
                    continue

                segment_stats[name]["mid"].append(float(np.median(vals)))
                for p in perc_bands:
                    alpha_tail = (1.0 - p) / 2.0
                    lo_pct = float(np.percentile(vals, alpha_tail * 100.0))
                    hi_pct = float(np.percentile(vals, (1.0 - alpha_tail) * 100.0))
                    segment_stats[name]["bands"][p]["lo"].append(lo_pct)
                    segment_stats[name]["bands"][p]["hi"].append(hi_pct)

        all_x.append(x0 + T_eval)
        for name in values_by_metric:
            metrics[name]["mid"].append(np.asarray(segment_stats[name]["mid"]))
            for p in perc_bands:
                metrics[name]["bands"][p]["lo"].append(np.asarray(segment_stats[name]["bands"][p]["lo"]))
                metrics[name]["bands"][p]["hi"].append(np.asarray(segment_stats[name]["bands"][p]["hi"]))

        if not is_final:
            jump_locs.append(x0 + T_end)

    if not all_x:
        return

    X_num = np.concatenate(all_x)
    X_dates = [start_date + timedelta(days=float(x)) for x in X_num]
    final_day = offsets[-1] + (float(results[-1].next_T) if results[-1].next_T else final_horizon_days)

    panel_specs = [
        (
            "Re",
            r"Effective reproduction number $rR_0$",
            COLORS["primary"],
            1.0,
            re_ylim,
        ),
        (
            "growth",
            r"Euler-Lotka growth-rate proxy $\frac{\left(rR_0\right)^{1/\alpha}-1}{\theta}$",
            "#D55E00",
            0.0,
            growth_ylim,
        ),
    ]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    sorted_bands = sorted(perc_bands, reverse=True)
    label_idx = max(0, len(X_dates) - 20)
    X_dates_arr = np.asarray(X_dates, dtype=object)

    def _nearest_finite_idx(mask: np.ndarray, target_idx: int) -> Optional[int]:
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            return None
        return int(idxs[np.argmin(np.abs(idxs - target_idx))])

    for ax, (name, ylabel, color, ref_value, ylim) in zip(axes, panel_specs):
        MID = np.concatenate(metrics[name]["mid"])
        ax.axhline(ref_value, color="#eeeeee", linestyle=":", linewidth=0.8, zorder=8, alpha=0.8)

        finite_band_values = []
        for i, p in enumerate(sorted_bands):
            LO = np.concatenate(metrics[name]["bands"][p]["lo"])
            HI = np.concatenate(metrics[name]["bands"][p]["hi"])
            good = np.isfinite(LO) & np.isfinite(HI)
            if not np.any(good):
                continue

            finite_band_values.extend([float(np.nanmin(LO[good])), float(np.nanmax(HI[good]))])
            alpha_vis = 0.15 + (0.07 * i)
            ax.fill_between(X_dates, LO, HI, color=color, alpha=alpha_vis, linewidth=0)

            band_label_idx = _nearest_finite_idx(good, label_idx)
            if band_label_idx is not None:
                label_y = HI[band_label_idx]
                span = max(float(np.nanmax(HI[good]) - np.nanmin(LO[good])), 1e-9)
                ax.text(
                    X_dates[band_label_idx], label_y + 0.015 * span, f" {int(p * 100)}% CI",
                    color=color, alpha=min(1.0, 1.1 * alpha_vis),
                    fontsize=7, va="bottom", ha="center", zorder=100,
                )

        good_mid = np.isfinite(MID)
        ax.plot(X_dates_arr[good_mid], MID[good_mid], color=color, lw=1.5, zorder=10)

        if np.any(good_mid):
            mid_label_idx = _nearest_finite_idx(good_mid, label_idx)
            if mid_label_idx is not None:
                median_y = MID[mid_label_idx]
                span = max(float(np.nanmax(MID[good_mid]) - np.nanmin(MID[good_mid])), 1e-9)
                ax.text(
                    X_dates[mid_label_idx], median_y + 0.02 * span, "median",
                    color=color, fontsize=7, va="bottom", ha="center", fontweight="normal",
                )

        for j_day in jump_locs:
            j_date = start_date + timedelta(days=j_day)
            ax.axvline(j_date, color=COLORS["secondary"], linestyle="-", lw=0.8, zorder=5)

        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color="#eeeeee", linestyle="-", which="major", zorder=0, alpha=0.5, lw=0.8)
        ax.xaxis.grid(False)

        if ylim is not None:
            ax.set_ylim(*ylim)
        elif finite_band_values:
            lo = min(min(finite_band_values), ref_value)
            hi = max(max(finite_band_values), ref_value)
            pad = 0.08 * max(hi - lo, 1e-9)
            ax.set_ylim(lo - pad, hi + pad)

    date_fmt = mdates.DateFormatter("%b %d")
    axes[-1].xaxis.set_major_formatter(date_fmt)
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[-1].set_xlim(start_date, start_date + timedelta(days=final_day))

    plt.tight_layout()
    plt.show()


def plot_last_posterior(
        results: Sequence[Any],
        figsize: Tuple[float, float] = (11.25, 4.3),
        pdf_estimator: str = 'scipy'
):
    set_journal_style()
    if not results: return
    res = results[-1]

    # --- Data Extraction ---
    arr = np.asarray(res.draws_array)
    R0, k, r, alpha, theta = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    Re = R0 * r
    alpha_theta = alpha * theta
    p0 = (k / (k + Re + 1e-12)) ** k

    # --- Configuration ---
    # (Data, Title, Bounds, KDE_Method)
    # We use 'reflect' for k because it's clipped. 'truncate' for others to keep tails thin.
    vars_config = [
        (R0, r"$R_0$", (0, 15), 'truncate'),
        (r, r"$r$", (0, 1), 'truncate'),
        (alpha, r"$\alpha$", (0, 30), 'truncate'),
        (theta, r"$\theta$", (0, 13), 'truncate'),
        (k, r"$k$", (0, 50), 'reflect'),  # <--- ONLY ONE REFLECTED
        (Re, r"$rR_0$", (0, 3), 'truncate'),  # <--- Corrected Title
        (alpha_theta, r"$\alpha\theta$", (2, 21), 'truncate'),
        (1 / np.sqrt(alpha), r"$\frac{1}{\sqrt{\alpha}}$", (0, 1.2), 'truncate'),
        (np.sqrt(alpha) * theta, r"$\sqrt{\alpha}\theta$", (0, 16), 'truncate'),
        (p0, r"$\left(\frac{k}{k+rR_0}\right)^k$", (0, 1), 'truncate'),
        (R0 / k, r"$\frac{r R_0}{k}$", (0, 1.2), 'truncate'),
        ((Re ** (1 / alpha) - 1) / theta, r"$\frac{\left(r R_0\right)^{\frac{1}{\alpha}} - 1}{\theta}$", (-0.15, 0.085),
         'truncate'),

    ]

    fig, axes = plt.subplots(2, 6, figsize=figsize, )
    axes = axes.flatten()

    for i, (data, title, bounds, method) in enumerate(vars_config):
        ax = axes[i]
        lo_b, hi_b = bounds

        # 1. Compute Density
        grid = np.linspace(lo_b, hi_b, 600)
        pdf = _kde_smart(data, grid, bounds, method, estimator=pdf_estimator)

        # 2. Calculate Stats
        med = np.median(data)
        q_lo = np.percentile(data, 2.5)
        q_hi = np.percentile(data, 97.5)

        # 3. Draw Outline (Primary Color)
        ax.plot(grid, pdf, color=COLORS['primary'], lw=1.5, zorder=10)

        # 4. Draw 95% Band (Secondary Fill)
        mask = (grid >= q_lo) & (grid <= q_hi)
        ax.fill_between(grid, 0, pdf, where=mask, color=COLORS['primary'], alpha=0.35, linewidth=0)

        # 5. Median Line (Dashed Black)
        med_height = np.interp(med, grid, pdf)
        ax.vlines(med, 0, med_height, color='#777', linestyle='--', lw=1, zorder=5)

        # 7. Axes & Ticks (Min, Median, Max)
        ax.set_xlim(lo_b, hi_b)
        tick_locs = [lo_b, med, hi_b]
        tick_labels = [
            _format_tick(lo_b, is_edge=True),
            _format_tick(med, is_edge=False),
            _format_tick(hi_b, is_edge=True)
        ]

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)

        trans = ax.get_xaxis_transform()
        tick_vals = [lo_b, med, hi_b]
        span = hi_b - lo_b

        # Check collision with Min, Median, Max
        # If too close, we simply don't show the CI text to keep it clean

        if not _is_too_close(q_lo, tick_vals, span) and lo_b < q_lo < hi_b:
            ax.text(q_lo, -0.02, f"{q_lo:.2f}", color=COLORS['primary'], fontsize=7,
                    ha='center', va='top', transform=trans, alpha=0.6)
            ax.plot([q_lo, q_lo], [0, 0.02], color=COLORS['primary'], lw=1, transform=trans, clip_on=False, alpha=0.6)

        if not _is_too_close(q_hi, tick_vals, span) and lo_b < q_hi < hi_b:
            ax.text(q_hi, -0.02, f"{q_hi:.2f}", color=COLORS['primary'], fontsize=7,
                    ha='center', va='top', transform=trans, alpha=0.6)
            ax.plot([q_hi, q_hi], [0, 0.02], color=COLORS['primary'], lw=1, transform=trans, clip_on=False, alpha=0.6)

        # 6. Styling
        ax.set_title(title, pad=6, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])  # No y-axis needed for shape visualization

        # Remove spines
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)

        # Adjust spacing to prevent x-axis label collision
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    plt.show()


def plot_cumulative_infections_datetime(
        results: Sequence[Any],
        start_date: datetime,
        *,
        obs_points: Optional[List[Tuple[datetime, int]]] = None,
        resolution_days: float = 0.25,
        perc_bands: Sequence[float] = (0.95, 0.8, 0.5),
        u_horizon_days: float = 55.0,
        figsize: Tuple[float, float] = (5.04, 3.1248)
):
    """
    Cumulative infections using Mean and Tail-Mean Bands.
    """
    set_journal_style()
    if not results: return
    res = results[-1]

    # --- Internal Helper: Tail Mean Logic ---
    def _tail_mean_band(matrix: np.ndarray, p_central: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates bands based on the mean of the tails, shrunk towards the global mean.
        This often produces smoother bands than raw quantiles for integer count data.
        """
        mu = matrix.mean(axis=0)
        if p_central >= 0.999:
            return mu, mu  # Fallback for ~100%

        N = matrix.shape[0]
        # t is the fraction of data in one tail (e.g. 0.025 for 95%)
        t = (1.0 - p_central) / 2.0
        # Average the top k and bottom k trajectories
        k = max(1, int(np.ceil(N * t)))

        sorted_cols = np.sort(matrix, axis=0)

        lower_raw = sorted_cols[:k, :].mean(axis=0)
        upper_raw = sorted_cols[-k:, :].mean(axis=0)

        # Shrinkage factor (gentle shrink towards mean for narrower bands)
        beta = p_central ** 0.7

        lower = mu + beta * (lower_raw - mu)
        upper = mu + beta * (upper_raw - mu)
        return lower, upper

    # --- 1. Setup Time Grid ---
    grid_numeric = np.arange(0, u_horizon_days + 1e-9, resolution_days)
    grid_dates = [start_date + timedelta(days=float(x)) for x in grid_numeric]

    # --- 2. Compute Statistics ---
    # Helper to get counts (N_trajectories x N_timepoints)
    cum_matrix = _cum_matrix_from_times(res.infection_times_2d, grid_numeric)

    # Calculate Mean (not Median)
    mean_curve = cum_matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    # --- 3. Plot Bands ---
    sorted_bands = sorted(perc_bands, reverse=True)

    for i, p in enumerate(sorted_bands):
        # Calculate Tail-Mean Band
        lower, upper = _tail_mean_band(cum_matrix, p)

        # Visual Alpha (matching your Timepath Re style)
        alpha_vis = 0.15 + (0.07 * i)

        ax.fill_between(grid_dates, lower, upper, color=COLORS['primary'], alpha=alpha_vis, lw=0)

        # Direct Labeling (at the end of the horizon)
        # We offset slightly in Y to sit on top of the edge
        # Check if we have data to label
        if len(grid_dates) > 0:
            ax.text(grid_dates[-10], upper[-10] + 0.25, f" {int(p * 100)}% CI",
                    color=COLORS['primary'], alpha=min(1.0, 1.1 * alpha_vis),
                    fontsize=7, va='bottom', ha='center')

    # --- 4. Plot Mean ---
    ax.plot(grid_dates, mean_curve, color=COLORS['primary'], lw=1.5, zorder=5)

    # Label Mean
    if len(grid_dates) > 0:
        ax.text(grid_dates[-10], mean_curve[-10] + 0.25, "mean",
                color=COLORS['primary'], fontsize=7, va='bottom', ha='center', fontweight='normal')

    # --- 5. Plot Observations ---
    if obs_points:
        obs_sorted = sorted(obs_points, key=lambda x: x[0])
        dates, incs = zip(*obs_sorted)
        cums = np.cumsum(incs)

        # Plot points
        ax.scatter(dates, cums, s=23, facecolors='white', edgecolors=COLORS['obs'],
                   lw=1.0, zorder=10)

        # Label the last observation directly
        ax.text(dates[-1], cums[-1] + 0.3, "Observed",
                color=COLORS['obs'], fontsize=7, va='bottom', ha='center', fontweight='normal')

    # --- 6. Formatting ---
    ax.yaxis.grid(True, color='#eeeeee', linestyle='-', which='major', zorder=0, alpha=0.5, lw=0.8)
    ax.xaxis.grid(False)

    # Date Axis
    date_fmt = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(date_fmt)
    # Weekly ticks (Mondays)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))

    # Limits & Labels
    ax.set_ylabel("Cumulative Infections")
    ax.set_ylim(bottom=0)

    # Extend X-limit significantly (+8 days) to make room for text labels
    ax.set_xlim(start_date, start_date + timedelta(days=u_horizon_days))

    plt.tight_layout()
    plt.show()


def plot_acceptance_bands(
        obs_points: Sequence[Tuple[datetime, int]],
        bands: Sequence[Any],
        figsize: Tuple[float, float] = (5.04, 3.1248),
        y_pad_factor: float = 0.15
):
    """
    Visualizes marginal acceptance window constraints on cumulative data.
    Includes:
    1. Smooth PCHIP Interpolant (Grey Background)
    2. Piecewise Linear Segmentation (Blue Dashed Lines)
    3. Cluster Brackets (Bottom Grey Lines)
    """
    set_journal_style()

    # --- 1. Data Prep ---
    obs_sorted = sorted(obs_points, key=lambda x: x[0])
    dates, incs = zip(*obs_sorted)
    dates_np = np.array(dates)
    cums = np.cumsum(incs)

    start_date = dates[0]
    final_date = dates[-1]

    # --- 1B. Build Interpolant & Linear Segmentation ---

    # 1. PCHIP (The Smooth Model)
    def to_seconds(t):
        return (t - start_date).total_seconds()

    x_obs = np.array([to_seconds(t) for t in dates])
    y_obs = np.array(cums, dtype=float)
    pchip = PchipInterpolator(x_obs, y_obs)

    # Dense grid for plotting smooth curve
    x_dense = np.linspace(x_obs[0], x_obs[-1], 500)
    y_dense = pchip(x_dense)
    dates_dense = [start_date + timedelta(seconds=s) for s in x_dense]

    # 2. DP Segmentation (The Linear Approx)
    # We reconstruct the optimal linear segments to visualize the logic
    best_segments = []
    try:
        # Params matching typical builder defaults
        step_days = 0.25
        min_seg_days = 1.0
        kmax = 4

        step_sec = step_days * 86400
        x_grid = np.arange(x_obs.min(), x_obs.max() + 1e-9, step_sec)
        f_grid = pchip(x_grid)
        n_grid = len(x_grid)
        min_pts = max(2, int(round(min_seg_days / step_days)))

        memo_cost = {}

        def get_cost(i, j):
            if (i, j) not in memo_cost:
                slen = j - i
                y0, y1 = f_grid[i], f_grid[j]
                slope = (y1 - y0) / max(1, slen)
                y_lin = y0 + np.arange(slen + 1) * slope
                memo_cost[(i, j)] = np.sum((f_grid[i:j + 1] - y_lin) ** 2)
            return memo_cost[(i, j)]

        best_score = float('inf')

        for K in range(1, kmax + 1):
            dp = np.full((K + 1, n_grid), 1e100)
            parent = np.full((K + 1, n_grid), -1, dtype=int)
            for j in range(min_pts, n_grid):
                dp[1, j] = get_cost(0, j);
                parent[1, j] = 0
            for k in range(2, K + 1):
                for j in range(k * min_pts, n_grid):
                    valid_i = range((k - 1) * min_pts, j - min_pts + 1)
                    if not valid_i: continue
                    costs = [dp[k - 1, i] + get_cost(i, j) for i in valid_i]
                    min_c = min(costs)
                    dp[k, j] = min_c
                    parent[k, j] = valid_i[costs.index(min_c)]

            sse = dp[K, n_grid - 1]
            if sse < 1e99:
                mse = max(sse / n_grid, 1e-12)
                # BIC-like score
                score = n_grid * np.log(mse) + 2 * K * np.log(n_grid)
                if score < best_score:
                    best_score = score
                    curr = n_grid - 1
                    path = []
                    for k_curr in range(K, 1, -1):
                        prev = parent[k_curr, curr]
                        path.append((prev, curr))
                        curr = prev
                    path.append((0, curr))
                    best_segments = path[::-1]
    except:
        best_segments = []  # Fallback if DP fails

    def _get_cumulative_at(t: datetime) -> int:
        past_indices = np.where(dates_np <= t)[0]
        if len(past_indices) == 0: return 0
        return cums[past_indices[-1]]

    # --- 1C. Re-Construct Clusters ---
    clusters = []
    sigma = 2;
    beta = 0.25;
    nw = 0.8
    z_beta = float(norm.ppf(0.5 + beta / 2.0))
    base_half = z_beta * sigma
    raw_intervals = []
    for i, t in enumerate(dates):
        gL = (dates[i] - dates[i - 1]).total_seconds() / 86400 if i > 0 else None
        gR = (dates[i + 1] - dates[i]).total_seconds() / 86400 if i < len(dates) - 1 else None
        if gL is None and gR is None:
            local = 0
        elif gL is None:
            local = nw * gR / 2
        elif gR is None:
            local = nw * gL / 2
        else:
            local = nw * (gL + gR) / 4
        half = max(base_half, local)
        raw_intervals.append((t - timedelta(days=half), t + timedelta(days=half)))
    if raw_intervals:
        raw_intervals.sort(key=lambda x: x[0])
        curr_s, curr_e = raw_intervals[0]
        for next_s, next_e in raw_intervals[1:]:
            if next_s <= curr_e:
                curr_e = max(curr_e, next_e)
            else:
                clusters.append((curr_s, curr_e));
                curr_s, curr_e = next_s, next_e
        clusters.append((curr_s, curr_e))

    # --- 2. Setup Figure ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 3. Plot Models & Observations ---

    # A. Smooth Interpolant (Background Guide)
    ax.plot(dates_dense, y_dense, color='#CCCCCC', lw=2.5, alpha=1.0, zorder=1, label="Interpolant")

    # B. Linear Segments (The Approximation) [NEW PART]
    if best_segments:
        for (idx_s, idx_e) in best_segments:
            # Map grid index back to date/value
            t_s = start_date + timedelta(seconds=x_grid[idx_s])
            t_e = start_date + timedelta(seconds=x_grid[idx_e])
            y_s = f_grid[idx_s]
            y_e = f_grid[idx_e]

            ax.plot([t_s, t_e], [y_s, y_e], color=COLORS['primary'], lw=1.0, ls='--', alpha=0.8, zorder=2)
            # ax.scatter([t_s, t_e], [y_s, y_e], s=8, color=COLORS['primary'], zorder=2)

    # C. Points
    ax.scatter(dates, cums, s=25, facecolors='white', edgecolors=COLORS['obs'], lw=1.0, zorder=10)

    # Label
    ax.text(dates[-1] - timedelta(days=0.1), cums[-1] + 0.3, "Observed", color=COLORS['obs'], fontsize=7, ha='center',
            va='bottom', zorder=10, fontweight='normal')

    # --- 4. Plot Acceptance Bands ---
    global_max_y = cums[-1]
    col_fill = mcolors.to_rgba(COLORS['primary'], alpha=0.15)
    col_edge = mcolors.to_rgba(COLORS['primary'], alpha=0.6)

    for i, band in enumerate(bands):
        if band.max_cases == 0:
            ax.axvspan(band.start_date, band.end_date, facecolor='none', edgecolor=COLORS['primary'], hatch='//////',
                       linewidth=0.0, alpha=0.15, zorder=0)
            continue

        # Handle Standard Band
        baseline = _get_cumulative_at(band.start_date)
        cum_min = baseline + band.min_cases
        cum_max = baseline + band.max_cases
        global_max_y = max(global_max_y, cum_max)

        x0 = mdates.date2num(band.start_date)
        width = mdates.date2num(band.end_date) - x0
        height = max(cum_max - cum_min, 0.1)
        y0 = cum_min

        # Draw Rectangle
        rect = Rectangle(
            (x0, y0), width, height,
            facecolor=col_fill, edgecolor=col_edge,
            linewidth=0.8, linestyle='-', zorder=5
        )
        ax.add_patch(rect)

        # Labels
        y_txt_offset = 0.08
        font_size = 6

        ax.text(band.end_date - timedelta(days=0.15), cum_min - y_txt_offset - 0.03, f"$L_{{{i + 1}}}={cum_min}$",
                color=COLORS['primary'], fontsize=font_size, ha='right', va='top',
                fontweight='bold', zorder=20)

        if cum_max > cum_min + 0.1:
            ax.text(band.end_date - timedelta(days=0.15), cum_max + y_txt_offset, f"$U_{{{i + 1}}}={cum_max}$",
                    color=COLORS['primary'], fontsize=font_size, ha='right', va='bottom',
                    fontweight='bold', zorder=20)

    # --- 5. Draw Clusters ---
    y_bracket = 0.4  # Position below x-axis
    for cl_start, cl_end in clusters[1:-1]:
        # Draw Bracket Line
        ax.plot([cl_start, cl_end], [y_bracket, y_bracket], color='#777777', lw=0.75, clip_on=False, zorder=10)

        # Draw Bracket Ends (vertical ticks)
        tick_h = 0.2
        ax.plot([cl_start, cl_start], [y_bracket - tick_h, y_bracket + tick_h], color='#777777', lw=0.75, clip_on=False,
                zorder=10)
        ax.plot([cl_end, cl_end], [y_bracket - tick_h, y_bracket + tick_h], color='#777777', lw=0.75, clip_on=False,
                zorder=10)

    # --- 6. Formatting ---
    date_fmt = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))

    ax.set_ylabel("Cumulative Cases")

    ax.set_xlim(start_date - timedelta(days=1), final_date + timedelta(days=1))
    ax.set_ylim(bottom=0, top=global_max_y * (1 + y_pad_factor))

    ax.yaxis.grid(True, color='#eeeeee', linestyle='-', zorder=0)
    ax.xaxis.grid(False)

    plt.tight_layout()
    plt.show()


def plot_rb_online_two_pane(
        results: Sequence[Any],
        start_date: datetime,
        *,
        ylim: Tuple[float, float] = (0, 1.05),
        figsize: Tuple[float, float] = (11.25, 4.3),
):
    """
    Plots online stopping probabilities (Unconditional vs Conditional).
    Style: Minimalist, Blue Gradient, Direct Labeling.
    """
    set_journal_style()
    if not results: return

    # --- 1. Data Prep ---
    offsets = []
    acc = 0.0
    for r in results:
        offsets.append(acc)
        if r.next_T is not None:
            acc += float(r.next_T)

    titles = [r"Unconditional", r"Conditional"]
    series_keys = ["p_uncond_mean", "p_cond_mean"]

    # Create a blue gradient palette for the snapshots
    # Start: Light Blue, End: Dark Blue
    # We use a colormap to pick N distinct shades
    cmap = plt.get_cmap("Blues")
    # Sample from 0.4 (mid blue) to 1.0 (dark blue)
    color_indices = np.linspace(0.4, 1.0, len(results))
    colors = [cmap(i) for i in color_indices]

    # --- 2. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Common X-axis limit calculation
    max_days = offsets[-1] + (float(results[-1].next_T) if results[-1].next_T else 20.0)

    for col_idx, (ax, key) in enumerate(zip(axes, series_keys)):

        # Iterate through snapshots
        for i, (res, x0) in enumerate(zip(results, offsets)):
            y_vals = getattr(res, key)
            x_vals = x0 + res.T_grid
            dates = [start_date + timedelta(days=float(x)) for x in x_vals]
            ax.plot(dates, y_vals, color=colors[i], lw=1.8, alpha=0.9, zorder=10 - i)

            # Event Dot (The moment the next observation actually happened)
            if res.next_T is not None:
                # Interpolate Y at exactly next_T
                x_event = x0 + float(res.next_T)
                y_event = float(np.interp(res.next_T, res.T_grid, y_vals))
                date_event = start_date + timedelta(days=x_event)

                # Draw "Hollow" dot (White fill, colored edge)
                ax.scatter([date_event], [y_event], s=23, facecolors='white', edgecolors=colors[i], lw=1.5, zorder=20)

                # Optional: Add small vertical dropline to show exactly when it stopped
                ax.axvline(date_event, ymax=y_event / ylim[1], color=colors[i], linestyle=':', lw=0.8, alpha=0.5)

        # --- Formatting per subplot ---
        ax.set_title(titles[col_idx], )
        ax.set_ylim(ylim)
        ax.set_xlim(start_date, start_date + timedelta(days=max_days))

        # Grid
        ax.yaxis.grid(True, color='#eeeeee', linestyle='-', zorder=0)
        ax.xaxis.grid(False)

        # Date Axis
        date_fmt = mdates.DateFormatter("%b %d")
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))

        # Label the lines directly on the first plot only (to avoid repetition)
        if col_idx == 0:
            # Label the peaks or start points? Start points are crowded.
            # Let's label the *peaks* of the first few curves.
            for i, (res, x0) in enumerate(zip(results, offsets)):
                # Only label if we have space, e.g., every other one or just first/last
                # Simple logic: Label the curve near its max value
                y_vals = getattr(res, key)
                idx_max = np.argmax(y_vals)
                x_max = x0 + res.T_grid[idx_max]
                y_max = y_vals[idx_max]
                d_max = start_date + timedelta(days=float(x_max))

                # Label text: "N=5"
                # Use annotatation with a small line pointing to the curve
                # ax.annotate(f"$N={res.n_obs}$", xy=(d_max, y_max), xytext=(0, 5),
                #             textcoords="offset points", ha='center', va='bottom',
                #             color=colors[i], fontsize=7, fontweight='bold')
                pass

                # --- 3. Shared Y Label ---
    axes[0].set_ylabel("Probability of Next Infection")

    # --- 4. Custom Legend (Manual Construction) ---
    # Since we used a gradient, a standard legend is bulky.
    # Let's make a small "Color Bar" style legend at the top right of the Conditional plot.

    # Dummy lines for legend
    legend_lines = [plt.Line2D([0], [0], color=c, lw=2) for c in colors]
    legend_labels = [f"$N={r.n_obs}$" for r in results]

    # Place legend inside the second plot, upper right
    axes[1].legend(legend_lines, legend_labels, loc='lower right',
                   frameon=False, fontsize=8, title="Observed Cases", title_fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)  # Tighten space between plots
    plt.show()


def plot_rb_online_left_pane(
        results: Sequence[Any],
        start_date: datetime,
        *,
        ylim: Tuple[float, float] = (0, 1.05),
        figsize: Tuple[float, float] = (5.04, 3.1248),
):
    """
    Single-pane version of `plot_rb_online_two_pane` using only the left subplot logic
    (Unconditional probabilities).
    """
    set_journal_style()
    if not results:
        return

    offsets = []
    acc = 0.0
    for r in results:
        offsets.append(acc)
        if r.next_T is not None:
            acc += float(r.next_T)

    cmap = plt.get_cmap("Blues")
    color_indices = np.linspace(0.4, 1.0, len(results))
    colors = [cmap(i) for i in color_indices]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    max_days = offsets[-1] + (float(results[-1].next_T) if results[-1].next_T else 20.0)
    key = "p_uncond_mean"

    for i, (res, x0) in enumerate(zip(results, offsets)):
        y_vals = np.asarray(getattr(res, key), dtype=float)
        x_vals = np.asarray(x0 + res.T_grid, dtype=float)

        if res.next_T is not None:
            x_event = x0 + float(res.next_T)
            y_event = float(np.interp(res.next_T, res.T_grid, y_vals))
            date_event = start_date + timedelta(days=x_event)

            # Solid up to event point, dashed after event point.
            x_pre = np.append(x_vals[x_vals < x_event], x_event)
            y_pre = np.append(y_vals[x_vals < x_event], y_event)
            x_post = np.concatenate(([x_event], x_vals[x_vals > x_event]))
            y_post = np.concatenate(([y_event], y_vals[x_vals > x_event]))

            dates_pre = [start_date + timedelta(days=float(x)) for x in x_pre]
            ax.plot(
                dates_pre, y_pre, color=colors[i], lw=1.8, linestyle='-',
                alpha=0.9, zorder=10 - i
            )

            if x_post.size >= 2:
                dates_post = [start_date + timedelta(days=float(x)) for x in x_post]
                ax.plot(
                    dates_post, y_post, color=colors[i], lw=2.0,
                    linestyle=(0, (8, 4)), dash_capstyle='butt',
                    alpha=1.0, zorder=10 - i
                )

            ax.scatter([date_event], [y_event], s=23, facecolors='white', edgecolors=colors[i], lw=1.5, zorder=20)
            ax.axvline(date_event, ymax=y_event / ylim[1], color=colors[i], linestyle=':', lw=0.8, alpha=0.5)
        else:
            dates = [start_date + timedelta(days=float(x)) for x in x_vals]
            ax.plot(dates, y_vals, color=colors[i], lw=1.8, alpha=0.9, zorder=10 - i)

    ax.set_title("Unconditional")
    ax.set_ylabel("Probability of Next Infection")
    ax.set_ylim(ylim)
    ax.set_xlim(start_date, start_date + timedelta(days=max_days))

    ax.yaxis.grid(True, color='#eeeeee', linestyle='-', zorder=0)
    ax.xaxis.grid(False)

    date_fmt = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))

    plt.tight_layout()
    plt.show()


def plot_rb_online_right_pane(
        results: Sequence[Any],
        start_date: datetime,
        *,
        ylim: Tuple[float, float] = (0, 1.05),
        figsize: Tuple[float, float] = (5.04, 3.1248),
):
    """
    Single-pane version of `plot_rb_online_two_pane` using only the right subplot logic
    (Conditional probabilities).
    """
    set_journal_style()
    if not results:
        return

    offsets = []
    acc = 0.0
    for r in results:
        offsets.append(acc)
        if r.next_T is not None:
            acc += float(r.next_T)

    cmap = plt.get_cmap("Blues")
    color_indices = np.linspace(0.4, 1.0, len(results))
    colors = [cmap(i) for i in color_indices]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    max_days = offsets[-1] + (float(results[-1].next_T) if results[-1].next_T else 20.0)
    key = "p_cond_mean"

    for i, (res, x0) in enumerate(zip(results, offsets)):
        y_vals = np.asarray(getattr(res, key), dtype=float)
        x_vals = np.asarray(x0 + res.T_grid, dtype=float)

        if res.next_T is not None:
            x_event = x0 + float(res.next_T)
            y_event = float(np.interp(res.next_T, res.T_grid, y_vals))
            date_event = start_date + timedelta(days=x_event)

            # Draw solid line up to the event point.
            x_pre = np.append(x_vals[x_vals < x_event], x_event)
            y_pre = np.append(y_vals[x_vals < x_event], y_event)
            dates_pre = [start_date + timedelta(days=float(x)) for x in x_pre]
            ax.plot(dates_pre, y_pre, color=colors[i], lw=1.8, linestyle='-', alpha=0.9, zorder=10 - i)

            # Draw dashed line after the event point.
            x_post = np.concatenate(([x_event], x_vals[x_vals > x_event]))
            y_post = np.concatenate(([y_event], y_vals[x_vals > x_event]))
            if x_post.size >= 2:
                dates_post = [start_date + timedelta(days=float(x)) for x in x_post]
                ax.plot(
                    dates_post, y_post, color=colors[i], lw=1.2,
                    linestyle=(0, (2, 0.5)), dash_capstyle='butt', alpha=1.0, zorder=10 - i
                )

            ax.scatter([date_event], [y_event], s=23, facecolors='white', edgecolors=colors[i], lw=1.5, zorder=20)
            # ax.axvline(date_event, ymax=y_event / ylim[1], color=colors[i], linestyle=':', lw=0.8, alpha=0.5)
        else:
            dates = [start_date + timedelta(days=float(x)) for x in x_vals]
            ax.plot(dates, y_vals, color=colors[i], lw=1.8, alpha=0.9, zorder=10 - i)

    # ax.set_title("Conditional")
    ax.set_ylabel("Probability of Outbreak Ending")
    ax.set_ylim(ylim)
    ax.set_xlim(start_date, start_date + timedelta(days=max_days))

    ax.yaxis.grid(True, color='#eeeeee', linestyle='-', zorder=0)
    ax.xaxis.grid(False)

    date_fmt = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))

    # Match two-pane style by keeping the observed-cases legend on the right pane.
    legend_lines = [plt.Line2D([0], [0], color=c, lw=2) for c in colors]
    legend_labels = [f"$N={r.n_obs}$" for r in results]
    ax.legend(legend_lines, legend_labels, loc='lower right',
              frameon=False, fontsize=8, title="Observed Cases", title_fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_rb_single_snapshot(
        result: Any,
        start_date: datetime,
        *,
        ylim: Tuple[float, float] = (0, 1.05),
        figsize: Tuple[float, float] = (5.04, 3.1248)  # Single column width
):
    """
    Plots a single snapshot's Unconditional vs Conditional probability curves on one axis.
    """
    set_journal_style()

    # --- 1. Data Prep ---
    # We assume 'result' is a single SnapshotResult object, not a list
    # The 'offset' is effectively 0 relative to this snapshot's start,
    # but we need to map T_grid to real dates starting from start_date.

    # Map numeric T_grid to dates
    dates = [start_date + timedelta(days=float(x)) for x in result.T_grid[:50]]

    # --- 2. Setup Figure ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 3. Plot Curves ---

    # A. Unconditional (Lighter or Dashed to distinguish)
    # Using 'band' color (lighter blue) for unconditional
    ax.plot(dates, result.p_uncond_mean[:50], color=COLORS['band'], lw=1.5, alpha=0.9, label="Unconditional")

    # B. Conditional (Darker/Primary)
    ax.plot(dates, result.p_cond_mean[:50], color=COLORS['primary'], lw=1.5, alpha=1.0, label="Conditional")

    # --- 4. Event Dot ---

    # --- 5. Direct Labeling ---
    # We place text near the end of the curves or where they separate
    # Using the last point of the grid for labeling

    last_date = dates[-4]
    last_y_uncond = result.p_uncond_mean[-1]
    last_y_cond = result.p_cond_mean[-1]

    # Conditional Label
    ax.text(last_date, last_y_cond + 0.02, " Conditional", color=COLORS['primary'], fontsize=8, va='bottom',
            ha='center')

    # Unconditional Label
    # Check for overlap: if they are too close, move unconditional down

    ax.text(last_date, last_y_uncond + 0.02, " Unconditional", color=COLORS['band'], fontsize=8, va='bottom',
            ha='center')

    # --- 6. Formatting ---
    ax.set_ylabel("Probability of the outbreak ending")
    ax.set_ylim(ylim)

    # Extend X-limit slightly for text
    ax.set_xlim(dates[0], dates[-1] + timedelta(days=4))

    # Date Axis
    date_fmt = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))

    # Minimalist Grid
    ax.yaxis.grid(True, color='#eeeeee', linestyle='-', zorder=0)
    ax.xaxis.grid(False)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    pars = (Parameters(
        R0=(0.25, 15),
        k=(0.2, 50),
        r=(0.01, 0.99),
        alpha=(0.01, 30),
        theta=(0.01, 13)
    ).require('R0 * r < 3')
            .require('1 < alpha * theta').require('alpha * theta < 21')
            .require('1/sqrt(alpha) >= 0.1').require('1/sqrt(alpha) <= 1.2')
            .require('1 <= sqrt(alpha) * theta').require('sqrt(alpha) * theta <= 16')
            # .require('((r * R0) ^ (1 / alpha) - 1) / theta <= 0.1')
            )

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

    snapshots = (3, 4, 5, 6, 7, 8)
    # snapshots = (8,)

    T_grid = np.arange(0, 70 + 1e-9, 1.0)

    best_kwargs_by_m = {
        2: {'sigma_days': 2.2568097755997485, 'beta': 0.7998708321056884, 'neighbor_weight': 1.1806520464866863,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.11245573995075447,
            'alpha': 0.32385439495197954, 'h_max': 0.12871399732163935, 'eps_share': 0.09924765304804734,
            'include_gap_windows': True, 'include_union_windows': True, 'max_unions_to_keep': 6,
            'gap_scale': 0.7560703267371328, },
        3: {'sigma_days': 0.5653056052984651, 'beta': 0.6529871512752563, 'neighbor_weight': 0.387089675346202,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.08749158534707935,
            'alpha': 0.14394029922091184, 'h_max': 0.0279233396513767, 'eps_share': 0.09833774700801128,
            'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 4,
            'gap_scale': 0.15707848510892658, },
        4: {'sigma_days': 3.7430822454871073, 'beta': 0.6186183122500877, 'neighbor_weight': 0.7761022110914765,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.06204652329785984,
            'alpha': 0.2536563022079606, 'h_max': 0.38087866375897683, 'eps_share': 0.06508600138947582,
            'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 3,
            'gap_scale': 0.291816139290914, },
        5: {'sigma_days': 2.7144407238423396, 'beta': 0.5661796416762115, 'neighbor_weight': 0.3821037348416254,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.05992387448283569,
            'alpha': 0.13045585214541777, 'h_max': 0.29078303850618525, 'eps_share': 0.09932394410370898,
            'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 2,
            'gap_scale': 0.5494360431590082, },
        6: {'sigma_days': 3.06015118575848, 'beta': 0.5053936723256046, 'neighbor_weight': 0.8892458447406966,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.05864907499636429,
            'alpha': 0.3767759340509563, 'h_max': 0.007239389104548999, 'eps_share': 0.06261011827627241,
            'include_gap_windows': True, 'include_union_windows': True, 'max_unions_to_keep': 3,
            'gap_scale': 0.25039868818075706, },
        7: {'sigma_days': 1.87448405791103, 'beta': 0.8122752531114021, 'neighbor_weight': 0.36814473528396524,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.11350999644046003,
            'alpha': 0.44260858510448914, 'h_max': 0.1860463899250351, 'eps_share': 0.0988640778045239,
            'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 6,
            'gap_scale': 0.15009926426544729, },

        # 8: {'sigma_days': 3.7781097376173833, 'beta': 0.5068644011368587, 'neighbor_weight': 0.11150380609174124,
        #     'grid_step_days': 0.28283975232293723, 'min_seg_days': 3.1876070745495713, 'kmax': 8,
        #     'baseline_p': 0.012782592311887171,
        #     'alpha': 0.39918253112411994, 'h_max': 0.11775991884158551, 'eps_share': 0.016036133279788863,
        #     'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 4,
        #     'gap_scale': 0.21347989807730383, 'include_global_total': True},

        8: {'sigma_days': 3.982802469310849, 'beta': 0.5620160669654297, 'neighbor_weight': 0.14374831069682925,
            'grid_step_days': 0.29535300036388434, 'min_seg_days': 4.829205983803487, 'kmax': 7,
            'baseline_p': 0.04699967547277244,
            'alpha': 0.39922441681969195, 'h_max': 0.10930906122596419, 'eps_share': 0.0011455595753183461,
            'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 1,
            'gap_scale': 0.25437829016143054, 'include_global_total': True},
    }

    # ---- Heavy run parameters (tune for your machine) ----
    scenario = Scenario([])
    results = run_all_snapshots_per_m(
        obs_points=obs_points,
        pars=pars,
        # builder_kwargs_by_m={i: best_kwargs_by_m[7] for i in snapshots},
        builder_kwargs_by_m=best_kwargs_by_m,
        snapshots=snapshots,
        num_trajectories=10_000_000_000,
        chunk_size=100_000,
        T_run=(obs_points[-1][0] - obs_points[0][0]).days + 40,
        max_cases=2000,
        max_workers=13,
        T_grid=T_grid,
        h=0.2,
        H_pad=10.0,
        min_required=30_000
    )

    start_date = obs_points[0][0]
    # plot_rb_online_two_pane(results, start_date)
    # plot_rb_online_right_pane(results, start_date)

    plot_timepath_Re_and_euler_lotka_growth_rate(results, start_date)

    # plot_rb_single_snapshot(results[0], obs_points[0][0])
    # plot_rb_single_snapshot(results[1], obs_points[1][0])
    # plot_rb_single_snapshot(results[2], obs_points[2][0])
    # plot_rb_single_snapshot(results[3], obs_points[3][0])
    # plot_rb_single_snapshot(results[4], obs_points[4][0])
    # plot_rb_single_snapshot(results[5], obs_points[5][0])
    # plot_rb_single_snapshot(results[6], obs_points[6][0])

    # plot_cumulative_infections_datetime(results, start_date=start_date, obs_points=obs_points, u_horizon_days=55)
    # plot_timepath_Re(results, start_date=start_date)
    # plot_last_posterior(results, pdf_estimator='scipy')
    #
    # plot_acceptance_bands(
    #     obs_points,
    #     build_acceptance_inequalities(obs_points=obs_points, simulation_start=start_date)[:-1]
    # )
