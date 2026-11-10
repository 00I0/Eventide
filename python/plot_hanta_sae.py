from __future__ import annotations

"""Hungarian online extinction probability curves, time-dependent parameter path, and posterior plots for the 2026 Hantavirus outbreak.

This script applies the fixed measles extinction algorithm (`python.extinction_probability`)
to the 2026 Hanta observation series using the pre-optimized acceptance-window construction
parameters. By default, it generates the animated Hungarian online extinction probability
curves (`rb_online_right_pane`) with step-by-step presentation outputs over an extended horizon
(defaulting to 50 days, ending on 2026-07-06).

Optional figures can be enabled via flags:
- `--plot-timepath`: 2x2 time-dependent parameter path (`timepath_re_growth_alpha_theta_p0`) from 2026-04-28 to 2026-06-08
- `--plot-posterior`: 2x2 posterior distribution plot conditioned on 10 quiet days (`posterior_conditioned_quiet_for_10d`)
- `--all-plots`: Plot all figures simultaneously
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.eventide import (
    ActiveSetSizeCollector,
    DrawCollector,
    IndexOffspringCriterion,
    InfectionTimeCollector,
    Parameters,
    Scenario,
    Simulator,
)
from python.extinction_probability import (
    conditional_extinction_probability,
    rb_cond_components_post,
)
from python.optimize_acceptance_windows import build_acceptance_inequalities

# ============================================================
# Observations (2026 Hantavirus outbreak)
# ============================================================
OBS_POINTS: List[Tuple[datetime, int]] = [
    (datetime(2026, 4, 6), 1),  # Case 1 (Index Case)
    (datetime(2026, 4, 24), 2),  # Case 2 and Case 3
    (datetime(2026, 4, 27), 1),  # Case 6 (Crew/Guide)
    (datetime(2026, 4, 28), 2),  # Case 4 and Case 8
    (datetime(2026, 4, 30), 1),  # Case 5 (Ship Doctor)
    (datetime(2026, 5, 1), 1),  # Case 7 (Disembarked early, onset in Switzerland)
    (datetime(2026, 5, 11), 1),  # Case 10
    (datetime(2026, 5, 12), 1),  # Case 9
    (datetime(2026, 5, 14), 1),  # Case 11
    (datetime(2026, 5, 17), 1),  # Case 12
]

# ============================================================
# Priors and acceptance criteria
# ============================================================
BASELINE_PRIORS: Dict[str, Tuple[float, float]] = {
    "R0": (0.0, 7.5),
    "k": (0.2, 40.0),
    "r": (0.01, 0.99),
    "alpha": (0.01, 30.0),
    "theta": (0.01, 20.0),
}

BASELINE_REQUIREMENTS: List[str] = [
    "R0 * r < 3",
    "alpha * theta > 15",
    "alpha * theta < 40",
    "(k / (k + R0)) ^ k > 0.25",
    "(k / (k + R0 * r)) ^ k > 0.35",
    "((R0 * r) ^ (1 / alpha) - 1) / theta < 0.05",
]

INDEX_OFFSPRING_BOUNDS: Tuple[int, int] = (3, 7)

# Optimized 2026 Hanta acceptance-window construction parameters per snapshot prefix m.
SNAPSHOT_BUILDER_KWARGS_BY_M: Dict[int, Dict[str, Any]] = {
    1: {
        "sigma_days": 1.1253077611883389,
        "beta": 0.9269668561471294,
        "neighbor_weight": 1.4025293253011095,
        "grid_step_days": 0.7009467884711393,
        "min_seg_days": 4.908118420398451,
        "kmax": 7,
        "baseline_p": 0.1839835231600347,
        "alpha": 0.033285974325294315,
        "h_max": 0.42878229034936355,
        "eps_share": 0.015041145162612917,
        "include_gap_windows": True,
        "include_union_windows": False,
        "max_unions_to_keep": 1,
        "gap_scale": 0.19042846636521432,
        "include_global_total": False,
    },
    2: {
        "sigma_days": 6.081300565314073,
        "beta": 0.974942697570574,
        "neighbor_weight": 0.9466095135488098,
        "grid_step_days": 0.28600913096519653,
        "min_seg_days": 3.1742404903073096,
        "kmax": 3,
        "baseline_p": 0.26647151466594277,
        "alpha": 0.07405030570102389,
        "h_max": 0.9475362071960769,
        "eps_share": 0.037857011907687935,
        "include_gap_windows": True,
        "include_union_windows": False,
        "max_unions_to_keep": 1,
        "gap_scale": 0.11569130163797606,
        "include_global_total": True,
    },
    3: {
        "sigma_days": 0.7653403148896346,
        "beta": 0.5987485961653242,
        "neighbor_weight": 1.6500754790932635,
        "grid_step_days": 0.3650895193940761,
        "min_seg_days": 1.4109653642254854,
        "kmax": 5,
        "baseline_p": 0.014789432422911065,
        "alpha": 0.39840034061498847,
        "h_max": 0.1308972657711816,
        "eps_share": 0.039999923460640276,
        "include_gap_windows": True,
        "include_union_windows": True,
        "max_unions_to_keep": 4,
        "gap_scale": 0.8082759755680968,
        "include_global_total": False,
    },
    4: {
        "sigma_days": 7.797135826075775,
        "beta": 0.5771000378758345,
        "neighbor_weight": 0.5951076064929567,
        "grid_step_days": 0.10286027476969772,
        "min_seg_days": 4.695535042493828,
        "kmax": 3,
        "baseline_p": 0.010344940919398766,
        "alpha": 0.29430663371538485,
        "h_max": 0.12129504877838976,
        "eps_share": 0.013240765166649696,
        "include_gap_windows": True,
        "include_union_windows": True,
        "max_unions_to_keep": 1,
        "gap_scale": 0.2933330295753821,
        "include_global_total": False,
    },
    5: {
        "sigma_days": 2.997890886490486,
        "beta": 0.500857012796471,
        "neighbor_weight": 0.11172096580404678,
        "grid_step_days": 0.10008039098466114,
        "min_seg_days": 4.771015548674616,
        "kmax": 7,
        "baseline_p": 0.016478174548807012,
        "alpha": 0.36847233840090193,
        "h_max": 0.10032616559324287,
        "eps_share": 0.09063341233668298,
        "include_gap_windows": False,
        "include_union_windows": False,
        "max_unions_to_keep": 6,
        "gap_scale": 0.2651847132294505,
        "include_global_total": True,
    },
    6: {
        "sigma_days": 7.991074504598734,
        "beta": 0.653809222878009,
        "neighbor_weight": 0.10930557297885897,
        "grid_step_days": 0.1211312366026706,
        "min_seg_days": 0.5118707082604929,
        "kmax": 8,
        "baseline_p": 0.1343024808675321,
        "alpha": 0.36999651398652245,
        "h_max": 0.3445638188588871,
        "eps_share": 0.06668312808060688,
        "include_gap_windows": False,
        "include_union_windows": True,
        "max_unions_to_keep": 1,
        "gap_scale": 0.39381732090998234,
        "include_global_total": False,
    },
    7: {
        "sigma_days": 5.873296919450529,
        "beta": 0.5824845115470392,
        "neighbor_weight": 0.291796990168927,
        "grid_step_days": 0.12331531726891505,
        "min_seg_days": 0.5823519413501396,
        "kmax": 6,
        "baseline_p": 0.12562279663847944,
        "alpha": 0.23691382509886752,
        "h_max": 0.1025574741102243,
        "eps_share": 0.015025082676865954,
        "include_gap_windows": True,
        "include_union_windows": True,
        "max_unions_to_keep": 1,
        "gap_scale": 0.10426792241931677,
        "include_global_total": True,
    },
    8: {
        "sigma_days": 4.030865507521659,
        "beta": 0.9307912574927261,
        "neighbor_weight": 1.3376802408030994,
        "grid_step_days": 0.3287166256320232,
        "min_seg_days": 2.4266296495384783,
        "kmax": 7,
        "baseline_p": 0.010261854691918907,
        "alpha": 0.37510170167668483,
        "h_max": 0.2182158786222464,
        "eps_share": 0.06200349156077785,
        "include_gap_windows": False,
        "include_union_windows": True,
        "max_unions_to_keep": 4,
        "gap_scale": 0.43705266298039136,
        "include_global_total": True,
    },
    9: {
        "sigma_days": 2.2224233697676192,
        "beta": 0.8563752728749588,
        "neighbor_weight": 2.1403570387301323,
        "grid_step_days": 0.6463033889253157,
        "min_seg_days": 0.6700350548491563,
        "kmax": 7,
        "baseline_p": 0.01710636813601208,
        "alpha": 0.3025610165973527,
        "h_max": 0.6279154110638231,
        "eps_share": 0.09950652720696691,
        "include_gap_windows": False,
        "include_union_windows": True,
        "max_unions_to_keep": 6,
        "gap_scale": 0.8389161575264481,
        "include_global_total": False,
    },
    10: {
        "sigma_days": 2.2224233697676192,
        "beta": 0.8563752728749588,
        "neighbor_weight": 2.1403570387301323,
        "grid_step_days": 0.6463033889253157,
        "min_seg_days": 0.6700350548491563,
        "kmax": 7,
        "baseline_p": 0.01710636813601208,
        "alpha": 0.3025610165973527,
        "h_max": 0.6279154110638231,
        "eps_share": 0.09950652720696691,
        "include_gap_windows": False,
        "include_union_windows": True,
        "max_unions_to_keep": 6,
        "gap_scale": 0.8389161575264481,
        "include_global_total": False,
    },
}

# ============================================================
# Simulation & Plotting Constants (matching plot_sae.py)
# ============================================================
TIMEPATH_T_GRID = np.arange(0.0, 100.0 + 1e-9, 0.25)
T_RUN = (OBS_POINTS[-1][0] - OBS_POINTS[0][0]).days + 65
NUM_TRAJECTORIES = 1_000_000_000_000_000
CHUNK_SIZE = 50_000
MAX_CASES = 200
MAX_WORKERS = 13
MIN_REQUIRED = 1_000  # Default small required draws for fast pilot/smoketest runs

ONE_COLUMN_WIDTH = 90 / 25.4
TWO_COLUMN_WIDTH = 190 / 25.4
ONE_COLUMN_HEIGHT = 70 / 25.4
TIMEPATH_HEIGHT = 75 / 25.4
SAVE_FIGURES = True
LANGUAGE: Literal["en", "hu"] = "hu"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "plots" / "hanta_2026_hu"
TICK_LABEL_SIZE = 7.25
AXIS_LABEL_SIZE = 8.0
LEGEND_FONT_SIZE = 7.25
PANEL_HEADING_SIZE = 9.0
MAIN_LINE_WIDTH = 1.0
OBSERVATION_ZORDER = 100

COLORS = {
    "primary": "#0072B2",
    "secondary": "#7F7F7F",
    "obs": "#000000",
    "axis": "#333333",
    "guide": "#BDBDBD",
    "grid": "#E5E5E5",
    "re": "#0072B2",
    "growth": "#D55E00",
    "alpha_theta": "#009E73",
    "p0": "#CC79A7",
}


@dataclass
class TimepathSnapshotResult:
    m: int
    t_star: float
    T_grid: np.ndarray
    p_uncond_mean: np.ndarray
    p_cond_mean: np.ndarray
    p_uncond_draws: np.ndarray
    p_cond_draws: np.ndarray
    draws_array: np.ndarray
    infection_times_2d: List[np.ndarray]
    n_obs: int
    next_T: Optional[float]
    stopped_pairs: List[Sequence[Tuple[float, float]]] | None
    accepted_count: int
    processed_count: int


# ============================================================
# Styling & Formatting (from plot_sae.py)
# ============================================================
def _save_figure(
        fig: plt.Figure,
        filename: str,
        output_dir: Path,
        save: bool = True,
        formats: Sequence[str] = ("pdf", "png"),
        dpi: int = 300,
) -> List[Path]:
    """Save a matplotlib figure to disk in both PDF and PNG formats."""
    saved_paths: List[Path] = []
    if not save:
        return saved_paths
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = output_dir / f"{filename}.{fmt}"
        fig.savefig(target, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        saved_paths.append(target)
        print(f"Saved: {target}")
    return saved_paths


def set_journal_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": AXIS_LABEL_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": PANEL_HEADING_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "axes.linewidth": 0.75,
        "lines.linewidth": MAIN_LINE_WIDTH,
        "axes.edgecolor": COLORS["axis"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def set_timepath_style() -> None:
    set_journal_style()


def _format_tick(value: float, is_edge: bool = False) -> str:
    """Format tick value: integer when close to whole, 2 decimals otherwise."""
    if abs(value) < 1e-9:
        return "0"
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_date(value: float, position: int) -> str:
    date = mdates.num2date(value)
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    return f"{date.day} {month_names[date.month]}"


def _format_hungarian_date(value: float, position: int) -> str:
    date = mdates.num2date(value)
    month_names = {
        1: "jan.", 2: "febr.", 3: "márc.", 4: "ápr.", 5: "máj.", 6: "jún.",
        7: "júl.", 8: "aug.", 9: "szept.", 10: "okt.", 11: "nov.", 12: "dec.",
    }
    return f"{month_names[date.month]} {date.day}."


def _sort_results_chronologically(results: Sequence[Any]) -> List[Any]:
    indexed = list(enumerate(results))
    indexed.sort(key=lambda x: (float(getattr(x[1], "t_star", 0.0)), x[0]))
    return [res for _, res in indexed]


def _filter_results_by_axis_start_date(
        ordered_results: Sequence[Any],
        start_date: datetime,
        axis_start_date: Optional[datetime],
) -> List[Any]:
    if axis_start_date is None:
        return list(ordered_results)
    return [
        res for res in ordered_results
        if start_date + timedelta(days=float(getattr(res, "t_star", 0.0))) >= axis_start_date
    ]


def _get_first_post_infection_times(result: Any) -> np.ndarray:
    t_star = float(result.t_star)
    first_posts = []
    for traj in result.infection_times_2d:
        ts = np.sort(np.asarray(traj, dtype=float))
        future = ts[ts > t_star]
        first_posts.append(float(future[0] - t_star) if future.size else np.inf)
    return np.asarray(first_posts, dtype=float)


def _retain_mask(first_post: np.ndarray, T: float, index: int) -> np.ndarray:
    if index == 0:
        return np.ones(first_post.shape[0], dtype=bool)
    return first_post > T


def _extract_metric_values(draws: np.ndarray) -> Dict[str, np.ndarray]:
    if draws.ndim != 2 or draws.shape[0] == 0:
        empty = np.empty(0, dtype=float)
        return {"Re": empty, "growth": empty, "alpha_theta": empty, "p0": empty}
    R0, k, reproduction, alpha, theta = draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3], draws[:, 4]
    effective = R0 * reproduction
    alpha_safe = np.maximum(alpha, 1e-12)
    theta_safe = np.maximum(theta, 1e-12)
    k_safe = np.maximum(k, 1e-12)
    return {
        "Re": effective,
        "growth": (np.power(np.maximum(effective, 1e-12), 1.0 / alpha_safe) - 1.0) / theta_safe,
        "alpha_theta": alpha * theta,
        "p0": np.power(k_safe / (k_safe + np.maximum(effective, 0.0)), k_safe),
    }


def _get_segment_end_days(
        ordered_results: Sequence[Any],
        final_horizon_days: float,
) -> Tuple[List[float], List[float]]:
    if final_horizon_days < 0:
        raise ValueError("final_horizon_days must be non-negative.")
    segment_ends: List[float] = []
    jump_days: List[float] = []
    for index, result in enumerate(ordered_results):
        start_day = float(result.t_star)
        if index < len(ordered_results) - 1:
            end_day = float(ordered_results[index + 1].t_star)
            if end_day < start_day:
                end_day = start_day + (float(result.next_T) if result.next_T is not None else 0.0)
            jump_days.append(end_day)
        else:
            end_day = start_day + final_horizon_days
        segment_ends.append(end_day)
    return segment_ends, jump_days


def _apply_common_date_axis(
        axes: Sequence[plt.Axes],
        start_date: datetime,
        final_day: float,
        axis_start_date: Optional[datetime] = None,
        axis_end_date: Optional[datetime] = None,
        lang: Literal["en", "hu"] = "hu",
) -> None:
    formatter = _format_hungarian_date if lang == "hu" else _format_date
    x_min = axis_start_date if axis_start_date is not None else start_date
    x_max = axis_end_date if axis_end_date is not None else (start_date + timedelta(days=float(final_day)))
    for axis in axes:
        axis.xaxis.set_major_formatter(FuncFormatter(formatter))
        axis.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        axis.tick_params(axis="x", labelrotation=0)
        axis.set_xlim(x_min, x_max)


def _kde_smart(
        x: np.ndarray,
        grid: np.ndarray,
        bounds: Tuple[float, float],
        method: Literal["reflect", "truncate"] = "truncate",
        estimator: str = "scipy",
) -> np.ndarray:
    """Compute normalized Gaussian KDE over `grid` with boundary handling, directly matching plot_sae.py."""
    x_valid = np.asarray(x, dtype=float)
    x_valid = x_valid[np.isfinite(x_valid)]
    if x_valid.size == 0:
        return np.zeros_like(grid)
    lo, hi = bounds
    if method == "reflect":
        bw_proxy = (np.max(x_valid) - np.min(x_valid)) * 0.4
        x_proc = np.concatenate([
            x_valid,
            (2 * lo - x_valid)[2 * lo - x_valid > lo - bw_proxy],
            (2 * hi - x_valid)[2 * hi - x_valid < hi + bw_proxy],
        ])
    else:
        x_proc = x_valid

    if estimator == "scipy":
        if len(x_proc) > 1 and np.ptp(x_proc) > 1e-12:
            kde = gaussian_kde(x_proc)
            pdf = kde(grid)
        elif len(x_proc) >= 1:
            bandwidth = max(grid[1] - grid[0], 1e-6)
            pdf = np.exp(-0.5 * ((grid - x_proc[0]) / bandwidth) ** 2)
        else:
            return np.zeros_like(grid)
    else:
        from sklearn.neighbors import KernelDensity
        kernel = estimator.split("_", 1)[1] if "_" in estimator else "epanechnikov"
        sigma = float(np.std(x_proc))
        q75, q25 = np.percentile(x_proc, [75, 25])
        iqr = q75 - q25
        sigma = min(sigma, iqr / 1.34) if iqr > 0 else sigma
        bandwidth = 0.9 * (sigma or 1.0) * len(x_proc) ** (-0.2)
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(x_proc[:, None])
        pdf = np.exp(kde.score_samples(grid[:, None]))

    integral = np.trapezoid(pdf, grid) if hasattr(np, "trapezoid") else np.trapz(pdf, grid)
    return pdf / integral if integral > 0 else pdf


def _get_last_draws_conditioned_quiet_for_T(
        results: Sequence[Any],
        T: Optional[float],
        *,
        t_star: Optional[float] = None,
) -> np.ndarray:
    """Extract posterior draws from the final snapshot conditioned on T quiet days, matching plot_sae.py."""
    if not results:
        return np.empty((0, 0))
    result = results[-1]
    draws = np.asarray(result.draws_array, dtype=float)
    if draws.size == 0 or T is None:
        return draws
    if T < 0:
        raise ValueError("T must be non-negative (days).")
    infection_times = list(getattr(result, "infection_times_2d", []))
    if not infection_times:
        return np.empty((0, draws.shape[1] if draws.ndim == 2 else 0))
    t_star = float(result.t_star if t_star is None else t_star)
    count = min(draws.shape[0], len(infection_times))
    first_post = []
    for trajectory in infection_times[:count]:
        times = np.sort(np.asarray(trajectory, dtype=float))
        future = times[times > t_star]
        first_post.append(float(future[0] - t_star) if future.size else np.inf)
    first_post_arr = np.asarray(first_post, dtype=float)
    return draws[:count][first_post_arr > float(T)]


# ============================================================
# Plotting Functions
# ============================================================
def plot_last_posterior_conditioned_quiet_for_T(
        results: Sequence[Any],
        T: float = 10.0,
        *,
        figsize: Tuple[float, float] = (115 / 25.4, 85 / 25.4),
        pdf_estimator: str = "scipy",
        t_star: Optional[float] = None,
        save: bool = SAVE_FIGURES,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        show: bool = True,
) -> Optional[plt.Figure]:
    """Plot the 2x2 posterior distribution for the last snapshot conditioned on T quiet days.

    Follows the exact visual styling and tick layout from `plot_sae.py`:
    - Unified primary blue outline and 95% CI fill
    - Dashed secondary line for median
    - Clean x-axis with edge ticks, median tick, and colored CI ticks
    - Invisible top/left/right spines, no y-ticks
    """
    set_journal_style()
    if not results:
        return None
    if T < 0:
        raise ValueError("T must be non-negative (days).")

    draws = _get_last_draws_conditioned_quiet_for_T(results, T=T, t_star=t_star)
    if draws.size == 0:
        print(f"Warning: No trajectories retained after conditioning on {T} quiet days.")
        return None
    print(f"Conditioned on {T:.1f} quiet days: {len(draws)} trajectories retained.")

    R0, k, reproduction, alpha, theta = draws.T
    effective = R0 * reproduction
    alpha_theta = alpha * theta
    p0 = (k / (k + effective + 1e-12)) ** k
    alpha_safe = np.maximum(alpha, 1e-12)
    theta_safe = np.maximum(theta, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        growth = (np.power(np.maximum(effective, 1e-12), 1.0 / alpha_safe) - 1.0) / theta_safe

    variables = [
        (effective, r"$R_c$", (0.0, 3.0)),
        (growth, r"$g$", (-0.15, 0.085)),
        (alpha_theta, r"$\alpha\theta$", (15.0, 50.0)),
        (p0, r"$P(N=0)$", (0.35, 0.95)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for axis, (data, title, bounds) in zip(axes.flat, variables):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        lower, upper = bounds
        grid = np.linspace(lower, upper, 600)
        density = _kde_smart(data, grid, bounds, "truncate", estimator=pdf_estimator)
        median = float(np.median(data))
        q_low, q_high = float(np.percentile(data, 2.5)), float(np.percentile(data, 97.5))

        # Outline and 95% CI shaded region
        axis.plot(grid, density, color=COLORS["primary"], lw=MAIN_LINE_WIDTH, zorder=10)
        axis.fill_between(
            grid, 0, density, where=(grid >= q_low) & (grid <= q_high),
            color=COLORS["primary"], alpha=0.35, linewidth=0,
        )

        # Median dashed vertical line
        axis.vlines(
            median, 0, float(np.interp(median, grid, density)),
            color=COLORS["secondary"], linestyle="--", lw=MAIN_LINE_WIDTH,
        )
        axis.set_xlim(lower, upper)

        # SAE tick formatting: min, median, max plus non-colliding colored CI bounds
        tick_specs = [
            (lower, _format_tick(lower, True), False),
            (upper, _format_tick(upper, True), False),
        ]
        if lower < median < upper:
            tick_specs.append((median, _format_tick(median), False))
        ci_specs = [
            (q_low, _format_tick(q_low), True),
            (q_high, _format_tick(q_high), True),
        ]
        axis_span = upper - lower
        if axis_span > 0:
            minimum_tick_gap = 0.075 * axis_span
            for ci_spec in sorted(ci_specs, key=lambda item: item[0]):
                if lower < ci_spec[0] < upper and all(
                        abs(ci_spec[0] - existing[0]) >= minimum_tick_gap for existing in tick_specs):
                    tick_specs.append(ci_spec)
            tick_specs.sort(key=lambda item: item[0])

        axis.set_xticks([spec[0] for spec in tick_specs])
        axis.set_xticklabels([spec[1] for spec in tick_specs])
        for tick, (_, _, is_ci) in zip(axis.xaxis.get_major_ticks(), tick_specs):
            if is_ci:
                tick.label1.set_color(COLORS["primary"])
                tick.tick1line.set_color(COLORS["primary"])
                tick.tick2line.set_color(COLORS["primary"])

        axis.set_xlim(lower, upper)
        axis.set_title(title, pad=6, fontsize=PANEL_HEADING_SIZE)
        axis.set_ylim(bottom=0)
        axis.set_yticks([])
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.spines["bottom"].set_color(COLORS["axis"])
        axis.spines["bottom"].set_linewidth(0.75)

    fig.tight_layout(w_pad=1.2, h_pad=1.2)
    _save_figure(fig, f"posterior_conditioned_quiet_for_{int(round(T))}d", output_dir=output_dir, save=save)
    if show:
        plt.show()
    return fig


def plot_timepath_re_growth_alpha_theta_p0(
        results: Sequence[Any],
        start_date: datetime,
        *,
        step: float = 0.1,
        final_horizon_days: float = 20.0,
        perc_bands: Sequence[float] = (0.95, 0.8, 0.5),
        figsize: Tuple[float, float] = (TWO_COLUMN_WIDTH, TIMEPATH_HEIGHT),
        ylims: Optional[Mapping[str, Tuple[float, float]]] = None,
        axis_start_date: Optional[datetime] = datetime(2026, 4, 28),
        axis_end_date: Optional[datetime] = datetime(2026, 6, 8),
        lang: Literal["en", "hu"] = LANGUAGE,
        save_path: Optional[Path] = None,
        save: bool = SAVE_FIGURES,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        show: bool = True,
) -> Optional[plt.Figure]:
    """Plot the 2x2 time-dependent parameter figure from plot_sae.py starting from 2026-04-28 to 2026-06-08."""
    set_timepath_style()
    results_filtered = [res for res in results if getattr(res, "m", 0) > 1]
    if not results_filtered:
        return None

    if axis_start_date is None:
        axis_start_date = datetime(2026, 4, 28)
    if axis_end_date is None:
        axis_end_date = datetime(2026, 6, 8)

    ordered = _filter_results_by_axis_start_date(
        _sort_results_chronologically(results_filtered),
        start_date,
        axis_start_date,
    )
    if not ordered:
        return None

    last_t_star = float(ordered[-1].t_star)
    last_date = start_date + timedelta(days=last_t_star)
    if axis_end_date is not None and axis_end_date > last_date:
        effective_final_horizon = (axis_end_date - last_date).total_seconds() / 86400.0
    else:
        effective_final_horizon = final_horizon_days

    segment_ends, jump_days = _get_segment_end_days(ordered, effective_final_horizon)
    metric_names = ("Re", "growth", "alpha_theta", "p0")
    metrics = {
        name: {"mid": [], "bands": {level: {"lo": [], "hi": []} for level in perc_bands}}
        for name in metric_names
    }
    x_segments = []
    for result, end_day in zip(ordered, segment_ends):
        horizon = end_day - float(result.t_star)
        if horizon <= 0:
            continue
        evaluation = np.linspace(0.0, horizon, max(1, int(np.floor(horizon / step)) + 1))
        draws = np.asarray(result.draws_array, dtype=float)
        first_post = _get_first_post_infection_times(result)
        count = min(len(draws), len(first_post))
        if count == 0:
            continue
        values = _extract_metric_values(draws[:count])
        first_post = first_post[:count]
        segment_stats = {
            name: {"mid": [], "bands": {level: {"lo": [], "hi": []} for level in perc_bands}}
            for name in metric_names
        }
        for index, elapsed in enumerate(evaluation):
            keep = _retain_mask(first_post, float(elapsed), index)
            for name in metric_names:
                finite = values[name][keep]
                finite = finite[np.isfinite(finite)]
                segment_stats[name]["mid"].append(float(np.median(finite)) if finite.size else np.nan)
                for level in perc_bands:
                    if finite.size:
                        tail = (1.0 - level) / 2
                        segment_stats[name]["bands"][level]["lo"].append(float(np.percentile(finite, tail * 100)))
                        segment_stats[name]["bands"][level]["hi"].append(
                            float(np.percentile(finite, (1 - tail) * 100))
                        )
                    else:
                        segment_stats[name]["bands"][level]["lo"].append(np.nan)
                        segment_stats[name]["bands"][level]["hi"].append(np.nan)
        x_segments.append(float(result.t_star) + evaluation)
        for name in metric_names:
            metrics[name]["mid"].append(np.asarray(segment_stats[name]["mid"]))
            for level in perc_bands:
                metrics[name]["bands"][level]["lo"].append(np.asarray(segment_stats[name]["bands"][level]["lo"]))
                metrics[name]["bands"][level]["hi"].append(np.asarray(segment_stats[name]["bands"][level]["hi"]))
    if not x_segments:
        return None
    x_numeric = np.concatenate(x_segments)
    x_dates = np.asarray([start_date + timedelta(days=float(value)) for value in x_numeric], dtype=object)
    panels = [
        ("Re", r"$R_c$", COLORS["re"], 1.0),
        ("growth", r"$g$", COLORS["growth"], 0.0),
        ("alpha_theta", r"$\alpha\theta$", COLORS["alpha_theta"], None),
        ("p0", r"$P(N=0)$", COLORS["p0"], None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    for axis, (name, label, color, reference) in zip(axes.flat, panels):
        mid = np.concatenate(metrics[name]["mid"])
        if reference is not None:
            axis.axhline(reference, color=COLORS["grid"], linestyle=":", linewidth=0.8)
        finite_values = []
        for index, level in enumerate(sorted(perc_bands, reverse=True)):
            low = np.concatenate(metrics[name]["bands"][level]["lo"])
            high = np.concatenate(metrics[name]["bands"][level]["hi"])
            good = np.isfinite(low) & np.isfinite(high)
            if np.any(good):
                finite_values.extend([float(np.nanmin(low[good])), float(np.nanmax(high[good]))])
                axis.fill_between(x_dates, low, high, color=color, alpha=0.14 + 0.07 * index, linewidth=0)
        good_mid = np.isfinite(mid)
        axis.plot(x_dates[good_mid], mid[good_mid], color=color, lw=MAIN_LINE_WIDTH)
        for jump_day in jump_days:
            axis.axvline(start_date + timedelta(days=jump_day), color=COLORS["secondary"], lw=0.8)
        axis.set_ylabel(label)
        axis.yaxis.grid(True, color=COLORS["grid"], linestyle="-", alpha=0.5, lw=0.8)
        if name == "growth":
            axis.set_ylim(-0.2, 0.1)
        elif ylims and name in ylims:
            axis.set_ylim(*ylims[name])
        elif finite_values:
            low, high = min(finite_values), max(finite_values)
            if reference is not None:
                low, high = min(low, reference), max(high, reference)
            padding = 0.08 * max(high - low, 1e-9)
            axis.set_ylim(low - padding, high + padding)
    _apply_common_date_axis(axes.flat, start_date, segment_ends[-1], axis_start_date=axis_start_date,
                            axis_end_date=axis_end_date, lang=lang)
    fig.tight_layout()
    _save_figure(fig, "timepath_re_growth_alpha_theta_p0", output_dir=output_dir, save=save)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    return fig


def plot_rb_online_right_pane(
        results: Sequence[Any],
        start_date: datetime,
        *,
        ylim: Tuple[float, float] = (0.0, 1.05),
        figsize: Tuple[float, float] = (ONE_COLUMN_WIDTH, 65 / 25.4),
        final_horizon_days: float = 50.0,
        axis_start_date: Optional[datetime] = None,
        axis_end_date: Optional[datetime] = None,
        lang: Literal["en", "hu"] = LANGUAGE,
        animate: bool = True,
        save_path: Optional[Path] = None,
        save: bool = SAVE_FIGURES,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        show: bool = True,
) -> Optional[plt.Figure]:
    """Plot the Hungarian online extinction probability curves with progressive animation over an extended horizon."""
    set_timepath_style()
    ordered = _filter_results_by_axis_start_date(
        _sort_results_chronologically(results),
        start_date,
        axis_start_date,
    )
    if not ordered:
        return None
    is_hu = lang == "hu"
    legend_title = "Megfigyelt esetek, $N$" if is_hu else "Observed cases, $N$"
    ylabel = "Valószínűség" if is_hu else "Probability that the outbreak is over"

    colors = [plt.get_cmap("Blues")(value) for value in np.linspace(0.4, 1.0, len(ordered))]
    fig, axis = plt.subplots(figsize=figsize)
    if axis_end_date is not None:
        max_days = (axis_end_date - start_date).total_seconds() / 86400.0
    else:
        max_days = max(
            float(result.t_star) + (float(result.next_T) if result.next_T is not None else final_horizon_days)
            for result in ordered
        )
    snapshot_artists: List[List[Any]] = []
    for index, result in enumerate(ordered):
        artists_for_snapshot: List[Any] = []
        y_values = np.asarray(result.p_cond_mean, dtype=float)
        snapshot_day = float(result.t_star)
        x_values = snapshot_day + np.asarray(result.T_grid, dtype=float)
        if not y_values.size or not x_values.size:
            snapshot_artists.append(artists_for_snapshot)
            continue
        if result.next_T is None:
            visible = x_values <= max_days
            x_plot = x_values[visible]
            y_plot = y_values[visible]
            line_full, = axis.plot(
                [start_date + timedelta(days=float(value)) for value in x_plot],
                y_plot,
                color=colors[index],
                lw=MAIN_LINE_WIDTH,
                zorder=10 - index,
            )
            artists_for_snapshot.append(line_full)
            snapshot_artists.append(artists_for_snapshot)
            continue
        event_x = snapshot_day + float(result.next_T)
        event_y = float(np.interp(result.next_T, result.T_grid, y_values))
        pre_x = np.append(x_values[x_values < event_x], event_x)
        pre_y = np.append(y_values[x_values < event_x], event_y)
        post_x = np.concatenate(([event_x], x_values[x_values > event_x]))
        post_y = np.concatenate(([event_y], y_values[x_values > event_x]))
        line_pre, = axis.plot(
            [start_date + timedelta(days=float(value)) for value in pre_x],
            pre_y,
            color=colors[index],
            lw=MAIN_LINE_WIDTH,
            zorder=10 - index,
        )
        artists_for_snapshot.append(line_pre)
        if post_x.size >= 2:
            line_post, = axis.plot(
                [start_date + timedelta(days=float(value)) for value in post_x],
                post_y,
                color=colors[index],
                lw=MAIN_LINE_WIDTH,
                linestyle=(0, (2, 0.5)),
                dash_capstyle="butt",
            )
            artists_for_snapshot.append(line_post)
        point_event = axis.scatter(
            [start_date + timedelta(days=event_x)],
            [event_y],
            s=23,
            facecolors="white",
            edgecolors=colors[index],
            lw=MAIN_LINE_WIDTH,
            zorder=OBSERVATION_ZORDER,
        )
        artists_for_snapshot.append(point_event)
        snapshot_artists.append(artists_for_snapshot)

    legend_handles = [plt.Line2D([0], [0], color=color, lw=MAIN_LINE_WIDTH) for color in colors]
    legend_labels = [f"$N={result.n_obs}$" for result in ordered]
    axis.set_ylabel(ylabel)
    axis.set_ylim(ylim)
    axis.yaxis.grid(True, color=COLORS["grid"], linestyle="-")
    _apply_common_date_axis([axis], start_date, max_days, axis_start_date=axis_start_date, axis_end_date=axis_end_date,
                            lang=lang)
    fig.tight_layout()

    # Measure top-right position of the complete legend so progressive steps share the same anchor
    ref_legend = axis.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        title=legend_title,
        title_fontsize=LEGEND_FONT_SIZE,
    )
    fig.canvas.draw()
    ref_bbox = ref_legend.get_window_extent()
    legend_top_right = tuple(axis.transAxes.inverted().transform((ref_bbox.x1, ref_bbox.y1)))
    ref_legend.remove()

    def _set_visible_count(visible_count: int) -> None:
        for idx, group in enumerate(snapshot_artists):
            is_vis = idx < visible_count
            for artist in group:
                artist.set_visible(is_vis)
        if axis.get_legend() is not None:
            axis.get_legend().remove()
        axis.legend(
            legend_handles[:visible_count],
            legend_labels[:visible_count],
            loc="upper right",
            bbox_to_anchor=legend_top_right,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            title=legend_title,
            title_fontsize=LEGEND_FONT_SIZE,
        )

    if animate and show:
        for step in range(1, len(ordered) + 1):
            _set_visible_count(step)
            plt.draw()
            plt.pause(0.35)

    if animate and save:
        for step in range(1, len(ordered) + 1):
            _set_visible_count(step)
            _save_figure(fig, f"rb_online_right_pane_step_{step}", output_dir=output_dir, save=save)

    _set_visible_count(len(ordered))
    _save_figure(fig, "rb_online_right_pane", output_dir=output_dir, save=save)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    return fig


# ============================================================
# Simulation Runners (using Measles Fixed Extinction Algorithm)
# ============================================================
def _build_parameters(
        *,
        priors: Mapping[str, Tuple[float, float]] = BASELINE_PRIORS,
        requirements: Sequence[str] = BASELINE_REQUIREMENTS,
) -> Parameters:
    pars = Parameters(
        R0=priors["R0"],
        k=priors["k"],
        r=priors["r"],
        alpha=priors["alpha"],
        theta=priors["theta"],
    )
    for req in requirements:
        pars = pars.require(req)
    return pars


def _cast_accept_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    if "kmax" in out:
        out["kmax"] = int(round(float(out["kmax"])))
    for name in ("include_gap_windows", "include_union_windows", "include_global_total"):
        if name in out:
            out[name] = bool(round(float(out[name])))
    if "max_unions_to_keep" in out:
        out["max_unions_to_keep"] = int(round(float(out["max_unions_to_keep"])))
    return out


def _build_criteria_for_prefix(
        obs_points: Sequence[Tuple[datetime, int]],
        m: int,
        *,
        index_offspring_bounds: Tuple[int, int] = INDEX_OFFSPRING_BOUNDS,
        **builder_kwargs: Any,
) -> Tuple[List[Any], datetime]:
    pts = obs_points[:m]
    sim_start = min(t for t, _ in pts)
    criteria = [IndexOffspringCriterion(*index_offspring_bounds)] + build_acceptance_inequalities(
        obs_points=pts,
        simulation_start=sim_start,
        **_cast_accept_kwargs(builder_kwargs),
    )
    return criteria, sim_start


def run_snapshot_with_counts(
        *,
        m: int,
        obs_points: Sequence[Tuple[datetime, int]],
        pars: Parameters,
        builder_kwargs: Mapping[str, Any],
        scenario: Optional[Scenario] = None,
        index_offspring_bounds: Tuple[int, int] = INDEX_OFFSPRING_BOUNDS,
        num_trajectories: int = NUM_TRAJECTORIES,
        chunk_size: int = CHUNK_SIZE,
        T_run: int = T_RUN,
        max_cases: int = MAX_CASES,
        max_workers: int = MAX_WORKERS,
        T_grid: np.ndarray = TIMEPATH_T_GRID,
        min_required: Optional[int] = MIN_REQUIRED,
) -> TimepathSnapshotResult:
    """Run ABC for the first `m` observations and compute fixed measles extinction probabilities."""
    if scenario is None:
        scenario = Scenario([])

    criteria, sim_start = _build_criteria_for_prefix(
        obs_points,
        m,
        index_offspring_bounds=index_offspring_bounds,
        **builder_kwargs,
    )

    collectors = [
        draws := DrawCollector(),
        active_set := ActiveSetSizeCollector(obs_points[m - 1][0]),
        infection_times := InfectionTimeCollector(),
    ]

    sim = Simulator(
        parameters=pars,
        sampler=pars.create_latin_hypercube_sampler(),
        start_date=sim_start,
        scenario=scenario,
        criteria=criteria,
        collectors=collectors,
        num_trajectories=num_trajectories,
        chunk_size=chunk_size,
        T_run=T_run,
        max_cases=max_cases,
        max_workers=max_workers,
        min_required=min_required,
    )

    started = time.time()
    sim.run()
    print(f"Run {m} took {time.time() - started:.2f} seconds")

    draws_array = np.asarray(draws, dtype=float)
    if draws_array.ndim == 1 and draws_array.size:
        draws_array = draws_array.reshape(1, -1)

    infection_times_2d = list(infection_times.infection_times)
    stopped_pairs = active_set.active_sets
    t_star = (active_set.collection_date - sim.start_date).total_seconds() / 86400.0
    accepted_count = int(sim.accepted or len(infection_times_2d))
    processed_count = int(sim.processed or 0)
    print(f"Run {m} accepted {len(infection_times_2d)}")

    T_grid = np.asarray(T_grid, dtype=float)
    if draws_array.size == 0 or not infection_times_2d:
        p_uncond_mean = np.full(T_grid.shape, np.nan)
        p_cond_mean = np.full(T_grid.shape, np.nan)
        p_uncond_draws = np.empty((0, T_grid.size))
        p_cond_draws = np.empty((0, T_grid.size))
    else:
        R0s, ks, rs, alphas, thetas = draws_array.T
        arguments = (infection_times_2d, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star)
        p_cond_mean = conditional_extinction_probability(
            *arguments,
            index_offspring_bounds=index_offspring_bounds,
        )
        g_cond_inf, g_cond_quiet = rb_cond_components_post(
            *arguments,
            index_offspring_bounds=index_offspring_bounds,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            p_cond_draws = np.where(g_cond_quiet > 0.0, g_cond_inf[:, None] / g_cond_quiet, np.nan)
            p_cond_draws = np.clip(p_cond_draws, 0.0, 1.0)
        p_uncond_mean = np.full(T_grid.shape, np.nan)
        p_uncond_draws = np.empty((0, T_grid.size))

    next_T = (
        (obs_points[m][0] - obs_points[m - 1][0]).total_seconds() / 86400.0
        if m < len(obs_points)
        else None
    )
    n_obs = int(sum(y for _, y in obs_points[:m]))

    return TimepathSnapshotResult(
        m=m,
        t_star=float(t_star),
        T_grid=T_grid,
        p_uncond_mean=p_uncond_mean,
        p_cond_mean=p_cond_mean,
        p_uncond_draws=p_uncond_draws,
        p_cond_draws=p_cond_draws,
        draws_array=draws_array,
        infection_times_2d=infection_times_2d,
        n_obs=n_obs,
        next_T=next_T,
        stopped_pairs=stopped_pairs,
        accepted_count=accepted_count,
        processed_count=processed_count,
    )


def run_all_snapshots_with_counts(
        *,
        obs_points: Sequence[Tuple[datetime, int]],
        pars: Parameters,
        builder_kwargs_by_m: Mapping[int, Mapping[str, Any]],
        snapshots: Sequence[int],
        scenario: Optional[Scenario] = None,
        index_offspring_bounds: Tuple[int, int] = INDEX_OFFSPRING_BOUNDS,
        num_trajectories: int = NUM_TRAJECTORIES,
        chunk_size: int = CHUNK_SIZE,
        T_run: int = T_RUN,
        max_cases: int = MAX_CASES,
        max_workers: int = MAX_WORKERS,
        T_grid: np.ndarray = TIMEPATH_T_GRID,
        min_required: Optional[int] = MIN_REQUIRED,
) -> List[TimepathSnapshotResult]:
    results: List[TimepathSnapshotResult] = []
    for m in snapshots:
        result = run_snapshot_with_counts(
            m=m,
            obs_points=obs_points,
            pars=pars,
            builder_kwargs=builder_kwargs_by_m.get(m, {}),
            scenario=scenario,
            index_offspring_bounds=index_offspring_bounds,
            num_trajectories=num_trajectories,
            chunk_size=chunk_size,
            T_run=T_run,
            max_cases=max_cases,
            max_workers=max_workers,
            T_grid=T_grid,
            min_required=min_required,
        )
        results.append(result)
    return results


def main(
        *,
        save: bool = SAVE_FIGURES,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        lang: Literal["en", "hu"] = LANGUAGE,
        animate: bool = True,
        show: bool = True,
        min_required: int = MIN_REQUIRED,
        final_horizon_days: float = 50.0,
        quiet_T: float = 10.0,
        step: float = 0.25,
        plot_rb: bool = True,
        plot_timepath: bool = False,
        plot_posterior: bool = False,
) -> None:
    obs_points = sorted(OBS_POINTS, key=lambda item: item[0])
    start_date = obs_points[0][0]
    pars = _build_parameters()
    snapshots = tuple(range(1, len(obs_points) + 1))

    missing = sorted(set(snapshots) - set(SNAPSHOT_BUILDER_KWARGS_BY_M.keys()))
    if missing:
        raise ValueError(f"Missing builder kwargs for snapshots: {missing}")

    results = run_all_snapshots_with_counts(
        obs_points=obs_points,
        pars=pars,
        builder_kwargs_by_m=SNAPSHOT_BUILDER_KWARGS_BY_M,
        snapshots=snapshots,
        scenario=Scenario([]),
        num_trajectories=NUM_TRAJECTORIES,
        chunk_size=CHUNK_SIZE,
        T_run=(obs_points[-1][0] - obs_points[0][0]).days + int(final_horizon_days) + 15,
        max_cases=MAX_CASES,
        max_workers=MAX_WORKERS,
        T_grid=TIMEPATH_T_GRID,
        min_required=min_required,
    )

    # 1. Extinction curve plot (by default only this is plotted, with extended horizon & progressive animation)
    if plot_rb:
        plot_rb_online_right_pane(
            results,
            start_date,
            final_horizon_days=final_horizon_days,
            axis_start_date=obs_points[2][0],
            axis_end_date=None,
            lang=lang,
            animate=animate,
            save=save,
            output_dir=output_dir,
            show=False,
        )

    # 2. Timepath parameter plot from 2026-04-28 to 2026-06-08 (optional)
    if plot_timepath:
        plot_timepath_re_growth_alpha_theta_p0(
            results,
            start_date,
            step=step,
            final_horizon_days=22.0,
            perc_bands=(0.95, 0.8, 0.5),
            axis_start_date=datetime(2026, 4, 28),
            axis_end_date=datetime(2026, 6, 8),
            lang=lang,
            save=save,
            output_dir=output_dir,
            show=False,
        )

    # 3. 2x2 posterior plot after 10 quiet days for the final snapshot (optional)
    if plot_posterior:
        plot_last_posterior_conditioned_quiet_for_T(
            results,
            T=quiet_T,
            save=save,
            output_dir=output_dir,
            show=False,
        )

    if show:
        plt.show()
    else:
        plt.close("all")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 2026 Hanta simulation with the fixed measles extinction algorithm and plot SAE figures."
    )
    parser.add_argument(
        "--lang",
        choices=("en", "hu"),
        default=LANGUAGE,
        help="Language for plot labels and titles (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save plot outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--no-animate",
        action="store_false",
        dest="animate",
        default=True,
        help="Disable progressive animation step outputs.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        dest="animate",
        default=True,
        help="Save progressive animation step PDFs/PNGs for presentation slides (default: True).",
    )
    parser.add_argument(
        "--no-rb",
        action="store_false",
        dest="plot_rb",
        default=True,
        help="Disable plotting the Hungarian online extinction probability curves.",
    )
    parser.add_argument(
        "--plot-timepath",
        action="store_true",
        default=False,
        help="Enable plotting the 2x2 time-dependent parameter path figure (default: False).",
    )
    parser.add_argument(
        "--plot-posterior",
        action="store_true",
        default=False,
        help="Enable plotting the 2x2 conditioned posterior distribution figure (default: False).",
    )
    parser.add_argument(
        "--all-plots",
        action="store_true",
        default=False,
        help="Plot all figures (rb curves, timepath, and conditioned posterior).",
    )
    parser.add_argument(
        "--min-required",
        type=int,
        default=MIN_REQUIRED,
        help="Minimum accepted trajectories per snapshot (default: %(default)s).",
    )
    parser.add_argument(
        "--final-horizon-days",
        type=float,
        default=50.0,
        help="Horizon in days after the final snapshot for rb curves (default: %(default)s).",
    )
    parser.add_argument(
        "--quiet-t",
        type=float,
        default=10.0,
        help="Quiet period duration in days for the conditioned posterior plot (default: %(default)s).",
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save",
        default=SAVE_FIGURES,
        help="Disable saving figures to disk.",
    )
    parser.add_argument(
        "--no-show",
        action="store_false",
        dest="show",
        default=True,
        help="Do not open interactive matplotlib windows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_rb = args.plot_rb or args.all_plots
    plot_timepath = args.plot_timepath or args.all_plots
    plot_posterior = args.plot_posterior or args.all_plots
    main(
        save=args.save,
        output_dir=args.output_dir,
        lang=args.lang,
        animate=args.animate,
        show=args.show,
        min_required=args.min_required,
        final_horizon_days=args.final_horizon_days,
        quiet_T=args.quiet_t,
        plot_rb=plot_rb,
        plot_timepath=plot_timepath,
        plot_posterior=plot_posterior,
    )
