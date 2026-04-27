from typing import Sequence, Tuple, Optional, Dict, Any

import numpy as np
from matplotlib import pyplot as plt

from python.plots.misc import SnapshotResult
from python.plots.style import _use_style, _legend_dedupe


def _cum_matrix_from_times(infection_times_2d: Sequence[Sequence[float]],
                           grid_days: np.ndarray) -> np.ndarray:
    if not infection_times_2d:
        return np.zeros((0, grid_days.size), dtype=float)
    rows = []
    for traj in infection_times_2d:
        t = np.sort(np.asarray(traj, float))
        rows.append(np.searchsorted(t, grid_days, side="right"))
    return np.vstack(rows).astype(float)


def _tail_mean_band(cum_matrix: np.ndarray, p_central: float) -> Tuple[np.ndarray, np.ndarray]:
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

    beta = p_central ** 0.7
    lower = mu + beta * (lower_raw - mu)
    upper = mu + beta * (upper_raw - mu)
    return lower, upper


def plot_cumulative_infections_last_numeric(
        results: Sequence["SnapshotResult"],
        *,
        resolution: float = 0.25,
        perc_bands: Sequence[float] = (0.95, 0.5, 0.2),
        cmap: str | Sequence[str] = "PuBu",
        scale: str = "linear",
        show_mean: bool = True,
        mean_style: Optional[Dict[str, Any]] = None,
        show_median: bool = False,
        median_style: Optional[Dict[str, Any]] = None,
        obs_points_days: Optional[Sequence[Tuple[float, int]]] = None,
):
    sty = _use_style(None)
    if not results:
        return
    res = results[-1]

    grid_days = np.arange(0.0, 55 + 1e-9, float(resolution))
    cum_matrix = _cum_matrix_from_times(res.infection_times_2d, grid_days)

    bands = sorted(perc_bands, reverse=True)
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
        positions = np.linspace(0.25, 0.85, num=len(bands)) if len(bands) > 1 else [0.6]
        band_colors = [cmap_obj(p) for p in positions]
    else:
        import matplotlib.colors as mcolors
        base = [mcolors.to_rgba(c) for c in cmap]
        if len(base) == 0:
            raise ValueError("Custom color list must contain at least one color.")
        band_colors = (base * (len(bands) // len(base) + 1))[:len(bands)]

    fig, ax = plt.subplots(figsize=sty.fig_pair, dpi=sty.dpi, constrained_layout=True)
    if scale.lower() == "log":
        ax.set_yscale("log")

    ax.set_xlabel("Days since start")
    ax.set_ylabel("Cumulative infections")

    computed = []
    for p in bands:
        lo, hi = _tail_mean_band(cum_matrix, p)
        computed.append((lo, hi, p))
    for (lo, hi, p), color in zip(computed, band_colors):
        if lo.size:
            ax.fill_between(grid_days, lo, hi, color=color, alpha=0.30, linewidth=0,
                            label=f"{int(round(p * 100))}% band")

    if show_mean and cum_matrix.size:
        mean_style = mean_style or {"color": "black", "lw": sty.lw_rb}
        ax.plot(grid_days, cum_matrix.mean(axis=0), label="mean", **mean_style)

    if show_median and cum_matrix.size:
        median_style = median_style or {"color": "gray", "lw": sty.lw_emp, "ls": "--"}
        ax.plot(grid_days, np.median(cum_matrix, axis=0), label="median", **median_style)

    if obs_points_days:
        obs_sorted = sorted(obs_points_days, key=lambda x: float(x[0]))
        xs, ys_inc = zip(*obs_sorted)
        ys_cum = np.cumsum(np.asarray(ys_inc, dtype=float))
        ax.scatter(xs, ys_cum, marker="o", s=sty.dot_area, linewidths=0.9, zorder=10,
                   edgecolors="white", color=sty.palette["POINTS"], label="Observed")

    _legend_dedupe(ax)
    ax.minorticks_on()
    ax.set_xlim(0.0, 55 if grid_days.size else 1.0)
    plt.show()
