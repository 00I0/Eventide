from typing import Tuple, Dict

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from python.plots.misc import SnapshotResult
from python.plots.style import _use_style


def _support_interval(x: np.ndarray, mass: float = 0.95) -> Tuple[float, float]:
    x = np.asarray(x);
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    xs = np.sort(x);
    n = xs.size
    if n == 1:
        return float(xs[0]), float(xs[0])
    m = max(1, min(int(np.floor(mass * n)), n - 1))
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
    x = np.asarray(x, float);
    x = x[np.isfinite(x)]
    grid = np.asarray(grid, float)
    if x.size == 0:
        return np.zeros_like(grid)
    h = max(1e-12, float(bw))

    X = x[:, None]
    G = grid[None, :]
    K = norm.pdf((G - X) / h) / h

    Ci = norm.cdf((hi - x) / h) - norm.cdf((lo - x) / h)
    Ci = np.maximum(Ci, 1e-15)
    weights = 1.0 / Ci
    f = (K * weights[:, None]).mean(axis=0)

    f[(grid < lo) | (grid > hi)] = 0.0
    mask = (grid >= lo) & (grid <= hi)
    area = np.trapz(f[mask], grid[mask])
    if area > 0:
        f /= area
    return f


def _collect_vars(res: SnapshotResult) -> Dict[str, np.ndarray]:
    arr = np.asarray(res.draws_array)
    R0 = arr[:, 0];
    k = arr[:, 1];
    r = arr[:, 2];
    a = arr[:, 3];
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
    return dict(
        R0=(0.25, 15.0), r=(0.01, 0.99),
        alpha=(0.01, 20.0), theta=(0.01, 20.0),
        k=(0.2, 10.0), Re=(0.0, 3.0),  # R0*r < 3
        alpha_theta=(3.0, 20.0), p0_Re=(0.0, 1.0)
    )


def plot_posterior_grid_single(res: SnapshotResult, *, mass=0.95, bw_adjust=1.0, n_grid=600):
    sty = _use_style(None)
    sv = _collect_vars(res)
    supports = _default_supports()
    var_specs = [
        ("R0", r"$R_0$"), ("r", r"$r$"), ("alpha", r"$\alpha$"), ("theta", r"$\theta$"),
        ("k", r"$k$"), ("Re", r"$rR_0$"), ("alpha_theta", r"$\alpha\theta$"),
        ("p0_Re", r"$\left(\frac{k}{k+rR_0}\right)^k$"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=sty.fig_pair, constrained_layout=True, dpi=sty.dpi)
    axes = axes.reshape(2, 4)
    face = mpl.colors.to_rgba(sty.palette["EMP"], 0.20)
    edge = sty.palette["EMP"]
    accent = sty.palette["ANA"]

    for idx, (key, title) in enumerate(var_specs):
        ax = axes.flat[idx]
        x = sv.get(key, np.array([]))
        lo, hi = supports[key]
        x = x[(x >= lo) & (x <= hi)]
        ax.set_title(title)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        if x.size < 2:
            continue
        h = bw_adjust * _scott_bw(x)
        grid = np.linspace(lo, hi, n_grid)
        pdf = _kde_truncated_gaussian(x, grid, lo, hi, h)
        hdi_lo, hdi_hi = _support_interval(x, mass=mass)
        hdi_lo = max(hdi_lo, lo);
        hdi_hi = min(hdi_hi, hi)
        mask = (grid >= hdi_lo) & (grid <= hdi_hi)
        ax.fill_between(grid[mask], 0.0, pdf[mask], color=face)
        ax.plot(grid, pdf, lw=sty.lw_emp, color=edge)
        med = float(np.median(x))
        med = min(max(med, lo), hi)
        j = int(np.clip(np.searchsorted(grid, med), 0, len(grid) - 1))
        ax.plot([med, med], [0.0, pdf[j]], lw=sty.lw_ana, color=accent)

        ax.minorticks_on()

    plt.show()
