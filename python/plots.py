from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import ceil
from typing import Optional, Sequence, Tuple, Literal, Any, List, Dict, overload, Union

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import gamma

from python.eventide import Hist1D, Hist2D, InfectionTimeCollector, DrawCollector, ActiveSetSizeCollector, TimeMatrix


def _empirical_stats(
        counts: np.ndarray,
        centers: np.ndarray,
        conf_level: float,
        method: Literal["symmetric", "shortest"]
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute mean, median, and a confidence interval from binned 1D histogram data.

    Args:
        counts: Bin counts.
        centers: Bin center values (same length as counts).
        conf_level: Desired confidence level in (0, 1).
        method: CI calculation method:
            * "symmetric": equal-tailed interval based on quantiles.
            * "shortest": the shortest interval containing the specified mass.

    Returns:
        Tuple (mean, median, (lo_ci, hi_ci)).
        Values are NaN if counts are empty.
    """
    sample = np.repeat(centers, counts.astype(int))
    if sample.size == 0:
        return float('nan'), float('nan'), (float('nan'), float('nan'))

    mean = float(sample.mean())
    median = float(np.median(sample))
    lo, hi = float('nan'), float('nan')

    if method == 'symmetric':
        a = 1 - conf_level
        lo, hi = np.quantile(sample, [a / 2, 1 - a / 2])
        lo, hi = float(lo), float(hi)

    if method == 'shortest':
        # noinspection PyPep8Naming
        N = sample.size
        k = int(np.floor(conf_level * N))
        s = np.sort(sample)
        widths = s[k:] - s[: N - k]
        i = np.argmin(widths)
        lo, hi = float(s[i]), float(s[i + k])

    return mean, median, (lo, hi)


# noinspection PyPep8Naming
def _compute_R1(
        data: np.ndarray,
        dist: stats.rv_continuous,
        params: Tuple[Any, ...],
        bins: int
) -> float:
    """
    Compute R1 (variance explained) between a histogram and a fitted PDF.

    The histogram is computed with bins and normalized to a density.

    Args:
        data: Sample values.
        dist: A SciPy continuous distribution object.
        params: Distribution parameters as accepted by dist.pdf.
        bins: Number of bins for the histogram.

    Returns:
        R1 score in [0, 1] or NaN if TSS is zero.
    """
    counts, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf_vals = dist.pdf(centers, *params)
    sse = np.sum((counts - pdf_vals) ** 2)
    tss = np.sum((counts - counts.mean()) ** 2)
    return 1 - (sse / tss) if tss > 0 else np.nan


class _Hist1DLayer:
    """
    Internal: Render a 1D histogram layer with optional stats overlays.

    Supports drawing mean, median, and confidence intervals on top of the bars.
    This is an implementation detail of: class: HistSpec.
    """
    __slots__ = (
        'collector', 'label',
        'show_mean', 'show_median', 'show_conf',
        'conf_level', 'conf_method',
        'bar_color', 'edge_color', 'bar_alpha'
    )

    def __init__(self,
                 collector: Hist1D,
                 label: str,
                 show_mean: bool,
                 show_median: bool,
                 show_conf: bool,
                 conf_level: float,
                 conf_method: Literal["symmetric", "shortest"],
                 bar_color: Optional[str],
                 edge_color: Optional[str],
                 bar_alpha: float
                 ):
        self.collector = collector
        self.label = label
        self.show_mean = show_mean
        self.show_median = show_median
        self.show_conf = show_conf
        self.conf_level = conf_level
        self.conf_method = conf_method
        self.bar_color = bar_color
        self.edge_color = edge_color
        self.bar_alpha = bar_alpha

    def plot(self, ax: plt.Axes) -> None:
        counts = np.asarray(self.collector)
        bins = counts.size
        lo, hi = self.collector.range
        edges = np.linspace(lo, hi, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = (hi - lo) / bins

        total = np.sum(counts)
        if total > 0:
            density = counts / (total * width)
        else:
            density = np.zeros_like(counts, dtype=float)

        ax.bar(
            centers, density,
            width=width, align='center',
            color=self.bar_color,
            edgecolor=self.edge_color,
            alpha=self.bar_alpha,
            label=self.label,
        )

        ax.set_xlim(lo, hi)
        ax.set_xlabel(self.collector.name)
        ax.grid(False)

        if not (self.show_mean or self.show_median or self.show_conf):
            return

        y_text = ax.get_ylim()[1] * 0.95
        mean, median, (lo_ci, hi_ci) = _empirical_stats(counts, centers, self.conf_level, self.conf_method)
        if self.show_conf:
            ax.axvspan(lo_ci, hi_ci, color='gray', alpha=0.3)
            ax.text(
                lo_ci, y_text,
                r'$\text{C.I.}_\text{low}$: ' + f'{lo_ci:.2f}',
                ha='center', va='top', fontsize='x-small',
                rotation=90,
                bbox=dict(
                    boxstyle='round, pad=0.35',
                    facecolor=(1, 1, 1, 0.6),
                    edgecolor='none'
                )
            )
            ax.text(
                hi_ci, y_text,
                r'$\text{C.I.}_\text{high}$: ' + f'{hi_ci:.2f}',
                ha='center', va='top', fontsize='x-small',
                rotation=90,
                bbox=dict(
                    boxstyle='round, pad=0.35',
                    facecolor=(1, 1, 1, 0.6),
                    edgecolor='none'
                )
            )
        if self.show_mean:
            ax.axvline(mean, color='C1', ls='--', lw=1.5)
            ax.text(
                mean, y_text,
                f'mean: {mean:.2f}',
                ha='center', va='top', color='C1', fontsize='x-small', rotation=90,
                bbox=dict(
                    boxstyle='round, pad=0.35',
                    facecolor=(1, 1, 1, 0.6),
                    edgecolor='none'
                )
            )
        if self.show_median:
            ax.axvline(median, color='C2', ls='-.', lw=1.5)
            ax.text(
                median, y_text,
                f'med: {median:.2f}',
                ha='center', va='top', color='C2', fontsize='x-small', rotation=90,
                bbox=dict(
                    boxstyle='round, pad=0.35',
                    facecolor=(1, 1, 1, 0.6),
                    edgecolor='none'
                )
            )


class _Hist2DLayer:
    """
    Internal: Render a 2D histogram layer as an image.

    This is an implementation detail of: class:HistSpec.
    """
    __slots__ = ('collector', 'label', 'cmap')

    def __init__(self, collector: Hist2D, label: str, cmap: str):
        self.collector = collector
        self.label = label
        self.cmap = cmap

    def plot(self, ax: plt.Axes) -> None:
        data = np.asarray(self.collector)
        (lox, hix), (loy, hiy) = self.collector.range
        ax.imshow(
            data.T,
            origin='lower',
            aspect='auto',
            extent=(lox, hix, loy, hiy),
            cmap=self.cmap
        )
        ax.set_xlabel(self.collector.var_names[0])
        ax.set_ylabel(self.collector.var_names[1])
        ax.set_title(self.label)
        ax.grid(False)


class _PdfLayer:
    """
    Internal: Render a fitted probability density function (PDF).

    Optionally overlays the empirical histogram used for fitting and can annotate the fit with an R1 score.
    This is an implementation detail of: class:HistSpec.
    """
    __slots__ = (
        'data', 'dist_name', 'bins', 'show_hist',
        'fit_method', 'show_R1', 'label',
        'line_color', 'line_alpha', 'hist_color', 'hist_alpha'
    )

    # noinspection PyPep8Naming
    def __init__(self,
                 data: np.ndarray,
                 dist_name: str,
                 bins: int,
                 show_hist: bool,
                 fit_method: Literal["mle", "ls"],
                 show_R1: bool,
                 label: str,
                 line_color: Optional[str],
                 line_alpha: float,
                 hist_color: Optional[str],
                 hist_alpha: float
                 ):
        self.data = data
        self.dist_name = dist_name
        self.bins = bins
        self.show_hist = show_hist
        self.fit_method = fit_method
        self.show_R1 = show_R1
        self.label = label
        self.line_color = line_color
        self.line_alpha = line_alpha
        self.hist_color = hist_color
        self.hist_alpha = hist_alpha

    def plot(self, ax: plt.Axes) -> None:
        arr = self.data
        lo, hi = arr.min(), arr.max()
        if self.show_hist:
            ax.hist(
                arr, bins=self.bins, density=True,
                color=self.hist_color, alpha=self.hist_alpha,
                label='Empirical'
            )

        ax.set_xlim(lo, hi)
        ax.grid(False)

        dist = getattr(stats, self.dist_name, None)
        if not dist:
            warnings.warn(f"No such dist '{self.dist_name}'")
            return

        label = self.label
        params = dist.fit(arr)
        if self.fit_method == 'ls':
            counts, edges = np.histogram(arr, bins=self.bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            try:
                params = curve_fit(lambda x, *p: dist.pdf(x, *p), centers, counts, p0=params)[0]
            except (RuntimeError, ValueError) as e:
                warnings.warn(f"LS fit failed for '{self.dist_name}': {e}")
        x = np.linspace(lo, hi, 400)

        if self.show_R1:
            label = f'{self.label} (R1: {100 * _compute_R1(arr, dist, params, self.bins):.1f}%)'

        ax.plot(x, dist.pdf(x, *params), color=self.line_color, alpha=self.line_alpha, lw=2, label=label)


@dataclass
class HistSpec:
    """
    Configuration for a histogram or PDF subplot.

    Wraps either a histogram collector (Hist1D/Hist2D) or raw data with a
    distribution name for PDF fitting, along with styling and overlay options.

    One of:
      * collector = Hist1D or Hist2D
      * data + dist_name for a fitted PDF

    1D-specific:
      * show_mean, show_median, show_conf, conf_level, conf_method

    PDF-specific:
      * bins, show_hist, fit_method, show_R1

    Args:
        collector: Hist1D or Hist2D instance (mutually exclusive with data/dist_name).
        data: Raw sample values for PDF mode.
        dist_name: Name of SciPy distribution for PDF mode.
        label: Title/legend label override.
        overlays: Additional HistSpecs to render on the same Axes (1D only).
    """

    # data sources
    collector: Hist1D | Hist2D | None = None
    data: Sequence[float] = None
    dist_name: str = None

    # common
    label: Optional[str] = None

    # 1D toggles
    show_mean: bool = False
    show_median: bool = False
    show_conf: bool = False
    conf_level: float = 0.95
    conf_method: Literal['symmetric', 'shortest'] = 'symmetric'

    # PDF toggles
    bins: int = 50
    show_hist: bool = True
    fit_method: Literal['mle', 'ls'] = 'mle'
    show_R1: bool = False

    # graphics defaults
    bar_color: Optional[str] = 'C0'
    edge_color: Optional[str] = None
    bar_alpha: float = 0.6

    cmap: str = 'viridis'

    hist_color: Optional[str] = 'C0'
    hist_alpha: float = 0.4
    line_color: Optional[str] = 'C1'
    line_alpha: float = 0.8

    overlays: List[HistSpec] = field(default_factory=list, init=False)

    def __post_init__(self):
        # validate exactly one of collector vs. data+dist_name
        has_col = self.collector is not None
        has_pdf = (self.data is not None or self.dist_name is not None)
        if not (has_col ^ has_pdf):
            raise ValueError('Must supply exactly one of collector or (data+dist_name)')

        # 2D restrictions
        if isinstance(self.collector, Hist2D):
            if any([self.show_mean, self.show_median, self.show_conf, self.data, self.dist_name]):
                raise ValueError('2D spec cannot have 1D toggles, PDF mode or overlays')

        if has_pdf:
            if any([self.show_mean, self.show_median, self.show_conf]):
                raise ValueError('PDF spec cannot have 1D toggles')

    # noinspection PyUnreachableCode
    def _make_layer(self, spec: HistSpec) -> _Hist1DLayer | _Hist2DLayer | _PdfLayer:
        """Construct the correct private layer for one spec."""
        lab = spec.label or (spec.collector.name if spec.collector else spec.dist_name)

        # 1D histogram
        if isinstance(spec.collector, Hist1D):
            return _Hist1DLayer(
                collector=spec.collector,
                label=lab,
                show_mean=spec.show_mean,
                show_median=spec.show_median,
                show_conf=spec.show_conf,
                conf_level=spec.conf_level,
                conf_method=spec.conf_method,
                bar_color=spec.bar_color,
                edge_color=spec.edge_color,
                bar_alpha=spec.bar_alpha
            )

        # 2D histogram
        if isinstance(spec.collector, Hist2D):
            return _Hist2DLayer(
                collector=spec.collector,
                label=lab,
                cmap=spec.cmap
            )

        # PDF layer
        if spec.data is not None:
            pdf_data = np.asarray(spec.data)
            show_hist = spec.show_hist
        elif isinstance(self.collector, Hist1D):
            counts = np.asarray(self.collector)  # binned counts
            lo, hi = self.collector.range
            bins = counts.size
            edges = np.linspace(lo, hi, bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            pdf_data = np.repeat(centers, counts.astype(int))
            show_hist = False
        else:
            raise ValueError('PDF spec requires data= or a Hist1D collector to inherit from')

        return _PdfLayer(
            data=pdf_data,
            dist_name=spec.dist_name,
            bins=spec.bins,
            show_hist=show_hist,
            fit_method=spec.fit_method,
            show_R1=spec.show_R1,
            label=lab,
            line_color=spec.line_color,
            line_alpha=spec.line_alpha,
            hist_color=spec.hist_color,
            hist_alpha=spec.hist_alpha
        )

    def overlay(self, *others: HistSpec) -> HistSpec:
        """
        Add additional layers on the same axes.

        Any PDF overlay with no explicit data will inherit the base collector’s samples automatically.
        """
        if isinstance(self.collector, Hist2D):
            raise ValueError('Cannot overlay onto a 2D spec')

        for ov in others:
            if isinstance(ov.collector, Hist2D):
                raise ValueError('Cannot overlay a 2D spec')
            self.overlays.append(ov)

        return self

    def plot(self, ax: plt.Axes) -> None:
        """Draw the base spec plus any overlays."""
        specs = [self] + self.overlays
        for spec in specs:
            layer = self._make_layer(spec)
            layer.plot(ax)

        if len(specs) > 1:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize='small', loc='upper right')

        elif ax.get_legend():
            ax.get_legend().remove()


class HistogramGrid:
    """
    Arrange and plot a grid of histogram/PDF subplots.

    Args:
        collectors: Sequence of HistSpec, Hist1D, or Hist2D instances.
        target_ratio: Desired figure width/height ratio for layout optimization.
        subplot_size: Side length in inches for each subplot.
        dpi: Figure resolution.
    """

    def __init__(
            self,
            collectors: Sequence[HistSpec | Hist1D | Hist2D],
            target_ratio: float = 16 / 9,
            subplot_size: float = 4.0,
            dpi: int = 200
    ):
        wrapped: list[HistSpec] = []
        for s in collectors:
            if isinstance(s, HistSpec):
                wrapped.append(s)
            elif isinstance(s, Hist1D):
                wrapped.append(HistSpec(collector=s))
            elif isinstance(s, Hist2D):
                wrapped.append(HistSpec(collector=s))

        self.specs = wrapped
        self.target_ratio = target_ratio
        self.subplot_size = subplot_size
        self.dpi = dpi

    def plot(self) -> plt.Figure:
        n = len(self.specs)
        rows, cols = min(
            ((ceil(n / c), c) for c in range(1, n + 1)),
            key=lambda x: abs(x[1] / x[0] - self.target_ratio)
        )
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(self.subplot_size * cols, self.subplot_size * rows),
            dpi=self.dpi
        )
        axes_flat = np.atleast_1d(axes).flatten()
        for ax, spec in zip(axes_flat, self.specs):
            spec.plot(ax)
        for ax in axes_flat[len(self.specs):]:
            fig.delaxes(ax)
        fig.tight_layout()
        return fig


def plot_histogram_grid(
        collectors: Sequence[Hist1D | Hist2D],
        target_ratio: float = 16 / 9,
        subplot_size: float = 5.0,
        dpi: int = 100
) -> Figure:
    """
    Convenience wrapper to create and display a HistogramGrid.

    Args:
        collectors: Sequence of Hist1D or Hist2D instances.
        target_ratio: Desired figure width/height ratio.
        subplot_size: Side length in inches for each subplot.
        dpi: Figure resolution.

    Returns:
        The Matplotlib Figure object.
    """
    fig = HistogramGrid(collectors, target_ratio, subplot_size, dpi).plot()
    plt.show()
    return fig


# noinspection PyPep8Naming
class ExtinctionProbabilityPlot:
    # noinspection PyUnresolvedReferences
    """
    Plot empirical and/or analytical extinction probabilities on one axis.

    The plot can show:
      * Analytical curves computed from accepted parameter draws and active-set sizes.
      * Empirical curves computed from a TimeMatrix of simulation outcomes.
      * Or both on the same axes.

    Example:
        >>> ep = ExtinctionProbabilityPlot(start_date, T_run, cutoff_date)
        >>> ep.add_analytical(draws, active_set_sizes, conf=(0.3, 0.6, 0.9))
        >>> ep.add_empirical(matrix)
        >>> fig = ep.plot(); fig.show()

    Args:
        start_date: Date corresponding to day 0 on the x-axis.
        cutoff_date: Start date for the plotting window (inclusive).
            If provided, both analytical and empirical series start at this date.
            Must satisfy 0 ≤ (cutoff_date - start_date).days ≤ T_run.
        T_run: Time horizon in days to plot (positive integer).
        dpi: Figure resolution in dots per inch.
        figsize: Figure size (width, height) in inches.
    """

    def __init__(
            self,
            start_date: datetime,
            cutoff_date: datetime,
            T_run: int,
            dpi: int = 100,
            figsize: Tuple[float, float] = (10, 5),
    ):
        self.start_date = start_date
        self.T_run = int(T_run)

        if cutoff_date is None:
            self.cutoff_day = 0
        else:
            cd = (cutoff_date - start_date).days
            if cd < 0 or cd > self.T_run:
                raise ValueError('cutoff_date must satisfy start_date ≤ cutoff_date ≤ start_date + T_run.')
            self.cutoff_day = int(cd)

        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.grid(True, linestyle=':')
        self.ax.set_ylim(0, 1.02)
        self.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Probability of Extinction')

    def add_analytical(
            self,
            draws: np.ndarray,
            active_set_sizes: np.ndarray,
            conf: Tuple[float, ...] = (0.3, 0.6, 0.9),
            colors: Optional[Sequence[str] | str] = None
    ) -> None:
        """
        Add analytical extinction probability: median + credible bands.

        Args:
            draws: Accepted draws, shape (N, 5) as [R0, k, r, alpha, theta].
            active_set_sizes: Shape (N,), aligned with draws.
            conf: Credible levels, e.g. (0.3, 0.6, 0.9). Values in (0,1).
            colors: Either a Matplotlib colormap name (str), or a list of colors:
                first color is used for the median line; subsequent colors are used for bands from widest → narrowest.
                If omitted, falls back to the Matplotlib color cycle.

        Raises:
            ValueError: If input shapes are inconsistent.
        """

        if draws.ndim != 2 or draws.shape[1] != 5:
            raise ValueError(f"'draws' must have shape (N, 5); got {draws.shape}")
        if active_set_sizes.ndim != 1 or active_set_sizes.shape[0] != draws.shape[0]:
            raise ValueError("'active_set_sizes' must be 1D with same length as draws.")

        R0s, ks, rs, alphas, thetas = draws.T
        Reffs = R0s * rs

        L = self.T_run - self.cutoff_day + 1
        quiet_windows_T = np.arange(1, L + 1)
        dates = [self.start_date + timedelta(days=int(d)) for d in range(self.cutoff_day, self.cutoff_day + L)]
        date_nums = mdates.date2num(dates)

        # extinction distributions over time
        probs = []
        for T in quiet_windows_T:
            F = gamma.cdf(T, a=alphas, scale=thetas)
            term1 = 1 + Reffs / ks
            term2 = 1 + (Reffs * F) / ks
            p = np.ones_like(term1)
            mask = term2 > 0
            p[mask] = (term1[mask] / term2[mask]) ** (-ks[mask])
            probs.append(np.clip(p ** active_set_sizes, 0, 1))
        M = np.vstack(probs)  # shape (T_run, N)

        # median + bands
        levels = tuple(sorted(conf))
        median = np.median(M, axis=1)
        band_bounds: list[tuple[float, np.ndarray, np.ndarray]] = []
        for lvl in sorted(conf):
            lower = np.percentile(M, (1 - lvl) * 100 / 2, axis=1)
            upper = np.percentile(M, 100 - (1 - lvl) * 100 / 2, axis=1)
            band_bounds.append((lvl, lower, upper))

        # --- colors ---
        if colors is None:
            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            median_color = cycle[0]
            band_palette = cycle[1:] or [cycle[0]]
            band_colors = [band_palette[i % len(band_palette)] for i in range(len(levels))]
        elif isinstance(colors, str):
            cmap_obj = plt.get_cmap(colors)
            n = 1 + len(levels)
            pos = np.linspace(0.15, 0.85, num=n)
            rgba = [cmap_obj(p) for p in pos]
            median_color = rgba[0]
            band_colors = rgba[1:]
        else:
            if len(colors) < 1:
                raise ValueError("colors list must have at least one color (for the median).")
            clist = [mcolors.to_rgba(c) for c in colors]
            median_color = clist[0]
            palette = clist[1:] or [clist[0]]
            band_colors = [palette[i % len(palette)] for i in range(len(levels))]

        for (lvl, lo, hi), c in zip(reversed(band_bounds), reversed(band_colors)):
            self.ax.fill_between(date_nums, lo, hi, alpha=0.25, color=c, label=f'{int(lvl * 100)}% CI')

        self.ax.plot(date_nums, median, lw=2, label='Analytical', color=median_color)

    def add_empirical(
            self,
            matrix: np.ndarray,
            color: str = 'C3',
            marker: str = '.'
    ) -> None:
        """
        Add empirical extinction probability from a TimeMatrix.

        For each day d, computes the fraction of trajectories that do not resurge after d:
        ext(d) = 1 - resurge/valid, where 'resurge' counts outcomes that restart after d and 'valid' counts all
        outcomes that could have resurged.

        Args:
            matrix: Square array with shape (T+2, T+2) produced by the TimeMatrix accumulator.
            color: Line/marker color.
            marker: Marker style.
        """

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 3:
            raise ValueError("'matrix' must be a square array with shape (T+2, T+2), T>=1.")

        T = matrix.shape[0] - 2
        d0 = self.cutoff_day
        d1 = min(self.T_run, T)
        if d0 > d1:
            return

        days = np.arange(d0, d1 + 1)
        dates = [self.start_date + timedelta(days=int(d)) for d in days]
        date_nums = mdates.date2num(dates)

        comp = np.empty_like(days, dtype=float)
        for i, d in enumerate(days):
            mask = np.arange(matrix.shape[1]) > d
            resurge = matrix[d + 1:, mask].sum()
            valid = matrix[:, mask].sum()
            comp[i] = 1 - (resurge / valid if valid > 0 else 0.0)

        self.ax.plot(date_nums, comp, lw=1.5, marker=marker, label='Empirical', color=color)

    def plot(self) -> Figure:
        """Finalize and display the plot."""
        self.ax.legend()
        self.fig.autofmt_xdate(rotation=30)
        self.fig.tight_layout()
        return self.fig


# noinspection PyPep8Naming
@overload
def plot_extinction(
        start_date: datetime,
        cutoff_date: datetime,
        T_run: int,
        *,
        analytical: Tuple[np.ndarray | DrawCollector, np.ndarray | ActiveSetSizeCollector],
        empirical: None = ...,
        analytical_colors: Optional[str | Sequence[str]] = ...,
        empirical_color: Optional[str] = ...,
        dpi: int = 100,
        figsize: Tuple[float, float] = (10, 5),
        conf: Tuple[float, ...] = (0.3, 0.6, 0.9),
) -> plt.Figure: ...


# noinspection PyPep8Naming
@overload
def plot_extinction(
        start_date: datetime,
        cutoff_date: datetime,
        T_run: int,
        *,
        analytical: None = ...,
        empirical: Union[np.ndarray, TimeMatrix],
        analytical_colors: Optional[str | Sequence[str]] = ...,
        empirical_color: Optional[str] = ...,
        dpi: int = 100,
        figsize: Tuple[float, float] = (10, 5),
        conf: Tuple[float, ...] = (0.3, 0.6, 0.9),
) -> plt.Figure: ...


# noinspection PyPep8Naming
@overload
def plot_extinction(
        start_date: datetime,
        cutoff_date: datetime,
        T_run: int,
        *,
        analytical: Tuple[np.ndarray | DrawCollector, np.ndarray | ActiveSetSizeCollector],
        empirical: Union[np.ndarray, TimeMatrix],
        analytical_colors: Optional[str | Sequence[str]] = ...,
        empirical_color: Optional[str] = ...,
        dpi: int = 100,
        figsize: Tuple[float, float] = (10, 5),
        conf: Tuple[float, ...] = (0.3, 0.6, 0.9),
) -> plt.Figure: ...


# noinspection PyPep8Naming
def plot_extinction(
        start_date: datetime,
        cutoff_date: datetime,
        T_run: int,
        *,
        analytical: Optional[Tuple[np.ndarray | DrawCollector, np.ndarray | ActiveSetSizeCollector]] = None,
        empirical: Optional[np.ndarray | TimeMatrix] = None,
        analytical_colors: Optional[str | Sequence[str]] = None,
        empirical_color: Optional[str] = None,
        dpi: int = 100,
        figsize: Tuple[float, float] = (10, 5),
        conf: Tuple[float, ...] = (0.3, 0.6, 0.9),
) -> plt.Figure:
    """Convenience: plot extinction probabilities (analytical, empirical, or both).

    Color API mirrors other plots:
      • analytical_colors: colormap name (str) or list of colors (first=median, rest=bands).
      • empirical_color: single color string for the empirical curve.

    Args:
        start_date: Day 0 on x-axis.
        cutoff_date: Day after which to start plotting.
        T_run: Horizon (days).
        analytical: Tuple (draws, active_set_sizes).
        empirical: TimeMatrix with shape (T+2, T+2).
        analytical_colors: Colormap name or list for analytical median+bands.
        empirical_color: Single color for empirical curve.
        dpi: Figure DPI.
        figsize: Figure size (w, h) in inches.
        conf: Credible levels for analytical bands.

    Returns:
        The Matplotlib Figure.

    Raises:
        ValueError: If neither analytical nor empirical data is provided
    """
    if analytical is None and empirical is None:
        raise ValueError("Provide at least one of 'analytical=(draws, active_set_sizes)' or 'empirical=matrix'.")

    ep = ExtinctionProbabilityPlot(start_date, cutoff_date, T_run, dpi=dpi, figsize=figsize)

    if analytical is not None:
        draws, active_set = analytical
        draws = np.asarray(draws)
        active_set = np.asarray(active_set)
        ep.add_analytical(draws, active_set, conf=conf, colors=analytical_colors)

    if empirical is not None:
        empirical = np.asarray(empirical)
        ep.add_empirical(empirical, color=(empirical_color or 'C3'))

    fig = ep.plot()
    plt.show()
    return fig


class CumulativeInfectionsPlot:
    # noinspection GrazieInspection, PyUnresolvedReferences
    """
        Plot cumulative infections over time with percentile bands and optional overlays.

        This class visualizes the distribution of cumulative infections from multiple simulated or observed trajectories.
        Percentile bands are computed as **tail-mean envelopes** (mean of the lowest/highest fraction of curves per
        timepoint), with shrinkage toward the mean for smaller central fractions.

        Supports:
          * Central-mass style band specification (e.g. 0.95 → 95% around the mean).
          * Either a Matplotlib colormap name or a list of explicit colors.
          * Mean and/or median overlay curves.
          * Observed cumulative points.
          * Custom figure size, DPI, and styles.

        Example:
            >>> plot = CumulativeInfectionsPlot(
            ...     infection_times,
            ...     start_date=datetime(2025, 3, 3),
            ...     end_date=datetime(2025, 5, 1),
            ...     perc_bands=(0.95, 0.5, 0.2),
            ...     cmap='OrRd',
            ...     scale='linear',
            ...     show_mean=True,
            ...     mean_style={'color': 'navy', 'lw': 2},
            ...     obs_points=[(datetime(2025, 3, 10), 5)]
            ... )
            >>> fig = plot.plot()

        Args:
            infection_time_matrix: Each element is a sequence of infection times (in days from simulation start)
                for one trajectory.
            start_date: Start date of the simulation/plot.
            end_date: End date of the simulation/plot.
            resolution: Grid step in days for cumulative counts.
            perc_bands: Central mass fractions for bands, e.g. (0.95, 0.5) for 95%, and 50%.
                Values must be in (0, 1].
            cmap: Colormap name (str) or list of explicit colors (widest band first).
            scale: Y-axis scale, either "linear" or "log".
            show_mean: If True, plot the mean curve.
            mean_style: Matplotlib style dict for the mean curve.
            show_median: If True, plot the median curve.
            median_style: Matplotlib style dict for the median curve.
            obs_points: Optional observed (date, count) pairs. Counts are assumed to be incident (not cumulative)
                and are cumulated for plotting.
            obs_style: Matplotlib style dict for observed points.
            dpi: Figure resolution in dots per inch.
            figsize: Figure size as (width, height) in inches.
            eps: Small number for numerical stability when dividing.

        Raises:
            ValueError: If style arguments are provided without enabling their curve, or if perc_bands values are out
            of (0, 1], or if a custom color list is  shorter than the number of bands.
            TypeError: If cmap is neither a str nor a sequence of colors.
        """

    def __init__(
            self,
            infection_time_matrix: Sequence[Sequence[float]],
            start_date: datetime,
            end_date: datetime,
            *,
            resolution: float = 0.1,
            perc_bands: Sequence[float] = (0.05, 0.20, 0.35),
            cmap: str | Sequence[str] = 'Blues',
            scale: Literal['linear', 'log'] = 'log',
            show_mean=False,
            mean_style: Optional[Dict[str, Any]] = None,
            show_median: bool = False,
            median_style: Optional[Dict[str, Any]] = None,
            obs_points: Optional[Sequence[Tuple[datetime, int]]] = None,
            obs_style: Optional[Dict[str, Any]] = None,
            dpi: int = 200,
            figsize: Tuple[float, float] = (10, 6),
            eps: float = 1e-9,  # numerical stability for sigma
    ):
        if mean_style and not show_mean:
            raise ValueError('mean_style specified but show_mean=False')
        if median_style and not show_median:
            raise ValueError('median_style specified but show_median=False')

        self.infection_time_matrix = [np.sort(np.asarray(t)) for t in infection_time_matrix]
        self.start_date = start_date
        self.end_date = end_date
        self.resolution = resolution
        self.perc_bands = tuple(sorted(perc_bands, reverse=True))
        self.cmap = cmap
        self.scale = scale
        self.show_mean = show_mean
        self.mean_style = mean_style or {'color': 'black', 'lw': 1.5}
        self.show_median = show_median
        self.median_style = median_style or {'color': 'gray', 'ls': '--', 'lw': 2}
        self.obs_points = sorted(obs_points, key=lambda x: x[0]) if obs_points else []
        self.obs_style = obs_style or {
            'marker': 'o', 'color': 'red', 'edgecolor': 'white', 's': 36, 'linewidths': 1, 'label': 'Observed'
        }
        self.dpi = dpi
        self.figsize = figsize
        self.eps = eps

        # precompute grid & cum matrix
        total_days = (end_date - start_date).total_seconds() / 86400.0
        self.grid = np.arange(0.0, total_days + resolution, resolution)

        cum = []
        for t in self.infection_time_matrix:
            cum.append(np.searchsorted(t, self.grid, side='right'))
        self.cum_matrix = np.vstack(cum)  # shape = (N, T)
        self.N, self.T = self.cum_matrix.shape
        self.sorted_matrix = np.sort(self.cum_matrix, axis=0)

    def _tail_mean_shrunk_band(self, p_central: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute a tail-mean envelope for a given central mass.

        The lower bound is the mean of the lowest k values per timepoint,
        the upper bound is the mean of the highest k values, where
        k = ceil(t * N) and t = (1 - p_central) / 2 is the per-side tail fraction.

        Bounds are then shrunk toward the mean curve so that as p_central → 0 they collapse to the mean.

        Args:
            p_central: Central mass fraction (0 < p ≤ 1).

        Returns:
            Tuple (lower, upper, t) where:
              * lower: Lower bound array (time dimension).
              * upper: Upper bound array (time dimension).
              * t: Per-side tail fraction actually used.
        """
        if not (0 < p_central <= 1.0):
            raise ValueError(f"perc_bands values must be in (0, 1], got {p_central}")

        # per-side tail fraction & tail size
        t = (1.0 - p_central) / 2.0
        mu = self.cum_matrix.mean(axis=0)  # overall mean curve

        if t <= 0:
            return mu.copy(), mu.copy(), 0.0

        k = max(1, int(np.ceil(self.N * t)))

        lower_raw = self.sorted_matrix[:k, :].mean(axis=0)
        upper_raw = self.sorted_matrix[-k:, :].mean(axis=0)

        # Shrink toward mean as p ↓.
        beta = p_central ** 0.7
        lower = mu + beta * (lower_raw - mu)
        upper = mu + beta * (upper_raw - mu)

        return lower, upper, t

    def plot(self) -> Figure:
        """
        Generate the cumulative infections plot.

        Returns:
            Matplotlib Figure object containing the plot.
        """
        # dates
        dates = [self.start_date + timedelta(days=float(d)) for d in self.grid]
        date_nums = mdates.date2num(dates)

        # fig/ax
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_yscale(self.scale)
        ax.set_xlim(mdates.date2num(self.start_date), mdates.date2num(self.end_date))
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative infections')
        ax.grid(True, ls='--', alpha=0.4)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis_date()
        fig.autofmt_xdate(rotation=30)

        n_bands = len(self.perc_bands)
        if isinstance(self.cmap, str):
            cmap_obj = plt.get_cmap(self.cmap)
            positions = np.linspace(0.15, 0.85, num=n_bands)
            color_list = [cmap_obj(pos) for pos in positions]
            get_color = lambda i: color_list[i]
        elif isinstance(self.cmap, Sequence):
            base_colors = [mcolors.to_rgba(c) for c in self.cmap]
            if len(base_colors) == 0:
                raise ValueError("Custom color list must contain at least one color.")
            color_list = [base_colors[i % len(base_colors)] for i in range(n_bands)]
            get_color = lambda i: color_list[i]

        # bands
        fractions = self.perc_bands
        bands = []
        for i, p in enumerate(fractions):
            lower, upper, _ = self._tail_mean_shrunk_band(p)
            color = get_color(i)
            bands.append((lower, upper, color, p))

        for lower, upper, color, p in reversed(bands):
            ax.fill_between(
                date_nums, lower, upper,
                color=color, alpha=0.35,
                label=f'{int(round(p * 100))}% band'
            )

        # overlays
        if self.show_mean:
            mean_curve = np.mean(self.cum_matrix, axis=0)
            ax.plot(date_nums, mean_curve, label='mean', **self.mean_style)
        if self.show_median:
            median_curve = np.median(self.cum_matrix, axis=0)
            ax.plot(date_nums, median_curve, label='median', **self.median_style)

        # observed points (cum)
        if self.obs_points:
            xs, ys = zip(*self.obs_points)
            ys_cum = np.cumsum(ys)
            ax.scatter(xs, ys_cum, **self.obs_style)

        # legend
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 1:
            ax.legend(fontsize='small', frameon=False)

        plt.tight_layout()
        return fig


def plot_cumulative_infections(
        infection_times: Sequence[Sequence[float]] | InfectionTimeCollector,
        start_date: datetime,
        end_date: datetime,
        *,
        resolution: float = 0.1,
        perc_bands: Sequence[float] = (0.05, 0.20, 0.35),
        cmap: str | Sequence[str] = 'PuBu',
        scale: Literal['linear', 'log'] = 'linear',
        show_mean: bool = False,
        mean_style: Optional[Dict[str, Any]] = None,
        show_median: bool = False,
        median_style: Optional[Dict[str, Any]] = None,
        obs_points: Optional[Sequence[Tuple[datetime, int]]] = None,
        obs_style: Optional[Dict[str, Any]] = None,
        dpi: int = 200,
        figsize: Tuple[float, float] = (10, 6)
) -> Figure:
    """
    Convenience wrapper to build and display a CumulativeInfectionsPlot.

    Args:
        infection_times: Each element is a sequence of infection times (in days from start_date) for one trajectory.
            Or if an InfectionTimeCollector is provided, the infection times are extracted from it.
        start_date: Start date of the simulation/plot.
        end_date: End date of the simulation/plot.
        resolution: Grid step in days for cumulative counts.
        perc_bands: Central mass fractions for bands, e.g. (0.95, 0.5).
        cmap: Colormap name (str) or list of explicit colors (widest band first).
        scale: Y-axis scale, either "linear" or "log"
        show_mean: Whether to plot the mean curve.
        mean_style: Style dict for the mean curve.
        show_median: Whether to plot the median curve.
        median_style: Style dict for the median curve.
        obs_points: Optional observed (date, count) pairs (incident counts).
        obs_style: Style dict for observed points.
        dpi: Figure resolution in dots per inch.
        figsize: Figure size as (width, height) in inches.

    Returns:
        Matplotlib Figure object.
    """
    plot = CumulativeInfectionsPlot(
        infection_times if isinstance(infection_times, Sequence) else infection_times.infection_times,
        start_date,
        end_date,
        resolution=resolution,
        perc_bands=perc_bands,
        cmap=cmap,
        scale=scale,
        show_mean=show_mean,
        mean_style=mean_style,
        show_median=show_median,
        median_style=median_style,
        obs_points=obs_points,
        obs_style=obs_style,
        dpi=dpi,
        figsize=figsize
    )
    fig = plot.plot()
    plt.show()
    return fig
