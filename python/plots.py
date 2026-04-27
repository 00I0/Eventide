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
from scipy.special import logsumexp, gammaincc
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
        ax.set_xlabel(self.label)
        ax.grid(False)

        if not (self.show_mean or self.show_median or self.show_conf):
            return

        y_text = ax.get_ylim()[1] * 0.95
        mean, median, (lo_ci, hi_ci) = _empirical_stats(counts, centers, self.conf_level, self.conf_method)
        if self.show_conf:
            ax.axvspan(lo_ci, hi_ci, color='C1', alpha=0.125, hatch='//')
            ax.text(
                lo_ci, y_text,
                r'$\text{C.I.}_\text{low}$: ' + f'{lo_ci:.2f}',
                ha='center', va='top', color='C1', fontsize='x-small',
                rotation=90,
                bbox=dict(
                    boxstyle='round, pad=0.35',
                    facecolor=(1, 1, 1, 0.8),
                    edgecolor='none'
                )
            )
            ax.text(
                hi_ci, y_text,
                r'$\text{C.I.}_\text{high}$: ' + f'{hi_ci:.2f}',
                ha='center', va='top', color='C1', fontsize='x-small',
                rotation=90,
                bbox=dict(
                    boxstyle='round, pad=0.35',
                    facecolor=(1, 1, 1, 0.8),
                    edgecolor='none'
                )
            )
        if self.show_mean:
            ax.axvline(mean, color='C3', ls='-.', lw=1.25)
            ax.text(
                mean, y_text,
                f'mean: {mean:.2f}',
                ha='center', va='top', color='C3', fontsize='x-small', rotation=90,
                bbox=dict(
                    boxstyle='round, pad=0.35',
                    facecolor=(1, 1, 1, 0.8),
                    edgecolor='none'
                )
            )
        if self.show_median:
            ax.axvline(median, color='C2', ls='--', lw=1.25)
            ax.text(
                median, y_text,
                f'median: {median:.2f}',
                ha='center', va='top', color='C2', fontsize='x-small', rotation=90,
                bbox=dict(
                    boxstyle='round, pad=0.35',
                    facecolor=(1, 1, 1, 0.8),
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
        # rows, cols = 3, 4
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
        collectors: Sequence[Hist1D | Hist2D | HistSpec],
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
        self.cutoff_date = cutoff_date

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
            active_set_times: list[list[float]],
            conf: Optional[Tuple[float, ...]] = (0.3, 0.6, 0.9),
            colors: Optional[Sequence[str] | str] = None,
            label: str = 'Analytical',
            *,
            method: Literal['ev', 'size_only'] = 'ev',
    ) -> None:
        """
        Plot the analytical probability that transmission has ended.

        This method computes the extinction probability for each posterior draw and day on the plotting grid,
        summarizes the median across draws, and (optionally) shades credible bands.

        * method='ev' (expected-value lag, **age-aware**):
          Uses the per-lineage ages at the cutoff and a deterministic observation.

        * method='size_only' (**age-agnostic** baseline):
          Ignores ages and uses a single-lineage probability raised to the number of active lineages.



        Args:
            draws: Array of accepted posterior draws with shape (N, 5) ordered as [R0, k, r, alpha, theta].
            active_set_times: For each draw, a list of infection times (floats) relative to simulation start.
            conf: Credible levels for shaded bands, e.g. (0.3, 0.6, 0.9). Set to None to disable bands.
            colors: Either a Matplotlib colormap name or a sequence of colors.
                The first color is used for the median line;
                subsequent colors are used for bands from widest → narrowest.
                If None, the Matplotlib color cycle is used.
            label: Legend label for the median curve.
              method: Analytical formulation to use: 'ev' or 'size_only' (see above).

        Raises:
            ValueError: If input shapes are inconsistent or an unknown method is given.

        Returns:
            None. The function adds the curve and (optionally) the bands to self.ax.
        """
        # ---------- validate & unpack ----------
        if draws.ndim != 2 or draws.shape[1] != 5:
            raise ValueError(f"'draws' must have shape (N, 5); got {draws.shape}")
        if not isinstance(active_set_times, list) or len(active_set_times) != draws.shape[0]:
            raise ValueError("'active_set_times' must be a list of length N (same as draws).")
        if method not in ('ev', 'size_only'):
            raise ValueError("method must be 'ev' or 'size_only'.")

        R0s, ks, rs, alphas, thetas = draws.T
        Reffs = R0s * rs

        ks = np.maximum(ks, 1e-12)
        thetas = np.maximum(thetas, np.finfo(float).tiny)
        tiny = np.finfo(float).tiny

        # ---------- day grid (aligned to empirical binning) ----------
        start_day = int(np.floor(self.cutoff_day))
        end_day = int(np.floor(self.T_run))
        if end_day < start_day:
            return

        days = np.arange(start_day, end_day + 1, dtype=int)
        T_vals = days - self.cutoff_day  # quiet-window length T for each day
        dates = [self.start_date + timedelta(days=int(d)) for d in days]
        date_nums = mdates.date2num(dates)

        N = draws.shape[0]
        L = len(days)
        P_mat = np.empty((L, N), dtype=float)

        # ---------- per-draw curve ----------
        for m in range(N):
            # unique infection times for this draw
            t_list = np.asarray(sorted(set(active_set_times[m])), dtype=float)
            n_active = int(t_list.size)

            k = float(ks[m])
            Reff = float(Reffs[m])
            alpha = float(alphas[m])
            theta = float(thetas[m])

            if n_active == 0:
                P_mat[:, m] = 1.0
                continue

            if method == 'size_only':
                # Age-agnostic: single-lineage probability ^ number of actives
                F_T = gamma.cdf(T_vals, a=alpha, scale=theta)  # (L,)
                term1 = 1.0 + Reff / k
                term2 = np.maximum(1.0 + (Reff * F_T) / k, tiny)
                p_single = (term1 / term2) ** (-k)  # (L,)
                P = np.power(p_single, n_active, dtype=float)  # (L,)
            else:
                # Age-aware EV method with deterministic lag δ = αθ
                a = self.cutoff_day - t_list  # (I,)
                a = np.clip(a, 0.0, None)
                delta = alpha * theta

                a_shift = np.maximum(a - delta, 0.0)  # (I,)
                a_shift_grid = a_shift[None, :] + T_vals[:, None]  # (L, I)

                Fa = gamma.cdf(a_shift, a=alpha, scale=theta)  # (I,)
                FaT = gamma.cdf(a_shift_grid, a=alpha, scale=theta)  # (L, I)

                A_i = 1.0 + (Reff * (1.0 - Fa)) / k  # (I,)
                B_i = 1.0 + (Reff * (FaT - Fa[None, :])) / k  # (L, I)
                A_i = np.maximum(A_i, tiny)
                B_i = np.maximum(B_i, tiny)

                # prod_i (A_i / B_i(T))^{-k} in log-space
                logA = np.log(A_i)[None, :]  # (1, I)
                logB = np.log(B_i)  # (L, I)
                log_prod = -k * np.sum(logA - logB, axis=1)  # (L,)
                P = np.exp(log_prod)  # (L,)

            np.clip(P, 0.0, 1.0, out=P)
            P_mat[:, m] = P

        # ---------- aggregate across draws ----------
        median = np.median(P_mat, axis=1)

        bands: list[tuple[float, np.ndarray, np.ndarray]] = []
        if conf:
            for lvl in sorted(conf):
                qlo = (1 - lvl) * 50.0
                qhi = 100.0 - qlo
                lo = np.percentile(P_mat, qlo, axis=1)
                hi = np.percentile(P_mat, qhi, axis=1)
                bands.append((lvl, lo, hi))

        # ---------- colors & plotting ----------
        if colors is None:
            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            median_color = cycle[0]
            band_palette = cycle[1:] or [cycle[0]]
            band_colors = [band_palette[i % len(band_palette)] for i in range(len(bands))]
        elif isinstance(colors, str):
            cmap_obj = plt.get_cmap(colors)
            n = 1 + len(bands)
            pos = np.linspace(0.15, 0.85, num=n)
            rgba = [cmap_obj(p) for p in pos]
            median_color = rgba[0]
            band_colors = rgba[1:]
        else:
            if len(colors) < 1:
                raise ValueError('colors list must have at least one color.')
            clist = [mcolors.to_rgba(c) for c in colors]
            median_color = clist[0]
            palette = clist[1:] or [clist[0]]
            band_colors = [palette[i % len(palette)] for i in range(len(bands))]

        for (lvl, lo, hi), c in zip(reversed(bands), reversed(band_colors)):
            self.ax.fill_between(date_nums, lo, hi, alpha=0.25, color=c, label=f'{int(lvl * 100)}% CI')

        self.ax.plot(date_nums, median, lw=2, label=label, color=median_color)

    def add_empirical(
            self,
            matrix: np.ndarray,
            color: str = 'C3',
            marker: str = '.'
    ) -> None:
        """
        Add empirical extinction probability from a TimeMatrix.

        For each day d, ext(d) = P(no future infections | no infections in (cutoff, d]).
        With the matrix encoding first-after-cutoff in the columns and T+1 meaning
        “never again”, this is:

        Args:
            matrix: Square array with shape (T+2, T+2) produced by the TimeMatrix accumulator.
            color: Line/marker color.
            marker: Marker style.
        """

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 3:
            raise ValueError("'matrix' must be a square array with shape (T+2, T+2), T>=1.")

        T = matrix.shape[0] - 2  # last simulated day
        d0 = int(np.floor(self.cutoff_day))
        d1 = min(int(np.floor(self.T_run)), T)
        if d0 > d1:
            return

        days = np.arange(d0, d1 + 1, dtype=int)
        dates = [self.start_date + timedelta(days=int(d)) for d in days]
        date_nums = mdates.date2num(dates)

        # precompute column sums
        col_sums = matrix.sum(axis=0)
        never_again = col_sums[T + 1]  # trajectories with no post-cutoff infections

        comp = np.empty_like(days, dtype=float)
        total = col_sums.sum()
        if total <= 0:
            comp.fill(0.0)
        else:
            for i, d in enumerate(days):
                valid = col_sums[d + 1:].sum()  # count(col > d)
                comp[i] = (never_again / valid) if valid > 0 else 1.0

        self.ax.plot(date_nums, comp, lw=1.5, marker=marker, label='Empirical', color=color)

    def plot(self) -> Figure:
        """Finalize and display the plot."""
        self.ax.legend()
        self.fig.autofmt_xdate(rotation=30)
        self.fig.tight_layout()
        return self.fig

    def add_analytical3(
            self,
            draws: np.ndarray,
            active_set_times: list[list[float]],
            conf: Optional[Tuple[float, ...]] = (0.3, 0.6, 0.9),
            colors: Optional[Sequence[str] | str] = None,
            label: str = "Analytical",
            *,
            method: Literal[
                "ev",  # age-aware; deterministic lag; plain Gamma kernel
                "size_only",  # age-agnostic baseline; plain Gamma kernel
                "ev_exp",  # age-aware; exponential removal    => Gamma with θ_eff
                "size_only_exp",  # age-agnostic; exponential removal => Gamma with θ_eff
                "ev_weibull",  # age-aware; Weibull removal        => numerically thinned Gamma
                "size_only_weibull",  # age-agnostic; Weibull removal     => numerically thinned Gamma
            ] = "ev",
            lambda_c: float = 0.0,  # exponential removal rate (1/mean); required for *_exp methods
            weibull_tau: float = 0.0,  # Weibull scale (τ_c); required for *_weibull methods
            weibull_nu: float = 0.0,  # Weibull shape (ν); required for *_weibull methods
    ) -> None:
        """Plot the analytical probability that transmission has ended.

        This routine computes, for each posterior draw and each plot day, the
        conditional probability that no further infections occur *after* that day,
        given that no infections have occurred between the cutoff and that day.
        The result is summarized by the median across draws and (optionally) shaded
        credible bands.

        Two modeling axes are supported:

        1) **Age handling**
           - ``method`` in {``"ev"``, ``"ev_exp"``, ``"ev_weibull"``}:
             *Age-aware*. Uses per-lineage ages at the cutoff along with a
             deterministic observation delay :math:`δ = α θ` (shape–scale Gamma).
           - ``method`` in {``"size_only"``, ``"size_only_exp"``, ``"size_only_weibull"``}:
             *Age-agnostic*. Uses a single-lineage probability raised to the number
             of active lineages; does not use per-lineage ages.

        2) **Removal (thinning) model**
           - Plain kernel (no removal): ``"ev"`` or ``"size_only"``.
           - **Exponential removal** with rate ``lambda_c``:
             Equivalent to a Gamma kernel with the same shape ``α`` and an
             *effective* scale :math:`θ_eff = θ / (1 + λ_c θ)`. Select
             ``"ev_exp"`` or ``"size_only_exp"``.
           - **Weibull removal** with scale ``weibull_tau`` and shape ``weibull_nu``:
             Uses the numerically thinned kernel
             :math:`F_eff(t) = ∫_0^t w_Γ(u; α, θ) · exp(-(u/τ)^ν) du`.
             Select ``"ev_weibull"`` or ``"size_only_weibull"``.

        All Gamma calls use **shape–scale** parameterization.

        Args:
          self: Object providing the plotting context. Must define:
            - ``start_date`` (date or datetime): calendar origin for x-axis ticks.
            - ``cutoff_day`` (float): first day to plot (absolute, in simulation units).
            - ``T_run`` (float): last day to plot (absolute, in simulation units).
            - ``ax`` (matplotlib.axes.Axes): target axes.
          draws: Array of accepted posterior draws, shape ``(N, 5)`` ordered as
            ``[R0, k, r, alpha, theta]``.
          active_set_times: For each draw ``m``, list of infection times (floats)
            of lineages active at the cutoff (relative to simulation start). Duplicate
            times are removed internally.
          conf: Credible levels for shaded bands (values in (0,1)). If ``None`` or
            empty, no bands are drawn.
          colors: Either a Matplotlib colormap name (``str``) or a sequence of colors.
            The first color is used for the median line; subsequent colors (if any)
            are used for bands, widest → narrowest. If ``None``, the Matplotlib cycle
            is used.
          label: Legend label for the median curve.
          method: Analytical formulation and removal model. See choices above.
          lambda_c: Exponential removal rate (1/mean removal time). Must be > 0
            when ``method`` ends with ``"_exp"``; ignored otherwise.
          weibull_tau: Weibull removal scale ``τ_c``. Must be > 0 when ``method``
            ends with ``"_weibull"``; ignored otherwise.
          weibull_nu: Weibull removal shape ``ν``. Must be > 0 when ``method``
            ends with ``"_weibull"``; ignored otherwise.

        Raises:
          ValueError: If inputs have inconsistent shapes or required removal parameters
            are missing/invalid for the chosen ``method``.
        """

        # ----------------------------
        # Helpers: effective F_obs(T)
        # ----------------------------
        def _F_plain(x: np.ndarray, a: float, th: float) -> np.ndarray:
            """Gamma CDF F_Gamma(x; a, th)."""
            return gamma.cdf(x, a=a, scale=th)

        def _F_exp(x: np.ndarray, a: float, th: float, lam: float) -> np.ndarray:
            """Effective CDF under exponential removal: Gamma(a, θ_eff)."""
            theta_eff = th / (1.0 + lam * th)
            return gamma.cdf(x, a=a, scale=theta_eff)

        def _F_weibull(x: np.ndarray, a: float, th: float, tau: float, nu: float) -> np.ndarray:
            """Effective CDF under Weibull removal via numerical thinning.

            Computes F_eff(t) = ∫_0^t w_Gamma(u; a, th) * exp(-(u/tau)^nu) du
            for vector x (t) using a shared integration grid for stability.
            """
            x = np.asarray(x, dtype=float)
            x_clipped = np.maximum(x, 0.0)
            xmax = float(np.max(x_clipped))
            if xmax <= 0:
                return np.zeros_like(x_clipped)

            # Integration grid: 0..xmax, with step based on both Gamma scale and Weibull scale.
            # Use a modest grid; cumulative trapezoid will be vectorized.
            grid_max = xmax
            # Heuristic step: finer of 0.1*min(theta, tau) and grid_max/2000, bounded below.
            h = max(min(0.1 * min(th, tau), grid_max / 2000.0), grid_max / 5000.0)
            u = np.arange(0.0, grid_max + h, h, dtype=float)  # (G,)
            # Gamma pdf · Weibull survival
            w = gamma.pdf(u, a=a, scale=th) * np.exp(-np.power(u / tau, nu))
            # Cumulative integral via trapezoid
            cum = np.cumsum((w[:-1] + w[1:]) * 0.5 * h)
            cum = np.concatenate(([0.0], cum))  # cum[j] ≈ ∫_0^{u[j]} ...
            # Interpolate F_eff at x
            return np.interp(x_clipped, u, cum, left=0.0, right=cum[-1])

        def _select_F(kind: str):
            """Return a callable F_obs(x, alpha, theta) according to removal model."""
            if kind in ("ev", "size_only"):
                return lambda x, a, th: _F_plain(x, a, th)
            if kind in ("ev_exp", "size_only_exp"):
                if lambda_c <= 0.0:
                    raise ValueError("lambda_c must be > 0 for '*_exp' methods.")
                return lambda x, a, th: _F_exp(x, a, th, lambda_c)
            if kind in ("ev_weibull", "size_only_weibull"):
                if weibull_tau <= 0.0 or weibull_nu <= 0.0:
                    raise ValueError("weibull_tau and weibull_nu must be > 0 for '*_weibull' methods.")
                return lambda x, a, th: _F_weibull(x, a, th, weibull_tau, weibull_nu)
            raise ValueError("Unknown method.")

        # ----------------------------
        # Validate & unpack inputs
        # ----------------------------
        if draws.ndim != 2 or draws.shape[1] != 5:
            raise ValueError(f"'draws' must have shape (N, 5); got {draws.shape}")
        if not isinstance(active_set_times, list) or len(active_set_times) != draws.shape[0]:
            raise ValueError("'active_set_times' must be a list of length N (same as draws).")

        R0s, ks, rs, alphas, thetas = draws.T
        Reffs = R0s * rs

        ks = np.maximum(ks, 1e-12)
        thetas = np.maximum(thetas, np.finfo(float).tiny)
        tiny = np.finfo(float).tiny

        # ----------------------------
        # Plot grid aligned to empirical
        # ----------------------------
        start_day = int(np.floor(self.cutoff_day))
        end_day = int(np.floor(self.T_run))
        if end_day < start_day:
            return

        days = np.arange(start_day, end_day + 1, dtype=int)
        T_vals = days - self.cutoff_day  # quiet-window length T for each day
        dates = [self.start_date + timedelta(days=int(d)) for d in days]
        date_nums = mdates.date2num(dates)

        N = draws.shape[0]
        L = len(days)
        P_mat = np.empty((L, N), dtype=float)

        # Choose effective CDF according to removal model encoded in method
        F_eff = _select_F(method)

        # ----------------------------
        # Compute per-draw curves
        # ----------------------------
        for m in range(N):
            # unique infection times for this draw (avoid double counting)
            t_list = np.asarray(sorted(set(active_set_times[m])), dtype=float)
            n_active = int(t_list.size)

            k = float(ks[m])
            Reff = float(Reffs[m])
            alpha = float(alphas[m])
            theta = float(thetas[m])

            if n_active == 0:
                P_mat[:, m] = 1.0
                continue

            if method.startswith("size_only"):
                # Age-agnostic: use a single-lineage probability with F_eff(T)
                F_T = F_eff(T_vals, alpha, theta)  # (L,)
                term1 = 1.0 + Reff / k
                term2 = np.maximum(1.0 + (Reff * F_T) / k, tiny)
                p_single = (term1 / term2) ** (-k)  # (L,)
                P = np.power(p_single, n_active, dtype=float)  # (L,)

            else:
                # Age-aware EV: per-lineage factors with deterministic lag δ = αθ
                a = np.clip(self.cutoff_day - t_list, 0.0, None)  # (I,)
                delta = alpha * theta

                # Shift by δ; evaluate F_eff at ages and ages+T grid
                a_shift = np.maximum(a - delta, 0.0)  # (I,)
                a_shift_grid = a_shift[None, :] + T_vals[:, None]  # (L, I)

                Fa = F_eff(a_shift, alpha, theta)  # (I,)
                FaT = F_eff(a_shift_grid, alpha, theta)  # (L, I)

                # Core per-lineage terms
                A_i = 1.0 + (Reff * (1.0 - Fa)) / k  # (I,)
                B_i = 1.0 + (Reff * (FaT - Fa[None, :])) / k  # (L, I)
                A_i = np.maximum(A_i, tiny)
                B_i = np.maximum(B_i, tiny)

                # Product over lineages: ∏ (A_i / B_i(T))^{-k}
                logA = np.log(A_i)[None, :]  # (1, I)
                logB = np.log(B_i)  # (L, I)
                log_prod = -k * np.sum(logA - logB, axis=1)  # (L,)
                P = np.exp(log_prod)  # (L,)

            np.clip(P, 0.0, 1.0, out=P)
            P_mat[:, m] = P

        # ----------------------------
        # Aggregate across draws
        # ----------------------------
        median = np.median(P_mat, axis=1)

        bands: list[tuple[float, np.ndarray, np.ndarray]] = []
        if conf:
            for lvl in sorted(conf):
                qlo = (1 - lvl) * 50.0
                qhi = 100.0 - qlo
                lo = np.percentile(P_mat, qlo, axis=1)
                hi = np.percentile(P_mat, qhi, axis=1)
                bands.append((lvl, lo, hi))

        # ----------------------------
        # Colors & plotting
        # ----------------------------
        if colors is None:
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            median_color = cycle[0]
            band_palette = cycle[1:] or [cycle[0]]
            band_colors = [band_palette[i % len(bands)] for i in range(len(bands))]
        elif isinstance(colors, str):
            cmap_obj = plt.get_cmap(colors)
            n = 1 + len(bands)
            pos = np.linspace(0.15, 0.85, num=n)
            rgba = [cmap_obj(p) for p in pos]
            median_color = rgba[0]
            band_colors = rgba[1:]
        else:
            if len(colors) < 1:
                raise ValueError("colors list must have at least one color.")
            clist = [mcolors.to_rgba(c) for c in colors]
            median_color = clist[0]
            palette = clist[1:] or [clist[0]]
            band_colors = [palette[i % len(bands)] for i in range(len(bands))]

        for (lvl, lo, hi), c in zip(reversed(bands), reversed(band_colors)):
            self.ax.fill_between(date_nums, lo, hi, alpha=0.25, color=c, label=f"{int(lvl * 100)}% CI")

        self.ax.plot(date_nums, median, lw=2, label=label, color=median_color)

    def add_analytical_first_after(
            self,
            draws: np.ndarray,
            active_set_times: list[list[float]],
            conf: Optional[Tuple[float, ...]] = (0.3, 0.6, 0.9),
            colors: Optional[Sequence[str] | str] = None,
            label: str = "Analytical",
            *,
            horizon: Literal["finite", "infinite"] = "finite",
            aggregate: Literal["median", "mean"] = "median",
    ) -> None:
        """Plot the analytical counterpart of the empirical 'first-after' estimator.

        This computes, for each posterior draw m and each plotted day d >= cutoff,
        the conditional probability that there are no infections **after** day d,
        given that there were no infections in (cutoff, d]. It uses only information
        available up to the cutoff (infection times before the cutoff and model
        parameters), and matches the empirical estimand when `horizon="finite"`.

        Let T = d - cutoff be the quiet-window length. For a draw with parameters
        (R0, k, r, alpha, theta), define R_eff = R0 * r, and for each lineage i in
        the active set at the cutoff, its age a_i = max(cutoff - t_i, 0). With the
        Gamma(shape=alpha, scale=theta) generation kernel and the NB–Gamma mixing,
        the survival of the first post-cutoff infection is

            S(T) = Π_i [ 1 + (R_eff / k) * Δ_i(T) ]^{-k},
            Δ_i(T) = F_Γ(a_i + T; α, θ) - F_Γ(a_i; α, θ),

        where F_Γ is the Gamma CDF in shape–scale parameterization. The plotted
        conditional probability equals S(T_h) / S(T), where:

          - horizon="finite":  T_h = T_run - cutoff  (matches empirical finite horizon)
          - horizon="infinite": T_h = ∞, i.e., Δ_i(T_h) = 1 - F_Γ(a_i; α, θ)

        The function aggregates per-draw curves across draws using either the median
        (default) or the mean, and optionally shades credible bands.

        Args:
          self: Object providing plotting context. Must define:
              - start_date (date or datetime): origin for x-axis ticks.
              - cutoff_day (float): first day to plot (absolute, simulation time).
              - T_run (float): last day to plot (absolute, simulation time).
              - ax (matplotlib.axes.Axes): target axes.
          draws: Array of accepted posterior draws with shape (N, 5) ordered as
            [R0, k, r, alpha, theta].
          active_set_times: For each draw m, a list of infection times (floats)
            for lineages active at the cutoff (relative to simulation start).
            Duplicates are removed internally per draw.
          conf: Credible levels for shaded bands (values in (0, 1)). If None or
            empty, no bands are drawn.
          colors: Either a Matplotlib colormap name (str) or a sequence of colors.
            The first color is used for the median/mean line; subsequent colors (if any)
            are used for bands from widest → narrowest. If None, uses the Matplotlib
            default color cycle.
          label: Legend label for the aggregated analytical curve.
          horizon: "finite" to match the empirical finite-horizon estimator
            (uses T_h = T_run - cutoff), or "infinite" for the theoretical infinite
            horizon (uses T_h = ∞).
          aggregate: "median" (robust summary; default) or "mean" (matches the
            expectation of an empirical proportion across draws).

        Raises:
          ValueError: If input shapes are inconsistent or if arguments are invalid.

        Returns:
          None. Adds the analytical curve and (optionally) credible bands to self.ax.
        """
        # ---- validate & unpack ----
        if draws.ndim != 2 or draws.shape[1] != 5:
            raise ValueError(f"'draws' must have shape (N, 5); got {draws.shape}")
        if not isinstance(active_set_times, list) or len(active_set_times) != draws.shape[0]:
            raise ValueError("'active_set_times' must be a list of length N (same as draws).")
        if horizon not in ("finite", "infinite"):
            raise ValueError("horizon must be 'finite' or 'infinite'.")
        if aggregate not in ("median", "mean"):
            raise ValueError("aggregate must be 'median' or 'mean'.")

        R0s, ks, rs, alphas, thetas = draws.T
        Reffs = R0s * rs

        # ks = np.maximum(ks, 1e-12)
        thetas = np.maximum(thetas, np.finfo(float).tiny)
        tiny = np.finfo(float).tiny

        # ---- day grid aligned to empirical binning ----
        start_day = int(np.floor(self.cutoff_day))
        end_day = int(np.floor(self.T_run))
        if end_day < start_day:
            return

        days = np.arange(start_day, end_day + 1, dtype=int)
        T_vals = days - self.cutoff_day  # quiet-window lengths for plotted days
        dates = [self.start_date + timedelta(days=int(d)) for d in days]
        date_nums = mdates.date2num(dates)

        # finite-horizon length from cutoff to T_run (nonnegative)
        T_h = float(self.T_run - self.cutoff_day)

        N = draws.shape[0]
        L = len(days)
        P_mat = np.empty((L, N), dtype=float)

        # ---- per-draw computation ----
        for m in range(N):
            # unique infection times (avoid double-counting a lineage)
            t_list = np.asarray(sorted(set(active_set_times[m])), dtype=float)
            I = int(t_list.size)

            k = float(ks[m])
            Reff = float(Reffs[m])
            alpha = float(alphas[m])
            theta = float(thetas[m])

            if I == 0:
                P_mat[:, m] = 1.0
                continue

            # ages at cutoff and base CDF at ages
            a = np.clip(self.cutoff_day - t_list, 0.0, None)  # (I,)
            Fa = gamma.cdf(a, a=alpha, scale=theta)  # (I,)

            # Δ_i(T) = F(a_i + T) - F(a_i)
            FaT = gamma.cdf(a[None, :] + T_vals[:, None], a=alpha, scale=theta)  # (L, I)
            Delta_T = np.maximum(FaT - Fa[None, :], 0.0)  # (L, I)

            # Δ_i(T_horizon)
            if horizon == "finite":
                FaH = gamma.cdf(a + T_h, a=alpha, scale=theta)  # (I,)
                Delta_H = np.maximum(FaH - Fa, 0.0)  # (I,)
            else:
                Delta_H = 1.0 - Fa  # (I,)

            # S(T) = Π_i [1 + (Reff/k) Δ_i(T)]^{-k}
            # Compute log S(T) for stability, then P(T) = S(T_h)/S(T).
            term_den = np.maximum(1.0 + (Reff / k) * Delta_T, tiny)  # (L, I)
            term_num = np.maximum(1.0 + (Reff / k) * Delta_H, tiny)  # (I,)

            logS_T = -k * np.sum(np.log(term_den), axis=1)  # (L,)
            logS_H = -k * np.sum(np.log(term_num))  # scalar

            # P_m(T) = exp(logS_H - logS_T)
            P = np.exp(logS_H - logS_T)  # (L,)
            np.clip(P, 0.0, 1.0, out=P)
            P_mat[:, m] = P

        # ---- aggregate across draws ----
        if aggregate == "median":
            center = np.median(P_mat, axis=1)
        else:
            center = np.mean(P_mat, axis=1)

        bands: list[tuple[float, np.ndarray, np.ndarray]] = []
        if conf:
            for lvl in sorted(conf):
                qlo = (1 - lvl) * 50.0
                qhi = 100.0 - qlo
                lo = np.percentile(P_mat, qlo, axis=1)
                hi = np.percentile(P_mat, qhi, axis=1)
                bands.append((lvl, lo, hi))

        # ---- colors & plotting ----
        if colors is None:
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            median_color = cycle[0]
            band_palette = cycle[1:] or [cycle[0]]
            band_colors = [band_palette[i % len(bands)] for i in range(len(bands))]
        elif isinstance(colors, str):
            cmap_obj = plt.get_cmap(colors)
            n = 1 + len(bands)
            pos = np.linspace(0.15, 0.85, num=n)
            rgba = [cmap_obj(p) for p in pos]
            median_color = rgba[0]
            band_colors = rgba[1:]
        else:
            if len(colors) < 1:
                raise ValueError("colors list must have at least one color.")
            clist = [mcolors.to_rgba(c) for c in colors]
            median_color = clist[0]
            palette = clist[1:] or [clist[0]]
            band_colors = [palette[i % len(bands)] for i in range(len(bands))]

        for (lvl, lo, hi), c in zip(reversed(bands), reversed(band_colors)):
            self.ax.fill_between(date_nums, lo, hi, alpha=0.25, color=c, label=f"{int(lvl * 100)}% CI")

        agg_label = f"{label} ({aggregate})" if aggregate in ("median", "mean") else label
        self.ax.plot(date_nums, center, lw=2, label=agg_label, color=median_color)

    def add_analytical_mixture(
            self,
            draws: np.ndarray,
            active_set_times: list[list[float]],
            conf: Optional[Tuple[float, ...]] = (0.3, 0.6, 0.9),
            colors: Optional[Sequence[str] | str] = None,
            label: str = "Analytical (mixture)",
            *,
            horizon: Literal["finite", "infinite"] = "infinite",
            bootstrap_samples: int = 0,
            random_state: Optional[int] = None,
    ) -> None:
        """Plot the analytical counterpart of the empirical first-after estimator
        using a *mixture (ratio-of-sums)* aggregation across draws.

        For each accepted draw ``m`` with parameters
        :math:`(R_{0m}, k_m, r_m, \\alpha_m, \\theta_m)` and active-set ages
        :math:`a_{mi}=\\max(t_\\star - t_{mi}, 0)`, the survival of the
        **first post-cutoff infection time** over a quiet window of length ``T`` is

        .. math::

            S_m(T)
            \\,=\\, \\prod_{i\\in\\mathcal I_m}\\big(1 + c_m\\,\\Delta_{mi}(T)\\big)^{-k_m},
            \\quad
            c_m = R_{\\rm eff,m}/k_m,
            \\quad
            \\Delta_{mi}(T) = F_\\Gamma(a_{mi}+T;\\alpha_m,\\theta_m) - F_\\Gamma(a_{mi};\\alpha_m,\\theta_m),

        where :math:`F_\\Gamma` is the Gamma CDF in **shape–scale** form and
        :math:`R_{\\rm eff,m}=R_{0m}r_m`.

        The empirical curve at day :math:`d=t_\\star+T` estimates the conditional probability

        .. math::

            \\Pr(\\text{no infection after }d \\mid \\text{no infection in }(t_\\star,d])
            \\,=\\, \\frac{\\Pr(Y>T_h)}{\\Pr(Y>T)}
            \\,=\\, \\frac{S_m(T_h)}{S_m(T)} ,

        with :math:`Y` the first-after time. Aggregating across a mixture of draws,
        the empirical *proportion* is a **ratio of counts**, whose expectation equals
        a **ratio of expectations**:

        .. math::

            p_{\\rm mix}(T)
            \\,=\\,
            \\frac{\\sum_m S_m(T_h)}{\\sum_m S_m(T)}.

        This function evaluates :math:`p_{\\rm mix}(T)` on the plotting grid, using
        only information available at the cutoff (parameters and pre-cutoff infection times).
        No post-cutoff simulation is required.

        Args:
          self: Object providing plotting context. Must define:
            - ``start_date`` (date or datetime): origin for x-axis ticks.
            - ``cutoff_day`` (float): first day to plot (absolute simulation time).
            - ``T_run`` (float): last day to plot (absolute simulation time).
            - ``ax`` (matplotlib.axes.Axes): target axes.
          draws: Array of accepted draws with shape ``(N, 5)`` ordered as
            ``[R0, k, r, alpha, theta]``. Gamma parameters use shape–scale.
          active_set_times: For each draw ``m``, list of infection times (floats)
            ``t_{mi}`` of lineages that are active at the cutoff; times are relative
            to the simulation start. Duplicates are removed internally per draw.
          conf: Credible levels for optional shaded bands (values in (0,1)). Bands are
            computed by bootstrap over draws when ``bootstrap_samples > 0``. Set to
            ``None`` or ``()`` to disable bands.
          colors: Either a Matplotlib colormap name (``str``) or a sequence of colors.
            The first color is used for the main curve; subsequent colors (if any) are
            used for bands from widest → narrowest. If ``None``, the Matplotlib cycle
            is used.
          label: Legend label for the analytical curve.
          horizon: Time horizon for the numerator:
            - ``"infinite"``: uses :math:`T_h=\\infty`, i.e.
              :math:`\\Delta_{mi}(\\infty)=1-F_\\Gamma(a_{mi};\\alpha_m,\\theta_m)`.
            - ``"finite"``: uses :math:`T_h = T_{\\max} = T_{\\rm run}-t_\\star`.
          bootstrap_samples: Number of bootstrap resamples of draws to build uncertainty
            bands for the mixture curve. Set to ``0`` to skip bands (fastest).
          random_state: Seed for the bootstrap RNG (for reproducibility). Ignored if
            ``bootstrap_samples == 0``.

        Raises:
          ValueError: If input shapes are inconsistent or arguments are invalid.

        Returns:
          None. The function adds the mixture curve (and optional bands) to ``self.ax``.
        """
        # -------- validate inputs --------
        if draws.ndim != 2 or draws.shape[1] != 5:
            raise ValueError(f"'draws' must have shape (N, 5); got {draws.shape}")
        if not isinstance(active_set_times, list) or len(active_set_times) != draws.shape[0]:
            raise ValueError("'active_set_times' must be a list of length N (same as draws).")
        if horizon not in ("finite", "infinite"):
            raise ValueError("horizon must be 'finite' or 'infinite'.")
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be >= 0.")

        R0s, ks, rs, alphas, thetas = draws.T
        Reffs = R0s * rs

        ks = np.maximum(ks, 1e-12)
        thetas = np.maximum(thetas, np.finfo(float).tiny)
        tiny = np.finfo(float).tiny

        # -------- plotting grid aligned to empirical bins --------
        start_day = int(np.floor(self.cutoff_day))
        end_day = int(np.floor(self.T_run))
        if end_day < start_day:
            return

        days = np.arange(start_day, end_day + 1, dtype=int)
        T_vals = days - self.cutoff_day  # quiet-window lengths for each plotted day
        dates = [self.start_date + timedelta(days=int(d)) for d in days]
        date_nums = mdates.date2num(dates)

        # finite horizon length from cutoff to end of run
        T_h = float(self.T_run - self.cutoff_day)

        N = draws.shape[0]
        L = len(days)

        # We'll compute log S_m(T) for all T (shape LxN) and log S_m(T_h) (shape N,)
        logS_T = np.empty((L, N), dtype=float)
        logS_H = np.empty((N,), dtype=float)

        # -------- per-draw survival computations --------
        for m in range(N):
            # unique infection times for this draw (avoid double counting)
            t_list = np.asarray(sorted(set(active_set_times[m])), dtype=float)

            k = float(ks[m])
            Reff = float(Reffs[m])
            alpha = float(alphas[m])
            theta = float(thetas[m])

            if t_list.size == 0:
                # No active lineages at cutoff -> S_m(T) = 1 for all T
                logS_T[:, m] = 0.0
                logS_H[m] = 0.0
                continue

            # Ages at cutoff and base Gamma CDF at ages
            a = np.clip(self.cutoff_day - t_list, 0.0, None)  # (I,)
            Fa = gamma.cdf(a, a=alpha, scale=theta)  # (I,)

            # Δ_i(T) over grid: F(a_i + T) - F(a_i)
            FaT = gamma.cdf(a[None, :] + T_vals[:, None], a=alpha, scale=theta)  # (L, I)
            Delta_T = np.maximum(FaT - Fa[None, :], 0.0)  # (L, I)

            # Δ_i(T_horizon)
            if horizon == "finite":
                FaH = gamma.cdf(a + T_h, a=alpha, scale=theta)  # (I,)
                Delta_H = np.maximum(FaH - Fa, 0.0)  # (I,)
            else:
                Delta_H = 1.0 - Fa  # (I,)

            # S_m(T) = Π_i [1 + (Reff/k) Δ_i(T)]^{-k}
            # Compute in log-space: logS_m(T) = -k * Σ_i log(1 + (Reff/k) Δ_i(T))
            term_T = np.maximum(1.0 + (Reff / k) * Delta_T, tiny)  # (L, I)
            logS_T[:, m] = -k * np.sum(np.log(term_T), axis=1)  # (L,)

            # S_m(T_h) similarly
            term_H = np.maximum(1.0 + (Reff / k) * Delta_H, tiny)  # (I,)
            logS_H[m] = -k * np.sum(np.log(term_H))  # ()

        # -------- mixture aggregation: ratio of sums across draws --------
        # p_mix(T) = sum_m S_m(T_h) / sum_m S_m(T) = exp( logsumexp(logS_H) - logsumexp(logS_T, axis=1) )
        log_num = logsumexp(logS_H)  # scalar
        log_den = logsumexp(logS_T, axis=1)  # (L,)
        p_mix = np.exp(log_num - log_den)  # (L,)
        np.clip(p_mix, 0.0, 1.0, out=p_mix)

        # -------- optional bootstrap bands over draws --------
        bands: list[tuple[float, np.ndarray, np.ndarray]] = []
        if conf and bootstrap_samples > 0:
            rng = np.random.default_rng(random_state)
            B = bootstrap_samples
            samples = np.empty((B, L), dtype=float)

            for b in range(B):
                idx = rng.integers(0, N, size=N)  # bootstrap resample of draws
                log_num_b = logsumexp(logS_H[idx])  # scalar
                log_den_b = logsumexp(logS_T[:, idx], axis=1)
                pb = np.exp(log_num_b - log_den_b)
                np.clip(pb, 0.0, 1.0, out=pb)
                samples[b, :] = pb

            for lvl in sorted(conf):
                qlo = (1 - lvl) * 50.0
                qhi = 100.0 - qlo
                lo = np.percentile(samples, qlo, axis=0)
                hi = np.percentile(samples, qhi, axis=0)
                bands.append((lvl, lo, hi))

        # -------- colors & plotting --------
        if colors is None:
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            main_color = cycle[0]
            band_palette = cycle[1:] or [cycle[0]]
            band_colors = [band_palette[i % len(bands)] for i in range(len(bands))]
        elif isinstance(colors, str):
            cmap_obj = plt.get_cmap(colors)
            n = 1 + len(bands)
            pos = np.linspace(0.15, 0.85, num=n)
            rgba = [cmap_obj(p) for p in pos]
            main_color = rgba[0]
            band_colors = rgba[1:]
        else:
            if len(colors) < 1:
                raise ValueError("colors list must have at least one color.")
            clist = [mcolors.to_rgba(c) for c in colors]
            main_color = clist[0]
            palette = clist[1:] or [clist[0]]
            band_colors = [palette[i % len(bands)] for i in range(len(bands))]

        for (lvl, lo, hi), c in zip(reversed(bands), reversed(band_colors)):
            self.ax.fill_between(date_nums, lo, hi, alpha=0.25, color=c, label=f"{int(lvl * 100)}% CI")

        self.ax.plot(date_nums, p_mix, lw=2, label=label, color=main_color)

    from typing import Literal, Optional, Sequence, Tuple
    import numpy as np

    def add_analytical_surrogate(
            self,
            draws: np.ndarray,
            active_set_times: list[list[float]],
            conf: Optional[Tuple[float, ...]] = (0.3, 0.6, 0.9),
            colors: Optional[Sequence[str] | str] = None,
            label: str = "Analytical",
            *,
            method: Literal["exact", "gompertz", "two_exp", "gamma_moment"] = "gompertz",
            horizon: Literal["infinite", "finite"] = "infinite",
            rho: float = 3.0,
            bootstrap_samples: int = 0,
            random_state: Optional[int] = None,
    ) -> None:
        """Plot an analytical approximation to the empirical “no-further-infection” curve,
        using only pre-cutoff information (parameters + infection times up to the cutoff).

        The target estimand matches the empirical first-after estimator. For a quiet
        window length T (days after cutoff), each accepted draw m has a first-after
        survival S_m(T). The empirical proportion is a ratio of counts across draws,
        whose expectation equals the **ratio of expectations**:

            p_mix(T) = sum_m S_m(T_h) / sum_m S_m(T),

        where T_h is the numerator’s horizon (∞ for infinite-horizon extinction, or
        T_run - cutoff for a finite horizon).

        This function provides several per-draw models for S_m(T) that depend only on
        (R0, k, r, alpha, theta) and the ages at cutoff a_{mi} = max(cutoff - t_{mi}, 0):

          • method="exact":     Exact NB–Gamma first-after survival.
          • method="gompertz":  Single-parameter hazard surrogate that matches the exact
                                start value and initial slope; closed form.
          • method="two_exp":   Two-exponential hazard surrogate that matches value,
                                slope, and curvature at 0; closed form (depends on rho).
          • method="gamma_moment": Moment-matched Gamma-driven hazard shape using
                                incomplete-gamma moments; closed form.

        All gamma calls use shape–scale parameterization.

        Args:
          draws: Accepted posterior draws, shape (N, 5) ordered as [R0, k, r, alpha, theta].
          active_set_times: For each draw m, list of infection times (floats) for lineages
            active at the cutoff (relative to simulation start). Duplicates are removed per draw.
          conf: Credible levels for shaded bands (values in (0, 1)). Set to None or () to skip.
          colors: Matplotlib colormap name (str) or list of colors. First color is the line;
            remaining colors (if any) are used for the bands from widest → narrowest.
          label: Legend label for the plotted curve.
          method: Per-draw survival model; see choices above.
          horizon: Numerator horizon. "infinite" uses T_h = ∞; "finite" uses T_h = T_run - cutoff.
          rho: Ratio β2/β1 used by method="two_exp" (β2 = rho * β1). Must be > 1.
          bootstrap_samples: If > 0, build uncertainty bands by bootstrapping draws (fast).
          random_state: RNG seed for bootstrap.

        Raises:
          ValueError: If inputs or options are invalid.

        Returns:
          None. Adds the analytical mixture curve (and optional bands) to self.ax.
        """
        # -------- validate inputs --------
        if draws.ndim != 2 or draws.shape[1] != 5:
            raise ValueError(f"'draws' must have shape (N, 5); got {draws.shape}")
        if not isinstance(active_set_times, list) or len(active_set_times) != draws.shape[0]:
            raise ValueError("'active_set_times' must be a list of length N (same as draws).")
        if method not in {"exact", "gompertz", "two_exp", "gamma_moment"}:
            raise ValueError("method must be one of {'exact','gompertz','two_exp','gamma_moment'}.")
        if horizon not in {"infinite", "finite"}:
            raise ValueError("horizon must be 'infinite' or 'finite'.")
        if method == "two_exp" and not (rho is not None and rho > 1.0):
            raise ValueError("for method='two_exp', rho must be > 1.")
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be >= 0.")

        # -------- unpack draws --------
        R0s, ks, rs, alphas, thetas = draws.T
        Reffs = R0s * rs
        ks = np.maximum(ks, 1e-12)
        thetas = np.maximum(thetas, np.finfo(float).tiny)
        tiny = np.finfo(float).tiny

        # -------- plotting grid --------
        start_day = int(np.floor(self.cutoff_day))
        end_day = int(np.floor(self.T_run))
        if end_day < start_day:
            return
        days = np.arange(start_day, end_day + 1, dtype=int)
        T_vals = days - self.cutoff_day  # quiet-window lengths for plotted days (float)
        dates = [self.start_date + timedelta(days=int(d)) for d in days]
        date_nums = mdates.date2num(dates)
        T_h = float(self.T_run - self.cutoff_day)  # finite horizon length

        N = draws.shape[0]
        L = len(days)

        # Will accumulate log S_m(T) for all T (LxN) and log S_m(T_h) (N,)
        logS_T = np.empty((L, N), dtype=float)
        logS_H = np.empty((N,), dtype=float)

        # -------- helper: compute exact S_m(T) ingredients from pre-cutoff ages --------
        def _ages_and_CDFs(t_list: np.ndarray, alpha: float, theta: float):
            """Compute ages at cutoff and base CDF/pdf at ages."""
            a = np.clip(self.cutoff_day - t_list, 0.0, None)  # (I,)
            Fa = gamma.cdf(a, a=alpha, scale=theta)  # (I,)
            fa = gamma.pdf(a, a=alpha, scale=theta)  # (I,)
            return a, Fa, fa

        # -------- per-draw computation of S_m(T) according to method --------
        for m in range(N):
            # unique infection times for this draw
            t_list = np.asarray(sorted(set(active_set_times[m])), dtype=float)

            k = float(ks[m])
            Reff = float(Reffs[m])
            alpha = float(alphas[m])
            theta = float(thetas[m])

            if t_list.size == 0:
                # No active lineages at cutoff -> S_m(T) ≡ 1
                logS_T[:, m] = 0.0
                logS_H[m] = 0.0
                continue

            a, Fa, fa = _ages_and_CDFs(t_list, alpha, theta)
            c = Reff / k

            if method == "exact":
                # Δ_i(T) = F(a_i + T) - F(a_i)
                FaT = gamma.cdf(a[None, :] + T_vals[:, None], a=alpha, scale=theta)  # (L,I)
                Delta_T = np.maximum(FaT - Fa[None, :], 0.0)  # (L,I)

                if horizon == "finite":
                    FaH = gamma.cdf(a + T_h, a=alpha, scale=theta)  # (I,)
                    Delta_H = np.maximum(FaH - Fa, 0.0)  # (I,)
                else:
                    Delta_H = 1.0 - Fa  # (I,)

                term_T = np.maximum(1.0 + c * Delta_T, tiny)
                term_H = np.maximum(1.0 + c * Delta_H, tiny)

                logS_T[:, m] = -k * np.sum(np.log(term_T), axis=1)
                logS_H[m] = -k * np.sum(np.log(term_H))
                continue

            # For surrogates we need: A_m = -log S_m(∞), h0, and optionally h1
            # Exact S_m(∞) (uses only pre-cutoff info)
            Delta_inf = 1.0 - Fa  # (I,)
            term_inf = np.maximum(1.0 + c * Delta_inf, tiny)  # (I,)
            A_m = k * np.sum(np.log(term_inf))  # A_m = -log S_m(∞) >= 0
            if A_m <= 0.0:
                # Degenerate: no chance of post-cutoff transmission
                logS_T[:, m] = 0.0
                logS_H[m] = 0.0
                continue

            # Initial hazard h(0) and (optionally) its derivative
            H0 = Reff * float(np.sum(fa))  # h(0) = R_eff * Σ f(a_i)
            # h'(0) = R_eff * Σ f'(a_i) - (R_eff^2/k) * Σ f(a_i)^2
            # f'_Gamma(a) = f(a) * ( (α-1)/a - 1/θ ) for a>0; for a=0 and α>1, limit = -∞, handle numerically:
            with np.errstate(divide="ignore", invalid="ignore"):
                fprime = fa * ((alpha - 1.0) / np.maximum(a, tiny) - 1.0 / theta)
            H1 = Reff * float(np.sum(fprime)) - (Reff * Reff / k) * float(np.sum(fa * fa))

            if method == "gompertz":
                # Hazard surrogate: h̃(T) = θ e^{-β T} with θ=H0, β=H0/A_m
                theta_h = H0
                beta_h = H0 / A_m if A_m > 0 else 0.0
                # S̃(T) = exp( - ∫_0^T h̃(u) du ) = exp( - A_m (1 - e^{-β T}) )
                S_T = np.exp(-A_m * (1.0 - np.exp(-beta_h * T_vals)))
                if horizon == "finite":
                    S_H = np.exp(-A_m * (1.0 - np.exp(-beta_h * T_h)))
                else:
                    S_H = np.exp(-A_m)  # S(∞) by construction
                logS_T[:, m] = np.log(np.maximum(S_T, tiny))
                logS_H[m] = np.log(np.maximum(S_H, tiny))
                continue

            if method == "two_exp":
                # Two-exponential hazard: h̃(T) = θ1 e^{-β1 T} + θ2 e^{-β2 T}, β2 = ρ β1
                # Matching constraints at 0:
                #   θ1 + θ2 = H0
                #   θ1/β1 + θ2/β2 = A_m
                #   -θ1 β1 - θ2 β2 = H1
                # Closed-form β1 from quadratic: ρ A_m β1^2 - H0(ρ+1) β1 - H1 = 0
                a_q = rho * A_m
                b_q = -H0 * (rho + 1.0)
                c_q = -H1
                disc = b_q * b_q - 4.0 * a_q * c_q
                # Guard for numerical issues; if fails, fall back to Gompertz
                if not (np.isfinite(disc) and disc >= 0):
                    # fallback: Gompertz
                    beta_h = H0 / A_m
                    S_T = np.exp(-A_m * (1.0 - np.exp(-beta_h * T_vals)))
                    S_H = np.exp(-A_m * (1.0 - np.exp(-beta_h * (T_h if horizon == "finite" else 1e12))))
                    if horizon == "infinite":
                        S_H = np.exp(-A_m)
                    logS_T[:, m] = np.log(np.maximum(S_T, tiny))
                    logS_H[m] = np.log(np.maximum(S_H, tiny))
                    continue

                beta1 = (-b_q + np.sqrt(disc)) / (2.0 * a_q)
                if not (np.isfinite(beta1) and beta1 > 0):
                    # fallback
                    beta_h = H0 / A_m
                    S_T = np.exp(-A_m * (1.0 - np.exp(-beta_h * T_vals)))
                    S_H = np.exp(-A_m) if horizon == "infinite" else np.exp(-A_m * (1.0 - np.exp(-beta_h * T_h)))
                    logS_T[:, m] = np.log(np.maximum(S_T, tiny))
                    logS_H[m] = np.log(np.maximum(S_H, tiny))
                    continue

                beta2 = rho * beta1
                # θ2 = -(H0 + H1/β1)/(ρ - 1), θ1 = H0 - θ2
                theta2 = -(H0 + H1 / beta1) / (rho - 1.0)
                theta1 = H0 - theta2

                # If nonpositive weights, fall back to Gompertz
                if (theta1 <= 0) or (theta2 <= 0):
                    beta_h = H0 / A_m
                    S_T = np.exp(-A_m * (1.0 - np.exp(-beta_h * T_vals)))
                    S_H = np.exp(-A_m) if horizon == "infinite" else np.exp(-A_m * (1.0 - np.exp(-beta_h * T_h)))
                    logS_T[:, m] = np.log(np.maximum(S_T, tiny))
                    logS_H[m] = np.log(np.maximum(S_H, tiny))
                    continue

                # S̃(T) = exp( -θ1/β1 (1 - e^{-β1 T}) - θ2/β2 (1 - e^{-β2 T}) )
                part1 = (theta1 / beta1) * (1.0 - np.exp(-beta1 * T_vals))
                part2 = (theta2 / beta2) * (1.0 - np.exp(-beta2 * T_vals))
                S_T = np.exp(- (part1 + part2))
                if horizon == "finite":
                    S_H = np.exp(- ((theta1 / beta1) * (1.0 - np.exp(-beta1 * T_h))
                                    + (theta2 / beta2) * (1.0 - np.exp(-beta2 * T_h))))
                else:
                    S_H = np.exp(- (theta1 / beta1 + theta2 / beta2))  # equals exp(-A_m)
                logS_T[:, m] = np.log(np.maximum(S_T, tiny))
                logS_H[m] = np.log(np.maximum(S_H, tiny))
                continue

            if method == "gamma_moment":
                # Hazard-shape surrogate: p̃(T) = exp( -A_m [1 - F_Γ(T; α_H, θ_H)] )
                # Moment approximation for h(T) uses: h(T) ≈ R_eff Σ f_Γ(a_i + T)
                # A_m is exact; moments approximate the shape only.
                # First moment M1 = ∫ T h(T) dT ≈ R_eff Σ E[(τ - a)_+]
                # Second moment M2 = ∫ T^2 h(T) dT ≈ R_eff Σ E[(τ - a)_+^2]
                # For τ ~ Gamma(α, θ), with x = a/θ, using regularized upper γ: Q(s,x) = Γ(s,x)/Γ(s) = gammaincc(s, x):
                x = a / theta
                Qa = gammaincc(alpha, x)  # P(τ > a)
                Qa1 = gammaincc(alpha + 1.0, x)  # scaled for E[τ 1_{τ>a}]
                Qa2 = gammaincc(alpha + 2.0, x)  # scaled for E[τ^2 1_{τ>a}]

                E_tau_gt = theta * alpha * Qa1  # E[τ 1_{τ>a}]
                E_tau2_gt = (theta ** 2) * alpha * (alpha + 1.0) * Qa2  # E[τ^2 1_{τ>a}]

                E_pos1 = E_tau_gt - a * Qa  # E[(τ - a)_+]
                E_pos2 = E_tau2_gt - 2.0 * a * E_tau_gt + (a ** 2) * Qa  # E[(τ - a)_+^2]

                M1 = Reff * float(np.sum(E_pos1))
                M2 = Reff * float(np.sum(E_pos2))
                # Mean and variance of hazard-shape g(T) = h(T) / A_m
                mu_H = M1 / A_m if A_m > 0 else 0.0
                var_H = max(M2 / A_m - mu_H * mu_H, 1e-12)

                # Convert to shape–scale for Gamma: α_H = μ^2/σ^2, θ_H = σ^2/μ
                alpha_H = max(mu_H * mu_H / var_H, 1e-9)
                theta_H = max(var_H / max(mu_H, 1e-12), 1e-9)

                # S̃(T) = exp( -A_m [1 - F_Γ(T; α_H, θ_H)] )
                G_T = gamma.cdf(T_vals, a=alpha_H, scale=theta_H)  # (L,)
                S_T = np.exp(-A_m * (1.0 - G_T))
                if horizon == "finite":
                    G_H = gamma.cdf(T_h, a=alpha_H, scale=theta_H)
                    S_H = np.exp(-A_m * (1.0 - G_H))
                else:
                    S_H = np.exp(-A_m)
                logS_T[:, m] = np.log(np.maximum(S_T, tiny))
                logS_H[m] = np.log(np.maximum(S_H, tiny))
                continue

        # -------- mixture aggregation: ratio of sums across draws --------
        # p_mix(T) = sum_m S_m(T_h) / sum_m S_m(T) = exp( logsumexp(logS_H) - logsumexp(logS_T, axis=1) )
        log_num = logsumexp(logS_H)  # scalar
        log_den = logsumexp(logS_T, axis=1)  # (L,)
        p_mix = np.exp(log_num - log_den)
        np.clip(p_mix, 0.0, 1.0, out=p_mix)

        # -------- optional bootstrap bands over draws --------
        bands: list[tuple[float, np.ndarray, np.ndarray]] = []
        if conf and bootstrap_samples > 0:
            rng = np.random.default_rng(random_state)
            B = bootstrap_samples
            samples = np.empty((B, L), dtype=float)
            for b in range(B):
                idx = rng.integers(0, N, size=N)
                log_num_b = logsumexp(logS_H[idx])
                log_den_b = logsumexp(logS_T[:, idx], axis=1)
                pb = np.exp(log_num_b - log_den_b)
                np.clip(pb, 0.0, 1.0, out=pb)
                samples[b, :] = pb
            for lvl in sorted(conf):
                qlo = (1 - lvl) * 50.0
                qhi = 100.0 - qlo
                lo = np.percentile(samples, qlo, axis=0)
                hi = np.percentile(samples, qhi, axis=0)
                bands.append((lvl, lo, hi))

        # -------- colors & plotting --------
        if colors is None:
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            main_color = cycle[0]
            band_palette = cycle[1:] or [cycle[0]]
            band_colors = [band_palette[i % len(bands)] for i in range(len(bands))]
        elif isinstance(colors, str):
            cmap_obj = plt.get_cmap(colors)
            n = 1 + len(bands)
            pos = np.linspace(0.15, 0.85, num=n)
            rgba = [cmap_obj(p) for p in pos]
            main_color = rgba[0]
            band_colors = rgba[1:]
        else:
            if len(colors) < 1:
                raise ValueError("colors list must have at least one color.")
            clist = [mcolors.to_rgba(c) for c in colors]
            main_color = clist[0]
            palette = clist[1:] or [clist[0]]
            band_colors = [palette[i % len(bands)] for i in range(len(bands))]

        for (lvl, lo, hi), c in zip(reversed(bands), reversed(band_colors)):
            self.ax.fill_between(date_nums, lo, hi, alpha=0.25, color=c, label=f"{int(lvl * 100)}% CI")

        method_label = {
            "exact": "Exact",
            "gompertz": "Gompertz",
            "two_exp": f"Two-exp (ρ={rho:g})",
            "gamma_moment": "Gamma-moment",
        }[method]
        the_label = f"{label} — {method_label} — {horizon}"
        self.ax.plot(date_nums, p_mix, lw=2, label=the_label, color=main_color)

    # --- analytic plotter matching the style of add_empirical ----------------------

    def add_analytic_parametric(
            self,
            parents_times_per_traj: List[Sequence[float]],
            draws: np.ndarray,  # shape (M,5): [R0, k, r, alpha, theta]
            color: str = 'C2',
            marker: str = '',
            label: str = 'Analytic (parametric; no-peek)',
            control_date: Optional["datetime"] = None,
    ) -> None:
        """
        Parameter-dependent analytic quiet-window extinction probability using only
        pre-cutoff information (no peeking past t_max), with **duplicates-encoded M_i**.

        **Input encoding (VERY IMPORTANT):**
        For trajectory m, `parents_times_per_traj[m]` must contain, for each parent
        with infection time t_i <= t_max, its infection time **at least once**, plus
        **one additional duplicate for each observed pre-cutoff child** of that parent.
        In other words, the multiplicity of a parent's time equals (1 + M_i).
        If two different parents have exactly the same time value, they will be
        merged by `np.unique` (rare; ignored as requested).

        Implements the formula
            P(B | A(T), data_at_tmax, Θ)
            = ∏_i [ (β + F(a_i + T)) / (β + 1) ]^{k + M_i },  with  β = k / R_post,
        where i ranges over parents (t_i <= t_max), a_i = t_max - t_i, and
        F is Gamma CDF with (alpha, theta).

        Parameters
        ----------
        parents_times_per_traj : list of sequences
            Per-trajectory arrays of duplicated parent infection times (days since start_date)
            as described above. Entries with t > t_max are ignored.
        draws : np.ndarray (M,5)
            Rows are [R0, k, r, alpha, theta] for each accepted trajectory.
            Post-cutoff R_post = R0 * r if controls are active at/ before t_max; else R_post = R0.
        color / marker / label : plotting style.
        control_date : datetime or None
            If provided and cutoff_date < control_date, we take R_post = R0 (controls not yet active).
            Otherwise R_post = R0 * r.

        Notes
        -----
        - This “no-peek” analytic conditions on the quiet window via F(a_i+T) and depends
          on (R,k,alpha,theta) through β and F. It will generally differ from an empirical
          estimator that *peeks* at post-cutoff seeds.
        """
        # --- validate inputs
        if draws.ndim != 2 or draws.shape[1] != 5:
            raise ValueError("draws must be shape (M,5): [R0, k, r, alpha, theta].")
        M = draws.shape[0]
        if len(parents_times_per_traj) != M:
            raise ValueError("parents_times_per_traj length must match draws.shape[0].")

        # --- day grid aligned with empirical routine
        cutoff_day = (self.cutoff_date - self.start_date).total_seconds() / 86400.0
        d0 = int(np.floor(cutoff_day))
        d1 = int(np.floor(self.T_run))
        if d0 > d1:
            return

        days = np.arange(d0, d1 + 1, dtype=int)
        T_grid = days - cutoff_day  # quiet-window length at each plotted day

        dates = [self.start_date + timedelta(days=int(d)) for d in days]
        date_nums = mdates.date2num(dates)

        # --- unpack draws and choose post-cutoff R
        R0 = draws[:, 0].astype(float)
        k_vals = draws[:, 1].astype(float)
        r_vals = draws[:, 2].astype(float)
        alpha_vals = draws[:, 3].astype(float)
        theta_vals = draws[:, 4].astype(float)

        use_controls = (control_date is None) or (self.cutoff_date >= control_date)
        R_post = (R0 * r_vals) if use_controls else R0

        # β = k / R_post; treat R_post <= 0 ⇒ β = +∞ (factor → 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            beta_vals = k_vals / R_post
        finite_beta = np.isfinite(beta_vals) & (beta_vals > 0.0)
        log_beta_plus_one = np.empty(M, dtype=float)
        log_beta_plus_one[finite_beta] = np.log1p(beta_vals[finite_beta])

        # --- build per-trajectory *unique* parents and their (1 + M_i) multiplicities
        # From duplicates-encoded times (filter to t_i <= t_max)
        parents_unique_times: List[np.ndarray] = []
        parents_mults: List[np.ndarray] = []  # multiplicity = 1 + M_i  (≥1 integers)
        for m in range(M):
            t_all = np.asarray(parents_times_per_traj[m], dtype=float)
            if t_all.size == 0:
                parents_unique_times.append(np.zeros(0, dtype=float))
                parents_mults.append(np.zeros(0, dtype=float))
                continue
            mask = (t_all <= cutoff_day)
            t_pre = t_all[mask]
            if t_pre.size == 0:
                parents_unique_times.append(np.zeros(0, dtype=float))
                parents_mults.append(np.zeros(0, dtype=float))
                continue
            unique_t, counts = np.unique(t_pre, return_counts=True)
            parents_unique_times.append(unique_t.astype(float))
            parents_mults.append(counts.astype(float))  # counts = 1 + M_i

        # --- compute the curve (average over trajectories of per-trajectory products)
        comp = np.empty_like(days, dtype=float)

        for ti, T in enumerate(T_grid):
            vals = []

            for m in range(M):
                t_par = parents_unique_times[m]
                mults = parents_mults[m]  # = 1 + M_i
                if t_par.size == 0:
                    vals.append(1.0)
                    continue

                # If β is not finite or ≤ 0, factor → 1 (for all parents)
                if not finite_beta[m]:
                    vals.append(1.0)
                    continue

                a_i = cutoff_day - t_par  # ages ≥ 0
                at = np.maximum(a_i + T, 0.0)  # guard negatives
                F_vec = gamma.cdf(at, a=alpha_vals[m], scale=theta_vals[m])  # vectorized

                beta = beta_vals[m]
                kval = k_vals[m]

                # exponent per parent: k + M_i = k + (mults - 1)
                exponents = kval + (mults - 1.0)

                # factor_i = ((β + F_i) / (β + 1))^{exponent_i}
                # compute in log-space for stability
                log_num = np.log(beta + F_vec)
                log_den = log_beta_plus_one[m]  # log(β + 1)
                log_prod = np.sum(exponents * (log_num - log_den))
                vals.append(float(np.exp(log_prod)))

            comp[ti] = float(np.median(vals)) if vals else 1.0

        # --- plot
        self.ax.plot(date_nums, comp, lw=1.7, marker=marker, label=label, color=color)
        self.ax.set_ylim(0.0, 1.02)
        # self.ax.set_ylabel("Extinction probability")
        # self.ax.grid(True)
        # self.fig.canvas.draw_idle()


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
        analytical: Tuple[np.ndarray | DrawCollector, list[list[float]] | ActiveSetSizeCollector],
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
        analytical: Optional[Tuple[np.ndarray | DrawCollector, list[list[float]] | ActiveSetSizeCollector]] = None,
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
        active_set = active_set if isinstance(active_set, list) else active_set.active_sets

        # ep.add_analytical(draws, active_set, None, colors=['C4'], label='analytical - ev', method='ev')
        # ep.add_analytical(draws, active_set, None, colors=['C5'], label='analytical - size_only', method='size_only')
        # ep.add_analytical3(draws, active_set, None, colors=['C6'], label='analytical3 - ev', method='ev', )

        # ep.add_analytical3(draws, active_set, conf=None, colors=['blue'], label='Analytical (EV)',
        #                    use_expected_compensation=True)
        # ep.add_analytical3(draws, active_set, conf=None, colors=['green'], label='Analytical (size-only)',
        #                    use_expected_compensation=False)

        # ep.add_analytical3(draws, active_set, conf=None, colors=['C5'], label='') # This just seems off.

        # ep.add_analytical_first_after(draws, active_set, conf=None, colors=['C5'], label='Analytical (med inf)',
        #                               horizon="infinite", aggregate='median')

        # no difference between number of bootstrap_samples.
        # ep.add_analytical_mixture(draws, active_set, conf=None, colors=['C4'], label='Analytical (0)',
        #                           bootstrap_samples=0)
        # ep.add_analytical_mixture(draws, active_set, conf=None, colors=['C5'], label='Analytical (3)',
        #                           bootstrap_samples=3)
        # ep.add_analytical_mixture(draws, active_set, conf=None, colors=['C6'], label='Analytical (7)',
        #                           bootstrap_samples=7)
        # ep.add_analytical_mixture(draws, active_set, conf=None, colors=['C7'], label='Analytical (15)',
        #                           bootstrap_samples=15)
        # ep.add_analytical_mixture(draws, active_set, conf=None, colors=['C8'], label='Analytical (30)',
        #                           bootstrap_samples=30)
        # ep.add_analytical_mixture(draws, active_set, conf=None, colors=['C9'], label='Analytical (50)',
        #                           bootstrap_samples=50)

        # ep.add_analytical_surrogate(draws, active_set, None, ['C4'], method='exact') # same as add_analytical_mixture
        # ep.add_analytical_surrogate(draws, active_set, None, ['C5'], method='gompertz')
        # ep.add_analytical_surrogate(draws, active_set, None, ['C6'], method='two_exp', rho=1.5, label="rho 1.5")
        # ep.add_analytical_surrogate(draws, active_set, None, ['C8'], method='two_exp', rho=3, label='rho 3')
        # ep.add_analytical_surrogate(draws, active_set, None, ['C9'], method='two_exp', rho=5, label='rho 5')
        # ep.add_analytical_surrogate(draws, active_set, None, ['C7'], method='gamma_moment')

        # ep.add_analytic_parametric(active_set, draws)

        # --- add_analytical (2 modes) ---
        # ep.add_analytical(draws, active_set, conf=None, colors=['C0'], label='Analytical — ev', method='ev')
        # ep.add_analytical(draws, active_set, conf=None, colors=['C1'], label='Analytical — size_only',
        #                   method='size_only')

        # --- add_analytical3 (6 modes) ---
        # ep.add_analytical3(draws, active_set, conf=None, colors=['C2'], label='Analytical3 — ev', method='ev')
        # ep.add_analytical3(draws, active_set, conf=None, colors=['C3'], label='Analytical3 — size_only',
        #                    method='size_only')
        # ep.add_analytical3(draws, active_set, conf=None, colors=['C4'], label='Analytical3 — ev_exp', method='ev_exp',
        #                    lambda_c=0.15)
        # ep.add_analytical3(draws, active_set, conf=None, colors=['C5'], label='Analytical3 — size_only_exp',
        #                    method='size_only_exp', lambda_c=0.15)
        # ep.add_analytical3(draws, active_set, conf=None, colors=['C6'], label='Analytical3 — ev_weibull',
        #                    method='ev_weibull', weibull_tau=7.0, weibull_nu=1.5)
        # ep.add_analytical3(draws, active_set, conf=None, colors=['C7'], label='Analytical3 — size_only_weibull',
        #                    method='size_only_weibull', weibull_tau=7.0, weibull_nu=1.5)

        # --- add_analytical_first_after (4 modes: horizon × aggregate) ---
        # ep.add_analytical_first_after(draws, active_set, conf=None, colors=['C8'], label='First-after — finite, median',
        #                               horizon='finite', aggregate='median')
        # ep.add_analytical_first_after(draws, active_set, conf=None, colors=['C9'],
        #                               label='First-after — infinite, median', horizon='infinite', aggregate='median')
        # ep.add_analytical_first_after(draws, active_set, conf=None, colors=['C10'],
        #                               label='First-after — finite, mean', horizon='finite', aggregate='mean')
        # ep.add_analytical_first_after(draws, active_set, conf=None, colors=['C11'],
        #                               label='First-after — infinite, mean', horizon='infinite', aggregate='mean')

        # --- add_analytical_mixture (2 modes: horizon) ---
        # ep.add_analytical_mixture(draws, active_set, conf=None, colors=['C12'], label='Mixture — finite',
        #                           horizon='finite', bootstrap_samples=0)
        # ep.add_analytical_mixture(draws, active_set, conf=None, colors=['C13'], label='Mixture — infinite',
        #                           horizon='infinite', bootstrap_samples=0)

        # --- add_analytical_surrogate (4 methods, infinite horizon) ---
        # ep.add_analytical_surrogate(draws, active_set, conf=None, colors=['C14'], label='Surrogate — exact',
        #                             method='exact', horizon='infinite')
        # ep.add_analytical_surrogate(draws, active_set, conf=None, colors=['C15'], label='Surrogate — gompertz',
        #                             method='gompertz', horizon='infinite')
        # ep.add_analytical_surrogate(draws, active_set, conf=None, colors=['C16'],
        #                             label='Surrogate — two_exp (ρ=3)', method='two_exp', horizon='infinite', rho=3)

        # --- add_analytic_parametric (no-peek parametric) ---
        ep.add_analytic_parametric(active_set, draws, color='k', label='No-peek parametric (duplicates)')

        # --- final cosmetics ---
        ep.ax.set_ylim(0.0, 1.02)
        ep.ax.set_ylabel("Probability (no further infections)")
        ep.ax.grid(True, alpha=0.3)
        ep.ax.legend(ncol=2, fontsize=8, frameon=False)
        ep.fig.tight_layout()

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
