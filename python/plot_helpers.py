import logging
from datetime import timedelta
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from fitter import Fitter


def create_histograms(hist_collectors, target_ratio: float = 16 / 9, subplot_size: float = 5.0):
    """
    Plot any number of 1D or 2D histograms in a grid whose overall aspect
    ratio is as close as possible to `target_ratio` (default 16:9).

    Args:
        hist_collectors (list): List of histogram-collector objects.
        target_ratio (float, optional): Desired figure width/height ratio.
        subplot_size (float, optional): Size (in inches) of each square subplot.
    """
    n = len(hist_collectors)

    best = None
    for cols in range(1, n + 1):
        rows = ceil(n / cols)
        ratio = cols / rows
        score = abs(ratio - target_ratio)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    _, rows, cols = best

    fig, axes = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size * rows))
    axes = axes.flatten()

    for ax, collector in zip(axes, hist_collectors):
        data = collector.numpy()
        title = collector.name

        if data.ndim == 1:
            lo, hi = collector.range
            bins = data.size
            edges = np.linspace(lo, hi, bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.bar(centers, data, width=(hi - lo) / bins, align='center')
            ax.set_xlabel(title, fontsize=14)
            ax.set_xlim(lo, hi)

        elif data.ndim == 2:
            (lox, hix), (loy, hiy) = collector.range
            extent = [lox, hix, loy, hiy]
            ax.imshow(data.T, aspect='auto', origin='lower', extent=extent)
            ax.set_xlabel(collector.var_names[0], fontsize=14)
            ax.set_ylabel(collector.var_names[1], fontsize=14)
            ax.set_xlim(lox, hix)
            ax.set_ylim(loy, hiy)

        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(False)

    # 4) Hide unused axes
    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


def fit_and_plot_histograms(hist_collectors, distributions=None, top_n=5, target_ratio: float = 16 / 9,
                            subplot_size: float = 4.0, timeout: int = 60, max_workers=12):
    """
    For each 1D histogram in `hist_collectors`, reconstruct raw samples,
    fit the given distributions (or all if None), and plot the histogram
    plus the top-N fitted PDFs in a grid that approximates `target_ratio`.

    Args:
        hist_collectors (list): objects with attributes:
            - numpy(): returns a 1D counts array
            - range: (lo, hi) tuple
            - name: string for title/label
        distributions (list or None): list of scipy.stats names to fit,
            or None to try them all.
        top_n (int): how many of the best-fit PDFs to overlay.
        target_ratio (float): desired figure width/height ratio.
        subplot_size (float): size (inches) of each subplot.
        timeout (int): seconds before giving up on a single fit.
    """
    logging.getLogger("fitter.fitter").setLevel(logging.CRITICAL)
    n = len(hist_collectors)

    # 1) choose best (rows, cols) to match target_ratio
    best = None
    for cols in range(1, n + 1):
        rows = ceil(n / cols)
        score = abs((cols / rows) - target_ratio)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    _, rows, cols = best

    fig, axes = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, collector in zip(axes, hist_collectors):
        counts = collector.numpy().astype(int)
        lo, hi = collector.range
        bins = len(counts)

        # 2) reconstruct raw samples
        edges = np.linspace(lo, hi, bins + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        data = np.repeat(mids, counts)

        # 3) fit distributions in parallel
        f = Fitter(
            data,
            distributions=distributions,
            bins=bins,
            timeout=timeout
        )
        f.fit(progress=False, max_workers=max_workers, prefer="processes")

        # 4) rank by SSE and grab top_n names
        df = f.df_errors.reset_index().rename(columns={'index': 'distribution'}).sort_values(by="sumsquare_error")

        df.to_csv(f'distributions_fitter_{collector.name}_2000.csv', index=False)
        top_dists = df['distribution'].iloc[:top_n].tolist()

        # 5) plot histogram + best PDFs
        ax.hist(data, bins=edges, density=True, alpha=0.4, label="Empirical")
        for name in top_dists:
            ax.plot(f.x, f.fitted_pdf[name], lw=2, label=name)
        ax.set_title(collector.name, fontsize=14)
        ax.set_xlim(lo, hi)
        ax.legend(fontsize="small")
        ax.grid(False)

    # hide any unused subplots
    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
    plt.savefig('../img/fitted.png')


def create_end_of_outbreak_plot(matrix, cutoff_day, start_date, T_run):
    days = np.arange(cutoff_day, T_run + 1)
    completion_prob = np.empty_like(days, dtype=float)

    for idx, d in enumerate(days):
        # for any final‐time index > d, check if the first re‐inf after d is also > d
        mask_after = np.arange(matrix.shape[1]) > d
        resurge = matrix[d + 1:, mask_after].sum()  # still “alive” (resurge) after day d
        valid = matrix[:, mask_after].sum()  # all sims that reach beyond day d
        completion_prob[idx] = 1.0 - (resurge / valid)

    # map day indices back to actual dates
    dates = [start_date + timedelta(days=int(d)) for d in days]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(dates, completion_prob, marker='o', lw=1.5, label="Completion prob.")

    threshold = 0.9
    ax.axhline(threshold, color='red', ls='--', lw=1)
    cross_idx = np.argmax(completion_prob >= threshold)
    if completion_prob[cross_idx] >= threshold:
        d0, p0 = dates[cross_idx], completion_prob[cross_idx]
        ax.axvline(d0, color='red', ls='--', lw=1)
        ax.scatter([d0], [p0], zorder=5)

    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_title("Estimated Completion Probability Over Time")
    ax.grid(True, ls=':')
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.show()
