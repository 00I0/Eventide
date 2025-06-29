from datetime import timedelta
from math import ceil

import matplotlib.pyplot as plt
import numpy as np


def create_histograms(hist_collectors, target_ratio: float = 16 / 9, subplot_size: float = 4.0):
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


def create_end_of_outbreak_plot(matrix, cutoff_day, start_date, T_run):
    days = np.arange(cutoff_day, T_run)
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
