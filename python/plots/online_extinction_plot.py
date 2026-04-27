from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt

from python.plots.misc import SnapshotResult
from python.plots.style import _use_style, _segment_offsets, _legend_dedupe


def plot_rb_online_two_pane_shifted(results: Sequence[SnapshotResult], *, show_next_dot=True, ylim=(0, 1)):
    sty = _use_style(None)
    if not results:
        return
    offsets = _segment_offsets(results)
    fig, axes = plt.subplots(1, 2, figsize=sty.fig_pair, sharex=False, sharey=True, constrained_layout=True)
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
            (line,) = ax.plot(x, y, lw=sty.lw_rb)
            if show_next_dot and (r.next_T is not None) and (r.T_grid[0] <= r.next_T <= r.T_grid[-1]):
                xd = x0 + r.next_T
                yd = float(np.interp(r.next_T, r.T_grid, y))
                colr = line.get_color()
                ax.scatter([xd], [yd], s=sty.dot_area, facecolors=colr, edgecolors="white", linewidths=0.9,
                           zorder=line.get_zorder() + 1)
        if col == 1:
            _legend_dedupe(ax)
    for a in axes:
        a.minorticks_on()
    plt.show()
