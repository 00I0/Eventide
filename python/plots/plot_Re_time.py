from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt

from python.plots.misc import SnapshotResult
from python.plots.style import _use_style, _segment_offsets, _legend_dedupe


def _first_post_after_tstar_per_draw(res: SnapshotResult) -> np.ndarray:
    ts = float(res.t_star)
    out = np.empty(len(res.infection_times_2d), dtype=float)
    for i, traj in enumerate(res.infection_times_2d):
        t = np.asarray(traj, float)
        rel = t - ts
        rel = rel[rel > 0.0]
        out[i] = rel[0] if rel.size else np.inf
    return out


def _summarize(vals: np.ndarray, band: tuple[float, float], agg="median") -> tuple[float, float, float]:
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    qlo, qhi = np.quantile(vals, band)
    mid = float(np.median(vals)) if agg == "median" else (float(np.mean(vals)) if agg == "mean" else float(agg(vals)))
    return mid, float(qlo), float(qhi)


def plot_timepath_Re(
        results: Sequence[SnapshotResult],
        *,
        step: float = 0.25,
        band: tuple[float, float] = (0.2, 0.8),
        summary: str = "median",
        draw_verticals: bool = True,
        clip_Re_max: float = 3.0,
        ensure_upward_jumps: bool = False,
):
    sty = _use_style(None)
    if not results:
        return

    offsets = _segment_offsets(results)
    payload = []
    for res in results:
        R0 = res.draws_array[:, 0]
        r = res.draws_array[:, 2]
        Re = R0 * r
        first_post = _first_post_after_tstar_per_draw(res)
        payload.append((res, Re, first_post))

    all_x, all_mid, all_lo, all_hi, jump_locs = [], [], [], [], []
    prev_end_mid = None
    tiny = 1e-9

    for (res, Re, first_post), x0 in zip(payload, offsets):
        if res.next_T is not None:
            T_end = float(res.next_T);
            open_end = True
        else:
            T_end = 20;
            open_end = False
        if T_end <= 0:
            continue

        if open_end:
            n_steps = max(1, int(np.floor((T_end - tiny) / step)) + 1)
            T_eval = np.linspace(0.0, max(0.0, T_end - tiny), n_steps)
        else:
            T_eval = np.arange(0.0, T_end + tiny, step)

        mids, los, his = [], [], []
        for j, T in enumerate(T_eval):
            vals = Re if j == 0 else Re[first_post > T]
            m, lo, hi = _summarize(vals, band, agg=summary)
            if ensure_upward_jumps and j == 0 and prev_end_mid is not None \
                    and np.isfinite(prev_end_mid) and np.isfinite(m) and m < prev_end_mid:
                m = prev_end_mid
            if np.isfinite(m):  m = float(np.clip(m, 0.0, clip_Re_max))
            if np.isfinite(lo): lo = float(np.clip(lo, 0.0, clip_Re_max))
            if np.isfinite(hi): hi = float(np.clip(hi, 0.0, clip_Re_max))
            mids.append(m);
            los.append(lo);
            his.append(hi)

        x = x0 + T_eval
        all_x.append(x);
        all_mid.append(np.asarray(mids))
        all_lo.append(np.asarray(los));
        all_hi.append(np.asarray(his))
        if len(mids):
            prev_end_mid = mids[-1]
        if res.next_T is not None:
            jump_locs.append(x0 + float(res.next_T))

    X = np.concatenate(all_x) if all_x else np.array([])
    MID = np.concatenate(all_mid) if all_mid else np.array([])
    LO = np.concatenate(all_lo) if all_lo else np.array([])
    HI = np.concatenate(all_hi) if all_hi else np.array([])

    fig, ax = plt.subplots(figsize=sty.fig_pair, constrained_layout=True, dpi=sty.dpi)
    ax.set_title(r"Effective reproduction number given no unobserved infections")
    ax.set_xlabel(r"Elapsed days since first $t_\star$")
    ax.set_ylabel(r"$rR_0$")

    if X.size:
        good = np.isfinite(MID)
        (ln,) = ax.plot(X[good], MID[good], lw=sty.lw_emp, color=sty.palette["INTERP"])
        color = ln.get_color()
        ax.fill_between(X[good], LO[good], HI[good], alpha=0.22, color=color,
                        label=f"central {int((band[1] - band[0]) * 100)}% band")
        ax.plot(X[good], MID[good], color=color, lw=sty.lw_emp, label="median")

    if draw_verticals:
        for xj in jump_locs:
            ax.axvline(xj, linestyle="--", linewidth=1.1, alpha=0.5, color=sty.palette["ANA"])

    _legend_dedupe(ax)

    ax.set_ylim(0, clip_Re_max)
    if X.size:
        ax.set_xlim(0.0, float(np.nanmax(X)) + step)

    ax.minorticks_on()
    plt.show()
