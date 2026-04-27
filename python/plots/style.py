# style_els.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Sequence

import matplotlib as mpl

from python.plots.misc import SnapshotResult

# ---- Figure width helpers ----------------------------------------------------
PT_PER_IN = 50  # TeX points → inches


def figure_size_from_tex(width_pt: float, *, fraction: float = 1.0, aspect: float = 0.62) -> Tuple[float, float]:
    """Return (width, height) in inches given a LaTeX width in points."""
    w_in = (width_pt / PT_PER_IN) * fraction
    h_in = w_in * aspect
    return (w_in, h_in)


ELS_5P_SINGLE_PT = 252.0
ELS_5P_DOUBLE_PT = 522.0


# ---- Style -------------------------------------------------------------------
@dataclass
class Style:
    # Sizing
    base_font: int = 10  # 5p reads naturally around 9 pt
    dpi: int = 300
    use_tex: bool = True  # recommend True for perfect font match
    serif_family: str = "serif"

    # Figure sizes (inches) at native publication widths
    fig_single: Tuple[float, float] = field(default_factory=lambda:
    figure_size_from_tex(ELS_5P_SINGLE_PT, aspect=0.62))
    fig_pair: Tuple[float, float] = field(default_factory=lambda:
    figure_size_from_tex(ELS_5P_DOUBLE_PT, aspect=0.4))

    # Default (two-pane)
    figsize: Tuple[float, float] = field(default_factory=lambda:
    figure_size_from_tex(ELS_5P_DOUBLE_PT, aspect=0.4))

    # Typography
    title_size: Optional[int] = None
    label_size: Optional[int] = None
    tick_size: Optional[int] = None
    title_weight: str = "regular"
    label_weight: str = "regular"
    annotation_size: int = 9

    # Axes & ticks (print-friendly)
    show_grid: bool = False
    grid_alpha: float = 0.12
    grid_linewidth: float = 0.6
    inward_ticks: bool = True
    axes_linewidth: float = 0.9
    tick_major_length: float = 5.0
    tick_minor_length: float = 2.5
    tick_major_width: float = 0.9
    tick_minor_width: float = 0.7

    # Lines / markers
    lw_emp: float = 1.0
    lw_ana: float = 1.0
    lw_rb: float = 1.0
    lw_default: float = 1.0
    marker_size: float = 4.5
    dot_area: float = 34.0

    # Dashing
    dash_empirical: Tuple[int, Tuple[float, ...]] = (0, (3.4, 1.6, 1.2, 1.6))

    # Color cycle (color-blind friendly; okabe-ito + neutrals)
    cycle: List[str] = field(default_factory=lambda: [
        "#0072B2",  # blue (empirical)
        "#E69F00",  # orange/amber (analytic)
        "#222222",  # near-black (RB)
        "#56B4E9",  # sky blue (alt)
        "#009E73",  # bluish green
        "#CC79A7",  # purple
        "#999999",  # grey
        "#D55E00",  # vermillion (alerts/points)
    ])

    # Named palette for explicit roles across the whole paper
    palette: Dict[str, str] = field(default_factory=lambda: dict(
        EMP="#0072B2",
        ANA="#E69F00",
        RB="#222222",
        SIM="#E69F00",
        HPD="#009E73",
        QUIET="#E9C87A",
        BAND="#A6C4EB",
        POINTS="#BB4430",
        PAREN="#6B6B6B",
        INTERP="#0072B2",
    ))

    tight_pad: float = 0.12
    _rc: Dict = field(default_factory=dict, init=False, repr=False)

    def build_rc(self) -> Dict:
        base = self.base_font
        title = self.title_size if self.title_size is not None else base + 3
        label = self.label_size if self.label_size is not None else base + 0
        tick = self.tick_size if self.tick_size is not None else base - 0

        # When not using LaTeX, pick a Times-like math font so it still blends in
        math_fontset = "stix" if not self.use_tex else "cm"

        rc = {
            # Fonts
            "text.usetex": bool(self.use_tex),
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}" if self.use_tex else "",
            "mathtext.fontset": math_fontset,
            "font.family": self.serif_family,
            "font.size": base,
            "axes.titlesize": title,
            "axes.titleweight": self.title_weight,
            "axes.labelsize": label,
            "axes.labelweight": self.label_weight,
            "xtick.labelsize": tick,
            "ytick.labelsize": tick,
            "axes.unicode_minus": False,

            # Spines / ticks
            "axes.linewidth": self.axes_linewidth,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "xtick.direction": "in" if self.inward_ticks else "out",
            "ytick.direction": "in" if self.inward_ticks else "out",
            "xtick.major.size": self.tick_major_length,
            "ytick.major.size": self.tick_major_length,
            "xtick.minor.size": self.tick_minor_length,
            "ytick.minor.size": self.tick_minor_length,
            "xtick.major.width": self.tick_major_width,
            "ytick.major.width": self.tick_major_width,
            "xtick.minor.width": self.tick_minor_width,
            "ytick.minor.width": self.tick_minor_width,

            # Lines / markers
            "lines.linewidth": self.lw_default,
            "lines.markersize": self.marker_size,

            # Grid
            "axes.grid": bool(self.show_grid),
            "grid.alpha": self.grid_alpha,
            "grid.linewidth": self.grid_linewidth,

            # Legend
            "legend.frameon": False,
            "legend.framealpha": 0.9,
            "legend.handlelength": 2.4,
            "legend.borderaxespad": 0.5,

            # Save / figure
            "pdf.fonttype": 42,  # embed TrueType
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "figure.figsize": list(self.figsize),
            "figure.dpi": self.dpi,
            "figure.autolayout": False,

            # Color cycle
            "axes.prop_cycle": mpl.cycler(color=self.cycle),
        }
        self._rc = rc
        return rc


# Global holder
_CURRENT_STYLE: Optional[Style] = None


def set_style(column: str = "double",
              *,
              base_font: int = 9,
              use_tex: bool = True,
              width_pt: Optional[float] = None,
              fraction: float = 1.0,
              aspect: float = 0.52,
              show_grid: bool = False,
              dpi: int = 300) -> Style:
    """
    Configure a style that blends with elsarticle 5p.
    - column: 'single' or 'double'
    - width_pt/fraction/aspect: override figure size from LaTeX widths if you want.
    """
    if width_pt is None:
        width_pt = ELS_5P_SINGLE_PT if column == "single" else ELS_5P_DOUBLE_PT
    figsize = figure_size_from_tex(width_pt, fraction=fraction, aspect=aspect)

    style = Style(
        figsize=figsize,
        base_font=base_font,
        use_tex=use_tex,
        show_grid=show_grid,
        dpi=dpi
    )
    mpl.rcParams.update(style.build_rc())
    global _CURRENT_STYLE
    _CURRENT_STYLE = style
    return style


def _use_style(style: Optional[Style]) -> Style:
    global _CURRENT_STYLE
    if style is None:
        _CURRENT_STYLE = _CURRENT_STYLE or set_style()
        mpl.rcParams.update(_CURRENT_STYLE._rc)
        return _CURRENT_STYLE
    mpl.rcParams.update(style._rc or style.build_rc())
    return style


def _segment_offsets(results: Sequence[SnapshotResult]) -> List[float]:
    offs, acc = [], 0.0
    for r in results:
        offs.append(acc)
        if r.next_T is not None:
            acc += float(r.next_T)
    return offs


def _legend_dedupe(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            H.append(h)
            L.append(l)
    if L:
        ax.legend(H, L, loc="upper left", frameon=False)


def days_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0


if __name__ == '__main__':
    print(figure_size_from_tex(ELS_5P_SINGLE_PT, aspect=0.62))
    print(figure_size_from_tex(ELS_5P_DOUBLE_PT, aspect=0.4))
