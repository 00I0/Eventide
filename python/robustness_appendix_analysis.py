from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from python.eventide import (
    ActiveSetSizeCollector,
    DrawCollector,
    IndexOffspringCriterion,
    InfectionTimeCollector,
    ParameterChangePoint,
    Parameters,
    Scenario,
    Simulator,
)
from python.on_the_fly import (
    rb_cond_components_post,
    rb_draws_cond_from_components,
    rb_draws_uncond_full_to_grid,
    rao_blackwell_uncond_over_post_full,
)
from python.optimize_acceptance_windows import build_acceptance_inequalities
from python.plots.misc import SnapshotResult

# ============================================================
# Configuration
# ============================================================

OBS_POINTS: List[Tuple[datetime, int]] = [
    (datetime(2018, 11, 2), 1),
    (datetime(2018, 11, 22), 2),
    (datetime(2018, 11, 28), 3),
    (datetime(2018, 12, 5), 4),
    (datetime(2018, 12, 11), 5),
    (datetime(2018, 12, 18), 6),
    (datetime(2018, 12, 26), 5),
    (datetime(2019, 1, 3), 4),
    (datetime(2019, 1, 10), 2),
    (datetime(2019, 1, 18), 1),
    (datetime(2019, 1, 25), 1),
]

SNAPSHOT_ID = len(OBS_POINTS)  # only the final snapshot is used here

T_RUN = (OBS_POINTS[-1][0] - OBS_POINTS[0][0]).days + 40
T_GRID = np.arange(0.0, 40.0 + 1e-9, 0.5)
MAX_CASES = 100
MAX_WORKERS = 13
NUM_TRAJECTORIES = 1_000_000_000_000_000
CHUNK_SIZE = 50_000
MIN_REQUIRED = 1_000
H = 0.2
H_PAD = 10.0
CONTROL_START = datetime(2019, 1, 1)

OUTPUT_DIR = Path("robustness_outputs/hanta2")

COLORS = {
    "baseline": "#111111",
    "prior_narrow": "#0072B2",
    "prior_wide": "#D55E00",
    "bio_loose": "#009E73",
    "bio_strict": "#CC79A7",
    "accept_loose": "#56B4E9",
}

BASELINE_PRIORS: Dict[str, Tuple[float, float]] = {
    "R0": (0.0, 7.5),
    "k": (0.2, 30.0),
    "r": (0.01, 0.99),
    "alpha": (0.01, 30.0),
    "theta": (0.01, 20.0),
}

BASELINE_REQUIREMENTS: List[str] = [
    "R0 * r < 3",
    "alpha * theta > 15",
    "alpha * theta < 60",
    "(k / (k + R0)) ^ k > 0.25",
    "(k / (k + R0 * r)) ^ k > 0.35",
    "((R0 * r) ^ (1 / alpha) - 1) / theta < 0.05",
]

OPTIMIZER_OUTPUT_FIELDS: Tuple[str, ...] = (
    "objective",
    "accepted",
    "sigma_days",
    "beta",
    "neighbor_weight",
    "grid_step_days",
    "min_seg_days",
    "kmax",
    "baseline_p",
    "alpha",
    "h_max",
    "eps_share",
    "include_gap_windows",
    "include_union_windows",
    "max_unions_to_keep",
    "gap_scale",
    "include_global_total",
)


def _builder_kwargs_from_optimizer_row(row: Sequence[Any]) -> Dict[str, Any]:
    if len(row) == len(OPTIMIZER_OUTPUT_FIELDS):
        values = dict(zip(OPTIMIZER_OUTPUT_FIELDS, row))
    elif len(row) == len(OPTIMIZER_OUTPUT_FIELDS) - 2:
        # Allow rows that contain only the acceptance-kwargs payload and omit
        # the leading objective / accepted columns.
        values = dict(zip(OPTIMIZER_OUTPUT_FIELDS[2:], row))
    else:
        raise ValueError(
            f"Unexpected optimizer row length {len(row)}; expected "
            f"{len(OPTIMIZER_OUTPUT_FIELDS)} or {len(OPTIMIZER_OUTPUT_FIELDS) - 2}."
        )
    return {
        "sigma_days": float(values["sigma_days"]),
        "beta": float(values["beta"]),
        "neighbor_weight": float(values["neighbor_weight"]),
        "grid_step_days": float(values["grid_step_days"]),
        "min_seg_days": float(values["min_seg_days"]),
        "kmax": int(values["kmax"]),
        "baseline_p": float(values["baseline_p"]),
        "alpha": float(values["alpha"]),
        "h_max": float(values["h_max"]),
        "eps_share": float(values["eps_share"]),
        "include_gap_windows": bool(values["include_gap_windows"]),
        "include_union_windows": bool(values["include_union_windows"]),
        "max_unions_to_keep": int(values["max_unions_to_keep"]),
        "gap_scale": float(values["gap_scale"]),
        "include_global_total": bool(values["include_global_total"]),
    }


SNAPSHOT_BUILDER_KWARGS_BY_M: Dict[int, Dict[str, Any]] = {
    1: _builder_kwargs_from_optimizer_row((
        3.876986677334013, 0.8536521235132486, 0.640374898332219, 0.10599357413613672, 0.7285707803290452, 2,
        0.1843105102607828, 0.39988138399181644, 0.4231284739255994, 0.09178407307432239, True, False, 0,
        0.8016084430512683, False,
    )),
    2: _builder_kwargs_from_optimizer_row((
        1.8445791914323304, 0.8816601601104272, 1.2102294660275816, 0.43546831487101323, 3.1987225675922053, 6,
        0.28376740693924385, 0.09799519411618896, 0.8185162430576272, 0.06512535606974085, True, True, 0,
        0.10186318097771685, True,
    )),
    3: _builder_kwargs_from_optimizer_row((
        7.81309491615046, 0.9578135534988547, 1.8279768796531257, 0.16046667759248917, 4.966058815762482, 2,
        0.06865373864095785, 0.010554750277145204, 0.1367819342609044, 0.07117480127990024, False, True, 6,
        0.8705002095780249, False,
    )),
    4: _builder_kwargs_from_optimizer_row((
        7.230142046451505, 0.5000307517236018, 2.034541151376431, 0.31096993492248015, 4.678181054040574, 2,
        0.1763631488167721, 0.04982099599959151, 0.10022453373717369, 0.0005548428848050443, False, True, 3,
        0.8919467963913615, True,
    )),
    5: _builder_kwargs_from_optimizer_row((
        5.896779157840167, 0.8894045810357124, 2.8366104994965755, 0.7448871585883066, 1.609828560076077, 3,
        0.048728197753616934, 0.09354187205294318, 0.28584425154799786, 0.03407536361980045, True, True, 3,
        0.11185380746783324, False,
    )),
    6: _builder_kwargs_from_optimizer_row((
        6.106501326327924, 0.5256891135165026, 1.3672741944106894, 0.7274105187796782, 1.14535435555878, 3,
        0.06682068388674232, 0.013870087712047747, 0.249889783048155, 0.032635876707872234, True, True, 6,
        0.8892829872491416, True,
    )),
    7: _builder_kwargs_from_optimizer_row((
        7.116888769886009, 0.7289981655377629, 0.4603085602769692, 0.3689880093118396, 4.7130775585359865, 2,
        0.17018271443712157, 0.10615833482828443, 0.5023157446544768, 0.0886387104281635, True, True, 5,
        0.7515885712089204, True,
    )),
    8: _builder_kwargs_from_optimizer_row((
        7.618148264912112, 0.6639606520497713, 0.7983152483309041, 0.30276960264247754, 4.030559854773017, 2,
        0.2859517471439936, 0.06965407744112657, 0.552419107409047, 0.022263962523991456, True, True, 3,
        0.3256759640980279, False,
    )),
    9: _builder_kwargs_from_optimizer_row((
        4.830514211689886, 0.8498009781272136, 0.2341760492678853, 0.47352688279069455, 2.1015806529196013, 3,
        0.04847992329718288, 0.3069766208772683, 0.9704615608663967, 0.000396162152668443, True, True, 4,
        0.6344008608356021, True,
    )),
    10: _builder_kwargs_from_optimizer_row((
        6.353369405193468, 0.7434956147057368, 0.1281014391336382, 0.32655880245142344, 4.985489866806982, 4,
        0.06579324087694745, 0.29314687776231696, 0.1532076599122092, 0.08214825792562591, False, False, 3,
        0.12278422433810095, True,
    )),
    11: _builder_kwargs_from_optimizer_row((
        7.422983570150037, 0.989262527201425, 2.14274118060158, 0.12442459103366947, 0.5000857976034738, 4,
        0.024567889698469254, 0.3979365013171724, 0.8513333723779887, 0.0682294608729182, False, False, 5,
        0.8999821695409332, True,
    )),
}

# Baseline acceptance settings use the final-snapshot optimizer configuration.
BASELINE_BUILDER_KWARGS: Dict[str, Any] = dict(SNAPSHOT_BUILDER_KWARGS_BY_M[SNAPSHOT_ID])


@dataclass(frozen=True)
class Spec:
    key: str
    title: str
    prior_overrides: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    requirements: Optional[Sequence[str]] = None
    builder_overrides: Mapping[str, Any] = field(default_factory=dict)


def _scale_builder(base: Mapping[str, Any], factor: float) -> Dict[str, Any]:
    out = dict(base)
    for name in ("baseline_p", "alpha", "h_max", "eps_share", "gap_scale"):
        if name in out:
            out[name] = float(out[name]) * factor
    return out


# -----------------------------------------------------------------
# Replace these with the final five perturbations of the appendix.
# The current choices are reasonable defaults only.
# -----------------------------------------------------------------


SPECS: List[Spec] = [
    Spec("baseline", "Alapbeállítás"),
    Spec(
        "prior_narrow",
        "Szűkebb prior",
        prior_overrides={
            "R0": (0.25, 6.0),
            "k": (0.1, 24.0),
            "alpha": (0.05, 24.0),
            "theta": (0.05, 16.0),
        },
    ),
    Spec(
        "prior_wide",
        "Tágabb prior",
        prior_overrides={
            "R0": (0.00, 9.0),
            "k": (0.01, 36.0),
            "alpha": (0.01, 36.0),
            "theta": (0.01, 24.0),
        },
    ),

    Spec(
        "bio_loose",
        "Enyhébb biológiai szűrés",
        requirements=[
            "R0 * r < 3.3",
            "alpha * theta > 13.5",
            "alpha * theta < 66",
            "(k / (k + R0)) ^ k > 0.23",
            "(k / (k + R0 * r)) ^ k > 0.32",
            "((R0 * r) ^ (1 / alpha) - 1) / theta < 0.06",
        ],
    ),

    Spec(
        "bio_strict",
        "Szigorúbb biológiai szűrés",
        requirements=[
            "R0 * r < 2.7",
            "alpha * theta > 16.5",
            "alpha * theta < 54",
            "(k / (k + R0)) ^ k > 0.27",
            "(k / (k + R0 * r)) ^ k > 0.38",
            "((R0 * r) ^ (1 / alpha) - 1) / theta < 0.045",
        ],
    ),

    Spec(
        "accept_loose",
        "Enyhébb elfogadási feltételek",
        builder_overrides=_scale_builder(BASELINE_BUILDER_KWARGS, 1.25),
    ),
]


# ============================================================
# Styling
# ============================================================

def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ============================================================
# Helpers
# ============================================================

def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(override)
    return out


def make_parameters(spec: Spec) -> Parameters:
    bounds = merge_dicts(BASELINE_PRIORS, spec.prior_overrides)
    pars = Parameters(
        R0=bounds["R0"],
        k=bounds["k"],
        r=bounds["r"],
        alpha=bounds["alpha"],
        theta=bounds["theta"],
    )
    reqs = list(BASELINE_REQUIREMENTS if spec.requirements is None else spec.requirements)
    for req in reqs:
        pars = pars.require(req)
    return pars


def _cast_accept_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    if "kmax" in out:
        out["kmax"] = int(round(float(out["kmax"])))
    if "max_unions_to_keep" in out:
        out["max_unions_to_keep"] = int(round(float(out["max_unions_to_keep"])))
    for name in ("include_gap_windows", "include_union_windows", "include_global_total"):
        if name in out:
            out[name] = bool(out[name])
    return out


def _scenario_controls_starting(*, simulation_start: datetime, control_start: datetime) -> Scenario:
    cps: List[ParameterChangePoint] = [ParameterChangePoint("r", simulation_start, "1.0")]
    if control_start > simulation_start:
        cps.append(ParameterChangePoint("r", control_start))
    return Scenario(cps)


def _run_final_snapshot(
        *,
        obs_points: Sequence[Tuple[datetime, int]],
        pars: Parameters,
        builder_kwargs: Mapping[str, Any],
        num_trajectories: int,
        chunk_size: int,
        T_run: int,
        max_cases: int,
        max_workers: int,
        T_grid: np.ndarray,
        h: float,
        H_pad: float,
        min_required: Optional[int],
        control_start: datetime,
) -> SnapshotResult:
    obs_sorted = sorted(list(obs_points), key=lambda x: x[0])
    m = len(obs_sorted)
    sim_start = min(t for t, _ in obs_sorted)
    criteria = [IndexOffspringCriterion(3, 7)] + build_acceptance_inequalities(
        obs_points=obs_sorted,
        simulation_start=sim_start,
        **_cast_accept_kwargs(builder_kwargs),
    )

    collectors = [
        draws := DrawCollector(),
        active_set := ActiveSetSizeCollector(obs_sorted[m - 1][0]),
        infection_times := InfectionTimeCollector(),
    ]
    scenario = _scenario_controls_starting(simulation_start=sim_start, control_start=control_start)
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
    sim.run()

    infection_times_2d = list(infection_times.infection_times)
    stopped_pairs = active_set.active_sets
    draws_array = np.asarray(draws, dtype=float)
    if draws_array.ndim == 1 and draws_array.size:
        draws_array = draws_array.reshape(1, -1)
    R0s, ks, rs, alphas, thetas = draws_array.T
    t_star = float((active_set.collection_date - sim.start_date).days)
    control_day = float((control_start - sim.start_date).days)

    T_fine, p_uncond_mean_fine, g_uncond_fine = rao_blackwell_uncond_over_post_full(
        infection_times_2d,
        stopped_pairs,
        R0s,
        rs,
        ks,
        alphas,
        thetas,
        T_max=float(T_grid[-1]),
        t_star=t_star,
        h=h,
        H_pad=H_pad,
        control_day=control_day,
    )
    p_uncond_mean = np.interp(T_grid, T_fine, p_uncond_mean_fine)
    p_uncond_draws = rb_draws_uncond_full_to_grid(T_fine, g_uncond_fine, T_grid)

    g_cond_inf, g_cond_quiet = rb_cond_components_post(
        infection_times_2d,
        stopped_pairs,
        R0s,
        rs,
        ks,
        alphas,
        thetas,
        T_grid,
        t_star,
        h=h,
        control_day=control_day,
    )
    p_cond_mean = (g_cond_inf.mean() / g_cond_quiet.mean(axis=0)) if g_cond_quiet.size else np.full_like(T_grid, np.nan)
    p_cond_draws = rb_draws_cond_from_components(g_cond_inf, g_cond_quiet) if g_cond_quiet.size else np.empty(
        (0, T_grid.size))

    return SnapshotResult(
        m=m,
        t_star=t_star,
        T_grid=np.asarray(T_grid, dtype=float),
        p_uncond_mean=p_uncond_mean,
        p_cond_mean=p_cond_mean,
        p_uncond_draws=p_uncond_draws,
        p_cond_draws=p_cond_draws,
        draws_array=draws_array,
        infection_times_2d=infection_times_2d,
        n_obs=int(sum(y for _, y in obs_sorted)),
        next_T=None,
        stopped_pairs=stopped_pairs,
    )


def run_spec(spec: Spec) -> Any:
    pars = make_parameters(spec)
    builder_kwargs = merge_dicts(BASELINE_BUILDER_KWARGS, spec.builder_overrides)

    result = _run_final_snapshot(
        obs_points=OBS_POINTS,
        pars=pars,
        builder_kwargs=builder_kwargs,
        num_trajectories=NUM_TRAJECTORIES,
        chunk_size=CHUNK_SIZE,
        T_run=T_RUN,
        max_cases=MAX_CASES,
        max_workers=MAX_WORKERS,
        T_grid=T_GRID,
        h=H,
        H_pad=H_PAD,
        min_required=MIN_REQUIRED,
        control_start=CONTROL_START,
    )
    return result


def get_draws(result: Any) -> np.ndarray:
    arr = np.asarray(result.draws_array, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError("Unexpected draws_array shape.")
    return arr


def _t_star_from_obs_points(obs_points: Sequence[Tuple[datetime, int]]) -> float:
    """Days between simulation start (= min obs date) and the final observation."""
    if not obs_points:
        return 0.0
    start = min(t for t, _ in obs_points)
    end = max(t for t, _ in obs_points)
    return float((end - start).days)


def get_draws_quiet_for_T(
        result: Any,
        T: float,
        *,
        t_star: Optional[float] = None,
) -> np.ndarray:
    """Filter draws to trajectories with *no infections* for T days after snapshot start.

    We keep a draw/trajectory if there are no infection times in (t_star, t_star + T],
    i.e. if the first post-snapshot infection time satisfies first_post > T.
    """
    if T < 0:
        raise ValueError("T must be non-negative.")

    draws = get_draws(result)
    infection_times_2d = list(getattr(result, "infection_times_2d", []))
    if draws.size == 0 or not infection_times_2d:
        return np.empty((0, 5), dtype=float)

    if t_star is None:
        if hasattr(result, "t_star"):
            t_star = float(getattr(result, "t_star"))
        else:
            t_star = _t_star_from_obs_points(OBS_POINTS)
    else:
        t_star = float(t_star)

    n = min(draws.shape[0], len(infection_times_2d))
    draws = draws[:n]
    infection_times_2d = infection_times_2d[:n]

    first_post: List[float] = []
    for traj in infection_times_2d:
        ts = np.sort(np.asarray(traj, dtype=float))
        future = ts[ts > t_star]
        first_post.append(float(future[0] - t_star) if future.size else np.inf)

    keep = np.asarray(first_post, dtype=float) > float(T)
    return draws[keep]


def posterior_components(result: Any) -> Dict[str, np.ndarray]:
    arr = get_draws(result)
    R = arr[:, 0]
    k = arr[:, 1]
    r = arr[:, 2]
    alpha = arr[:, 3]
    theta = arr[:, 4]

    rr = R * r
    alpha_theta = alpha * theta
    p0 = (k / (k + rr + 1e-12)) ** k
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        growth = (np.power(rr, 1.0 / alpha) - 1.0) / theta

    return {
        "R": R,
        "rR": rr,
        "alpha_theta": alpha_theta,
        "p0": p0,
        "growth": growth,
    }


def posterior_components_from_draws(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute derived posterior quantities from a draws array (n,5)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError("Unexpected draws array shape.")

    R = arr[:, 0]
    k = arr[:, 1]
    r = arr[:, 2]
    alpha = arr[:, 3]
    theta = arr[:, 4]

    rr = R * r
    alpha_theta = alpha * theta
    p0 = (k / (k + rr + 1e-12)) ** k
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        growth = (np.power(rr, 1.0 / alpha) - 1.0) / theta

    return {
        "R": R,
        "rR": rr,
        "alpha_theta": alpha_theta,
        "p0": p0,
        "growth": growth,
    }


def summary_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "median": float("nan"),
            "q025": float("nan"),
            "q975": float("nan"),
        }
    return {
        "median": float(np.median(x)),
        "q025": float(np.quantile(x, 0.025)),
        "q975": float(np.quantile(x, 0.975)),
    }


def build_summary_quiet_for_T(
        results: Mapping[str, Any],
        *,
        T: float,
        t_star: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build a per-spec summary table for the quiet-for-T filtered posteriors."""
    rows: List[Dict[str, Any]] = []
    for spec in SPECS:
        draws = get_draws_quiet_for_T(results[spec.key], T=T, t_star=t_star)
        comps = posterior_components_from_draws(draws) if draws.size else {
            "R": np.asarray([], dtype=float),
            "rR": np.asarray([], dtype=float),
            "alpha_theta": np.asarray([], dtype=float),
            "p0": np.asarray([], dtype=float),
            "growth": np.asarray([], dtype=float),
        }

        row: Dict[str, Any] = {
            "spec": spec.key,
            "title": spec.title,
            "n_retained": int(draws.shape[0]),
        }
        for name, values in comps.items():
            stats = summary_stats(values)
            row[f"{name}_median"] = stats["median"]
            row[f"{name}_q025"] = stats["q025"]
            row[f"{name}_q975"] = stats["q975"]
        rows.append(row)

    if not rows:
        return rows

    baseline = rows[0]
    for row in rows:
        for key in ("R_median", "rR_median", "alpha_theta_median", "p0_median", "growth_median"):
            b = float(baseline.get(key, float("nan")))
            v = float(row.get(key, float("nan")))
            row[f"delta_{key}"] = float(v - b) if np.isfinite(v) and np.isfinite(b) else float("nan")
    return rows


def print_quiet_for_T_summary_table(rows: Sequence[Mapping[str, Any]], *, T: float) -> None:
    """Print a compact comparison table for quiet-for-T filtered posteriors."""
    if not rows:
        print(f"\nQuiet-for-T posterior summary (T={T:g}): <no rows>")
        return

    print(f"\nQuiet-for-T posterior summary (condition: no infections for T={T:g} days)")
    col_spec = 14
    col_n = 7
    col_val = 20
    header = (
        f"{'Spec':<{col_spec}}"
        f"{'n':>{col_n}}"
        f"{'R (med)':>{col_val}}"
        f"{'rR (med)':>{col_val}}"
        f"{'aθ (med)':>{col_val}}"
        f"{'p0 (med)':>{col_val}}"
        f"{'g (med)':>{col_val}}"
    )
    print(header)
    print("-" * len(header))

    baseline = rows[0]
    for row in rows:
        name = str(row.get("spec", ""))
        n = int(row.get("n_retained", 0))

        def _fmt_med(key: str) -> str:
            v = float(row.get(key, float("nan")))
            if row.get("spec") == baseline.get("spec"):
                return pretty_number(v)
            dv = float(row.get(f"delta_{key}", float("nan")))
            sign = "+" if np.isfinite(dv) and dv >= 0 else ""
            return f"{pretty_number(v)} ({sign}{pretty_number(dv)})"

        line = (
            f"{name:<{col_spec}}"
            f"{n:>{col_n}d}"
            f"{_fmt_med('R_median'):>{col_val}}"
            f"{_fmt_med('rR_median'):>{col_val}}"
            f"{_fmt_med('alpha_theta_median'):>{col_val}}"
            f"{_fmt_med('p0_median'):>{col_val}}"
            f"{_fmt_med('growth_median'):>{col_val}}"
        )
        print(line)


def hdi_interval(x: np.ndarray, mass: float = 0.8) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    xs = np.sort(x)
    n = xs.size
    if n == 1:
        return float(xs[0]), float(xs[0])
    m = max(1, min(int(np.floor(mass * n)), n - 1))
    widths = xs[m:] - xs[: n - m]
    j = int(np.argmin(widths))
    return float(xs[j]), float(xs[j + m])


def expand_interval(lo: float, hi: float, factor: float = 1.2) -> Tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float("nan"), float("nan")
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * factor
    return float(mid - half), float(mid + half)


def build_baseline_interval_rows(result: Any) -> List[Dict[str, Any]]:
    arr = get_draws(result)
    R0 = arr[:, 0]
    k = arr[:, 1]
    r = arr[:, 2]
    alpha = arr[:, 3]
    theta = arr[:, 4]

    rr = R0 * r
    alpha_theta = alpha * theta
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        requirement_values = {
            "R0 * r": rr,
            "alpha * theta": alpha_theta,
            "sqrt(alpha) * theta": np.sqrt(alpha) * theta,
            "R0 / k": R0 / k,
            "R0 * r / k": rr / k,
            "(k / (k + R0 * r)) ^ k": (k / (k + rr + 1e-12)) ** k,
            "((R0 * r) ^ (1 / alpha) - 1) / theta": (np.power(rr, 1.0 / alpha) - 1.0) / theta,
        }

    series = [
        ("parameter", "R0", R0),
        ("parameter", "k", k),
        ("parameter", "r", r),
        ("parameter", "alpha", alpha),
        ("parameter", "theta", theta),
        ("requirement", "R0 * r", requirement_values["R0 * r"]),
        ("requirement", "alpha * theta", requirement_values["alpha * theta"]),
        ("requirement", "sqrt(alpha) * theta", requirement_values["sqrt(alpha) * theta"]),
        ("requirement", "R0 / k", requirement_values["R0 / k"]),
        ("requirement", "R0 * r / k", requirement_values["R0 * r / k"]),
        ("requirement", "(k / (k + R0 * r)) ^ k", requirement_values["(k / (k + R0 * r)) ^ k"]),
        ("requirement", "((R0 * r) ^ (1 / alpha) - 1) / theta",
         requirement_values["((R0 * r) ^ (1 / alpha) - 1) / theta"]),
    ]

    rows: List[Dict[str, Any]] = []
    for kind, name, values in series:
        lo80, hi80 = hdi_interval(values, mass=0.8)
        lo120, hi120 = expand_interval(lo80, hi80, factor=1.2)
        rows.append({
            "kind": kind,
            "name": name,
            "hdi80_lo": lo80,
            "hdi80_hi": hi80,
            "interval120_lo": lo120,
            "interval120_hi": hi120,
        })
    return rows


def print_baseline_interval_report(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\nBaseline posterior intervals (80% HDI and expanded 120%-width interval):")
    print("type         name                                     80% HDI                      120% interval")
    print("-" * 108)
    for row in rows:
        kind = str(row["kind"]).ljust(12)
        name = str(row["name"]).ljust(40)
        hdi = f"[{pretty_number(float(row['hdi80_lo']))}, {pretty_number(float(row['hdi80_hi']))}]".ljust(28)
        ext = f"[{pretty_number(float(row['interval120_lo']))}, {pretty_number(float(row['interval120_hi']))}]"
        print(f"{kind}{name}{hdi}{ext}")


def first_crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.flatnonzero(y >= level)
    if idx.size == 0:
        return float("nan")
    j = int(idx[0])
    if j == 0:
        return float(x[0])
    x0, x1 = x[j - 1], x[j]
    y0, y1 = y[j - 1], y[j]
    if y1 == y0:
        return float(x1)
    w = (level - y0) / (y1 - y0)
    return float(x0 + w * (x1 - x0))


def cumulative_matrix(infection_times_2d: Sequence[Sequence[float]], grid: np.ndarray) -> np.ndarray:
    rows = []
    for traj in infection_times_2d:
        t = np.sort(np.asarray(traj, dtype=float))
        rows.append(np.searchsorted(t, grid, side="right"))
    if not rows:
        return np.zeros((0, len(grid)))
    return np.vstack(rows).astype(float)


def rb_curve(result: Any, key: str = "p_cond_mean") -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(getattr(result, key), dtype=float)
    x = np.asarray(result.T_grid, dtype=float)
    return x, y


def kde_on_grid(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        out = np.zeros_like(grid)
        if x.size == 1:
            out[np.argmin(np.abs(grid - x[0]))] = 1.0
        return out
    kde = gaussian_kde(x)
    pdf = kde(grid)
    area = np.trapezoid(pdf, grid)
    if area > 0:
        pdf = pdf / area
    return pdf


def pretty_number(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def slugify(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


def hu_date_formatter(x: float, pos: int) -> str:
    dt = mdates.num2date(x)
    hu_month = {
        1: "jan.",
        2: "febr.",
        3: "márc.",
        4: "ápr.",
        5: "máj.",
        6: "jún.",
        7: "júl.",
        8: "aug.",
        9: "szept.",
        10: "okt.",
        11: "nov.",
        12: "dec.",
    }
    return f"{hu_month[dt.month]} {dt.day:02d}"


# ============================================================
# Summaries
# ============================================================

def build_summary(results: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_rows: List[Dict[str, Any]] = []
    posterior_rows: List[Dict[str, Any]] = []

    for spec in SPECS:
        res = results[spec.key]
        comps = posterior_components(res)
        t_rb, p_rb = rb_curve(res, key="p_cond_mean")

        raw_row: Dict[str, Any] = {
            "spec": spec.key,
            "title": spec.title,
            "p_rb_0": float(p_rb[0]),
            "T95": first_crossing(t_rb, p_rb, 0.95),
            "n_accepted": int(len(get_draws(res))),
        }
        for name, values in comps.items():
            stats = summary_stats(values)
            raw_row[f"{name}_median"] = stats["median"]
            raw_row[f"{name}_q025"] = stats["q025"]
            raw_row[f"{name}_q975"] = stats["q975"]
        raw_rows.append(raw_row)

        for name, values in comps.items():
            stats = summary_stats(values)
            posterior_rows.append({
                "spec": spec.key,
                "title": spec.title,
                "quantity": name,
                **stats,
            })

    baseline = raw_rows[0]
    for row in raw_rows:
        for key in ("R_median", "rR_median", "alpha_theta_median", "p0_median", "growth_median", "p_rb_0", "T95"):
            row[f"delta_{key}"] = float(row[key] - baseline[key]) if np.isfinite(row[key]) and np.isfinite(
                baseline[key]) else float("nan")

    return raw_rows, posterior_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    cols = [
        ("R_median", r"$R$"),
        ("rR_median", r"$rR$"),
        ("alpha_theta_median", r"$\alpha\theta$"),
        ("p0_median", r"$p_0$"),
        ("growth_median", r"$g_{\mathrm{EL}}$"),
        ("p_rb_0", r"$\widehat p_{\mathrm{RB}}(0)$"),
        ("T95", r"$T_{0.95}$"),
    ]
    baseline = rows[0]
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l" + "c" * len(cols) + "}\n")
        f.write("\\toprule\n")
        f.write("Specifikáció & " + " & ".join(label for _, label in cols) + r" \\" + "\n")
        f.write("\\midrule\n")
        for row in rows:
            parts = [row["title"]]
            for key, _ in cols:
                val = row[key]
                dval = row[f"delta_{key}"] if f"delta_{key}" in row else float("nan")
                if row["spec"] == baseline["spec"]:
                    parts.append(pretty_number(val))
                else:
                    sign = "+" if dval >= 0 else ""
                    parts.append(f"{pretty_number(val)} ({sign}{pretty_number(dval)})")
            f.write(" & ".join(parts) + r" \\" + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


# ============================================================
# Plotting
# ============================================================

def plot_rb_curves(results: Mapping[str, Any], out_path: Path, key: str = "p_cond_mean") -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for spec in SPECS:
        t, y = rb_curve(results[spec.key], key=key)
        ax.plot(t, y, lw=2.0 if spec.key == "baseline" else 1.8, color=COLORS.get(spec.key), label=spec.title)
    ax.set_xlabel(r"$T$ nap a $t_\star$ időpont után")
    ax.set_ylabel("Valószínűség")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, np.max(T_GRID))
    ax.yaxis.grid(True, color="#eeeeee")
    ax.xaxis.grid(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_incidence_medians(results: Mapping[str, Any], out_path: Path) -> None:
    set_style()
    start_date = OBS_POINTS[0][0]
    horizon_days = T_RUN
    grid = np.arange(0.0, horizon_days + 1e-9, 0.5)
    dates = [start_date + timedelta(days=float(x)) for x in grid]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for spec in SPECS:
        res = results[spec.key]
        mat = cumulative_matrix(res.infection_times_2d, grid)
        med = np.mean(mat, axis=0) if len(mat) else np.zeros_like(grid)
        ax.plot(dates, med, lw=2.0 if spec.key == "baseline" else 1.7, color=COLORS.get(spec.key), label=spec.title)

    obs_sorted = sorted(OBS_POINTS, key=lambda x: x[0])
    obs_dates, obs_incs = zip(*obs_sorted)
    obs_cums = np.cumsum(obs_incs)
    ax.scatter(obs_dates, obs_cums, s=24, facecolors="white", edgecolors="#000000", lw=1.0, zorder=10,
               label="Megfigyelt")

    ax.set_ylabel("Kumulatív fertőzések")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(hu_date_formatter))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.yaxis.grid(True, color="#eeeeee")
    ax.xaxis.grid(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def metric_bounds(all_posteriors: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for quantity in ["R", "rR", "alpha_theta", "p0", "growth"]:
        vals = np.concatenate(
            [all_posteriors[key][quantity][np.isfinite(all_posteriors[key][quantity])] for key in all_posteriors])
        lo = float(np.quantile(vals, 0.005))
        hi = float(np.quantile(vals, 0.995))
        span = max(hi - lo, 1e-6)
        lo -= 0.08 * span
        hi += 0.08 * span
        if quantity == "p0":
            lo, hi = max(0.0, lo), min(1.0, hi)
        bounds[quantity] = (lo, hi)
    return bounds


def plot_posteriors_vs_baseline(results: Mapping[str, Any], out_path: Path) -> None:
    set_style()
    baseline_key = "baseline"
    compare_specs = SPECS[1:]
    if not compare_specs:
        return

    posteriors = {spec.key: posterior_components(results[spec.key]) for spec in SPECS}
    bounds = metric_bounds(posteriors)
    quantities = [
        ("R", r"$R$"),
        ("rR", r"$rR$"),
        ("alpha_theta", r"$\alpha\theta$"),
        ("p0", r"$p_0$"),
        ("growth", r"$g_{\mathrm{EL}}$"),
    ]

    n_rows = len(compare_specs)
    n_cols = len(quantities)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.5, max(2.6 * n_rows, 3.0)),
        squeeze=False,
    )

    for i, spec in enumerate(compare_specs):
        for j, (name, title) in enumerate(quantities):
            ax = axes[i, j]
            lo, hi = bounds[name]
            grid = np.linspace(lo, hi, 500)
            base_pdf = kde_on_grid(posteriors[baseline_key][name], grid)
            alt_pdf = kde_on_grid(posteriors[spec.key][name], grid)

            baseline_label = "Alapbeállítás" if i == 0 and j == 0 else None
            alt_label = spec.title if j == 0 else None
            ax.plot(grid, base_pdf, color=COLORS[baseline_key], lw=1.8, label=baseline_label)
            ax.plot(grid, alt_pdf, color=COLORS[spec.key], lw=1.8, label=alt_label)

            if i == 0:
                ax.set_title(title)
            ax.set_yticks([])
            ax.set_xlim(lo, hi)
            if j == 0:
                ax.set_ylabel(spec.title)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # if handles:
    #     fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(out_path)
    plt.close(fig)


def plot_posteriors_vs_baseline_quiet_for_T(
        results: Mapping[str, Any],
        out_path: Path,
        *,
        T: float,
        t_star: Optional[float] = None,
        print_summary: bool = True,
) -> None:
    """Posterior KDE comparison plot conditioned on a quiet period of length T.

    This mirrors :func:`plot_posteriors_vs_baseline`, but filters each spec's
    posterior draws to trajectories with *no infections* in (t_star, t_star + T].
    """
    set_style()
    baseline_key = "baseline"
    compare_specs = SPECS[1:]
    if not compare_specs:
        return

    quiet_draws: Dict[str, np.ndarray] = {
        spec.key: get_draws_quiet_for_T(results[spec.key], T=T, t_star=t_star)
        for spec in SPECS
    }
    if quiet_draws.get(baseline_key, np.empty((0, 5))).shape[0] == 0:
        # Nothing to compare if the baseline has no retained draws.
        return

    if print_summary:
        quiet_rows = build_summary_quiet_for_T(results, T=T, t_star=t_star)
        print_quiet_for_T_summary_table(quiet_rows, T=T)

    posteriors = {
        spec.key: posterior_components_from_draws(quiet_draws[spec.key])
        for spec in SPECS
    }
    bounds = metric_bounds(posteriors)

    quantities = [
        ("R", r"$R$"),
        ("rR", r"$rR$"),
        ("alpha_theta", r"$\alpha\theta$"),
        ("p0", r"$p_0$"),
        ("growth", r"$g_{\mathrm{EL}}$"),
    ]

    n_rows = len(compare_specs)
    n_cols = len(quantities)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.5, max(2.6 * n_rows, 3.0)),
        squeeze=False,
    )

    for i, spec in enumerate(compare_specs):
        for j, (name, title) in enumerate(quantities):
            ax = axes[i, j]
            lo, hi = bounds[name]
            grid = np.linspace(lo, hi, 500)
            base_pdf = kde_on_grid(posteriors[baseline_key][name], grid)
            alt_pdf = kde_on_grid(posteriors[spec.key][name], grid)

            baseline_label = "Alapbeállítás" if i == 0 and j == 0 else None
            alt_label = spec.title if j == 0 else None
            ax.plot(grid, base_pdf, color=COLORS[baseline_key], lw=1.8, label=baseline_label)
            ax.plot(grid, alt_pdf, color=COLORS[spec.key], lw=1.8, label=alt_label)

            if i == 0:
                ax.set_title(title)
            ax.set_yticks([])
            ax.set_xlim(lo, hi)
            if j == 0:
                ax.set_ylabel(spec.title)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # if handles:
    #     fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.01))

    # Annotate the quiet-period condition.
    # fig.suptitle(fr"Conditioned on no infections for $T={T:g}$ days", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    for spec in SPECS:
        print(f"Running / loading specification: {spec.title}")
        results[spec.key] = run_spec(spec)

    # baseline_interval_rows = build_baseline_interval_rows(results["baseline"])
    # print_baseline_interval_report(baseline_interval_rows)
    # exit()
    # write_csv(OUTPUT_DIR / "baseline_intervals.csv", baseline_interval_rows)

    raw_rows, posterior_rows = build_summary(results)
    write_csv(OUTPUT_DIR / "summary_metrics.csv", raw_rows)
    write_csv(OUTPUT_DIR / "posterior_summary.csv", posterior_rows)
    write_latex_summary(OUTPUT_DIR / "summary_table.tex", raw_rows)

    plot_rb_curves(results, OUTPUT_DIR / "figures" / "rb_curves 2.pdf", key="p_cond_mean")
    plot_incidence_medians(results, OUTPUT_DIR / "figures" / "incidence_medians 2.pdf")
    plot_posteriors_vs_baseline(results, OUTPUT_DIR / "figures" / "posterior_compare_all 2.pdf")
    plot_posteriors_vs_baseline_quiet_for_T(results, OUTPUT_DIR / "figures" / "posterior_compare_all_quiet 2.pdf",
                                            T=10.0)

    print(f"Done. Outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
