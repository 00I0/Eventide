import time
from datetime import datetime
# ======================================================================
# Styles (unified + journal-ready)
# ======================================================================
from typing import Optional
from typing import Sequence, Any

import numpy as np

from python.eventide import (
    Parameters, Simulator, Scenario,
    DrawCollector, ActiveSetSizeCollector, InfectionTimeCollector,
    IndexOffspringCriterion
)
from python.optimize_acceptance_windows import build_acceptance_inequalities
from python.plots.exintction_plot import plot_all_with_probability_bands
from python.plots.extinct_probs import rb_cond_components_post, rao_blackwell_uncond_over_post_full
from python.plots.misc import SnapshotResult
from python.plots.online_extinction_plot import plot_rb_online_two_pane_shifted
from python.plots.plot_Re_time import plot_timepath_Re
from python.plots.plot_acceptance_construction import plot_construction_side_by_side
from python.plots.plot_cumulative import plot_cumulative_infections_last_numeric
from python.plots.plot_posteriors import plot_posterior_grid_single
from python.plots.plot_selected_Hus import plot_selected_Hus
from python.plots.style import Style, set_style

# ======================================================================
# Data structures
# ======================================================================

_CURRENT_STYLE: Optional[Style] = None

from typing import Dict, List, Optional


# ======================================================================
# Snapshot runners
# ======================================================================

def run_snapshot(
        m: int,
        obs_points,
        pars,
        builder_kwargs: Dict[str, Any],
        num_trajectories: int = 800_000,
        chunk_size: int = 100_000,
        T_run: int = 70,
        max_cases: int = 1000,
        max_workers: int = 8,
        T_grid: np.ndarray = np.arange(0, 70 + 1e-9, 1.0),
        h: float = 0.2,
        H_pad: float = 10.0,
        min_required: Optional[int] = None
) -> SnapshotResult:
    """Build acceptance from the first m observations, run ABC, compute RB curves on T_grid."""

    def _cast_accept_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(d)
        if "kmax" in out: out["kmax"] = int(round(float(out["kmax"])))
        for b in ("include_gap_windows", "include_union_windows"):
            if b in out: out[b] = bool(round(float(out[b])))
        if "max_unions_to_keep" in out: out["max_unions_to_keep"] = int(round(float(out["max_unions_to_keep"])))
        if "mode" in out: out["mode"] = str(out["mode"])
        return out

    def _build_criteria_for_prefix(obs_points, m, **kwargs):
        pts = obs_points[:m]
        sim_start = min(t for t, _ in pts)
        crit = [IndexOffspringCriterion(2, 5)] + build_acceptance_inequalities(
            obs_points=pts, simulation_start=sim_start, **_cast_accept_kwargs(kwargs)
        )
        return crit, sim_start

    crit, sim_start = _build_criteria_for_prefix(obs_points, m, **builder_kwargs)

    collectors = [
        draws := DrawCollector(),
        active_set := ActiveSetSizeCollector(obs_points[m - 1][0]),
        itimes := InfectionTimeCollector(),
    ]
    sim = Simulator(
        parameters=pars,
        sampler=pars.create_latin_hypercube_sampler(),
        start_date=min(t for t, _ in obs_points),
        scenario=Scenario([]),
        criteria=crit,
        collectors=collectors,
        num_trajectories=num_trajectories,
        chunk_size=chunk_size,
        T_run=T_run,
        max_cases=max_cases,
        max_workers=max_workers,
        min_required=min_required
    )
    sim.run()

    infection_times_2d = list(itimes.infection_times)
    stopped_pairs = active_set.active_sets
    R0s, ks, rs, alphas, thetas = np.asarray(draws).T
    t_star = (active_set.collection_date - sim.start_date).days

    Tf, pU_mean_exact, gU_draws_fine = rao_blackwell_uncond_over_post_full(
        infection_times_2d, stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_max=float(T_grid[-1]), t_star=t_star, h=h, H_pad=H_pad
    )
    p_uncond_mean = np.interp(T_grid, Tf, pU_mean_exact)
    p_uncond_draws = np.vstack([np.interp(T_grid, Tf, row) for row in gU_draws_fine])

    gC_inf, gC_quiet = rb_cond_components_post(
        infection_times_2d, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=h
    )
    p_cond_mean = (gC_inf.mean() / gC_quiet.mean(axis=0)) if gC_quiet.size else np.full_like(T_grid, np.nan)
    p_cond_draws = gC_inf[:, None] / gC_quiet if gC_quiet.size else np.empty((0, T_grid.size))

    print('Run', m, 'accepted', len(infection_times_2d))

    next_T = None
    if m < len(obs_points):
        delta = (obs_points[m][0] - obs_points[m - 1][0]).total_seconds() / 86400.0
        next_T = float(delta)

    n_obs = int(sum(y for _, y in obs_points[:m]))

    return SnapshotResult(
        m=m,
        t_star=t_star,
        T_grid=T_grid,
        p_uncond_mean=p_uncond_mean,
        p_cond_mean=p_cond_mean,
        p_uncond_draws=p_uncond_draws,
        p_cond_draws=p_cond_draws,
        draws_array=np.asarray(draws),
        infection_times_2d=infection_times_2d,
        next_T=next_T,
        n_obs=n_obs,
        stopped_pairs=stopped_pairs
    )


def run_all_snapshots_per_m(
        obs_points,
        pars,
        builder_kwargs_by_m: Dict[int, Dict[str, Any]],
        snapshots: Sequence[int],
        *,
        num_trajectories: int = 1_000_000,
        chunk_size: int = 100_000,
        T_run: int = 70,
        max_cases: int = 1000,
        max_workers: int = 12,
        T_grid: np.ndarray = np.arange(0, 70 + 1e-9, 1.0),
        h: float = 0.2,
        H_pad: float = 10.0,
        min_required: Optional[int] = None
) -> List[SnapshotResult]:
    results = []
    for m in (m for m in snapshots if m >= 3):
        kw = builder_kwargs_by_m.get(m, {})
        res = run_snapshot(
            m=m,
            obs_points=obs_points,
            pars=pars,
            builder_kwargs=kw,
            num_trajectories=num_trajectories,
            chunk_size=chunk_size,
            T_run=T_run,
            max_cases=max_cases,
            max_workers=max_workers,
            T_grid=T_grid,
            h=h,
            H_pad=H_pad,
            min_required=min_required,
        )
        results.append(res)
    return results


def main():
    # --- Activate a uniform, journal-ready style across *all* plots
    set_style(column="double", base_font=14, dpi=320, use_tex=False, show_grid=False)

    # ---- parameters
    pars = (Parameters(
        R0=(0.25, 15),
        k=(0.2, 10),
        r=(0.01, 0.99),
        alpha=(0.01, 20),
        theta=(0.01, 20)
    ).require('R0 * r < 3')
            .require('3 < alpha * theta').require('alpha * theta < 20')
            .require('1/sqrt(alpha) >= 0.1').require('1/sqrt(alpha) <= 0.9')
            .require('1 <= sqrt(alpha) * theta').require('sqrt(alpha) * theta <= 15'))

    obs_points = [
        (datetime(2025, 3, 3), 1),
        (datetime(2025, 3, 20), 3),
        (datetime(2025, 3, 24), 1),
        (datetime(2025, 3, 25), 1),
        (datetime(2025, 3, 30), 1),
        (datetime(2025, 4, 1), 2),
        (datetime(2025, 4, 4), 1),
        (datetime(2025, 4, 17), 1),
    ]
    snapshots = (3, 4, 5, 6, 7, 8)
    T_grid = np.arange(0, 70 + 1e-9, 1.0)

    # ---- per-snapshot acceptance kwargs
    best_kwargs_by_m = {
        2: {'sigma_days': 1.0, 'beta': 0.75, 'neighbor_weight': 0.8,
            'grid_step_days': 0.7069396345569126, 'min_seg_days': 0.5498366748209828, 'kmax': 5,
            'baseline_p': 0.15535618566466364,
            'alpha': 0.3993493824281206, 'h_max': 0.36168539770090363, 'eps_share': 0.07926160513478236,
            'include_gap_windows': False, 'include_union_windows': True, 'max_unions_to_keep': 1,
            'gap_scale': 0.7992419483068927, 'mode': 'segment'},
        3: {'sigma_days': 0.5966211660241121, 'beta': 0.5966211660241121, 'neighbor_weight': 0.30681432190053654,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 5, 'baseline_p': 0.13802210868325532,
            'alpha': 0.07610820650842659, 'h_max': 0.004574550308592357, 'eps_share': 0.05630074728705349,
            'include_gap_windows': False, 'include_union_windows': False, 'max_unions_to_keep': 5,
            'gap_scale': 0.1501261393189749, 'mode': 'cluster'},
        4: {'sigma_days': 3.8528152014595296, 'beta': 0.7834260755055544, 'neighbor_weight': 0.5995991966482781,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 5, 'baseline_p': 0.14709316211475382,
            'alpha': 0.387266136503176, 'h_max': 0.0002883690811074091, 'eps_share': 0.04971483766473928,
            'include_gap_windows': True, 'include_union_windows': True, 'max_unions_to_keep': 5,
            'gap_scale': 0.15364873314885064, 'mode': 'cluster'},
        5: {'sigma_days': 0.4138019095729616, 'beta': 0.9640140542294829, 'neighbor_weight': 0.3349658451776068,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 5, 'baseline_p': 0.07244378182825628,
            'alpha': 0.299871027309736, 'h_max': 0.3525137985689808, 'eps_share': 0.013971158538885366,
            'include_gap_windows': False, 'include_union_windows': True, 'max_unions_to_keep': 6,
            'gap_scale': 0.8906859038362628, 'mode': 'cluster'},
        6: {'sigma_days': 1.0, 'beta': 0.75, 'neighbor_weight': 0.8,
            'grid_step_days': 0.16628237119014017, 'min_seg_days': 0.8739716473789757, 'kmax': 5,
            'baseline_p': 0.05420222962320582, 'alpha': 0.22390683694143176, 'h_max': 0.19886424623959195,
            'eps_share': 0.0018096283765457303, 'include_gap_windows': True, 'include_union_windows': True,
            'max_unions_to_keep': 1, 'gap_scale': 0.7358464682803482, 'mode': 'segment'},
        7: {'sigma_days': 0.8554052419608095, 'beta': 0.7063167204522096, 'neighbor_weight': 0.3006068585907883,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 5, 'baseline_p': 0.0551662673770968,
            'alpha': 0.29763668754480394, 'h_max': 0.07723938993735249, 'eps_share': 0.0614888223914195,
            'include_gap_windows': True, 'include_union_windows': True, 'max_unions_to_keep': 6,
            'gap_scale': 0.7887632675348197, 'mode': 'cluster'},

        # 8: {'sigma_days': 1.87448405791103, 'beta': 0.8122752531114021, 'neighbor_weight': 0.36814473528396524,
        #     'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 4, 'baseline_p': 0.11350999644046003,
        #     'alpha': 0.44260858510448914, 'h_max': 0.1860463899250351, 'eps_share': 0.0988640778045239,
        #     'include_gap_windows': True, 'include_union_windows': False, 'max_unions_to_keep': 6,
        #     'gap_scale': 0.15009926426544729, 'mode': 'cluster'},

        8: {'sigma_days': 0.31798182271094916, 'beta': 0.9531742271317549, 'neighbor_weight': 0.3052971316181496,
            'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 5,
            'baseline_p': 0.061612817304115365, 'alpha': 0.30952093482926424, 'h_max': 0.006882237183009204,
            'eps_share': 0.08763955930821622, 'include_gap_windows': True, 'include_union_windows': True,
            'max_unions_to_keep': 4, 'gap_scale': 0.8999651561495379, 'mode': 'cluster'},

        # 8: {'sigma_days': 1.0, 'beta': 0.75, 'neighbor_weight': 0.8,
        #     'grid_step_days': 0.3717008410804425, 'min_seg_days': 1.3749050049077112, 'kmax': 6,
        #     'baseline_p': 0.05000019698741225, 'alpha': 0.274251428198899, 'h_max': 0.02446491644928228,
        #     'eps_share': 0.02657521874931045, 'include_gap_windows': True, 'include_union_windows': True,
        #     'max_unions_to_keep': 6, 'gap_scale': 0.8999651561495379, 'mode': 'segment'},

        # 8: {'sigma_days': 0.2586792100075642, 'beta': 0.8602035868562967, 'neighbor_weight': 0.45073426051423593,
        #     'grid_step_days': 0.25, 'min_seg_days': 1.0, 'kmax': 5,
        #     'baseline_p': 0.059449554079719774, 'alpha': 0.36946731625462104, 'h_max': 0.0016948505908678524,
        #     'eps_share': 0.07136140411363777, 'include_gap_windows': False, 'include_union_windows': True,
        #     'max_unions_to_keep': 4, 'gap_scale': 0.7692023454814878, 'mode': 'cluster'},
    }

    builder_kwargs_by_m = dict(best_kwargs_by_m)

    # ---- single consolidated run
    t0 = time.time()
    results = run_all_snapshots_per_m(
        obs_points=obs_points,
        pars=pars,
        builder_kwargs_by_m=builder_kwargs_by_m,
        snapshots=snapshots,
        num_trajectories=200_000_000_000,
        chunk_size=100_000,
        T_run=70,
        max_cases=1000,
        max_workers=13,
        T_grid=T_grid,
        h=0.2,
        H_pad=10.0,
        min_required=10_000
    )
    print("Total runtime (all snapshots):", time.time() - t0)

    # ---- reuse LAST snapshot for the 3 one-off plots (no second simulation needed)
    last = results[-1]
    draws_arr = last.draws_array
    R0s, ks, rs, alphas, thetas = draws_arr.T

    plot_all_with_probability_bands(
        last.infection_times_2d, last.stopped_pairs, R0s, rs, ks, alphas, thetas,
        last.T_grid, last.t_star, h=0.2, H_pad=10.0,
    )
    plot_selected_Hus(R0s, T_max=70, alphas=alphas, ks=ks, rs=rs, thetas=thetas)
    plot_construction_side_by_side(obs_points)

    # ---- online multi-snapshot plots (unchanged numerics)
    plot_rb_online_two_pane_shifted(results, show_next_dot=True, ylim=(0, 1))
    plot_posterior_grid_single(results[-1], mass=0.95, bw_adjust=1.0, n_grid=600)
    plot_timepath_Re(results, step=0.25, band=(0.025, 0.975), summary="median", draw_verticals=True)
    plot_cumulative_infections_last_numeric(
        results,
        resolution=0.25,
        perc_bands=(0.95, 0.5, 0.2),
        cmap="PuBu",
        scale="linear",
        show_mean=True,
        show_median=False,
        obs_points_days=[((day - datetime(2025, 3, 6)).days, cases) for (day, cases) in obs_points]
    )


if __name__ == '__main__':
    main()
