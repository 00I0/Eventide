import time
from datetime import datetime

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from python.eventide import (
    Parameters, Simulator, Scenario,
    InfectionTimeCollector
)
from python.optimize_acceptance_windows import build_acceptance_inequalities

# --- Parameters ---
pars = Parameters(
    R0=(0.25, 15),
    k=(0.2, 10),
    r=(0.01, 0.99),
    alpha=(0.01, 20),
    theta=(0.01, 20)
).require('R0 * r < 1').require('10 < alpha * theta').require('alpha * theta < 14')

sim_start = datetime(2025, 3, 3)
PLOT_HORIZON_DAYS = 90  # only show first 90 days since sim_start
T_run = 150  # simulation horizon
T_grid = np.arange(0.0, T_run + 1e-9, 1.0)

# --- Observations (full list); we will iterate over prefixes [:2], [:3], ... ---
obs_points = [
    (datetime(2025, 3, 6), 1),
    (datetime(2025, 3, 21), 3),
    (datetime(2025, 3, 25), 1),
    (datetime(2025, 3, 26), 1),
    (datetime(2025, 3, 30), 1),
    (datetime(2025, 4, 2), 2),
    (datetime(2025, 4, 17), 1),
]

# We'll call your provided helper directly as requested.
# from somewhere import build_acceptance_inequalities

# --- Plot style (PGF/TeX) ---
mpl.use('pgf')
mpl.rcParams.update({
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': r'''
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{siunitx}
        \usepackage[T1]{fontenc}
        \usepackage{lmodern}
    ''',
    'font.family': 'serif',
    'font.size': 12,
    'lines.linewidth': 1.0,
    'lines.markersize': 3.0,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.unicode_minus': False,
    'axes.prop_cycle': mpl.cycler(color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']),
})


# --- Estimator + CI utilities (conditional on quiet window) ---
def empirical_cond_over(infection_times, T_grid, t_star):
    """
    Estimate p(T) = P(B | A(T)), with:
      A(T): no infection in (t_star, t_star+T]
      B:    no infection after t_star at all
    infection_times: list of arrays of infection times (days since sim_start)
    t_star: cutoff (days since sim_start)
    Returns: (p_hat[T], se[T])
    """
    M = len(infection_times)
    first_post = np.full(M, np.inf, dtype=float)
    has_post = np.zeros(M, dtype=bool)
    for m, times in enumerate(infection_times):
        ts = np.sort(np.asarray(times, dtype=float))
        j = np.searchsorted(ts, t_star, side='right')  # strictly after t_star
        if j < ts.size:
            first_post[m] = ts[j]
            has_post[m] = True
    B_mask = ~has_post  # no post-cutoff infections

    p = np.empty(T_grid.size, dtype=float)
    se = np.empty(T_grid.size, dtype=float)
    for j, T in enumerate(T_grid):
        cutoff = t_star + T
        # Quiet window is (t_star, cutoff] => first post > cutoff
        A_mask = (first_post > cutoff)
        denom = np.count_nonzero(A_mask)
        numer = np.count_nonzero(A_mask & B_mask)
        if denom > 0:
            pj = numer / denom
            p[j] = pj
            se[j] = np.sqrt(pj * (1 - pj) / denom)  # binomial s.e. conditional on A
        else:
            p[j] = np.nan
            se[j] = np.nan
    return p, se


def cond_denominator_counts(infection_times, T_grid, t_star):
    """
    n_A(T): number of trajectories satisfying A(T) (quiet through t_star+T).
    """
    M = len(infection_times)
    first_post = np.full(M, np.inf, dtype=float)
    for m, times in enumerate(infection_times):
        ts = np.sort(np.asarray(times, dtype=float))
        j = np.searchsorted(ts, t_star, side='right')
        if j < ts.size:
            first_post[m] = ts[j]
    nA = np.array([(first_post > (t_star + T)).sum() for T in T_grid], dtype=float)
    return nA


def wilson_interval(phat, n, z=1.96):
    """
    Vectorized Wilson score interval (95% when z=1.96).
    """
    phat = np.asarray(phat, dtype=float)
    n = np.asarray(n, dtype=float)
    out_lo = np.full_like(phat, np.nan)
    out_hi = np.full_like(phat, np.nan)

    mask = n > 0
    if not np.any(mask):
        return out_lo, out_hi

    p = phat[mask]
    nn = n[mask]
    denom = 1.0 + (z ** 2) / nn
    center = (p + (z ** 2) / (2 * nn)) / denom
    half = z * np.sqrt((p * (1 - p) / nn) + (z ** 2) / (4 * nn ** 2)) / denom
    out_lo[mask] = np.clip(center - half, 0.0, 1.0)
    out_hi[mask] = np.clip(center + half, 0.0, 1.0)
    return out_lo, out_hi


# --- Run progressive simulations and plot ---
fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8))
t0 = time.time()

for k in range(2, len(obs_points) + 1):
    this_obs = obs_points[:k]
    cutoff_dt = this_obs[-1][0]
    t_star = (cutoff_dt - sim_start).days  # cutoff in days since sim_start

    # Build criteria for THIS prefix
    criteria = build_acceptance_inequalities(
        obs_points=this_obs,
        simulation_start=sim_start,
        sigma_days=3.5138784768847935,
        beta=0.5448705790647912,
        neighbor_weight=0.3259905700578138,
        grid_step_days=0.29607190994514637,
        min_seg_days=1.1388690204854988,
        kmax=4,
        baseline_p=0.07466034737192961,
        alpha=0.3887850260822494,
        h_max=0.16418840893328812,
        eps_share=0.0999993345398409,
        include_gap_windows=False,
        include_union_windows=True,
        max_unions_to_keep=6,
        gap_scale=0.3312202106199904,
        mode='cluster'
    )

    # Collect infection times only
    infection_times_collector = InfectionTimeCollector()

    sim = Simulator(
        parameters=pars,
        sampler=pars.create_latin_hypercube_sampler(),
        start_date=sim_start,
        scenario=Scenario([]),  # no change points for this task
        criteria=criteria,
        collectors=[infection_times_collector],
        num_trajectories=1_00_000_000,  # keep as requested; reduce during local testing if needed
        chunk_size=100_000,
        T_run=T_run,
        max_cases=1000,
        max_workers=13,
    )
    sim.run()

    infection_times = infection_times_collector.infection_times

    # Empirical conditional probability and denominators
    pC_emp, _ = empirical_cond_over(infection_times, T_grid, t_star)
    n_cond = cond_denominator_counts(infection_times, T_grid, t_star)
    loC, hiC = wilson_interval(pC_emp, n_cond, z=1.96)  # 95%

    # Absolute x-axis (days since sim_start); curves "shift" by t_star
    x_abs = t_star + T_grid

    # Only show the first 90 days since sim_start
    m90 = (x_abs <= PLOT_HORIZON_DAYS) & ~np.isnan(pC_emp)

    # Plot line + 95% CI band, legend without "(k=...)"
    (line,) = ax.plot(x_abs[m90], pC_emp[m90], lw=1.2,
                      label=f"cutoff {cutoff_dt:%Y-%m-%d}")
    ax.fill_between(x_abs[m90], loC[m90], hiC[m90],
                    alpha=0.35, linewidth=0, color=line.get_color())

print(f"Total runtime for {len(obs_points) - 1} prefixes: {time.time() - t0:.2f}s")

# --- Finalize single plot ---
ax.set_xlabel("Days since simulation start")
ax.set_ylabel("Empirical conditional probability (95% CI)")
ax.set_xlim(0, PLOT_HORIZON_DAYS)
ax.set_ylim(-0.02, 1.02)
ax.set_title(r"Empirical conditional probability with progressive cutoffs")
ax.legend(frameon=False, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("../img/empirical_conditional_progressive.pgf")
# plt.show()
