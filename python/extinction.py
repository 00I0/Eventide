import time
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gamma

from python.eventide import (Parameters, IntervalCriterion, IndexOffspringCriterion, Simulator,
                             Scenario, ActiveSetSizeCollector, ParameterChangePoint)
from python.eventide.collectors import TimeMatrix, DrawCollector

pars = Parameters(
    R0=(0.25, 15),
    k=(0.2, 10),
    r=(0.01, 0.99),
    alpha=(0.01, 20),
    theta=(0.01, 20)
).require('R0 * r < 10').require('1 < alpha * theta').require('alpha * theta < 50')

sampler = pars.create_latin_hypercube_sampler()
criteria = [
    IntervalCriterion(datetime(2025, 3, 3), datetime(2025, 4, 4), 9, 11),
    IntervalCriterion(datetime(2025, 4, 1), datetime(2025, 4, 4), 1, 11),
    IntervalCriterion(datetime(2025, 4, 4), datetime(2025, 4, 14), 0, 0),
    IntervalCriterion(datetime(2025, 4, 14), datetime(2025, 4, 17), 1, 1),
    IndexOffspringCriterion(2, 5)
]

collectors = [
    time_matrix := TimeMatrix(datetime(2025, 4, 17)),
    draws := DrawCollector(),
    active_set_size := ActiveSetSizeCollector(datetime(2025, 4, 17))
]

sim = Simulator(
    parameters=pars,
    sampler=sampler,
    start_date=datetime(2025, 3, 3),
    scenario=Scenario([
        ParameterChangePoint('R0', datetime(2025, 4, 14), '0.5 * R0'),  # rábapordány
        ParameterChangePoint('R0', datetime(2025, 4, 17))
    ]),
    criteria=criteria,
    collectors=collectors,
    num_trajectories=1_000_000_000,
    chunk_size=100_000,
    T_run=90,
    max_cases=1000,
    max_workers=13,
)

now = time.time()
sim.run()
print('Runtime:', time.time() - now)
print('Trajectories accepted (time matrix):', time_matrix.numpy().sum())

# create_end_of_outbreak_plot(time_matrix.numpy(), time_matrix.cutoff_day(sim.start_date), sim.start_date, sim.T_run)

time_mtx_np = time_matrix.numpy()
draws_np = draws.numpy()
active_set_size_np = active_set_size.numpy()


def plot_analytical_extinction(draws, start_date, t_max_date, T_run, active_set_sizes):
    """
    Calculates and plots the analytical extinction probability based on a
    posterior distribution of parameters.

    Args:
        draws (np.ndarray): The (N_accepted, 5) array of accepted parameters
                            from the DrawCollector. Columns are (R0, k, r, alpha, theta).
        start_date (datetime): The start date of the simulation.
        t_max_date (datetime): The date of the last observed case (t_max).
        T_run (int): Total number of days the simulation runs for plotting range.
    """
    R0s = draws[:, 0]
    ks = draws[:, 1]
    rs = draws[:, 2]
    alphas = draws[:, 3]
    thetas = draws[:, 4]

    R_effs = R0s * rs

    t_max_day = (t_max_date - start_date).days
    dates = [t_max_date + timedelta(days=d) for d in range(1, T_run - t_max_day)]
    if not dates:
        print("Warning: No days to plot for analytical probability. T_run might be too short.")
        return

    # The quiet window T is the number of days since t_max
    quiet_windows_T = np.arange(1, len(dates) + 1)

    # ---  Calculate P_ext(T) for each T and each parameter set ---
    prob_distributions = np.zeros((len(quiet_windows_T), len(R0s)))

    for i, T in enumerate(quiet_windows_T):
        F_T = gamma.cdf(T, a=alphas, scale=thetas)

        term1 = 1 + R_effs / ks
        term2 = 1 + (R_effs * F_T) / ks

        probs_for_this_T = np.full_like(term1, 1.0)
        valid_mask = term2 > 0
        probs_for_this_T[valid_mask] = (term1[valid_mask] / term2[valid_mask]) ** (-ks[valid_mask])

        prob_distributions[i, :] = np.clip(probs_for_this_T ** active_set_sizes, 0, 1)

    median_probs = np.median(prob_distributions, axis=1)
    lower_ci = np.percentile(prob_distributions, 35, axis=1)
    upper_ci = np.percentile(prob_distributions, 65, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.set_title("Analytical Extinction Probability Over Time")
    ax.plot(dates, median_probs, marker='', lw=2, label="Median Analytical Probability")

    print("\nAnalytical extinction probabilities")
    for date, prob in zip(dates, median_probs):
        print(date.strftime('%Y-%m-%d'), f'{prob: .4f}')

    ax.fill_between(dates, lower_ci, upper_ci, alpha=0.2, label="30% Credible Interval")

    threshold = 0.9
    ax.axhline(threshold, color='red', ls='--', lw=1)

    try:
        cross_idx = np.where(median_probs >= threshold)[0][0]
        d0, p0 = dates[cross_idx], median_probs[cross_idx]
        ax.axvline(d0, color='red', ls='--', lw=1)
        ax.scatter([d0], [p0], zorder=5, c='red')
    except IndexError:
        print(f"Analytical probability did not reach the {threshold * 100:.0f}% threshold.")

    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability of Extinction")
    ax.grid(True, ls=':')
    ax.legend()
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_extinction_probabilities(draws=None, start_date=None, T_run=None, active_set_sizes=None, matrix=None,
                                  cutoff_date=None):
    """
    Plots analytical and/or empirical extinction probabilities on the same axes.

    Args:
        draws (np.ndarray, optional): (N_accepted, 5) array of accepted parameters
        start_date (datetime, optional): Start date of the simulation.
        T_run (int, optional): Total number of days in plotting range.
        active_set_sizes (int or array-like, optional): Active set sizes for analytical.
        matrix (np.ndarray, optional): Simulation matrix for empirical probability.
        cutoff_date (datetime, optional): Start day for empirical probability.

    Usage examples:
        # Only analytical
        plot_extinction_probabilities(draws=..., start_date=..., t_max_date=..., T_run=..., active_set_sizes=...)

        # Only empirical
        plot_extinction_probabilities(matrix=..., cutoff_day=..., start_date=..., T_run=...)

        # Both
        plot_extinction_probabilities(draws=..., start_date=..., t_max_date=..., T_run=..., active_set_sizes=..., matrix=..., cutoff_day=...)
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    if cutoff_date is None or start_date is None or T_run is None:
        raise ValueError("cutoff_date, start_date and T_run must be specified")

    cutoff_day = (cutoff_date - start_date).days
    days = np.arange(cutoff_day, T_run + 1)

    x_vals = [start_date + timedelta(days=int(d)) for d in days]
    x_label = "Date"

    # --- Analytical Extinction Probability ---
    if draws is not None and active_set_sizes is not None:
        R0s = draws[:, 0]
        ks = draws[:, 1]
        rs = draws[:, 2]
        alphas = draws[:, 3]
        thetas = draws[:, 4]
        R_effs = R0s * rs

        quiet_windows_T = np.arange(1, len(days) + 1)
        prob_distributions = np.zeros((len(quiet_windows_T), len(R0s)))

        for i, T in enumerate(quiet_windows_T):
            F_T = gamma.cdf(T, a=alphas, scale=thetas)
            term1 = 1 + R_effs / ks
            term2 = 1 + (R_effs * F_T) / ks
            probs_for_this_T = np.full_like(term1, 1.0)
            valid_mask = term2 > 0
            probs_for_this_T[valid_mask] = (term1[valid_mask] / term2[valid_mask]) ** (-ks[valid_mask])
            prob_distributions[i, :] = np.clip(probs_for_this_T ** active_set_sizes, 0, 1)

        median_probs = np.median(prob_distributions, axis=1)
        lower_ci = np.percentile(prob_distributions, 35, axis=1)
        upper_ci = np.percentile(prob_distributions, 65, axis=1)

        ax.plot(x_vals, median_probs, lw=2, label="Analytical Median")
        ax.fill_between(x_vals, lower_ci, upper_ci, alpha=0.2, label="Analytical 30% CI")

    # --- Empirical (Simulation-based) Probability ---
    if matrix is not None:
        completion_prob = np.empty_like(days, dtype=float)
        for idx, d in enumerate(days):
            mask_after = np.arange(matrix.shape[1]) > d
            resurge = matrix[d + 1:, mask_after].sum()
            valid = matrix[:, mask_after].sum()
            completion_prob[idx] = 1.0 - (resurge / valid)
        ax.plot(x_vals, completion_prob, marker='.', lw=1.5, label="Empirical (Simulation)")

    # --- Finishing up ---
    ax.axhline(0.9, color='red', ls='--', lw=1, label="90% Threshold")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Probability of Extinction")
    ax.set_title("Probability of Infection Being Over")
    ax.grid(True, ls=':')
    ax.legend()
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.show()


#
# plot_analytical_extinction(draws.numpy(), start_date=sim.start_date, t_max_date=datetime(2025, 4, 17),
#                            T_run=sim.T_run, active_set_sizes=active_set_size.numpy())

plot_extinction_probabilities(draws=draws.numpy(), start_date=sim.start_date, T_run=sim.T_run,
                              active_set_sizes=active_set_size.numpy(), matrix=time_matrix.numpy(),
                              cutoff_date=datetime(2025, 4, 17))
