import time
from datetime import datetime
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from python.eventide import Simulator, Parameters, IntervalCriterion, Scenario, ParameterChangePoint, \
    IndexOffspringCriterion
from python.eventide.collectors import InfectionTimeCollector

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
    infection_times := InfectionTimeCollector()
    # time_matrix := TimeMatrix(datetime(2025, 4, 17)),
    # draws := DrawCollector(),
    # active_set_size := ActiveSetSizeCollector(datetime(2025, 4, 17))
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
print('Trajectories:', len(infection_times.numpy()))


def plot_cumulative_infections(infection_times, start_date, end_date, resolution=0.1, obs_points=None):
    """
    infection_times : List[List[float]]
        Each inner list is sorted (or unsorted) floats = days since start_date.
    start_date, end_date : datetime.datetime
        Absolute bounds of the plot.
    resolution : float
        Step in days for the internal time grid (e.g. 0.1 days).
    """
    # 1) Build fine time grid (days since start_date)
    total_days = (end_date - start_date).total_seconds() / 86400.0
    t_grid = np.arange(0, total_days + resolution, resolution)

    # 2) Compute cumulative counts for each trajectory
    cum = []
    for traj in infection_times:
        times = np.sort(np.asarray(traj))
        counts = np.searchsorted(times, t_grid, side='right')
        cum.append(counts)
    cum = np.vstack(cum)  # shape = (n_traj, n_time)

    N, T = cum.shape

    # 3) trimmed‐mean curve (drop ⌊0.25%×N⌋ from each tail)
    trim_n = int(np.floor(N * 0.0025))

    sorted_c = np.sort(cum, axis=0)
    trimmed = sorted_c[trim_n: N - trim_n, :]
    mean_curve = trimmed.mean(axis=0)

    # 3) Compute mean curve
    # mean_curve = cum.mean(axis=0)

    # 4) Compute "average of extremes" bands (90%, 60%, 30%)
    n_traj = trimmed.shape[0]
    intervals = [0.05, 0.20, 0.35]
    bands = {}
    for frac in intervals:
        k = max(1, int(np.ceil(n_traj * frac)))
        sorted_c = np.sort(trimmed, axis=0)
        lower = sorted_c[:k, :].mean(axis=0)
        upper = sorted_c[-k:, :].mean(axis=0)
        bands[frac] = (lower, upper)

    lo90, hi90 = bands[0.05]
    lo60, hi60 = bands[0.20]
    lo30, hi30 = bands[0.35]

    # 5) Convert grid to datetime objects
    dates = [start_date + timedelta(days=float(d)) for d in t_grid]

    # 6) Plot mean + bands
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(dates, lo90 + 1, hi90 + 1, alpha=0.2, label='90% band')
    ax.fill_between(dates, lo60 + 1, hi60 + 1, alpha=0.3, label='60% band')
    ax.fill_between(dates, lo30 + 1, hi30 + 1, alpha=0.4, label='30% band')
    ax.plot(dates, mean_curve + 1, color='black', lw=1.5, label='Mean')

    if obs_points:
        # sort by date
        obs_sorted = sorted(obs_points, key=lambda x: x[0])
        obs_dates, daily_counts = zip(*obs_sorted)
        # cumulative sum of the daily counts
        cum_counts = np.cumsum(daily_counts)
        ax.scatter(
            obs_dates,
            cum_counts + 1,
            color='red',
            marker='o',
            edgecolor='white',
            zorder=5,
            label='Observed cumulative'
        )

    # 8) Weekly ticks on the x‑axis
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    # 9) Labels, grid, legend
    ax.set_xlim(start_date, end_date)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative infections')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=False)

    plt.show()


plot_cumulative_infections(infection_times.numpy(), start_date=sim.start_date, end_date=datetime(2025, 6, 30),
                           obs_points=[
                               (datetime(2025, 3, 6), 1),
                               (datetime(2025, 3, 21), 3),
                               (datetime(2025, 3, 25), 1),
                               (datetime(2025, 3, 26), 1),
                               (datetime(2025, 3, 30), 1),
                               (datetime(2025, 4, 2), 2),
                               (datetime(2025, 4, 17), 1)
                           ])
