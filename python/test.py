import time
from datetime import datetime

from python.eventide import Parameters, Simulator, Scenario, ParameterChangePoint, IntervalCriterion, \
    IndexOffspringCriterion, Hist1D, Hist2D, InfectionTimeCollector, DrawCollector, ActiveSetSizeCollector, TimeMatrix
from python.plots import plot_extinction, plot_histogram_grid, HistSpec, plot_cumulative_infections

pars = Parameters(
    R0=(0.25, 15),
    k=(0.2, 10),
    r=(0.01, 0.99),
    alpha=(0.01, 20),
    theta=(0.01, 20)
).require('R0 * r < 10').require('1 < alpha * theta').require('alpha * theta < 50')

sim = Simulator(
    parameters=pars,
    sampler=pars.create_latin_hypercube_sampler(),
    start_date=datetime(2025, 3, 3),
    scenario=Scenario([
        ParameterChangePoint('R0', datetime(2025, 4, 14), '0.5 * R0'),  # rábapordány
        ParameterChangePoint('R0', datetime(2025, 4, 17))
    ]),
    criteria=[
        IntervalCriterion(datetime(2025, 3, 3), datetime(2025, 4, 4), 8, 10),
        IntervalCriterion(datetime(2025, 4, 1), datetime(2025, 4, 4), 1, 11),
        IntervalCriterion(datetime(2025, 4, 4), datetime(2025, 4, 14), 0, 0),
        IntervalCriterion(datetime(2025, 4, 14), datetime(2025, 4, 17), 1, 1),
        IndexOffspringCriterion(2, 5)
    ],
    collectors=[
        R0 := Hist1D('R0', range=pars.R0_range, bins=50),
        r := Hist1D('r', range=pars.r_range, bins=50),
        k := Hist1D('k', range=pars.k_range, bins=50),
        alpha := Hist1D('alpha', range=pars.alpha_range, bins=200),
        theta := Hist1D('theta', range=pars.theta_range, bins=200),
        R0_r_product := Hist1D('r * R0', range=(0, 6), bins=200),
        alpha_theta_product := Hist1D('alpha * theta', range=(0, 200), bins=200),
        rR0_joint := Hist2D(('r', 'R0'), range=((0.01, 0.99), (0.25, 15.0)), bins=50),
        joint := Hist2D(('r * R0', 'alpha * theta'), range=((0, 10), (0, 50)), bins=50),
        infection_times := InfectionTimeCollector(),
        draws := DrawCollector(),
        active_set_size := ActiveSetSizeCollector(datetime(2025, 4, 17)),
        time_matrix := TimeMatrix(datetime(2025, 4, 17))
    ],
    num_trajectories=100_000_000,
    chunk_size=100_000,
    T_run=80,
    max_cases=100,
    max_workers=12,
)

now = time.time()
sim.run()
print('Runtime:', time.time() - now)

plot_histogram_grid([R0,
                     HistSpec(collector=R0, show_mean=True, show_conf=True, conf_level=0.6, show_median=True,
                              bar_color='skyblue', bar_alpha=0.7)
                    .overlay(HistSpec(dist_name='gamma', line_color='C3', line_alpha=0.9, show_R1=True)),
                     r, k, alpha, theta, R0_r_product, alpha_theta_product, rR0_joint, joint])

plot_cumulative_infections(infection_times, sim.start_date, sim.end_date, show_mean=True,
                           perc_bands=(.05, .35, .8), obs_points=[
        (datetime(2025, 3, 6), 1),
        (datetime(2025, 3, 21), 3),
        (datetime(2025, 3, 25), 1),
        (datetime(2025, 3, 26), 1),
        (datetime(2025, 3, 30), 1),
        (datetime(2025, 4, 2), 2),
        (datetime(2025, 4, 17), 1)
    ])

print(time_matrix.cutoff_day(sim.start_date))
print(sim.T_run)
plot_extinction(sim.start_date, time_matrix.cutoff_date, sim.T_run, analytical=(draws, active_set_size),
                analytical_colors='viridis',
                empirical=time_matrix)
