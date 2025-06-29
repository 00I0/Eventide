import time
from datetime import datetime

from python.eventide import (Parameters, IntervalCriterion, IndexOffspringCriterion, Hist1D, Hist2D, Simulator,
                             Scenario, TimeMatrix, ParameterChangePoint)
from python.plot_helpers import create_histograms, create_end_of_outbreak_plot

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
    R0 := Hist1D('R0', range=(0.25, 15), bins=200),
    r := Hist1D('r', range=(0.01, 0.99), bins=200),
    k := Hist1D('k', range=(0.2, 10), bins=200),
    alpha := Hist1D('alpha', range=(0.01, 20), bins=200),
    theta := Hist1D('theta', range=(0.01, 20), bins=200),
    R0_r_product := Hist1D('r * R0', range=(0.01 * 0.25, 10), bins=200),
    alpha_theta_product := Hist1D('alpha * theta', range=(1, 50), bins=200),
    rR0_joint := Hist2D(('r', 'R0'), range=((0.01, 0.99), (0.25, 15.0)), bins=50),
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
    T_run=80,
    max_cases=1000,
    max_workers=12,
)

now = time.time()
sim.run()
print('Runtime:', time.time() - now)
print('Trajectories accepted (time matrix):', time_matrix.numpy().sum())
print('Trajectories accepted (param hist) :', R0.numpy().sum())

create_histograms(collectors[1:])
create_end_of_outbreak_plot(time_matrix.numpy(), time_matrix.cutoff_day(sim.start_date), sim.start_date, sim.T_run)
