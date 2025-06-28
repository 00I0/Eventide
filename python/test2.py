import time
from datetime import datetime

import matplotlib.pyplot as plt

from python.eventide import (Parameters, IntervalCriterion, OffspringCriterion, Hist1D, Hist2D, Simulator,
                             Scenario, TimeMatrix)

start_date = datetime(2025, 3, 3)
pre_pause_date = datetime(2025, 4, 4)
cutoff_date = datetime(2025, 4, 17)
current_date = datetime(2025, 4, 27)

pre_pause_day = (pre_pause_date - start_date).days
cut_off_day = (cutoff_date - start_date).days
current_day = (current_date - start_date).days

pars = Parameters(
    R0=(0.25, 15),
    k=(0.2, 10),
    r=(0.01, 0.99),
    alpha=(0.01, 20),
    theta=(0.01, 20)
).require('R0 * r < 10').require('1 < alpha * theta').require('alpha * theta < 50')

sampler = pars.create_latin_hypercube_sampler()
criteria = [
    IntervalCriterion(0.0, pre_pause_day, 9, 11),
    IntervalCriterion(pre_pause_day - 3, pre_pause_day, 1, 11),
    IntervalCriterion(pre_pause_day, cut_off_day - 3, 0, 0),
    IntervalCriterion(cut_off_day - 3, cut_off_day, 1, 1),
    OffspringCriterion(2, 5)
]
T_run = 80
collectors = [
    time_matrix := TimeMatrix(T_run, cut_off_day),
    R0 := Hist1D('R0', range=(0.25, 15), bins=200),
    r := Hist1D('r', range=(0.01, 0.99), bins=200),
    k := Hist1D('k', range=(0.2, 10), bins=200),
    alpha := Hist1D('alpha', range=(0.01, 20), bins=200),
    theta := Hist1D('theta', range=(0.01, 20), bins=200),
    R0_r_product := Hist1D('r * R0', range=(0.01 * 0.25, 0.99 * 15), bins=200),
    alpha_theta_product := Hist1D('alpha * theta', range=(0.01 * 0.01, 20 * 20), bins=200),
    rR0_joint := Hist2D(('r', 'R0'), range=((0.01, 0.99), (0.25, 15.0)), bins=50)
]

sim = Simulator(
    sampler=sampler,
    scenario=Scenario([]),
    criteria=criteria,
    collectors=collectors,
    num_trajectories=1_000_000_000,
    chunk_size=100_000,
    T_run=T_run,
    max_cases=1000,
    max_workers=12,
    cutoff_day=cut_off_day
)

now = time.time()
sim.run()
print('Runtime:', time.time() - now)

print('Trajectories accepted (time matrix):', time_matrix.numpy().sum())
print('Trajectories accepted (param hist) :', R0.numpy().sum())

plt.figure(figsize=(5, 5))
plt.imshow(rR0_joint.numpy(), extent=[0.01, 0.99, 0.25, 15.0], aspect='auto', origin='lower')
plt.colorbar(label='Count')
plt.xlabel('r')
plt.ylabel('R0')
plt.title('Joint Heatmap: R0 vs r')
plt.tight_layout()
plt.show()
