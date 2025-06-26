import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import eventide

# 1) Define parameters (fixed or stochastic)
params = [
    eventide.Parameter("R0", 0.25, 15),
    eventide.Parameter("k", 0.2, 10),
    eventide.Parameter("r", 0.01, 0.99),
    eventide.Parameter("alpha", 0.01, 20),
    eventide.Parameter("theta", 0.01, 20),
]

sampler = eventide.LatinHypercubeSampler(params, scramble=True)

# cp_set = eventide.ParameterChangePoint(30.0, "r", 1.0)
# cp_restore = eventide.ParameterChangePoint(60.0, "r")
# scenario = eventide.Scenario([cp_set, cp_restore])
scenario = eventide.Scenario([])

# 3) Acceptance criteria

start_date = datetime(2025, 3, 3)
pre_pause_date = datetime(2025, 4, 4)
cutoff_date = datetime(2025, 4, 17)
current_date = datetime(2025, 4, 27)

pre_pause_day = (pre_pause_date - start_date).days
cut_off_day = (cutoff_date - start_date).days
current_day = (current_date - start_date).days

IC1 = eventide.IntervalCriterion(0.0, pre_pause_day, 9, 11)
IC2 = eventide.IntervalCriterion(pre_pause_day - 3, pre_pause_day, 1, 11)
IC3 = eventide.IntervalCriterion(pre_pause_day, cut_off_day - 3, 0, 0)
IC4 = eventide.IntervalCriterion(cut_off_day - 3, cut_off_day, 1, 1)
offs = eventide.OffspringCriterion(2, 5)
criteria = [offs, IC1, IC2, IC3, IC4]

# 4) Data collectors
T_run = 80
N_TRAJ = 100_000_000
tm_col = eventide.TimeMatrixCollector(T_run, cut_off_day)
ph_col = eventide.DrawHistogramCollector(params, bins=200)
jh_col = eventide.JointHeatmapCollector(
    R0_min=0.25, R0_max=15.0,
    r_min=0.01, r_max=0.99,
    bins=50
)
dm1_col = eventide.DerivedMarginalCollector(
    eventide.Product.R0_r, min=0.0, max=10.0, bins=200
)
dm2_col = eventide.DerivedMarginalCollector(
    eventide.Product.AlphaTheta, min=1.0, max=50.0, bins=200
)

collectors = [tm_col, ph_col, jh_col, dm1_col, dm2_col]

# 5) Run simulation
sim = eventide.Simulator(
    sampler=sampler,
    scenario=scenario,
    criteria=criteria,
    collectors=collectors,
    num_trajectories=N_TRAJ,
    chunk_size=100_000,
    T_run=T_run,
    max_cases=1000,
    max_workers=12,
    cutoff_day=cut_off_day
)
now = time.time()
sim.run()
print('Runtime: ', time.time() - now)

# 6) Retrieve results into NumPy arrays
time_matrix = np.array(tm_col.matrix())  # shape (81×81)
param_hist = np.array(ph_col.histogram())  # shape (5×200)
heatmap_2d = np.array(jh_col.heatmap())  # shape (50×50)
marg_rR0 = np.array(dm1_col.histogram())  # shape (200,)
marg_alphaTheta = np.array(dm2_col.histogram())  # shape (200,)

accepted = time_matrix.sum()
print(f"Trajectories accepted (time matrix): {accepted:_} / {N_TRAJ:_}")
print(f"Trajectories accepted (param hist): {param_hist.sum(axis=1)}")
if accepted == 0:
    print("⚠️  All collectors are zero. Try relaxing or removing your acceptance criteria.")

# --- A) Plot time‐matrix ---
plt.figure(figsize=(6, 5))
plt.imshow(time_matrix,
           origin='lower',
           extent=[0, T_run, 0, T_run],
           aspect='auto')
plt.colorbar(label='Count of trajectories')
plt.xlabel('First infection after cutoff (days)')
plt.ylabel('Final infection time (days)')
plt.title('Time‐Matrix: Final vs. First‐After‐Cutoff')
plt.tight_layout()
plt.show()

# --- B) Plot joint heatmap of R0 vs r ---
plt.figure(figsize=(5, 5))
plt.imshow(heatmap_2d,
           origin='lower',
           extent=[0.01, 0.99, 0.25, 15.0],  # [r_min, r_max, R0_min, R0_max]
           aspect='auto')
plt.colorbar(label='Count')
plt.xlabel('r')
plt.ylabel('R0')
plt.title('Joint Heatmap: R0 vs r')
plt.tight_layout()
plt.show()

# --- C) Combine all 7 histograms into one figure ---
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()

# 1–5: Parameter histograms
param_names = [p.name for p in sampler.parameters()]
for i, name in enumerate(param_names):
    ax = axes[i]
    counts = param_hist[i]
    bins = counts.shape[0]
    lo, hi = params[i].min, params[i].max
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts, width=(edges[1] - edges[0]))
    ax.set_title(name)
    ax.set_xlabel(name)
    ax.set_ylabel('Count')

# 6: Derived marginal R0 × r
ax = axes[5]
counts = marg_rR0
bins = counts.shape[0]
edges = np.linspace(0.0, 10.0, bins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
ax.bar(centers, counts, width=(edges[1] - edges[0]))
ax.set_title('R0 × r')
ax.set_xlabel('R0 × r')
ax.set_ylabel('Count')

# 7: Derived marginal α × θ
ax = axes[6]
counts = marg_alphaTheta
bins = counts.shape[0]
edges = np.linspace(1.0, 50.0, bins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
ax.bar(centers, counts, width=(edges[1] - edges[0]))
ax.set_title('α × θ')
ax.set_xlabel('α × θ')
ax.set_ylabel('Count')

# Turn off the unused subplots
for j in (7, 8):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
