import os
import time
from datetime import datetime

import numpy as np
import scipy.stats.distributions as distns
from fitter import Fitter
from matplotlib import pyplot as plt

from python.eventide import (Parameters, IntervalCriterion, IndexOffspringCriterion, Simulator,
                             Scenario, ParameterChangePoint)
from python.eventide.collectors import DrawCollector

pars = Parameters(
    R0=(0.25, 15),
    k=(0.2, 10),
    r=(0.01, 0.99),
    alpha=(0.01, 20),
    theta=(0.01, 20)
).require('R0 * r < 10').require('1 < alpha * theta').require('alpha * theta < 50')

sampler = pars.create_latin_hypercube_sampler()
criteria = [
    IntervalCriterion(datetime(2025, 3, 3), datetime(2025, 4, 4), 8, 10),
    IntervalCriterion(datetime(2025, 4, 1), datetime(2025, 4, 4), 1, 11),
    IntervalCriterion(datetime(2025, 4, 4), datetime(2025, 4, 14), 0, 0),
    IntervalCriterion(datetime(2025, 4, 14), datetime(2025, 4, 17), 1, 1),
    IndexOffspringCriterion(2, 5)
]

collectors = [
    draw_collector := DrawCollector(),
    # time_matrix := TimeMatrix(datetime(2025, 4, 17)),
    # R0 := Hist1D('R0', range=pars.R0_range, bins=200),
    # r := Hist1D('r', range=pars.r_range, bins=200),
    # k := Hist1D('k', range=pars.k_range, bins=200),
    # alpha := Hist1D('alpha', range=pars.alpha_range, bins=200),
    # theta := Hist1D('theta', range=pars.theta_range, bins=200),
    # R0_r_product := Hist1D('r * R0', range=(0, 6), bins=200),
    # alpha_theta_product := Hist1D('alpha * theta', range=(0, 200), bins=200),
    # rR0_joint := Hist2D(('r', 'R0'), range=((0.01, 0.99), (0.25, 15.0)), bins=50),
    # joint := Hist2D(('r * R0', 'alpha * theta'), range=((0, 10), (0, 50)), bins=50)
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
    num_trajectories=100_000_000_000,
    chunk_size=100_000,
    T_run=48,
    max_cases=100,
    max_workers=12,
)

now = time.time()
sim.run()
print('Runtime:', time.time() - now)
# print('Trajectories accepted (time matrix):', time_matrix.numpy().sum())
# print('Trajectories accepted (param hist) :', R0.numpy().sum())

print(draw_collector.numpy().shape)
# print(draw_collector.numpy())
print('\n\n')

# assume draw_collector is already defined; flatten into 1D arrays
R0 = draw_collector.numpy()[:, 0].flatten()
k = draw_collector.numpy()[:, 1].flatten()
r = draw_collector.numpy()[:, 2].flatten()
alpha = draw_collector.numpy()[:, 3].flatten()
theta = draw_collector.numpy()[:, 4].flatten()
r_times_R0 = r * R0
alpha_times_theta = alpha * theta

variables = {
    'R0': R0,
    'k': k,
    'r': r,
    'alpha': alpha,
    'theta': theta,
    'r_times_R0': r_times_R0,
    'alpha_times_theta': alpha_times_theta
}


def empirical_cdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


# make sure output directory exists
os.makedirs('fits2', exist_ok=True)

for name, data in variables.items():
    # 1) fit with a 60 s timeout
    f = Fitter(data,
               distributions=None,
               bins=50,
               timeout=60)  # stop any individual fit after 60 s
    f.fit(max_workers=-1, progress=False)

    # 2) get and clean the error table
    df = f.df_errors.copy()
    df = df.loc[~df.index.str.startswith('_')]
    df = df.rename(columns={
        'sumsquare_error': 'sse',
        'ks_statistic': 'ks_stat',
        'ks_pvalue': 'ks_pvalue',
        'kl_div': 'kl_div'
    })
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3) compute empirical CDF once
    x_ecdf, y_ecdf = empirical_cdf(data)
    # pre-compute total sum of squares for R1
    tss = np.sum((y_ecdf - np.mean(y_ecdf)) ** 2)

    mses = []
    r1s = []
    params_list = []

    for dist_name in df.index:
        params = f.fitted_param.get(dist_name)
        params_list.append(params)
        if params is None:
            mses.append(np.nan)
            r1s.append(np.nan)
            continue

        try:
            cdf = getattr(distns, dist_name).cdf
            y_fit = cdf(x_ecdf, *params)
            mask = ~np.isnan(y_fit)
            mse = np.mean((y_ecdf[mask] - y_fit[mask]) ** 2)
            mses.append(mse)

            # compute SSE from the MSE
            sse = mse * np.sum(mask)
            # R1 = 1 - SSE/TSS
            r1 = 1 - sse / tss if tss > 0 else np.nan
            r1s.append(r1)
        except Exception:
            mses.append(np.nan)
            r1s.append(np.nan)

    df['mse'] = mses
    df['r1'] = r1s
    df['params'] = params_list

    # 4) drop any row where sse is NaN, then save CSV
    df = df.loc[~df['sse'].isna()]
    out = df.reset_index().rename(columns={'index': 'distribution'})
    csv_path = f'fits2/fit_small_support_results_{name}.csv'
    out.to_csv(csv_path, index=False)
    print(f"✔ Saved full fit results for '{name}' → {csv_path}")

    # 5) pick top-5 by R1 (largest)
    best5 = df['r1'].nlargest(5).index.tolist()

    plt.figure(figsize=(10, 6), dpi=300)
    # plot histogram
    plt.hist(data, bins=200, density=True, alpha=0.5,
             color='gray', label='data')
    x = np.linspace(data.min(), data.max(), 400)

    plotted = []
    for dist_name in best5:
        params = f.fitted_param.get(dist_name)
        if params is None:
            continue
        try:
            pdf = getattr(distns, dist_name).pdf
            y_pdf = pdf(x, *params)
            r1 = df.at[dist_name, 'r1']
            plt.plot(x, y_pdf, lw=2,
                     label=f"{dist_name} (R1={r1:.3f})")
            plotted.append(dist_name)
        except Exception:
            continue

    title_scores = " & ".join(
        [f"{d}: {df.at[d, 'r1']:.3f}" for d in plotted]
    ) if plotted else "no valid fits"
    plt.title(f"{name} — top {len(plotted)} by R1")
    plt.xlabel(name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    plot_path = f'fits2/fit_small_support_plot_{name}_best5_r1.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✔ Saved top‐5 R1 plot for '{name}' → {plot_path}")
