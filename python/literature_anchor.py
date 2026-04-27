import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import loggamma
from scipy.stats import gamma as gamma_dist

from python.eventide import DrawCollector, Scenario, \
    Simulator, Parameters, Hist1D, Hist2D, IndexOffspringCriterion
from python.eventide_viz import HistSpec, plot_histogram_grid
from python.optimize_acceptance_windows import build_acceptance_inequalities

# plt.style.use('fivethirtyeight')
if False:
    mpl.use('pgf')
    mpl.rcParams.update({
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.texsystem': 'pdflatex',
        'pgf.preamble': r'''
            \usepackage{amsmath}
            \usepackage{siunitx}
            \usepackage[T1]{fontenc}
            \usepackage{lmodern}
        ''',

        # Fonts
        'font.family': 'serif',  # or 'sans-serif' to match your journal
        'font.size': 12,
        # 'axes.titlesize': 8,
        # 'axes.labelsize': 8,
        # 'legend.fontsize': 7,
        # 'xtick.labelsize': 7,
        # 'ytick.labelsize': 7,

        # Lines & ticks (subtle but crisp)
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

        # Grid (light, y-only by default)
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3,

        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Save tight
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.unicode_minus': False,

        'axes.prop_cycle': mpl.cycler(color=['#4C72B0', '#333333', '#E69F00']),
    })


# ------------------------------------------------------------
# Support-based (HDI/shortest) interval utilities
# ------------------------------------------------------------
def support_interval(x, mass=0.95):
    """
    Shortest (highest-density) interval covering `mass` of x.
    Returns (low, high). Works on 1D arrays of samples.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    xs = np.sort(x)
    n = xs.size
    if n == 1:
        return xs[0], xs[0]
    m = int(np.floor(mass * n))
    m = max(1, min(m, n - 1))
    # sliding window of width m
    widths = xs[m:] - xs[:n - m]
    j = np.argmin(widths)
    return xs[j], xs[j + m]


def _fmt_pct(x):
    """
    Format a percentile like:
      2.5  -> "02.5"
      25   -> "25"
      97.5 -> "97.5"
    """
    s = f"{x:.1f}".rstrip("0").rstrip(".")
    if x < 10 and not s.startswith("0"):
        s = "0" + s
    return s


def add_equal_tailed_and_sbp(vec, label, percents=(0.50, 0.95)):
    """
    percents: iterable of central masses, e.g. (0.50, 0.65, 0.95).
              For each p, we report:
                - equal-tailed quantiles at (1-p)/2 and 1-(1-p)/2
                - smallest-support (HDI) interval of mass p
    """
    vec = np.asarray(vec)
    out = {
        "metric": label,
        "median": np.quantile(vec, 0.5),
    }

    # Equal-tailed quantiles for each requested central mass p
    # lower = (1-p)/2, upper = 1 - lower
    for p in percents:
        lower = 0.5 * (1.0 - p)
        upper = 1.0 - lower
        q_lo = np.quantile(vec, lower)
        q_hi = np.quantile(vec, upper)
        out[f"q{_fmt_pct(100 * lower)}"] = q_lo
        out[f"q{_fmt_pct(100 * upper)}"] = q_hi

    # Smallest-support (HDI) intervals
    for p in percents:
        lo, hi = support_interval(vec, mass=p)
        tag = int(round(100 * p))
        out[f"sbp{tag}_low"] = lo
        out[f"sbp{tag}_high"] = hi

    return out


# ------------------------------------------------------------
# Negative Binomial helpers (mean=Rbar, dispersion=k)
# ------------------------------------------------------------
def nb_pmf(n, Rbar, k):
    n = np.asarray(n)
    Rbar = np.asarray(Rbar)
    k = np.asarray(k)
    logC = loggamma(n + k) - loggamma(k) - loggamma(n + 1.0)
    logp0 = k * np.log(k / (k + Rbar))
    logp1 = n * np.log(Rbar / (k + Rbar))
    return np.exp(logC + logp0 + logp1)


def nb_p_zero(Rbar, k):
    return (k / (k + Rbar)) ** k


def nb_tail_prob_ge_m(m, Rbar, k):
    ns = np.arange(m)
    pmfs = np.vstack([nb_pmf(n, Rbar, k) for n in ns])
    return 1.0 - pmfs.sum(axis=0)


def nb_prob_between(a, b, Rbar, k):
    ns = np.arange(a, b + 1)
    pmfs = np.vstack([nb_pmf(n, Rbar, k) for n in ns])
    return pmfs.sum(axis=0)


# ------------------------------------------------------------
# Gamma generation-interval helpers (shape=alpha, scale=theta)
# ------------------------------------------------------------
def gamma_quantiles(alpha, theta, qs):
    return np.vstack([gamma_dist.ppf(q, a=alpha, scale=theta) for q in qs])


# Euler–Lotka growth rate for Gamma GI:
# Rbar * (1 + theta*g)^(-alpha) = 1  ->  g = ((Rbar)^(1/alpha) - 1)/theta
def euler_lotka_growth(Rbar, alpha, theta):
    return ((Rbar ** (1.0 / alpha)) - 1.0) / theta


# ------------------------------------------------------------
# Main summaries from accepted draws
# ------------------------------------------------------------
def compute_summaries_with_sbp(draws, percents=(0.50, 0.95)):
    """
    draws: (N_accept x 5) in order [R0, k, r, alpha, theta]
    percents: iterable of central masses to report (e.g., [0.5, 0.65, 0.95])
      For each p in percents, we report:
        - equal-tailed quantiles at (1-p)/2 and 1-(1-p)/2
        - smallest-support (HDI) interval of mass p
    Returns:
      - summary DataFrame with equal-tailed quantiles AND support-based (HDI) intervals
      - per_draw dict of useful vectors
    """
    draws = np.asarray(draws)
    R0 = draws[:, 0]
    k = draws[:, 1]
    r = draws[:, 2]
    alpha = draws[:, 3]
    theta = draws[:, 4]

    Re = r * R0
    red = 1.0 - r
    pr_subcritical = np.mean(Re < 1)

    # Generation-time derived per draw
    gt_mean = alpha * theta
    gt_sd = np.sqrt(alpha) * theta
    gt_cv = 1.0 / np.sqrt(alpha)

    q25, q50, q75 = gamma_quantiles(alpha, theta, qs=[0.25, 0.5, 0.75])
    p_le_7 = gamma_dist.cdf(7.0, a=alpha, scale=theta)
    p_le_14 = gamma_dist.cdf(14.0, a=alpha, scale=theta)

    # Overdispersion functionals (index vs post-control)
    p0_idx = nb_p_zero(R0, k)
    tail3_idx = nb_tail_prob_ge_m(3, R0, k)
    tail5_idx = nb_tail_prob_ge_m(5, R0, k)
    p_3to5_idx = nb_prob_between(3, 5, R0, k)

    p0_ctl = nb_p_zero(Re, k)
    tail3_ctl = nb_tail_prob_ge_m(3, Re, k)
    tail5_ctl = nb_tail_prob_ge_m(5, Re, k)

    g_R0 = euler_lotka_growth(R0, alpha, theta)
    g_Re = euler_lotka_growth(Re, alpha, theta)

    rows = []
    # Reproduction numbers
    rows.append(add_equal_tailed_and_sbp(R0, "R0 (index, pre-control)", percents))
    rows.append(add_equal_tailed_and_sbp(Re, "Re = r*R0 (post-control)", percents))
    rows.append({"metric": "Pr(Re < 1)", "median": pr_subcritical})
    rows.append(add_equal_tailed_and_sbp(red, "Reduction (1 - r)", percents))

    # Generation-time stats
    rows.append(add_equal_tailed_and_sbp(gt_mean, "Gen time mean (days)", percents))
    rows.append(add_equal_tailed_and_sbp(gt_sd, "Gen time SD (days)", percents))
    rows.append(add_equal_tailed_and_sbp(gt_cv, "Gen time CV", percents))
    rows.append(add_equal_tailed_and_sbp(q50, "Gen time median (days)", percents))
    rows.append(add_equal_tailed_and_sbp(q75 - q25, "Gen time IQR (days)", percents))
    rows.append(add_equal_tailed_and_sbp(p_le_7, "P(W ≤ 7 days)", percents))
    rows.append(add_equal_tailed_and_sbp(p_le_14, "P(W ≤ 14 days)", percents))

    # Overdispersion functionals at index
    rows.append(add_equal_tailed_and_sbp(p0_idx, "Index: P(N=0) at R0", percents))
    rows.append(add_equal_tailed_and_sbp(tail3_idx, "Index: P(N≥3) at R0", percents))
    rows.append(add_equal_tailed_and_sbp(tail5_idx, "Index: P(N≥5) at R0", percents))
    rows.append(add_equal_tailed_and_sbp(p_3to5_idx, "Index: P(3≤N≤5) at R0", percents))

    # Overdispersion functionals post-control
    rows.append(add_equal_tailed_and_sbp(p0_ctl, "Post-control: P(N=0) at Re", percents))
    rows.append(add_equal_tailed_and_sbp(tail3_ctl, "Post-control: P(N≥3) at Re", percents))
    rows.append(add_equal_tailed_and_sbp(tail5_ctl, "Post-control: P(N≥5) at Re", percents))

    # Growth/decay rates
    rows.append(add_equal_tailed_and_sbp(g_R0, "Euler–Lotka growth rate at R0 (1/day)", percents))
    rows.append(add_equal_tailed_and_sbp(g_Re, "Euler–Lotka growth rate at Re (1/day)", percents))

    summary = pd.DataFrame(rows)

    per_draw = {
        "R0": R0, "Re": Re, "Pr_Re_lt_1": (Re < 1).astype(float), "reduction": red,
        "gt_mean": gt_mean, "gt_sd": gt_sd, "gt_cv": gt_cv,
        "gt_median": q50, "gt_iqr": (q75 - q25),
        "P_W_le_7": p_le_7, "P_W_le_14": p_le_14,
        "idx_P_N_eq_0": p0_idx, "idx_P_N_ge_3": tail3_idx, "idx_P_N_ge_5": tail5_idx,
        "idx_P_3to5": p_3to5_idx,
        "ctl_P_N_eq_0": p0_ctl, "ctl_P_N_ge_3": tail3_ctl, "ctl_P_N_ge_5": tail5_ctl,
        "g_R0": g_R0, "g_Re": g_Re,
        "alpha": alpha, "theta": theta, "k": k, "r": r
    }
    return summary, per_draw


# ------------------------------------------------------------
def main():
    pars = (
        Parameters(
            R0=(0.25, 15),
            k=(0.2, 10),
            r=(0.01, 0.99),
            alpha=(0.01, 20),
            theta=(0.01, 10)
        ).require('R0 * r < 3')
        .require('2 < alpha * theta').require('alpha * theta < 26')
        .require('R0 / k < 1.2').require('(r * R0) / k < 0.4')
        .require('((r * R0) ^ (1/ alpha) - 1) / theta < 0.1')
        .require('1 / sqrt(alpha) >= 0.1').require('1 / sqrt(alpha) <= 1.0')
        .require('1.0 <= sqrt(alpha) * theta').require('sqrt(alpha) * theta <= 15')
        .require('(k/(k + r * R0))^k > 0.05').require('(k/(k + r * R0))^k < 0.95')
    )
    epsilon = timedelta(days=1)
    obs_points = [
        (datetime(2025, 3, 6), 1),
        (datetime(2025, 3, 21), 3),
        (datetime(2025, 3, 25), 1),
        (datetime(2025, 3, 26), 1),
        (datetime(2025, 3, 30), 1),
        (datetime(2025, 4, 2), 2),
        (datetime(2025, 4, 17), 1),
    ]
    sim_start = min(t for t, _ in obs_points)

    sim = Simulator(
        parameters=pars,
        sampler=pars.create_latin_hypercube_sampler(),
        start_date=datetime(2025, 3, 6),
        scenario=Scenario([
            # ParameterChangePoint('R0', datetime(2025, 4, 17) - epsilon, '0.5 * R0'),  # rábapordány
            # ParameterChangePoint('R0', datetime(2025, 4, 17) + epsilon, '0.5 * R0'),
        ]),
        criteria=[IndexOffspringCriterion(2, 5)] + build_acceptance_inequalities(
            obs_points=obs_points,
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
        )  # [
        #     IntervalCriterion(
        #         datetime(2025, 3, 6),
        #         datetime(2025, 4, 2) + epsilon,
        #         8, 10
        #     ),
        #     IntervalCriterion(
        #         datetime(2025, 4, 2) - epsilon,
        #         datetime(2025, 4, 2) + epsilon,
        #         1, 10
        #     ),
        #     IntervalCriterion(
        #         datetime(2025, 4, 2) + epsilon,
        #         datetime(2025, 4, 17) - epsilon,
        #         0, 0
        #     ),
        #     IntervalCriterion(
        #         datetime(2025, 4, 17) - epsilon,
        #         datetime(2025, 4, 17) + epsilon,
        #         1, 1
        #     ),
        #     IndexOffspringCriterion(2, 5)
        # ]
        ,
        collectors=[
            draw_collector := DrawCollector(),

            R0 := Hist1D('R0', range=pars.R0_range, ),
            r := Hist1D('r', range=pars.r_range, ),
            k := Hist1D('k', range=pars.k_range, ),
            alpha := Hist1D('alpha', range=pars.alpha_range, ),
            theta := Hist1D('theta', range=pars.theta_range, ),
            R0_r_product := Hist1D('r * R0', range=(0, 3.2), ),
            alpha_theta_product := Hist1D('alpha * theta', range=(0, 30), ),
            rR0_joint := Hist2D(('r', 'R0'), range=((0.01, 0.99), (0.25, 15.0)), ),
            joint := Hist2D(('r * R0', 'alpha * theta'), range=((0, 10), (0, 50)), ),

            Rr_over_k := Hist1D('R0*r / k', range=(0, 0.42), ),
            R_over_k := Hist1D('R0 / k', range=(0, 1.25), ),
            Rr_euler_lotka := Hist1D('((r * R0) ^ (1/ alpha) - 1) / theta ', range=(0, 0.115), ),
            R_euler_lotka := Hist1D('(R0 ^ (1/ alpha) - 1) / theta', range=(0, 1), ),

            interval_sd := Hist1D('sqrt(alpha) * theta', range=(0.5, 23), ),
            interval_cv := Hist1D('1/ sqrt(alpha)', range=(0, 3), ),
            interval_mode := Hist1D('(alpha - 1) * theta', range=(0, 25), ),

            offspring_heterogeneity_rR0 := Hist1D('(k/(k + R0*r))^k', range=(0, 1)),
            offspring_heterogeneity_R0 := Hist1D('(k/(k + R0))^k', range=(0, 0.5)),

            growth_elasticity_R0 := Hist1D('alpha * (R0^(1/alpha) - 1)', range=(0, 10), ),
            growth_elasticity_rR0 := Hist1D('alpha * ((r*R0)^(1/alpha) - 1)', range=(0, 1.5), ),

        ],
        num_trajectories=1_000_000_000,
        chunk_size=100_000,
        T_run=45,
        max_cases=100,
        max_workers=13,
    )

    now = time.time()
    sim.run()
    print('Runtime:', time.time() - now)

    fig_threshold = plot_histogram_grid([
        HistSpec(R0, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest', label=r'$R_0$', ),
        HistSpec(r, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest'),
        HistSpec(k, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest'),
        HistSpec(alpha, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest'),
        HistSpec(theta, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest'),

        HistSpec(R0_r_product, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest',
                 label=r'$rR_0$'),
        HistSpec(alpha_theta_product, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest',
                 label=r'$\alpha \theta$'),
        HistSpec(interval_sd, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest',
                 label=r'$\sqrt{\alpha} \theta$'),

        HistSpec(interval_cv, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest',
                 label=r'$\frac{1}{\sqrt{\alpha}}$'),

        HistSpec(R_over_k, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest',
                 label=r'$\frac{R_0}{k}$'),
        HistSpec(Rr_over_k, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest',
                 label=r'$\frac{rR_0}{k}$'),
        # HistSpec(R_euler_lotka, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest'),
        # HistSpec(Rr_euler_lotka, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest'),
        #

        # HistSpec(interval_cv, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest'),
        # HistSpec(interval_mode, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest'),
        #
        HistSpec(offspring_heterogeneity_rR0, show_median=True, show_conf=True, conf_level=0.95,
                 conf_method='shortest', label=r'$\left(\frac{k}{k + rR_0}\right)^k$'),
        # HistSpec(offspring_heterogeneity_R0, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest'),
        #
        # HistSpec(growth_elasticity_R0, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest'),
        HistSpec(Rr_euler_lotka, show_median=True, show_conf=True, conf_level=0.95, conf_method='shortest',
                 label=r'$\frac{(rR_0)^{1/\alpha}-1}{\theta}$'),

        # rR0_joint
    ], dpi=300, target_ratio=(np.sqrt(5) + 1) / 2)

    # fig_threshold.savefig('../img/threshold_distributions.pgf')
    plt.show()

    draws = np.asarray(draw_collector)
    summary_table, per_draw = compute_summaries_with_sbp(draws, percents=[0.50, 0.65, 0.95])
    print('Number of accepted trajectories:', draws.shape[0])
    print(summary_table.to_string(index=False))
    # summary_table.to_csv('data/summary_table_00010.csv')


if __name__ == '__main__':
    main()
