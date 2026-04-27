import numpy as np
from matplotlib import pyplot as plt

from python.plots.extinct_probs import empirical_uncond_over, analytic_uncond_over, empirical_cond_over, \
    analytic_cond_over, rao_blackwell_uncond_over_post_full, rb_cond_components_post
from python.plots.style import _use_style


def pointwise_hpd_from_draws(draws, level=0.95):
    draws = np.asarray(draws, float)
    M, nT = draws.shape
    q = max(1, int(np.floor(level * M)))
    lo = np.empty(nT, float);
    hi = np.empty(nT, float)
    for j in range(nT):
        col = np.sort(draws[:, j])
        widths = col[q - 1:] - col[:M - q + 1]
        idx = int(np.argmin(widths))
        lo[j] = col[idx];
        hi[j] = col[idx + q - 1]
    return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)


def simultaneous_band_from_draws(draws, level=0.95, eps=1e-12, center='mean'):
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError(f"`draws` must be 2D (M, nT); got shape {draws.shape}")
    M, nT = draws.shape
    if isinstance(center, str) and center == 'mean':
        mu = draws.mean(axis=0)
    else:
        mu = np.asarray(center, dtype=float)
        if mu.shape != (nT,):
            raise ValueError(f"`center` must have shape (nT,), got {mu.shape}")
    sd = draws.std(axis=0, ddof=1) if M > 1 else np.zeros(nT, dtype=float)
    sd = np.maximum(sd, eps)
    tvals = np.max(np.abs((draws - mu) / sd), axis=1)
    alpha = 1.0 - float(level)
    c = float(np.quantile(tvals, 1.0 - alpha))
    lo = np.clip(mu - c * sd, 0.0, 1.0)
    hi = np.clip(mu + c * sd, 0.0, 1.0)
    inside = np.all((draws >= lo) & (draws <= hi), axis=1)
    realized = float(np.mean(inside))
    return mu, (lo, hi), c, realized


def plot_all_with_probability_bands(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_grid, t_star, h=0.2, H_pad=10.0,
        show_pointwise_hpd=False, level_hpd=0.95,
        show_simultaneous=False, level_sim=0.95,
):
    sty = _use_style(None)

    pU_emp, _ = empirical_uncond_over(infection_times, T_grid, t_star)
    pU_ana = analytic_uncond_over(stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=h, U_max=T_grid[-1])
    pC_emp, _ = empirical_cond_over(infection_times, T_grid, t_star)
    pC_ana = analytic_cond_over(stopped_pairs, T_grid, t_star)

    T_fine, pU_rb_mean_exact, gU_uncond_full = rao_blackwell_uncond_over_post_full(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas,
        T_max=float(T_grid[-1]), t_star=t_star, h=h, H_pad=H_pad
    )
    pU_rb_mean = np.interp(T_grid, T_fine, pU_rb_mean_exact)
    pU_draws = np.vstack([np.interp(T_grid, T_fine, row) for row in gU_uncond_full])

    gC_inf_post, gC_quiet_post = rb_cond_components_post(
        infection_times, stopped_pairs, R0s, rs, ks, alphas, thetas, T_grid, t_star, h=h
    )
    pC_rb_mean = gC_inf_post.mean() / gC_quiet_post.mean(axis=0)
    pC_draws = gC_inf_post[:, None] / gC_quiet_post

    U_lo_hpd = U_hi_hpd = C_lo_hpd = C_hi_hpd = None
    U_lo_sim = U_hi_sim = U_c = U_real = None
    C_lo_sim = C_hi_sim = C_c = C_real = None
    if show_pointwise_hpd:
        U_lo_hpd, U_hi_hpd = pointwise_hpd_from_draws(pU_draws, level=level_hpd)
        C_lo_hpd, C_hi_hpd = pointwise_hpd_from_draws(pC_draws, level=level_hpd)
    if show_simultaneous:
        _, (U_lo_sim, U_hi_sim), U_c, U_real = simultaneous_band_from_draws(
            pU_draws, level=level_sim, center=pU_rb_mean
        )
        _, (C_lo_sim, C_hi_sim), C_c, C_real = simultaneous_band_from_draws(
            pC_draws, level=level_sim, center=pC_rb_mean
        )

    COL_EMP = sty.palette["EMP"]
    COL_ANA = sty.palette["ANA"]
    COL_RB = sty.palette["RB"]
    COL_SIM = sty.palette["SIM"]
    COL_HPD = sty.palette["HPD"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=sty.fig_pair,
        sharex=True, sharey=True,
        dpi=sty.dpi,
        # constrained_layout=True
    )

    # Unconditional
    ax = axes[0]
    ax.plot(T_grid, pU_emp, color=COL_EMP, lw=sty.lw_emp, ls=sty.dash_empirical,
            dash_capstyle='butt', solid_capstyle='butt',
            label=r"$\widehat p_{\mathrm{uncond}}^{(\mathrm{E})}(T)$")
    ax.plot(T_grid, pU_ana, color=COL_ANA, lw=sty.lw_ana, solid_capstyle='butt',
            label=r"$ p_{\mathrm{uncond}}^{(\mathrm{SF})}(T)$ ")
    if show_simultaneous and U_lo_sim is not None:
        ax.fill_between(T_grid, U_lo_sim, U_hi_sim, color=COL_SIM, alpha=0.16, linewidth=0,
                        label=f"RB Simul. {int(100 * level_sim)}%")
    if show_pointwise_hpd and U_lo_hpd is not None:
        ax.fill_between(T_grid, U_lo_hpd, U_hi_hpd, color=COL_HPD, alpha=0.24, linewidth=0,
                        label=f"RB HPD {int(100 * level_hpd)}%")
    ax.plot(T_grid, pU_rb_mean, color=COL_RB, lw=sty.lw_rb,
            label=r"$ p_{\mathrm{uncond}}^{(\mathrm{RB})}(T)$")
    ax.set_title(r"Unconditional")
    ax.set_xlabel(r"$T$ days since $t_\star$")
    ax.set_ylabel("Probability")
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=False)

    # Conditional
    ax = axes[1]
    ax.plot(T_grid, pC_emp, color=COL_EMP, lw=sty.lw_emp, ls=sty.dash_empirical,
            dash_capstyle='butt', solid_capstyle='butt',
            label=r"$\widehat p_{\mathrm{cond}}^{(\mathrm{E})}(T)$")
    ax.plot(T_grid, pC_ana, color=COL_ANA, lw=sty.lw_ana, solid_capstyle='butt',
            label=r"$ p_{\mathrm{cond}}^{(\mathrm{SF})}(T)$")
    if show_simultaneous and C_lo_sim is not None:
        ax.fill_between(T_grid, C_lo_sim, C_hi_sim, color=COL_SIM, alpha=0.16, linewidth=0,
                        label=f"RB Simul. {int(100 * level_sim)}%")
    if show_pointwise_hpd and C_lo_hpd is not None:
        ax.fill_between(T_grid, C_lo_hpd, C_hi_hpd, color=COL_HPD, alpha=0.24, linewidth=0,
                        label=f"RB HPD {int(100 * level_hpd)}%")
    ax.plot(T_grid, pC_rb_mean, color=COL_RB, lw=sty.lw_rb,
            label=r"$ p_{\mathrm{cond}}^{(\mathrm{RB})}(T)$")
    ax.set_title(r"Conditional")
    ax.set_xlabel(r"$T$ days since $t_\star$")
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=False)
    for a in axes:
        a.minorticks_on()

    plt.show()

    return dict(
        uncond=dict(emp=pU_emp, ana=pU_ana, rb_mean=pU_rb_mean, draws=pU_draws),
        cond=dict(emp=pC_emp, ana=pC_ana, rb_mean=pC_rb_mean, draws=pC_draws),
    )
