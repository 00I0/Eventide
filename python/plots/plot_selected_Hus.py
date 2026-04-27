import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from python.plots.extinct_probs import compute_H_grid
from python.plots.style import _use_style


def plot_selected_Hus(R0s, T_max: float, alphas, ks, rs, thetas):
    sty = _use_style(None)
    R_eff = R0s * rs
    df = pd.DataFrame({
        "R": R_eff,
        "k": ks,
        "alpha": alphas,
        "theta": thetas,
        "meanGI": alphas * thetas
    })

    R_targets = np.quantile(df["R"], [0.10, 0.50, 0.90])
    k_targets = np.quantile(df["k"], [0.25, 0.75])

    selected_idx = []
    for Rt in R_targets:
        for kt in k_targets:
            dist = (df["R"] - Rt) ** 2 + (df["k"] - kt) ** 2
            cand = dist.idxmin()
            selected_idx.append(cand)
    selected_idx = sorted(set(int(i) for i in selected_idx))

    def bgw_ext_prob(R, k, tol=1e-12, itmax=2000):
        if R <= 1.0:
            return 1.0
        beta = k / R
        q = 1.0
        for _ in range(itmax):
            q_new = (beta / (beta + 1.0 - q)) ** k
            if abs(q_new - q) < tol:
                return float(q_new)
            q = q_new
        return float(q)

    U_plot = T_max
    h_H = 0.2
    uu = np.arange(0, U_plot + 1e-9, h_H)

    fig, ax = plt.subplots(1, 1, figsize=sty.fig_single, constrained_layout=True, dpi=sty.dpi)
    palette = sty.cycle

    for j, idx in enumerate(selected_idx):
        R, k, a, th = float(df.at[idx, "R"]), float(df.at[idx, "k"]), float(df.at[idx, "alpha"]), float(
            df.at[idx, "theta"])
        H = compute_H_grid(R, k, a, th, U_max=U_plot, h=h_H)
        q = bgw_ext_prob(R, k)
        col = palette[j % len(palette)]
        ax.plot(uu, H, color=col, lw=sty.lw_ana,
                label=rf"$R_{{\mathrm{{eff}}}}={R:.2f},\ k={k:.2f},\ \alpha={a:4.2f},\ \theta={th:4.2f}$")
        H0 = (k / (k + R)) ** k
        ax.plot([0], [H0], marker='o', ms=sty.marker_size, color=col)
        ax.hlines(q, 0, U_plot, color=col, lw=1.0, linestyles=':', alpha=0.55)

    ax.set_xlabel(r"$u$ (days since seed)")
    ax.set_ylabel(r"$H(u)$")
    ax.set_xlim(0, U_plot)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(r"Single-seed finite-horizon survival $H(u)$")
    ax.legend(frameon=False, loc="lower right", ncol=1)
    ax.minorticks_on()
    plt.show()
