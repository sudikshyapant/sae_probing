"""Visualisation helpers for SAE probing regime experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _mask(xs, ys):
    """Drop (x, y) pairs where y is NaN."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    valid  = ~np.isnan(ys)
    return xs[valid], ys[valid]


def plot_figure5(
    n_values,         scarcity_no_sae,  scarcity_sae,
    ratio_values,     imbalance_no_sae, imbalance_sae,
    fraction_values,  noise_logreg,     noise_sae,
    dataset_name: str = "154_athlete_sport_football",
    color: str = "#1f77b4",
    save_path: Path | None = None,
) -> plt.Figure:
    """Replicate Figure 5 of Kantamneni et al. (2025) for a single dataset.

    Layout: 2 rows × 3 columns
    - Top row    — Test AUC curves
                   (solid = non-SAE quiver / LogReg,  dashed = SAE quiver / SAE probe)
    - Bottom row — SAE Δ(AUC) = sae_auc − no_sae_auc with shaded fill

    Parameters
    ----------
    n_values        : x-axis for data-scarcity column (log-scaled)
    scarcity_no_sae : test AUC of non-SAE quiver at each n
    scarcity_sae    : test AUC of SAE quiver at each n
    ratio_values    : x-axis for class-imbalance column
    imbalance_no_sae, imbalance_sae : analogous for imbalance regime
    fraction_values : x-axis for label-noise column
    noise_logreg    : test AUC of bare LogReg at each corruption fraction
    noise_sae       : test AUC of SAE probe at each corruption fraction
    dataset_name    : used in the figure title
    color           : single colour for all lines (one dataset = one colour)
    save_path       : if given, figure is saved here at 150 dpi

    Returns
    -------
    matplotlib Figure
    """
    ALPHA = 0.15

    fig, axes = plt.subplots(
        2, 3, figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.45},
    )

    col_titles = ["Data Scarcity", "Class Imbalance", "Label Noise"]
    x_labels   = ["Number of Training Samples",
                  "Ratio of Positive Class",
                  "Fraction Corrupted"]
    xs_list    = [n_values,     ratio_values,     fraction_values]
    no_s_list  = [scarcity_no_sae, imbalance_no_sae, noise_logreg]
    sae_list   = [scarcity_sae,    imbalance_sae,    noise_sae]
    leg_no_sae = ["Non-SAE Quiver", "Non-SAE Quiver", "Logistic Regression"]
    leg_sae    = ["SAE Quiver",     "SAE Quiver",     "SAE Probe (k=128)"]

    for col in range(3):
        xs       = xs_list[col]
        no_sae   = no_s_list[col]
        with_sae = sae_list[col]

        # ── top: AUC curves ──────────────────────────────────────────────
        ax = axes[0, col]
        x_ns, y_ns = _mask(xs, no_sae)
        x_ws, y_ws = _mask(xs, with_sae)

        ax.plot(x_ns, y_ns, color=color, linewidth=2, linestyle="-",
                label=leg_no_sae[col])
        ax.plot(x_ws, y_ws, color=color, linewidth=2, linestyle="--",
                label=leg_sae[col])

        if col == 0:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_ylim(0.4, 1.05)
        ax.set_title(col_titles[col], fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.set_ylabel(f"Test AUC\n{dataset_name}", fontsize=8)

        # ── bottom: Δ AUC ────────────────────────────────────────────────
        ax_d  = axes[1, col]
        delta = np.asarray(with_sae, dtype=float) - np.asarray(no_sae, dtype=float)
        x_d, d_d = _mask(xs, delta)

        ax_d.axhline(0, color="black", linewidth=1, linestyle="--")
        ax_d.plot(x_d, d_d, color=color, linewidth=1.5)
        ax_d.fill_between(x_d, 0, d_d, alpha=ALPHA, color=color)

        if col == 0:
            ax_d.set_xscale("log")
            ax_d.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax_d.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax_d.set_ylabel("SAE Δ(AUC)", fontsize=9)

        ax_d.set_xlabel(x_labels[col], fontsize=9)
        ax_d.grid(True, alpha=0.3)

    fig.suptitle(
        f"Figure 5 (replication) — {dataset_name}\n"
        "Solid = non-SAE quiver | Dashed = SAE quiver  "
        "| Bottom row = SAE improvement",
        fontsize=10,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig
