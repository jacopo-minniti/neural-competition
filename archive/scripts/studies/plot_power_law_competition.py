"""Plot the task law, prospective forecasts, width contrasts, and reopening."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def main():
    root = Path(__file__).resolve().parents[2] / "docs/evidence/power_law_competition"
    primary = json.loads((root / "summary.json").read_text())
    transfer = json.loads((root / "transfer32_summary.json").read_text())
    mechanism = json.loads((root / "mechanism_summary.json").read_text())
    blue, orange = "#2466A6", "#BD5B32"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
    ax = axes[0, 0]
    for count, alpha, color in [(24, 1.15, blue), (32, 1.4, orange)]:
        rank = np.arange(1, count + 1)
        q = rank.astype(float)**-alpha
        q *= 1.2 / q.sum()
        ax.loglog(rank, q, marker=".", color=color, label=f"{count} factors, exponent {alpha}")
    ax.set(title="Associated tasks retain exact power-law marginals",
           xlabel="Factor frequency rank", ylabel="Occurrence probability per example")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    values, lows, highs, labels = [], [], [], []
    for cohort, count in [(primary, 24), (transfer, 32)]:
        for case in ["prototype", "associated"]:
            row = next(r for r in cohort["forecast_scores"]
                       if (r["case"], r["method"], r["horizon"]) == (case, "bilinear", 16))
            values.append(row["response_error_ratio"])
            lo, hi = row["ratio_seed_bootstrap95"]
            lows.append(values[-1] - lo)
            highs.append(hi - values[-1])
            labels.append(f"{count} factors\n{'Prototype' if case == 'prototype' else 'Continuous'}")
    ax.bar(np.arange(4), values, color=[blue, blue, orange, orange], width=.58)
    ax.errorbar(np.arange(4), values, yerr=[lows, highs], fmt="none", color="#333333", capsize=4)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
    ax.set(title="Conditional response prediction transfers",
           ylabel="16-step RMSE / persistence RMSE", ylim=(0, 1.08), xticks=np.arange(4), xticklabels=labels)
    ax.text(.04, .84, "Lower is better; six-seed bootstrap intervals", transform=ax.transAxes, fontsize=9)
    ax = axes[1, 0]
    for case, color in [("prototype", blue), ("associated", orange)]:
        row = next(r for r in mechanism["width_contrasts"] if r["case"] == case and r["step"] == 2000)
        ax.plot([50, 100], [row["widths"][str(m)]["observed"] for m in [50, 100]],
                color=color, marker="o", label=f"{case.capitalize()}: observed")
        ax.plot([50, 100], [row["widths"][str(m)]["predicted"] for m in [50, 100]],
                color=color, marker="x", linestyle="--", label=f"{case.capitalize()}: predicted")
    ax.set(title="Width protection is conditional, not universal",
           xlabel="Width", ylabel="Probability of 10% weakening over 16 steps", xticks=[50, 100], ylim=(0, .01))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 1]
    recovery = mechanism["reversibility"]
    values = [r["conditional_reopening"] for r in recovery]
    ax.bar([0, 1], values, width=.5, color=[blue, orange])
    for index, value in enumerate(values):
        ax.text(index, value + .025, f"{value:.1%}", ha="center")
    ax.set(title="Many short-window availability losses reopen",
           ylabel="Reopened by step 16, given closed at step 4",
           xticks=[0, 1], xticklabels=["Prototype", "Associated continuous"], ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.text(.03, .93, r"Availability resolution $\beta_*=0.1$", transform=ax.transAxes, fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Many-factor competition: conditional forecasts and their limits", fontsize=14)
    fig.savefig(root / "summary.png", dpi=170)
    fig.savefig(root / "summary.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
