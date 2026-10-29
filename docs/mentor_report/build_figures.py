"""Report figures from retained full-cohort evidence, without new training."""

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "images"
TABLES = ROOT / "docs/tables"
BLUE, ORANGE, GREY = "#225b8b", "#c26b35", "#777777"
plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
    }
)

# Preserve the original whole-distribution figure, including its uncertainty bands.
(OUT / "distributions.png").write_bytes(
    subprocess.check_output(
        ["git", "show", "b03c53f:docs/figures/distributions/gradient_distributions_0.2.png"],
        cwd=ROOT,
    )
)

# The one-step comparison is Table 1; this figure tests cumulative cost instead.
f = pd.read_csv(TABLES / "history/effect_magnitude/cumulative_per_seed.csv")
f = f.query('view == "reference" and width in [8, 16]')
fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.9), layout="constrained")
for ax, condition, title in zip(
    axes,
    ["teacher8/momentum", "k8_teacher2027/adam_001"],
    ["Momentum: cost falls, fit improves", "Adam: cost rises, fit improves"],
):
    g = f[f.condition == condition].groupby("width")[["penalty", "final_loss"]].mean()
    x = g.loc[[8, 16], "penalty"].to_numpy()
    y = 2 * g.loc[[8, 16], "final_loss"].to_numpy()
    ax.annotate(
        "",
        xy=(x[1], y[1]),
        xytext=(x[0], y[0]),
        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1},
    )
    for i, (width, color) in enumerate([(8, BLUE), (16, ORANGE)]):
        ax.scatter(x[i], y[i], color=color, s=24, zorder=3)
        ax.annotate(
            f"Width {width}",
            (x[i], y[i]),
            xytext=(0, -12 if condition == "teacher8/momentum" and i == 0 else 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set(
        title=title,
        xlabel=r"Accumulated positive pair cost $C_+$",
        ylabel="Final reference MSE",
        yscale="log",
    )
    span = x.max() - x.min()
    ax.set_xlim(max(0, x.min() - 0.35 * span), x.max() + 0.35 * span)
    if condition == "teacher8/momentum":
        ax.set_ylim(1e-11, 3e-2)
    else:
        ax.set_ylim(0.03, 0.16)
        ax.set_yticks([0.04, 0.06, 0.1], labels=["0.04", "0.06", "0.10"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.savefig(OUT / "cumulative_cost.pdf", bbox_inches="tight")
plt.close(fig)

f = pd.read_csv(TABLES / "residual_growth/summary.csv")
f = f.query(
    'study=="residual_growth" and collection=="replication20" and degrees==30 and completed==10000 and arm in ["w8_fixed", "w32_fixed"]'
)
roles = pd.read_csv(TABLES / "residual_growth/roles.csv")
roles = roles.query(
    'collection=="replication20" and degrees==30 and ratio==-3 and budget==10000 and arm in ["w8_fixed", "w32_fixed"]'
)
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.25), layout="constrained")
for opt, marker in [("sgd", "o"), ("momentum", "s"), ("adam", "^")]:
    for w, color, shift in [(8, BLUE, -0.10), (32, ORANGE, 0.10)]:
        vals = [
            200
            * f[
                (f.optimizer == opt) & (f.ratio == ratio) & (f.arm == f"w{w}_fixed")
            ].mean_loss.item()
            for ratio in [3, -3]
        ]
        axes[0].plot(
            np.arange(2) + shift, vals, marker=marker, color=color, alpha=0.8, ls="--", lw=0.7
        )
axes[0].set(
    xticks=[0, 1],
    xticklabels=["Same sign (1, 3)", "Opposing (1, −3)"],
    yscale="log",
    ylabel="Test MSE / target variance (%)",
    title="Output signs change the width gap",
)
for w, color in [(8, BLUE), (32, ORANGE)]:
    axes[0].plot([], [], color=color, label=f"Width {w}")
axes[0].legend(frameon=False, fontsize=8)
for j, (w, color) in enumerate([(8, BLUE), (32, ORANGE)]):
    vals = [
        roles[(roles.optimizer == opt) & (roles.width == w)].final_positive.mean()
        for opt in ["sgd", "momentum", "adam"]
    ]
    axes[1].bar(np.arange(3) + (j - 0.5) * 0.28, vals, width=0.26, color=color)
axes[1].axhline(4, color=GREY, ls="--", lw=1)
axes[1].text(
    0.97, 4.35, "Required: 4", ha="right", fontsize=8, transform=axes[1].get_yaxis_transform()
)
axes[1].set(
    xticks=range(3),
    xticklabels=["SGD", "Mom.", "Adam"],
    ylim=(0, 16),
    ylabel="Mean positive-output units",
    title="Positive-role availability",
)
fig.savefig(OUT / "signed_roles.pdf", bbox_inches="tight")
plt.close(fig)

f = pd.read_csv(TABLES / "feature_profile/endpoints.csv")
f = f[f.collection == "main"]
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), layout="constrained", sharey=True)
for ax, opt in zip(axes, ["sgd", "momentum", "adam"]):
    g = f[f.optimizer == opt]
    vals = [g[g.arm == arm].test_loss.mean() * 200 for arm in ["ordinary8", "early", "late"]]
    ax.plot(range(3), vals, "o-", color=BLUE)
    wide = g[g.arm == "ordinary32"].test_loss.mean() * 200
    ax.axhline(wide, color=ORANGE, ls="--", lw=1.4)
    for x, v in enumerate(vals):
        ax.annotate(
            f"{v:.3g}%", (x, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8
        )
    ax.set(
        xticks=range(3),
        xticklabels=["Ordinary 8", "Early", "Late"],
        yscale="log",
        ylim=(0.002, 10),
        xlim=(-0.45, 2.45),
        title={"sgd": "SGD", "momentum": "Momentum", "adam": "Adam"}[opt],
    )
    ax.tick_params(axis="x", labelsize=8)
axes[0].set_ylabel("Test MSE / target variance (%)")
fig.savefig(OUT / "readout_timing.pdf", bbox_inches="tight")
plt.close(fig)
