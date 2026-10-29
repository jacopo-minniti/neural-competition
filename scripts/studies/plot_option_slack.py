"""Summarize forecast improvement and the tests against simple commitment."""

import json

import matplotlib
import numpy as np
from explore_conditional_editability import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    theory = ROOT / "docs/theory"
    retention = json.loads((theory / "option_retention/results.json").read_text())["results"]
    scores = json.loads((theory / "tangent_sgd_forecast/analysis.json").read_text())[
        "fresh_test_seeds"
    ]["scores"]
    transport = json.loads((theory / "option_transport/results.json").read_text())["results"]
    shortcut = json.loads((theory / "shortcut_commitment/results.json").read_text())["results"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for width, color in [(32, "#2969a2"), (128, "#be6925")]:
        for arm, style in [("ordinary", "-"), ("frozen_tangent", "--")]:
            steps = sorted({r["step"] for r in retention})
            values = [
                np.mean(
                    [
                        r["train_loss"]
                        for r in retention
                        if r["width"] == width and r["arm"] == arm and r["step"] == s
                    ]
                )
                for s in steps
            ]
            axes[0, 0].plot(
                steps,
                values,
                style,
                color=color,
                label=f"{width}: " + ("ordinary" if arm == "ordinary" else "retained tangent"),
            )
    axes[0, 0].set(
        title="Preserving all features does not rescue narrow SGD",
        xlabel="SGD step",
        ylabel="Training cross-entropy",
    )
    axes[0, 0].legend(fontsize=8)
    keys = ["loss_gradient", "independent_tangent", "conditional_tangent"]
    axes[0, 1].bar(
        range(3), [scores[k]["rmse"] for k in keys], color=["gray", "#2969a2", "#be6925"]
    )
    axes[0, 1].set(
        xticks=range(3),
        xticklabels=["Loss/gradient", "Independent\nbatches", "Known batch\nschedule"],
        ylabel="Late gain-difference RMSE",
        title="Fresh-seed forecasting: distinguish information used",
    )
    for width, color in [(32, "#2969a2"), (128, "#be6925")]:
        group = [r for r in transport if r["width"] == width and r["step"] == 0]
        values = [
            100 * np.mean([r["audits"][i]["gate_crossing_fraction"] for r in group]) for i in (0, 1)
        ]
        axes[1, 0].plot([0.005, 0.05], values, "-o", color=color, label=f"Width {width}")
        values = [
            np.mean(
                [
                    r["validation"]["exception_loss"]
                    for r in shortcut
                    if r["width"] == width and r["arm"] == arm and r["step"] == 2000
                ]
            )
            for arm in ["ordinary", "early_mask", "late_mask"]
        ]
        axes[1, 1].plot(range(3), values, "-o", color=color, label=f"Width {width}")
    axes[1, 0].set(
        title="Early gate flux is lower at greater width",
        xlabel="Full-gradient step size",
        ylabel="Probe gate crossings (%)",
    )
    axes[1, 0].legend()
    axes[1, 1].set(
        title="Cue removal helps both early and late",
        xticks=range(3),
        xticklabels=["Ordinary", "Early masking", "Late masking"],
        ylabel="Held-out exception cross-entropy",
    )
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(theory / "option_retention/summary.png", dpi=180)


if __name__ == "__main__":
    main()
