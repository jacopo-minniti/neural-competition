"""Plot descriptive state discrimination and phase-wise response retention."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    root = Path("docs/evidence/competition_reserve")
    summary = json.loads((root / "summary.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), layout="constrained")
    labels = ["Current\nresponse", "Implementation\nparticipation", "Protected\nreserve", "Completion\nreserve"]
    names = ["current", "participation", "protected", "completion"]
    for offset, cohort, label, color in [(-.18, "confirmation", "24 factors", "#247c96"),
                                        (.18, "transfer32", "32 factors · fresh seeds", "#d17b35")]:
        data = summary[cohort]
        axes[0].bar(np.arange(4) + offset, [data["overall"]["auc"][k] for k in names],
                    width=.36, label=label, color=color)
        axes[1].plot([4, 8, 16], [100*data["by_width"][str(w)]["loss_fraction"] for w in [4, 8, 16]],
                     "o-", color=color, label=label, linewidth=2)
        axes[1].plot([4, 8, 16], [100*data["by_width"][str(w)]["predicted_loss_fraction"] for w in [4, 8, 16]],
                     "--", color=color, alpha=.7)
    axes[0].set(xticks=np.arange(4), xticklabels=labels, ylim=(.5, 1),
                ylabel="AUC for later loss of half current response",
                title="Reserve distinguishes vulnerability")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    axes[1].set(xticks=[4, 8, 16], xlabel="Width (all exceed exact capacity of 3)",
                ylabel="Contrasts losing half their current response (%)",
                title="Losses at matched common-learning progress", ylim=(0, 8))
    axes[1].text(.98, .94, "Solid: observed full training\nDashed: coherent-pressure approximation",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9)
    fig.suptitle("Current state and competition reserve", fontsize=16)
    fig.savefig(root / "summary.png", dpi=180)
    fig.savefig(root / "summary.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
