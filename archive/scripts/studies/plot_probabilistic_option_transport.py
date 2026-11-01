"""Plot analytic pressure-indexed predictions, without empirical fitting."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from probabilistic_option_transport import blind_probability, transition_probabilities


def main():
    exposure = np.linspace(0, 3, 241)
    colors = ["#2466A6", "#C85B36"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), layout="constrained")
    for theta, label, color in zip([np.pi / 3, np.pi / 12],
                                   [r"$\theta=60^\circ$", r"$\theta=15^\circ$"], colors):
        axes[0].plot(exposure, [blind_probability(theta, s) for s in exposure],
                     label=label, color=color, linewidth=2)
    axes[0].set(title="Geometry changes the direction of transport",
                ylabel="Single-block blindness probability", ylim=(0, .5))
    for width, color in zip([4, 8], colors):
        values = [transition_probabilities(np.pi / 3, s, width) for s in exposure]
        axes[1].plot(exposure, [v["closed_given_initially_open"] for v in values],
                     color=color, linewidth=2, label=f"Width {width}")
        axes[2].plot(exposure, [v["open_given_initially_closed"] for v in values],
                     color=color, linewidth=2, label=f"Width {width}")
    axes[1].set(title="Width reduces loss after initial coverage",
                ylabel="P(closed now | open initially)", ylim=(0, .022))
    axes[2].set(title="Width improves endpoint reopening",
                ylabel="P(open now | closed initially)", ylim=(0, 1))
    for ax in axes:
        ax.set_xlabel("Accumulated common pressure s")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
    fig.suptitle("Derived option transport laws\n"
                 "Exact at fixed pressure exposure; not fixed-time training failure rates",
                 fontsize=13)
    dest = Path(__file__).resolve().parents[2] / "docs/evidence/probabilistic_competition"
    fig.savefig(dest / "pressure_laws.png", dpi=170)
    fig.savefig(dest / "pressure_laws.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
