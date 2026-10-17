#!/usr/bin/env python3
"""Plot net batch-gradient agreement from saved pre-update gradients; no training."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from neural_competition.analysis import load_checkpoint
from neural_competition.distributions import LABELS, read_study, seed_mean_interval


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--collection", default="pier_distributions")
    parser.add_argument("--optimizer", default="sgd_1")
    parser.add_argument("--widths", nargs="+", type=int, default=[8, 16])
    args = parser.parse_args()
    output = args.root / "runs/pier_study/analysis"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    metrics, _ = read_study(args.root, names=(args.collection,))
    chosen = metrics[(metrics.optimizer == args.optimizer) & metrics.width.isin(args.widths)]
    runs = chosen.groupby(["width", "seed"])
    if chosen.empty or set(chosen.width) != set(args.widths):
        raise ValueError("Requested completed width/optimizer cases are missing")
    # Use the same recorded steps for every seed and width, without interpolating
    # loss-triggered snapshots that occur at different times in different runs.
    common_steps = sorted(set.intersection(*(set(part.step) for _, part in runs)))
    chosen = chosen[chosen.step.isin(common_steps)]
    rows = []
    for _, row in chosen.iterrows():
        snapshot = load_checkpoint(row.run_path, int(row.step))
        gradients = snapshot["probe_gradients"].double().flatten(1)
        gradient = snapshot["mean_gradient"].double().flatten()
        signal = gradient.square().sum().item()
        energy = gradients.square().sum(1).mean().item()
        rows.append(
            dict(
                width=int(row.width),
                seed=int(row.seed),
                step=int(row.step),
                agreement=signal / energy if energy > 0 else float("nan"),
                mean_gradient_squared_norm=signal,
                batch_gradient_energy=energy,
                train_mse=2 * snapshot["train_loss_before"],
                test_mse=2 * snapshot["test_loss_before"],
            )
        )
    frame = pd.DataFrame(rows).sort_values(["width", "seed", "step"])
    estimates = []
    for (width, step), part in frame.groupby(["width", "step"]):
        mean, low, high = seed_mean_interval(part.agreement)
        estimates.append(
            dict(
                width=width,
                step=step,
                n_seeds=len(part),
                agreement_mean=mean,
                agreement_low=low,
                agreement_high=high,
                train_mse_median=part.train_mse.median(),
                test_mse_median=part.test_mse.median(),
            )
        )
    summary = pd.DataFrame(estimates)
    name = f"agreement_{args.optimizer}_" + "_".join(f"w{w}" for w in args.widths)
    frame.to_csv(output / f"{name}_seeds.csv", index=False)
    summary.to_csv(output / f"{name}_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for index, width in enumerate(args.widths):
        color = plt.get_cmap("tab10")(index % 10)
        part = summary[summary.width == width].sort_values("step")
        ax.plot(
            part.step,
            part.agreement_mean,
            color=color,
            lw=2.2,
            marker="o",
            ms=4,
            label=f"width {width}",
        )
        ax.fill_between(part.step, part.agreement_low, part.agreement_high, color=color, alpha=0.17)
    ax.set_xscale("symlog", linthresh=20, linscale=0.8)
    ax.set_xticks([0, 20, 100, 1000, 10000, 100000], ["0", "20", "100", "1k", "10k", "100k"])
    ax.set(
        xlabel="completed training updates (log scale after 20)",
        ylabel="A: net gradient agreement",
        ylim=(0, max(1.02, summary.agreement_high.max() * 1.03)),
        title=r"$A_t=\|\nabla L(\theta_t)\|^2\,/\,\mathbb{E}_B\|g_B(\theta_t)\|^2$",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    counts = frame.groupby("width").seed.nunique().tolist()
    config = json.loads((args.root / "runs" / args.collection / "config.json").read_text())
    seed_label = (
        f"{counts[0]} seeds per width" if len(set(counts)) == 1 else f"seed counts: {counts}"
    )
    optimizer_label = LABELS.get(args.optimizer, args.optimizer)
    fig.suptitle(
        f"Teacher: {config['teacher_width']} neurons · {optimizer_label} · {seed_label}",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.015,
        "Lines: seed means. Bands: pointwise 95% bootstrap intervals across seeds. "
        f"{config['probe_batches']} probe batches per state.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))
    print(f"Saved: {output / name}")


if __name__ == "__main__":
    main()
