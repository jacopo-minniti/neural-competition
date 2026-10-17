#!/usr/bin/env python3
"""Enumerate same-batch neuron-pair interference at saved SGD/momentum states."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from analyze_neurons import effects, save_figure, selection
from compare_momentum import LABELS, PROFILES, statistics

from neural_competition.analysis import load_checkpoint
from neural_competition.neurons import candidate_descent
from neural_competition.pairs import (
    pair_loss_effects,
    pair_statistics,
    source_neuron_changes,
    summarize_batches,
)

METRICS = [
    "frequency",
    "severity",
    "event_severity",
    "eligible_fraction",
    "helpful_neuron_fraction",
    "positive_interaction_fraction",
    "unordered_frequency",
    "events_per_batch",
    "harm_per_batch",
    "event_benefit_share",
    "harm_over_full_gain",
]
GROUPS = ["optimizer", "comparison", "comparison_kind", "probe_set", "scale"]
COLORS = {8: "#2563eb", 16: "#df6b20"}


def make_plots(summary, output):
    primary = summary[(summary.probe_set == "original128") & (summary.scale == 1)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    profiles = ["sgd_1", "momentum_01"]
    for row_index, comparison in enumerate(["mse_0.2", "mse_0.01"]):
        for ax, metric, title in zip(
            axes[row_index],
            ["frequency", "event_severity", "severity"],
            [
                "Frequency among helpful pairs",
                "Severity on interfering cases",
                "Aggregate harm / eligible benefit",
            ],
        ):
            for width, offset in [(8, -0.12), (16, 0.12)]:
                for j, optimizer in enumerate(profiles):
                    rows = primary[
                        (primary.comparison == comparison)
                        & (primary.width == width)
                        & (primary.optimizer == optimizer)
                    ]
                    if rows.empty:
                        continue
                    p = rows.iloc[0]
                    if not np.isfinite(p[metric]):
                        continue
                    ax.errorbar(
                        j + offset,
                        100 * p[metric],
                        yerr=[
                            [100 * (p[metric] - p[metric + "_low"])],
                            [100 * (p[metric + "_high"] - p[metric])],
                        ],
                        color=COLORS[width],
                        fmt="o",
                        capsize=4,
                        label=f"width {width}" if j == 0 else None,
                    )
            ax.set(
                xticks=range(2),
                xticklabels=[LABELS[p] for p in profiles],
                title=f"MSE ≈ {comparison[4:]}\n{title}",
                ylabel="percent",
                ylim=(0, None),
            )
            ax.tick_params(axis="x", labelsize=8)
            ax.title.set_fontsize(10)
            ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Same batch, different neurons: helpful alone, harmful after another neuron", fontsize=12
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.text(
        0.5,
        0.013,
        "All pairs, both orders; 128 batches/state. Early: 40 paired seeds; later: SGD 8, momentum 7. Conditional severity uses event-positive seeds. 95% intervals.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output, "pair_interference")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, metric, title in zip(
        axes,
        ["frequency", "severity"],
        ["Strong interference frequency", "Harm / eligible standalone benefit"],
    ):
        for j, (optimizer, scale) in enumerate([(o, s) for o in profiles for s in [0.01, 1.0]]):
            for width, offset in [(8, -0.1), (16, 0.1)]:
                p = summary[
                    (summary.optimizer == optimizer)
                    & (summary.scale == scale)
                    & (summary.width == width)
                    & (summary.comparison == "mse_0.2")
                    & (summary.probe_set == "original128")
                ].iloc[0]
                ax.errorbar(
                    j + offset,
                    100 * p[metric],
                    yerr=[
                        [100 * (p[metric] - p[metric + "_low"])],
                        [100 * (p[metric + "_high"] - p[metric])],
                    ],
                    color=COLORS[width],
                    fmt="o",
                    capsize=4,
                    label=f"width {width}" if j == 0 else None,
                )
        ax.set(
            title=title,
            ylabel="percent",
            xticks=range(4),
            xticklabels=["SGD · 1%", "SGD · full", "Momentum · 1%", "Momentum · full"],
            ylim=(0, None),
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False)
    fig.suptitle("Early MSE ≈ .2: finite-step sensitivity of pair interference", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.text(
        0.5,
        0.01,
        "40 paired seeds. Step shortening changes candidate updates at fixed states; all neuron updates are computed before combining them.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output, "pair_step_sensitivity")


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "runs/pier_study/pairs"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    selected = selection(root, PROFILES)
    selected = selected[
        (selected.comparison_kind == "matched_loss")
        | ((selected.optimizer != "sgd_01") & selected.step.isin([100, 700, 1999, 19999]))
    ]
    selected.to_csv(output / "selected_states.csv", index=False)
    data = torch.load(root / "runs/pier_distributions/data.pt", weights_only=True)
    original = data["probe_indices"]
    fresh_root = root / "runs/pier_study/momentum"
    fresh_indices = torch.load(fresh_root / "fresh_source_indices.pt", weights_only=True)
    fresh_gradients = torch.load(fresh_root / "fresh_source_gradients.pt", weights_only=True)
    torch.save({"original128": original, "fresh32": fresh_indices}, output / "source_indices.pt")
    records, batches, pairs, sensitivity = [], [], [], []
    sources = {}
    for index, row in enumerate(selected.itertuples()):
        snapshot = load_checkpoint(row.run_path, row.step)
        sources[row.source_checkpoint] = hashlib.sha256(
            Path(row.source_checkpoint).read_bytes()
        ).hexdigest()
        identity = {
            key: getattr(row, key)
            for key in [
                "optimizer",
                "comparison",
                "comparison_kind",
                "width",
                "seed",
                "step",
                "train_mse",
            ]
        }
        probe_sets = ["original128", "fresh32"] if row.comparison == "mse_0.2" else ["original128"]
        for probe_set in probe_sets:
            if probe_set == "original128":
                indices, gradients = original, snapshot["probe_gradients"]
            else:
                indices = fresh_indices
                gradients = fresh_gradients[f"{row.optimizer}_w{row.width}_seed{row.seed}"]
            x, y = data["x_train"][indices], data["y_train"][indices].double()
            descent = candidate_descent(snapshot, gradients)
            for scale in [0.01, 1.0]:
                updates = -scale * snapshot["optimizer"]["lr"] * descent
                prediction, changes = source_neuron_changes(
                    snapshot["parameters_before"], updates, x
                )
                residual = prediction - y
                baseline = 0.5 * residual.square().mean(1)
                alone, interaction = pair_loss_effects(residual, changes)
                total = changes.sum(-1)
                full_gain = -(residual * total + 0.5 * total.square()).mean(1)
                per_batch, per_pair = pair_statistics(alone, interaction, baseline)
                record = summarize_batches(per_batch)
                gain = full_gain.mean().item()
                record["harm_over_full_gain"] = (
                    record["harm_per_batch"] / gain if gain > 0 else float("nan")
                )
                base = {
                    **identity,
                    "probe_set": probe_set,
                    "scale": scale,
                    "source_batches": len(indices),
                }
                records.append(
                    {
                        **base,
                        **record,
                        "full_source_gain": full_gain.mean().item(),
                        "baseline_loss": baseline.mean().item(),
                    }
                )
                for b in range(len(indices)):
                    batches.append(
                        {
                            **base,
                            "source_batch": b,
                            "full_source_gain": full_gain[b].item(),
                            "baseline_loss": baseline[b].item(),
                            **{k: v[b].item() for k, v in per_batch.items()},
                        }
                    )
                for r in range(row.width):
                    for s in range(row.width):
                        if r != s:
                            pairs.append(
                                {
                                    **base,
                                    "r": r,
                                    "s": s,
                                    **{k: v[r, s].item() for k, v in per_pair.items()},
                                }
                            )
                if probe_set == "original128":
                    for tolerance in [0.0, 1e-8]:
                        alternate, _ = pair_statistics(
                            alone, interaction, baseline, relative_tolerance=tolerance
                        )
                        sensitivity.append(
                            {
                                **base,
                                "relative_tolerance": tolerance,
                                **summarize_batches(alternate),
                            }
                        )
                upper = torch.triu_indices(row.width, row.width, offset=1)
                name = f"{row.collection}_w{row.width}_seed{row.seed}_{row.optimizer}_step{row.step}_{probe_set}_scale{scale:g}.npz"
                np.savez_compressed(
                    raw / name,
                    standalone=alone.numpy(),
                    interaction=interaction[:, upper[0], upper[1]].numpy(),
                    r=upper[0].numpy(),
                    s=upper[1].numpy(),
                    baseline_loss=baseline.numpy(),
                )
        if index % 20 == 0:
            print(f"All-pair experiments: {index + 1}/{len(selected)} states", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "pair_seeds.csv", index=False)
    pd.DataFrame(batches).to_csv(output / "pair_batches.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "pair_neurons.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / "tolerance_sensitivity.csv", index=False)
    summary = statistics(frame, GROUPS, METRICS)
    summary.to_csv(output / "pair_summary.csv", index=False)
    effects(frame, GROUPS, METRICS).to_csv(output / "pair_effects.csv", index=False)
    make_plots(summary, output)
    source_inputs = [
        root / "runs/pier_distributions/data.pt",
        fresh_root / "fresh_source_indices.pt",
        fresh_root / "fresh_source_gradients.pt",
    ]
    for path in source_inputs:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "teacher_width": 8,
        "widths": [8, 16],
        "source_batches": {"original128": 128, "fresh32": 32},
        "scales": [0.01, 1.0],
        "relative_tolerance": 1e-10,
        "source_sha256": sources,
        "event": "A_r<0, A_s<0, A_s+I_rs>0; r->s ordered; tolerance relative to source batch loss",
        "severity": "sum eligible positive marginal harm / sum eligible standalone second-neuron improvement",
        "uncertainty": "4000 paired training-seed bootstrap samples; seed 741; fixed teacher/data",
        "no_retraining": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(frame)} state/scale/probe summaries under {output}", flush=True)


if __name__ == "__main__":
    main()
