#!/usr/bin/env python3
"""Measure P/N and exact neuron prediction changes using saved SGD states."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neural_competition.analysis import load_checkpoint
from neural_competition.distributions import (
    matched_states,
    paired_width_effect,
    read_study,
    seed_mean_interval,
)
from neural_competition.neurons import (
    gradient_masses,
    movement_statistics,
    prediction_moves,
    source_improvement,
)

COLORS = {8: "#2563eb", 16: "#df6b20"}
PN_METRICS = ["P", "N", "global_P", "global_N", "offset_negative", "p_negative"]
MOVE_METRICS = [
    "R",
    "R_perpendicular",
    "cancellation_fraction",
    "solo_harm_offset_fraction",
    "solo_harm_per_source_gain",
    "joint_harm_per_source_gain",
    "joint_benefit_per_source_gain",
    "joint_harm_fraction",
    "neuron_harm_fraction",
    "source_gain",
    "loss_change",
    "linear_progress",
    "joint_energy",
    "individual_energy",
    "perpendicular_energy",
]


def selection(root):
    metrics, _ = read_study(
        root, names=("pier_distributions", "pier_optimizer_pilot", "pier_early", "pier_early_extra")
    )
    metrics = metrics[metrics.width.isin([8, 16])].copy()
    views = []
    for collection, optimizer in [
        ("pier_distributions", "sgd_1"),
        ("pier_optimizer_pilot", "sgd_01"),
    ]:
        part = metrics[(metrics.collection == collection) & (metrics.optimizer == optimizer)]
        common_steps = sorted(
            set.intersection(*(set(p.step) for _, p in part.groupby(["width", "seed"])))
        )
        for step in common_steps:
            chosen = part[part.step == step].copy()
            chosen["comparison"] = f"step_{step}"
            chosen["comparison_kind"] = "equal_steps"
            views.append(chosen)
        chosen = matched_states(part, target=0.01, tolerance=0.25)
        chosen = chosen[chosen.loss_eligible].copy()
        paired = chosen.groupby("seed").width.nunique()
        chosen = chosen[chosen.seed.isin(paired[paired == 2].index)].copy()
        chosen["comparison"] = "mse_0.01"
        chosen["comparison_kind"] = "matched_loss"
        views.append(chosen)
    early = metrics[
        metrics.collection.isin(["pier_early", "pier_early_extra"]) & (metrics.optimizer == "sgd_1")
    ]
    chosen = matched_states(early, target=0.2, tolerance=0.1)
    chosen = chosen[chosen.loss_eligible].copy()
    paired = chosen.groupby("seed").width.nunique()
    chosen = chosen[chosen.seed.isin(paired[paired == 2].index)].copy()
    chosen["comparison"] = "mse_0.2"
    chosen["comparison_kind"] = "matched_loss"
    views.append(chosen)
    result = pd.concat(views, ignore_index=True)
    result["source_checkpoint"] = [
        str(Path(r.run_path) / "checkpoints" / f"step_{int(r.step):06d}.pt")
        for r in result.itertuples()
    ]
    return result


def estimate(frame, groups, metrics):
    rows = []
    for identity, part in frame.groupby(groups):
        record = dict(zip(groups, identity))
        record["n_seeds"] = part.seed.nunique()
        for metric in metrics:
            mean, low, high = seed_mean_interval(part[metric])
            record.update({metric: mean, metric + "_low": low, metric + "_high": high})
        record["train_mse_mean"] = part.train_mse.mean()
        if "n_examples" in part:
            record["n_examples_min"] = part.n_examples.min()
            record["n_examples_mean"] = part.n_examples.mean()
        rows.append(record)
    return pd.DataFrame(rows)


def effects(frame, groups, metrics):
    rows = []
    for identity, part in frame.groupby(groups):
        for metric in metrics:
            result = paired_width_effect(part, metric)
            result.update(dict(zip(groups, identity)))
            rows.append(result)
    return pd.DataFrame(rows)


def ratio(a, b):
    return a / b if b > 0 else float("nan")


def save_figure(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pn(summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4))
    early = summary[
        (summary.optimizer == "sgd_1") & (summary.comparison == "mse_0.2") & (summary.probes == 512)
    ]
    for offset, width in zip([-0.17, 0.17], [8, 16]):
        row = early[early.width == width].iloc[0]
        values = np.array([row.P, row.N])
        error = np.array(
            [
                [row.P - row.P_low, row.N - row.N_low],
                [row.P_high - row.P, row.N_high - row.N],
            ]
        )
        axes[0].bar(
            np.arange(2) + offset,
            values,
            width=0.3,
            color=COLORS[width],
            yerr=error,
            capsize=3,
            label=f"width {width}",
        )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["P: positive", "N: negative"],
        ylabel="neuron mass / gradient energy",
        title="Early: matched MSE ≈ 0.2\n40 paired seeds, 512 probes",
    )
    axes[0].legend(frameon=False, fontsize=9)
    for ax, metric in zip(axes[1:], ["P", "N"]):
        for width in [8, 16]:
            part = summary[
                (summary.optimizer == "sgd_1")
                & (summary.width == width)
                & (summary.comparison_kind == "equal_steps")
                & (summary.probes == 128)
            ]
            part = part.copy()
            part["step"] = part.comparison.str.removeprefix("step_").astype(int)
            part = part.sort_values("step")
            ax.plot(
                part.step,
                part[metric],
                color=COLORS[width],
                marker="o",
                ms=3,
                label=f"width {width}",
            )
            ax.fill_between(
                part.step,
                part[metric + "_low"],
                part[metric + "_high"],
                color=COLORS[width],
                alpha=0.15,
            )
        ax.set(
            xscale="symlog",
            xlabel="completed updates",
            ylabel=metric,
            title=f"{metric} across training · SGD lr 1",
        )
        ax.set_xscale("symlog", linthresh=20)
        ax.set_xticks([0, 100, 1000, 10000, 100000], ["0", "100", "1k", "10k", "100k"])
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Neuron-level conflicting demands: teacher 8; both layers train", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.text(
        0.5,
        0.01,
        "Intervals: 95% bootstrap across seeds. Equal-step curves compare different losses; late ratios approach numerical floors.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output, "neuron_PN")


def plot_cancellation(summary, output, source_batches=32):
    chosen = summary[
        (summary.scale == 1)
        & summary.comparison.isin(["mse_0.2", "mse_0.01"])
        & summary.optimizer.isin(["sgd_1", "sgd_01"])
    ]
    labels = [
        ("sgd_1", "mse_0.2", "LR 1 · MSE ≈ .2"),
        ("sgd_1", "mse_0.01", "LR 1 · MSE ≈ .01"),
        ("sgd_01", "mse_0.01", "LR .1 · MSE ≈ .01"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    settings = [
        (
            "all",
            "R_perpendicular",
            "Interaction of side effects\nR perpendicular (<1: net cancellation)",
        ),
        (
            "well_fit_0.05",
            "cancellation_fraction",
            "Cancellation on accurate examples\n1 − |total move| / sum |individual moves|",
        ),
        (
            "all",
            "joint_harm_per_source_gain",
            "Actual collateral damage\nheld-out harm / source-batch improvement",
        ),
    ]
    for ax, (subset, metric, title) in zip(axes, settings):
        for offset, width in zip([-0.12, 0.12], [8, 16]):
            part = chosen[(chosen.width == width) & (chosen.subset == subset)]
            for index, (optimizer, comparison, _) in enumerate(labels):
                rows = part[(part.optimizer == optimizer) & (part.comparison == comparison)]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                ax.errorbar(
                    index + offset,
                    row[metric],
                    yerr=[
                        [row[metric] - row[metric + "_low"]],
                        [row[metric + "_high"] - row[metric]],
                    ],
                    color=COLORS[width],
                    fmt="o",
                    capsize=4,
                    label=f"width {width}" if index == 0 else None,
                )
        ax.set(title=title, xticks=range(3), xticklabels=[r[2] for r in labels])
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.18)
    axes[0].axhline(1, ls=":", color="gray", lw=1)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    fig.suptitle(
        f"Exact neuron prediction changes: {source_batches} candidate SGD batches per saved state",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.text(
        0.5,
        0.012,
        "40 / 8 / 5 paired seeds, respectively; 95% seed-bootstrap intervals. Accurate examples satisfy |residual| ≤ .05 at BOTH widths.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output, "neuron_prediction_interactions")


def plot_step_sensitivity(summary, output):
    early = summary[(summary.optimizer == "sgd_1") & (summary.comparison == "mse_0.2")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, subset, title in zip(
        axes,
        ["all", "well_fit_0.05"],
        ["All test examples", "Accurate at both widths: |residual| ≤ .05"],
    ):
        for width, offset in [(8, -0.1), (16, 0.1)]:
            part = early[(early.width == width) & (early.subset == subset)].sort_values("scale")
            x = np.arange(len(part)) + offset
            values = part.joint_harm_per_source_gain.to_numpy()
            ax.errorbar(
                x,
                values,
                yerr=np.array(
                    [
                        values - part.joint_harm_per_source_gain_low.to_numpy(),
                        part.joint_harm_per_source_gain_high.to_numpy() - values,
                    ]
                ),
                color=COLORS[width],
                marker="o",
                linestyle="none",
                capsize=4,
                label=f"width {width}",
            )
        ax.set(
            title=title,
            xticks=[0, 1],
            xticklabels=["1% of saved SGD direction", "Full SGD step"],
            ylabel="held-out harm / source-batch improvement",
            ylim=(0, None),
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False)
    fig.suptitle("The collateral-harm difference depends on step length", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.text(
        0.5,
        0.01,
        "SGD lr 1 states near MSE .2; 40 paired seeds; pointwise 95% seed-bootstrap intervals. No retraining.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output, "neuron_step_sensitivity")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--source-batches", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.source_batches <= 128:
        parser.error("--source-batches must be between 1 and 128")
    root = args.root
    output = root / "runs/pier_study/neurons"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    selected = selection(root)
    selected.to_csv(output / "selected_states.csv", index=False)
    data = torch.load(root / "runs/pier_distributions/data.pt", weights_only=True)
    # Every collection must really share targets and the fixed probe batches.
    for collection in selected.collection.unique():
        other = torch.load(root / "runs" / collection / "data.pt", weights_only=True)
        for key in ["x_train", "y_train", "x_test", "y_test", "probe_indices"]:
            torch.testing.assert_close(data[key], other[key], atol=0, rtol=0)
    source_indices = data["probe_indices"][: args.source_batches]
    torch.save(source_indices, output / "source_indices.pt")
    x_source, y_source = data["x_train"][source_indices], data["y_train"][source_indices]
    x_test, y_test = data["x_test"].double(), data["y_test"].double()
    snapshots = {}
    masses, neuron_rows = [], []
    for index, row in enumerate(selected.itertuples()):
        path = row.source_checkpoint
        if path not in snapshots:
            snapshots[path] = load_checkpoint(row.run_path, int(row.step))
        snapshot = snapshots[path]
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
        for probes in [128, 512] if row.comparison == "mse_0.2" else [128]:
            if probes == 512:
                fresh_path = (
                    root
                    / "runs/pier_study/tail_checks"
                    / (f"w{row.width:04d}_seed{row.seed}_{row.optimizer}.pt")
                )
                fresh = torch.load(fresh_path, weights_only=True)
                torch.testing.assert_close(
                    fresh["parameters_before"], snapshot["parameters_before"]
                )
                gradients = fresh["probe_gradients"]
            else:
                gradients = snapshot["probe_gradients"]
            record, units = gradient_masses(gradients)
            masses.append({**identity, "probes": probes, **record})
            neuron_rows.extend({**identity, "probes": probes, **unit} for unit in units)
        if index % 50 == 0:
            print(f"P/N: {index + 1}/{len(selected)} states", flush=True)
    mass = pd.DataFrame(masses)
    mass.to_csv(output / "gradient_masses_seeds.csv", index=False)
    pd.DataFrame(neuron_rows).to_csv(output / "gradient_masses_neurons.csv", index=False)
    mass_groups = ["optimizer", "comparison", "comparison_kind", "probes"]
    mass_summary = estimate(mass, [*mass_groups, "width"], PN_METRICS)
    mass_summary.to_csv(output / "gradient_masses_summary.csv", index=False)
    effects(mass, mass_groups, PN_METRICS).to_csv(
        output / "gradient_masses_effects.csv", index=False
    )
    plot_pn(mass_summary, output)

    # Do not interpret numerical-floor states as evidence of a mechanism.
    functional = selected[(selected.step <= 19999) | (selected.comparison_kind == "matched_loss")]
    records, per_batch, gram_rows = [], [], []
    seen = 0
    for identity, pair in functional.groupby(["optimizer", "comparison", "seed"]):
        optimizer, comparison, seed = identity
        states = {
            int(row.width): (row, snapshots[row.source_checkpoint]) for row in pair.itertuples()
        }
        if set(states) != {8, 16}:
            raise ValueError(f"Unpaired functional comparison {identity}")
        residuals = {}
        for width, (_, snapshot) in states.items():
            prediction, _, _ = prediction_moves(
                snapshot["parameters_before"],
                torch.zeros_like(snapshot["actual_step"])[None],
                x_test,
            )
            residuals[width] = prediction - y_test
        masks = {"all": torch.ones(len(y_test), dtype=torch.bool)}
        for threshold in [0.05, 0.1]:
            masks[f"well_fit_{threshold:g}"] = (residuals[8].abs() <= threshold) & (
                residuals[16].abs() <= threshold
            )
        for width, (row, snapshot) in states.items():
            lr = snapshot["optimizer"]["lr"]
            base = {
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
            for scale in [0.01, 1.0]:
                updates = -lr * scale * snapshot["probe_gradients"][: args.source_batches].double()
                prediction, moves, crossings = prediction_moves(
                    snapshot["parameters_before"], updates, x_test
                )
                gain = source_improvement(
                    snapshot["parameters_before"], updates, x_source, y_source
                )
                for subset, mask in masks.items():
                    if not mask.any():
                        continue
                    stats, gram = movement_statistics(prediction - y_test, moves, mask)
                    record = {key: values.nanmean().item() for key, values in stats.items()}
                    record["R"] = ratio(record["joint_energy"], record["individual_energy"])
                    record["R_perpendicular"] = ratio(
                        record["perpendicular_energy"], record["perpendicular_individual_energy"]
                    )
                    record["cancellation_fraction"] = 1 - ratio(
                        record["absolute_joint_movement"], record["absolute_individual_movement"]
                    )
                    record["solo_harm_offset_fraction"] = ratio(
                        record["offset_solo_harm"], record["solo_harm"]
                    )
                    record["source_gain"] = gain.mean().item()
                    record["source_harm_fraction"] = (gain < 0).double().mean().item()
                    record["gate_crossing_fraction"] = crossings[:, mask].double().mean().item()
                    for metric in ["solo_harm", "joint_harm", "joint_benefit"]:
                        record[metric + "_per_source_gain"] = ratio(
                            record[metric], record["source_gain"]
                        )
                    identity_row = {
                        **base,
                        "scale": scale,
                        "subset": subset,
                        "n_examples": mask.sum().item(),
                    }
                    records.append({**identity_row, **record})
                    for b in range(args.source_batches):
                        per_batch.append(
                            {
                                **identity_row,
                                "source_batch": b,
                                "source_gain": gain[b].item(),
                                **{k: v[b].item() for k, v in stats.items()},
                            }
                        )
                    if subset == "all" and scale == 1:
                        matrix = gram.mean(0)
                        gram_rows.extend(
                            {**base, "r": r, "s": s, "inner_product": matrix[r, s].item()}
                            for r in range(width)
                            for s in range(width)
                        )
                # A deterministic illustration, selected by seed before looking at effects.
                if optimizer == "sgd_1" and comparison == "mse_0.2" and seed == 0 and scale == 1:
                    np.savez_compressed(
                        output / f"example_w{width}_seed0.npz",
                        residual=(prediction - y_test).numpy(),
                        moves=moves.numpy(),
                        source_gain=gain.numpy(),
                        well_fit=masks["well_fit_0.05"].numpy(),
                    )
            seen += 1
            if seen % 10 == 0:
                print(f"Prediction changes: {seen}/{len(functional)} states", flush=True)
    moves = pd.DataFrame(records)
    moves.to_csv(output / "prediction_changes_seeds.csv", index=False)
    pd.DataFrame(per_batch).to_csv(output / "prediction_changes_batches.csv", index=False)
    pd.DataFrame(gram_rows).to_csv(output / "prediction_gram_neurons.csv", index=False)
    move_groups = ["optimizer", "comparison", "comparison_kind", "scale", "subset"]
    move_summary = estimate(moves, [*move_groups, "width"], MOVE_METRICS)
    move_summary.to_csv(output / "prediction_changes_summary.csv", index=False)
    effects(moves, move_groups, MOVE_METRICS).to_csv(
        output / "prediction_changes_effects.csv", index=False
    )
    plot_cancellation(move_summary, output, args.source_batches)
    plot_step_sensitivity(move_summary, output)
    manifest = {
        "teacher_width": 8,
        "widths": [8, 16],
        "source_batches": args.source_batches,
        "test_examples": len(y_test),
        "scales": [0.01, 1.0],
        "well_fit_thresholds": [0.05, 0.1],
        "source_steps": "float64 theta - scale * lr * saved float32 probe_gradient; no retraining",
        "uncertainty": "paired training-seed bootstrap, 4000 replicates, seed 741",
        "source_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in snapshots
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Done: {len(snapshots)} unique checkpoints; {len(mass)} P/N rows; {len(moves)} movement rows."
    )
    print(output)


if __name__ == "__main__":
    main()
