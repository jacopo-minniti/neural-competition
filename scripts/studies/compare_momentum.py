#!/usr/bin/env python3
"""Compare width/interference under SGD and momentum without retraining."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from analyze_neurons import MOVE_METRICS, effects, estimate, ratio, save_figure, selection

from neural_competition.analysis import load_checkpoint
from neural_competition.diagnostics import alignment_arrays, distribution_summary
from neural_competition.distributions import seed_mean_interval
from neural_competition.neurons import (
    candidate_descent,
    gradient_masses,
    movement_statistics,
    prediction_moves,
    source_improvement,
    step_transfer_masses,
)

PROFILES = [
    ("pier_distributions", "sgd_1"),
    ("pier_distributions", "momentum_01"),
    ("pier_optimizer_pilot", "sgd_01"),
]
LABELS = {"sgd_1": "SGD lr 1", "momentum_01": "Momentum lr .1, β=.9", "sgd_01": "SGD lr .1"}
COLORS = {8: "#2563eb", 16: "#df6b20"}
GRADIENT_METRICS = [
    "P",
    "N",
    "global_N",
    "p_negative",
    "A",
    "step_P",
    "step_N",
    "step_global_N",
    "step_p_negative",
    "skewness",
    "excess_kurtosis",
]
ACTUAL_METRICS = [
    "linear_harm_fraction",
    "finite_harm_fraction",
    "loss_change",
    "actual_P",
    "actual_N",
    "actual_global_N",
]
TRAJECTORY_METRICS = ["batch_sharpness_ratio", "step_previous_cosine", "step_norm", "test_mse"]
MOVEMENT_METRICS = [
    *MOVE_METRICS,
    "source_harm_fraction",
    "gain_nonpositive",
    "harm_over_train_loss",
    "net_transfer_per_source_gain",
]


def statistics(frame, groups, metrics):
    summary = estimate(frame, [*groups, "width"], metrics)
    counts = frame.groupby([*groups, "width"])[metrics].count().reset_index()
    counts = counts.rename(columns={k: k + "_n_valid" for k in metrics})
    return summary.merge(counts, on=[*groups, "width"])


def record_movement(stats, gain, train_mse):
    result = {key: values.nanmean().item() for key, values in stats.items()}
    for name, numerator, denominator in [
        ("R", "joint_energy", "individual_energy"),
        ("R_perpendicular", "perpendicular_energy", "perpendicular_individual_energy"),
        ("solo_harm_offset_fraction", "offset_solo_harm", "solo_harm"),
    ]:
        result[name] = ratio(result[numerator], result[denominator])
    result["cancellation_fraction"] = 1 - ratio(
        result["absolute_joint_movement"], result["absolute_individual_movement"]
    )
    result["source_gain"] = gain.mean().item()
    result["source_harm_fraction"] = (gain < 0).double().mean().item()
    result["gain_nonpositive"] = float(result["source_gain"] <= 0)
    result["harm_over_train_loss"] = result["joint_harm"] / (0.5 * train_mse)
    for metric in ["solo_harm", "joint_harm", "joint_benefit"]:
        result[metric + "_per_source_gain"] = ratio(result[metric], result["source_gain"])
    result["net_transfer_per_source_gain"] = ratio(-result["loss_change"], result["source_gain"])
    return result


def write_contrasts(gradients, actual, movements, output):
    """Momentum width effect minus SGD width effect, using common paired seeds."""
    rows = []
    for domain, frame, groups, metrics in [
        ("gradient", gradients, ["comparison", "probes"], GRADIENT_METRICS),
        ("actual_step", actual, ["comparison", "probes", "scale"], ACTUAL_METRICS),
        ("movement", movements, ["comparison", "direction", "scale", "subset"], MOVEMENT_METRICS),
    ]:
        frame = frame[
            (frame.comparison_kind == "matched_loss")
            & frame.optimizer.isin(["sgd_1", "momentum_01"])
        ]
        for identity, part in frame.groupby(groups):
            for metric in metrics:
                pivot = part.pivot(index="seed", columns=["optimizer", "width"], values=metric)
                required = [(o, w) for o in ["sgd_1", "momentum_01"] for w in [8, 16]]
                if not all(key in pivot for key in required):
                    continue
                pivot = pivot[required].dropna()
                contrast = (
                    pivot["momentum_01", 16]
                    - pivot["momentum_01", 8]
                    - pivot["sgd_1", 16]
                    + pivot["sgd_1", 8]
                )
                mean, low, high = seed_mean_interval(contrast)
                rows.append(
                    {
                        "domain": domain,
                        **dict(zip(groups, identity)),
                        "metric": metric,
                        "n": len(contrast),
                        "mean": mean,
                        "low": low,
                        "high": high,
                    }
                )
    pd.DataFrame(rows).to_csv(output / "optimizer_width_contrasts.csv", index=False)


def recorded_step(snapshot, gradients, x, y):
    """The saved actual step, evaluated on full minibatches in float64."""
    delta = snapshot["actual_step"].double()
    descent = -delta
    g = gradients.double()
    energy = g.square().sum((1, 2)).mean()
    denominator = (energy * descent.square().sum()).sqrt()
    scores = (g * descent).sum(-1) / denominator
    local_P = scores.clamp_min(0).sum(1).mean().item()
    local_N = (-scores).clamp_min(0).sum(1).mean().item()
    global_N = (-scores.sum(1)).clamp_min(0).mean().item()
    linear_harm = (scores.sum(1) < 0).double().mean().item()
    rows = []
    for scale in [0.01, 1.0]:
        before, moves, _ = prediction_moves(
            snapshot["parameters_before"], scale * delta[None], x.flatten(0, 1)
        )
        change = moves[0].sum(-1).reshape(y.shape)
        residual = before.reshape(y.shape) - y.double()
        losses = (residual * change + 0.5 * change.square()).mean(1)
        rows.append(
            {
                "scale": scale,
                "linear_harm_fraction": linear_harm,
                "finite_harm_fraction": (losses > 0).double().mean().item(),
                "loss_change": losses.mean().item(),
                "actual_P": local_P,
                "actual_N": local_N,
                "actual_global_N": global_N,
            }
        )
    return rows


def interval_points(ax, frame, metric, profiles):
    for width, offset in [(8, -0.13), (16, 0.13)]:
        for position, optimizer in enumerate(profiles):
            part = frame[(frame.optimizer == optimizer) & (frame.width == width)]
            if part.empty:
                continue
            row = part.iloc[0]
            if not np.isfinite(row[metric]):
                continue
            ax.errorbar(
                position + offset,
                row[metric],
                yerr=[[row[metric] - row[metric + "_low"]], [row[metric + "_high"] - row[metric]]],
                color=COLORS[width],
                marker="o",
                capsize=4,
                label=f"width {width}" if position == 0 else None,
            )
    ax.set_xticks(range(len(profiles)), [LABELS[p] for p in profiles], fontsize=8)
    ax.grid(axis="y", alpha=0.18)


def plots(grad, actual, movement, output):
    profiles = ["sgd_1", "momentum_01"]
    early = grad[(grad.comparison == "mse_0.2") & (grad.probes == 512)]
    realized = actual[
        (actual.comparison == "mse_0.2") & (actual.probes == 512) & (actual.scale == 1)
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, field, title in zip(
        axes[0],
        ["P", "N", "A"],
        ["Positive neuron mass P", "Negative neuron mass N", "Raw gradient agreement A"],
    ):
        interval_points(ax, early, field, profiles)
        ax.set_title(title)
    interval_points(axes[1, 0], early, "p_negative", profiles)
    axes[1, 0].set_title("Negative raw gradient pairs")
    interval_points(axes[1, 1], realized, "linear_harm_fraction", profiles)
    axes[1, 1].set_title("Actual update: first-order harmful batches")
    interval_points(axes[1, 2], realized, "finite_harm_fraction", profiles)
    axes[1, 2].set_title("Actual update: batches whose loss increases")
    for ax in axes.flat:
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Teacher 8: comparable early MSE ≈ .2 · 40 paired seeds · 512 probes", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.text(
        0.5,
        0.01,
        "Pointwise 95% seed-bootstrap intervals. Raw gradients and actual optimizer directions answer different questions.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output, "momentum_gradients")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row_index, comparison in enumerate(["mse_0.2", "mse_0.01"]):
        frame = movement[
            (movement.comparison == comparison)
            & (movement.direction == "optimizer")
            & (movement.scale == 1)
            & (movement.subset == "well_fit_0.05")
        ]
        selected_profiles = profiles if comparison == "mse_0.2" else [*profiles, "sgd_01"]
        for ax, metric, title in zip(
            axes[row_index],
            ["cancellation_fraction", "R", "joint_harm_per_source_gain"],
            ["Partial cancellation", "Net interaction energy R", "Harm / source improvement"],
        ):
            interval_points(ax, frame, metric, selected_profiles)
            ax.set_title(f"MSE ≈ {comparison[4:]}: {title}")
        axes[row_index, 1].axhline(1, ls=":", color="gray")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Exact neuron movements on predictions accurate at BOTH widths", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(
        0.5,
        0.013,
        "32 possible optimizer steps per state; 40 early seeds, later 8 / 7 / 5 seeds. Ratios with nonpositive source improvement are undefined.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output, "momentum_neurons")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row_index, optimizer in enumerate(profiles):
        for ax, subset, title in zip(
            axes[row_index],
            ["all", "well_fit_0.05"],
            ["All test examples", "Accurate at both widths"],
        ):
            frame = movement[
                (movement.optimizer == optimizer)
                & (movement.comparison == "mse_0.2")
                & (movement.direction == "optimizer")
                & (movement.subset == subset)
            ]
            for width, offset in [(8, -0.1), (16, 0.1)]:
                p = frame[frame.width == width].sort_values("scale")
                values = p.joint_harm_per_source_gain.to_numpy()
                ax.errorbar(
                    np.arange(len(p)) + offset,
                    values,
                    yerr=np.array(
                        [
                            values - p.joint_harm_per_source_gain_low.to_numpy(),
                            p.joint_harm_per_source_gain_high.to_numpy() - values,
                        ]
                    ),
                    color=COLORS[width],
                    fmt="o",
                    capsize=4,
                    label=f"width {width}",
                )
            ax.set(
                title=f"{LABELS[optimizer]} · {title}",
                xticks=[0, 1],
                xticklabels=["1% step", "Full step"],
                ylabel="harm / source-batch improvement",
                ylim=(0, None),
            )
            ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Step-length sensitivity at the same early states", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.text(
        0.5,
        0.01,
        "40 paired seeds. Shortening a diagnostic step does not retrain the network at a different learning rate.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output, "momentum_step_sensitivity")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["A", "P", "N"]):
        for optimizer, linestyle in [("sgd_1", "-"), ("momentum_01", "--")]:
            for width in [8, 16]:
                p = grad[
                    (grad.optimizer == optimizer)
                    & (grad.width == width)
                    & (grad.comparison_kind == "equal_steps")
                    & (grad.probes == 128)
                ].copy()
                p["step"] = p.comparison.str[5:].astype(int)
                p = p.sort_values("step")
                ax.plot(
                    p.step,
                    p[metric],
                    color=COLORS[width],
                    ls=linestyle,
                    marker="o",
                    ms=3,
                    label=f"{'SGD' if optimizer == 'sgd_1' else 'Momentum'} · width {width}",
                )
        ax.set_xscale("symlog", linthresh=20)
        ax.set_xticks([0, 100, 1000, 10000, 100000], ["0", "100", "1k", "10k", "100k"])
        ax.set(title=metric, xlabel="completed updates")
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Raw gradient geometry over training · 10 seeds per width/profile")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.text(
        0.5,
        0.01,
        "Equal steps compare different losses. Lines connect saved states; late ratios may involve numerical floors.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output, "momentum_agreement")


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "runs/pier_study/momentum"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    selected = selection(root, PROFILES)
    selected = selected[
        (selected.optimizer != "sgd_01") | (selected.comparison_kind == "matched_loss")
    ]
    selected.to_csv(output / "selected_states.csv", index=False)
    trajectory_groups = ["optimizer", "comparison", "comparison_kind"]
    statistics(selected, trajectory_groups, TRAJECTORY_METRICS).to_csv(
        output / "trajectory_summary.csv", index=False
    )
    effects(selected, trajectory_groups, TRAJECTORY_METRICS).to_csv(
        output / "trajectory_effects.csv", index=False
    )
    data = torch.load(root / "runs/pier_distributions/data.pt", weights_only=True)
    for collection in selected.collection.unique():
        other = torch.load(root / "runs" / collection / "data.pt", weights_only=True)
        for key in ["x_train", "y_train", "x_test", "y_test", "probe_indices"]:
            torch.testing.assert_close(data[key], other[key], atol=0, rtol=0)
    fresh_indices = torch.load(
        root / "runs/pier_study/tail_checks/probe_indices.pt", weights_only=True
    )
    snapshot_cache, source_hashes = {}, {}
    gradients, neurons, actual = [], [], []
    for index, row in enumerate(selected.itertuples()):
        path = row.source_checkpoint
        if path not in snapshot_cache:
            snapshot_cache[path] = load_checkpoint(row.run_path, int(row.step))
            source_hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        snapshot = snapshot_cache[path]
        identity = {
            k: getattr(row, k)
            for k in [
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
                    / f"w{row.width:04d}_seed{row.seed}_{row.optimizer}.pt"
                )
                fresh = torch.load(fresh_path, weights_only=True)
                torch.testing.assert_close(
                    fresh["parameters_before"], snapshot["parameters_before"]
                )
                g = fresh["probe_gradients"]
                indices = fresh_indices
                source_hashes[str(fresh_path)] = hashlib.sha256(fresh_path.read_bytes()).hexdigest()
            else:
                g = snapshot["probe_gradients"]
                indices = data["probe_indices"]
            raw, units = gradient_masses(g)
            raw["A"] = ratio(
                snapshot["mean_gradient"].double().square().sum().item(), raw["gradient_energy"]
            )
            raw.update(step_transfer_masses(g, candidate_descent(snapshot, g)))
            dot, _ = alignment_arrays(g.double())
            moments = distribution_summary("dot", dot)
            raw["skewness"] = moments["dot_skewness"]
            raw["excess_kurtosis"] = moments["dot_excess_kurtosis"]
            gradients.append({**identity, "probes": probes, **raw})
            neurons.extend({**identity, "probes": probes, **unit} for unit in units)
            actual.extend(
                {**identity, "probes": probes, **result}
                for result in recorded_step(
                    snapshot, g, data["x_train"][indices], data["y_train"][indices]
                )
            )
        if index % 50 == 0:
            print(f"Gradients/recorded steps: {index + 1}/{len(selected)}", flush=True)
    gradient_frame = pd.DataFrame(gradients)
    actual_frame = pd.DataFrame(actual)
    gradient_frame.to_csv(output / "gradient_seeds.csv", index=False)
    pd.DataFrame(neurons).to_csv(output / "gradient_neurons.csv", index=False)
    actual_frame.to_csv(output / "actual_step_seeds.csv", index=False)
    groups = ["optimizer", "comparison", "comparison_kind", "probes"]
    grad_summary = statistics(gradient_frame, groups, GRADIENT_METRICS)
    grad_summary.to_csv(output / "gradient_summary.csv", index=False)
    effects(gradient_frame, groups, GRADIENT_METRICS).to_csv(
        output / "gradient_effects.csv", index=False
    )
    actual_summary = statistics(actual_frame, [*groups, "scale"], ACTUAL_METRICS)
    actual_summary.to_csv(output / "actual_step_summary.csv", index=False)
    effects(actual_frame, [*groups, "scale"], ACTUAL_METRICS).to_csv(
        output / "actual_step_effects.csv", index=False
    )

    indices = data["probe_indices"][:32]
    torch.save(indices, output / "candidate_source_indices.pt")
    x_source, y_source = data["x_train"][indices], data["y_train"][indices]
    x_test, y_test = data["x_test"].double(), data["y_test"].double()
    functional = selected[(selected.step <= 19999) | (selected.comparison_kind == "matched_loss")]
    records, batches = [], []
    seen = 0
    for _, pair in functional.groupby(["optimizer", "comparison", "seed"]):
        states = {int(r.width): (r, snapshot_cache[r.source_checkpoint]) for r in pair.itertuples()}
        assert set(states) == {8, 16}
        residuals = {}
        for width, (_, snapshot) in states.items():
            pred, _, _ = prediction_moves(
                snapshot["parameters_before"],
                torch.zeros_like(snapshot["actual_step"])[None],
                x_test,
            )
            residuals[width] = pred - y_test
        masks = {"all": torch.ones(len(y_test), dtype=torch.bool)}
        for threshold in [0.05, 0.1]:
            masks[f"well_fit_{threshold:g}"] = (residuals[8].abs() <= threshold) & (
                residuals[16].abs() <= threshold
            )
        for width, (row, snapshot) in states.items():
            base = {
                k: getattr(row, k)
                for k in [
                    "optimizer",
                    "comparison",
                    "comparison_kind",
                    "width",
                    "seed",
                    "step",
                    "train_mse",
                ]
            }
            modes = (
                ["optimizer", "gradient_only"] if row.optimizer == "momentum_01" else ["optimizer"]
            )
            for direction in modes:
                descent = candidate_descent(
                    snapshot,
                    snapshot["probe_gradients"][:32],
                    include_history=direction == "optimizer",
                )
                for scale in [0.01, 1.0]:
                    updates = -snapshot["optimizer"]["lr"] * scale * descent
                    _, moves, crossings = prediction_moves(
                        snapshot["parameters_before"], updates, x_test
                    )
                    gain = source_improvement(
                        snapshot["parameters_before"], updates, x_source, y_source
                    )
                    for subset, mask in masks.items():
                        stats, _ = movement_statistics(residuals[width], moves, mask)
                        result = record_movement(stats, gain, row.train_mse)
                        result["gate_crossing_fraction"] = crossings[:, mask].double().mean().item()
                        identity = {
                            **base,
                            "direction": direction,
                            "scale": scale,
                            "subset": subset,
                            "n_examples": mask.sum().item(),
                        }
                        records.append({**identity, **result})
                        batches.extend(
                            {
                                **identity,
                                "source_batch": b,
                                "source_gain": gain[b].item(),
                                **{k: v[b].item() for k, v in stats.items()},
                            }
                            for b in range(32)
                        )
            seen += 1
            if seen % 20 == 0:
                print(f"Neuron changes: {seen}/{len(functional)}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "movement_seeds.csv", index=False)
    pd.DataFrame(batches).to_csv(output / "movement_batches.csv", index=False)
    groups = ["optimizer", "comparison", "comparison_kind", "direction", "scale", "subset"]
    move_summary = statistics(frame, groups, MOVEMENT_METRICS)
    move_summary.to_csv(output / "movement_summary.csv", index=False)
    effects(frame, groups, MOVEMENT_METRICS).to_csv(output / "movement_effects.csv", index=False)
    write_contrasts(gradient_frame, actual_frame, frame, output)
    plots(grad_summary, actual_summary, move_summary, output)
    manifest = {
        "teacher_width": 8,
        "widths": [8, 16],
        "candidate_batches": 32,
        "test_examples": 4096,
        "scales": [0.01, 1.0],
        "source_sha256": source_hashes,
        "direction": "momentum: -(lr)*(beta*pre_step_buffer + g_B); freeze history across candidate B",
        "gradient_only": "same momentum-trained state, just -lr*g_B; no history",
        "uncertainty": "4000 paired seed-bootstrap draws; seed 741; conditional on fixed teacher/data",
        "ratio_policy": "H/U undefined if mean source improvement U <= 0; n_valid counts retained",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Saved {len(snapshot_cache)} checkpoints / {len(frame)} movement rows: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
