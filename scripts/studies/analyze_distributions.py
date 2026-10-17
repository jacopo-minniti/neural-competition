#!/usr/bin/env python3
"""Rebuild distribution figures and seed-level estimates without training."""

import argparse
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
    LABELS,
    analyze_states,
    finite_step_scales,
    matched_states,
    paired_width_effect,
    read_study,
    seed_mean_interval,
)

COLORS = dict(zip([8, 9, 10, 16, 32], plt.get_cmap("viridis")(np.linspace(0.08, 0.9, 5))))
MAIN = ["sgd_1", "momentum_01", "adam_0001"]


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def estimate_table(frame, group, metrics):
    rows = []
    for identity, part in frame.groupby(group):
        identity = identity if isinstance(identity, tuple) else (identity,)
        record = dict(zip(group, identity))
        record["n_seeds"] = part["seed"].nunique()
        for metric in metrics:
            mean, low, high = seed_mean_interval(part[metric])
            record.update({metric: mean, f"{metric}_low": low, f"{metric}_high": high})
        rows.append(record)
    return pd.DataFrame(rows)


def plot_training(metrics, outcomes, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    fixed = [0, 20, 100, 700, 1999, 4999, 19999, 59999, 99999]
    for ax, optimizer in zip(axes, MAIN):
        part = metrics[(metrics.optimizer == optimizer) & metrics.step.isin(fixed)]
        for width, group in part.groupby("width"):
            grouped = group.groupby("step").test_mse
            q = grouped.quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(q.index + 1, q[0.5], color=COLORS[width], label=f"width {width}")
            ax.fill_between(q.index + 1, q[0.25], q[0.75], color=COLORS[width], alpha=0.12)
        ax.set(
            xscale="log", yscale="log", xlabel="updates", title=LABELS[optimizer], ylim=(1e-12, 2)
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("test MSE: median and middle 50% of seeds")
    axes[-1].legend(fontsize=8)
    fig.suptitle("One fixed teacher; 10 consecutive training seeds per setting")
    fig.tight_layout()
    save_figure(fig, output, "training")
    final = outcomes.copy()
    final["good_fit"] = final.test_mse < 1e-6
    final.groupby(["collection", "optimizer", "width"]).agg(
        n=("seed", "count"),
        mean_mse=("test_mse", "mean"),
        median_mse=("test_mse", "median"),
        good_fits=("good_fit", "sum"),
    ).to_csv(output / "outcomes_summary.csv")


def plot_cdf(arrays, selected, output, target=0.01):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for j, optimizer in enumerate(MAIN):
        for i, field in enumerate(["raw_dot", "scaled_dot"]):
            ax = axes[i, j]
            group = selected[selected.optimizer == optimizer]
            for width in sorted(group.width.unique()):
                distributions = [
                    arrays[(optimizer, int(width), int(seed))][field]
                    for seed in group[group.width == width].seed
                ]
                pooled = np.concatenate(distributions)
                lo, hi = np.quantile(pooled, [0.001, 0.999])
                grid = np.linspace(lo, hi, 250)
                cdfs = np.array([np.searchsorted(np.sort(x), grid) / len(x) for x in distributions])
                mean = cdfs.mean(axis=0)
                rng = np.random.default_rng(854)
                indices = rng.integers(len(cdfs), size=(1000, len(cdfs)))
                boot = cdfs[indices].mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975], axis=0)
                ax.plot(grid, mean, color=COLORS[width], label=f"width {width}")
                ax.fill_between(grid, low, high, color=COLORS[width], alpha=0.1)
            ax.axvline(0, color="black", lw=0.6)
            if i == 0:
                ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
            ax.set(
                xlabel="gB · gC" if i == 0 else "gB · gC / mean ||gB||²",
                ylabel="fraction ≤ x",
                title=LABELS[optimizer],
                ylim=(0, 1),
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Near training MSE {target:g}: full distributions and 95% seed bootstrap bands")
    fig.tight_layout()
    save_figure(fig, output, f"gradient_distributions_{target:g}")


def plot_tails(arrays, selected, output, target=0.01):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    grid = np.linspace(0, 5, 180)
    for ax, optimizer in zip(axes, MAIN):
        group = selected[selected.optimizer == optimizer]
        for width, part in group.groupby("width"):
            curves = []
            for seed in part.seed:
                x = arrays[(optimizer, int(width), int(seed))]["scaled_dot"]
                z = (x - x.mean()) / x.std()
                curves.append([(np.mean(z < -t), np.mean(z > t)) for t in grid])
            curves = np.asarray(curves).mean(axis=0)
            ax.plot(grid, curves[:, 1], color=COLORS[width], label=f"width {width}")
            ax.plot(grid, curves[:, 0], color=COLORS[width], ls="--")
        ax.set(
            yscale="log",
            ylim=(1e-4, 1),
            xlabel="distance from mean (standard deviations)",
            title=LABELS[optimizer],
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("tail probability: right solid, left dashed")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Tail asymmetry near MSE {target:g}; no power-law fit implied")
    fig.tight_layout()
    save_figure(fig, output, f"gradient_tails_{target:g}")


def plot_summaries(summary, output, target=0.01):
    fields = [
        "cosine_negative_fraction",
        "scaled_dot_skewness",
        "scaled_dot_excess_kurtosis",
        "mean_cosine_negative_fraction",
        "step_alignment_negative_fraction",
        "batch_sharpness_ratio",
    ]
    titles = [
        "negative gradient pairs",
        "dot-product skewness",
        "dot-product excess kurtosis",
        "gradient opposes full gradient",
        "actual step: first-order harm",
        "batch sharpness (paper definition)",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, metric, title in zip(axes.flat, fields, titles):
        for optimizer in MAIN:
            part = summary[summary.optimizer == optimizer].sort_values("width")
            ax.errorbar(
                part.width,
                part[metric],
                yerr=[part[metric] - part[f"{metric}_low"], part[f"{metric}_high"] - part[metric]],
                marker="o",
                label=LABELS[optimizer],
                capsize=3,
            )
        ax.set(xlabel="student width", title=title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Near training MSE {target:g}: seed means and 95% bootstrap intervals")
    fig.tight_layout()
    save_figure(fig, output, f"width_statistics_{target:g}")


def plot_evolution(metrics, output):
    fields = [
        "pair_dot_skewness",
        "pair_cosine_negative_fraction",
        "batch_sharpness_ratio",
        "step_previous_cosine",
        "probe_linear_harm_fraction",
        "probe_loss_increase_fraction",
    ]
    titles = [
        "pair-dot skewness",
        "negative gradient pairs",
        "batch sharpness (paper definition)",
        "consecutive-step cosine",
        "predicted harmful probe steps",
        "actual harmful probe steps",
    ]
    fixed = [20, 100, 700, 1999, 4999, 19999, 59999, 99999]
    for optimizer in MAIN:
        part = metrics[(metrics.optimizer == optimizer) & metrics.step.isin(fixed)]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, field, title in zip(axes.flat, fields, titles):
            for width, group in part.groupby("width"):
                q = group.groupby("step")[field].quantile([0.25, 0.5, 0.75]).unstack()
                ax.plot(q.index + 1, q[0.5], color=COLORS[width], label=f"width {width}")
                ax.fill_between(q.index + 1, q[0.25], q[0.75], color=COLORS[width], alpha=0.1)
            ax.set(xscale="log", xlabel="updates", title=title)
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{LABELS[optimizer]}: evolution (median and middle 50% of seeds)")
        fig.tight_layout()
        save_figure(fig, output, f"evolution_{optimizer}")


def plot_directions(arrays, selected, output):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    fields = ["mean_cosine", "moment_alignment", "step_alignment"]
    titles = [
        "probe gradient · full gradient",
        "probe gradient · updated moment",
        "probe gradient · descent direction",
    ]
    grid = np.linspace(-1, 1, 201)
    for j, optimizer in enumerate(MAIN):
        group = selected[selected.optimizer == optimizer]
        for i, (field, title) in enumerate(zip(fields, titles)):
            ax = axes[i, j]
            for width, part in group.groupby("width"):
                curves = []
                for seed in part.seed:
                    x = arrays[(optimizer, int(width), int(seed))][field]
                    x = np.sort(x[np.isfinite(x)])
                    if len(x):
                        curves.append(np.searchsorted(x, grid) / len(x))
                if curves:
                    ax.plot(
                        grid, np.mean(curves, axis=0), color=COLORS[width], label=f"width {width}"
                    )
            ax.axvline(0, color="black", lw=0.6)
            ax.set(xlabel="cosine (negative = opposing)", ylabel="fraction ≤ x", title=title)
            ax.grid(alpha=0.2)
            if i == 0:
                ax.text(
                    0.5,
                    1.22,
                    LABELS[optimizer],
                    transform=ax.transAxes,
                    ha="center",
                    fontweight="bold",
                )
            if i == 1 and optimizer == "sgd_1":
                ax.text(0.5, 0.5, "SGD has no momentum buffer", transform=ax.transAxes, ha="center")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Different directions at the same early loss (MSE ≈ 0.2)", y=1.01)
    fig.tight_layout()
    save_figure(fig, output, "gradient_moment_step_distributions")


def plot_step_scales(frame, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for i, stage in enumerate(["early", "near_fit"]):
        for j, optimizer in enumerate(MAIN):
            ax = axes[i, j]
            part = frame[(frame.optimizer == optimizer) & (frame.stage == stage)]
            for width, group in part.groupby("width"):
                q = (
                    group.groupby("scale")
                    .actual_harm_fraction.quantile([0.25, 0.5, 0.75])
                    .unstack()
                )
                ax.plot(q.index, q[0.5], marker=".", color=COLORS[width], label=f"width {width}")
                ax.fill_between(q.index, q[0.25], q[0.75], color=COLORS[width], alpha=0.1)
            ax.set(
                xscale="log",
                xlabel="multiplier of the saved actual update",
                ylim=(0, 1),
                title=f"{LABELS[optimizer]}, {stage.replace('_', ' ')}",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("fraction of probe losses increasing")
    axes[1, 0].set_ylabel("fraction of probe losses increasing")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Rescaling one realized step at a frozen state (float64 loss evaluation)")
    fig.tight_layout()
    save_figure(fig, output, "finite_step_intervention")


def analyze_controls(root, metrics, outcomes, output):
    _, initial = read_study(root, names=("pier_initialization_control",))
    zero = outcomes[(outcomes.optimizer == "adam_0001") & outcomes.width.isin([8, 16])].copy()
    zero["initialization"] = "zero output weights"
    initial["initialization"] = "random output weights, sd 0.1"
    pd.concat([zero, initial]).groupby(["initialization", "width"]).agg(
        n=("seed", "count"),
        median_test_mse=("test_mse", "median"),
        mean_test_mse=("test_mse", "mean"),
    ).to_csv(output / "initialization_summary.csv")
    signs = []
    for _, row in pd.concat([outcomes, initial]).iterrows():
        final = torch.load(Path(row.run_path) / "final.pt", weights_only=True)
        a = final["parameters"][:, -1]
        signs.append(
            {
                **row.to_dict(),
                "positive_units": (a > 0).sum().item(),
                "negative_units": (a < 0).sum().item(),
            }
        )
    pd.DataFrame(signs).to_csv(output / "sign_diagnostics.csv", index=False)
    folder = root / "runs/pier_stability_pilot"
    summaries, histories = [], []
    for path in sorted(folder.glob("w*/state.json")):
        state = json.loads(path.read_text())
        spec = state["optimizer"]
        identity = dict(width=state["width"], seed=state["seed"], optimizer=spec["label"])
        frame = pd.read_csv(path.parent / "metrics.csv")
        frame["train_mse"] = 2 * frame.train_loss
        frame["test_mse"] = 2 * frame.test_loss
        reference = (
            2 / spec["lr"] if spec["name"] == "sgd" else 2 * (1 - spec["momentum"]) / spec["lr"]
        )
        frame["sharpness_reference_ratio"] = frame.batch_sharpness_ratio / reference
        for key, value in identity.items():
            frame[key] = value
        histories.append(frame)
        summaries.append(
            {
                **identity,
                "status": state["status"],
                "final_test_mse": 2 * state.get("final_test_loss", np.nan),
                "final_train_mse": 2 * state.get("final_train_loss", np.nan),
                "max_observed_train_mse": frame.train_mse.max(),
                "late_sharpness_reference_ratio": (
                    frame.tail(3).sharpness_reference_ratio.mean()
                    if state["status"] == "complete"
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "stability_runs.csv", index=False)
    summary.groupby(["optimizer", "width"]).agg(
        n=("seed", "count"),
        complete=("status", lambda x: (x == "complete").sum()),
        diverged=("status", lambda x: (x == "diverged").sum()),
        median_mse_completed=("final_test_mse", "median"),
        median_late_reference_ratio=("late_sharpness_reference_ratio", "median"),
    ).to_csv(output / "stability_summary.csv")
    history = pd.concat(histories)
    history.to_csv(output / "stability_metrics.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    names = dict(
        sgd_2="SGD lr 2",
        sgd_4="SGD lr 4",
        momentum_02="Momentum lr 0.2",
        momentum_04="Momentum lr 0.4",
    )
    for i, width in enumerate([8, 16]):
        part = history[
            (history.width == width) & history.step.isin([20, 100, 700, 1999, 4999, 9999, 19999])
        ]
        for ax, field, title in zip(
            axes[i],
            ["test_mse", "sharpness_reference_ratio", "step_previous_cosine"],
            ["test MSE", "batch sharpness / reference", "consecutive-step cosine"],
        ):
            for optimizer, group in part.groupby("optimizer"):
                q = group.groupby("step")[field].quantile([0.25, 0.5, 0.75]).unstack()
                ax.plot(q.index + 1, q[0.5], label=names[optimizer])
                ax.fill_between(q.index + 1, q[0.25], q[0.75], alpha=0.1)
            ax.set(xscale="log", xlabel="updates", title=f"width {width}: {title}")
            if field == "test_mse":
                ax.set(yscale="log", ylim=(1e-10, 2))
            if field == "sharpness_reference_ratio":
                ax.axhline(1, color="black", ls="--", lw=0.7)
                ax.set(ylim=(0, 2))
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("High-rate pilot: five seeds; 3 divergent starts counted in the table")
    fig.tight_layout()
    save_figure(fig, output, "stability_pilot")


def analyze_continuations(root, output):
    metrics, outcomes = read_study(root, names=("pier_stability_continuation",))
    outcomes.to_csv(output / "continuation_outcomes.csv", index=False)
    summary = outcomes.groupby("optimizer").agg(
        n=("seed", "count"),
        median_test_mse=("test_mse", "median"),
        mean_test_mse=("test_mse", "mean"),
    )
    summary.to_csv(output / "continuation_summary.csv")
    metrics.to_csv(output / "continuation_metrics.csv", index=False)
    fixed = [0, 20, 100, 499, 1999, 4999, 9999]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for optimizer, part in metrics[metrics.step.isin(fixed)].groupby("optimizer"):
        for ax, field, title in zip(
            axes,
            ["test_mse", "batch_sharpness_ratio", "step_previous_cosine"],
            ["test MSE", "batch sharpness", "consecutive-step cosine"],
        ):
            q = part.groupby("step")[field].quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(q.index, q[0.5], label=f"learning rate {optimizer.split('_')[-1]}")
            ax.fill_between(q.index, q[0.25], q[0.75], alpha=0.12)
            ax.set(xscale="symlog", xlabel="additional updates from saved state", title=title)
            if field == "test_mse":
                ax.set(yscale="log", ylim=(1e-6, 2))
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Rate intervention at identical width-16 states after 20k SGD(lr 4) updates; five paired seeds"
    )
    fig.tight_layout()
    save_figure(fig, output, "stability_continuation")
    data = torch.load(root / "runs/pier_stability_pilot/data.pt", weights_only=True)
    indices = data["probe_indices"]
    scale_frames = []
    for seed in range(5):
        snapshot = load_checkpoint(
            root / f"runs/pier_stability_pilot/w0016_seed{seed}_sgd_4", 19999
        )
        frame = finite_step_scales(snapshot, data["x_train"][indices], data["y_train"][indices])
        frame["seed"] = seed
        scale_frames.append(frame)
    pd.concat(scale_frames).to_csv(output / "stability_step_scales.csv", index=False)
    effects = []
    pivot = outcomes.pivot(index="seed", columns="optimizer", values="test_mse")
    for optimizer in ["sgd_1", "sgd_2", "sgd_5"]:
        log_ratio = np.log10(pivot[optimizer] / pivot.sgd_4)
        mean, low, high = seed_mean_interval(log_ratio)
        effects.append(
            dict(optimizer=optimizer, n=len(log_ratio), mean_log10_ratio=mean, low=low, high=high)
        )
    pd.DataFrame(effects).to_csv(output / "continuation_paired_effects.csv", index=False)


def summarize_tail_validation(root, output):
    folder = root / "runs/pier_study/tail_checks"
    fields = ["scaled_dot_skewness", "scaled_dot_excess_kurtosis", "cosine_negative_fraction"]
    frames, effects = [], []
    for size, filename in [(128, "original_128.csv"), (512, "validation_512.csv")]:
        frame = pd.read_csv(folder / filename)
        summary = estimate_table(frame, ["optimizer", "width"], fields)
        summary["probe_batches"] = size
        frames.append(summary)
        for optimizer, part in frame.groupby("optimizer"):
            for field in fields:
                effects.append(
                    dict(
                        optimizer=optimizer, probe_batches=size, **paired_width_effect(part, field)
                    )
                )
    pd.concat(frames).to_csv(output / "tail_validation_summary.csv", index=False)
    pd.DataFrame(effects).to_csv(output / "tail_validation_effects.csv", index=False)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    args = parser.parse_args()
    output = args.root / "runs/pier_study/analysis"
    output.mkdir(parents=True, exist_ok=True)
    metrics, outcomes = read_study(args.root)
    metrics.to_csv(output / "checkpoint_metrics.csv", index=False)
    outcomes.to_csv(output / "outcomes.csv", index=False)
    matched = matched_states(metrics)
    matched.to_csv(output / "loss_matching.csv", index=False)
    selected, arrays = analyze_states(matched[matched.loss_eligible], cache_arrays=True)
    selected.to_csv(output / "matched_statistics.csv", index=False)
    fields = [
        "raw_dot_mean",
        "raw_dot_std",
        "scaled_dot_mean",
        "scaled_dot_skewness",
        "scaled_dot_excess_kurtosis",
        "scaled_dot_q01",
        "scaled_dot_q99",
        "cosine_negative_fraction",
        "mean_cosine_negative_fraction",
        "step_alignment_negative_fraction",
        "moment_alignment_negative_fraction",
        "batch_sharpness_ratio",
        "probe_loss_increase_fraction",
        "train_mse",
        "step_previous_cosine",
        "gradient_signal_ratio",
    ]
    summary = estimate_table(selected, ["collection", "optimizer", "width"], fields)
    summary.to_csv(output / "matched_summary.csv", index=False)
    effects = []
    for optimizer, part in selected.groupby("optimizer"):
        for metric in fields:
            for large in [10, 16, 32]:
                effects.append(
                    {"optimizer": optimizer, **paired_width_effect(part, metric, large=large)}
                )
    pd.DataFrame(effects).to_csv(output / "paired_effects.csv", index=False)
    # A shared update count complements matching on the attained loss.
    equal, _ = analyze_states(
        metrics[(metrics.step == 1999) & (metrics.collection == "pier_distributions")]
    )
    equal.to_csv(output / "step2000_statistics.csv", index=False)
    estimate_table(equal, ["optimizer", "width"], fields).to_csv(
        output / "step2000_summary.csv", index=False
    )
    plot_training(metrics, outcomes, output)
    plot_cdf(arrays, selected[selected.collection == "pier_distributions"], output)
    plot_tails(arrays, selected[selected.collection == "pier_distributions"], output)
    plot_summaries(summary[summary.collection == "pier_distributions"], output)
    plot_evolution(metrics, output)
    early_metrics, _ = read_study(args.root, names=("pier_early", "pier_early_extra"))
    early_match = matched_states(early_metrics, target=0.2, tolerance=0.1)
    early_match.to_csv(output / "early_loss_matching.csv", index=False)
    early, early_arrays = analyze_states(early_match[early_match.loss_eligible], cache_arrays=True)
    early.to_csv(output / "early_statistics.csv", index=False)
    early_summary = estimate_table(early, ["optimizer", "width"], fields)
    early_summary.to_csv(output / "early_summary.csv", index=False)
    early_effects = []
    for optimizer, part in early.groupby("optimizer"):
        for metric in fields:
            for large in [10, 16, 32]:
                early_effects.append(
                    {"optimizer": optimizer, **paired_width_effect(part, metric, large=large)}
                )
    pd.DataFrame(early_effects).to_csv(output / "early_paired_effects.csv", index=False)
    plot_cdf(early_arrays, early, output, target=0.2)
    plot_tails(early_arrays, early, output, target=0.2)
    plot_summaries(early_summary, output, target=0.2)
    plot_directions(early_arrays, early, output)
    data = torch.load(args.root / "runs/pier_distributions/data.pt", weights_only=True)
    indices = data["probe_indices"]
    x_probe, y_probe = data["x_train"][indices], data["y_train"][indices]
    scales = []
    for label, states in [("early", early), ("near_fit", selected)]:
        for _, row in states.iterrows():
            snapshot = load_checkpoint(row.run_path, int(row.step))
            frame = finite_step_scales(snapshot, x_probe, y_probe)
            for key in ["optimizer", "width", "seed", "step", "train_mse", "collection"]:
                frame[key] = row[key]
            frame["stage"] = label
            scales.append(frame)
    scales = pd.concat(scales, ignore_index=True)
    scales.to_csv(output / "finite_step_scales.csv", index=False)
    plot_step_scales(scales, output)
    analyze_controls(args.root, metrics, outcomes, output)
    analyze_continuations(args.root, output)
    summarize_tail_validation(args.root, output)
    print(outcomes.groupby(["collection", "optimizer", "width"]).test_mse.agg(["count", "median"]))
    print(f"Saved figures and tables: {output}")


if __name__ == "__main__":
    main()
