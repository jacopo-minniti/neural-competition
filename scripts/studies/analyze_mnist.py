#!/usr/bin/env python3
"""Publish MNIST seed-level results, paired contrasts, and scientific figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "docs/tables/mnist"
FIGURES = ROOT / "docs/figures/mnist"
OPTIMIZERS = ["sgd", "momentum", "adam"]
WIDTHS = [16, 32, 64, 128]
COLORS = dict(zip(WIDTHS, ["#bd3b35", "#d08b23", "#328477", "#315cb3"]))
NAMES = {"sgd": "SGD", "momentum": "Momentum", "adam": "Adam"}


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(741)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return float(values.mean()), *np.quantile(draws, [0.025, 0.975]).tolist()


def summarize(frame, keys, metrics):
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            if metric in group:
                values = group[metric].dropna()
                mean, low, high = interval(values)
                base = dict(zip(keys, key))
                if "metric" in base:
                    base["statistic"] = base.pop("metric")
                rows.append(base | dict(metric=metric, mean=mean, low=low, high=high, n=len(values)))
    return pd.DataFrame(rows)


def save(frame, name):
    frame.to_csv(TABLES / (name + ".csv.gz"), index=False, float_format="%.17g",
                 compression={"method": "gzip", "mtime": 0})


def read_collection(root, collection):
    frames = {key: [] for key in ["epochs", "diagnostics", "transfers", "step_controls", "budgets"]}
    finals, audits = [], []
    for marker in sorted((root / collection).glob("*/complete.json")):
        result = json.loads(marker.read_text())
        task = result["task"]
        assert not result["diverged"]
        metadata = {key: task[key] for key in ["loss", "optimizer", "width", "seed", "lr"]}
        metadata |= dict(collection=collection, run=marker.parent.name)
        for name in frames:
            table = pd.read_csv(marker.parent / (name + ".csv.gz"))
            frames[name].append(table.assign(**metadata))
        budget = frames["budgets"][-1].iloc[-1]
        final = dict(**metadata, train_loss=result["train"]["train_loss"],
                     train_accuracy=result["train"]["train_accuracy"],
                     val_loss=result["train"]["val_loss"],
                     val_accuracy=result["train"]["val_accuracy"],
                     test_loss=result["test"]["loss"], test_accuracy=result["test"]["accuracy"],
                     completed=result["completed"], seconds=result["seconds"])
        final |= {key: float(budget[key]) for key in budget.index
                  if key.startswith("cumulative_") or key.startswith("sampling_se_") or key in [
                      "actual_progress", "initial_reference_loss", "reference_loss"]}
        gain = final["cumulative_gain"]
        final |= dict(harm_per_gain=final["cumulative_harm"] / gain,
                      pair_plus_per_gain=final["cumulative_pair_plus"] / gain,
                      net_all_per_gain=final["cumulative_net_all_from_endpoints"] / gain,
                      progress_per_gain=final["actual_progress"] / gain,
                      sampled_pair_progress=(gain - final["cumulative_harm"]
                                             - final["cumulative_pair_net"]),
                      sampling_relative_se_gain=final["sampling_se_gain"] / gain)
        finals.append(final)
        audits.append(dict(**metadata, max_budget_error=result["max_budget_error"],
                           max_pair_identity_error=result["max_pair_identity_error"],
                           max_logit_additivity_error=result["max_logit_additivity_error"]))
    return pd.DataFrame(finals), {key: pd.concat(value, ignore_index=True)
                                 for key, value in frames.items()}, pd.DataFrame(audits)


def use_dense_main(root, finals, budgets):
    """Replace sampled main G/H/C_all by independently audited every-update sums."""
    markers = sorted((root / "dense_audit").glob("*/complete.json"))
    assert len(markers) == 120, "Complete every-update accounting for the whole main cohort"
    curves, comparisons = [], []
    for marker in markers:
        audit = json.loads(marker.read_text())
        task = audit["task"]
        run = marker.parent.name
        fmask = (finals.collection == "main") & (finals.run == run)
        bmask = (budgets.collection == "main") & (budgets.run == run)
        assert fmask.sum() == 1
        exact = pd.read_csv(marker.parent / "exact_curves.csv.gz").set_index("completed")
        for field in ["gain", "harm", "net_all"]:
            comparisons.append(dict(loss=task["loss"], optimizer=task["optimizer"],
                                    width=task["width"], seed=task["seed"], field=field,
                                    **audit[field]))
            value = audit[field]["exact"]
            finals.loc[fmask, "cumulative_" + field] = value
            if field == "net_all":
                finals.loc[fmask, "cumulative_net_all_from_endpoints"] = value
            for row in budgets.index[bmask]:
                end = int(budgets.loc[row, "completed"])
                budgets.loc[row, "cumulative_" + field] = exact.loc[end, field]
                if field == "net_all":
                    budgets.loc[row, "cumulative_net_all_from_endpoints"] = exact.loc[end, field]
        metadata = {key: task[key] for key in ["loss", "optimizer", "width", "seed", "lr"]}
        curves.append(exact.reset_index().assign(**metadata))
        assert abs(audit["gain"]["exact"] - audit["harm"]["exact"]
                   - audit["net_all"]["exact"] - float(finals.loc[fmask, "actual_progress"].iloc[0])) < 1e-9
    for field, numerator in [("harm_per_gain", "cumulative_harm"),
                             ("pair_plus_per_gain", "cumulative_pair_plus"),
                             ("net_all_per_gain", "cumulative_net_all_from_endpoints"),
                             ("progress_per_gain", "actual_progress")]:
        mask = finals.collection == "main"
        finals.loc[mask, field] = finals.loc[mask, numerator] / finals.loc[mask, "cumulative_gain"]
    finals["accounting"] = np.where(finals.collection == "main", "every_update", "sampled_updates")
    save(pd.concat(curves, ignore_index=True), "exact_cumulative_curves")
    save(pd.DataFrame(comparisons), "sampling_audit")
    return finals, budgets


def complete_tuned(frame, identity):
    """Width-tuned comparisons reuse main runs where the chosen rate is unchanged."""
    main = frame[(frame.collection == "main") & frame.width.isin([16, 128])].copy()
    changed = frame[frame.collection == "width_tuned"].copy()
    main = main.set_index(identity)
    changed = changed.set_index(identity)
    reused = main.drop(index=changed.index, errors="ignore").reset_index()
    reused["source_collection"] = "main"
    changed = changed.reset_index().assign(source_collection="width_tuned")
    result = pd.concat([reused, changed], ignore_index=True)
    result["collection"] = "width_tuned_complete"
    return pd.concat([frame, result], ignore_index=True)


def paired(frame, keys, metrics):
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        narrow = group[group.width == 16].set_index("seed")
        for width in [32, 64, 128]:
            wide = group[group.width == width].set_index("seed")
            shared = narrow.index.intersection(wide.index)
            for seed in shared:
                for metric in metrics:
                    if metric not in group:
                        continue
                    a, b = narrow.loc[seed, metric], wide.loc[seed, metric]
                    rows.append(dict(zip(keys, key)) | dict(
                        baseline=16, width=width, seed=seed, metric=metric,
                        narrow=a, wide=b, difference=b - a,
                        relative_change=b / a - 1 if a != 0 else np.nan))
    return pd.DataFrame(rows)


def load_matched(root):
    rows, transfers, coverage, audits = [], [], [], []
    for collection in ["main", "lower_lr", "width_tuned"]:
        folder = "matched" if collection == "main" else "matched_" + collection
        for marker in sorted((root / folder).glob("*/complete.json")):
            result = json.loads(marker.read_text())
            audits.append(dict(collection=collection, run=marker.parent.name,
                               accepted=result["accepted"], targets=result["targets"],
                               replay_max_error=result["replay_max_error"]))
            for name, target in [("interactions", rows), ("transfers", transfers),
                                 ("coverage", coverage)]:
                target.append(pd.read_csv(marker.parent / (name + ".csv.gz"))
                              .assign(collection=collection))
    return (pd.concat(rows, ignore_index=True), pd.concat(transfers, ignore_index=True),
            pd.concat(coverage, ignore_index=True), pd.DataFrame(audits))


def interaction_seed_summary(frame):
    keys = ["collection", "loss", "optimizer", "width", "seed", "target_loss", "fraction", "view"]
    fields = ["gain", "harm", "progress", "net_all", "pair_plus", "pair_minus", "pair_net",
              "eligible", "strict_events", "strict_harm", "eligible_gain", "harmful_neurons",
              "neurons"]
    summary = frame.groupby(keys)[fields].sum().reset_index()
    counts = frame.groupby(keys).size().reset_index(name="sources")
    summary = summary.merge(counts, on=keys, validate="one_to_one")
    for field in ["harm", "progress", "net_all", "pair_plus", "pair_minus", "pair_net"]:
        summary[field + "_per_gain"] = summary[field] / summary.gain.where(summary.gain > 0)
    summary["strict_frequency"] = summary.strict_events / summary.eligible.where(
        summary.eligible > 0)
    summary["strict_severity"] = summary.strict_harm / summary.eligible_gain.where(
        summary.eligible_gain > 0)
    summary["strict_harm_per_gain"] = summary.strict_harm / summary.gain.where(summary.gain > 0)
    summary["harmful_neuron_frequency"] = summary.harmful_neurons / summary.neurons
    summary["mean_progress"] = summary.progress / summary.sources
    summary["mean_strict_events"] = summary.strict_events / summary.sources
    summary["mean_strict_harm"] = summary.strict_harm / summary.sources
    summary["mean_eligible_pairs"] = summary.eligible / summary.sources
    summary["pair_remainder_per_gain"] = (summary.net_all - summary.pair_net) / summary.gain
    return summary


def save_figure(fig, name):
    fig.savefig(FIGURES / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def figures(epochs, budgets, finals, matched, transfers):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for row, loss in enumerate(["mse", "ce"]):
        for col, optimizer in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            group = epochs[(epochs.collection == "main") & (epochs.loss == loss)
                           & (epochs.optimizer == optimizer)]
            for width in WIDTHS:
                local = group[group.width == width]
                for _, run in local.groupby("seed"):
                    ax.plot(run.epoch, run.train_loss, color=COLORS[width], alpha=0.13, lw=0.8)
                mean = local.groupby("epoch").train_loss.mean()
                ax.plot(mean.index, mean, color=COLORS[width], label=f"width {width}", lw=2)
            ax.set_yscale("log")
            ax.set_title(f"{NAMES[optimizer]} · {'squared loss' if loss == 'mse' else 'cross-entropy'}")
            ax.grid(alpha=0.15)
            if col == 0:
                ax.set_ylabel("Full training loss")
            if row == 1:
                ax.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("MNIST: one hidden ReLU layer · five paired seeds · constant learning rates")
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    save_figure(fig, "training")
    for loss in ["mse", "ce"]:
        fig, axes = plt.subplots(4, 3, figsize=(13, 11), sharex=True)
        fields = ["cumulative_gain", "cumulative_harm", "cumulative_net_all_from_endpoints",
                  "actual_progress"]
        titles = ["Standalone gains G", "Standalone harm H", "Total interaction cost",
                  "Actual reference loss decrease"]
        for col, optimizer in enumerate(OPTIMIZERS):
            group = budgets[(budgets.collection == "main") & (budgets.loss == loss)
                            & (budgets.optimizer == optimizer)]
            for row, (field, title) in enumerate(zip(fields, titles)):
                ax = axes[row, col]
                for width in WIDTHS:
                    local = group[group.width == width].groupby("completed")
                    curve = local[["epoch", field]].mean()
                    ax.plot(curve.epoch, curve[field], color=COLORS[width], label=f"width {width}")
                ax.grid(alpha=0.15)
                if col == 0:
                    ax.set_ylabel(title + "\n(loss units)")
                if row == 0:
                    ax.set_title(NAMES[optimizer])
                if row == 3:
                    ax.set_xlabel("Epoch")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.suptitle(f"MNIST {loss.upper()}: exact every-update G, H and total interaction cost\n"
                     "Fixed 256-image reference; five-seed means; actual training updates")
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        save_figure(fig, "cumulative_" + loss)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for row, loss in enumerate(["mse", "ce"]):
        for col, optimizer in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            group = budgets[(budgets.collection == "main") & (budgets.loss == loss)
                            & (budgets.optimizer == optimizer)]
            for width in WIDTHS:
                curve = group[group.width == width].groupby("completed")[["epoch",
                                               "cumulative_pair_plus"]].mean()
                ax.plot(curve.epoch, curve.cumulative_pair_plus, color=COLORS[width],
                        label=f"width {width}")
            ax.set_title(f"{NAMES[optimizer]} · {loss.upper()}")
            ax.grid(alpha=0.15)
            if col == 0:
                ax.set_ylabel("Estimated accumulated C+\n(loss units)")
            if row == 1:
                ax.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Positive pair cost over training · same-step diagnostics, sampled updates\n"
                 "C+ includes diminishing returns; it is not a causal loss reduction")
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    save_figure(fig, "cumulative_pair_plus")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    metrics = ["harmful_frequency", "harm_per_gain", "strict_frequency", "pair_plus_per_gain"]
    metric_titles = ["Harmful batch directions (%)", "Standalone H / G (%)",
                     "Strict pair reversals (%)", "Pair C+ / G (%)"]
    for row, (loss, target) in enumerate([("mse", 0.1), ("ce", 0.3)]):
        for col, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[row, col]
            data = transfers if metric == "harmful_frequency" else matched[
                (matched.fraction == 1) & (matched.view == "reference")]
            data = data[(data.collection == "main") & (data.loss == loss)
                        & (data.target_loss == target)]
            for optimizer, color in zip(OPTIMIZERS, ["#3263a8", "#bd6c1e", "#3b8b62"]):
                x, means, lows, highs = [], [], [], []
                for width in WIDTHS:
                    values = data[(data.optimizer == optimizer) & (data.width == width)][metric]
                    mean, low, high = interval(100 * values)
                    x.append(width)
                    means.append(mean)
                    lows.append(low)
                    highs.append(high)
                ax.plot(x, means, marker="o", color=color, label=NAMES[optimizer])
                ax.fill_between(x, lows, highs, color=color, alpha=0.13)
            ax.set_xscale("log", base=2)
            ax.set_xticks(WIDTHS, WIDTHS)
            ax.grid(alpha=0.15)
            ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{'Squared loss' if loss == 'mse' else 'Cross-entropy'}\n"
                              f"training loss ≈ {target}")
            if row == 1:
                ax.set_xlabel("Hidden neurons")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Matched full training loss · common fresh diagnostic batches across widths\n"
                 "Full optimizer steps; reference set 256; exploratory 95% seed-bootstrap intervals")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, "matched_metrics")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, (loss, target) in enumerate([("mse", 0.1), ("ce", 0.3)]):
        for col, optimizer in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            for fraction, style, color in [(1.0, "-", "#3263a8"), (0.01, "--", "#b56820")]:
                local = matched[(matched.collection == "main") & (matched.loss == loss)
                                & (matched.optimizer == optimizer) & (matched.target_loss == target)
                                & (matched.fraction == fraction) & (matched.view == "reference")]
                vals = []
                for width in WIDTHS:
                    values = local[local.width == width].mean_progress / fraction
                    vals.append(interval(values))
                means, low, high = np.array(vals).T
                ax.plot(WIDTHS, means, style, marker="o", color=color,
                        label="Actual full step" if fraction == 1 else "1% step × 100")
                ax.fill_between(WIDTHS, low, high, color=color, alpha=0.12)
            ax.axhline(0, color="grey", lw=0.7)
            ax.set_xscale("log", base=2)
            ax.set_xticks(WIDTHS, WIDTHS)
            ax.grid(alpha=0.15)
            ax.set_title(f"{NAMES[optimizer]} · {loss.upper()}")
            if col == 0:
                ax.set_ylabel("Reference loss decrease\nper unit step fraction")
            if row == 1:
                ax.set_xlabel("Hidden neurons")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Finite-step effects at matched training loss\n"
                 "Same direction scaled locally; 16 fresh sources per seed; five paired seeds")
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    save_figure(fig, "step_size")


def full_objective_tables():
    raw = pd.read_csv(TABLES / "full_objective_probes.csv.gz")
    keys = ["loss", "optimizer", "width", "seed", "target_loss", "fraction"]
    metrics = ["progress", "linear_logit_work", "curvature_cost", "prediction_rms"]
    per_seed = raw.groupby(keys)[metrics].mean().reset_index()
    save(per_seed, "full_objective_per_seed")
    save(summarize(per_seed, [key for key in keys if key != "seed"], metrics),
         "full_objective_summary")
    contrasts = paired(per_seed, ["loss", "optimizer", "target_loss", "fraction"], metrics)
    save(contrasts, "full_objective_paired_per_seed")
    save(summarize(contrasts, ["loss", "optimizer", "target_loss", "fraction", "width", "metric"],
                   ["difference", "relative_change"]), "full_objective_paired_summary")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, (loss, target) in enumerate([("mse", 0.1), ("ce", 0.3)]):
        for col, optimizer in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            group = per_seed[(per_seed.loss == loss) & (per_seed.optimizer == optimizer)
                             & (per_seed.target_loss == target) & (per_seed.fraction == 1)]
            for offset, field, color, label in [(-0.23, "linear_logit_work", "#3263a8", "Logit work"),
                                                (0, "curvature_cost", "#cf8b24", "Curvature cost"),
                                                (0.23, "progress", "#398c69", "Actual progress")]:
                for i, width in enumerate([16, 128]):
                    mean, low, high = interval(group[group.width == width][field])
                    ax.bar(i + offset, mean, width=0.22, color=color,
                           label=label if i == 0 else None)
                    ax.errorbar(i + offset, mean, yerr=[[mean - low], [high - mean]],
                                color="black", capsize=2, lw=0.8)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xticks([0, 1], ["width 16", "width 128"])
            ax.set_title(f"{NAMES[optimizer]} · {loss.upper()}")
            ax.grid(axis="y", alpha=0.15)
            if col == 0:
                ax.set_ylabel("Full training loss units\nper fresh candidate update")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Actual objective effects at matched training loss\n"
                 "All 55,000 training examples · eight common fresh sources per seed · five seeds")
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    save_figure(fig, "full_objective")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "runs/mnist")
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    parts = [read_collection(args.input, name) for name in ["main", "lower_lr", "width_tuned"]]
    finals = pd.concat([p[0] for p in parts], ignore_index=True)
    frames = {key: pd.concat([p[1][key] for p in parts], ignore_index=True)
              for key in parts[0][1]}
    audits = pd.concat([p[2] for p in parts], ignore_index=True)
    assert len(finals) == 200
    finals, frames["budgets"] = use_dense_main(args.input, finals, frames["budgets"])
    finals = complete_tuned(finals, ["loss", "optimizer", "width", "seed"])
    save(finals, "final_per_seed")
    metrics = ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "test_loss",
               "test_accuracy", "cumulative_gain", "cumulative_harm", "cumulative_pair_plus",
               "cumulative_pair_net", "cumulative_net_all_from_endpoints", "actual_progress",
               "harm_per_gain", "pair_plus_per_gain", "net_all_per_gain", "progress_per_gain",
               "sampling_relative_se_gain"]
    final_summary = summarize(finals, ["collection", "loss", "optimizer", "width"], metrics)
    save(final_summary, "final_summary")
    contrasts = paired(finals, ["collection", "loss", "optimizer"], metrics)
    save(contrasts, "final_paired_per_seed")
    contrast_summary = summarize(contrasts, ["collection", "loss", "optimizer", "baseline",
                                            "width", "metric"], ["difference", "relative_change"])
    save(contrast_summary, "final_paired_summary")
    # Keep all compact source diagnostics, not just selected examples.
    for name, frame in frames.items():
        save(frame, name)
    save(audits, "audits")
    raw, transfer, coverage, replay = load_matched(args.input)
    save(raw, "matched_raw_interactions")
    save(coverage, "matched_coverage")
    save(replay, "replay_audits")
    summary = interaction_seed_summary(raw)
    identity = ["loss", "optimizer", "width", "seed", "target_loss", "fraction", "view"]
    summary = complete_tuned(summary, identity)
    transfer = complete_tuned(transfer, ["loss", "optimizer", "width", "seed", "target_loss"])
    save(summary, "matched_interactions_per_seed")
    save(transfer, "matched_transfers_per_seed")
    m_metrics = ["mean_progress", "progress_per_gain", "harm_per_gain", "net_all_per_gain",
                 "pair_plus_per_gain", "pair_minus_per_gain", "strict_frequency", "strict_severity",
                 "strict_harm_per_gain", "harmful_neuron_frequency", "pair_remainder_per_gain",
                 "mean_strict_events", "mean_strict_harm", "mean_eligible_pairs"]
    m_keys = ["collection", "loss", "optimizer", "width", "target_loss", "fraction", "view"]
    save(summarize(summary, m_keys, m_metrics), "matched_interactions_summary")
    transfer_metrics = ["harmful_frequency", "harmful_gain", "helpful_gain", "net_first_order_gain",
                        "p_step", "n_step", "raw_harmful_frequency", "raw_p", "raw_n",
                        "gradient_energy", "direction_energy", "finite_probe_gain",
                        "finite_probe_prediction"]
    t_keys = ["collection", "loss", "optimizer", "width", "target_loss"]
    save(summarize(transfer, t_keys, transfer_metrics), "matched_transfers_summary")
    contrasts = paired(summary, [k for k in m_keys if k != "width"], m_metrics)
    save(contrasts, "matched_interactions_paired_per_seed")
    save(summarize(contrasts, [k for k in m_keys if k != "width"] + ["width", "metric"],
                   ["difference", "relative_change"]),
         "matched_interactions_paired_summary")
    contrasts = paired(transfer, [k for k in t_keys if k != "width"], transfer_metrics)
    save(contrasts, "matched_transfers_paired_per_seed")
    save(summarize(contrasts, [k for k in t_keys if k != "width"] + ["width", "metric"],
                   ["difference", "relative_change"]),
         "matched_transfers_paired_summary")
    figures(frames["epochs"], frames["budgets"], finals, summary, transfer)
    full_objective_tables()
    audit = dict(training_runs=200, tuning_runs=96, matched_runs=len(replay),
                 matched_states=int(replay.accepted.sum()), matched_targets=int(replay.targets.sum()),
                 training_updates=int(finals[finals.collection != "width_tuned_complete"].completed.sum()),
                 max_mse_pair_identity_error=float(audits.max_pair_identity_error.max()),
                 max_budget_error=float(audits.max_budget_error.max()),
                 max_logit_additivity_error=float(audits.max_logit_additivity_error.max()),
                 max_replay_error=float(replay.replay_max_error.max()),
                 dense_replay_runs=120,
                 full_objective_states=119, full_objective_sources=8,
                 matched_sources=16, matched_batch_probes=64, seeds=5,
                 max_matching_relative_error=float(coverage.relative_loss_error.abs().max()))
    (TABLES / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(final_summary[(final_summary.collection == "main") & final_summary.metric.isin([
        "train_loss", "test_accuracy", "pair_plus_per_gain", "harm_per_gain"])]
          .pivot(index=["loss", "optimizer", "width"], columns="metric", values="mean").to_string())


if __name__ == "__main__":
    main()
