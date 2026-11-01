"""Summarize frequency-pressure studies with seed-clustered uncertainty."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def interval(values, rng):
    values = np.asarray(values)
    means = values[rng.integers(len(values), size=(4000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), low=float(np.quantile(means, .025)),
                high=float(np.quantile(means, .975)), clusters=len(values))


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def diagnostic_summary(records, rng):
    rows = []
    for record in records:
        width, step = record["width"], record["step"]
        t = np.array(record["transport"])
        singles = [1 << i for i in range(8)]
        cross = t[1:9, singles][~np.eye(8, dtype=bool)]
        negshare = np.maximum(-cross, 0).sum() / max(np.abs(cross).sum(), 1e-300)
        row = dict(width=width, step=step, seed=record["seed"], loss=record["loss"],
                   negative_cross_fraction=float(np.mean(cross < 0)),
                   negative_cross_share=float(negshare),
                   modal_cancellation=record["modal_cancellation"],
                   factor_cancellation=record["factor_cancellation"],
                   fixed_interaction_response=record["strength"][256],
                   moving_demand_response=record["strength"][-1])
        if "gate_flip_mass" in record:
            row["gate_flip_mass"] = record["gate_flip_mass"]
        for name in ["finite_loss", "finite_rate"]:
            if name in record:
                row[name] = record[name]
        rows.append(row)
    summary = []
    for step in sorted({r["step"] for r in rows}):
        for width in sorted({r["width"] for r in rows}):
            selected = [r for r in rows if r["step"] == step and r["width"] == width]
            if not selected:
                continue
            summary.append(dict(step=step, width=width, **{
                key: interval([r[key] for r in selected], rng) for key in [
                    "negative_cross_fraction", "negative_cross_share", "modal_cancellation",
                    "factor_cancellation", "fixed_interaction_response", "moving_demand_response"]}))
    return rows, summary


def forecast_summary(folder, rng):
    predictions = read_lines(folder / "predictions.jsonl")
    actual = read_lines(folder / "actual.jsonl")
    rows, summaries = [], []
    for prediction, observation in zip(predictions, actual):
        assert all(prediction[k] == observation[k] for k in ["step", "seed", "width"])
        initial_rate = prediction["initial"]["fixed_rate"][0]
        for horizon, forecast in prediction["predictions"].items():
            measured = observation["actual"][horizon]
            samples = np.array(measured["loss"])
            predicted_samples = np.array(forecast["gaussian"]["loss"])
            actual_probability = float(np.mean(np.array(measured["fixed_rate"]) < .9 * initial_rate))
            predicted_probability = float(np.mean(
                np.array(forecast["gaussian"]["fixed_rate"]) < .9 * initial_rate))
            rows.append(dict(step=prediction["step"], seed=prediction["seed"],
                             width=prediction["width"], horizon=int(horizon),
                             initial_loss=prediction["initial"]["loss"],
                             actual_loss=float(samples.mean()),
                             actual_mean_se=float(samples.std(ddof=1) / np.sqrt(len(samples))),
                             predicted_mean_se=float(predicted_samples.std(ddof=1)
                                                     / np.sqrt(len(predicted_samples))),
                             gaussian=float(predicted_samples.mean()),
                             mean_gradient=forecast["mean_gradient"]["loss"],
                             actual_weakening=actual_probability,
                             predicted_weakening=predicted_probability,
                             brier=actual_probability * (1 - predicted_probability)**2
                             + (1 - actual_probability) * predicted_probability**2))
    for horizon in sorted({r["horizon"] for r in rows}):
        selected = [r for r in rows if r["horizon"] == horizon]
        errors = {name: np.array([r[name] - r["actual_loss"] for r in selected])
                  for name in ["initial_loss", "mean_gradient", "gaussian"]}
        seeds = sorted({r["seed"] for r in selected})
        gaussian_mse = [np.mean([e**2 for e, r in zip(errors["gaussian"], selected)
                                 if r["seed"] == seed]) for seed in seeds]
        mse_ci = interval(gaussian_mse, rng)
        summaries.append(dict(horizon=horizon, states=len(selected), original_seeds=len(seeds),
                              rmse={k: float(np.sqrt(np.mean(e**2))) for k, e in errors.items()},
                              gaussian_rmse_seed_ci={k: float(np.sqrt(mse_ci[k]))
                                                     for k in ["mean", "low", "high"]},
                              gaussian_bias=float(errors["gaussian"].mean()),
                              actual_mc_se_rms=float(np.sqrt(np.mean([
                                  r["actual_mean_se"]**2 for r in selected]))),
                              actual_weakening=float(np.mean([
                                  r["actual_weakening"] for r in selected])),
                              predicted_weakening=float(np.mean([
                                  r["predicted_weakening"] for r in selected])),
                              weakening_calibration_mae=float(np.mean([
                                  abs(r["actual_weakening"] - r["predicted_weakening"])
                                  for r in selected])),
                              brier=float(np.mean([r["brier"] for r in selected]))))
    return rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/parity_pressure")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1507100)
    result = dict(uncertainty="4000 seed-cluster bootstrap draws; particles and checkpoints are not clusters",
                  training=[], diagnostics={}, forecasts={}, finite_attribution={},
                  matched_progress=[], time_to_loss=[], smooth_forecasts={})
    run_names = ["development_v2", "confirmation", "centered", "flatter", "geometric",
                 "uniform_activity", "uniform_label", "uniform_init", "small_init", "balanced",
                 "half_rate", "long_budget"]
    for name in run_names:
        folder = root / name
        if not (folder / "curves.npz").exists():
            continue
        protocol = json.loads((folder / "protocol.json").read_text())
        curves = np.load(folder / "curves.npz")
        archive = out / "runs" / name
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / "protocol.json", archive / "protocol.json")
        if name in ["development_v2", "confirmation"]:
            shutil.copy2(folder / "curves.npz", archive / "curves.npz")
        else:
            keep = np.isin(curves["steps"], [0, 100, 500, 1000, 2000, 4000, 6000,
                                            10000, 20000, 40000, 50000, 100000])
            np.savez_compressed(archive / "curves.npz", **{k: curves[k][keep] for k in curves.files})
        shutil.copytree(folder / "sources", archive / "sources", dirs_exist_ok=True)
        for step in [10000, 20000, 40000, 100000]:
            index = np.flatnonzero(curves["steps"] == step)
            if not len(index):
                continue
            loss = curves["loss"][index[0]]
            gain = loss[0] - loss[-1]
            result["training"].append(dict(run=name, step=step, widths=protocol["widths"],
                                           loss_mean=loss.mean(1).tolist(),
                                           audit_loss_mean=curves["audit_loss"][index[0]].mean(1).tolist(),
                                           gain=interval(gain, rng), wide_wins=int(np.sum(gain > 0)),
                                           seeds=len(gain)))
        records = json.loads((folder / "diagnostics.json").read_text())
        if records:
            rows, summary = diagnostic_summary(records, rng)
            result["diagnostics"][name] = dict(rows=rows, summary=summary)
            for threshold in [.08, .04, .02]:
                for width in protocol["widths"]:
                    values, times = [], []
                    for seed in sorted({r["seed"] for r in rows}):
                        series = [r for r in rows if r["width"] == width and r["seed"] == seed]
                        for before, after in zip(series, series[1:]):
                            if before["loss"] >= threshold >= after["loss"]:
                                fraction = (before["loss"] - threshold) / (before["loss"] - after["loss"])
                                values.append((1 - fraction) * before["negative_cross_share"]
                                              + fraction * after["negative_cross_share"])
                                times.append((1 - fraction) * before["step"] + fraction * after["step"])
                                break
                    if values:
                        result["matched_progress"].append(dict(run=name, threshold=threshold,
                            width=width, reached=len(values), total=protocol["diagnostic_seeds"],
                            interpolated=True, adverse_share=interval(values, rng),
                            mean_interpolated_step=float(np.mean(times))))
        if name == "confirmation":
            for threshold in [.10, .08, .04, .02]:
                for i, width in enumerate(protocol["widths"]):
                    hits = []
                    for seed in range(protocol["seeds"]):
                        indices = np.flatnonzero(curves["loss"][:, i, seed] <= threshold)
                        if len(indices):
                            hits.append(int(curves["steps"][indices[0]]))
                    result["time_to_loss"].append(dict(threshold=threshold, width=width,
                        reached=len(hits), total=protocol["seeds"],
                        conditional_median_step=float(np.median(hits)) if hits else None))
        if name in ["development_v2", "confirmation"]:
            loc = {int(s): i for i, s in enumerate(curves["steps"])}
            forecasts = [r for r in read_lines(folder / "forecasts.jsonl") if r["step"] >= 2000]
            smooth_summary = []
            for horizon in protocol["horizons"]:
                actual = np.array([curves["loss"][loc[r["step"] + horizon],
                    protocol["widths"].index(r["width"]), r["seed"]] for r in forecasts])
                metrics = {}
                for key in ["persistence", "first_order", "gauss_newton", "transported"]:
                    predicted = np.array([r["forecasts"][str(horizon)][key] for r in forecasts])
                    metrics[key] = dict(rmse=float(np.sqrt(np.mean((predicted - actual)**2))),
                                        negative_loss_predictions=int(np.sum(predicted < 0)))
                smooth_summary.append(dict(horizon=horizon, metrics=metrics))
            result["smooth_forecasts"][name] = smooth_summary
    forecast_names = ["development_forecasts", "confirmation_forecasts", "geometric_forecasts",
                      "uniform_forecasts"]
    for name in forecast_names:
        folder = root / name
        if not (folder / "actual.jsonl").exists():
            continue
        rows, summary = forecast_summary(folder, rng)
        result["forecasts"][name] = dict(rows=rows, summary=summary)
        archive = out / "runs" / name
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / "protocol.json", archive / "protocol.json")
        shutil.copytree(folder / "sources", archive / "sources", dirs_exist_ok=True)
        for filename in ["predictions.jsonl", "actual.jsonl"]:
            compact = read_lines(folder / filename)
            for record in compact:
                if filename == "predictions.jsonl":
                    for predictions in record["predictions"].values():
                        for method, values in predictions.items():
                            predictions[method] = {k: v for k, v in values.items()
                                                   if k in {"loss", "fixed_rate"}}
                else:
                    for horizon, values in record["actual"].items():
                        record["actual"][horizon] = {k: v for k, v in values.items()
                                                     if k in {"loss", "fixed_rate"}}
            with gzip.open(archive / (filename + ".gz"), "wb") as handle:
                handle.write(("\n".join(json.dumps(r) for r in compact) + "\n").encode())
    finite = root / "finite_audit"
    if (finite / "results.jsonl").exists():
        compact = []
        for record in read_lines(finite / "results.jsonl"):
            mean = np.mean(record["transport"], axis=0)
            cross = mean[1:9, :8][~np.eye(8, dtype=bool)]
            demanded = mean[1:9, -1]
            compact.append(dict(step=record["step"], width=record["width"], seed=record["seed"],
                                negative_cross_share=float(np.maximum(-cross, 0).sum()
                                                           / np.abs(cross).sum()),
                                negative_cross_fraction=float(np.mean(cross < 0)),
                                opposed_crossing_fraction=float(np.mean(
                                    record["opposed_crossing_fraction"])),
                                current_demand_adverse_share=float(np.maximum(-demanded, 0).sum()
                                    / max(np.abs(demanded).sum(), 1e-300)),
                                opposed_current_demand_sources=bool(
                                    np.any(demanded < 0) and np.any(demanded > 0)),
                                max_identity_error=record["max_identity_error"],
                                mean_transport=mean.tolist(),
                                mean_loss_terms={k: float(np.mean(v))
                                                 for k, v in record["loss_terms"].items()}))
        summary = []
        for step in sorted({r["step"] for r in compact}):
            for width in [8, 32]:
                selected = [r for r in compact if r["step"] == step and r["width"] == width]
                summary.append(dict(step=step, width=width, **{
                    k: interval([r[k] for r in selected], rng) for k in [
                        "negative_cross_share", "negative_cross_fraction", "opposed_crossing_fraction",
                        "current_demand_adverse_share", "opposed_current_demand_sources"]}))
        result["finite_attribution"] = dict(rows=compact, summary=summary)
        archive = out / "runs" / "finite_audit"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(finite / "protocol.json", archive / "protocol.json")
        shutil.copytree(finite / "sources", archive / "sources", dirs_exist_ok=True)
    # Save the failed smooth local forecasts, including their prospective values.
    for name in ["development_v2", "confirmation"]:
        source = root / name / "forecasts.jsonl"
        with gzip.open(out / "runs" / name / "smooth_forecasts.jsonl.gz", "wb") as handle:
            handle.write(source.read_bytes())
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copy2(Path(__file__), out / "analyze_frequency_pressure.py")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    curves = np.load(root / "confirmation" / "curves.npz")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    colors = ["#b44832", "#a07a13", "#24689b"]
    for i, (width, color) in enumerate(zip([8, 16, 32], colors)):
        for ax, key, title in [(axes[0, 0], "loss", "Population loss: ordinary SGD"),
                               (axes[0, 1], "audit_loss", "Uniform audit loss: different risk")]:
            values = curves[key][:, i]
            mean, se = values.mean(1), values.std(1, ddof=1) / np.sqrt(values.shape[1])
            ax.plot(curves["steps"], mean, label=f"width {width}", color=color)
            ax.fill_between(curves["steps"], np.maximum(mean - 1.96 * se, 1e-8),
                            mean + 1.96 * se, color=color, alpha=.13)
            ax.set(title=title, xlabel="SGD steps", ylabel="Squared loss / 2", yscale="log")
        summary = [r for r in result["diagnostics"]["confirmation"]["summary"] if r["width"] == width]
        times = np.array([r["step"] for r in summary])
        for ax, key, title in [
            (axes[1, 0], "fixed_interaction_response", "Fixed interaction-correction response"),
            (axes[1, 1], "negative_cross_share", "Within-gate adverse cross-pressure share")]:
            ax.plot(times, [r[key]["mean"] for r in summary], color=color, label=f"width {width}")
            ax.fill_between(times, [r[key]["low"] for r in summary],
                            [r[key]["high"] for r in summary], color=color, alpha=.13)
            ax.set(title=title, xlabel="SGD steps", ylabel="Response" if key.startswith("fixed") else "Share")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Fresh-seed parity study: fixed distribution, all parameters trained", fontsize=13)
    fig.text(.5, .01, "Bottom right is a smooth diagnostic; finite gate effects prevent interpreting it as loss mediation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    for suffix in ["png", "svg"]:
        fig.savefig(out / f"width_and_pressure.{suffix}", dpi=170)
    plt.close(fig)
    forecasts = result["forecasts"]["confirmation_forecasts"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    selected = [r for r in forecasts["rows"] if r["horizon"] == 1000]
    for width, color in [(8, colors[0]), (32, colors[2])]:
        rows = [r for r in selected if r["width"] == width]
        actual = np.array([r["initial_loss"] - r["actual_loss"] for r in rows])
        predicted = np.array([r["initial_loss"] - r["gaussian"] for r in rows])
        axes[0].scatter(actual, predicted, s=24, color=color, alpha=.8, label=f"width {width}")
        axes[1].scatter([r["actual_weakening"] for r in rows],
                        [r["predicted_weakening"] for r in rows], s=24, color=color, alpha=.8)
    for ax in axes[:2]:
        low, high = ax.get_xlim()
        ax.plot([low, high], [low, high], color=".5", linewidth=1)
    axes[0].set(title="Loss reduction over next 1,000 steps", xlabel="Observed conditional mean",
                ylabel="Predicted conditional mean")
    axes[0].legend(frameon=False)
    axes[1].set(title="10% fixed-demand response weakening", xlabel="Observed probability",
                ylabel="Predicted probability", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
    summary = forecasts["summary"]
    for key, label, color in [("initial_loss", "Persistence", ".5"),
                              ("mean_gradient", "Mean-gradient forecast", "#a07a13"),
                              ("gaussian", "Gate-aware Gaussian forecast", "#24689b")]:
        axes[2].plot([r["horizon"] for r in summary], [r["rmse"][key] for r in summary],
                     marker="o", color=color, label=label)
    axes[2].set(title="Conditional mean loss error", xlabel="Forecast horizon (steps)",
                ylabel="RMSE", xscale="log", yscale="log")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(out / f"conditional_forecasts.{suffix}", dpi=170)
    plt.close(fig)
    print(json.dumps(dict(training_comparisons=len(result["training"]),
                          forecast_cohorts=list(result["forecasts"]))))


if __name__ == "__main__":
    main()
