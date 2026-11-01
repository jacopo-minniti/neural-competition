"""Aggregate completion-reserve option statistics; no SGD trajectory forecasts."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def auc(labels, score):
    positive, negative = score[labels], score[~labels]
    if not len(positive) or not len(negative):
        return None
    return float(np.mean((positive[:, None] > negative) + .5 * (positive[:, None] == negative)))


def interval(values, rng):
    values = np.asarray(values)
    draws = values[rng.integers(len(values), size=(4000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), low=float(np.quantile(draws, .025)),
                high=float(np.quantile(draws, .975)), clusters=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/reserve_bridge")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(200731)
    result = dict(target="aggregate half-response-loss probability over natural completion phases",
                  uncertainty="seed-bank bootstrap; factors and phases are dependent", cohorts={})
    for name in ["development", "confirmation_v2", "step_reference", "half_step", "association_transfer"]:
        folder = root / name
        if not (folder / "results.json").exists():
            continue
        raw = json.loads((folder / "results.json").read_text())
        rows = raw["rows"]
        summaries = []
        seed_rates = {}
        for setting in raw["protocol"]["settings_spec"]:
            for width in [4, 8, 16]:
                selected = [r for r in rows if r["setting"] == setting["name"] and r["width"] == width]
                for phase in [None, 0, 1]:
                    subset = [r for r in selected if phase is None or r["phase"] == phase]
                    for audit in ["prototype", "factor"]:
                        initial = np.array([r["initial"][audit + "_response"] for r in subset])
                        observed = np.array([r["actual"][audit + "_response"] for r in subset])
                        predicted = np.array([r["predicted" if audit == "prototype" else "predicted_factor"]
                                              for r in subset])
                        event, forecast = observed < .5 * initial, predicted < .5 * initial
                        ratio = predicted / np.maximum(initial, 1e-300)
                        seeds = sorted({r["seed"] for r in subset})
                        seed_rate = [event[[i for i, r in enumerate(subset) if r["seed"] == seed]].mean()
                                     for seed in seeds]
                        seed_pred = [forecast[[i for i, r in enumerate(subset) if r["seed"] == seed]].mean()
                                     for seed in seeds]
                        q = np.array(subset[0]["q"])
                        weighted_event = event @ q / q.sum()
                        weighted_forecast = forecast @ q / q.sum()
                        seed_weighted = [np.mean([v for v, r in zip(weighted_event, subset)
                                                   if r["seed"] == seed]) for seed in seeds]
                        seed_weighted_pred = [np.mean([v for v, r in zip(weighted_forecast, subset)
                                                        if r["seed"] == seed]) for seed in seeds]
                        if phase is None:
                            seed_rates[setting["name"], width, audit] = np.array(seed_rate)
                        summaries.append(dict(setting=setting["name"], width=width, phase=phase,
                            audit=audit, actual_events=int(event.sum()), predicted_events=int(forecast.sum()),
                            trials=event.size, observed=interval(seed_rate, rng), predicted=interval(seed_pred, rng),
                            observed_qweighted=interval(seed_weighted, rng),
                            predicted_qweighted=interval(seed_weighted_pred, rng),
                            completion_auc=auc(event, 1 - ratio), current_auc=auc(event, -initial),
                            precision=float(np.sum(event & forecast) / max(forecast.sum(), 1)),
                            recall=float(np.sum(event & forecast) / max(event.sum(), 1)),
                            integrated_other_to_head=float(np.mean([
                                r["other_to_head_update_length"] for r in subset])),
                            initial_other_to_head=float(np.mean([r["initial_other_to_head"]
                                for r in subset])) if "initial_other_to_head" in subset[0] else None,
                            mean_end_step=float(np.mean([r["end_step"] for r in subset])),
                            mean_phase_loss=float(np.mean([r["actual"]["loss"] for r in subset]))))
        differences = [dict(setting=setting["name"], audit=audit,
            width4_minus_width16=interval(seed_rates[setting["name"], 4, audit]
                                         - seed_rates[setting["name"], 16, audit], rng))
            for setting in raw["protocol"]["settings_spec"] for audit in ["prototype", "factor"]]
        result["cohorts"][name] = dict(protocol=raw["protocol"], summary=summaries,
                                      paired_width_differences=differences)
        archive = out / name
        archive.mkdir(parents=True, exist_ok=True)
        for filename in ["protocol.json", "predictions.jsonl", "results.json"]:
            if filename.endswith(".jsonl") or filename == "results.json":
                with gzip.open(archive / (filename + ".gz"), "wb") as handle:
                    handle.write((folder / filename).read_bytes())
            else:
                shutil.copy2(folder / filename, archive / filename)
        shutil.copytree(folder / "sources", archive / "sources", dirs_exist_ok=True)
    reference_file, half_file = root / "step_reference/results.json", root / "half_step/results.json"
    if reference_file.exists() and half_file.exists():
        full = json.loads(reference_file.read_text())["rows"]
        half = json.loads(half_file.read_text())["rows"]
        keyed = {tuple(r[k] for k in ["setting", "seed", "width", "phase"]): r for r in half}
        disagreements, observations, errors = 0, 0, []
        for a in full:
            b = keyed[tuple(a[k] for k in ["setting", "seed", "width", "phase"])]
            ea = np.array(a["actual"]["prototype_response"]) < .5 * np.array(a["initial"]["prototype_response"])
            eb = np.array(b["actual"]["prototype_response"]) < .5 * np.array(b["initial"]["prototype_response"])
            disagreements += int(np.sum(ea != eb))
            observations += len(ea)
            errors.extend(abs(np.array(a["actual"]["prototype_response"])
                              - np.array(b["actual"]["prototype_response"])))
        result["step_sensitivity"] = dict(disagreements=disagreements, observations=observations,
            mean_absolute_response_error=float(np.mean(errors)), maximum_response_error=float(max(errors)))
    # Use the existing parity dataset and checkpoints only to audit the removed premises.
    parity_folder = Path("runs/parity_pressure/confirmation")
    if (parity_folder / "checkpoints.npz").exists():
        from reserve_bridge import forward, velocity
        from run_reserve_bridge import metric_norm

        from neural_competition.parity import panels

        protocol = json.loads((parity_folder / "protocol.json").read_text())
        population, _ = panels(protocol["q"])
        x = np.column_stack([population["x"], np.ones(len(population["x"]))])
        head = np.flatnonzero(population["x"].sum(1) == 0)[0]
        head_weight = np.zeros(len(x))
        head_weight[head] = population["p"][head]
        checkpoints = np.load(parity_folder / "checkpoints.npz")
        diagnostic = dict(p_empty=float(head_weight[head]), empty_target=float(population["y"][head]),
                          empty_dot_support_min=float(np.min(x @ x[head])),
                          empty_dot_support_max=float(np.max(x @ x[head])), widths={})
        for width in [8, 16, 32]:
            state = tuple(checkpoints[f"0_{width}_{key}"] for key in ["w", "a", "c"])
            e = population["y"] - forward(state, x)
            head_update = velocity(state, x, population["y"], head_weight)
            full_update = velocity(state, x, population["y"], population["p"])
            diagnostic["widths"][width] = dict(
                median_head_to_full_norm=float(np.median(metric_norm(head_update, width)
                                                        / metric_norm(full_update, width))),
                median_absolute_head_residual=float(np.median(abs(e[:, head]))))
        result["parity_premises"] = diagnostic
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copy2(Path(__file__), out / "analyze_reserve_bridge.py")
    confirmation = result["cohorts"]["confirmation_v2"]["summary"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    rows = [r for r in confirmation if r["setting"] == "cooccurrence"
            and r["phase"] is None and r["audit"] == "prototype"]
    for key, label, color in [("observed", "Observed", "#24689b"),
                              ("predicted", "Unchanged reserve prediction", "#bd602a")]:
        axes[0].plot([r["width"] for r in rows], [100 * r[key]["mean"] for r in rows],
                     marker="o", label=label, color=color)
        axes[0].fill_between([r["width"] for r in rows], [100 * r[key]["low"] for r in rows],
                              [100 * r[key]["high"] for r in rows], alpha=.14, color=color)
    axes[0].set(title="First relaxation: co-occurring factors", xlabel="Width",
                ylabel="Half-response losses (%)", xticks=[4, 8, 16])
    axes[0].legend(frameon=False, fontsize=8)
    names = ["cooccurrence", "more_other_pressure", "boundary_30", "boundary_60"]
    rows = [next(r for r in confirmation if r["setting"] == name and r["width"] == 4
                 and r["phase"] is None and r["audit"] == "prototype") for name in names]
    masses = [.04, .15, .3, .6]
    for key, label, color in [("observed", "Observed", "#24689b"),
                              ("predicted", "Unchanged reserve prediction", "#bd602a")]:
        axes[1].plot(masses, [100 * r[key]["mean"] for r in rows], marker="o", color=color, label=label)
    axes[1].set(title="Boundary: stronger competing pressure", xlabel="Expected active factor count",
                ylabel="Half-response losses at width 4 (%)")
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(out / f"first_relaxation.{suffix}", dpi=170)
    plt.close(fig)
    print(json.dumps(dict(cohorts=list(result["cohorts"]))))


if __name__ == "__main__":
    main()
