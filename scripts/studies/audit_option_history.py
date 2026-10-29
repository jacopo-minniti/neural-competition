"""Read-only recomputation of selected historical claims from raw saved records."""

import json
from pathlib import Path

import numpy as np
from analyze_option_mismatch import features, fit_predict

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/theory"


def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def main():
    f = read(D / "option_mismatch_tracking/forecasts.jsonl")
    o = read(D / "option_mismatch_tracking/outcomes.jsonl")
    by = {(r["seed"], r["step"]): r for r in f}
    end = [r for r in f if r["step"] == 800]
    result = {
        "mnist_final": {
            side: {
                k: float(np.mean([r[side][k] for r in end]))
                for k in ["full_loss", "validation_loss", "breadth", "mass"]
            }
            for side in ["narrow", "wide"]
        }
    }
    base = [
        "time",
        "full_loss_gap",
        "full_loss_mean",
        "gradient_norm_squared_gap",
        "gradient_norm_squared_mean",
    ]
    extra = [
        "value_gap",
        "alignment_gap",
        "residual_strength_percentile_gap",
        "breadth_gap",
        "mass_gap",
    ]
    o = [r for r in o if r["start"] >= 100]
    z = [features(by[r["seed"], r["start"]]) for r in o]
    y = np.array([r["gain_difference"] for r in o])
    cal = np.array([r["seed"] < 281805 for r in o])
    result["late_forecast"] = {}
    for name, fields in [("baseline", base), ("options", base + extra)]:
        x = np.array([[r[k] for k in fields] for r in z])
        p = fit_predict(x, y, cal, ~cal)
        result["late_forecast"][name] = float(np.sqrt(np.mean((y[~cal] - p) ** 2)))
    result["weakest_half"] = {}
    for m in [32, 128]:
        a = []
        for seed in range(281801, 281809):
            with np.load(D / f"option_mismatch_tracking/spectra_{seed}.npz") as ar:
                load = ar[f"{m}_800_residual_loading"]
                a.append(float(load[: len(load) // 2].sum() / load.sum()))
        result["weakest_half"][str(m)] = float(np.mean(a))
    result["common_demand_ratio"] = float(
        np.mean([r["cross"]["excess_value_wide"] / r["cross"]["excess_value_narrow"] for r in end])
    )
    result["tangent_fresh"] = {}
    fr = read(D / "tangent_sgd_fresh/forecasts.jsonl")
    out = {(r["seed"], r["start"]): r for r in read(D / "tangent_sgd_fresh/outcomes.jsonl")}
    fr = [r for r in fr if r["start"] >= 100]
    for name in ["conditional", "independent"]:
        result["tangent_fresh"][name] = float(
            np.sqrt(
                np.mean([(r[name] - out[r["seed"], r["start"]]["difference"]) ** 2 for r in fr])
            )
        )
    retention = json.loads((D / "option_retention/results.json").read_text())["results"]
    result["retention"] = {
        f"{m}_{arm}": float(
            np.mean(
                [
                    r["train_loss"]
                    for r in retention
                    if r["width"] == m and r["arm"] == arm and r["step"] == 8000
                ]
            )
        )
        for m in [32, 128]
        for arm in ["ordinary", "frozen_tangent"]
    }
    trans = json.loads((D / "option_transport/results.json").read_text())["results"]
    result["transport_initial"] = {
        str(m): float(
            np.mean(
                [
                    a["gate_crossing_fraction"]
                    for r in trans
                    if r["width"] == m and r["step"] == 0
                    for a in r["audits"]
                    if a["h"] == 0.05
                ]
            )
        )
        for m in [32, 128]
    }
    pred = np.load(D / "polarization_width_predictions.npz")
    val = np.load(D / "polarization_width_validation.npz")
    result["polarization"] = {}
    for case in ["base", "near", "rare", "clusters"]:
        result["polarization"][case] = {}
        for m in [4, 6, 8, 12]:
            key = f"{case}_m{m}"
            p, a = pred[key + "_predicted"], val[key + "_loss"]
            result["polarization"][case][str(m)] = dict(
                predicted_mean=float(p.mean()),
                actual_mean=float(a.mean()),
                false_failure=int(np.sum((p > 1e-8) & (a < 1e-8))),
                missed_failure=int(np.sum((p < 1e-8) & (a > 1e-8))),
            )
    output = D / "option_transport_resilience/history_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
