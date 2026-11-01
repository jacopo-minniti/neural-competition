"""Summarize held-out probability forecasts with dependence-aware seed resampling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def normal_cdf(x):
    return np.array([.5 * (1 + math.erf(v / np.sqrt(2))) for v in np.ravel(x)]).reshape(x.shape)


def summarize(records):
    actual = np.stack([r["actual"] for r in records])
    prediction = np.stack([r["prediction"] for r in records])
    initial = np.stack([r["initial"] for r in records])
    close = np.stack([r["close"] for r in records])
    pclose = np.stack([r["pclose"] for r in records])
    mse = np.mean((prediction - actual) ** 2)
    baseline = np.mean((initial - actual) ** 2)
    prevalence = float(close.mean())
    brier = np.mean(pclose**2 - 2 * pclose * close + close)
    oracle_constant_brier = prevalence * (1 - prevalence)
    return dict(states=len(records),
                mean_strength_rmse=float(np.sqrt(mse)),
                persistence_rmse=float(np.sqrt(baseline)),
                response_error_ratio=float(np.sqrt(mse / max(baseline, 1e-300))),
                observed_10pct_loss_probability=prevalence,
                predicted_10pct_loss_probability=float(pclose.mean()),
                brier=float(brier), constant_prevalence_brier=oracle_constant_brier,
                brier_skill=float(1 - brier / max(oracle_constant_brier, 1e-300)),
                expected_lost_count_mae=float(np.mean(np.abs(pclose.sum(1) - close.sum(1)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()
    folder = Path(args.input)
    states = [json.loads(line) for line in (folder / "states.jsonl").read_text().splitlines()]
    protocol = json.loads((folder / "protocol.json").read_text())
    first = np.load(folder / f"{states[0]['key']}_forecast.npz")
    methods = ["frozen", "block4", "smooth"]
    if "bilinear_h1" in first or f"bilinear_h{protocol['horizons'][0]}" in first:
        methods += ["linear", "bilinear"]
    primary = "bilinear" if "bilinear" in methods else "block4"
    rows, boundary, joint = [], [], []
    for state in states:
        forecast = np.load(folder / f"{state['key']}_forecast.npz")
        outcomes = np.load(folder / f"{state['key']}_actual.npz")
        initial = forecast["initial_strength"]
        for horizon in protocol["horizons"]:
            truth = outcomes[f"h{horizon}"]
            actual_close = (truth < .9 * initial).mean(0)
            labels = {"frozen": f"h{horizon}", "block4": f"block_h{horizon}", "smooth": None,
                      "linear": f"linear_h{horizon}", "bilinear": f"bilinear_h{horizon}"}
            for method in methods:
                label = labels[method]
                if label is None:
                    prediction = initial + horizon * protocol["rate"] * forecast["mean"]
                    std = np.sqrt(np.maximum(np.diag(forecast["covariance"]), 0)
                                  * horizon * protocol["rate"]**2 / protocol["batch"])
                    pclose = normal_cdf((.9 * initial - prediction) / np.maximum(std, 1e-30))
                else:
                    samples = forecast[label]
                    prediction = samples.mean(0)
                    pclose = (samples < .9 * initial).mean(0)
                    joint.append(dict(**state, horizon=horizon, method=method,
                                      observed_any=float(np.mean(np.any(truth < .9 * initial, axis=1))),
                                      predicted_any=float(np.mean(np.any(samples < .9 * initial, axis=1))),
                                      independent_any=float(1 - np.prod(1 - pclose)),
                                      observed_count_variance=float(np.var(np.sum(truth < .9 * initial, axis=1))),
                                      predicted_count_variance=float(np.var(np.sum(samples < .9 * initial, axis=1)))))
                    for resolution in protocol["availability_resolutions"]:
                        available = initial >= resolution
                        if np.any(available):
                            yp = np.mean(truth[:, available] < resolution, axis=0)
                            pp = np.mean(samples[:, available] < resolution, axis=0)
                            boundary.append(dict(**state, horizon=horizon, method=method,
                                                 resolution=resolution, at_risk=int(available.sum()),
                                                 observed=float(yp.sum()), predicted=float(pp.sum()),
                                                 brier_sum=float(np.sum(pp**2 - 2 * pp * yp + yp))))
                rows.append(dict(**state, horizon=horizon, method=method, actual=truth.mean(0),
                                 prediction=prediction, initial=initial, close=actual_close, pclose=pclose))
    grouped = []
    rng = np.random.default_rng(14061)
    for case in protocol["cases"]:
        for horizon in protocol["horizons"]:
            for method in methods:
                group = [r for r in rows if (r["case"], r["horizon"], r["method"]) == (case, horizon, method)]
                point = summarize(group)
                seed_groups = [[r for r in group if r["seed"] == seed] for seed in range(protocol["seeds"])]
                draws = []
                for _ in range(1000):
                    sampled = [row for index in rng.integers(len(seed_groups), size=len(seed_groups))
                               for row in seed_groups[index]]
                    draws.append(summarize(sampled)["response_error_ratio"])
                grouped.append(dict(case=case, horizon=horizon, method=method, **point,
                                    ratio_seed_bootstrap95=np.quantile(draws, [.025, .975]).tolist()))
    init_groups = []
    for case in protocol["cases"]:
        for kind in protocol["initializations"]:
            for horizon in protocol["horizons"]:
                group = [r for r in rows if (r["case"], r["initialization"], r["horizon"], r["method"])
                         == (case, kind, horizon, primary)]
                init_groups.append(dict(case=case, initialization=kind, horizon=horizon, **summarize(group)))
    availability = []
    for case in protocol["cases"]:
        for horizon in protocol["horizons"]:
            for resolution in protocol["availability_resolutions"]:
                group = [r for r in boundary if (r["case"], r["horizon"], r["resolution"], r["method"])
                         == (case, horizon, resolution, primary)]
                count = sum(r["at_risk"] for r in group)
                availability.append(dict(case=case, horizon=horizon, resolution=resolution,
                                         factor_states_at_risk=count,
                                         observed_probability=sum(r["observed"] for r in group) / count,
                                         predicted_probability=sum(r["predicted"] for r in group) / count,
                                         brier=sum(r["brier_sum"] for r in group) / count))
    natural = []
    for case in protocol["cases"]:
        for kind in protocol["initializations"]:
            for width in protocol["widths"]:
                group = [s for s in states if (s["case"], s["initialization"], s["width"], s["step"])
                         == (case, kind, width, max(protocol["checkpoints"]))]
                natural.append(dict(case=case, initialization=kind, width=width,
                                    **{key: float(np.mean([s[key] for s in group])) for key in [
                                        "validation_loss", "mean_strength", "tail_strength",
                                        "mean_diversity", "tail_diversity"]}))
    result = dict(protocol=protocol, unique_states=len(states), independent_seed_banks=protocol["seeds"],
                  forecast_scores=grouped, by_initialization=init_groups,
                  availability=availability, natural=natural,
                  primary_method=primary,
                  joint_event_summary={k: float(np.mean([r[k] for r in joint if r["method"] == primary]))
                                       for k in ["observed_any", "predicted_any", "independent_any",
                                                 "observed_count_variance", "predicted_count_variance"]})
    output = Path(args.output) if args.output else folder.parent / f"{folder.name}_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(scores=grouped, availability=availability, natural=natural), indent=2))


if __name__ == "__main__":
    main()
