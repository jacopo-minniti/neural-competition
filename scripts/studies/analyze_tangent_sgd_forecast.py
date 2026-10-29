"""Compare tangent forecasts with the unchanged loss/gradient baseline."""

import json

import numpy as np
from analyze_option_mismatch import features, fit_predict, score
from explore_conditional_editability import ROOT

OLD = ROOT / "docs/theory/option_mismatch_tracking"
OUT = ROOT / "docs/theory/tangent_sgd_forecast"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    old_forecasts = {(r["seed"], r["step"]): r for r in read(OLD / "forecasts.jsonl")}
    outcomes = read(OLD / "outcomes.jsonl")
    fields = [
        "time",
        "full_loss_gap",
        "full_loss_mean",
        "gradient_norm_squared_gap",
        "gradient_norm_squared_mean",
    ]
    calibration = [r for r in outcomes if r["seed"] < 281805 and r["start"] >= 100]
    cx = np.array(
        [[features(old_forecasts[r["seed"], r["start"]])[f] for f in fields] for r in calibration]
    )
    cy = np.array([r["gain_difference"] for r in calibration])
    all_result = {}
    saved = []
    for directory, name in [
        (OUT, "previous_test_seeds"),
        (ROOT / "docs/theory/tangent_sgd_fresh", "fresh_test_seeds"),
    ]:
        forecasts = read(directory / "forecasts.jsonl")
        actual = {(r["seed"], r["start"]): r for r in read(directory / "outcomes.jsonl")}
        rows = [r for r in forecasts if r["seed"] >= 281805 and r["start"] >= 100]
        y = np.array([actual[r["seed"], r["start"]]["difference"] for r in rows])
        x = []
        for r in rows:
            if name == "previous_test_seeds":
                z = features(old_forecasts[r["seed"], r["start"]])
                x.append([z[f] for f in fields])
            else:
                x.append(
                    [
                        np.log1p(r["start"]),
                        r["loss_wide"] - r["loss_narrow"],
                        (r["loss_wide"] + r["loss_narrow"]) / 2,
                        r["gradient_wide"] - r["gradient_narrow"],
                        (r["gradient_wide"] + r["gradient_narrow"]) / 2,
                    ]
                )
        joined = np.concatenate([cx, np.array(x)])
        target = np.concatenate([cy, y])
        mask = np.arange(len(joined)) < len(cx)
        baseline = fit_predict(joined, target, mask, ~mask)
        predictions = {
            "zero": np.zeros(len(y)),
            "loss_gradient": baseline,
            "conditional_tangent": np.array([r["conditional"] for r in rows]),
            "independent_tangent": np.array([r["independent"] for r in rows]),
        }
        seeds = np.array([r["seed"] for r in rows])
        all_result[name] = dict(
            scores={p: score(y, v) for p, v in predictions.items()},
            per_seed={
                str(seed): {
                    p: score(y[seeds == seed], v[seeds == seed]) for p, v in predictions.items()
                }
                for seed in sorted(set(seeds))
            },
        )
        for i, r in enumerate(rows):
            saved.append(
                dict(
                    seed=r["seed"],
                    step=r["start"],
                    actual=float(y[i]),
                    **{p: float(v[i]) for p, v in predictions.items()},
                )
            )
    # Replay agrees with old uninterrupted trajectories, rather than using new outcomes silently.
    new = read(OUT / "outcomes.jsonl")
    old = {(r["seed"], r["start"]): r["gain_difference"] for r in outcomes}
    all_result["max_replay_discrepancy"] = max(
        abs(r["difference"] - old[r["seed"], r["start"]]) for r in new
    )
    all_result["caveat"] = (
        "Same 140 late intervals per test cohort; four seeds per cohort. Conditional forecasts know exogenous batch identities. MC forecasts do not. No coefficients fitted to test outcomes."
    )
    (OUT / "analysis.json").write_text(json.dumps(all_result, indent=2) + "\n")
    (OUT / "predictions.json").write_text(json.dumps(saved, indent=2) + "\n")
    print(json.dumps(all_result, indent=2))


if __name__ == "__main__":
    main()
