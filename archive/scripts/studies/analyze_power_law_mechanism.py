"""Width contrasts, reversible losses, and final-risk uncertainty for the main cohort."""

import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[2] / "docs/evidence/power_law_competition"
    folder = root / "centered_confirmation"
    states = [json.loads(x) for x in (folder / "states.jsonl").read_text().splitlines()]
    rows, recovery = [], []
    for state in states:
        f = np.load(folder / f"{state['key']}_forecast.npz")
        a = np.load(folder / f"{state['key']}_actual.npz")
        initial = f["initial_strength"]
        observed = (a["h16"] < .9 * initial).mean(0)
        predicted = (f["bilinear_h16"] < .9 * initial).mean(0)
        rows.append(dict(**state, observed=float(observed.mean()), predicted=float(predicted.mean())))
        closed4 = (initial >= .1) & (a["h4"] < .1)
        restored = closed4 & (a["h16"] >= .1)
        recovery.append(dict(case=state["case"], seed=state["seed"],
                             closed4=int(closed4.sum()), reopened16=int(restored.sum())))
    rng = np.random.default_rng(414892)
    contrasts = []
    for case in ["prototype", "associated"]:
        for step in [0, 200, 2000]:
            group = [r for r in rows if r["case"] == case and r["step"] == step]
            means = {}
            for width in [50, 100]:
                wr = [r for r in group if r["width"] == width]
                means[str(width)] = {k: float(np.mean([r[k] for r in wr]))
                                     for k in ["observed", "predicted", "validation_loss", "mean_diversity"]}
            diffs = []
            for seed in range(6):
                sr = [r for r in group if r["seed"] == seed]
                diffs.append(np.mean([r["observed"] for r in sr if r["width"] == 100])
                             - np.mean([r["observed"] for r in sr if r["width"] == 50]))
            draws = np.array(diffs)[rng.integers(6, size=(5000, 6))].mean(1)
            contrasts.append(dict(case=case, step=step, widths=means,
                                  wide_minus_narrow=float(np.mean(diffs)),
                                  seed_bootstrap95=np.quantile(draws, [.025, .975]).tolist()))
    reversibility = []
    for case in ["prototype", "associated"]:
        group = [r for r in recovery if r["case"] == case]
        closed = sum(r["closed4"] for r in group)
        reopened = sum(r["reopened16"] for r in group)
        reversibility.append(dict(case=case, closed_by_step4=closed, reopened_by_step16=reopened,
                                  conditional_reopening=reopened / closed))
    result = dict(width_contrasts=contrasts, reversibility=reversibility,
                  caution="Six seed clusters, not independent factors or fork outcomes; checkpoint comparisons are not causal mediation.")
    (root / "mechanism_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
