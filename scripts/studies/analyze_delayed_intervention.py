"""Original-loss outcomes of the prospective response-matched timing intervention."""

import json
from pathlib import Path

import numpy as np
from analyze_option_transport_resilience import read, summary

ROOT = Path(__file__).resolve().parents[2] / "docs/theory/option_transport_resilience"


def main():
    folder = ROOT / "delayed_intervention"
    rows = read(folder / "trajectory.jsonl")
    ix = {(r["case"], r["width"], r["arm"], r["time"], r["seed"]): r for r in rows}
    result = {}
    for case in sorted({r["case"] for r in rows}):
        result[case] = {}
        for m in [4, 8]:
            result[case][str(m)] = {}
            for arm in sorted({r["arm"] for r in rows}):
                rr = [
                    r
                    for r in rows
                    if r["case"] == case
                    and r["width"] == m
                    and r["arm"] == arm
                    and r["time"] == 2000
                ]
                baseline = np.array([ix[case, m, "ordinary", 2000, r["seed"]]["loss"] for r in rr])
                loss = np.array([r["loss"] for r in rr])
                result[case][str(m)][arm] = dict(
                    loss=summary(loss),
                    dense_loss=summary([r["dense_loss"] for r in rr]),
                    loss_change=summary(loss - baseline),
                    rescued=int(np.sum((baseline > 1e-7) & (loss < 1e-7))),
                    induced=int(np.sum((baseline < 1e-7) & (loss > 1e-7))),
                )
    matching = read(folder / "matching.jsonl")
    result["matching"] = {
        k: max(r[k] for r in matching) for k in ["function_error", "kernel_error"]
    }
    result["clone_max_loss_difference"] = max(
        abs(r["loss"] - ix[r["case"], r["width"], "ordinary", r["time"], r["seed"]]["loss"])
        for r in rows
        if r["arm"] == "clone_at_2"
    )
    (ROOT / "intervention_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
