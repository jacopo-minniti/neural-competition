"""Halve deterministic intervention steps at the same pre-existing checkpoints."""

import json
from pathlib import Path

import numpy as np
from power_law_tasks import dataset, initialize, response, sgd_step, specification
from probe_power_law_pressure import pressure_step


def main():
    folder = Path(__file__).resolve().parents[2] / "docs/evidence/power_law_competition/pressure"
    previous = json.loads((folder / "results.json").read_text())
    rows = []
    for row in previous["records"]:
        bank = previous["protocol"]["seed_bank"] + row["seed"]
        train, _, audit, _ = dataset(specification(richer=row["case"] == "associated"), bank + 10000)
        state = initialize(row["width"], train["x"].shape[1], row["initialization"], bank)
        rng = np.random.default_rng(bank + 20000)
        for _ in range(previous["protocol"]["checkpoint"]):
            state = sgd_step(state, train, .025, 32, rng)
        coarse = np.load(folder / f"{row['key']}_actual.npz")
        initial = response(state, audit)
        for arm in ["common", "refinement", "full"]:
            fine = tuple(x.copy() for x in state)
            for _ in range(32):
                fine = pressure_step(fine, train, arm, .0125)
            effect = response(fine, audit) - initial
            old = coarse[arm] - initial
            rows.append(dict(case=row["case"], arm=arm, key=row["key"],
                             coarse=old.tolist(), fine=effect.tolist()))
    summary = []
    for case in ["prototype", "associated"]:
        for arm in ["common", "refinement", "full"]:
            group = [r for r in rows if (r["case"], r["arm"]) == (case, arm)]
            old, fine = [np.array([r[k] for r in group]) for k in ["coarse", "fine"]]
            summary.append(dict(case=case, arm=arm, coarse_mean=float(old.mean()),
                                fine_mean=float(fine.mean()),
                                absolute_rmse=float(np.sqrt(np.mean((old - fine)**2)))))
    (folder / "step_sensitivity.json").write_text(json.dumps(dict(
        status="Follow-up numerical audit after coarse intervention results; no model fitting",
        coarse_step=.025, fine_step=.0125, exposure_time=.4, summary=summary, records=rows), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
