"""Same-checkpoint common/refinement pressure interventions and prospective decoding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from power_law_tasks import dataset, forward, initialize, response, sgd_step, specification


def pressure_step(state, train, component, rate, reference=None):
    w, a, c = state
    x, y, p = train["x"], train["y"], train["p"]
    if reference is None:
        pre = w @ x.T
        gate, h = pre > 0, np.maximum(pre, 0)
    else:
        pre0 = reference[0] @ x.T
        gate = pre0 > 0
        h = np.maximum(pre0, 0) + gate * ((w - reference[0]) @ x.T)
    e = y - c - np.mean(a[:, None] * h, axis=0)
    common_design = np.column_stack([np.ones(len(x)), train["z"]])
    common = common_design @ np.linalg.solve(
        common_design.T @ (p[:, None] * common_design), common_design.T @ (p * e))
    if component == "common":
        e = common
    elif component == "refinement":
        e = e - common
    elif component != "full":
        raise ValueError(component)
    dw = a[:, None] * ((gate * (p * e)) @ x)
    da = h @ (p * e)
    dc = p @ e
    return w + rate * dw, a + rate * da, c + rate * dc


def run():
    out = Path(__file__).resolve().parents[2] / "docs/evidence/power_law_competition/pressure"
    out.mkdir(exist_ok=True)
    protocol = dict(seed_bank=106100, seeds=6, initializations=["fanin_gaussian", "fanin_uniform"],
                    cases=["prototype", "associated"], checkpoint=200, widths=[50, 100],
                    pressure_steps=16, continuation_steps=100, rate=.025, batch=32,
                    comparison="Same-state common-only, refinement-only, full, and no-pressure",
                    model="Bilinear local dynamics with one midpoint gate refresh",
                    caution="Changes function and speed; not pure option mediation or natural-width mediation")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    for case in protocol["cases"]:
        spec = specification(richer=case == "associated")
        for seed in range(protocol["seeds"]):
            bank = protocol["seed_bank"] + seed
            train, validation, audit, _ = dataset(spec, bank + 10000)
            for kind in protocol["initializations"]:
                for width in protocol["widths"]:
                    state = initialize(width, train["x"].shape[1], kind, bank)
                    rng = np.random.default_rng(bank + 20000)
                    for _ in range(protocol["checkpoint"]):
                        state = sgd_step(state, train, .025, 32, rng)
                    initial = response(state, audit)
                    predictions, actuals, after_loss = {}, {}, {}
                    key = f"{case}_{kind}_{seed}_{width}"
                    for arm in ["common", "refinement", "full"]:
                        predicted = tuple(v.copy() for v in state)
                        reference = predicted
                        for step in range(16):
                            if step == 8:
                                reference = predicted
                            predicted = pressure_step(predicted, train, arm, .025, reference)
                        predictions[arm] = response(predicted, audit)
                    # This file is committed before any corresponding intervention outcome.
                    np.savez_compressed(out / f"{key}_forecast.npz", initial=initial, **predictions)
                    for arm in ["none", "common", "refinement", "full"]:
                        actual = tuple(v.copy() for v in state)
                        if arm != "none":
                            for _ in range(16):
                                actual = pressure_step(actual, train, arm, .025)
                        actuals[arm] = response(actual, audit)
                        rng = np.random.default_rng(bank + 70000)
                        for _ in range(100):
                            actual = sgd_step(actual, train, .025, 32, rng)
                        after_loss[arm] = float(.5 * np.sum(validation["p"] * (
                            validation["y"] - forward(actual, validation["x"]))**2))
                    np.savez_compressed(out / f"{key}_actual.npz", **actuals)
                    records.append(dict(key=key, case=case, initialization=kind, seed=seed,
                                        width=width, after_loss=after_loss,
                                        initial_mean_strength=float(initial.mean()),
                                        common_mean_change=float(np.mean(actuals["common"] - initial)),
                                        common_weakened_fraction=float(np.mean(actuals["common"] < .9 * initial))))
    summaries = []
    for case in protocol["cases"]:
        group = [r for r in records if r["case"] == case]
        for arm in ["common", "refinement", "full"]:
            effects, forecasts = [], []
            for row in group:
                f = np.load(out / f"{row['key']}_forecast.npz")
                a = np.load(out / f"{row['key']}_actual.npz")
                forecasts.extend(f[arm] - f["initial"])
                effects.extend(a[arm] - f["initial"])
            effects, forecasts = np.array(effects), np.array(forecasts)
            valid = np.abs(effects) > 1e-5
            summaries.append(dict(case=case, arm=arm, states=len(group),
                                  mean_effect=float(effects.mean()),
                                  error_ratio=float(np.sqrt(np.mean((effects - forecasts)**2)
                                                            / max(np.mean(effects**2), 1e-300))),
                                  sign_accuracy=float(np.mean(np.sign(effects[valid]) == np.sign(forecasts[valid]))),
                                  nontrivial_effects=int(valid.sum()),
                                  after_loss_difference=float(np.mean([
                                      r["after_loss"][arm] - r["after_loss"]["none"] for r in group]))))
    (out / "results.json").write_text(json.dumps(dict(protocol=protocol, records=records,
                                                      summaries=summaries), indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    run()
