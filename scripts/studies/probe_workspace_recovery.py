"""Post-pilot discriminators: fixed-feature rescue, symmetry, and long recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from explore_recruitable_workspace import (
    ROOT,
    append,
    copy,
    direction,
    evolve,
    expand,
    predict,
    regions,
    stamp,
    task,
)


def custom(s, x, y, vx, vy, off, voff, steps=1200, freeze=False, rate=0.06):
    s = copy(s)
    rows = []
    for t in range(steps + 1):
        if t in [0, 20, 100, 300, 1200, 6000]:
            tr = predict(s, x) - off - y
            va = predict(s, vx) - voff - vy
            rows.append(
                dict(
                    step=t,
                    train=float(np.mean(tr**2) / 2),
                    validation=float(np.mean(va**2) / 2),
                    validation_regions=[float(np.mean(va[k] ** 2) / 2) for k in regions(vx)],
                )
            )
        if t < steps:
            vel, _ = direction(s, x, y + off)
            if freeze:
                vel = (np.zeros_like(vel[0]), vel[1], vel[2])
            s = tuple(a + rate * d for a, d in zip(s, vel))
    return s, rows


def exact_target(seed):
    r = np.random.default_rng(seed + 70000)
    centers = np.array([-0.60, 0.0, 0.57]) + r.uniform(-0.035, 0.035, 3)
    widths = np.array([0.25, 0.28, 0.23])
    heights = np.array([1.0, -0.8, 0.7]) * r.uniform(0.9, 1.1, 3)
    knots = np.array([[c - w, c, c + w] for c, w in zip(centers, widths)]).ravel()
    coef = np.array([h / w * np.array([1.0, -2.0, 1.0]) for h, w in zip(heights, widths)]).ravel()
    return np.column_stack([np.ones(9), -knots]), 9 * coef, np.array([0.2, 0.0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    src, out = ROOT / a.source, ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Choose fresh output.")
    prot = json.loads((src / "protocol.json").read_text())
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                created=stamp(),
                source=a.source,
                seeds=prot["seeds"],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                note="Designed after pilot; saved before these new continuations. No outcome-selected seeds.",
                forecasts=[
                    "Fixed-feature rescue would favor initial added coverage over nonlinear replacement.",
                    "Tiny clone perturbations may not provide accessible alternative implementations at this budget.",
                    "Long ordinary continuation and fast control may remove the apparent obstruction.",
                    "Dormant-resource recycling changes function by zero on bounded task support; improvement need not imply dynamic loss.",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    audits = []
    for seed in prot["seeds"]:
        z = np.load(src / f"states_{seed}.npz")
        x, y, vx, vy = [z[k] for k in ["x", "y", "vx", "vy"]]
        d, vd = z["demand"], z["vdemand"]
        s = tuple(z[f"9_peer_{k}"] for k in ["w", "a", "c"])
        clone = expand(s, "clone", seed)
        rng = np.random.default_rng(seed + 110000)
        delta = rng.normal(size=(9, 2)) * 1e-3
        pert = copy(clone)
        pert[0][:9] += delta
        pert[0][9:18] -= delta
        dormant = (np.abs(s[0][:, 0]) + s[0][:, 1]) < 0
        recycled = copy(s)
        recycled[0][dormant] = rng.normal(size=(int(dormant.sum()), 2))
        recycled[1][dormant] = 0
        for arm, state, freeze, steps in [
            ("narrow_readout", s, True, 1200),
            ("fresh_readout", expand(s, "fresh", seed), True, 1200),
            ("perturbed_clone", pert, False, 1200),
            ("recycled_dormant", recycled, False, 1200),
            ("narrow_long_assay", s, False, 6000),
        ]:
            # Recenter perturbation too: same exact initial residual in every assay.
            oo, vv = predict(state, x), predict(state, vx)
            _, rows = custom(state, x, d, vx, vd, oo, vv, steps, freeze)
            for row in rows:
                append(out / "assays.jsonl", dict(seed=seed, arm=arm, **row))
        for m, rate in [(9, 0.06), (36, 0.06), (9, 0.12)]:
            start = tuple(z[f"{m}_initial_{k}"] for k in ["w", "a", "c"])
            _, rows = evolve(start, x, y, 30000, rate, vx, vy, [3000, 6000, 12000, 30000])
            for row in rows:
                append(out / "long.jsonl", dict(seed=seed, width=m, rate=rate, **row))
        ex = exact_target(seed)
        xx = np.column_stack([np.linspace(-1, 1, 10001), np.ones(10001)])
        tx, ty = task(seed, 2048, True)
        represent_error = float(np.max(np.abs(predict(ex, tx) - ty)))
        assert represent_error < 1e-12
        audits.append(
            dict(
                seed=seed,
                represent_error=represent_error,
                initially_dormant=int(
                    np.sum(np.abs(z["9_initial_w"][:, 0]) + z["9_initial_w"][:, 1] < 0)
                ),
                peer_dormant=int(dormant.sum()),
                recycled_function_error=float(
                    np.max(np.abs(predict(recycled, xx) - predict(s, xx)))
                ),
                perturb_function_error=float(np.max(np.abs(predict(pert, xx) - predict(s, xx)))),
            )
        )
        print(json.dumps(dict(seed=seed, done=True)), flush=True)
    (out / "audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    (out / "complete.json").write_text(json.dumps(dict(completed=stamp())) + "\n")


if __name__ == "__main__":
    main()
