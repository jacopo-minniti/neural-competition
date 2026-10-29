"""Restore a random feature direction at fixed width, prediction and kernel trace.

The control shares the exact readout redistribution and zero output coefficient,
but retains the old incoming feature. Treatment changes only that zero-output
feature's direction and matches its training-measure activation norm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from toy_option_resources import advance, forward, kernel, measure, task, velocity

ROOT = Path(__file__).resolve().parents[2]


def recycle(state, data, dense, rng):
    w, a, c = (v[0].copy() for v in state)
    m = len(a)
    x, p = data["x"], data["p"]
    h = np.maximum(x @ w.T, 0)
    before = forward(state, x)[0]
    dense_before = forward(state, dense["x"])[0]
    candidates = []
    for j in range(m):
        energy = np.sum(p * h[:, j] ** 2)
        if energy < 1e-12:
            continue
        remaining = np.arange(m) != j
        design = np.column_stack([h[:, remaining] / np.sqrt(m), np.ones(len(x))])
        rhs = a[j] * h[:, j] / m
        delta = np.linalg.pinv(design * np.sqrt(p[:, None]), rcond=1e-11) @ (rhs * np.sqrt(p))
        aa = a.copy()
        aa[j] = 0
        aa[remaining] += np.sqrt(m) * delta[:-1]
        ss = (w[None].copy(), aa[None], np.array([c + delta[-1]]))
        error = max(
            np.max(abs(forward(ss, x)[0] - before)),
            np.max(abs(forward(ss, dense["x"])[0] - dense_before)),
        )
        if error < 1e-8:
            cost = float(np.sum(delta**2) + a[j] ** 2 / m)
            candidates.append((cost, j, ss, energy, error))
    if not candidates:
        return None
    cost, j, control, energy, error = min(candidates, key=lambda t: t[0])
    for _ in range(100):
        new_w = rng.normal(size=w.shape[1])
        hh = np.maximum(x @ new_w, 0)
        e = np.sum(p * hh**2)
        if e > 1e-10:
            break
    else:
        raise RuntimeError("No nonzero random feature draw")
    treatment = tuple(v.copy() for v in control)
    treatment[0][0, j] = new_w * np.sqrt(energy / e)
    function_error = max(
        np.max(abs(forward(treatment, x) - forward(control, x))),
        np.max(abs(forward(treatment, dense["x"]) - forward(control, dense["x"]))),
    )
    trace_error = abs(np.trace(kernel(treatment, data)[0]) - np.trace(kernel(control, data)[0]))
    assert function_error < 1e-10 and trace_error < 1e-8
    return (
        control,
        treatment,
        dict(
            unit=j,
            redistribution_cost=cost,
            preservation_error=error,
            function_error=float(function_error),
            trace_error=float(trace_error),
            old_w_norm=float(np.linalg.norm(w[j])),
            new_w_norm=float(np.linalg.norm(treatment[0][0, j])),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["discrete", "continuous", "separate_attribute", "two_refinements"],
    )
    parser.add_argument("--time", type=int, default=20)
    parser.add_argument("--end", type=float, default=500)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    src, out = ROOT / args.input, ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists()
    protocol = dict(
        input=args.input,
        cases=args.cases,
        time=args.time,
        end=args.end,
        repeats=args.repeats,
        dt=0.05,
        selection="All checkpoint states; exact prediction-preserving readout redistribution required on training and dense grids. No selection by future outcome.",
        treatment="Random replacement of one zero-output feature, activation energy matched to its old direction",
        control="Same redistributed readouts and zero output unit, with original incoming feature retained",
        limits="Same width/function/kernel trace, not same full kernel or parameter norm. Random directions are not selected for task alignment.",
        sources={},
        input_hashes={},
    )
    for name in [Path(__file__).name, "toy_option_resources.py"]:
        path = Path(__file__).with_name(name)
        protocol["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / name).write_bytes(path.read_bytes())
    for name in args.cases:
        n = task(name)["capacity"]
        for m in [n, 2 * n]:
            file = f"{name}_{m}_states.npz"
            protocol["input_hashes"][file] = hashlib.sha256((src / file).read_bytes()).hexdigest()
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for name in args.cases:
        data, dense = task(name), task(name, True)
        for m in [data["capacity"], 2 * data["capacity"]]:
            raw = np.load(src / f"{name}_{m}_states.npz")
            base = tuple(raw[f"ordinary_{args.time}_{k}"] for k in ["w", "a", "c"])
            starts = []
            meta = []
            for seed in range(len(base[0])):
                for repeat in range(args.repeats):
                    rng = np.random.default_rng(295900 + 1000 * seed + repeat)
                    s = tuple(v[seed : seed + 1].copy() for v in base)
                    result = recycle(s, data, dense, rng)
                    row = dict(
                        case=name, width=m, seed=seed, repeat=repeat, eligible=result is not None
                    )
                    if result is None:
                        with (out / "ineligible.jsonl").open("a") as f:
                            f.write(json.dumps(row) + "\n")
                        continue
                    control, treatment, matching = result
                    for arm, state in [("control", control), ("recycled", treatment)]:
                        starts.append(state)
                        meta.append(dict(**row, arm=arm, **matching))
            if not starts:
                continue
            states = tuple(np.concatenate([s[k] for s in starts]) for k in range(3))
            checkpoints = {
                0: args.time,
                round((100 - args.time) / 0.05): 100,
                round((args.end - args.time) / 0.05): args.end,
            }
            for i in range(round((args.end - args.time) / 0.05) + 1):
                if i in checkpoints:
                    obs = measure(states, data)
                    dense_loss = 0.5 * np.sum(
                        dense["p"] * (forward(states, dense["x"]) - dense["y"]) ** 2, axis=1
                    )
                    with (out / "outcomes.jsonl").open("a") as f:
                        for j, row in enumerate(meta):
                            f.write(
                                json.dumps(
                                    dict(
                                        **row,
                                        time=checkpoints[i],
                                        dense_loss=float(dense_loss[j]),
                                        **{k: float(v[j]) for k, v in obs.items()},
                                    ),
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                if i < round((args.end - args.time) / 0.05):
                    states = advance(states, velocity(states, data), 0.05)
            print(name, m, "eligible", len(starts) // (2 * args.repeats), flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
