"""Natural-width transitions and component-pressure interventions on toy families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from toy_option_resources import (
    CASES,
    advance,
    common,
    forward,
    measure,
    norm2,
    task,
    transport_audit,
    velocity,
)

ROOT = Path(__file__).resolve().parents[2]


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=295001)
    parser.add_argument("--end", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["ordinary", "early_no_common", "matched_movement", "late_no_common"],
    )
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists(), "Preserve previous experiment"
    protocol = dict(
        seed=args.seed,
        count=args.count,
        cases={k: CASES[k] for k in args.cases},
        arms=args.arms,
        dt=args.dt,
        end=args.end,
        scale=0.002,
        early_window=[0, 20],
        late_window=[100, 120],
        widths="capacity witness width and twice that width, nested initial units",
        pressure="Orthogonal conditional-mean residual on attribute groups versus context-dependent residual",
        control="Full-gradient direction with per-seed optimizer-metric velocity norm matched to early_no_common path at each step",
        sources={},
    )
    for name in [Path(__file__).name, "toy_option_resources.py"]:
        path = Path(__file__).with_name(name)
        protocol["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / name).write_bytes(path.read_bytes())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    times = sorted(
        set(
            [float(v) for v in range(0, 21, 2)]
            + [30.0, 50.0, 100.0, 120.0, 200.0, 500.0, 1000.0, 2000.0, args.end]
        )
    )
    times = [t for t in times if t <= args.end]
    steps = {round(t / args.dt): t for t in times}
    for name in args.cases:
        data = task(name)
        dense = task(name, dense=True)
        n = data["capacity"]
        rng = np.random.default_rng(args.seed)
        iw = 0.002 * rng.normal(size=(args.count, 2 * n, data["x"].shape[1]))
        ia = 0.002 * rng.normal(size=(args.count, 2 * n))
        for m in [n, 2 * n]:
            states = {
                arm: (iw[:, :m].copy(), ia[:, :m].copy(), np.zeros(args.count)) for arm in args.arms
            }
            raw = {}
            for i in range(round(args.end / args.dt) + 1):
                t = i * args.dt
                if i in steps:
                    for arm, state in states.items():
                        obs = measure(state, data)
                        dense_loss = 0.5 * np.sum(
                            dense["p"] * (forward(state, dense["x"]) - dense["y"]) ** 2, axis=1
                        )
                        for seed in range(args.count):
                            append(
                                out / "trajectory.jsonl",
                                dict(
                                    case=name,
                                    width=m,
                                    arm=arm,
                                    seed=seed,
                                    time=steps[i],
                                    dense_loss=float(dense_loss[seed]),
                                    **{k: float(v[seed]) for k, v in obs.items()},
                                ),
                            )
                        if t in [0, 20, 100, args.end]:
                            for key, v in zip(["w", "a", "c"], state):
                                raw[f"{arm}_{int(t)}_{key}"] = v.copy()
                    if "ordinary" in states and t in [2, 6, 10, 20, 100]:
                        tr = transport_audit(states["ordinary"], data, args.dt)
                        for seed in range(args.count):
                            append(
                                out / "transport.jsonl",
                                dict(
                                    case=name,
                                    width=m,
                                    seed=seed,
                                    time=t,
                                    **{k: v[seed].tolist() for k, v in tr.items()},
                                ),
                            )
                if i == round(args.end / args.dt):
                    break
                updates = {}
                for arm, state in states.items():
                    if arm == "matched_movement" and t < 20:
                        continue
                    e = data["y"] - forward(state, data["x"])
                    if (arm == "early_no_common" and t < 20) or (
                        arm == "late_no_common" and 100 <= t < 120
                    ):
                        e -= common(e, data)
                    updates[arm] = velocity(state, data, e)
                if "matched_movement" in states and t < 20:
                    assert "early_no_common" in states
                    v = velocity(states["matched_movement"], data)
                    target = norm2(updates["early_no_common"], m)
                    scale = np.sqrt(target / np.maximum(norm2(v, m), 1e-300))
                    updates["matched_movement"] = tuple(
                        vv * scale.reshape((len(scale),) + (1,) * (vv.ndim - 1)) for vv in v
                    )
                    assert np.max(abs(norm2(updates["matched_movement"], m) - target)) < 1e-10
                for arm in states:
                    states[arm] = advance(states[arm], updates[arm], args.dt)
            np.savez_compressed(out / f"{name}_{m}_states.npz", **raw)
            print(
                json.dumps(
                    dict(
                        case=name,
                        width=m,
                        end=args.end,
                        losses={
                            a: float(
                                np.mean(
                                    0.5
                                    * np.sum(
                                        data["p"] * (forward(s, data["x"]) - data["y"]) ** 2, axis=1
                                    )
                                )
                            )
                            for a, s in states.items()
                        },
                    )
                ),
                flush=True,
            )
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
