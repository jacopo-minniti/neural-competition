"""Equal current options, different gate-resolved signed implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from implementation_moments import (
    hierarchy,
    moment_kernel,
    moment_output,
    moment_step,
    moments,
    response_from_kernel,
)
from toy_option_resources import advance, forward, kernel, velocity

ROOT = Path(__file__).resolve().parents[2]


def identity_bank(rng, count, m, data):
    assert m % 4 == 0
    w = np.zeros((count, m, data["x"].shape[1]))
    a = np.zeros((count, m))
    for j in range(0, m, 4):
        v = rng.normal(size=(count, data["x"].shape[1]))
        v[:, -1] = 0
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        bias = np.max(abs(v @ data["x"].T), axis=1) + 0.5
        w[:, j] = v
        w[:, j + 1] = -v
        w[:, j + 2] = v
        w[:, j + 2, -1] += bias
        w[:, j + 3, -1] = bias
        a[:, j : j + 4] = np.array([1, -1, -1, 1])
    return (0.002 * w, 0.002 * a, np.zeros(count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=296201)
    parser.add_argument("--end", type=float, default=2000)
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists()
    configs = [dict(n=1, ratio=1.0, overlap=o) for o in [1.0, 0.0]] + [
        dict(n=4, ratio=0.1, overlap=o) for o in [1.0, 0.0]
    ]
    protocol = dict(
        count=args.count,
        seed=args.seed,
        end=args.end,
        configs=configs,
        dt=0.05,
        horizons=[0.05, 0.5, 2.0, 10.0],
        construction="Zero-function ReLU identity blocks with readouts +/-a. Same predictions, full K and B at time zero, same width and metric; signed gate-resolved q changes. Artificial causal necessity test, not natural-width initialization.",
        sources={},
    )
    for name in [Path(__file__).name, "implementation_moments.py", "toy_option_resources.py"]:
        path = Path(__file__).with_name(name)
        protocol["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / name).write_bytes(path.read_bytes())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for config in configs:
        data = hierarchy(**config)
        m = 4 if config["n"] == 1 else 12
        name = f"n{config['n']}_o{config['overlap']:g}"
        plus = identity_bank(np.random.default_rng(args.seed), args.count, m, data)
        minus = (plus[0].copy(), -plus[1], plus[2].copy())
        assert np.max(abs(forward(plus, data["x"]))) < 1e-14
        np.testing.assert_allclose(kernel(plus, data), kernel(minus, data), atol=1e-14)
        states = {"plus": plus, "minus": minus}
        resources = {
            arm: [moments(tuple(v[j : j + 1] for v in state), data) for j in range(args.count)]
            for arm, state in states.items()
        }
        raw = {
            f"{arm}_{k}": v.copy()
            for arm, state in states.items()
            for k, v in zip(["w", "a", "c"], state)
        }
        np.savez_compressed(out / f"{name}_initial.npz", **raw)
        start_gates = plus[0] @ data["x"].T > 0
        crossed = {arm: np.zeros(args.count, dtype=bool) for arm in states}
        k0 = kernel(plus, data)
        frozen = np.zeros((args.count, len(data["x"])))
        times = {
            1: 0.05,
            10: 0.5,
            40: 2.0,
            200: 10.0,
            2000: 100.0,
            round(args.end / 0.05): args.end,
        }
        for i in range(1, round(args.end / 0.05) + 1):
            if i <= 200:
                frozen += 0.05 * np.einsum("sij,sj->si", k0, data["root"] * data["y"] - frozen)
                for arm in resources:
                    resources[arm] = [moment_step(r, data, 0.05) for r in resources[arm]]
            for arm in states:
                states[arm] = advance(states[arm], velocity(states[arm], data), 0.05)
                crossed[arm] |= np.any(
                    (states[arm][0] @ data["x"].T > 0) != start_gates, axis=(1, 2)
                )
            if i in times:
                for arm, state in states.items():
                    kk = kernel(state, data)
                    ff = forward(state, data["x"]) * data["root"]
                    loss = 0.5 * np.sum((ff - data["root"] * data["y"]) ** 2, axis=1)
                    with (out / "outcomes.jsonl").open("a") as f:
                        for j in range(args.count):
                            row = dict(
                                case=name,
                                **config,
                                width=m,
                                seed=j,
                                arm=arm,
                                time=times[i],
                                loss=float(loss[j]),
                                crossed=bool(crossed[arm][j]),
                            )
                            if i <= 200:
                                kr = moment_kernel(resources[arm][j], data)
                                fr = moment_output(resources[arm][j], data)
                                b = response_from_kernel(kk[j])
                                br = response_from_kernel(kr)
                                b0 = response_from_kernel(k0[j])
                                row.update(
                                    moment_function_error=float(np.linalg.norm(fr - ff[j])),
                                    frozen_function_error=float(np.linalg.norm(frozen[j] - ff[j])),
                                    moment_response_error=float(np.linalg.norm(br - b)),
                                    frozen_response_error=float(np.linalg.norm(b0 - b)),
                                    predicted_loss=float(
                                        0.5 * np.sum((fr - data["root"] * data["y"]) ** 2)
                                    ),
                                    response=b.tolist(),
                                    predicted_response=br.tolist(),
                                )
                            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(name, float(np.mean(loss)), flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
