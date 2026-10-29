"""Exploratory step refinement of gate versus smooth nonlinear loss work.

Motivated by nonvanishing toy corrections at dt=.025. These are repeated
numerical discretizations of the same 64 starts, not additional independent runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task
from trace_observed_learning import ROOT, append, toy_forward, toy_tangent, toy_velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Preserve previous result")
    protocol = dict(
        seed=283010,
        count=64,
        widths=[4, 8],
        horizon=20,
        rates=[0.05, 0.025, 0.0125, 0.00625],
        cases=[["discrete", 1, 0.0], ["continuous", 3, 0.15], ["dense", 9, 0.15]],
        purpose="Exploratory numerical gate audit after half-step nonlinearity persisted",
        sources={},
    )
    for path in [
        Path(__file__),
        Path(__file__).with_name("trace_observed_learning.py"),
        Path(__file__).with_name("explore_stable_distinctions.py"),
    ]:
        protocol["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        (out / path.name).write_bytes(path.read_bytes())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for name, nodes, spread in protocol["cases"]:
        x, p, y = task(nodes, spread, 0.1)
        rng = np.random.default_rng(protocol["seed"])
        iw = 0.002 * rng.normal(size=(64, 8, 3))
        ia = 0.002 * rng.normal(size=(64, 8))
        for width in protocol["widths"]:
            for dt in protocol["rates"]:
                s = iw[:, :width].copy(), ia[:, :width].copy(), np.zeros(64)
                sums = np.zeros((64, 4))
                loss0 = 0.5 * np.sum(p * (y - toy_forward(s, x)[0]) ** 2, axis=1)
                for _ in range(round(20 / dt)):
                    f, pre = toy_forward(s, x)
                    e = y - f
                    velocity = toy_velocity(s, x, p, e)
                    linear = dt * toy_tangent(s, x, velocity)
                    new = tuple(v + dt * dv for v, dv in zip(s, velocity))
                    fn, pren = toy_forward(new, x)
                    virtual = new[2][:, None] + np.mean(
                        new[1][..., None] * (pre > 0) * pren, axis=1
                    )
                    smooth = dt**2 * np.mean(
                        velocity[1][..., None] * (pre > 0) * (velocity[0] @ x.T), axis=1
                    )
                    gate = fn - virtual
                    assert np.max(abs(fn - f - linear - smooth - gate)) < 1e-12
                    sums += np.column_stack(
                        [
                            (p * e * linear).sum(1),
                            (p * e * smooth).sum(1),
                            (p * e * gate).sum(1),
                            0.5 * (p * (fn - f) ** 2).sum(1),
                        ]
                    )
                    s = new
                loss = 0.5 * np.sum(p * (y - toy_forward(s, x)[0]) ** 2, axis=1)
                assert (
                    np.max(abs(loss0 - loss - sums[:, 0] - sums[:, 1] - sums[:, 2] + sums[:, 3]))
                    < 1e-10
                )
                for i in range(64):
                    append(
                        out / "outcomes.jsonl",
                        dict(
                            case=name,
                            width=width,
                            dt=dt,
                            seed=i,
                            loss=float(loss[i]),
                            signal=float(sums[i, 0]),
                            smooth=float(sums[i, 1]),
                            gate=float(sums[i, 2]),
                            cost=float(sums[i, 3]),
                        ),
                    )
                print(
                    json.dumps(
                        dict(
                            case=name,
                            width=width,
                            dt=dt,
                            mean_loss=float(loss.mean()),
                            smooth=float(sums[:, 1].mean()),
                            gate=float(sums[:, 2].mean()),
                        )
                    ),
                    flush=True,
                )
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
