"""Prospective gate-aware pressure forecasts, validated on ordinary future SGD."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from neural_competition import frequency_pressure as fp
from neural_competition import parity


def outcomes(state, origin, x, y, p, bits, q):
    e = y - fp.forward(state, x)
    reference = y - fp.forward(origin, x)
    norm = np.sum(p * reference**2)
    fixed_rate = fp.update_power(fp.velocity(state, x, reference, p)) / norm
    gates = state[0] @ x.T > 0
    previous = origin[0] @ x.T > 0
    return dict(loss=(.5 * e**2 @ p).tolist(),
                conditional_loss=((.5 * e**2 * p) @ bits / q).tolist(),
                fixed_rate=np.atleast_1d(fixed_rate).tolist(),
                closing_mass=(np.mean(gates < previous, axis=-2) @ p).tolist(),
                opening_mass=(np.mean(gates > previous, axis=-2) @ p).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--steps", type=int, nargs="+", default=[2000, 6000, 10000])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--seed-bank", type=int, default=937100)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Preserve prior cohorts")
    training = json.loads((run / "protocol.json").read_text())
    checkpoints = np.load(run / "checkpoints.npz")
    q = np.array(training["q"])
    pop, _ = parity.panels(q)
    x, y, p = fp.inputs(pop["x"], q, training["encoding"]), pop["y"], pop["p"]
    rate, batch = training["rate"], training["batch"]
    protocol = dict(**vars(args), training=training,
                    forecast="state-dependent moment-matched Gaussian gradient noise; all gates refreshed",
                    validation="independent ordinary SGD minibatches, all parameters updated",
                    weakening="fixed initial residual kernel response falls by at least 10%",
                    independence="future prediction/validation RNGs independent; states clustered by seed",
                    sources={})
    (out / "sources").mkdir()
    for source in [Path(__file__), Path(fp.__file__)]:
        content = source.read_bytes()
        protocol["sources"][source.name] = hashlib.sha256(content).hexdigest()
        (out / "sources" / source.name).write_bytes(content)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    cases = []
    started = time.monotonic()
    for step in args.steps:
        for seed in range(args.seeds):
            for width in args.widths:
                origin = tuple(checkpoints[f"{step}_{width}_{name}"][seed] for name in ["w", "a", "c"])
                cases.append((step, seed, width, origin))
    for index, (step, seed, width, origin) in enumerate(cases):
        states = tuple(np.broadcast_to(v, (args.particles,) + v.shape).copy() for v in origin)
        mean_state = tuple(v.copy() for v in origin)
        rng = np.random.default_rng(args.seed_bank + step * 100 + seed)
        predictions = {}
        initial = outcomes(origin, origin, x, y, p, pop["x"], q)
        for horizon in range(1, max(args.horizons) + 1):
            states = fp.gaussian_forecast_step(states, x, y, p, rate, batch, rng)
            e = y - fp.forward(mean_state, x)
            mean_state = fp.advance(mean_state, fp.velocity(mean_state, x, e, p), rate)
            if horizon in args.horizons:
                predictions[horizon] = dict(gaussian=outcomes(states, origin, x, y, p, pop["x"], q),
                                            mean_gradient=outcomes(mean_state, origin, x, y, p,
                                                                   pop["x"], q))
        with (out / "predictions.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(step=step, seed=seed, width=width, initial=initial,
                                         predictions=predictions)) + "\n")
        if index % 6 == 0:
            print(json.dumps(dict(phase="prediction", cases=index + 1,
                                  seconds=round(time.monotonic() - started, 1))), flush=True)
    # Every forecast is saved before any validation continuation is run.
    for index, (step, seed, width, origin) in enumerate(cases):
        states = tuple(np.broadcast_to(v, (args.particles,) + v.shape).copy() for v in origin)
        rng = np.random.default_rng(args.seed_bank + 10000000 + step * 100 + seed)
        actual = {}
        for horizon in range(1, max(args.horizons) + 1):
            ids = rng.choice(len(p), size=(args.particles, batch), p=p)
            xb, yb = x[ids], y[ids]
            e = yb - fp.forward(states, xb)
            states = fp.advance(states, fp.velocity(states, xb, e, 1 / batch), rate)
            if horizon in args.horizons:
                actual[horizon] = outcomes(states, origin, x, y, p, pop["x"], q)
        with (out / "actual.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(step=step, seed=seed, width=width, actual=actual)) + "\n")
        if index % 6 == 0:
            print(json.dumps(dict(phase="validation", cases=index + 1,
                                  seconds=round(time.monotonic() - started, 1))), flush=True)
    print(json.dumps(dict(complete=True, seconds=round(time.monotonic() - started, 1))), flush=True)


if __name__ == "__main__":
    main()
