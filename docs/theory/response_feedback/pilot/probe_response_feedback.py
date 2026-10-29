"""Minimal response-feedback forecasts under initially response-matched changes.

One scalar feedback coefficient, with a demand-relaxation rate, predicts the
loss effect of slowing coefficient/feature feedback. No coefficients fitted.
Full-gradient branches isolate response transport from minibatch variability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from trace_observed_learning import ROOT, append


def copy_state(s):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in s.items()}


def from_checkpoint(path):
    d = torch.load(path, weights_only=True)
    return dict(
        w=d["w"].double(),
        b=d["b"].double(),
        a=d["a"].double(),
        c=d["c"].double(),
        norm=np.sqrt(len(d["w"])),
        mw=1.0,
        ma=1.0,
    )


def expand(s, k=4, uniform=False):
    t = copy_state(s)
    t["w"] = s["w"].repeat(k, 1)
    t["b"] = s["b"].repeat(k)
    fractions = (
        torch.ones(k, dtype=torch.float64) / k if uniform else torch.tensor([1.0] + [0.0] * (k - 1))
    )
    t["a"] = torch.cat([f * s["a"] for f in fractions], dim=1)
    t["ma"] = 1 / k
    t["mw"] = 1 / float(fractions.square().sum())
    return t


def cache(s, x):
    z = x @ s["w"].T + s["b"]
    h = z.relu()
    return h @ s["a"].T / s["norm"] + s["c"], z, h


def velocity(s, x, demand, cached=None):
    f, z, h = cache(s, x) if cached is None else cached
    back = (demand @ s["a"]) * (z > 0) / s["norm"]
    return dict(
        w=s["mw"] * back.T @ x / len(x),
        b=s["mw"] * back.mean(0),
        a=s["ma"] * demand.T @ h / (len(x) * s["norm"]),
        c=demand.mean(0),
    )


def power(s, x, demand, cached=None):
    v = velocity(s, x, demand, cached)
    return float(
        (v["w"].square().sum() + v["b"].square().sum()) / s["mw"]
        + v["a"].square().sum() / s["ma"]
        + v["c"].square().sum()
    )


def update(s, v, rate):
    t = copy_state(s)
    for key in v:
        t[key] += rate * v[key]
    return t


def loss(f, y):
    return float(torch.nn.functional.cross_entropy(f, y))


def forecast(s, x, y, rate, k, horizons):
    f, z, h = cache(s, x)
    p = f.softmax(-1)
    e = torch.nn.functional.one_hot(y, 10) - p
    v = velocity(s, x, e, (f, z, h))
    df = (((x @ v["w"].T + v["b"]) * (z > 0)) @ s["a"].T + h @ v["a"].T) / s["norm"] + v["c"]
    signal = power(s, x, e, (f, z, h))
    curvature = float(((p * df.square()).sum(1) - (p * df).sum(1).square()).mean())
    lam = curvature / signal
    # Exact single-step difference in functions at matched initial response.
    next_s = update(s, v, rate)
    dh = cache(next_s, x)[2] - h
    interaction = rate * (dh @ v["a"].T) / s["norm"]
    q = float((e * interaction).sum(1).mean()) / rate**2
    # Within-cell identity: directional response transport = 4 q.
    # It need not describe other simultaneous gate transport.
    rows = []
    for horizon in horizons:
        leading = -(1 - 1 / k) * q * rate**2 * horizon * (2 * horizon - 1)
        # Single loaded mode closure: K evolves under residual drive.
        # Initial effective energy is S/(2 lambda); no claim it equals total CE.
        initial_r = np.sqrt(signal / lam)
        gains = []
        valid = True
        for fraction in [1.0, 1 / k]:
            r = initial_r
            stiffness = lam
            feedback = 4 * fraction * q / initial_r**3
            for _ in range(horizon):
                dr = -rate * stiffness * r - rate**2 * feedback * r * r / 4
                dk = rate * feedback * r
                r += dr
                stiffness += dk
                valid &= bool(np.isfinite(r) and np.isfinite(stiffness) and stiffness > 0)
            gains.append(0.5 * (initial_r**2 - r * r))
        rows.append(
            dict(
                horizon=horizon,
                zero=0.0,
                leading=leading,
                relaxed=gains[1] - gains[0],
                relaxed_valid=valid,
            )
        )
    return dict(
        signal=signal, curvature_rate=lam, feedback=q, initial_loss=loss(f, y), predictions=rows
    ), e


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="docs/theory/observed_learning/mnist")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rate", type=float, default=0.01)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 20, 100, 400, 800])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40])
    args = parser.parse_args()
    src = ROOT / args.input
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Preserve old experiment")
    prior = json.loads((src / "protocol.json").read_text())
    protocol = dict(
        input=args.input,
        seeds=prior["seeds"],
        steps=args.steps,
        rate=args.rate,
        horizons=args.horizons,
        widths=[32, 128],
        copies=4,
        training="full-gradient CE",
        target="reserve gain minus ordinary gain, same initial function and kernel",
        hypotheses=[
            "Frozen response predicts zero treatment effect.",
            "Signed coefficient-feature feedback predicts effect sign.",
            "A single relaxing mode extends a quadratic local forecast.",
        ],
        sources={},
        checkpoint_hashes={},
    )
    for path in [Path(__file__), Path(__file__).with_name("trace_observed_learning.py")]:
        protocol["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        (out / path.name).write_bytes(path.read_bytes())
    for seed in prior["seeds"]:
        for width in [32, 128]:
            for step in args.steps:
                path = src / f"model_{seed}_{width}_{step}.pt"
                protocol["checkpoint_hashes"][path.name] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = torch.load(ROOT / "data/mnist/prepared.pt", weights_only=True)
    x, y = data["xtrain"][:8192].double(), data["train_labels"][:8192]
    vx, vy = data["xval"][:1024].double(), data["val_labels"][:1024]
    torch.set_num_threads(2)
    with torch.no_grad():
        for seed in prior["seeds"]:
            for step in args.steps:
                for width in [32, 128]:
                    base = from_checkpoint(src / f"model_{seed}_{width}_{step}.pt")
                    prediction, e = forecast(base, x, y, args.rate, 4, args.horizons)
                    append(
                        out / "forecasts.jsonl",
                        dict(seed=seed, width=width, step=step, **prediction),
                    )
                    ordinary, reserve, clones = (
                        copy_state(base),
                        expand(base),
                        expand(base, uniform=True),
                    )
                    assert (
                        abs(loss(cache(ordinary, x)[0], y) - loss(cache(reserve, x)[0], y)) < 1e-12
                    )
                    assert abs(power(ordinary, x, e) - power(reserve, x, e)) < 1e-10
                    for i in range(1, max(args.horizons) + 1):
                        for s in [ordinary, reserve, clones]:
                            cached = cache(s, x)
                            demand = torch.nn.functional.one_hot(y, 10) - cached[0].softmax(-1)
                            v = velocity(s, x, demand, cached)
                            for key in v:
                                s[key] += args.rate * v[key]
                        if i in args.horizons:
                            lo = loss(cache(ordinary, x)[0], y)
                            lr = loss(cache(reserve, x)[0], y)
                            lc = loss(cache(clones, x)[0], y)
                            assert abs(lc - lo) < 1e-10
                            append(
                                out / "outcomes.jsonl",
                                dict(
                                    seed=seed,
                                    width=width,
                                    step=step,
                                    horizon=i,
                                    ordinary_loss=lo,
                                    reserve_loss=lr,
                                    effect=lo - lr,
                                    ordinary_validation=loss(cache(ordinary, vx)[0], vy),
                                    reserve_validation=loss(cache(reserve, vx)[0], vy),
                                    clone_error=lc - lo,
                                    ordinary_fixed_strength=power(ordinary, x, e),
                                    reserve_fixed_strength=power(reserve, x, e),
                                ),
                            )
                print(json.dumps(dict(seed=seed, step=step, completed=True)), flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
