"""Predict a useful correction's response under pressure from other MNIST classes.

One focal class is assigned by seed before branch outcomes. All original labels
remain unchanged. Compare full-task exposure with peer-only exposure, then both
receive full-task recovery. Current function, kernel, data and rate initially match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from probe_response_feedback import (
    cache,
    copy_state,
    freeze_gates,
    from_checkpoint,
    loss,
    option_value,
    power,
    velocity,
)
from trace_observed_learning import ROOT, append


def evolve(s, x, y, vx, vy, focal, demand, probability, rate, decode=False):
    mask = y == focal
    histories = {}
    for history in ["full", "peers"]:
        state = copy_state(s)
        records = {}
        for t in range(101):
            if t in [0, 20, 40, 100]:
                f = cache(state, x)[0]
                fv = cache(state, vx)[0]
                row = dict(
                    loss=loss(f, y),
                    focal_loss=loss(f[mask], y[mask]),
                    validation=loss(fv, vy),
                    focal_validation=loss(fv[vy == focal], vy[vy == focal]),
                )
                if t in [0, 20]:
                    row.update(
                        fixed_power=power(state, x, demand),
                        **option_value(state, x, demand, probability),
                    )
                if decode:
                    physical = {k: v for k, v in state.items() if not k.startswith("reference_")}
                    pf = cache(physical, x)[0]
                    pv = cache(physical, vx)[0]
                    row["decoded"] = dict(
                        loss=loss(pf, y),
                        focal_loss=loss(pf[mask], y[mask]),
                        validation=loss(pv, vy),
                        focal_validation=loss(pv[vy == focal], vy[vy == focal]),
                    )
                    if t in [0, 20]:
                        row["decoded"].update(
                            fixed_power=power(physical, x, demand),
                            **option_value(physical, x, demand, probability),
                        )
                records[str(t)] = row
            if t == 100:
                break
            use = ~mask if history == "peers" and t < 20 else slice(None)
            cached = cache(state, x[use])
            e = torch.nn.functional.one_hot(y[use], 10) - cached[0].softmax(-1)
            v = velocity(state, x[use], e, cached)
            for key in v:
                state[key] += rate * v[key]
        histories[history] = records
    return histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="docs/theory/observed_learning/mnist")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[20, 100])
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    src = ROOT / args.input
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Preserve previous experiment")
    prior = json.loads((src / "protocol.json").read_text())
    protocol = dict(
        input=args.input,
        seeds=prior["seeds"],
        widths=[32, 128],
        steps=args.steps,
        focal_class="seed modulo 10, fixed before branch outcomes",
        pressure_steps=20,
        recovery_steps=80,
        rate=0.05,
        tau=1.0,
        training="full-gradient original cross-entropy",
        capability="Fixed initial focal residual, fixed initial full-task curvature, full 8192-image measure",
        model="fixed-gate bilinear, no fitted coefficients",
        decode_physical_response=args.decode,
        claim="Test whether peer pressure actually weakens useful options; sign not assumed",
        sources={},
    )
    for path in [
        Path(__file__),
        Path(__file__).with_name("probe_response_feedback.py"),
        Path(__file__).with_name("trace_observed_learning.py"),
        Path(__file__).with_name("audit_conditional_editability.py"),
    ]:
        protocol["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        (out / path.name).write_bytes(path.read_bytes())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = torch.load(ROOT / "data/mnist/prepared.pt", weights_only=True)
    x, y = data["xtrain"][:8192].double(), data["train_labels"][:8192]
    vx, vy = data["xval"][:1024].double(), data["val_labels"][:1024]
    torch.set_num_threads(2)
    with torch.no_grad():
        for seed in prior["seeds"]:
            focal = seed % 10
            for step in args.steps:
                for width in [32, 128]:
                    base = from_checkpoint(src / f"model_{seed}_{width}_{step}.pt")
                    p = cache(base, x)[0].softmax(-1)
                    d = (torch.nn.functional.one_hot(y, 10) - p) * (y == focal)[:, None]
                    predicted = evolve(
                        freeze_gates(base), x, y, vx, vy, focal, d, p, 0.05, decode=args.decode
                    )
                    append(
                        out / "forecasts.jsonl",
                        dict(seed=seed, width=width, step=step, focal=focal, histories=predicted),
                    )
                    actual = evolve(base, x, y, vx, vy, focal, d, p, 0.05)
                    append(
                        out / "outcomes.jsonl",
                        dict(seed=seed, width=width, step=step, focal=focal, histories=actual),
                    )
                print(json.dumps(dict(seed=seed, step=step, complete=True)), flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
