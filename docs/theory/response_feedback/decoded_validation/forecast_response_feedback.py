"""Predict matched-response intervention effects with a fixed-gate bilinear model.

The only dynamics omitted are activation-pattern changes. All loss residuals,
both trainable layers, their coupling, optimizer metric and data measure remain.
Paired intervention differences cancel their identical initial tangent model.
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
    expand,
    freeze_gates,
    from_checkpoint,
    linearize,
    loss,
    power,
    velocity,
)
from trace_observed_learning import ROOT, append


def run(states, x, y, vx, vy, horizons, rate, batches, decode=False):
    records = {}
    for i in range(1, max(horizons) + 1):
        for s in states.values():
            ids = batches[i - 1] if batches is not None else slice(None)
            cached = cache(s, x[ids])
            e = torch.nn.functional.one_hot(y[ids], 10) - cached[0].softmax(-1)
            v = velocity(s, x[ids], e, cached)
            for key in v:
                s[key] += rate * v[key]
        if i in horizons:
            records[i] = {
                arm: dict(loss=loss(cache(s, x)[0], y), validation=loss(cache(s, vx)[0], vy))
                for arm, s in states.items()
            }
            if decode:
                for arm, s in states.items():
                    physical = {k: v for k, v in s.items() if not k.startswith("reference_")}
                    records[i][arm]["decoded_loss"] = loss(cache(physical, x)[0], y)
                    records[i][arm]["decoded_validation"] = loss(cache(physical, vx)[0], vy)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="docs/theory/observed_learning/mnist")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 20, 100, 800])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20, 80])
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--independent-forecast", action="store_true")
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    src = ROOT / args.input
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Preserve prior record")
    prior = json.loads((src / "protocol.json").read_text())
    protocol = dict(
        input=args.input,
        seeds=prior["seeds"],
        steps=args.steps,
        rate=args.rate,
        horizons=args.horizons,
        widths=[32, 128],
        copies=args.copies,
        batch=args.batch,
        streams=args.streams,
        independent_forecast=args.independent_forecast,
        decode_physical_response=args.decode,
        model="Fixed activation masks, fully interacting bilinear layers and exact cross-entropy",
        target="Reserve gain minus ordinary gain under initially matched response",
        forecast_information=(
            "Current parameters/data; independent forecast batch streams"
            if args.independent_forecast
            else "Current parameters/data; SGD forecasts condition on declared exogenous batch schedule"
        ),
        fitted_parameters=0,
        sources={},
        checkpoint_hashes={},
    )
    for path in [
        Path(__file__),
        Path(__file__).with_name("probe_response_feedback.py"),
        Path(__file__).with_name("trace_observed_learning.py"),
    ]:
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
                    for stream in range(args.streams):
                        batches = None
                        if args.batch:
                            generator = torch.Generator().manual_seed(
                                seed * 10000 + step * 100 + stream
                            )
                            batches = [
                                torch.randint(len(x), (args.batch,), generator=generator)
                                for _ in range(max(args.horizons))
                            ]
                        forecast_batches = batches
                        if args.batch and args.independent_forecast:
                            fg = torch.Generator().manual_seed(
                                seed * 10000 + step * 100 + stream + 900000000
                            )
                            forecast_batches = [
                                torch.randint(len(x), (args.batch,), generator=fg)
                                for _ in range(max(args.horizons))
                            ]
                        e = torch.nn.functional.one_hot(y, 10) - cache(base, x)[0].softmax(-1)
                        reserve = expand(base, k=args.copies)
                        match = dict(
                            function=float((cache(base, x)[0] - cache(reserve, x)[0]).abs().max()),
                            loaded_response=abs(power(base, x, e) - power(reserve, x, e)),
                        )
                        assert max(match.values()) < 1e-10
                        prediction_states = dict(
                            ordinary=freeze_gates(base),
                            reserve=freeze_gates(reserve),
                            linear=linearize(base),
                        )
                        if args.decode:
                            prediction_states["linear_reserve"] = linearize(reserve)
                        predicted = run(
                            prediction_states,
                            x,
                            y,
                            vx,
                            vy,
                            args.horizons,
                            args.rate,
                            forecast_batches,
                            decode=args.decode,
                        )
                        for h, values in predicted.items():
                            append(
                                out / "forecasts.jsonl",
                                dict(
                                    seed=seed,
                                    width=width,
                                    step=step,
                                    stream=stream,
                                    horizon=h,
                                    matching=match,
                                    initial_loss=loss(cache(base, x)[0], y),
                                    effect=values["ordinary"]["loss"] - values["reserve"]["loss"],
                                    validation_effect=values["ordinary"]["validation"]
                                    - values["reserve"]["validation"],
                                    **values,
                                ),
                            )
                        actual = run(
                            dict(ordinary=copy_state(base), reserve=reserve),
                            x,
                            y,
                            vx,
                            vy,
                            args.horizons,
                            args.rate,
                            batches,
                        )
                        for h, values in actual.items():
                            append(
                                out / "outcomes.jsonl",
                                dict(
                                    seed=seed,
                                    width=width,
                                    step=step,
                                    stream=stream,
                                    horizon=h,
                                    effect=values["ordinary"]["loss"] - values["reserve"]["loss"],
                                    validation_effect=values["ordinary"]["validation"]
                                    - values["reserve"]["validation"],
                                    **values,
                                ),
                            )
                print(json.dumps(dict(seed=seed, step=step, complete=True)), flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
