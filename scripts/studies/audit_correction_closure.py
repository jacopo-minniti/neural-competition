"""Retrospective separation of loss curvature from response renewal in recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from probe_causal_response import evolve, physical, quadratic_gains, signed_expand
from probe_response_feedback import (
    cache,
    copy_state,
    freeze_gates,
    from_checkpoint,
    linearize,
    loss,
    velocity,
)
from trace_observed_learning import ROOT, append


def curve(state, x, target, offset, steps=20, decode=False):
    s = copy_state(state)
    initial = loss(cache(s, x)[0] + offset, target)
    result = {}
    for t in range(steps):
        cached = cache(s, x)
        residual = target - (cached[0] + offset).softmax(-1)
        v = velocity(s, x, residual, cached)
        for k in v:
            s[k] += 0.05 * v[k]
        if t + 1 in [1, 5, 20]:
            result[str(t + 1)] = initial - loss(
                cache(physical(s) if decode else s, x)[0] + offset, target
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists()
    cases = json.loads((ROOT / "docs/evidence/causal_response/pilot_cases.json").read_text())
    protocol = dict(
        cases=cases,
        kind="Retrospective diagnostic on selected development cases; actual post-pressure parameters supplied",
        sources={},
    )
    for name in [Path(__file__).name, "probe_causal_response.py", "probe_response_feedback.py"]:
        p = Path(__file__).with_name(name)
        protocol["sources"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
        (out / name).write_bytes(p.read_bytes())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = torch.load(ROOT / "data/mnist/prepared.pt", weights_only=True)
    x, y = data["xtrain"][:8192].double(), data["train_labels"][:8192]
    labels = torch.nn.functional.one_hot(y, 10)
    torch.set_num_threads(2)
    with torch.no_grad():
        for case in cases:
            for width in [32, 128]:
                base = from_checkpoint(
                    ROOT
                    / f"docs/evidence/observed_learning/mnist_confirmation/model_{case['seed']}_{width}_{case['step']}.pt"
                )
                f0 = cache(base, x)[0]
                p0 = f0.softmax(-1)
                target = p0.clone()
                target[y == case["focal"]] = labels[y == case["focal"]].double()
                demand = target - p0
                arms = {}
                for arm in ["none", "peer", "signed"]:
                    s = signed_expand(base) if arm == "signed" else copy_state(base)
                    if arm != "none":
                        ids = y != case["focal"]
                        s = evolve(s, x[ids], labels[ids], 20, 0.05)
                    offset = f0 - cache(s, x)[0]
                    arms[arm] = dict(
                        quadratic=quadratic_gains(s, x, demand, p0, 20, 0.05),
                        tangent_ce=curve(linearize(s), x, target, offset),
                        bilinear_ce=curve(freeze_gates(s), x, target, offset, decode=True),
                        actual=curve(s, x, target, offset),
                    )
                append(out / "results.jsonl", dict(**case, width=width, arms=arms))
                print(case["seed"], width, flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
