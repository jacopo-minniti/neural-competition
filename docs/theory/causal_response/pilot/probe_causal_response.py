"""Matched-response pressure interventions and common-output correction probes.

Forecasts precede outcomes. The anchor is a frozen output offset, not a parameter
update: it gives all post-pressure states identical predictions, demand and loss
curvature while retaining their different Jacobians and optimizer geometry.
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
    loss,
    option_value,
    power,
    velocity,
)
from trace_observed_learning import ROOT, append


def physical(state):
    return {k: v for k, v in state.items() if not k.startswith("reference_")}


def evolve(state, x, target, steps, rate, offset=None):
    s = copy_state(state)
    for _ in range(steps):
        cached = cache(s, x)
        f = cached[0] if offset is None else cached[0] + offset
        e = target - f.softmax(-1)
        v = velocity(s, x, e, cached)
        for key in v:
            s[key] += rate * v[key]
    return s


def measure(s, x, y, vx, vy, focal, demand, p0):
    f, vf = cache(s, x)[0], cache(s, vx)[0]
    return dict(
        loss=loss(f, y),
        focal_loss=loss(f[y == focal], y[y == focal]),
        validation=loss(vf, vy),
        focal_validation=loss(vf[vy == focal], vy[vy == focal]),
        fixed_power=power(s, x, demand),
        **option_value(s, x, demand, p0),
    )


def branches(base, x, y, vx, vy, focal, steps, recovery, rate, predicted):
    f0, vf0 = cache(base, x)[0], cache(base, vx)[0]
    p0, vp0 = f0.softmax(-1), vf0.softmax(-1)
    labels, vlabels = torch.nn.functional.one_hot(y, 10), torch.nn.functional.one_hot(vy, 10)
    # Original labels for the focal class; initial predictions as background targets.
    target, vtarget = p0.clone(), vp0.clone()
    target[y == focal] = labels[y == focal].double()
    vtarget[vy == focal] = vlabels[vy == focal].double()
    demand = target - p0
    initial = dict(
        loss=loss(f0, y),
        correction_loss=loss(f0, target),
        correction_validation=loss(vf0, vtarget),
        fixed_power=power(base, x, demand),
        **option_value(base, x, demand, p0),
    )
    results = {}
    for name in ["none", "full", "peer", "reserve"]:
        state = expand(base) if name == "reserve" else copy_state(base)
        before = cache(state, x)[0]
        assert float((before - f0).abs().max()) < 1e-10
        assert abs(power(state, x, demand) - initial["fixed_power"]) < 1e-10
        if name != "none":
            use = y != focal if name in ["peer", "reserve"] else slice(None)
            state = evolve(
                freeze_gates(state) if predicted else state, x[use], labels[use], steps, rate
            )
            state = physical(state)
        row = measure(state, x, y, vx, vy, focal, demand, p0)
        # Common predictions at the start of correction, including on validation.
        offset = f0 - cache(state, x)[0]
        voffset = vf0 - cache(state, vx)[0]
        start = freeze_gates(state) if predicted else state
        corrected = physical(evolve(start, x, target, recovery, rate, offset))
        row["anchor_gain"] = initial["correction_loss"] - loss(
            cache(corrected, x)[0] + offset, target
        )
        row["anchor_validation_gain"] = initial["correction_validation"] - loss(
            cache(corrected, vx)[0] + voffset, vtarget
        )
        # Unmodified original-task recovery is a separate outcome, never conflated
        # with the common-output mechanism probe.
        recovered = physical(evolve(start, x, labels, recovery, rate))
        row["native_final_loss"] = loss(cache(recovered, x)[0], y)
        row["native_final_focal_loss"] = loss(cache(recovered, x[y == focal])[0], y[y == focal])
        results[name] = row
    return dict(initial=initial, arms=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="docs/theory/observed_learning/mnist_confirmation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", required=True, help="JSON list of seed, step, focal triples")
    parser.add_argument("--pressure", type=int, default=20)
    parser.add_argument("--recovery", type=int, default=20)
    args = parser.parse_args()
    src, out = ROOT / args.input, ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists(), "Preserve previous experiment"
    cases = json.loads(Path(args.cases).read_text())
    protocol = dict(
        input=args.input,
        cases=cases,
        widths=[32, 128],
        pressure=args.pressure,
        recovery=args.recovery,
        rate=0.05,
        copies=4,
        tau=1,
        model="Fixed-gate bilinear evolution, physical decoding, reset masks at declared phase change",
        anchor="Fixed output offset to pre-pressure predictions; focal hard targets and initial background probabilities",
        sources={},
        checkpoint_hashes={},
    )
    for name in [
        Path(__file__).name,
        "probe_response_feedback.py",
        "trace_observed_learning.py",
        "audit_conditional_editability.py",
        "explore_conditional_editability.py",
    ]:
        path = Path(__file__).with_name(name)
        protocol["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        (out / name).write_bytes(path.read_bytes())
    for case in cases:
        for width in [32, 128]:
            name = f"model_{case['seed']}_{width}_{case['step']}.pt"
            protocol["checkpoint_hashes"][name] = hashlib.sha256(
                (src / name).read_bytes()
            ).hexdigest()
    data_path = ROOT / "data/mnist/prepared.pt"
    protocol["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = torch.load(data_path, weights_only=True)
    x, y = data["xtrain"][:8192].double(), data["train_labels"][:8192]
    vx, vy = data["xval"][:1024].double(), data["val_labels"][:1024]
    torch.set_num_threads(2)
    with torch.no_grad():
        for case in cases:
            for width in [32, 128]:
                base = from_checkpoint(src / f"model_{case['seed']}_{width}_{case['step']}.pt")
                for predicted, file in [(True, "forecasts.jsonl"), (False, "outcomes.jsonl")]:
                    result = branches(
                        base,
                        x,
                        y,
                        vx,
                        vy,
                        case["focal"],
                        args.pressure,
                        args.recovery,
                        0.05,
                        predicted,
                    )
                    append(out / file, dict(**case, width=width, **result))
                print(json.dumps(dict(**case, width=width, complete=True)), flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
