"""Timing control: add matched-response copies only after peer pressure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from probe_causal_response import evolve, measure, physical, signed_expand
from probe_response_feedback import cache, freeze_gates, from_checkpoint, loss, option_value, power
from trace_observed_learning import ROOT, append


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src, out = ROOT / args.input, ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists()
    cases = json.loads(Path(args.cases).read_text())
    protocol = dict(
        input=args.input,
        cases=cases,
        widths=[32, 128],
        pressure=20,
        recovery=20,
        rate=0.05,
        refresh=10,
        purpose="Late-only signed copies distinguish pressure-history preservation from altered correction dynamics.",
        sources={},
        checkpoint_hashes={},
    )
    for name in [
        Path(__file__).name,
        "probe_causal_response.py",
        "probe_response_feedback.py",
        "trace_observed_learning.py",
        "audit_conditional_editability.py",
    ]:
        path = Path(__file__).with_name(name)
        protocol["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / name).write_bytes(path.read_bytes())
    for c in cases:
        for w in [32, 128]:
            name = f"model_{c['seed']}_{w}_{c['step']}.pt"
            protocol["checkpoint_hashes"][name] = hashlib.sha256(
                (src / name).read_bytes()
            ).hexdigest()
    data_path = ROOT / "data/mnist/prepared.pt"
    protocol["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = torch.load(data_path, weights_only=True)
    x, y = data["xtrain"][:8192].double(), data["train_labels"][:8192]
    vx, vy = data["xval"][:1024].double(), data["val_labels"][:1024]
    labels = torch.nn.functional.one_hot(y, 10)
    torch.set_num_threads(2)
    with torch.no_grad():
        for case in cases:
            for width in [32, 128]:
                base = from_checkpoint(src / f"model_{case['seed']}_{width}_{case['step']}.pt")
                f0 = cache(base, x)[0]
                p0 = f0.softmax(-1)
                target = p0.clone()
                target[y == case["focal"]] = labels[y == case["focal"]].double()
                demand = target - p0
                initial = dict(
                    correction_loss=loss(f0, target), **option_value(base, x, demand, p0)
                )
                for predicted, file in [(True, "forecasts.jsonl"), (False, "outcomes.jsonl")]:
                    refresh = 10 if predicted else 0
                    use = y != case["focal"]
                    post = physical(
                        evolve(
                            freeze_gates(base) if predicted else base,
                            x[use],
                            labels[use],
                            20,
                            0.05,
                            refresh=refresh,
                        )
                    )
                    state = signed_expand(post)
                    assert abs(power(state, x, demand) - power(post, x, demand)) < 1e-10
                    assert float((cache(state, x)[0] - cache(post, x)[0]).abs().max()) < 1e-10
                    row = measure(state, x, y, vx, vy, case["focal"], demand, p0)
                    offset = f0 - cache(state, x)[0]
                    start = freeze_gates(state) if predicted else state
                    trace = {}
                    evolve(start, x, target, 20, 0.05, offset, refresh=refresh, trace=trace)
                    row["anchor_gains"] = {
                        k: initial["correction_loss"] - v for k, v in trace.items()
                    }
                    native = physical(evolve(start, x, labels, 20, 0.05, refresh=refresh))
                    row["native_final_loss"] = loss(cache(native, x)[0], y)
                    append(
                        out / file,
                        dict(
                            **case,
                            width=width,
                            refresh=refresh,
                            initial=initial,
                            arms=dict(late_signed=row),
                        ),
                    )
                print(case["seed"], width, flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
