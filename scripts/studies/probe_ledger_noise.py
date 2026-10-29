"""One-step batch-size intervention at fixed ordinary-training MNIST states.

No continuation forecast is fitted. Outcomes are full-training loss changes,
with identical initial parameters, loss, Jacobian and step size across arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from explore_conditional_editability import Net
from trace_observed_learning import ROOT, append, forward, gradients, state, tangent

from neural_competition.learning_ledger import cross_entropy_ledger, dot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    source, out = ROOT / args.input, ROOT / args.output
    source_protocol = json.loads((source / "protocol.json").read_text())
    assert (source / "complete.json").exists()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Refusing to overwrite prior probe")
    protocol = dict(
        source=str(source.relative_to(ROOT)),
        seeds=source_protocol["seeds"],
        widths=[32, 128],
        steps=[100, 800],
        batch_sizes=[128, 512, 2048, "full"],
        repeats=args.repeats,
        rate=0.05,
        sampling="with replacement, nested within repetition",
        precision="float64 analytic SGD from identical saved float32 ordinary checkpoints",
        hypotheses=[
            "Larger batches reduce convex cost and increase expected one-step gain.",
            "Feature nonlinearity may limit the quadratic-noise explanation.",
        ],
        scope="Immediate counterfactual, not long-run batch-size or generalization claim",
        checkpoint_hashes={},
        sources={},
    )
    for seed in protocol["seeds"]:
        for width in protocol["widths"]:
            for step in protocol["steps"]:
                path = source / f"model_{seed}_{width}_{step}.pt"
                protocol["checkpoint_hashes"][path.name] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    for path in [
        Path(__file__),
        Path(__file__).with_name("trace_observed_learning.py"),
        ROOT / "src/neural_competition/learning_ledger.py",
    ]:
        protocol["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        (out / path.name).write_bytes(path.read_bytes())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = torch.load(ROOT / "data/mnist/prepared.pt", weights_only=True)
    x, y = data["xtrain"][:8192].double(), data["train_labels"][:8192]
    torch.set_num_threads(2)
    with torch.no_grad():
        for seed in protocol["seeds"]:
            for step in protocol["steps"]:
                for width in protocol["widths"]:
                    model = Net(width)
                    model.load_state_dict(
                        torch.load(source / f"model_{seed}_{width}_{step}.pt", weights_only=True)
                    )
                    old = state(model)
                    cached = forward(old, x)
                    full, _ = gradients(old, x, y, cached)
                    p = cached[0].softmax(-1)
                    generator = torch.Generator().manual_seed(seed + step + 910000)
                    for repeat in range(args.repeats):
                        ids = torch.randint(len(x), (2048,), generator=generator)
                        arms = [128, 512, 2048] + (["full"] if repeat == 0 else [])
                        for size in arms:
                            if size == "full":
                                grad = full
                            else:
                                selected = ids[:size]
                                grad, _ = gradients(
                                    old,
                                    x[selected],
                                    y[selected],
                                    tuple(v[selected] for v in cached),
                                )
                            delta = tuple(-0.05 * v for v in grad)
                            new = tuple(v + dv for v, dv in zip(old, delta))
                            linear = tangent(old, x, cached, delta)
                            row = cross_entropy_ledger(cached[0], forward(new, x)[0], y, linear)
                            row.update(
                                seed=seed,
                                width=width,
                                step=step,
                                repeat=repeat,
                                size=size,
                                signal=0.05 * float(dot(full, full)),
                                quadratic_cost=float(
                                    0.5
                                    * (
                                        (p * linear.square()).sum(1) - (p * linear).sum(1).square()
                                    ).mean()
                                ),
                            )
                            append(out / "outcomes.jsonl", row)
                print(
                    json.dumps(dict(seed=seed, step=step, status="paired probes complete")),
                    flush=True,
                )
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
