"""Fresh ordinary SGD checkpoints for locked causal response tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from explore_conditional_editability import Net
from trace_observed_learning import ROOT, append


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=284101)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists()
    sources = {}
    for path in [Path(__file__), Path(__file__).with_name("explore_conditional_editability.py")]:
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / path.name).write_bytes(path.read_bytes())
    data_path = ROOT / "data/mnist/prepared.pt"
    protocol = dict(
        seeds=list(range(args.seed, args.seed + args.count)),
        widths=[32, 128],
        steps=100,
        checkpoints=[20, 100],
        rate=0.05,
        batch=128,
        training_examples=8192,
        sources=sources,
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        note="Fresh independent initialization seeds; ordinary original-label MNIST SGD; no interventions yet.",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = torch.load(data_path, weights_only=True)
    x, y = data["xtrain"][:8192], data["train_labels"][:8192]
    vx, vy = data["xval"][:1024], data["val_labels"][:1024]
    torch.set_num_threads(2)
    for seed in protocol["seeds"]:
        for width in protocol["widths"]:
            torch.manual_seed(seed)
            model = Net(width)
            opt = torch.optim.SGD(model.parameters(), lr=0.05)
            generator = torch.Generator().manual_seed(seed + 500)
            for step in range(1, 101):
                ids = torch.randint(len(x), (128,), generator=generator)
                opt.zero_grad()
                torch.nn.functional.cross_entropy(model(x[ids]), y[ids]).backward()
                opt.step()
                if step in protocol["checkpoints"]:
                    torch.save(model.state_dict(), out / f"model_{seed}_{width}_{step}.pt")
                    with torch.no_grad():
                        append(
                            out / "checkpoints.jsonl",
                            dict(
                                seed=seed,
                                width=width,
                                step=step,
                                loss=float(torch.nn.functional.cross_entropy(model(x), y)),
                                validation=float(torch.nn.functional.cross_entropy(model(vx), vy)),
                            ),
                        )
        print(seed, flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
