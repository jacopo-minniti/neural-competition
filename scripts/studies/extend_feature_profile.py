#!/usr/bin/env python3
"""Extend five preselected paired starts after stopping all readout interventions."""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch

from neural_competition.tasks import paired_gaussian_task
from neural_competition.training import advance, components

ROOT = Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    folder = ROOT / "runs/feature_profile/extension"
    folder.mkdir(exist_ok=True)
    sources = [
        Path(__file__),
        ROOT / "src/neural_competition/training.py",
        ROOT / "src/neural_competition/tasks.py",
        ROOT / "src/neural_competition/residual_growth.py",
        ROOT / "docs/theory/feature_profile_protocol.md",
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (folder / "sources").mkdir(exist_ok=True)
    for p in sources:
        (folder / "sources" / (hashes[str(p.relative_to(ROOT))] + ".txt")).write_bytes(
            p.read_bytes()
        )
    for kind, opt in enumerate(["sgd", "momentum", "adam"]):
        if (folder / f"{opt}.json").exists():
            continue
        path = ROOT / f"runs/feature_profile/main/d9027_a30_r-3_{opt}_none.pt"
        saved = torch.load(path, weights_only=True)
        keep = [i for i, m in enumerate(saved["models"]) if m["seed"] < 105]
        models = [saved["models"][i] for i in keep]
        state = saved["states"][10000]
        p, first, second = [state[k][keep].clone() for k in ["parameters", "first", "second"]]
        norm = torch.tensor([[math.sqrt(m["width"])] for m in models], dtype=torch.float64)
        seeds = torch.tensor([m["seed"] - 100 for m in models])
        rates = torch.ones_like(p)
        # Padding/constant feature are frozen; all genuine neurons now use ordinary rates.
        for i, m in enumerate(models):
            rates[i, m["width"] :] = 0
        rngs = [torch.Generator().manual_seed(95000 + s) for s in range(100, 105)]
        for rng in rngs:
            torch.randint(4096, (10000, 128), generator=rng)
        d = paired_gaussian_task(30, -3, 9027)
        rows = []
        for t in range(10000, 100001, 1000):
            if t in [10000, 30000, 100000]:
                pred = components(p, d["xtest"][None].expand(len(p), -1, -1), norm).sum(-1)
                losses = 0.5 * (pred - d["ytest"]).square().mean(1)
                for m, loss in zip(models, losses):
                    rows.append(dict(optimizer=opt, **m, step=t, test_loss=float(loss)))
            if t == 100000:
                break
            ids = torch.stack([torch.randint(4096, (1000, 128), generator=rng) for rng in rngs], 1)
            p, first, second = advance(
                p,
                first,
                second,
                d["xtrain"],
                d["ytrain"],
                ids,
                seeds,
                norm,
                rates,
                [0.3, 0.03, 0.003][kind],
                kind,
                False,
                t,
            )
        pd.DataFrame(rows).to_csv(folder / f"{opt}.csv", index=False, float_format="%.17g")
        torch.save(
            dict(parameters=p, first=first, second=second, models=models), folder / f"{opt}.pt"
        )
        (folder / f"{opt}.json").write_text(
            json.dumps(
                dict(
                    sources=hashes,
                    initial_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    seed_start=100,
                    seeds=5,
                    initial_step=10000,
                    final_step=100000,
                    protocol="All interventions stop at 10k, including fast readout rate; preserve optimizer state",
                ),
                indent=2,
            )
            + "\n"
        )
        print(
            opt,
            pd.DataFrame(rows).query("step==100000").groupby("arm").test_loss.mean().to_dict(),
            flush=True,
        )


if __name__ == "__main__":
    main()
