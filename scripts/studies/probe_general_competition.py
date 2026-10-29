import json
import math
from pathlib import Path

import torch

from neural_competition.training import components

torch.set_num_threads(1)
out = Path("runs/general_competition")
out.mkdir(parents=True, exist_ok=True)
rows = []
for distribution in ["gaussian", "mixture", "uniform"]:
    g = torch.Generator().manual_seed(7341)
    x = torch.randn(8192, 20, generator=g, dtype=torch.float64)
    if distribution == "mixture":
        x = 0.6 * x + 0.8 * (2 * torch.randint(2, x.shape, generator=g) - 1)
    if distribution == "uniform":
        x = math.sqrt(3) * (2 * torch.rand(8192, 20, generator=g, dtype=torch.float64) - 1)
    v = torch.zeros(8, 20, dtype=torch.float64)
    for k in range(4):
        v[2 * k, 2 * k] = v[2 * k + 1, 2 * k] = math.cos(math.pi / 6)
        v[2 * k, 2 * k + 1] = 0.5
        v[2 * k + 1, 2 * k + 1] = -0.5
    beta = torch.tensor([1.0, -3.0] * 4, dtype=torch.float64)
    y = (x @ v.T).relu() @ beta
    y = y / y[:4096].std()
    from neural_competition.sign_survival import advance_protected

    for kind, opt in enumerate(["sgd", "momentum", "adam"]):
        models = [
            (seed, arm)
            for seed in range(300, 310)
            for arm in ["ordinary8", "ordinary32", "protected8"]
        ]
        p = torch.zeros(30, 32, 22, dtype=torch.float64)
        rates = torch.ones_like(p)
        norm = torch.tensor(
            [[math.sqrt(32 if arm == "ordinary32" else 8)] for seed, arm in models],
            dtype=torch.float64,
        )
        for i, (seed, arm) in enumerate(models):
            w = 32 if arm == "ordinary32" else 8
            p[i, :w, :20] = torch.randn(
                w, 20, generator=torch.Generator().manual_seed(93000 + seed), dtype=torch.float64
            ) / math.sqrt(20)
            a = 0.1 * torch.randn(
                w, generator=torch.Generator().manual_seed(94000 + seed), dtype=torch.float64
            )
            s = torch.ones(w)
            s[: w // 2] = -1
            s = s[torch.randperm(w, generator=torch.Generator().manual_seed(96000 + seed))]
            p[i, :w, -1] = a.abs() * s
            rates[i, w:] = 0
        signs = p[:, :, -1].sign()
        first = torch.zeros_like(p)
        second = first.clone()
        rngs = [torch.Generator().manual_seed(95000 + s) for s in range(300, 310)]
        seeds = torch.tensor([s - 300 for s, a in models])
        protect = torch.tensor([a == "protected8" for s, a in models])
        for step in range(0, 10000, 100):
            ids = torch.stack([torch.randint(4096, (100, 128), generator=g) for g in rngs], 1)
            p, first, second, *_ = advance_protected(
                p,
                first,
                second,
                x[:4096],
                y[:4096],
                ids,
                seeds,
                norm,
                rates,
                [0.3, 0.03, 0.003][kind],
                kind,
                step,
                signs,
                protect,
                torch.full((30,), -1),
            )
        pred = components(p, x[4096:][None].expand(30, -1, -1), norm).sum(-1)
        loss = (pred - y[4096:]).square().mean(1)
        for i, (seed, arm) in enumerate(models):
            rows.append(
                dict(
                    distribution=distribution,
                    optimizer=opt,
                    seed=seed,
                    arm=arm,
                    mse=float(loss[i]),
                    positive=int((p[i, :, -1] > 0).sum()),
                )
            )
        torch.save(dict(parameters=p, models=models, x=x, y=y), out / f"{distribution}_{opt}.pt")
        print(
            distribution,
            opt,
            {
                a: float(loss[[j for j, (s, b) in enumerate(models) if a == b]].mean())
                for a in ["ordinary8", "ordinary32", "protected8"]
            },
            flush=True,
        )
(out / "results.json").write_text(json.dumps(rows, indent=2))
