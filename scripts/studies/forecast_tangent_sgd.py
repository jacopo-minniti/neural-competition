"""Prospective nonlinear-loss forecasts on full-data tangent features.

Separate conditional (known batch stream) and independent-stream MC forecasts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from explore_conditional_editability import ROOT, Net
from explore_option_retention import Tangent, evaluate
from torch.nn import functional as F
from track_option_mismatch import evaluate as gradient_evaluate

OUT = ROOT / "docs/theory/tangent_sgd_forecast"
torch.set_num_threads(2)


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def evolve(models, batches, x, y):
    opts = {m: torch.optim.SGD(net.parameters(), lr=0.05) for m, net in models.items()}
    for batch in batches:
        for m, net in models.items():
            opts[m].zero_grad()
            F.cross_entropy(net(x[batch]), y[batch]).backward()
            opts[m].step()


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(281801, 281809)))
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    OUT = ROOT / args.output if not str(args.output).startswith("/") else Path(args.output)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "forecasts.jsonl").exists():
        raise RuntimeError("Preserving existing forecasts; choose a new directory for a rerun.")
    protocol = dict(
        seeds=args.seeds,
        calibration_seeds=list(range(281801, 281805)),
        test_seeds=[s for s in args.seeds if s not in range(281801, 281805)],
        steps=800,
        interval=20,
        rate=0.05,
        independent_streams=4,
        target="wide minus narrow full-training gain",
        note="Forecasts recorded before actual outcomes. Conditional forecast knows only exogenous batch schedule; MC forecast uses independent paired streams. No fitted coefficients.",
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = torch.load(ROOT / "data/mnist/prepared.pt", weights_only=True)
    x, y = data["xtrain"][:8192], data["train_labels"][:8192]
    for seed in protocol["seeds"]:
        models = {}
        for width in (32, 128):
            torch.manual_seed(seed)
            models[width] = Net(width)
        generator = torch.Generator().manual_seed(seed + 500)
        base = {m: evaluate(net, x, y) for m, net in models.items()}
        for step in range(0, 800, 20):
            gradients = {}
            for width, model in models.items():
                gradient_evaluate(model, x, y, gradient=True)
                gradients[width] = sum(float(p.grad.square().sum()) for p in model.parameters())
            actual_batches = [torch.randint(len(x), (128,), generator=generator) for _ in range(20)]
            predictions = []
            for stream in range(5):
                if stream == 0:
                    batches = actual_batches
                else:
                    rng = torch.Generator().manual_seed(seed * 10000 + step * 10 + stream)
                    batches = [torch.randint(len(x), (128,), generator=rng) for _ in range(20)]
                tangent = {m: Tangent(net) for m, net in models.items()}
                evolve(tangent, batches, x, y)
                ending = {m: evaluate(net, x, y) for m, net in tangent.items()}
                predictions.append(
                    dict(
                        stream=stream,
                        narrow_gain=base[32] - ending[32],
                        wide_gain=base[128] - ending[128],
                        difference=(base[128] - ending[128]) - (base[32] - ending[32]),
                    )
                )
            forecast = dict(
                seed=seed,
                start=step,
                loss_narrow=base[32],
                loss_wide=base[128],
                gradient_narrow=gradients[32],
                gradient_wide=gradients[128],
                predictions=predictions,
                conditional=predictions[0]["difference"],
                independent=float(np.mean([r["difference"] for r in predictions[1:]])),
            )
            append(OUT / "forecasts.jsonl", forecast)
            evolve(models, actual_batches, x, y)
            ending = {m: evaluate(net, x, y) for m, net in models.items()}
            outcome = dict(
                seed=seed,
                start=step,
                narrow_gain=base[32] - ending[32],
                wide_gain=base[128] - ending[128],
                difference=(base[128] - ending[128]) - (base[32] - ending[32]),
            )
            append(OUT / "outcomes.jsonl", outcome)
            base = ending
            if step % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            seed=seed,
                            step=step,
                            actual=outcome["difference"],
                            conditional=forecast["conditional"],
                            independent=forecast["independent"],
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
