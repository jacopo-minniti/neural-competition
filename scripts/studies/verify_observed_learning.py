"""Independent uninstrumented SGD replay and historical-loss re-evaluation."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import torch
from explore_conditional_editability import Net
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/theory/observed_learning"


def main():
    torch.set_num_threads(2)
    data = torch.load(ROOT / "data/mnist/prepared.pt", weights_only=True)
    x, y = data["xtrain"][:8192], data["train_labels"][:8192]
    result = dict(
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            platform=platform.platform(),
        ),
        replay=[],
    )
    for name in ["mnist", "mnist_confirmation"]:
        folder = OUT / name
        protocol = json.loads((folder / "protocol.json").read_text())
        for seed in protocol["seeds"]:
            for width in protocol["widths"]:
                torch.manual_seed(seed)
                model = Net(width)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                generator = torch.Generator().manual_seed(seed + 500)
                for _ in range(800):
                    ids = torch.randint(len(x), (128,), generator=generator)
                    optimizer.zero_grad()
                    F.cross_entropy(model(x[ids]), y[ids]).backward()
                    optimizer.step()
                expected = torch.load(folder / f"model_{seed}_{width}_800.pt", weights_only=True)
                error = max(
                    float((v - expected[k]).abs().max()) for k, v in model.state_dict().items()
                )
                assert error == 0
                result["replay"].append(
                    dict(cohort=name, seed=seed, width=width, max_parameter_error=error)
                )
        print(name, "uninstrumented replay exact", flush=True)
    prior = ROOT / "docs/theory/option_mismatch_tracking"
    rows = [json.loads(line) for line in (prior / "forecasts.jsonl").read_text().splitlines()]
    errors = []
    with torch.no_grad():
        for r in rows:
            if r["step"] != 800:
                continue
            for width, side in [(32, "narrow"), (128, "wide")]:
                model = Net(width)
                model.load_state_dict(
                    torch.load(prior / f"final_{r['seed']}_{width}.pt", weights_only=True)
                )
                value = float(F.cross_entropy(model.double()(x.double()), y))
                errors.append(abs(value - r[side]["full_loss"]))
    result["historical_mnist_max_loss_difference"] = max(errors)
    assert max(errors) < 1e-7
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(replayed=len(result["replay"]), historical_max=max(errors))), flush=True)


if __name__ == "__main__":
    main()
