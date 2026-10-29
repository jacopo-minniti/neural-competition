"""Preserve initialization tangent features, covering the early commitment window."""

import json

import torch
from explore_conditional_editability import ROOT, Net
from explore_option_retention import OUT, Tangent, evaluate
from torch.nn import functional as F


def main():
    data = torch.load(ROOT / "data/mnist/prepared.pt", weights_only=True)
    x, y = data["xtrain"][:8192], data["train_labels"][:8192]
    vx, vy = data["xval"][:1024], data["val_labels"][:1024]
    rows = []
    for seed in (281805, 281806):
        models = {}
        for width in (32, 128):
            torch.manual_seed(seed)
            models[width] = Tangent(Net(width))
        opts = {m: torch.optim.SGD(net.parameters(), lr=0.05) for m, net in models.items()}
        generator = torch.Generator().manual_seed(seed + 500)
        for step in range(2001):
            if step in (800, 2000):
                for width, net in models.items():
                    rows.append(
                        dict(
                            seed=seed,
                            width=width,
                            step=step,
                            train_loss=evaluate(net, x, y),
                            validation_loss=evaluate(net, vx, vy),
                        )
                    )
            if step < 2000:
                batch = torch.randint(len(x), (128,), generator=generator)
                for width, net in models.items():
                    opts[width].zero_grad()
                    F.cross_entropy(net(x[batch]), y[batch]).backward()
                    opts[width].step()
    (OUT / "initial_freeze.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
