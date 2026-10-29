"""Can preserving the current tangent features rescue ordinary narrow SGD?"""

import json

import numpy as np
import torch
from explore_conditional_editability import ROOT, Net
from torch import nn
from torch.nn import functional as F

OUT = ROOT / "docs/theory/option_retention"
SOURCE = ROOT / "docs/theory/option_mismatch_tracking"
torch.set_num_threads(2)


class Tangent(nn.Module):
    """Same initial function and Jacobian, permanently fixed derivative features."""

    def __init__(self, model):
        super().__init__()
        for name, parameter in model.named_parameters():
            self.register_buffer(name + "0", parameter.detach().clone())
            self.register_parameter(name, nn.Parameter(torch.zeros_like(parameter)))

    def forward(self, x):
        with torch.no_grad():
            z = F.linear(x, self.w0, self.b0)
            h = z.relu()
            initial = F.linear(h, self.a0) / np.sqrt(len(self.w0)) + self.c0
        delta_h = (z > 0) * F.linear(x, self.w, self.b)
        return (
            initial
            + (F.linear(delta_h, self.a0) + F.linear(h, self.a)) / np.sqrt(len(self.w0))
            + self.c
        )


def evaluate(model, x, y):
    with torch.no_grad():
        return sum(
            float(F.cross_entropy(model(x[i : i + 1024]), y[i : i + 1024], reduction="sum"))
            for i in range(0, len(x), 1024)
        ) / len(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        seeds=[281805, 281806],
        widths=[32, 128],
        start=800,
        continuation=7200,
        arms=["ordinary", "frozen_tangent"],
        rate=0.05,
        batch=128,
        checkpoints=[0, 20, 100, 400, 1200, 3200, 7200],
        note="Saved before branch results. Original task, labels, optimizer rate and matched minibatches; tangent model freezes all derivative features.",
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = torch.load(ROOT / "data/mnist/prepared.pt", weights_only=True)
    x, y = data["xtrain"][:8192], data["train_labels"][:8192]
    vx, vy = data["xval"][:1024], data["val_labels"][:1024]
    rows = []
    for seed in protocol["seeds"]:
        models = {}
        for width in protocol["widths"]:
            net = Net(width)
            net.load_state_dict(torch.load(SOURCE / f"final_{seed}_{width}.pt", weights_only=True))
            models[width, "ordinary"] = net
            models[width, "frozen_tangent"] = Tangent(net)
        opts = {key: torch.optim.SGD(model.parameters(), lr=0.05) for key, model in models.items()}
        generator = torch.Generator().manual_seed(seed + 500)
        for _ in range(800):
            torch.randint(len(x), (128,), generator=generator)
        for step in range(7201):
            if step in protocol["checkpoints"]:
                for (width, arm), model in models.items():
                    row = dict(
                        seed=seed,
                        width=width,
                        arm=arm,
                        step=800 + step,
                        train_loss=evaluate(model, x, y),
                        validation_loss=evaluate(model, vx, vy),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                (OUT / "results.json").write_text(
                    json.dumps(dict(protocol=protocol, results=rows), indent=2) + "\n"
                )
            if step < 7200:
                batch = torch.randint(len(x), (128,), generator=generator)
                for key, model in models.items():
                    opts[key].zero_grad()
                    F.cross_entropy(model(x[batch]), y[batch]).backward()
                    opts[key].step()
        for (width, arm), model in models.items():
            torch.save(model.state_dict(), OUT / f"final_{seed}_{width}_{arm}.pt")


if __name__ == "__main__":
    main()
