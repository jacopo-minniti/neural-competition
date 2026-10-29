"""Task-structured stress test: an easy shared cue with visual exceptions.

No teacher network or assigned neurons. Early/late cue masking probes commitment.
"""

import json

import numpy as np
import torch
from explore_conditional_editability import ROOT, Net
from explore_option_retention import Tangent
from torch import nn
from torch.nn import functional as F

OUT = ROOT / "docs/theory/shortcut_commitment"
torch.set_num_threads(2)


class ContextNet(Net):
    def __init__(self, width):
        super().__init__(width)
        self.w = nn.Parameter(torch.randn(width, 794) / np.sqrt(794))


def dataset(x, y, seed):
    rng = torch.Generator().manual_seed(seed)
    exceptional = torch.rand(len(y), generator=rng) < 0.05
    other = (y + torch.randint(1, 10, (len(y),), generator=rng)) % 10
    cue = torch.where(exceptional, other, y)
    return torch.cat([x, 8 * F.one_hot(cue, 10).float()], 1), exceptional


def evaluate(model, x, y, exceptional):
    with torch.no_grad():
        losses = torch.cat(
            [
                F.cross_entropy(model(x[i : i + 1024]), y[i : i + 1024], reduction="none")
                for i in range(0, len(x), 1024)
            ]
        )
    return dict(
        loss=float(losses.mean()),
        exception_loss=float(losses[exceptional].mean()),
        regular_loss=float(losses[~exceptional].mean()),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        seeds=[281901, 281902],
        widths=[32, 128],
        rate=0.05,
        steps=2000,
        exception_probability=0.05,
        cue_amplitude=8,
        arms=["ordinary", "early_mask", "late_mask", "no_cue", "tangent_after100"],
        note="Protocol saved before outcomes. Masking covers steps 0-99 or 700-799. Tangent branch starts from ordinary step100. No parameter roles or teacher are given.",
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = torch.load(ROOT / "data/mnist/prepared.pt", weights_only=True)
    x, ex = dataset(data["xtrain"][:8192], data["train_labels"][:8192], 281900)
    y = data["train_labels"][:8192]
    vx, vex = dataset(data["xval"][:2048], data["val_labels"][:2048], 281903)
    vy = data["val_labels"][:2048]
    rows = []
    for seed in protocol["seeds"]:
        models = {}
        for width in (32, 128):
            torch.manual_seed(seed)
            original = ContextNet(width).state_dict()
            for arm in ("ordinary", "early_mask", "late_mask", "no_cue"):
                models[width, arm] = ContextNet(width)
                models[width, arm].load_state_dict(original)
        opts = {k: torch.optim.SGD(net.parameters(), lr=0.05) for k, net in models.items()}
        rng = torch.Generator().manual_seed(seed + 1)
        for step in range(2001):
            if step == 100:
                for width in (32, 128):
                    models[width, "tangent_after100"] = Tangent(models[width, "ordinary"])
                    opts[width, "tangent_after100"] = torch.optim.SGD(
                        models[width, "tangent_after100"].parameters(), lr=0.05
                    )
            if step in (0, 100, 400, 800, 2000):
                for (width, arm), net in models.items():
                    px, pv = x, vx
                    if arm == "no_cue":
                        px = x.clone()
                        pv = vx.clone()
                        px[:, -10:] = 0
                        pv[:, -10:] = 0
                    row = dict(
                        seed=seed,
                        width=width,
                        arm=arm,
                        step=step,
                        train=evaluate(net, px, y, ex),
                        validation=evaluate(net, pv, vy, vex),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                (OUT / "results.json").write_text(
                    json.dumps(dict(protocol=protocol, results=rows), indent=2) + "\n"
                )
            if step < 2000:
                batch = torch.randint(len(x), (128,), generator=rng)
                for (width, arm), net in models.items():
                    bx = x[batch]
                    if (
                        arm == "no_cue"
                        or (arm == "early_mask" and step < 100)
                        or (arm == "late_mask" and 700 <= step < 800)
                    ):
                        bx = bx.clone()
                        bx[:, -10:] = 0
                    opts[width, arm].zero_grad()
                    F.cross_entropy(net(bx), y[batch]).backward()
                    opts[width, arm].step()


if __name__ == "__main__":
    main()
