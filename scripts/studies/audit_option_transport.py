"""Signed option transport under the training gradient, on a fixed common demand."""

import copy
import json

import numpy as np
import torch
from explore_conditional_editability import ROOT, Net

OUT = ROOT / "docs/theory/option_transport"
SOURCE = ROOT / "docs/theory/option_mismatch_tracking"
torch.set_num_threads(2)


def strength(model, x, d, create_graph=False, fixed_gate=None):
    if fixed_gate is None:
        output = model(x)
    else:
        hidden = (x @ model.w.T + model.b) * fixed_gate
        output = hidden @ model.a.T / np.sqrt(len(model.w)) + model.c
    scalar = (output * d).sum() / np.sqrt(len(x))
    derivatives = torch.autograd.grad(scalar, tuple(model.parameters()), create_graph=create_graph)
    return sum(v.square().sum() for v in derivatives)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = torch.load(ROOT / "data/mnist/prepared.pt", weights_only=True)
    x, y = data["xtrain"][:8192].double(), data["train_labels"][:8192]
    ids = torch.randperm(5000, generator=torch.Generator().manual_seed(281920))[:128]
    qx, qy = data["xval"][ids].double(), data["val_labels"][ids]
    rows = []
    for seed in range(281801, 281809):
        for step in (0, 800):
            models = {}
            for width in (32, 128):
                torch.manual_seed(seed)
                model = Net(width).double()
                if step:
                    model.load_state_dict(
                        torch.load(SOURCE / f"final_{seed}_{width}.pt", weights_only=True)
                    )
                models[width] = model
            with torch.no_grad():
                common = (
                    torch.nn.functional.one_hot(qy, 10)
                    - sum(m(qx).softmax(-1) for m in models.values()) / 2
                )
                common /= common.norm()
            for width, model in models.items():
                model.zero_grad()
                for start in range(0, len(x), 1024):
                    loss = torch.nn.functional.cross_entropy(
                        model(x[start : start + 1024]), y[start : start + 1024], reduction="sum"
                    ) / len(x)
                    loss.backward()
                gradient = [p.grad.detach().clone() for p in model.parameters()]
                k = strength(model, qx, common, create_graph=True)
                sensitivity = torch.autograd.grad(k, tuple(model.parameters()), allow_unused=True)
                velocity = -sum(
                    (v * g).sum() for v, g in zip(sensitivity, gradient) if v is not None
                )
                incoming_load = (
                    (
                        (gradient[0].square().sum(1) + gradient[1].square())
                        / (model.w.square().sum(1) + model.b.square()).clamp_min(1e-12)
                    )
                    .mean()
                    .sqrt()
                )
                observations = []
                original_gate = (qx @ model.w.T + model.b > 0).detach()
                for h in (0.005, 0.05):
                    moved = copy.deepcopy(model)
                    with torch.no_grad():
                        for p, g in zip(moved.parameters(), gradient):
                            p.add_(g, alpha=-h)
                    actual = float(strength(moved, qx, common).detach())
                    fixed = float(strength(moved, qx, common, fixed_gate=original_gate).detach())
                    changed_gate = (qx @ moved.w.T + moved.b > 0).detach()
                    observations.append(
                        dict(
                            h=h,
                            actual_change=actual - float(k.detach()),
                            predicted_change=h * float(velocity.detach()),
                            smooth_finite_change=fixed - float(k.detach()),
                            gate_change=actual - fixed,
                            gate_crossing_fraction=float(
                                (changed_gate != original_gate).double().mean()
                            ),
                        )
                    )
                rows.append(
                    dict(
                        seed=seed,
                        width=width,
                        step=step,
                        kappa=float(k.detach()),
                        velocity=float(velocity.detach()),
                        relative_velocity=float((velocity / k).detach()),
                        incoming_relative_load=float(incoming_load.detach()),
                        audits=observations,
                    )
                )
    (OUT / "results.json").write_text(
        json.dumps(
            dict(
                results=rows,
                note="Demands fixed during each comparison, common across widths, derived from current held-out residuals. Transport derivative includes smooth terms; ReLU gate crossings are audited separately.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
