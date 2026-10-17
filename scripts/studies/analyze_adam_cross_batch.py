#!/usr/bin/env python3
"""Quick raw-gradient and Adam-direction batch transfer from saved trajectories."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from run_trajectory import evaluate

from neural_competition.distributions import seed_mean_interval
from neural_competition.model import ShallowReLU, loss
from neural_competition.neurons import candidate_descent, gradient_masses, step_transfer_masses
from neural_competition.trajectory import adam_step, batch_gradient

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/pier_study/robustness/k8_teacher2027"
OUT = ROOT / "runs/pier_study/adam_cross_batch"
WIDTHS = [8, 16, 32]
PROFILES = ["adam_001", "adam_01"]
PROBES = 128
PROBE_SEED = 20261021
METRICS = [
    "train_mse",
    "updates",
    "P",
    "N",
    "p_negative",
    "A",
    "pair_mean",
    "step_P",
    "step_N",
    "step_p_negative",
    "step_global_N",
    "step_net",
    "step_conflict_share",
    "expected_gain",
    "expected_harm",
    "source_harm_fraction",
]


def select_state(p, first, second, i, model, updates, comparison):
    w = model["width"]
    return dict(
        **model,
        updates=updates,
        comparison=comparison,
        parameters=p[i, :w].clone(),
        first=first[i, :w].clone(),
        second=second[i, :w].clone(),
    )


def states(folder, data):
    config = json.loads((folder / "config.json").read_text())
    models = config["models"]
    chosen = [i for i, m in enumerate(models) if m["width"] in WIDTHS]
    widths = torch.tensor([m["width"] for m in models])[:, None]
    seed_map = torch.tensor([m["seed"] for m in models])
    norm = widths.double().sqrt().float()
    p = torch.load(folder / "initial.pt", weights_only=True)["parameters"]
    first, second = torch.zeros_like(p), torch.zeros_like(p)
    generators = [torch.Generator().manual_seed(12026 + s) for s in range(5)]
    indices = torch.stack(
        [torch.randint(len(data["x_train"]), (1000, 64), generator=g) for g in generators],
        dim=1,
    )
    selected, nearest, distance = [], {}, {}
    checking = True
    lr = config["lr"]
    for t in range(1, 1001):
        ids = indices[t - 1].index_select(0, seed_map)
        p, first, second = adam_step(
            p, first, second, data["x_train"][ids], data["y_train"][ids], norm, lr, t
        )
        if t in [100, 1000]:
            selected += [
                select_state(p, first, second, i, models[i], t, f"step_{t}") for i in chosen
            ]
        if (checking and (t % 5 == 0 if t <= 200 else t % 25 == 0)) or t == 1000:
            mse = evaluate(p, widths, data["x_train"], data["y_train"])
            if checking:
                for i in chosen:
                    d = abs(np.log(mse[i] / 0.2))
                    if d < distance.get(i, np.inf):
                        distance[i] = d
                        nearest[i] = select_state(p, first, second, i, models[i], t, "mse_0.2")
                        nearest[i]["train_mse"] = float(mse[i])
                checking = not all(mse[i] < 0.18 for i in chosen)
    for state in nearest.values():
        if not 0.18 <= state["train_mse"] <= 0.22:
            raise RuntimeError(f"No early loss match: {state['width']}, {state['seed']}")
        selected.append(state)
    # Replay uses exactly the original paired minibatch streams and training precision.
    saved = pd.read_csv(folder / "losses.csv").query("updates == 1000")
    errors = [
        abs(
            mse[i]
            - saved[(saved.width == m["width"]) & (saved.seed == m["seed"])].train_mse.iloc[0]
        )
        for i, m in enumerate(models)
    ]
    replay_error = float(max(errors))
    assert replay_error < 2e-12, replay_error
    for t in [10000, 50000, 100000]:
        checkpoint = torch.load(folder / "checkpoints" / f"step_{t:06d}.pt", weights_only=True)
        selected += [
            select_state(
                checkpoint["parameters"],
                checkpoint["velocity"],
                checkpoint["second"],
                i,
                models[i],
                t,
                f"step_{t}",
            )
            for i in chosen
        ]
    return selected, config, replay_error


def gradients(p, x, y):
    return batch_gradient(
        p[None].expand(len(x), -1, -1),
        x.double(),
        y.double(),
        torch.full((len(x), 1), len(p) ** 0.5, dtype=torch.float64),
    )


def audit(state, spec, g, h, x, y):
    """Independent autograd and restored torch.optim.Adam at every analyzed state."""
    p = state["parameters"].double()
    model = ShallowReLU(x.shape[-1], len(p)).double()
    with torch.no_grad():
        model.weight.copy_(p[:, :-2])
        model.bias.copy_(p[:, -2])
        model.output.copy_(p[:, -1])
    loss(model, x[0].double(), y[0].double()).backward()
    automatic = model.blocks(tuple(v.grad for v in model.parameters()))
    grad_error = float((automatic - g[0]).abs().max())
    parameter = torch.nn.Parameter(p.clone())
    opt = torch.optim.Adam([parameter], lr=spec["lr"], foreach=False)
    opt.state[parameter].update(
        step=torch.tensor(float(state["updates"])),
        exp_avg=state["first"].double().clone(),
        exp_avg_sq=state["second"].double().clone(),
    )
    parameter.grad = g[0].clone()
    opt.step()
    update_error = float((parameter.detach() - (p - spec["lr"] * h[0])).abs().max())
    assert grad_error < 1e-11 and update_error < 1e-12, (grad_error, update_error)
    return grad_error, update_error


def measure(state, spec, data, indices):
    p = state["parameters"].double()
    x, y = data["x_train"][indices], data["y_train"][indices]
    g = gradients(p, x, y)
    snapshot = dict(
        optimizer=spec,
        step=state["updates"],
        optimizer_before={"exp_avg": state["first"], "exp_avg_sq": state["second"]},
    )
    h = candidate_descent(snapshot, g)
    raw, _ = gradient_masses(g)
    transfer = step_transfer_masses(g, h)
    q = torch.einsum("brd,crd->bc", h, g)
    offdiag = ~torch.eye(PROBES, dtype=torch.bool)
    scale = (g.square().sum((1, 2)).mean() * h.square().sum((1, 2)).mean()).sqrt()
    net = float(q[offdiag].mean() / scale)
    assert abs(net - (transfer["step_P"] - transfer["step_N"])) < 1e-12
    flat = g.flatten(1)
    full = gradients(p, data["x_train"][None], data["y_train"][None])[0]
    mse = float(evaluate(p[None], torch.tensor([[len(p)]]), data["x_train"], data["y_train"])[0])
    record = {k: state[k] for k in ["width", "seed", "updates", "comparison"]}
    record.update(
        **raw,
        **transfer,
        train_mse=mse,
        step_net=net,
        A=float(full.square().sum() / flat.square().sum(1).mean()),
        step_conflict_share=transfer["step_N"] / (transfer["step_P"] + transfer["step_N"]),
        expected_gain=float(spec["lr"] * q[offdiag].mean()),
        expected_harm=float(spec["lr"] * (-q[offdiag]).clamp_min(0).mean()),
        source_harm_fraction=float((q.diag() < 0).double().mean()),
        **dict(
            zip(["gradient_audit_error", "optimizer_audit_error"], audit(state, spec, g, h, x, y))
        ),
    )
    arrays = dict(
        parameters=p,
        first=state["first"],
        second=state["second"],
        gradients=g,
        descent=h,
        raw_dots=flat @ flat.T,
        adam_dots=q,
    )
    return record, arrays


def summaries(frame):
    rows, effects = [], []
    for identity, part in frame.groupby(["profile", "comparison", "width"]):
        row = dict(zip(["profile", "comparison", "width"], identity))
        row["n_seeds"] = len(part)
        for metric in METRICS:
            mean, low, high = seed_mean_interval(part[metric])
            row.update({metric: mean, metric + "_low": low, metric + "_high": high})
        rows.append(row)
    for identity, part in frame.groupby(["profile", "comparison"]):
        for width in [16, 32]:
            for metric in METRICS:
                pair = part.pivot(index="seed", columns="width", values=metric)
                mean, low, high = seed_mean_interval(pair[width] - pair[8])
                effects.append(
                    dict(
                        profile=identity[0],
                        comparison=identity[1],
                        width=width,
                        baseline=8,
                        metric=metric,
                        mean=mean,
                        low=low,
                        high=high,
                        n_seeds=len(pair),
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(effects)


def plots(summary):
    colors = {8: "#2563eb", 16: "#df6b20", 32: "#129177"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    metrics = ["step_p_negative", "step_P", "step_N"]
    titles = [
        "Harmful Adam batch directions (%)",
        "Positive neuron transfer P",
        "Negative neuron transfer N",
    ]
    for row, profile in enumerate(PROFILES):
        for col, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[row, col]
            for width in WIDTHS:
                part = summary[
                    (summary.profile == profile)
                    & (summary.width == width)
                    & summary.comparison.str.startswith("step_")
                ].sort_values("updates")
                factor = 100 if metric == "step_p_negative" else 1
                ax.plot(
                    part.updates,
                    factor * part[metric],
                    "o-",
                    color=colors[width],
                    label=f"width {width}",
                )
                ax.fill_between(
                    part.updates,
                    factor * part[metric + "_low"],
                    factor * part[metric + "_high"],
                    color=colors[width],
                    alpha=0.12,
                )
            ax.set(xscale="log", xlabel="Completed updates", title=title, ylim=(0, None))
            ax.grid(alpha=0.18)
        axes[row, 0].set_ylabel("Adam lr " + ("0.001" if row == 0 else "0.01"))
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Teacher 8 · 5 paired seeds · 128 fresh batches per state\nShading: pointwise seed-bootstrap 95% intervals",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT / f"adam_cross_batch_trajectory.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, metric, title in zip(axes, metrics, titles):
        factor = 100 if metric == "step_p_negative" else 1
        for j, profile in enumerate(PROFILES):
            for width in WIDTHS:
                part = summary[
                    (summary.profile == profile)
                    & (summary.width == width)
                    & (summary.comparison == "mse_0.2")
                ].iloc[0]
                pos = j + (WIDTHS.index(width) - 1) * 0.18
                ax.errorbar(
                    pos,
                    factor * part[metric],
                    yerr=[
                        [factor * (part[metric] - part[metric + "_low"])],
                        [factor * (part[metric + "_high"] - part[metric])],
                    ],
                    color=colors[width],
                    marker="o",
                    capsize=3,
                    label=f"width {width}" if j == 0 else None,
                )
        ax.set(xticks=[0, 1], xticklabels=["Adam .001", "Adam .01"], title=title, ylim=(0, None))
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Early comparison near training MSE 0.2 · 5 paired seeds", fontsize=11)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT / f"adam_cross_batch_matched_loss.{suffix}", dpi=180)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "states"
    cache.mkdir(exist_ok=True)
    data = torch.load(STUDY / "data.pt", weights_only=True)
    rng = torch.Generator().manual_seed(PROBE_SEED)
    indices = torch.randint(len(data["x_train"]), (PROBES, 64), generator=rng)
    torch.save(indices, OUT / "probe_indices.pt")
    records, replay_errors = [], {}
    for profile in PROFILES:
        selected, config, replay_error = states(STUDY / profile, data)
        replay_errors[profile] = replay_error
        for state in selected:
            record, arrays = measure(state, config["optimizer_spec"], data, indices)
            record["profile"] = profile
            records.append(record)
            name = f"{profile}_w{state['width']}_s{state['seed']}_{state['comparison']}.pt"
            torch.save({**record, **arrays}, cache / name)
        print(
            f"{profile}: {len(selected)} states complete; replay error {replay_error:.3g}",
            flush=True,
        )
    frame = pd.DataFrame(records)
    summary, effects = summaries(frame)
    frame.to_csv(OUT / "per_seed.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    effects.to_csv(OUT / "effects.csv", index=False)
    plots(summary)
    sources = [
        "scripts/studies/analyze_adam_cross_batch.py",
        "src/neural_competition/neurons.py",
        "src/neural_competition/trajectory.py",
        "scripts/studies/run_trajectory.py",
    ]
    manifest = dict(
        teacher_width=8,
        teacher_seed=2027,
        widths=WIDTHS,
        profiles=PROFILES,
        training_seeds=list(range(5)),
        probe_seed=PROBE_SEED,
        probe_batches=PROBES,
        batch_size=64,
        states=len(frame),
        ordered_pairs_per_state=PROBES * (PROBES - 1),
        replay_error=replay_errors,
        arithmetic="float64 diagnostic at saved float32 training state",
        pairing="all ordered off-diagonal pairs; shared fresh training probe batches",
        history="both pre-step Adam moments frozen across candidate batches",
        max_gradient_audit_error=float(frame.gradient_audit_error.max()),
        max_optimizer_audit_error=float(frame.optimizer_audit_error.max()),
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        data_sha256=hashlib.sha256((STUDY / "data.pt").read_bytes()).hexdigest(),
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
