"""Recompute seed-level summaries, provenance audits and observed-learning plots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from explore_stable_distinctions import task
from trace_observed_learning import toy_forward, toy_tangent, toy_velocity

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/evidence/observed_learning"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stats(values):
    v = np.asarray(values, dtype=float)
    return dict(
        mean=float(v.mean()),
        se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
        n=len(v),
        positive=int(np.sum(v > 0)),
    )


def ledger(path):
    with np.load(path) as raw:
        return dict(zip(raw["columns"], raw["values"].T))


def provenance(folder):
    protocol = json.loads((folder / "protocol.json").read_text())
    assert (folder / "complete.json").exists(), folder
    for source, expected in protocol["sources"].items():
        assert hashlib.sha256((folder / Path(source).name).read_bytes()).hexdigest() == expected
    return protocol


def mnist(folder):
    protocol = provenance(folder)
    seeds = protocol["seeds"]
    result, data = {}, {}
    checkpoints = read(folder / "checkpoints.jsonl")
    panels = read(folder / "panels.jsonl")
    for width in [32, 128]:
        data[width] = {seed: ledger(folder / f"ledger_{seed}_{width}.npz") for seed in seeds}
    closure = max(np.max(abs(d["closure"])) for side in data.values() for d in side.values())
    tangent_error = max(
        np.max(abs(d["tangent"] - d["signal"] - d["sampling"] - d["rounding"]))
        for side in data.values()
        for d in side.values()
    )
    assert closure < 1e-12 and tangent_error < 1e-12
    result["verification"] = dict(
        max_step_closure=float(closure), max_parameter_output_work_difference=float(tangent_error)
    )
    result["stages"] = {}
    for start in [0, 100]:
        stage = {}
        for width in [32, 128]:
            stage[str(width)] = {
                key: stats([d[key][start:].sum() for d in data[width].values()])
                for key in ["gain", "signal", "sampling", "nonlinear", "cost", "rounding"]
            }
            stage[str(width)]["initial_loss"] = stats(
                [d["loss"][start] for d in data[width].values()]
            )
            stage[str(width)]["final_loss"] = stats(
                [d["next_loss"][-1] for d in data[width].values()]
            )
        stage["wide_minus_narrow"] = {
            key: stats(
                [data[128][s][key][start:].sum() - data[32][s][key][start:].sum() for s in seeds]
            )
            for key in ["gain", "signal", "sampling", "nonlinear", "cost", "rounding"]
        }
        result["stages"][str(start)] = stage
    result["final_gap"] = stats(
        [data[32][s]["next_loss"][-1] - data[128][s]["next_loss"][-1] for s in seeds]
    )
    result["validation_gap"] = stats(
        [
            next(
                r["validation_loss"]
                for r in checkpoints
                if r["seed"] == s and r["width"] == 32 and r["step"] == 800
            )
            - next(
                r["validation_loss"]
                for r in checkpoints
                if r["seed"] == s and r["width"] == 128 and r["step"] == 800
            )
            for s in seeds
        ]
    )
    result["panel"] = {}
    for step in [0, 20, 100, 400, 800]:
        for size in [64, 256, 1024]:
            rows = [r for r in panels if r["step"] == step and r["size"] == size]
            record = {}
            for key in ["own", "corrected", "cross"]:
                # Average squared error over draws/states WITHIN seed first.
                record[key + "_relative_rmse"] = stats(
                    [
                        np.sqrt(
                            np.mean(
                                [
                                    (r[key] / r["full_signal"] - 1) ** 2
                                    for r in rows
                                    if r["seed"] == seed
                                ]
                            )
                        )
                        for seed in seeds
                    ]
                )
                record[key + "_ratio"] = stats(
                    [
                        np.mean([r[key] / r["full_signal"] for r in rows if r["seed"] == seed])
                        for seed in seeds
                    ]
                )
            record["predicted_bias_ratio"] = stats(
                [
                    np.mean(
                        [r["expected_bias"] / r["full_signal"] for r in rows if r["seed"] == seed]
                    )
                    for seed in seeds
                ]
            )
            record["negative_corrected_fraction"] = stats(
                [np.mean([r["corrected"] < 0 for r in rows if r["seed"] == seed]) for seed in seeds]
            )
            for key in ["own", "corrected", "cross"]:
                seed_rmse, seed_sign = [], []
                for seed in seeds:
                    w = {r["repeat"]: r for r in rows if r["seed"] == seed and r["width"] == 128}
                    n = {r["repeat"]: r for r in rows if r["seed"] == seed and r["width"] == 32}
                    truth = w[0]["full_signal"] - n[0]["full_signal"]
                    errors = [w[j][key] - n[j][key] - truth for j in w]
                    seed_rmse.append(np.sqrt(np.mean(np.square(errors))))
                    seed_sign.append(np.mean([(w[j][key] - n[j][key]) * truth > 0 for j in w]))
                record[key + "_width_difference_rmse"] = stats(seed_rmse)
                record[key + "_width_sign_accuracy"] = stats(seed_sign)
            result["panel"][f"{step}_{size}"] = record
    return result, data


def toy(folder):
    protocol = provenance(folder)
    raw = np.load(folder / "trajectories.npz")
    result = {}
    for name, nodes, spread, detail in [
        ("discrete", 1, 0.0, 2.0),
        ("continuous", 3, 0.15, 2.0),
        ("no_refinement", 1, 0.0, 0.0),
    ]:
        case = {}
        for width in [4, 8]:
            key = f"{name}_{width}"
            trace = raw[key + "_trace"]
            expected = trace[0, :, 0] - trace[:, :, 4] - trace[:, :, 5] + trace[:, :, 6]
            detail_expected = (
                trace[0, :, 2] - trace[:, :, 7] - trace[:, :, 8] - trace[:, :, 9] + trace[:, :, 10]
            )
            closure = float(np.max(abs(trace[:, :, 0] - expected)))
            detail_closure = float(np.max(abs(trace[:, :, 2] - detail_expected)))
            assert closure < 1e-9 and detail_closure < 1e-9
            x, p, _ = task(nodes, spread, 0.1)
            y = 2 * x[:, 1] + detail * (x[:, 0] > 0) * x[:, 1]
            s = tuple(raw[key + "_" + k] for k in ["w", "a", "c"])
            e = y - toy_forward(s, x)[0]
            d = ((x[:, 0] > 0) - 0.1) * x[:, 1] / (0.3 * np.sqrt(p @ x[:, 1] ** 2))
            loading = e @ (p * d)
            power = (p * e * toy_tangent(s, x, toy_velocity(s, x, p, e))).sum(1)
            dpower = loading * (
                toy_tangent(s, x, toy_velocity(s, x, p, loading[:, None] * d)) @ (p * d)
            )
            case[str(width)] = dict(
                final_loss=stats(trace[-1, :, 0]),
                dense_loss=stats(raw[key + "_dense_loss"]),
                failures=int(np.sum(trace[-1, :, 0] > 1e-7)),
                max_cumulative_closure=closure,
                max_detail_closure=detail_closure,
                final_total_signal=stats(power),
                final_detail_self_power=stats(dpower),
                cumulative={
                    k: stats(trace[-1, :, i]) for i, k in enumerate(protocol["columns"]) if i >= 4
                },
            )
        case["gap"] = stats(raw[name + "_4_trace"][-1, :, 0] - raw[name + "_8_trace"][-1, :, 0])
        result[name] = case
    return result


def probe(folder):
    protocol = provenance(folder)
    rows = read(folder / "outcomes.jsonl")
    expected = len(protocol["seeds"]) * 2 * 2 * (3 * protocol["repeats"] + 1)
    assert len(rows) == expected
    for path, expected_hash in protocol["checkpoint_hashes"].items():
        assert (
            hashlib.sha256((ROOT / protocol["source"] / path).read_bytes()).hexdigest()
            == expected_hash
        )
    result = {}
    for step in [100, 800]:
        for width in [32, 128]:
            sub = [r for r in rows if r["step"] == step and r["width"] == width]
            record = {}
            for size in [128, 512, 2048, "full"]:
                group = [r for r in sub if r["size"] == size]
                record[str(size)] = {
                    key: stats(
                        [
                            np.mean([r[key] for r in group if r["seed"] == s])
                            for s in protocol["seeds"]
                        ]
                    )
                    for key in ["gain", "signal", "tangent", "nonlinear", "cost", "quadratic_cost"]
                }
            for size in [512, 2048, "full"]:
                record[f"{size}_minus_128"] = {
                    key: stats(
                        [
                            np.mean([r[key] for r in sub if r["seed"] == s and r["size"] == size])
                            - np.mean([r[key] for r in sub if r["seed"] == s and r["size"] == 128])
                            for s in protocol["seeds"]
                        ]
                    )
                    for key in ["gain", "cost", "nonlinear", "tangent"]
                }
            result[f"{step}_{width}"] = record
    result["max_closure"] = max(abs(r["closure"]) for r in rows)
    return result


def figures(result, data):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {32: "#b65336", 128: "#267aa5"}
    mean = {
        w: {k: np.mean([d[k] for d in side.values()], axis=0) for k in next(iter(side.values()))}
        for w, side in data.items()
    }
    for w in [32, 128]:
        d = mean[w]
        axes[0, 0].plot(np.r_[d["loss"], d["next_loss"][-1]], color=colors[w], label=f"width {w}")
        axes[0, 1].plot(np.cumsum(d["signal"]), color=colors[w], ls="--", label=f"{w}: signal")
        axes[0, 1].plot(np.cumsum(d["gain"]), color=colors[w], label=f"{w}: realized gain")
    axes[0, 0].set(
        title="Ordinary MNIST training (8 fresh pairs)", xlabel="SGD step", ylabel="Cross-entropy"
    )
    axes[0, 0].legend()
    axes[0, 1].set(
        title="Full gradient alone overcounts progress",
        xlabel="SGD step",
        ylabel="Cumulative loss units",
    )
    axes[0, 1].legend(ncol=2, fontsize=8)
    initial = mean[32]["loss"][0] - mean[128]["loss"][0]
    observed = mean[32]["next_loss"] - mean[128]["next_loss"]
    naive = initial + np.cumsum(mean[128]["signal"] - mean[32]["signal"])
    exact = initial + np.cumsum(
        mean[128]["signal"]
        + mean[128]["sampling"]
        + mean[128]["rounding"]
        + mean[128]["nonlinear"]
        - mean[128]["cost"]
        - mean[32]["signal"]
        - mean[32]["sampling"]
        - mean[32]["rounding"]
        - mean[32]["nonlinear"]
        + mean[32]["cost"]
    )
    axes[1, 0].plot(naive, color="#999999", label="Signal only")
    axes[1, 0].plot(exact, color="#267aa5", lw=3, label="Complete ledger")
    axes[1, 0].plot(observed, color="black", ls="--", label="Observed gap")
    axes[1, 0].set(
        title="Positive gap favors width 128", xlabel="SGD step", ylabel="Narrow − wide CE"
    )
    axes[1, 0].legend(fontsize=8)
    for key, label, color in [
        ("own", "Panel self-response", "#b65336"),
        ("corrected", "Variance-corrected", "#267aa5"),
        ("cross", "Panel × full gradient", "#55a16d"),
    ]:
        sizes = np.array([64, 256, 1024])
        values = [
            result["mnist_confirmation"]["panel"][f"800_{s}"][key + "_relative_rmse"]["mean"]
            for s in sizes
        ]
        axes[1, 1].plot(sizes, values, "o-", label=label, color=color)
    axes[1, 1].set(
        xscale="log",
        yscale="log",
        title="Measurement error at step 800",
        xlabel="Panel size",
        ylabel="RMSE / true signal",
    )
    axes[1, 1].legend(fontsize=8)
    fig.savefig(OUT / "mnist_ledger.png", dpi=180)
    plt.close(fig)
    raw = np.load(OUT / "toy/trajectories.npz")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for j, case in enumerate(["discrete", "continuous"]):
        for width in [4, 8]:
            v = raw[f"{case}_{width}_trace"].mean(1)
            t = raw[f"{case}_{width}_times"]
            axes[j].plot(t, v[:, 0], label=f"width {width}")
        axes[j].set(
            xscale="symlog",
            yscale="log",
            xlabel="Training time",
            ylabel="Half MSE",
            title=case.capitalize(),
        )
        axes[j].legend()
    v = raw["discrete_4_trace"].mean(1)
    t = raw["discrete_4_times"]
    axes[2].plot(t, v[:, 7], label="Detail self-response")
    axes[2].plot(t, -v[:, 8], label="Opposing cross-response", ls="--")
    axes[2].plot(t, v[0, 2] - v[:, 2], label="Realized detail reduction")
    axes[2].set(
        xlabel="Training time", ylabel="Cumulative loss units", title="Positive strength can cancel"
    )
    axes[2].legend(fontsize=8)
    fig.savefig(OUT / "toy_ledger.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, width in zip(axes, [32, 128]):
        r = result["batch_probe"][f"800_{width}"]
        for key, label in [("gain", "Actual loss gain"), ("cost", "Convex cost")]:
            means = [r[str(size)][key]["mean"] for size in [128, 512, 2048, "full"]]
            ses = [r[str(size)][key]["se"] for size in [128, 512, 2048, "full"]]
            ax.errorbar(range(4), means, yerr=ses, marker="o", label=label)
        ax.set(
            xticks=range(4),
            xticklabels=["128", "512", "2048", "Full"],
            title=f"Width {width}, step 800",
            xlabel="Counterfactual batch size",
            ylabel="One-step CE units",
        )
        ax.legend(fontsize=8)
    fig.savefig(OUT / "batch_intervention.png", dpi=180)
    plt.close(fig)


def main():
    result = {}
    for name in ["mnist", "mnist_confirmation"]:
        result[name], data = mnist(OUT / name)
    for name in ["toy", "toy_half_step"]:
        result[name] = toy(OUT / name)
    result["batch_probe"] = probe(OUT / "batch_probe")
    provenance(OUT / "boundary_audit")
    boundary = read(OUT / "boundary_audit/outcomes.jsonl")
    assert len(boundary) == 3 * 2 * 4 * 64
    result["boundary_audit"] = {}
    for case in ["discrete", "continuous", "dense"]:
        for width in [4, 8]:
            for dt in [0.05, 0.025, 0.0125, 0.00625]:
                rows = [
                    r
                    for r in boundary
                    if r["case"] == case and r["width"] == width and r["dt"] == dt
                ]
                result["boundary_audit"][f"{case}_{width}_{dt}"] = {
                    key: stats([r[key] for r in rows])
                    for key in ["loss", "signal", "smooth", "gate", "cost"]
                }
    result["uncertainty"] = (
        "SE over initialization seeds/pairs; panel draws, times and batch probes are not independent seeds."
    )
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    figures(result, data)
    print(
        json.dumps(
            {
                k: result["mnist_confirmation"][k]
                for k in ["verification", "final_gap", "validation_gap"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
