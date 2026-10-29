"""Seed-level evaluation of the compact response-feedback model and its limits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/theory/response_feedback"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check(folder):
    p = json.loads((folder / "protocol.json").read_text())
    assert (folder / "complete.json").exists(), folder
    for source, digest in p["sources"].items():
        assert hashlib.sha256((folder / Path(source).name).read_bytes()).hexdigest() == digest
    for checkpoint, digest in p.get("checkpoint_hashes", {}).items():
        assert hashlib.sha256((ROOT / p["input"] / checkpoint).read_bytes()).hexdigest() == digest
    return p


def stats(values):
    x = np.array(values, dtype=float)
    return dict(
        mean=float(x.mean()),
        se=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
        seeds=len(x),
    )


def scores(rows, field="effect"):
    y = np.array([r[field] for r in rows])
    p = np.array([r["pred_" + field] for r in rows])
    seeds = sorted({r["seed"] for r in rows})
    # Resample seeds, retaining dependence across widths, horizons and checkpoints.
    errors = np.array(
        [
            np.mean([(r["pred_" + field] - r[field]) ** 2 for r in rows if r["seed"] == s])
            for s in seeds
        ]
    )
    baselines = np.array([np.mean([r[field] ** 2 for r in rows if r["seed"] == s]) for s in seeds])
    indices = np.random.default_rng(281028).integers(len(seeds), size=(10000, len(seeds)))
    ratios = np.sqrt(errors[indices].mean(1) / baselines[indices].mean(1))
    return dict(
        n_seed_states=len(rows),
        seeds=len(seeds),
        actual=stats([np.mean([r[field] for r in rows if r["seed"] == s]) for s in seeds]),
        predicted=stats(
            [np.mean([r["pred_" + field] for r in rows if r["seed"] == s]) for s in seeds]
        ),
        rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        zero_rmse=float(np.sqrt(np.mean(y**2))),
        relative_rmse=float(np.linalg.norm(p - y) / np.linalg.norm(y))
        if np.linalg.norm(y) > 0
        else None,
        sign_accuracy=float(np.mean(p * y > 0)),
        relative_rmse_seed_bootstrap_95=np.quantile(ratios, [0.025, 0.975]).tolist(),
        seed_rmse=stats(
            [
                np.sqrt(
                    np.mean([(r["pred_" + field] - r[field]) ** 2 for r in rows if r["seed"] == s])
                )
                for s in seeds
            ]
        ),
        improves_zero_seeds=int(
            sum(
                np.mean([(r["pred_" + field] - r[field]) ** 2 for r in rows if r["seed"] == s])
                < np.mean([r[field] ** 2 for r in rows if r["seed"] == s])
                for s in seeds
            )
        ),
    )


def analyze(folder):
    protocol = check(folder)
    forecasts = read(folder / "forecasts.jsonl")
    outcomes = read(folder / "outcomes.jsonl")

    def key(r):
        return tuple(r[k] for k in ["seed", "width", "step", "stream", "horizon"])

    forecasts = {key(r): r for r in forecasts}
    expected = (
        len(protocol["seeds"])
        * 2
        * len(protocol["steps"])
        * protocol["streams"]
        * len(protocol["horizons"])
    )
    assert len(forecasts) == len(outcomes) == expected
    assert len({key(r) for r in outcomes}) == expected
    groups = {}
    for o in outcomes:
        f = forecasts[key(o)]
        row = {k: o[k] for k in ["seed", "width", "step", "horizon"]}
        row.update(
            effect=o["effect"],
            pred_effect=f["effect"],
            validation_effect=o["validation_effect"],
            pred_validation_effect=f["validation_effect"],
            ordinary_gain=f["initial_loss"] - o["ordinary"]["loss"],
            predicted_gain=f["initial_loss"] - f["ordinary"]["loss"],
        )
        if "linear" in f:
            row["linear_gain"] = f["initial_loss"] - f["linear"]["loss"]
        if "decoded_loss" in f["ordinary"]:
            row["decoded_effect"] = f["ordinary"]["decoded_loss"] - f["reserve"]["decoded_loss"]
            row["decoded_linear_effect"] = (
                f["linear"]["decoded_loss"] - f["linear_reserve"]["decoded_loss"]
            )
            row["decoded_gain"] = f["initial_loss"] - f["ordinary"]["decoded_loss"]
            row["decoded_linear_gain"] = f["initial_loss"] - f["linear"]["decoded_loss"]
        groups.setdefault((o["seed"], o["width"], o["step"], o["horizon"]), []).append(row)
    averaged = []
    for group in groups.values():
        r = {k: group[0][k] for k in ["seed", "width", "step", "horizon"]}
        r.update({k: float(np.mean([v[k] for v in group])) for k in group[0] if k not in r})
        averaged.append(r)
    result = {
        "primary": scores(
            [r for r in averaged if r["step"] in [20, 100] and r["horizon"] in [20, 80]]
        ),
        "settings": {},
    }
    if all("decoded_effect" in r for r in averaged):
        result["decoded_primary"] = scores(
            [
                dict(r, pred_effect=r["decoded_effect"])
                for r in averaged
                if r["step"] in [20, 100] and r["horizon"] in [20, 80]
            ]
        )
        result["decoded_linear_primary"] = scores(
            [
                dict(r, pred_effect=r["decoded_linear_effect"])
                for r in averaged
                if r["step"] in [20, 100] and r["horizon"] in [20, 80]
            ]
        )
    for step in protocol["steps"]:
        for horizon in protocol["horizons"]:
            selected = [r for r in averaged if r["step"] == step and r["horizon"] == horizon]
            record = scores(selected)
            if all("decoded_effect" in r for r in selected):
                for name in ["decoded", "decoded_linear"]:
                    record[name] = scores(
                        [dict(r, pred_effect=r[name + "_effect"]) for r in selected]
                    )
                    record[name + "_gain_rmse"] = float(
                        np.sqrt(
                            np.mean(
                                [(r[name + "_gain"] - r["ordinary_gain"]) ** 2 for r in selected]
                            )
                        )
                    )
            record["validation"] = scores(selected, "validation_effect")
            record["widths"] = {
                str(w): scores([r for r in selected if r["width"] == w]) for w in [32, 128]
            }
            record["ordinary_gain_rmse"] = float(
                np.sqrt(
                    np.mean([(r["predicted_gain"] - r["ordinary_gain"]) ** 2 for r in selected])
                )
            )
            if all("linear_gain" in r for r in selected):
                record["linear_gain_rmse"] = float(
                    np.sqrt(
                        np.mean([(r["linear_gain"] - r["ordinary_gain"]) ** 2 for r in selected])
                    )
                )
                by = {(r["seed"], r["width"]): r for r in selected}
                target = []
                bilinear = []
                linear = []
                for seed in protocol["seeds"]:
                    wide, narrow = by[seed, 128], by[seed, 32]
                    target.append(wide["ordinary_gain"] - narrow["ordinary_gain"])
                    bilinear.append(wide["predicted_gain"] - narrow["predicted_gain"])
                    linear.append(wide["linear_gain"] - narrow["linear_gain"])
                record["ordinary_width_gain_rmse"] = {
                    name: float(np.sqrt(np.mean((np.array(v) - target) ** 2)))
                    for name, v in [
                        ("bilinear", bilinear),
                        ("linear", linear),
                        ("zero", np.zeros(len(target))),
                    ]
                }
            result["settings"][f"{step}_{horizon}"] = record
    result["matching_max"] = max(max(r["matching"].values()) for r in forecasts.values())
    return result, averaged


def scalar(folder):
    protocol = check(folder)
    forecasts = {(r["seed"], r["width"], r["step"]): r for r in read(folder / "forecasts.jsonl")}
    outcomes = read(folder / "outcomes.jsonl")
    assert len(outcomes) == len(protocol["seeds"]) * 2 * len(protocol["steps"]) * len(
        protocol["horizons"]
    )
    result = {"clone_max_error": max(r["clone_error"] for r in outcomes), "settings": {}}
    for step in protocol["steps"]:
        for horizon in protocol["horizons"]:
            selected = []
            for r in outcomes:
                if r["step"] != step or r["horizon"] != horizon:
                    continue
                f = forecasts[r["seed"], r["width"], r["step"]]
                p = next(v for v in f["predictions"] if v["horizon"] == horizon)
                selected.append(dict(r, **{k: p[k] for k in ["leading", "relaxed"]}))
            result["settings"][f"{step}_{horizon}"] = {
                name: scores([dict(r, pred_effect=r[name]) for r in selected])
                for name in ["leading", "relaxed"]
            }
    return result


def peer(folder):
    protocol = check(folder)
    f = {(r["seed"], r["width"], r["step"]): r for r in read(folder / "forecasts.jsonl")}
    o = read(folder / "outcomes.jsonl")
    assert len(o) == len(protocol["seeds"]) * len(protocol["steps"]) * 2
    assert len(f) == len({(r["seed"], r["width"], r["step"]) for r in o}) == len(o)
    rows = []
    for r in o:
        forecast = f[r["seed"], r["width"], r["step"]]
        h = r["histories"]
        p = forecast["histories"]
        b = h["peers"]["0"]
        a = h["peers"]["20"]
        pred = p["peers"]["20"]
        row = {k: r[k] for k in ["seed", "width", "step", "focal"]}
        row.update(
            effect=a["value"] / b["value"] - 1,
            pred_effect=pred["value"] / b["value"] - 1,
            power_change=a["fixed_power"] / b["fixed_power"] - 1,
            focal_loss_change=a["focal_loss"] - b["focal_loss"],
            peer_vs_full_value=(a["value"] - h["full"]["20"]["value"]) / b["value"],
            recovered_loss_gap=h["peers"]["100"]["loss"] - h["full"]["100"]["loss"],
            recovered_focal_gap=h["peers"]["100"]["focal_loss"] - h["full"]["100"]["focal_loss"],
            cg_residual=max(
                t.get("cg_residual", 0.0) for hist in h.values() for t in hist.values()
            ),
        )
        if "decoded" in pred:
            row["decoded_effect"] = pred["decoded"]["value"] / b["value"] - 1
            row["decoded_peer_vs_full"] = (
                pred["decoded"]["value"] - p["full"]["20"]["decoded"]["value"]
            ) / b["value"]
        rows.append(row)
    result = dict(
        scores=scores(rows),
        capability_weakened=sum(r["effect"] < 0 for r in rows),
        focal_loss_increased=sum(r["focal_loss_change"] > 0 for r in rows),
        rows=len(rows),
        loss_harmed_without_weakening=sum(
            r["focal_loss_change"] > 0 and r["effect"] >= 0 for r in rows
        ),
        max_cg_residual=max(r["cg_residual"] for r in rows),
    )
    if all("decoded_effect" in r for r in rows):
        decoded = [dict(r, pred_effect=r["decoded_effect"]) for r in rows]
        result["decoded_scores"] = scores(decoded)
        result["decoded_peer_vs_full_scores"] = scores(
            [
                dict(r, effect=r["peer_vs_full_value"], pred_effect=r["decoded_peer_vs_full"])
                for r in rows
            ]
        )
    result["peer_value_below_full_control"] = sum(r["peer_vs_full_value"] < 0 for r in rows)
    result["recovery"] = {
        field: stats(
            [np.mean([r[field] for r in rows if r["seed"] == s]) for s in protocol["seeds"]]
        )
        for field in ["recovered_loss_gap", "recovered_focal_gap", "peer_vs_full_value"]
    }
    result["settings"] = {
        str(step): dict(
            capability=scores([r for r in rows if r["step"] == step]),
            focal_loss_change=stats(
                [
                    np.mean(
                        [
                            r["focal_loss_change"]
                            for r in rows
                            if r["seed"] == s and r["step"] == step
                        ]
                    )
                    for s in protocol["seeds"]
                ]
            ),
        )
        for step in protocol["steps"]
    }
    return result, rows


def figures(result, rows, peer_rows):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    primary = [r for r in rows if r["step"] in [20, 100] and r["horizon"] in [20, 80]]
    for width, color in [(32, "#b65336"), (128, "#267aa5")]:
        selected = [r for r in primary if r["width"] == width]
        axes[0].scatter(
            [r["effect"] for r in selected],
            [r["pred_effect"] for r in selected],
            color=color,
            label=f"width {width}",
            s=24,
        )
    bounds = axes[0].get_xlim()
    axes[0].plot(bounds, bounds, color="black", ls="--", lw=1)
    axes[0].set(
        xlabel="Actual intervention effect (CE)",
        ylabel="Predicted effect",
        title="Independent-stream SGD forecast",
    )
    axes[0].legend(fontsize=8)
    axes[0].xaxis.set_major_locator(MaxNLocator(4))
    for width, color in [(32, "#b65336"), (128, "#267aa5")]:
        selected = [
            r for r in primary if r["width"] == width and r["step"] == 20 and r["horizon"] == 80
        ]
        axes[1].bar(
            str(width),
            np.mean([r["effect"] for r in selected]),
            color=color,
            yerr=np.std([r["effect"] for r in selected], ddof=1) / np.sqrt(len(selected)),
        )
    axes[1].set(
        title="Slowing feedback usually hurts",
        xlabel="Base width",
        ylabel="Reserve − ordinary gain (CE)",
    )
    axes[1].axhline(0, color="black", lw=0.8)
    for width, color in [(32, "#b65336"), (128, "#267aa5")]:
        selected = [r for r in peer_rows if r["width"] == width]
        axes[2].scatter(
            [r["effect"] * 100 for r in selected],
            [r["focal_loss_change"] for r in selected],
            color=color,
            s=25,
        )
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set(
        xlabel="Change in focal response value (%)",
        ylabel="Change in focal CE",
        title="Peer pressure: loss ≠ option loss",
    )
    fig.savefig(OUT / "summary.png", dpi=180)
    plt.close(fig)


def comparisons(decoded_rows, peer_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    labels = ["Internal bilinear", "Physical bilinear", "Physical linear"]
    fields = ["pred_effect", "decoded_effect", "decoded_linear_effect"]
    colors = ["#89959f", "#267aa5", "#b65336"]
    for i, (label, field, color) in enumerate(zip(labels, fields, colors)):
        values = [
            scores([dict(r, pred_effect=r[field]) for r in decoded_rows if r["horizon"] == h])[
                "relative_rmse"
            ]
            for h in [20, 80]
        ]
        axes[0].bar(np.arange(2) + (i - 1) * 0.24, values, width=0.23, label=label, color=color)
    axes[0].axhline(1, ls="--", color="black", lw=1, label="Zero-effect baseline")
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["20 steps", "80 steps"],
        ylabel="RMSE / zero-effect RMSE",
        title="Matched-response effect forecast",
    )
    axes[0].legend(fontsize=8)
    for field, label, color in [
        ("pred_effect", "Internal bilinear", "#89959f"),
        ("decoded_effect", "Physical bilinear", "#267aa5"),
    ]:
        axes[1].scatter(
            [100 * r["effect"] for r in peer_rows],
            [100 * r[field] for r in peer_rows],
            s=22,
            color=color,
            label=label,
        )
    lo, hi = axes[1].get_xlim()
    axes[1].plot([lo, hi], [lo, hi], color="black", ls="--", lw=1)
    axes[1].set(
        xlabel="Actual focal option-value change (%)",
        ylabel="Predicted change (%)",
        title="Other-class pressure: 20 steps",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close(fig)


def main():
    result = {"scalar_pilot": scalar(OUT / "pilot")}
    for name in ["bilinear_pilot", "sgd_pilot", "bilinear_validation", "sgd_validation"]:
        result[name], rows = analyze(OUT / name)
    for name in ["decoded_sgd_pilot", "decoded_validation"]:
        result[name], decoded_rows = analyze(OUT / name)
    for name in ["peer_pilot", "peer_decoded_pilot", "peer_validation"]:
        result[name], peer_rows = peer(OUT / name)
    # Adding forecast readouts must not change the actual experimental branches.
    peer_old = read(OUT / "peer_pilot/outcomes.jsonl")
    peer_new = read(OUT / "peer_decoded_pilot/outcomes.jsonl")
    assert peer_old == peer_new
    keys = ["seed", "width", "step", "stream", "horizon"]
    old = {tuple(r[k] for k in keys): r for r in read(OUT / "sgd_pilot/outcomes.jsonl")}
    new = read(OUT / "decoded_sgd_pilot/outcomes.jsonl")
    assert all(old[tuple(r[k] for k in keys)] == r for r in new)
    result["repeated_branch_audit"] = (
        "All shared pilot outcomes identical across readout ablations."
    )
    result["inference"] = (
        "SE over initialization seeds; widths, checkpoints, horizons and streams share those seeds."
    )
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    figures(result, rows, peer_rows)
    comparisons(decoded_rows, peer_rows)
    print(
        json.dumps(
            {name: result[name]["primary"] for name in ["bilinear_validation", "sgd_validation"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
