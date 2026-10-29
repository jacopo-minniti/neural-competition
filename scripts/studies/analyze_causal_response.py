"""Evaluate locked causal response forecasts without treating repeated rows as seeds."""

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
OUT = ROOT / "docs/theory/causal_response"


def read(path):
    return [json.loads(v) for v in path.read_text().splitlines()]


def audit(folder):
    assert (folder / "complete.json").exists(), folder
    p = json.loads((folder / "protocol.json").read_text())
    for name, digest in p["sources"].items():
        assert hashlib.sha256((folder / Path(name).name).read_bytes()).hexdigest() == digest
    for name, digest in p.get("checkpoint_hashes", {}).items():
        assert hashlib.sha256((ROOT / p["input"] / name).read_bytes()).hexdigest() == digest
    if "data_sha256" in p:
        assert (
            hashlib.sha256((ROOT / "data/mnist/prepared.pt").read_bytes()).hexdigest()
            == p["data_sha256"]
        )
    return p


def stats(rows, field):
    seeds = sorted({r["seed"] for r in rows})
    x = np.array([np.mean([r[field] for r in rows if r["seed"] == s]) for s in seeds])
    return dict(
        mean=float(x.mean()),
        se=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
        seeds=len(x),
    )


def score(rows, field="prediction"):
    y = np.array([r["actual"] for r in rows])
    p = np.array([r[field] for r in rows])
    seeds = sorted({r["seed"] for r in rows})
    err = np.array(
        [np.mean([(r[field] - r["actual"]) ** 2 for r in rows if r["seed"] == s]) for s in seeds]
    )
    zero = np.array([np.mean([r["actual"] ** 2 for r in rows if r["seed"] == s]) for s in seeds])
    index = np.random.default_rng(284128).integers(len(seeds), size=(10000, len(seeds)))
    interval = np.quantile(
        np.sqrt(err[index].mean(1) / zero[index].mean(1)), [0.025, 0.975]
    ).tolist()
    return dict(
        rows=len(rows),
        seeds=len(seeds),
        rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        zero_rmse=float(np.sqrt(np.mean(y * y))),
        relative_rmse=float(np.linalg.norm(p - y) / np.linalg.norm(y)),
        sign_accuracy=float(np.mean(p * y > 0)),
        bootstrap_95=interval,
        actual=stats(rows, "actual"),
        better_than_zero_seeds=int(np.sum(err < zero)),
    )


def replication(folder, timing_folder):
    audit(folder)
    forecasts = {(r["seed"], r["width"]): r for r in read(folder / "forecasts.jsonl")}
    outcomes = read(folder / "outcomes.jsonl")
    assert len(forecasts) == len(outcomes) == 16
    rows = []
    for o in outcomes:
        f = forecasts[o["seed"], o["width"]]
        for h in [1, 5, 20]:
            for name, left, right in [
                ("pressure", "peer", "none"),
                ("preservation", "signed", "peer"),
            ]:
                rows.append(
                    dict(
                        seed=o["seed"],
                        width=o["width"],
                        horizon=h,
                        target=name,
                        actual=o["arms"][left]["anchor_gains"][str(h)]
                        - o["arms"][right]["anchor_gains"][str(h)],
                        prediction=f["arms"][left]["anchor_gains"][str(h)]
                        - f["arms"][right]["anchor_gains"][str(h)],
                    )
                )
    result = dict(primary=score(rows), settings={})
    for name in ["pressure", "preservation"]:
        for h in [1, 5, 20]:
            selected = [r for r in rows if r["target"] == name and r["horizon"] == h]
            result["settings"][f"{name}_{h}"] = dict(
                score=score(selected),
                actual_negative=sum(r["actual"] < 0 for r in selected),
                predicted_negative=sum(r["prediction"] < 0 for r in selected),
            )
    result["rows"] = rows
    result["timing"] = timing(timing_folder, outcomes, forecasts)
    return result


def timing(folder, outcomes, forecasts):
    audit(folder)
    late_f = {(r["seed"], r["width"]): r for r in read(folder / "forecasts.jsonl")}
    late_o = {(r["seed"], r["width"]): r for r in read(folder / "outcomes.jsonl")}
    rows = []
    for o in outcomes:
        key = o["seed"], o["width"]
        if key not in late_o:
            continue
        late = late_o[key]["arms"]["late_signed"]
        lf = late_f[key]["arms"]["late_signed"]
        f = forecasts[key]["arms"]
        # A late expansion cannot have preserved a response lost during pressure.
        for field in ["value", "fixed_power", "loss"]:
            assert abs(late[field] - o["arms"]["peer"][field]) < 1e-8
        for h in [1, 5, 20]:
            rows.append(
                dict(
                    seed=o["seed"],
                    width=o["width"],
                    horizon=h,
                    actual=o["arms"]["signed"]["anchor_gains"][str(h)]
                    - late["anchor_gains"][str(h)],
                    prediction=f["signed"]["anchor_gains"][str(h)] - lf["anchor_gains"][str(h)],
                    early_vs_peer=o["arms"]["signed"]["anchor_gains"][str(h)]
                    - o["arms"]["peer"]["anchor_gains"][str(h)],
                    pressure=o["arms"]["peer"]["anchor_gains"][str(h)]
                    - o["arms"]["none"]["anchor_gains"][str(h)],
                )
            )
    return dict(
        score=score(rows),
        rows=rows,
        positive_history_effects={
            str(h): sum(r["actual"] > 0 for r in rows if r["horizon"] == h) for h in [1, 5, 20]
        },
    )


def main():
    audit(OUT / "checkpoints")
    audit(OUT / "class8_checkpoints")
    forecasts = {}
    outcomes = []
    for name in ["validation_a", "validation_b"]:
        p = audit(OUT / name)
        fs = read(OUT / name / "forecasts.jsonl")
        os = read(OUT / name / "outcomes.jsonl")
        assert len(fs) == 4 * len(p["cases"]) and len(os) == 2 * len(p["cases"])
        for r in fs:
            k = (r["seed"], r["width"], r["step"], r["focal"], r["refresh"])
            assert k not in forecasts
            forecasts[k] = r
        outcomes.extend(os)
    assert len({(r["seed"], r["width"]) for r in outcomes}) == len(outcomes) == 20
    rows = []
    state_rows = []
    for o in outcomes:
        a = o["arms"]
        initial = o["initial"]
        state = dict(
            seed=o["seed"],
            width=o["width"],
            focal=o["focal"],
            peer_value_change=a["peer"]["value"] / initial["value"] - 1,
            signed_value_change=a["signed"]["value"] / initial["value"] - 1,
            native_signed_cost=a["signed"]["native_final_loss"] - a["peer"]["native_final_loss"],
            native_signed_focal_cost=a["signed"]["native_final_focal_loss"]
            - a["peer"]["native_final_focal_loss"],
        )
        state_rows.append(state)
        for h in ["1", "5", "20"]:
            for target, left, right in [
                ("pressure", "peer", "none"),
                ("preservation", "signed", "peer"),
            ]:
                row = dict(
                    seed=o["seed"],
                    width=o["width"],
                    focal=o["focal"],
                    horizon=int(h),
                    target=target,
                    actual=a[left]["anchor_gains"][h] - a[right]["anchor_gains"][h],
                    baseline_gain=a["none"]["anchor_gains"][h],
                    actual_quadratic=a[left]["quadratic_gains"][h] - a[right]["quadratic_gains"][h],
                )
                row["relative_effect"] = row["actual"] / row["baseline_gain"]
                for refresh in [0, 10]:
                    f = forecasts[o["seed"], o["width"], o["step"], o["focal"], refresh]["arms"]
                    row[f"prediction_{refresh}"] = (
                        f[left]["anchor_gains"][h] - f[right]["anchor_gains"][h]
                    )
                rows.append(row)
    result = dict(primary={str(r): score(rows, f"prediction_{r}") for r in [0, 10]}, settings={})
    for target in ["pressure", "preservation"]:
        for h in [1, 5, 20]:
            selected = [r for r in rows if r["target"] == target and r["horizon"] == h]
            result["settings"][f"{target}_{h}"] = {
                str(k): score(selected, f"prediction_{k}") for k in [0, 10]
            }
            result["settings"][f"{target}_{h}"]["oracle_quadratic"] = score(
                selected, "actual_quadratic"
            )
    harm = [
        r
        for r in rows
        if r["target"] == "pressure" and r["horizon"] == 1 and r["prediction_10"] < 0
    ]
    chosen = {(r["seed"], r["width"]) for r in harm}
    treatment = [
        r
        for r in rows
        if r["target"] == "preservation" and r["horizon"] == 1 and (r["seed"], r["width"]) in chosen
    ]
    result["forecast_selected_harm"] = dict(
        count=len(harm),
        actual_harmed=sum(r["actual"] < 0 for r in harm),
        actual_rescued=sum(r["actual"] > 0 for r in treatment),
        states=[
            dict(
                seed=r["seed"],
                width=r["width"],
                focal=r["focal"],
                predicted=r["prediction_10"],
                actual=r["actual"],
            )
            for r in harm
        ],
    )
    result["width"] = {}
    for w in [32, 128]:
        selected = [
            dict(r, damage=max(0, -r["relative_effect"]))
            for r in rows
            if r["width"] == w and r["horizon"] == 1 and r["target"] == "pressure"
        ]
        states = [r for r in state_rows if r["width"] == w]
        result["width"][str(w)] = dict(
            relative_first_step_damage=stats(selected, "damage"),
            first_step_impaired=sum(r["actual"] < 0 for r in selected),
            option_weakened=sum(r["peer_value_change"] < 0 for r in states),
            native_signed_cost=stats(states, "native_signed_cost"),
            drift_reduced=sum(
                abs(r["signed_value_change"]) < abs(r["peer_value_change"]) for r in states
            ),
            absolute_drift_ratio=float(
                np.mean([abs(r["signed_value_change"]) for r in states])
                / np.mean([abs(r["peer_value_change"]) for r in states])
            ),
        )
    paired = {}
    for r in rows:
        if r["target"] == "pressure" and r["horizon"] == 1:
            paired.setdefault(r["seed"], {})[r["width"]] = max(0, -r["relative_effect"])
    differences = np.array([v[32] - v[128] for v in paired.values()])
    sample = np.random.default_rng(284129).choice(differences, size=(10000, len(differences)))
    result["width"]["paired_damage_narrow_minus_wide"] = dict(
        mean=float(differences.mean()),
        se=float(differences.std(ddof=1) / np.sqrt(len(differences))),
        bootstrap_95=np.quantile(sample.mean(1), [0.025, 0.975]).tolist(),
    )
    result["counts"] = {
        f"{target}_{h}_negative": sum(
            r["actual"] < 0 for r in rows if r["target"] == target and r["horizon"] == h
        )
        for target in ["pressure", "preservation"]
        for h in [1, 5, 20]
    }
    result["state_rows"] = state_rows
    result["rows"] = rows
    for name in ["pilot", "signed_pilot", "quadratic_pilot", "closure_audit"]:
        audit(OUT / name)
    # Exact branch replication: adding a diagnostic must not alter the outcomes.
    old = {(r["seed"], r["width"]): r for r in read(OUT / "signed_pilot/outcomes.jsonl")}
    for r in read(OUT / "quadratic_pilot/outcomes.jsonl"):
        for arm, values in r["arms"].items():
            for k, v in old[r["seed"], r["width"]]["arms"][arm].items():
                assert values[k] == v, (r["seed"], r["width"], arm, k)
    result["closure_audit"] = {}
    records = read(OUT / "closure_audit/results.jsonl")
    for model in ["quadratic", "tangent_ce", "bilinear_ce"]:
        cr = []
        for r in records:
            a = r["arms"]
            for h in ["1", "5", "20"]:
                for left, right in [("peer", "none"), ("signed", "peer")]:
                    cr.append(
                        dict(
                            seed=r["seed"],
                            actual=a[left]["actual"][h] - a[right]["actual"][h],
                            prediction=a[left][model][h] - a[right][model][h],
                        )
                    )
        result["closure_audit"][model] = score(cr)
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["broad_timing_audit"] = timing(
        OUT / "timing_audit",
        outcomes,
        {(r["seed"], r["width"]): r for r in forecasts.values() if r["refresh"] == 10},
    )
    result["class8_replication"] = replication(OUT / "class8_replication", OUT / "class8_timing")
    result["inference"] = (
        "Independent seeds are the replication unit. Each seed has one focal class; class and seed effects are not separated."
    )
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(rows, state_rows)
    plot_windows(rows, result["class8_replication"]["rows"])
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["primary", "forecast_selected_harm", "width", "counts", "closure_audit"]
            },
            indent=2,
        )
    )


def plot(rows, states):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for h, color in [(1, "#267aa5"), (5, "#a88924"), (20, "#b65336")]:
        r = [v for v in rows if v["horizon"] == h]
        ax[0].scatter(
            [v["actual"] for v in r],
            [v["prediction_10"] for v in r],
            s=20,
            color=color,
            label=f"{h} step" if h == 1 else f"{h} steps",
        )
    limits = ax[0].get_xlim()
    ax[0].plot(limits, limits, "k--", lw=1)
    ax[0].set(
        xlabel="Actual correction-gain contrast",
        ylabel="Predicted contrast",
        title="Fresh MNIST intervention forecasts",
    )
    ax[0].legend(fontsize=8)
    ax[0].xaxis.set_major_locator(MaxNLocator(4))
    ax[0].ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    for w, color in [(32, "#b65336"), (128, "#267aa5")]:
        r = [v for v in states if v["width"] == w]
        ax[1].scatter(
            [100 * v["peer_value_change"] for v in r],
            [100 * v["signed_value_change"] for v in r],
            s=25,
            color=color,
            label=f"width {w}",
        )
    limits = ax[1].get_xlim()
    ax[1].plot(limits, limits, "k--", lw=1)
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].axvline(0, color="grey", lw=0.6)
    ax[1].set(
        xlabel="Peer-induced option change (%)",
        ylabel="Change with signed copies (%)",
        title="Preservation suppresses both signs",
    )
    ax[1].legend(fontsize=8)
    for w, color in [(32, "#b65336"), (128, "#267aa5")]:
        for target, style in [("pressure", "-"), ("preservation", "--")]:
            y = []
            e = []
            for h in [1, 5, 20]:
                r = [
                    v
                    for v in rows
                    if v["width"] == w and v["target"] == target and v["horizon"] == h
                ]
                s = stats(r, "actual")
                y.append(s["mean"])
                e.append(s["se"])
            ax[2].errorbar(
                [1, 5, 20],
                y,
                yerr=e,
                color=color,
                linestyle=style,
                marker="o",
                capsize=2,
                label=f"{w}: {target}",
            )
    ax[2].axhline(0, color="black", lw=0.8)
    ax[2].set(
        xlabel="Correction steps", ylabel="Gain contrast (CE)", title="Finite-budget consequences"
    )
    ax[2].legend(fontsize=7)
    fig.savefig(OUT / "summary.png", dpi=180)
    plt.close(fig)


def plot_windows(rows, replicated):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    chosen = {
        (r["seed"], r["width"])
        for r in rows
        if r["target"] == "pressure" and r["prediction_10"] < 0
    }
    for seed, width in sorted(chosen):
        selected = sorted(
            [
                r
                for r in rows
                if r["seed"] == seed and r["width"] == width and r["target"] == "pressure"
            ],
            key=lambda r: r["horizon"],
        )
        axes[0].plot(
            [r["horizon"] for r in selected],
            [100 * r["relative_effect"] for r in selected],
            marker="o",
            label=f"seed {seed} / w{width}",
        )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(
        xlabel="Correction steps",
        ylabel="Peer effect / no-pressure gain (%)",
        title="Forecast-selected risk is budget-dependent",
        xticks=[1, 5, 20],
    )
    axes[0].legend(fontsize=7)
    original = next(
        r["actual"]
        for r in rows
        if r["seed"] == 284109
        and r["width"] == 32
        and r["target"] == "preservation"
        and r["horizon"] == 20
    )
    means, errors = [original], [0.0]
    for width in [32, 128]:
        selected = [
            r
            for r in replicated
            if r["width"] == width and r["horizon"] == 20 and r["target"] == "preservation"
        ]
        s = stats(selected, "actual")
        means.append(s["mean"])
        errors.append(s["se"])
    axes[1].bar(
        ["Original case\nwidth 32", "8 fresh seeds\nwidth 32", "8 fresh seeds\nwidth 128"],
        means,
        yerr=errors,
        capsize=3,
        color=["#89959f", "#b65336", "#267aa5"],
    )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(
        ylabel="Early copies − ordinary gain (CE)", title="The digit-8 rescue does not replicate"
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.savefig(OUT / "correction_windows.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
