"""Score frozen transition forecasts with seed-cluster uncertainty and calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from implementation_moments import observe
from probabilistic_options import clock, episodes, probability, weighted_task

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/theory/probabilistic_options"
FRESH = BASE / "fresh"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def cluster_interval(values, rows):
    seeds = sorted(set(r["seed"] for r in rows))
    num = []
    den = []
    for seed in seeds:
        chosen = [i for i, r in enumerate(rows) if r["seed"] == seed]
        w = np.array([rows[i]["weight"] for i in chosen])
        v = np.array([values[i] for i in chosen])
        num.append(np.sum(w * v))
        den.append(w.sum())
    num, den = map(np.array, [num, den])
    rng = np.random.default_rng(297501)
    index = rng.integers(len(seeds), size=(4000, len(seeds)))
    boot = num[index].sum(1) / den[index].sum(1)
    return dict(
        mean=float(num.sum() / den.sum()),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        seeds=len(seeds),
    )


def metrics(rows, event):
    y = np.array([r["actual"] for r in rows])
    w = np.array([r["weight"] for r in rows])
    result = {}
    for model in rows[0]["predictions"]:
        p = np.array([r["predictions"][model] for r in rows])
        clipped = np.clip(p, 1e-10, 1 - 1e-10)
        result[model] = dict(
            brier=float(np.average((p - y) ** 2, weights=w)),
            log_loss=float(
                np.average(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped), weights=w)
            ),
            mean_prediction=float(np.average(p, weights=w)),
        )
    chosen = np.array([r["predictions"][event] for r in rows])
    base = np.array([r["predictions"][event + "_constant"] for r in rows])
    no_width = np.array([r["predictions"][event + "_no_width"] for r in rows])
    return dict(
        n=len(rows),
        observed=cluster_interval(y, rows),
        scores=result,
        brier_improvement_over_constant=cluster_interval((base - y) ** 2 - (chosen - y) ** 2, rows),
        brier_improvement_over_no_width=cluster_interval(
            (no_width - y) ** 2 - (chosen - y) ** 2, rows
        ),
        comparator_improvement_over_selected={
            model: cluster_interval(
                (chosen - y) ** 2 - (np.array([r["predictions"][model] for r in rows]) - y) ** 2,
                rows,
            )
            for model in [
                event + "_task",
                event + "_width",
                event + "_options",
                event + "_resource",
            ]
        },
    )


def main():
    assert (FRESH / "complete.json").exists()
    protocol = json.loads((FRESH / "protocol.json").read_text())
    locked = json.loads((FRESH / "forecasts_locked.json").read_text())
    assert (
        hashlib.sha256((FRESH / "forecasts.jsonl").read_bytes()).hexdigest()
        == locked["forecast_sha256"]
    )
    assert (
        hashlib.sha256((FRESH / "models.json").read_bytes()).hexdigest() == protocol["model_sha256"]
    )
    assert (
        hashlib.sha256((ROOT / protocol["model_path"]).read_bytes()).hexdigest()
        == protocol["model_sha256"]
    )
    for name, digest in protocol["sources"].items():
        assert hashlib.sha256((FRESH / name).read_bytes()).hexdigest() == digest
    fit = json.loads((FRESH / "models.json").read_text())
    for path, digest in fit["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for name, digest in json.loads((BASE / "locked/source_manifest.json").read_text()).items():
        assert hashlib.sha256((BASE / "locked" / name).read_bytes()).hexdigest() == digest
    paths = {}
    for r in read(FRESH / "trajectory.jsonl"):
        paths.setdefault((r["bank"], r["case"], r["width"], r["seed"]), []).append(r)
    # Recompute measurements from saved parameter states, independently of JSON labels.
    audited_states = 0
    for config in protocol["configs"]:
        data = weighted_task(config["n"], config["family"], config["parameter"], config["overlap"])
        for factor in protocol["factors"]:
            width = factor * data["capacity"]
            stem = f"{config['case']}_m{width}"
            for time in [0, 20, protocol["end"]]:
                suffix = "landmark" if time <= 20 else "final"
                prefix = f"{time}_" if time <= 20 else ""
                with np.load(FRESH / f"{stem}_{suffix}.npz") as saved:
                    state = tuple(saved[prefix + key] for key in ["w", "a", "c"])
                actual = observe(state, data)
                rows = [
                    next(
                        r
                        for r in paths[protocol["seed"], config["case"], width, seed]
                        if r["time"] == time
                    )
                    for seed in range(protocol["count"])
                ]
                for key, values in actual.items():
                    np.testing.assert_allclose(
                        values, [r[key] for r in rows], rtol=1e-9, atol=1e-10
                    )
                audited_states += len(rows)
    closure, _, opening = episodes(paths)
    truth = {}
    for event, records in [("closure", closure), ("opening", opening)]:
        for r in records:
            truth[event, r["case"], r["width"], r["seed"], r["component"], r.get("horizon", 20)] = r
    forecasts = read(FRESH / "forecasts.jsonl")
    for r in forecasts:
        target = truth[r["event"], r["case"], r["width"], r["seed"], r["component"], r["horizon"]]
        assert r["features"] == target["features"]
        r["actual"] = target["event"]
        for name, prediction in r["predictions"].items():
            model = fit["selected"][name]
            exposure = None if r["event"] == "closure" else [clock(r["horizon"], model["clock"])]
            replay = probability(model, [r["features"]], exposure)[0]
            assert abs(replay - prediction) < 1e-12
    assert len(forecasts) == len(truth)
    result = dict(
        model_sha256=protocol["model_sha256"],
        forecast_sha256=locked["forecast_sha256"],
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remeasured_parameter_states=audited_states,
        scope="Frozen first-passage forecasts; structural support is not finite-budget mastery. Uncertainty resamples original seed indices across all families, widths, components and horizons.",
        groups=[],
        calibration=[],
        by_width=[],
        mastery=[],
    )
    for event in ["closure", "opening"]:
        for split in ["all", "same", "transfer"]:
            subset = [
                r
                for r in forecasts
                if r["event"] == event and (split == "all" or r["split"] == split)
            ]
            if subset:
                result["groups"].append(
                    dict(event=event, split=split, horizon="all", **metrics(subset, event))
                )
            if event == "opening":
                for t in [50, 100, 500, 2000]:
                    subset = [
                        r
                        for r in forecasts
                        if r["event"] == event
                        and r["horizon"] == t
                        and (split == "all" or r["split"] == split)
                    ]
                    if subset:
                        result["groups"].append(
                            dict(event=event, split=split, horizon=t, **metrics(subset, event))
                        )
        subset = [r for r in forecasts if r["event"] == event]
        for bin_index in range(10):
            low = bin_index / 10
            block = [r for r in subset if min(int(r["predictions"][event] * 10), 9) == bin_index]
            if not block:
                continue
            result["calibration"].append(
                dict(
                    event=event,
                    low=float(low),
                    n=len(block),
                    predicted=float(
                        np.average(
                            [r["predictions"][event] for r in block],
                            weights=[r["weight"] for r in block],
                        )
                    ),
                    observed=cluster_interval([r["actual"] for r in block], block),
                )
            )
        for split in ["same", "transfer"]:
            for factor in [1, 2, 3]:
                subset = [
                    r
                    for r in forecasts
                    if r["event"] == event
                    and r["split"] == split
                    and abs(np.exp(r["features"]["log_width"]) - factor) < 1e-6
                ]
                if subset:
                    result["by_width"].append(
                        dict(event=event, split=split, factor=factor, **metrics(subset, event))
                    )
    for case in sorted(set(r["case"] for r in forecasts)):
        for width in sorted(set(r["width"] for r in forecasts if r["case"] == case)):
            end = [p[-1] for k, p in paths.items() if k[1:3] == (case, width)]
            for j in range(end[0]["n"]):
                result["mastery"].append(
                    dict(
                        case=case,
                        width=width,
                        component=j + 1,
                        frequency=end[0]["q"][j],
                        mean_loss=float(np.mean([r["loss"] for r in end])),
                        covered=sum(r["coverage"][j] >= 0.95 for r in end),
                        mastered=sum(abs(r["gap_error"][j]) <= 0.1 for r in end),
                        mean_gap=float(np.mean([abs(r["gap_error"][j]) for r in end])),
                    )
                )
    (BASE / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (BASE / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    plot(result)
    for r in result["groups"]:
        if r["horizon"] == "all":
            print(r["event"], r["split"], r["n"], r["scores"][r["event"]])


def plot(result):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for event, color in [("closure", "#b45309"), ("opening", "#0369a1")]:
        rr = [r for r in result["calibration"] if r["event"] == event]
        pred = np.array([r["predicted"] for r in rr])
        obs = np.array([r["observed"]["mean"] for r in rr])
        bounds = np.array([r["observed"]["ci95"] for r in rr])
        axes[0].errorbar(
            pred,
            obs,
            yerr=np.maximum(np.stack([obs - bounds[:, 0], bounds[:, 1] - obs]), 0),
            fmt="o-",
            capsize=3,
            label=event,
            color=color,
        )
    axes[0].plot([0, 1], [0, 1], ls="--", color="gray")
    axes[0].set(
        xlabel="Frozen predicted probability",
        ylabel="Observed frequency",
        title="Fresh-seed calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend(frameon=False)
    for event, ax in zip(["closure", "opening"], axes[1:]):
        for split, color in [("same", "#0f766e"), ("transfer", "#7c3aed")]:
            rr = [r for r in result["by_width"] if r["event"] == event and r["split"] == split]
            x = [r["factor"] for r in rr]
            obs = np.array([r["observed"]["mean"] for r in rr])
            bounds = np.array([r["observed"]["ci95"] for r in rr])
            ax.errorbar(
                x,
                obs,
                yerr=np.maximum(np.stack([obs - bounds[:, 0], bounds[:, 1] - obs]), 0),
                fmt="o-",
                capsize=3,
                color=color,
                label=f"{split}: observed",
            )
            ax.plot(
                x,
                [r["scores"][event]["mean_prediction"] for r in rr],
                "--",
                color=color,
                label=f"{split}: forecast",
            )
        ax.set(
            xticks=[1, 2, 3],
            xlabel="Width / sufficient target width",
            ylim=(0, 1),
            title=f"{event.capitalize()} by width",
        )
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Frozen support-transition forecasts; intervals resample seed clusters\nOpening pools four deadlines and differently selected closed subsets at each width; not final mastery.",
        fontsize=11,
    )
    fig.savefig(BASE / "fresh_forecasts.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
