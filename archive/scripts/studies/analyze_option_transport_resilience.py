"""Seed-paired summaries; all displayed errors are half MSE, not option units."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/evidence/option_transport_resilience"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summary(values):
    v = np.asarray(values)
    return dict(
        mean=float(v.mean()),
        se=float(v.std(ddof=1) / np.sqrt(len(v))),
        positive=int(np.sum(v > 0)),
        n=len(v),
    )


def analyze(folder):
    rows = read(folder / "trajectory.jsonl")
    ix = {(r["seed"], r["width"], r["history"], r["arm"], r["step"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    result = dict(means=[], contrasts=[])
    for width in [9, 36]:
        for t in sorted({r["step"] for r in rows}):
            for history in ["full", "peer"]:
                for arm in ["ordinary", "reserve"] + (["double_rate"] if width == 9 else []):
                    result["means"].append(
                        dict(
                            width=width,
                            step=t,
                            history=history,
                            arm=arm,
                            **{
                                key: summary([ix[s, width, history, arm, t][key] for s in seeds])
                                for key in [
                                    "train",
                                    "validation",
                                    "common_value",
                                    "common_strength",
                                ]
                            },
                        )
                    )

            def vals(history, arm, key):
                return np.array([ix[s, width, history, arm, t][key] for s in seeds])

            for key in ["validation", "common_value"]:
                normal = vals("full", "ordinary", key)
                altered = vals("full", "reserve", key)
                peer = vals("peer", "ordinary", key)
                peer_altered = vals("peer", "reserve", key)
                result["contrasts"].append(
                    dict(
                        width=width,
                        step=t,
                        metric=key,
                        full_reserve_minus_ordinary=summary(altered - normal),
                        peer_reserve_minus_ordinary=summary(peer_altered - peer),
                        exposure_effect_ordinary=summary(peer - normal),
                        exposure_effect_reserve=summary(peer_altered - altered),
                        difference_in_differences=summary(
                            (peer_altered - altered) - (peer - normal)
                        ),
                    )
                )
    forecasts = read(folder / "forecasts.jsonl")
    local = read(folder / "local_outcomes.jsonl")
    fidx = {(r["seed"], r["width"], r["history"], r["arm"]): r for r in forecasts}
    result["matching"] = {
        k: max(r[k] for r in forecasts) for k in ["kernel_match_error", "function_match_error"]
    }
    result["smooth_forecast_factor_error"] = max(
        abs(
            r["smooth_demand_change"]
            - 0.25 * fidx[r["seed"], r["width"], r["history"], "ordinary"]["smooth_demand_change"]
        )
        for r in forecasts
        if r["arm"] == "reserve"
    )
    result["local_forecast_rmse"] = {
        arm: float(
            np.sqrt(
                np.mean(
                    [
                        (
                            r["actual"]
                            - fidx[r["seed"], r["width"], r["history"], arm]["smooth_demand_change"]
                        )
                        ** 2
                        for r in local
                        if r["arm"] == arm
                    ]
                )
            )
        )
        for arm in ["ordinary", "reserve"]
    }
    result["width_gap"] = {
        str(t): summary(
            [
                ix[s, 9, "full", "ordinary", t]["validation"]
                - ix[s, 36, "full", "ordinary", t]["validation"]
                for s in seeds
            ]
        )
        for t in [0, 400, 1600, 3400]
    }
    return result


def main():
    results = {name: analyze(OUT / name) for name in ["main", "affine"]}
    (OUT / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    for name, r in results.items():
        print(name, "matching", r["matching"], "width_gap", r["width_gap"])
        for row in r["means"]:
            if row["step"] in [0, 400, 3400]:
                print(
                    {k: row[k] for k in ["width", "step", "history", "arm"]},
                    "loss",
                    row["validation"]["mean"],
                    "value",
                    row["common_value"]["mean"],
                )
        for row in r["contrasts"]:
            if row["step"] in [400, 3400] and row["metric"] == "validation":
                print(row)
    # Publication-style static figure: the hypotheses are directional, not scalar counts.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    main = results["main"]
    for width, color in [(9, "#a54032"), (36, "#246e93")]:
        for arm, ls in [("ordinary", "-"), ("reserve", "--")]:
            rows = [
                r
                for r in main["means"]
                if r["width"] == width and r["arm"] == arm and r["history"] == "full"
            ]
            axes[0].plot(
                [r["step"] for r in rows],
                [r["validation"]["mean"] for r in rows],
                ls,
                color=color,
                label=f"{width}: {arm}",
            )
        for j, metric in [(1, "common_value"), (2, "validation")]:
            rows = [r for r in main["contrasts"] if r["width"] == width and r["metric"] == metric]
            axes[j].errorbar(
                [r["step"] for r in rows],
                [r["difference_in_differences"]["mean"] for r in rows],
                yerr=[r["difference_in_differences"]["se"] for r in rows],
                color=color,
                label=f"Width {width}",
            )
    axes[0].set(title="Original-task learning", ylabel="Held-out half MSE")
    axes[1].set(title="Reserve changes exposure effect", ylabel="Common-demand option value (DiD)")
    axes[2].set(title="Does exposure protection persist?", ylabel="Held-out loss (DiD)")
    for ax in axes:
        ax.set_xlabel("Updates after matched checkpoint")
        ax.axvline(400, color="gray", ls=":", lw=1)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1:]:
        ax.axhline(0, color="black", lw=0.6)
    fig.savefig(OUT / "summary.png", dpi=180)


if __name__ == "__main__":
    main()
