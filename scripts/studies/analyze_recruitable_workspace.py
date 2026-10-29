"""Seed-paired summaries; never count checkpoints as independent repetitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/theory/recruitable_workspace"


def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def stat(x):
    x = np.asarray(x)
    return dict(
        mean=float(x.mean()),
        se=float(x.std(ddof=1) / np.sqrt(len(x))),
        n=len(x),
        positive=int((x > 1e-12).sum()),
        values=x.tolist(),
    )


def summarize(folder):
    p = OUT / folder
    prot = json.loads((p / "protocol.json").read_text())
    seeds = prot["seeds"]
    ordinary = read(p / "ordinary.jsonl")
    assays = read(p / "assays.jsonl")
    interv = read(p / "interventions.jsonl")

    def vec(rows, **filters):
        return np.array(
            [
                next(
                    r["validation"]
                    for r in rows
                    if r["seed"] == s and all(r[k] == v for k, v in filters.items())
                )
                for s in seeds
            ]
        )

    result = dict(
        seeds=seeds, affine=prot["affine"], ordinary={}, assay={}, interventions={}, contrasts={}
    )
    for step in [0, 200, 600, 1200, 3000]:
        result["ordinary"][str(step)] = {
            str(m): stat(vec(ordinary, width=m, step=step)) for m in [9, 36]
        }
    for m in [9, 36]:
        for h in ["initial", "pause", "peer", "full"]:
            result["assay"][f"{m}_{h}"] = {
                str(t): stat(vec(assays, width=m, history=h, step=t))
                for t in [0, 20, 100, 300, 1200]
            }
    for arm in ["clone", "fresh", "fresh_gain_matched", "fast", "slow_time_matched"]:
        t = 2400 if arm == "slow_time_matched" else 1200
        result["interventions"][arm] = stat(vec(interv, arm=arm, step=t))
    for t in [300, 1200]:

        def v(m, h):
            return vec(assays, width=m, history=h, step=t)

        for m in [9, 36]:
            result["contrasts"][f"peer_minus_pause_{m}_t{t}"] = stat(v(m, "peer") - v(m, "pause"))
        result["contrasts"][f"pressure_extra_narrow_t{t}"] = stat(
            (v(9, "peer") - v(9, "pause")) - (v(36, "peer") - v(36, "pause"))
        )
        for h in ["initial", "pause", "peer", "full"]:
            result["contrasts"][f"width_gap_{h}_t{t}"] = stat(v(9, h) - v(36, h))
        # Complement of the exposed region, all three choices rotated over seeds.
        q = {}
        for m in [9, 36]:
            q[m] = []
            for s in seeds:
                both = {
                    h: next(
                        r["validation_regions"]
                        for r in assays
                        if r["seed"] == s
                        and r["width"] == m
                        and r["history"] == h
                        and r["step"] == t
                    )
                    for h in ["pause", "peer"]
                }
                k = [i for i in range(3) if i != s % 3]
                q[m].append(np.mean(np.array(both["peer"])[k] - np.array(both["pause"])[k]))
            result["contrasts"][f"withheld_peer_minus_pause_{m}_t{t}"] = stat(q[m])
        result["contrasts"][f"withheld_pressure_extra_narrow_t{t}"] = stat(
            np.array(q[9]) - np.array(q[36])
        )
    base = vec(assays, width=9, history="peer", step=1200)
    for arm in ["fresh", "fresh_gain_matched", "fast", "slow_time_matched"]:
        t = 2400 if arm == "slow_time_matched" else 1200
        result["contrasts"][f"base_minus_{arm}"] = stat(base - vec(interv, arm=arm, step=t))
    result["contrasts"]["fresh_minus_fast"] = stat(
        vec(interv, arm="fresh", step=1200) - vec(interv, arm="fast", step=1200)
    )
    result["clone_max_error"] = float(np.max(np.abs(base - vec(interv, arm="clone", step=1200))))
    result["contrasts"]["ordinary_width_gap_3000"] = stat(
        vec(ordinary, width=9, step=3000) - vec(ordinary, width=36, step=3000)
    )
    result["contrasts"]["ordinary_gain_wide_minus_narrow_200_3000"] = stat(
        (vec(ordinary, width=36, step=200) - vec(ordinary, width=36, step=3000))
        - (vec(ordinary, width=9, step=200) - vec(ordinary, width=9, step=3000))
    )
    return result


def main():
    result = {
        k: summarize(k)
        for k in ["pilot", "confirmation", "extra_refinement", "affine", "active_initialization"]
    }
    for rec, longkey in [("recovery", "long"), ("recovery_active", "long_active")]:
        rows = read(OUT / rec / "assays.jsonl")
        lon = read(OUT / rec / "long.jsonl")
        result[rec] = {}
        for arm in sorted({r["arm"] for r in rows}):
            ts = sorted({r["step"] for r in rows if r["arm"] == arm})
            result[rec][arm] = {
                str(t): stat([r["validation"] for r in rows if r["arm"] == arm and r["step"] == t])
                for t in ts
            }
        result[longkey] = {
            f"{m}_{rate}": {
                str(t): stat(
                    [
                        r["validation"]
                        for r in lon
                        if r["width"] == m and r["rate"] == rate and r["step"] == t
                    ]
                )
                for t in [3000, 6000, 12000, 30000]
            }
            for m, rate in [(9, 0.06), (36, 0.06), (9, 0.12)]
        }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, d in result.items():
        if name in ["long", "recovery", "long_active", "recovery_active"]:
            print(name, {k: {t: round(z["mean"], 6) for t, z in v.items()} for k, v in d.items()})
        else:
            print(
                name,
                {
                    k: {j: round(z[j], 7) for j in ["mean", "se", "positive"]}
                    for k, z in d["contrasts"].items()
                },
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = result["confirmation"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for m, color in [(9, "#2166ac"), (36, "#b35806")]:
        ts = [0, 200, 600, 1200, 3000]
        axs[0].plot(
            ts, [d["ordinary"][str(t)][str(m)]["mean"] for t in ts], label=f"Width {m}", color=color
        )
        hs = ["initial", "pause", "peer", "full"]
        axs[1].errorbar(
            range(4),
            [d["assay"][f"{m}_{h}"]["1200"]["mean"] for h in hs],
            yerr=[d["assay"][f"{m}_{h}"]["1200"]["se"] for h in hs],
            label=f"Width {m}",
            color=color,
            marker="o",
        )
    axs[0].set(title="Ordinary task learning", xlabel="GD updates", ylabel="Held-out half MSE")
    axs[1].set(
        title="Same residual, 1,200-step correction",
        xticks=range(4),
        xticklabels=["Initial", "Paused", "One context", "All contexts"],
        ylabel="Held-out correction error",
    )
    arms = ["clone", "fresh", "fresh_gain_matched", "fast", "slow_time_matched"]
    axs[2].bar(
        range(5),
        [d["interventions"][a]["mean"] for a in arms],
        color=["#888", "#b35806", "#d9a66d", "#2166ac", "#77aadd"],
    )
    axs[2].errorbar(
        range(5),
        [d["interventions"][a]["mean"] for a in arms],
        yerr=[d["interventions"][a]["se"] for a in arms],
        fmt="none",
        color="black",
    )
    axs[2].set(
        title="Resource and rate controls",
        xticks=range(5),
        xticklabels=["Clone", "Fresh", "Fresh\ngain match", "2× rate", "½ rate\n2× steps"],
        ylabel="Held-out correction error",
    )
    axs[0].legend()
    axs[1].legend()
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "summary.png", dpi=180)


if __name__ == "__main__":
    main()
