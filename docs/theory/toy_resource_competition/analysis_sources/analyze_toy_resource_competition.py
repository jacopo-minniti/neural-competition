"""Audit saved toy-family experiments, retain negative controls, and draw figures."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from probe_delayed_option_recovery import measure as recovery_measure
from toy_option_resources import task

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/theory/toy_resource_competition"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def mean_ci(values):
    values = np.array(values)
    rng = np.random.default_rng(295701)
    boot = values[rng.integers(len(values), size=(4000, len(values)))].mean(1)
    return dict(
        mean=float(values.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist(), n=len(values)
    )


def correlation(a, b):
    if min(np.std(a), np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main():
    sources = {}
    for folder in BASE.iterdir():
        if not folder.is_dir() or not (folder / "protocol.json").exists():
            continue
        assert (folder / "complete.json").exists(), folder
        protocol = json.loads((folder / "protocol.json").read_text())
        for name, expected in protocol["sources"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
        for name, expected in protocol.get("input_hashes", {}).items():
            assert (
                hashlib.sha256((ROOT / protocol["input"] / name).read_bytes()).hexdigest()
                == expected
            )
        if "input_protocol_hash" in protocol:
            assert (
                hashlib.sha256(
                    (BASE / "recycle_confirmation/protocol.json").read_bytes()
                ).hexdigest()
                == protocol["input_protocol_hash"]
            )
        sources[folder.name] = hashlib.sha256((folder / "protocol.json").read_bytes()).hexdigest()
    rows = read(BASE / "confirmation/trajectory.jsonl")
    ordinary = {
        (r["case"], r["width"], r["seed"], r["time"]): r for r in rows if r["arm"] == "ordinary"
    }
    groups = sorted(set(k[:2] for k in ordinary))
    results = dict(
        protocol_hashes=sources,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status="Descriptive and causal mechanism audit, not prospective outcome forecasts. Thresholds chosen after pilot, applied unchanged to fresh cohort.",
        definition="Substantial obstruction: loss > 5% of initial centered-refinement target energy. Latent coverage >= .95; repeat analyses at .01/.10 obstruction fractions and .90/.99 coverage cutoffs.",
        natural=[],
        pressure=[],
        events=[],
        recycling=[],
        step_check=[],
        initialization=[],
    )
    for case, m in groups:
        data = task(case)
        reference = float(0.5 * np.sum(data["p"] * data["ref_y"] ** 2))
        end = [ordinary[case, m, seed, 2000] for seed in range(64)]
        paths = [
            [r for k, r in sorted(ordinary.items()) if k[:3] == (case, m, seed)]
            for seed in range(64)
        ]
        transition = {}
        for cutoff in [0.90, 0.95, 0.99]:
            initial = np.array([p[0]["coverage"] >= cutoff for p in paths])
            closes = np.array(
                [
                    init and any(r["coverage"] < cutoff for r in p[1:])
                    for init, p in zip(initial, paths)
                ]
            )
            recovered = closes & np.array([p[-1]["coverage"] >= cutoff for p in paths])
            transition[str(cutoff)] = dict(
                initial=int(initial.sum()),
                closed=int(closes.sum()),
                recovered=int(recovered.sum()),
                initial_covered_failures=int(
                    sum(init and r["loss"] > 0.05 * reference for init, r in zip(initial, end))
                ),
            )
        rec = dict(
            case=case,
            width=m,
            reference=reference,
            dense_loss=mean_ci([r["dense_loss"] for r in end]),
            failures={
                str(frac): sum(r["loss"] > frac * reference for r in end)
                for frac in [0.01, 0.05, 0.10]
            },
            transitions=transition,
            null_energy=float(np.mean([r["null_energy"] for r in end])),
            readout_floor=float(np.mean([r["readout_floor"] for r in end])),
        )
        integrals = []
        for path in paths:
            p = [r for r in path if r["time"] >= 20]
            integrals.append(
                float(
                    np.trapezoid(
                        [r["loss"] - r["loaded_gain_20"] for r in p], [r["time"] for r in p]
                    )
                    / (2000 - 20)
                )
            )
        rec["time_mean_unmet_response_demand20"] = float(np.mean(integrals))
        raw = np.load(BASE / "confirmation" / f"{case}_{m}_states.npz")
        state = tuple(raw[f"ordinary_100_{key}"] for key in ["w", "a", "c"])
        recovery = recovery_measure(state, data["x"], data["p"], data["y"])
        certified = (recovery["blind_recovery_clock"] > 1900) & (
            recovery["blind_floor_lower"] > 0.05 * reference
        )
        rec["flow_blind_bound_count"] = int(certified.sum())
        rec["flow_bound_violations_in_euler"] = int(
            np.sum(
                certified
                & (np.array([r["loss"] for r in end]) < recovery["blind_floor_lower"] - 1e-9)
            )
        )
        rec["common_initial_energy"] = float(np.mean([p[0]["common_energy"] for p in paths]))
        rec["refinement_initial_energy"] = float(
            np.mean([p[0]["refinement_energy"] for p in paths])
        )
        results["natural"].append(rec)
        for arm in ["early_no_common", "matched_movement", "late_no_common"]:
            arm_rows = sorted(
                [
                    r
                    for r in rows
                    if (r["case"], r["width"], r["arm"], r["time"]) == (case, m, arm, 2000)
                ],
                key=lambda r: r["seed"],
            )
            results["pressure"].append(
                dict(
                    case=case,
                    width=m,
                    arm=arm,
                    dense_loss=float(np.mean([r["dense_loss"] for r in arm_rows])),
                    excess_over_ordinary=mean_ci(
                        [r["dense_loss"] - b["dense_loss"] for r, b in zip(arm_rows, end)]
                    ),
                )
            )
    event_rows = read(BASE / "events_confirmation/events.jsonl")
    first = {}
    for r in event_rows:
        if r["before"] - r["full"] > 0.03:
            first.setdefault((r["case"], r["width"], r["seed"]), r)
    for case, m in groups:
        events = [r for k, r in first.items() if k[:2] == (case, m)]
        classifications = collections.Counter()
        failed = collections.Counter()
        initial_covered = collections.Counter()
        early = collections.Counter()
        reference = 0.5 * np.sum(task(case)["p"] * task(case)["ref_y"] ** 2)
        for r in events:
            code = ("C" if r["before"] - r["common"] > 0.03 else "") + (
                "R" if r["before"] - r["refinement"] > 0.03 else ""
            ) or "joint"
            classifications[code] += 1
            failed[code] += int(ordinary[case, m, r["seed"], 2000]["loss"] > 0.05 * reference)
            initial_covered[code] += ordinary[case, m, r["seed"], 0]["coverage"] >= 0.95
            early[code] += r["common_energy"] > r["refinement_energy"]
        results["events"].append(
            dict(
                case=case,
                width=m,
                first_large_loss_counts=dict(classifications),
                eventual_failures=dict(failed),
                initially_covered=dict(initial_covered),
                common_demand_dominates=dict(early),
            )
        )
    rr = read(BASE / "recycle_confirmation/outcomes.jsonl")
    paired = {(r["case"], r["width"], r["seed"], r["repeat"], r["arm"], r["time"]): r for r in rr}
    for case, m in groups:
        seeds = sorted(set(r["seed"] for r in rr if (r["case"], r["width"]) == (case, m)))
        if not seeds:
            continue
        gains, control_losses, treatment_losses, ordinary_losses = [], [], [], []
        stats = {
            k: []
            for k in [
                "refine_strength_20",
                "min_refine_strength_20",
                "breadth_20",
                "implementation_diversity",
                "loaded_gain_20",
                "refine_strength_1000",
                "min_refine_strength_1000",
                "breadth_1000",
                "loaded_gain_1000",
                "coverage",
            ]
        }
        better = worse = total = 0
        positive_response_better = positive_response_count = 0
        for seed in seeds:
            trial_gains = []
            changes = {k: [] for k in stats}
            for repeat in range(4):
                c0, t0, ce, te = [
                    paired[case, m, seed, repeat, arm, t]
                    for arm, t in [
                        ("control", 20),
                        ("recycled", 20),
                        ("control", 2000),
                        ("recycled", 2000),
                    ]
                ]
                g = ce["dense_loss"] - te["dense_loss"]
                trial_gains.append(g)
                control_losses.append(ce["dense_loss"])
                treatment_losses.append(te["dense_loss"])
                better += g > 1e-4
                worse += g < -1e-4
                total += 1
                positive_response_count += t0["loaded_gain_20"] - c0["loaded_gain_20"] > 1e-5
                positive_response_better += (g > 1e-4) and (
                    t0["loaded_gain_20"] - c0["loaded_gain_20"] > 1e-5
                )
                for k in stats:
                    changes[k].append(t0[k] - c0[k])
            gains.append(np.mean(trial_gains))
            ordinary_losses.append(ordinary[case, m, seed, 2000]["dense_loss"])
            for k in stats:
                stats[k].append(np.mean(changes[k]))
        results["recycling"].append(
            dict(
                case=case,
                width=m,
                eligible=len(seeds),
                total_seeds=64,
                repeats=4,
                mean_control_loss=float(np.mean(control_losses)),
                mean_recycled_loss=float(np.mean(treatment_losses)),
                mean_ordinary_loss_on_eligible=float(np.mean(ordinary_losses)),
                paired_gain_seed_bootstrap=mean_ci(gains),
                improved_trials=int(better),
                worsened_trials=int(worse),
                trials=total,
                positive_immediate_loaded_gain_trials=int(positive_response_count),
                improved_and_positive_immediate_gain=int(positive_response_better),
                descriptive_correlations={k: correlation(gains, v) for k, v in stats.items()},
                max_function_error=max(
                    r["function_error"] for r in rr if (r["case"], r["width"]) == (case, m)
                ),
                max_trace_error=max(
                    r["trace_error"] for r in rr if (r["case"], r["width"]) == (case, m)
                ),
            )
        )
    steps = read(BASE / "step_check/trajectory.jsonl")
    for case, m in sorted(set((r["case"], r["width"]) for r in steps)):
        end = sorted(
            [r for r in steps if (r["case"], r["width"], r["time"]) == (case, m, 2000)],
            key=lambda r: r["seed"],
        )
        results["step_check"].append(
            dict(
                case=case,
                width=m,
                mean_dense_loss=float(np.mean([r["dense_loss"] for r in end])),
                paired_change=mean_ci(
                    [
                        r["dense_loss"] - ordinary[case, m, r["seed"], 2000]["dense_loss"]
                        for r in end
                    ]
                ),
            )
        )
    initial = np.load(BASE / "initial/initial.npz")
    for case in ["discrete", "continuous", "separate_attribute", "two_refinements"]:
        for m in [2, 4, 6, 8, 12, 16]:
            cov, strength = initial[f"{case}_{m}_coverage"], initial[f"{case}_{m}_strength"]
            results["initialization"].append(
                dict(
                    case=case,
                    width=m,
                    n=len(cov),
                    coverage95=float(np.mean(cov >= 0.95)),
                    coverage999999=float(np.mean(cov >= 1 - 1e-6)),
                    strength_mean=float(strength.mean()),
                    strength_q05=float(np.quantile(strength, 0.05)),
                    strength_q95=float(np.quantile(strength, 0.95)),
                )
            )
    analysis_sources = BASE / "analysis_sources"
    analysis_sources.mkdir(exist_ok=True)
    results["analysis_dependency_hashes"] = {}
    for name in [
        Path(__file__).name,
        "toy_option_resources.py",
        "probe_delayed_option_recovery.py",
        "explore_stable_distinctions.py",
        "explore_transported_gates.py",
    ]:
        source = Path(__file__).with_name(name)
        results["analysis_dependency_hashes"][name] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        (analysis_sources / name).write_bytes(source.read_bytes())
    fine_events = read(BASE / "events_step_check/events.jsonl")
    first_fine = {}
    for r in fine_events:
        if r["before"] - r["full"] > 0.03:
            first_fine.setdefault((r["case"], r["width"], r["seed"]), r)
    results["events_step_check"] = []
    for case, m in sorted(set((r["case"], r["width"]) for r in fine_events)):
        counter = collections.Counter()
        for key, r in first_fine.items():
            if key[:2] != (case, m):
                continue
            code = ("C" if r["before"] - r["common"] > 0.03 else "") + (
                "R" if r["before"] - r["refinement"] > 0.03 else ""
            ) or "joint"
            counter[code] += 1
        results["events_step_check"].append(
            dict(case=case, width=m, first_large_loss_counts=dict(counter))
        )
    (BASE / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    plot(results, ordinary)
    print(
        json.dumps(
            {k: results[k] for k in ["natural", "events", "recycling", "step_check"]}, indent=2
        )
    )


def plot(results, ordinary):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    names = ["discrete", "continuous", "separate_attribute"]
    colors = ["#b45309", "#0369a1"]
    for col, name in enumerate(names):
        for j, m in enumerate([4, 8]):
            rr = sorted(
                [r for r in results["natural"] if r["case"] == name and r["width"] == m],
                key=lambda r: r["width"],
            )[0]
            tr = rr["transitions"]["0.95"]
            vals = [tr["initial"], tr["closed"], tr["recovered"], rr["failures"]["0.05"]]
            axes[0, col].bar(
                np.arange(4) + (j - 0.5) * 0.36,
                np.array(vals) / 64,
                width=0.34,
                label=f"width {m}",
                color=colors[j],
            )
            times = sorted(set(k[-1] for k in ordinary if k[:2] == (name, m)))
            # Both actual residual loading and response availability are visible.
            mean_loss = [
                np.mean(
                    [r["loss"] for k, r in ordinary.items() if k[:2] == (name, m) and k[-1] == t]
                )
                for t in times
            ]
            mean_unmet = [
                np.mean(
                    [
                        r["loss"] - r["loaded_gain_20"]
                        for k, r in ordinary.items()
                        if k[:2] == (name, m) and k[-1] == t
                    ]
                )
                for t in times
            ]
            axes[1, col].plot(
                times, np.maximum(mean_loss, 1e-8), color=colors[j], label=f"loss, width {m}"
            )
            axes[1, col].plot(
                times,
                np.maximum(mean_unmet, 1e-8),
                color=colors[j],
                ls="--",
                label=f"unmet τ=20, width {m}",
            )
        axes[0, col].set(
            title=name.replace("_", " "),
            xticks=range(4),
            xticklabels=[
                "initial\ncoverage",
                "later\nclosure",
                "recovered\nat end",
                "substantial\nerror",
            ],
            ylim=(0, 1.05),
        )
        axes[1, col].set(xscale="symlog", yscale="log", xlabel="optimizer time", ylim=(1e-6, 5))
    axes[0, 0].set_ylabel("Fraction of 64 runs")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_ylabel("Mean loss / unmet response demand")
    axes[1, 2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Latent initial support is often present; its later loss and recovery differ with width\nCoverage means ≥95% of fixed refinement direction. Dashed curves are descriptive, not forecasts.",
        fontsize=12,
    )
    fig.savefig(BASE / "natural_transitions.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    chosen = [
        r
        for r in results["recycling"]
        if (r["case"], r["width"])
        in [("discrete", 4), ("separate_attribute", 8), ("two_refinements", 6)]
    ]
    x = np.arange(len(chosen))
    gains = np.array([r["paired_gain_seed_bootstrap"]["mean"] for r in chosen])
    bounds = np.array([r["paired_gain_seed_bootstrap"]["ci95"] for r in chosen])
    axes[0].errorbar(
        x,
        gains,
        yerr=np.stack([gains - bounds[:, 0], bounds[:, 1] - gains]),
        fmt="o",
        capsize=5,
        color="#0f766e",
        label="mean and 95% seed bootstrap interval",
    )
    axes[0].axhline(0, color="#6b7280", lw=1)
    axes[0].set(
        xticks=x,
        xticklabels=[
            f"{r['case'].replace('_', ' ')}\nwidth {r['width']}; {r['eligible']}/64 eligible"
            for r in chosen
        ],
        ylabel="Paired terminal loss reduction",
        title="Recycling: same width, predictions and kernel trace",
    )
    axes[0].legend(frameon=False, fontsize=8)
    chosen = [
        r
        for r in results["natural"]
        if (r["case"], r["width"])
        in [("discrete", 4), ("continuous", 4), ("separate_attribute", 8)]
    ]
    for j, arm in enumerate(["ordinary", "early_no_common", "matched_movement"]):
        val = []
        for r in chosen:
            val.append(
                r["dense_loss"]["mean"]
                if arm == "ordinary"
                else next(
                    s["dense_loss"]
                    for s in results["pressure"]
                    if (s["case"], s["width"], s["arm"]) == (r["case"], r["width"], arm)
                )
            )
        axes[1].bar(x + (j - 1) * 0.25, val, 0.23, label=arm.replace("_", " "))
    axes[1].set(
        xticks=x,
        xticklabels=[f"{r['case'].replace('_', ' ')}\nwidth {r['width']}" for r in chosen],
        ylabel="Mean terminal loss",
        title="Removing common pressure can impair useful creation",
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(BASE / "interventions.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
