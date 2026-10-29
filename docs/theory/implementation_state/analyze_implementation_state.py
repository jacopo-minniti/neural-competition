"""Reconstruct hierarchy transitions, resource ablations, and causal comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from implementation_moments import hierarchy, moments
from toy_option_resources import kernel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/theory/implementation_state"


def read(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def ci(v):
    a = np.array(v)
    rng = np.random.default_rng(296501)
    b = a[rng.integers(len(a), size=(4000, len(a)))].mean(1)
    return dict(mean=float(a.mean()), ci95=np.quantile(b, [0.025, 0.975]).tolist(), n=len(a))


def main():
    provenance = {}
    for folder in BASE.iterdir():
        if not folder.is_dir() or not (folder / "protocol.json").exists():
            continue
        assert (folder / "complete.json").exists(), folder
        protocol = json.loads((folder / "protocol.json").read_text())
        for name, digest in protocol["sources"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol.get("input_hashes", {}).items():
            assert (
                hashlib.sha256((ROOT / protocol["input"] / name).read_bytes()).hexdigest() == digest
            )
        provenance[folder.name] = hashlib.sha256(
            (folder / "protocol.json").read_bytes()
        ).hexdigest()
    rows = read(BASE / "confirmation/trajectory.jsonl") + read(BASE / "wider/trajectory.jsonl")
    result = dict(
        provenance=provenance,
        natural=[],
        numerical=[],
        forecasts=[],
        option_matched=[],
        moment_mixed=[],
        resource_overlap=[],
        interpretation="Mechanism study. Per-refinement directions audit spectral options; they are not independent options. Transition counts censor new closures after first mastery. Correlations do not establish mediation.",
    )
    groups = sorted(set((r["case"], r["width"]) for r in rows))
    for case, m in groups:
        rr = [r for r in rows if (r["case"], r["width"]) == (case, m)]
        paths = [
            sorted([r for r in rr if r["seed"] == seed], key=lambda r: r["time"])
            for seed in range(48)
        ]
        config = {k: rr[0][k] for k in ["n", "ratio", "overlap"]}
        data = hierarchy(**config)
        for j in range(data["n"]):
            counts = {}
            for cutoff in [0.90, 0.95, 0.99]:
                initial = closed = recovered = created = 0
                durations = []
                for path in paths:
                    first_master = next(
                        (r["time"] for r in path if abs(r["gap_error"][j]) <= 0.1), float("inf")
                    )
                    covered = path[0]["coverage"][j] >= cutoff
                    initial += covered
                    missing = next(
                        (
                            r["time"]
                            for r in path
                            if r["time"] < first_master and r["coverage"][j] < cutoff
                        ),
                        None,
                    )
                    did_close = covered and missing is not None
                    closed += did_close
                    if did_close:
                        back = next(
                            (
                                r["time"]
                                for r in path
                                if r["time"] > missing and r["coverage"][j] >= cutoff
                            ),
                            None,
                        )
                        recovered += back is not None
                        if back is not None:
                            durations.append(back - missing)
                    created += (not covered) and any(r["coverage"][j] >= cutoff for r in path[1:])
                counts[str(cutoff)] = dict(
                    initial=initial,
                    demanded_closure=closed,
                    recovered=recovered,
                    recruited_from_initial_miss=created,
                    median_observed_recovery_delay=None
                    if not durations
                    else float(np.median(durations)),
                )
            end = [p[-1] for p in paths]
            budget = []
            for t in [20, 100, 500, 1000, 2000, 4000]:
                at = [r for r in rr if r["time"] == t]
                budget.append(
                    dict(
                        time=t,
                        mean_absolute_gap=float(np.mean([abs(r["gap_error"][j]) for r in at])),
                        mastered=sum(abs(r["gap_error"][j]) <= 0.1 for r in at),
                        mean_strength1000=float(np.mean([r["strength_1000"][j] for r in at])),
                        mean_coverage=float(np.mean([r["coverage"][j] for r in at])),
                    )
                )
            result["natural"].append(
                dict(
                    case=case,
                    width=m,
                    component=j + 1,
                    frequency=float(data["q"][j]),
                    transitions=counts,
                    budgets=budget,
                    terminal_gap=ci([abs(r["gap_error"][j]) for r in end]),
                    terminal_loss=ci([r["loss"] for r in end]),
                    failure_sensitivity={
                        str(c): sum(abs(r["gap_error"][j]) > c for r in end)
                        for c in [0.05, 0.1, 0.2]
                    },
                )
            )
        # The task attribute-overlap knob is not assumed to equal implementation overlap.
        folder = "wider" if rr[0]["factor"] == 4 else "confirmation"
        raw = np.load(BASE / folder / f"{case}_m{m}.npz")
        shared = 2 * data["z"][:, 0] + np.sum(data["q"] * data["coeff"]) * (data["z"] @ data["v"])
        shared = shared * data["root"]
        shared /= np.linalg.norm(shared)
        for t in [0, 20, 100]:
            state = tuple(raw[f"t{t}_{k}"] for k in ["w", "a", "c"])
            overlaps = []
            groupcounts = []
            for seed in range(48):
                resource = moments(tuple(v[seed : seed + 1] for v in state), data)
                groupcounts.append(len(resource["groups"]))
                block = np.array(
                    [
                        g["g"][:, None] * (g["R"] + g["s"] * data["gram"]) * g["g"][None, :]
                        for g in resource["groups"]
                    ]
                )
                common = np.maximum(np.einsum("i,gij,j->g", shared, block, shared), 0)
                detail = np.maximum(
                    np.einsum("ni,gnk,ki->gi", data["directions"], block, data["directions"]), 0
                )
                overlap = np.sum(np.sqrt(common[:, None] * detail), axis=0) / np.sqrt(
                    np.maximum(common.sum() * detail.sum(0), 1e-300)
                )
                overlaps.append(overlap)
            result["resource_overlap"].append(
                dict(
                    case=case,
                    width=m,
                    time=t,
                    mean_group_overlap=np.mean(overlaps, axis=0).tolist(),
                    mean_gate_groups=float(np.mean(groupcounts)),
                )
            )
    for collection in ["step_check", "finer_step"]:
        fine = read(BASE / collection / "trajectory.jsonl")
        for case, m in sorted(set((r["case"], r["width"]) for r in fine)):
            end = sorted(
                [r for r in fine if (r["case"], r["width"], r["time"]) == (case, m, 4000)],
                key=lambda r: r["seed"],
            )
            original = sorted(
                [r for r in rows if (r["case"], r["width"], r["time"]) == (case, m, 4000)],
                key=lambda r: r["seed"],
            )
            result["numerical"].append(
                dict(
                    collection=collection,
                    case=case,
                    width=m,
                    loss=ci([r["loss"] for r in end]),
                    change=ci([r["loss"] - b["loss"] for r, b in zip(end, original)]),
                    component_gap=np.mean(np.abs([r["gap_error"] for r in end]), axis=0).tolist(),
                )
            )
    result["frozen_stability"] = []
    for start in [2, 20, 100]:
        eigensteps = []
        for n, o in [(1, 0), (4, 0), (4, 1)]:
            ratio = 1 if n == 1 else 0.1
            data = hierarchy(n, ratio, o)
            case = f"n{n}_r{ratio:g}_o{o}"
            for m in [2 + 2 * n, 4 + 4 * n]:
                bank = np.load(BASE / "confirmation" / f"{case}_m{m}.npz")
                state = tuple(bank[f"t{start}_{k}"][:12] for k in ["w", "a", "c"])
                eigensteps.extend(0.05 * np.linalg.eigvalsh(kernel(state, data))[:, -1])
        result["frozen_stability"].append(
            dict(
                start=start,
                count=len(eigensteps),
                unstable=int(np.sum(np.array(eigensteps) >= 2)),
                max_step_eigenvalue=float(max(eigensteps)),
            )
        )
    forecasts = read(BASE / "boundary_probe/forecasts.jsonl")
    for start in [2, 20, 100]:
        for h in [0.05, 0.5, 5.0]:
            rr = [r for r in forecasts if r["start"] == start and r["horizon"] == h]

            def ratio(a, b):
                return float(
                    np.sqrt(sum(r[a] ** 2 for r in rr) / max(sum(r[b] ** 2 for r in rr), 1e-300))
                )

            nc = [r for r in rr if not r["crossed"]]
            result["forecasts"].append(
                dict(
                    start=start,
                    horizon=h,
                    n=len(rr),
                    crossed=sum(r["crossed"] for r in rr),
                    response_rmse_ratio=ratio("moment_response_error", "frozen_response_error"),
                    function_rmse_ratio=ratio("moment_function_error", "frozen_function_error"),
                    max_particle_function_error=max(r["particle_function_error"] for r in rr),
                    max_no_crossing_response_error=max(
                        [r["moment_response_error"] for r in nc], default=0
                    ),
                )
            )
    matched = read(BASE / "option_matched/outcomes.jsonl")
    lookup = {(r["case"], r["seed"], r["arm"], r["time"]): r for r in matched}
    for case in sorted(set(r["case"] for r in matched)):
        for t in [0.05, 0.5, 2.0, 10.0, 2000.0]:
            differences = [
                lookup[case, j, "plus", t]["loss"] - lookup[case, j, "minus", t]["loss"]
                for j in range(48)
            ]
            record = dict(
                case=case,
                time=t,
                paired_loss_difference=ci(differences),
                individual_loss_difference_count=sum(abs(v) > 1e-4 for v in differences),
            )
            if t <= 10:
                actual = []
                predicted = []
                for j in range(48):
                    a, b = lookup[case, j, "plus", t], lookup[case, j, "minus", t]
                    actual.append(np.array(a["response"]) - np.array(b["response"]))
                    predicted.append(
                        np.array(a["predicted_response"]) - np.array(b["predicted_response"])
                    )
                actual, predicted = map(np.array, [actual, predicted])
                record["response_contrast_rms"] = float(np.sqrt(np.mean(actual**2)))
                record["moment_contrast_error_ratio"] = float(
                    np.linalg.norm(actual - predicted) / max(np.linalg.norm(actual), 1e-30)
                )
                selected = [
                    j
                    for j in range(48)
                    if not lookup[case, j, "plus", t]["crossed"]
                    and not lookup[case, j, "minus", t]["crossed"]
                ]
                record["no_crossing_pairs"] = len(selected)
                record["no_crossing_error_max"] = (
                    float(np.max(abs(actual[selected] - predicted[selected]))) if selected else None
                )
            result["option_matched"].append(record)
    for collection in ["boundary_probe", "early_moment_mix"]:
        mixed = read(BASE / collection / "mixed_outcomes.jsonl")
        lookup = {(r["case"], r["width"], r["seed"], r["arm"], r["time"]): r for r in mixed}
        for case, m in sorted(set(k[:2] for k in lookup)):
            keys = [k for k in lookup if k[:2] == (case, m) and k[-2:] == ("control", 2000)]
            effects = [lookup[k]["loss"] - lookup[k[:3] + ("mixed", 2000)]["loss"] for k in keys]
            result["moment_mixed"].append(
                dict(
                    collection=collection,
                    case=case,
                    width=m,
                    eligible=len(keys),
                    loss_gain=ci(effects),
                    improvements=sum(v > 1e-4 for v in effects),
                    harms=sum(v < -1e-4 for v in effects),
                    divergence_count=sum(
                        lookup[k].get("first_gate_divergence", -1) >= 0 for k in keys
                    )
                    if collection == "early_moment_mix"
                    else None,
                    max_moment_error=max(
                        r["moment_matching_error"]
                        for r in mixed
                        if (r["case"], r["width"]) == (case, m)
                    ),
                )
            )
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (BASE / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (BASE / "analyze_implementation_state.py").write_bytes(Path(__file__).read_bytes())
    plot(result)
    print(json.dumps({k: result[k] for k in ["numerical", "forecasts", "moment_mixed"]}, indent=2))


def plot(result):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")
    colors = {10: "#b45309", 20: "#0369a1", 40: "#0f766e"}
    for col, j in enumerate(range(1, 5)):
        for m in [10, 20, 40]:
            r = next(
                r
                for r in result["natural"]
                if (r["case"], r["width"], r["component"]) == ("n4_r0.1_o0", m, j)
            )
            times = [b["time"] for b in r["budgets"]]
            ax[0, col].plot(
                times,
                [b["mean_absolute_gap"] for b in r["budgets"]],
                color=colors[m],
                label=f"width {m}",
            )
            ax[1, col].plot(times, [b["mean_strength1000"] for b in r["budgets"]], color=colors[m])
        ax[0, col].set(
            title=f"Refinement {j}; frequency {r['frequency']:.5f}",
            xscale="log",
            yscale="log",
            ylim=(1e-5, 2),
        )
        ax[0, col].axhline(0.1, color="gray", lw=0.7, ls=":")
        ax[1, col].set(xscale="log", ylim=(0, 1.05), xlabel="optimizer time")
    ax[0, 0].set_ylabel("Mean relative contrast error")
    ax[1, 0].set_ylabel("Directional strength, τ=1000")
    ax[0, 0].legend(frameon=False)
    fig.suptitle(
        "A frequency hierarchy separates usable response from remaining demand\nAll refinements remain in the objective; 48 nested-width seeds, separate attributes."
    )
    fig.savefig(BASE / "hierarchy_budget.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for m, color in [(10, colors[10]), (20, colors[20]), (40, colors[40])]:
        rr = [
            next(
                r
                for r in result["natural"]
                if (r["case"], r["width"], r["component"]) == ("n4_r0.1_o0", m, j)
            )
            for j in range(1, 5)
        ]
        ax[0].plot(
            range(1, 5),
            [r["transitions"]["0.95"]["demanded_closure"] / 48 for r in rr],
            marker="o",
            color=color,
            label=f"width {m}: closure",
        )
    ax[0].set(
        xticks=range(1, 5),
        xlabel="Refinement rank (rarer →)",
        ylabel="Fraction of runs",
        title="Demanded closure falls with width\nAll observed closures here subsequently recover",
        ylim=(0, 1),
    )
    ax[0].legend(frameon=False, fontsize=7)
    rr = [r for r in result["moment_mixed"] if r["collection"] == "early_moment_mix"]
    positions = np.arange(len(rr))
    ax[1].bar(
        positions - 0.18,
        [r["improvements"] / r["eligible"] for r in rr],
        0.34,
        label="improves loss",
        color="#0f766e",
    )
    ax[1].bar(
        positions + 0.18,
        [r["harms"] / r["eligible"] for r in rr],
        0.34,
        label="worsens loss",
        color="#b45309",
    )
    ax[1].set(
        xticks=positions,
        xticklabels=[
            f"N={1 if r['case'].startswith('n1') else 4}; o={r['case'][-1]}\nwidth {r['width']}\n{r['eligible']} eligible"
            for r in rr
        ],
        ylabel="Fraction changing terminal loss by >1e−4",
        title="Same options AND moments; different later outcomes\nEarly intervention; no general rescue benefit",
    )
    ax[1].legend(frameon=False)
    fig.savefig(BASE / "resource_interventions.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
