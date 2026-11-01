"""Audit delayed-demand chronology, forecast coverage, and recovery counterexamples."""

import json
from pathlib import Path

import numpy as np
from analyze_option_transport_resilience import read, summary

ROOT = Path(__file__).resolve().parents[2] / "docs/evidence/option_transport_resilience"


def analyze(folder):
    outcomes = read(folder / "outcomes.jsonl")
    forecasts = read(folder / "forecasts.jsonl")
    by = {(r["case"], r["width"], r["seed"]): r for r in outcomes}
    raw = np.load(folder / "trajectories.npz")
    result = {}
    for case in sorted({r["case"] for r in outcomes}):
        result[case] = {}
        for m in [4, 8]:
            rows = [r for r in outcomes if r["case"] == case and r["width"] == m]
            # This is a numerical summary marker, not the definition of an option.
            failed = [r for r in rows if r["loss"] > 1e-7]
            row = dict(
                loss=summary([r["loss"] for r in rows]),
                dense_loss=summary([r["dense_loss"] for r in rows]),
                above_1e_minus7=len(failed),
                initially_missing=sum(r["first_null_time"] == 0 for r in failed),
                dynamic_loss_before_crossover=sum(
                    0 < r["first_null_time"] < r["crossover_time"] for r in failed
                ),
                forecasts={},
            )
            for t in [20, 100]:
                fs = [
                    r for r in forecasts if r["case"] == case and r["width"] == m and r["time"] == t
                ]
                stats = {}
                for kind, exitkey in [("cell", "first_gate_exit"), ("blind", "first_blind_exit")]:
                    floor = "cell_floor_lower" if kind == "cell" else "blind_floor_lower"
                    clock = "recovery_clock" if kind == "cell" else "blind_recovery_clock"
                    if floor not in fs[0]:
                        continue
                    pos = [r for r in fs if r[floor] > 1e-8]
                    cover = [r for r in pos if r[clock] > 2000 - t]
                    stats[kind] = dict(
                        positive_floors=len(pos),
                        floor_mean=float(np.mean([r[floor] for r in fs])),
                        short_100_coverage=sum(r[clock] > 100 for r in pos),
                        endpoint_coverage=len(cover),
                        endpoint_floor_violations=sum(
                            by[case, m, r["seed"]]["loss"] < r[floor] - 1e-8 for r in cover
                        ),
                        exit_bound_violations=sum(
                            0 <= by[case, m, r["seed"]][exitkey][f"{t}.0"] < r[clock] - 1e-7
                            for r in pos
                        ),
                        uncertified_floor_overestimates=sum(
                            by[case, m, r["seed"]]["loss"] < r[floor] - 1e-7 for r in pos
                        ),
                    )
                row["forecasts"][str(t)] = stats
            key = f"{case}_{m}"
            cols = list(raw[key + "_columns"])
            tr = raw[key + "_trace"]
            row["energy_trace"] = {
                str(t): dict(
                    main=float(tr[i, :, cols.index("main_energy")].mean()),
                    detail=float(tr[i, :, cols.index("detail_energy")].mean()),
                    loss=float(tr[i, :, cols.index("loss")].mean()),
                )
                for i, t in enumerate(raw[key + "_times"])
                if t in [0, 2, 4, 6, 8, 10, 20, 100, 2000]
            }
            result[case][str(m)] = row
        narrow = [r for r in outcomes if r["case"] == case and r["width"] == 4]
        result[case]["paired_loss_gap"] = summary(
            [r["loss"] - by[case, 8, r["seed"]]["loss"] for r in narrow]
        )
    return result


def main():
    result = {
        name: analyze(ROOT / name)
        for name in ["delayed", "delayed_confirmation", "delayed_half_step"]
    }
    (ROOT / "delayed_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    for cohort, info in result.items():
        print(cohort)
        for case, rr in info.items():
            for m in ["4", "8"]:
                r = rr[m]
                print(
                    case,
                    m,
                    "loss",
                    r["loss"],
                    "dynamic_before",
                    r["dynamic_loss_before_crossover"],
                    "forecasts100",
                    r["forecasts"]["100"],
                )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    raw = np.load(ROOT / "delayed_confirmation/trajectories.npz")
    for m, color in [(4, "#a54032"), (8, "#246e93")]:
        key = f"delayed_{m}"
        cols = list(raw[key + "_columns"])
        tr = raw[key + "_trace"]
        tt = raw[key + "_times"]
        for col, ls in [("main_energy", "-"), ("detail_energy", "--")]:
            axes[0].plot(
                tt[tt <= 20],
                tr[tt <= 20, :, cols.index(col)].mean(1),
                ls,
                color=color,
                label=f"m={m}, {col.split('_')[0]}",
            )
    axes[0].set_yscale("log")
    axes[0].set(
        title="Shared rule before contextual refinement",
        xlabel="Gradient-flow time",
        ylabel="Residual energy",
    )
    cases = ["delayed", "rarer", "continuous", "no_refinement", "larger_init"]
    for m, offset, color in [("4", -0.15, "#a54032"), ("8", 0.15, "#246e93")]:
        rr = [result["delayed_confirmation"][case][m]["loss"] for case in cases]
        axes[1].bar(
            np.arange(5) + offset,
            [r["mean"] for r in rr],
            0.3,
            yerr=[r["se"] for r in rr],
            color=color,
            label=f"Width {m}",
        )
    axes[1].set_xticks(
        range(5), ["Delayed", "Rarer", "Continuous", "No detail", "Larger init"], rotation=25
    )
    axes[1].set(title="Fresh-cohort outcome at time 2,000", ylabel="Half MSE")
    rr = [result["delayed_confirmation"][case]["4"]["forecasts"]["100"] for case in cases]
    for kind, offset, color in [("cell", -0.15, "#8f9396"), ("blind", 0.15, "#498b76")]:
        axes[2].bar(
            np.arange(5) + offset,
            [r[kind]["endpoint_coverage"] for r in rr],
            0.3,
            color=color,
            label=f"{kind} bound",
        )
    axes[2].set_xticks(
        range(5), ["Delayed", "Rarer", "Continuous", "No detail", "Larger init"], rotation=25
    )
    axes[2].set(
        title="Conservative long-budget coverage", ylabel="Width-4 cases covered (out of 64)"
    )
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(ROOT / "delayed_summary.png", dpi=180)


if __name__ == "__main__":
    main()
