#!/usr/bin/env python3
"""Cumulative, activity-matched and loss-matched interaction diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_competition.trajectory import FIELDS, VIEWS

COLORS = {8: "#2563eb", 16: "#df6b20"}
INDEX = {key: i for i, key in enumerate(FIELDS)}


def ratio(a, b):
    return a / b if b > 0 else np.nan


def packed(sums):
    g, h, p, c = [sums[INDEX[key]] for key in ["solo_gain", "solo_harm", "penalty", "credit"]]
    return dict(
        gain=g,
        harm=h,
        penalty=p,
        credit=c,
        net=p - c,
        S_plus=ratio(p, g),
        S_minus=ratio(c, g),
        S_net=ratio(p - c, g),
        solo_harm_ratio=ratio(h, g),
        retained_gain=ratio(g - h - p + c, g),
        frequency=ratio(sums[6], sums[7]),
        helpful_frequency=ratio(sums[9], sums[8]),
        strong_frequency=ratio(sums[10], 2 * sums[8]),
        penalty_per_energy=ratio(p, sums[12]),
    )


def interval(values, seed=741):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(len(values), size=(4000, len(values)))].mean(1)
    return values.mean(), *np.quantile(boot, [0.025, 0.975])


def curve(ax, frame, x, y, *, factor=1.0, log=False):
    for width, part in frame.groupby("width"):
        table = part.pivot(index="seed", columns=x, values=y).sort_index(axis=1)
        xx, values = table.columns.to_numpy(), table.to_numpy() * factor
        mean = np.nanmean(values, axis=0)
        rng = np.random.default_rng(741)
        weights = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), 1500)
        finite = np.isfinite(values)
        den = weights @ finite.astype(float)
        boot = (weights @ np.nan_to_num(values)) / np.where(den > 0, den, np.nan)
        low, high = np.nanquantile(boot, [0.025, 0.975], axis=0)
        if log:
            mean, low, high = [np.maximum(v, 1e-14) for v in [mean, low, high]]
            ax.set_yscale("log")
        ax.plot(xx, mean, color=COLORS[width], label=f"width {width}", lw=1.8)
        ax.fill_between(xx, low, high, color=COLORS[width], alpha=0.16, linewidth=0)


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def collect(folder):
    config = json.loads((folder / "config.json").read_text())
    if json.loads((folder / "state.json").read_text())["status"] != "complete":
        raise ValueError(f"Training is not complete: {folder}")
    raw = np.load(folder / "updates.npy", mmap_mode="r")
    steps = len(raw)
    points = np.unique(
        np.r_[
            np.arange(1, min(101, steps + 1)),
            np.geomspace(1, steps, 750).astype(int),
            np.arange(1000, steps + 1, 1000),
            steps,
        ]
    )
    caps = raw[:, :, :, INDEX["solo_gain"]].sum(0).min(0)
    bins = {"overview": np.geomspace(1e-9, 2.0, 76), "primary": np.geomspace(0.02, 0.2, 13)}
    cumulative, windows, phases, progress, finals, audits = [], [], [], [], [], []
    label = config["optimizer"]
    for i, model in enumerate(config["models"]):
        reference_mse = 2 * np.asarray(raw[:, i, 1, 0])
        for v, view in enumerate(VIEWS):
            a = np.array(raw[:, i, v, :])
            base = dict(optimizer=label, view=view, **model)
            sums = a.sum(0)
            finals.append(dict(**base, **packed(sums), initial_loss=a[0, 0], final_loss=a[-1, 1]))
            cs = np.cumsum(a, axis=0)
            for t in points:
                cumulative.append(
                    dict(
                        **base,
                        updates=int(t),
                        **packed(cs[t - 1]),
                        reference_mse=reference_mse[t - 1],
                    )
                )
            for start in range(0, steps, 1000):
                window = a[start : min(start + 1000, steps)]
                windows.append(
                    dict(**base, updates=min(start + 1000, steps), **packed(window.sum(0)))
                )
            u = np.r_[0.0, cs[:, 2]]
            grid = np.linspace(0, caps[v], 401)
            p = np.interp(grid, u, np.r_[0.0, cs[:, 4]])
            c = np.interp(grid, u, np.r_[0.0, cs[:, 5]])
            for x, pp, cc in zip(grid, p, c):
                progress.append(
                    dict(
                        **base,
                        U=x,
                        penalty=pp,
                        credit=cc,
                        net=pp - cc,
                        S_plus=ratio(pp, x),
                        S_net=ratio(pp - cc, x),
                    )
                )
            for bin_set, edges in bins.items():
                assignment = np.searchsorted(edges, reference_mse, side="right") - 1
                good = (assignment >= 0) & (assignment < len(edges) - 1)
                ids = assignment[good]
                counts = np.bincount(ids, minlength=len(edges) - 1)
                totals = np.stack(
                    [
                        np.bincount(ids, weights=a[good, k], minlength=len(edges) - 1)
                        for k in range(a.shape[1])
                    ],
                    axis=1,
                )
                for k, n in enumerate(counts):
                    if n:
                        phases.append(
                            dict(
                                **base,
                                bin_set=bin_set,
                                bin=k,
                                lower=edges[k],
                                upper=edges[k + 1],
                                mse=np.sqrt(edges[k] * edges[k + 1]),
                                visits=int(n),
                                **packed(totals[k]),
                            )
                        )
            local = a[:, 1] - a[:, 0] - (a[:, 3] - a[:, 2] + a[:, 4] - a[:, 5])
            telescope = sums[3] - sums[2] + sums[4] - sums[5] - (a[-1, 1] - a[0, 0])
            adjacency = np.max(np.abs(a[1:, 0] - a[:-1, 1])) if view == "reference" else np.nan
            audits.append(
                dict(
                    **base,
                    max_local_error=float(np.abs(local).max()),
                    telescope_error=float(telescope) if view == "reference" else np.nan,
                    reference_adjacency_error=adjacency,
                )
            )
    return [pd.DataFrame(x) for x in [cumulative, windows, phases, progress, finals, audits]]


def phase_summary(phases):
    rows = []
    for identity, group in phases.groupby(["optimizer", "view", "bin_set", "bin"]):
        group = group[np.isfinite(group.S_plus)]
        common = group.groupby("seed").width.nunique()
        common = common[common == 2].index
        group = group[group.seed.isin(common)]
        for width, part in group.groupby("width"):
            base = dict(zip(["optimizer", "view", "bin_set", "bin"], identity))
            row = dict(
                **base,
                width=width,
                paired_seeds=len(common),
                mse=part.mse.iloc[0],
                lower=part.lower.iloc[0],
                upper=part.upper.iloc[0],
            )
            for metric in ["S_plus", "S_net", "helpful_frequency", "frequency", "strong_frequency"]:
                mean, low, high = interval(part[metric])
                row.update({metric: mean, metric + "_low": low, metric + "_high": high})
            rows.append(row)
    return pd.DataFrame(rows)


def plots(label, cumulative, windows, phases, progress, losses, output, config):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    c = cumulative[cumulative.optimizer == label]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.1), layout="constrained")
    curve(axes[0, 0], losses, "updates", "train_mse", log=True)
    axes[0, 0].set_title("Full training MSE")
    specs = [
        ("penalty", "Accumulated unfavorable penalty", 1.0),
        ("net", "Accumulated net interaction penalty", 1.0),
        ("S_plus", "Cumulative penalty / standalone gains", 100.0),
        ("S_minus", "Cumulative favorable credit / gains", 100.0),
        ("S_net", "Cumulative net penalty / gains", 100.0),
    ]
    for ax, (metric, title, factor) in zip(axes.flat[1:], specs):
        curve(ax, c[c.view == "reference"], "updates", metric, factor=factor)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("% of standalone gains" if factor == 100 else "Half-MSE units")
    for ax in axes.flat:
        ax.set_xlabel("Training updates")
        ax.grid(alpha=0.15)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0, 0].legend()
    n_seeds = losses.seed.nunique()
    fig.suptitle(
        f"{label.upper()}: {config['steps']:,} actual updates, {n_seeds} paired seeds\nInteraction accounting on a fixed {config['reference_size']}-example training reference; means and 95% seed intervals"
    )
    save(fig, output, f"{label}_cumulative")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row, view in enumerate(VIEWS):
        part = progress[(progress.optimizer == label) & (progress.view == view)]
        for ax, metric, title in zip(
            axes[row],
            ["penalty", "net", "S_plus"],
            [
                "Accumulated unfavorable penalty",
                "Accumulated net penalty",
                "Penalty / standalone gains",
            ],
        ):
            curve(ax, part[part.U > 0], "U", metric, factor=100.0 if metric == "S_plus" else 1.0)
            ax.set_title(f"{view.capitalize()}: {title}", fontsize=10)
            ax.set_xlabel("Accumulated standalone helpful gains U")
            ax.set_ylabel("%" if metric == "S_plus" else "Half-MSE units")
            ax.grid(alpha=0.15)
    axes[0, 0].legend()
    fig.suptitle(
        f"{label.upper()}: compare at equal accumulated standalone gains\nOnly the U range reached by every run is shown; all {n_seeds} paired seeds remain in each curve"
    )
    save(fig, output, f"{label}_equal_gains")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row, view in enumerate(VIEWS):
        part = phases[
            (phases.optimizer == label) & (phases.view == view) & (phases.bin_set == "overview")
        ]
        part = part[part.paired_seeds >= 3]
        for ax, metric, title in zip(
            axes[row, :2],
            ["S_plus", "S_net"],
            ["Unfavorable penalty / gains", "Net interaction penalty / gains"],
        ):
            for width, p in part.groupby("width"):
                p = p.sort_values("mse")
                ax.plot(p.mse, 100 * p[metric], color=COLORS[width], label=f"width {width}")
                ax.fill_between(
                    p.mse,
                    100 * p[metric + "_low"],
                    100 * p[metric + "_high"],
                    color=COLORS[width],
                    alpha=0.16,
                )
            ax.set_title(f"{view.capitalize()}: {title}", fontsize=10)
            ax.set_ylabel("% of standalone gains")
        coverage = part[part.width == 8].sort_values("mse")
        axes[row, 2].step(coverage.mse, coverage.paired_seeds, where="mid", color="#555555")
        axes[row, 2].set_ylim(0, n_seeds + 0.5)
        axes[row, 2].set_title("Paired seeds available", fontsize=10)
        for ax in axes[row]:
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xlabel("Fixed-reference MSE (same loss bin)")
            ax.grid(alpha=0.15)
    axes[0, 0].legend()
    fig.suptitle(
        f"{label.upper()}: compare at the same reference loss\nRatios pool all visits within each bin; lower-loss bins can contain fewer and different seeds"
    )
    save(fig, output, f"{label}_matched_loss")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row, view in enumerate(VIEWS):
        part = windows[(windows.optimizer == label) & (windows.view == view)]
        for ax, metric, title in zip(
            axes[row],
            ["S_plus", "helpful_frequency", "strong_frequency"],
            [
                "Interaction penalty / gains",
                "Unfavorable pairs among helpful pairs",
                "Strict reversal frequency (ordered)",
            ],
        ):
            curve(ax, part, "updates", metric, factor=100.0)
            ax.set_title(f"{view.capitalize()}: {title}", fontsize=9)
            ax.set_xlabel("Training updates")
            ax.set_ylabel("%")
            ax.grid(alpha=0.15)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0, 0].legend()
    fig.suptitle(
        f"{label.upper()}: local behavior throughout training\nEach point pools 1,000 consecutive actual updates; undefined ratios remain missing"
    )
    save(fig, output, f"{label}_local_windows")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["penalty", "S_plus", "S_net"],
        [
            "Accumulated unfavorable penalty",
            "Cumulative penalty / gains",
            "Cumulative net penalty / gains",
        ],
    ):
        factor = 1.0 if metric == "penalty" else 100.0
        curve(ax, c[c.view == "source"], "updates", metric, factor=factor)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training updates")
        ax.set_ylabel("Half-MSE units" if metric == "penalty" else "% of standalone gains")
        ax.grid(alpha=0.15)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].legend()
    fig.suptitle(
        f"{label.upper()}: accumulated effects on the actual source minibatches\nThe evaluation batch changes each step; these totals do not telescope to one fixed objective"
    )
    save(fig, output, f"{label}_source_cumulative")


def comparison_plot(cumulative, progress, phases, output, labels):
    """One compact reference-set figure with all three comparison clocks."""
    fig, axes = plt.subplots(
        len(labels), 3, figsize=(13, 3.5 * len(labels)), layout="constrained", squeeze=False
    )
    for row, label in enumerate(labels):
        c = cumulative[(cumulative.optimizer == label) & (cumulative.view == "reference")]
        p = progress[
            (progress.optimizer == label) & (progress.view == "reference") & (progress.U > 0)
        ]
        curve(axes[row, 0], c, "updates", "S_plus", factor=100.0)
        curve(axes[row, 1], p, "U", "S_plus", factor=100.0)
        selected = phases[
            (phases.optimizer == label)
            & (phases.view == "reference")
            & (phases.bin_set == "primary")
        ]
        for width, part in selected.groupby("width"):
            part = part.sort_values("mse")
            axes[row, 2].plot(
                part.mse, 100 * part.S_plus, color=COLORS[width], label=f"width {width}"
            )
            axes[row, 2].fill_between(
                part.mse,
                100 * part.S_plus_low,
                100 * part.S_plus_high,
                color=COLORS[width],
                alpha=0.16,
            )
        axes[row, 0].set_title(f"{label.upper()}: cumulative over training", fontsize=10)
        axes[row, 1].set_title(f"{label.upper()}: equal standalone gains", fontsize=10)
        axes[row, 2].set_title(f"{label.upper()}: same loss, 0.2 to 0.02", fontsize=10)
        axes[row, 0].set_xlabel("Training updates")
        axes[row, 0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axes[row, 1].set_xlabel("Accumulated standalone helpful gains U")
        axes[row, 2].set_xlabel("Fixed-reference MSE")
        axes[row, 2].set_xscale("log")
        axes[row, 2].set_xticks([0.2, 0.1, 0.05, 0.02], labels=["0.2", "0.1", "0.05", "0.02"])
        axes[row, 2].minorticks_off()
        axes[row, 2].invert_xaxis()
        for ax in axes[row]:
            ax.set_ylabel("Penalty / standalone gains (%)")
            ax.grid(alpha=0.15)
    axes[0, 0].legend()
    fig.suptitle(
        "Unfavorable neuron interactions: training totals and controls for progress\nFixed reference set; seed means and pointwise 95% bootstrap intervals"
    )
    save(fig, output, "comparison_summary")


def effects(frame, comparison):
    rows = []
    for identity, group in frame.groupby(["optimizer", "view"]):
        for metric in ["penalty", "net", "S_plus", "S_net", "S_minus", "retained_gain"]:
            if metric not in group:
                continue
            pivot = group.pivot(index="seed", columns="width", values=metric).dropna()
            if 8 not in pivot or 16 not in pivot or not len(pivot):
                continue
            mean, low, high = interval(pivot[16] - pivot[8])
            rows.append(
                dict(
                    optimizer=identity[0],
                    view=identity[1],
                    comparison=comparison,
                    metric=metric,
                    paired_seeds=len(pivot),
                    width8=pivot[8].mean(),
                    width16=pivot[16].mean(),
                    difference=mean,
                    low=low,
                    high=high,
                )
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/pier_study/trajectory"))
    parser.add_argument("--optimizers", nargs="+", default=["momentum", "sgd"])
    parser.add_argument("--plots-only", action="store_true", help="Reuse the saved summary CSVs")
    args = parser.parse_args()
    output = args.root / "analysis"
    output.mkdir(exist_ok=True)
    if args.plots_only:
        cumulative = pd.read_csv(output / "cumulative.csv")
        windows = pd.read_csv(output / "windows.csv")
        phase_means = pd.read_csv(output / "matched_loss_summary.csv")
        progress = pd.read_csv(output / "equal_gains.csv")
        for label in args.optimizers:
            losses = pd.read_csv(args.root / label / "losses.csv")
            config = json.loads((args.root / label / "config.json").read_text())
            plots(label, cumulative, windows, phase_means, progress, losses, output, config)
        comparison_plot(cumulative, progress, phase_means, output, args.optimizers)
        return
    collected = [collect(args.root / label) for label in args.optimizers]
    cumulative, windows, phases, progress, finals, audits = [
        pd.concat([x[i] for x in collected], ignore_index=True) for i in range(6)
    ]
    phase_means = phase_summary(phases)
    for name, frame in [
        ("cumulative", cumulative),
        ("windows", windows),
        ("loss_bins", phases),
        ("equal_gains", progress),
        ("final_seeds", finals),
        ("audit", audits),
        ("matched_loss_summary", phase_means),
    ]:
        frame.to_csv(output / f"{name}.csv", index=False)
    if audits.max_local_error.max() > 1e-10 or audits.telescope_error.abs().max() > 1e-9:
        raise AssertionError("Exact interaction accounting failed")
    primary = phases[(phases.bin_set == "primary") & np.isfinite(phases.S_plus)]
    counts = primary.groupby(["optimizer", "view", "width", "seed"]).bin.nunique()
    eligible = counts[counts == 12].reset_index()
    primary = primary.merge(eligible.drop(columns="bin"), on=["optimizer", "view", "width", "seed"])
    primary = (
        primary.groupby(["optimizer", "view", "width", "seed"])[["S_plus", "S_minus", "S_net"]]
        .mean()
        .reset_index()
    )
    primary.to_csv(output / "equal_loss_bin_average.csv", index=False)
    end_progress = progress.groupby(["optimizer", "view", "width", "seed"]).tail(1)
    comparisons = pd.concat(
        [
            effects(finals, "final_update"),
            effects(primary, "equal_loss_bins_0.2_to_0.02"),
            effects(end_progress, "equal_accumulated_gains"),
        ],
        ignore_index=True,
    )
    comparisons.to_csv(output / "paired_effects.csv", index=False)
    for label in args.optimizers:
        losses = pd.read_csv(args.root / label / "losses.csv")
        config = json.loads((args.root / label / "config.json").read_text())
        plots(label, cumulative, windows, phase_means, progress, losses, output, config)
    comparison_plot(cumulative, progress, phase_means, output, args.optimizers)
    print(
        comparisons[comparisons.metric.isin(["penalty", "S_plus", "S_net"])].to_string(index=False)
    )
    print("Maximum telescoping error:", audits.telescope_error.abs().max())


if __name__ == "__main__":
    main()
