#!/usr/bin/env python3
"""Compare teacher sizes, student widths and optimizers without pooling teachers."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_trajectory import collect, interval, packed

TABLES = ["cumulative", "windows", "phases", "progress", "finals", "audits"]
PROFILES = ["momentum_01", "adam_001", "adam_01"]
LABELS = {
    "momentum_01": "Momentum lr 0.1",
    "momentum_003": "Momentum lr 0.03",
    "adam_001": "Adam lr 0.001",
    "adam_01": "Adam lr 0.01",
}
PALETTE = ["#2563eb", "#12a085", "#dd7b16", "#9358bd"]
BINS = {
    "overview": np.geomspace(1e-9, 8.0, 81),
    "primary": np.geomspace(0.02, 0.2, 13),
    "early": np.geomspace(0.1, 0.5, 9),
}


def cached_collect(folder, refresh=False):
    cache = folder / "analysis_cache"
    signature = dict(
        schema=1,
        size=(folder / "updates.npy").stat().st_size,
        mtime=(folder / "updates.npy").stat().st_mtime_ns,
        helper=hashlib.sha256(
            (inspect.getsource(collect) + inspect.getsource(packed)).encode()
        ).hexdigest(),
        bins={k: v.tolist() for k, v in BINS.items()},
    )
    marker = cache / "complete.json"
    if not refresh and marker.exists() and json.loads(marker.read_text()) == signature:
        return [pd.read_csv(cache / f"{name}.csv.gz") for name in TABLES]
    frames = collect(folder, bin_ranges=BINS)
    cache.mkdir(exist_ok=True)
    for name, frame in zip(TABLES, frames):
        frame.to_csv(cache / f"{name}.csv.gz", index=False)
    marker.write_text(json.dumps(signature, indent=2) + "\n")
    return frames


def fitting(folder):
    data = pd.read_csv(folder / "losses.csv")
    last = data.updates.max()
    rows = []
    for (width, seed), part in data.groupby(["width", "seed"]):
        window = part[part.updates > last - 10_000]
        end = part[part.updates == last].iloc[0]
        rows.append(
            dict(
                width=width,
                seed=seed,
                window_mse=window.train_mse.mean(),
                window_test_mse=window.test_mse.mean(),
                final_mse=end.train_mse,
                best_mse=part.train_mse.min(),
                best_test_mse=part.test_mse.min(),
            )
        )
    return pd.DataFrame(rows), data


def equal_bins(phases, kind):
    part = phases[(phases.bin_set == kind) & np.isfinite(phases.S_plus)]
    keys = ["condition", "view", "width", "seed"]
    n = len(BINS[kind]) - 1
    complete = part.groupby(keys).bin.nunique()
    complete = complete[complete == n].reset_index().drop(columns="bin")
    part = part.merge(complete, on=keys)
    return part.groupby(keys)[["S_plus", "S_minus", "S_net"]].mean().reset_index()


def contrast(frame, jobs, scope, metrics):
    result = []
    for job in jobs:
        k = job["teacher_width"]
        subset = frame[frame.condition == job["id"]]
        for view, group in subset.groupby("view"):
            for width in job["widths"]:
                if width == k:
                    continue
                for metric in metrics:
                    pivot = group.pivot(index="seed", columns="width", values=metric)
                    if k not in pivot or width not in pivot:
                        continue
                    pivot = pivot[[k, width]].dropna()
                    if not len(pivot):
                        continue
                    base, wide = pivot[k].to_numpy(), pivot[width].to_numpy()
                    mean, low, high = interval(wide - base)
                    rng = np.random.default_rng(741)
                    ids = rng.integers(len(base), size=(4000, len(base)))
                    denominator = base[ids].mean(1)
                    ratios = np.divide(
                        wide[ids].mean(1),
                        denominator,
                        out=np.full(len(ids), np.nan),
                        where=denominator > 0,
                    )
                    result.append(
                        dict(
                            condition=job["id"],
                            teacher_width=k,
                            teacher_seed=job["teacher_seed"],
                            profile=job["profile"],
                            role=job["role"],
                            output_scale=job.get("output_scale", 0.0),
                            view=view,
                            scope=scope,
                            baseline=k,
                            width=width,
                            metric=metric,
                            n=len(pivot),
                            baseline_mean=base.mean(),
                            wide_mean=wide.mean(),
                            difference=mean,
                            low=low,
                            high=high,
                            ratio=wide.mean() / base.mean() if base.mean() > 0 else np.nan,
                            ratio_low=np.nanquantile(ratios, 0.025),
                            ratio_high=np.nanquantile(ratios, 0.975),
                        )
                    )
    return pd.DataFrame(result)


def summary(finals):
    rows = []
    for (condition, view, width), p in finals.groupby(["condition", "view", "width"]):
        first = p.iloc[0]
        row = dict(
            condition=condition,
            view=view,
            width=width,
            teacher_width=first.teacher_width,
            teacher_seed=first.teacher_seed,
            profile=first.profile,
            role=first.role,
            output_scale=first.output_scale,
            n=p.seed.nunique(),
            stable_successes=int((p.window_mse < 1e-6).sum()),
            best_successes=int((p.best_mse < 1e-6).sum()),
        )
        for metric in [
            "window_mse",
            "window_test_mse",
            "final_mse",
            "best_mse",
            "penalty",
            "net",
            "S_plus",
            "S_minus",
            "S_net",
            "frequency",
            "helpful_frequency",
            "strong_frequency",
        ]:
            mean, lo, hi = interval(p[metric])
            row.update({metric: mean, metric + "_low": lo, metric + "_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def band(ax, frame, x, y, color, label, *, scale=1.0, style="-", log=False):
    table = frame.pivot(index="seed", columns=x, values=y).sort_index(axis=1)
    if table.empty:
        return
    values = table.to_numpy() * scale
    mean = np.nanmean(values, axis=0)
    rng = np.random.default_rng(741)
    weights = rng.multinomial(len(table), np.full(len(table), 1 / len(table)), 800)
    den = weights @ np.isfinite(values).astype(float)
    boot = (weights @ np.nan_to_num(values)) / np.where(den > 0, den, np.nan)
    lo, hi = np.nanquantile(boot, [0.025, 0.975], axis=0)
    if log:
        mean, lo, hi = [np.maximum(v, 1e-14) for v in [mean, lo, hi]]
        ax.set_yscale("log")
    ax.plot(table.columns, mean, color=color, label=label, ls=style, lw=1.7)
    ax.fill_between(table.columns, lo, hi, color=color, alpha=0.14, linewidth=0)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def core_plots(finals, phases, progress, jobs, out):
    core = [j for j in jobs if j["role"] == "core"]
    for name, ylabel, mode in [
        ("fitting", "Mean train MSE over final 10k updates", "fit"),
        ("raw_penalty", "Accumulated unfavorable penalty (half-MSE)", "raw"),
        ("unfavorable_pair_frequency", "Unfavorable pairs among helpful pairs (%)", "freq"),
        ("strict_pair_frequency", "Strict reversal frequency among helpful pairs (%)", "strict"),
        ("normalized_controls", "Penalty / standalone helpful gains (%)", "controls"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), layout="constrained")
        for job in core:
            ax = axes[[8, 16].index(job["teacher_width"]), PROFILES.index(job["profile"])]
            part = finals[finals.condition == job["id"]]
            if mode == "fit":
                part = part[part.view == "reference"]
                band(ax, part, "width", "window_mse", PALETTE[0], "Train", log=True)
                band(ax, part, "width", "window_test_mse", PALETTE[2], "Test", log=True, style="--")
            elif mode == "raw":
                for color, view in zip(PALETTE, ["reference", "source"]):
                    band(
                        ax,
                        part[part.view == view],
                        "width",
                        "penalty",
                        color,
                        view.capitalize(),
                        log=True,
                    )
            elif mode in ["freq", "strict"]:
                metric = "helpful_frequency" if mode == "freq" else "strong_frequency"
                for color, view in zip(PALETTE, ["reference", "source"]):
                    band(
                        ax,
                        part[part.view == view],
                        "width",
                        metric,
                        color,
                        view.capitalize(),
                        scale=100.0,
                    )
            else:
                band(
                    ax,
                    part[part.view == "reference"],
                    "width",
                    "S_plus",
                    PALETTE[0],
                    "All 100k updates",
                    scale=100.0,
                )
                p = progress[(progress.condition == job["id"]) & (progress.view == "reference")]
                p = p.groupby(["width", "seed"]).tail(1)
                band(ax, p, "width", "S_plus", PALETTE[1], "Same accumulated gains", scale=100.0)
                p = phases[(phases.condition == job["id"]) & (phases.view == "reference")]
                p = equal_bins(p, "primary")
                band(ax, p, "width", "S_plus", PALETTE[2], "Same loss: 0.2 to 0.02", scale=100.0)
            ax.set_title(f"Teacher {job['teacher_width']}: {LABELS[job['profile']]}", fontsize=10)
            ax.set_xticks(job["widths"])
            ax.tick_params(axis="x", labelrotation=40)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
            ax.set_xlabel("Student width")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(alpha=0.16)
        axes[0, 0].legend(fontsize=8)
        for ax in axes.flat:
            if not ax.lines:
                ax.text(0.5, 0.5, "Not completed", ha="center", transform=ax.transAxes)
        fig.suptitle(
            "Robustness across teacher size, width and optimizer\n100,000 updates; five paired seeds; means and pointwise 95% seed intervals"
        )
        save(fig, out, f"robustness_{name}")


def forest(effects, jobs, out, role, view="reference", show_loss=True):
    selected = [j for j in jobs if (j["role"] == "core") == (role == "core")]
    if not selected:
        return
    specs = [
        ("fitting", "window_mse", "Final-window train MSE"),
        ("fixed_budget", "penalty", "Total unfavorable penalty"),
        ("equal_activity", "penalty", "Penalty at equal useful activity"),
        ("primary_loss", "S_plus", "Normalized penalty at same loss"),
    ]
    if not show_loss:
        specs = specs[:3]
    fig, axes = plt.subplots(
        1, len(specs), figsize=(14, 0.55 * len(selected) + 2), layout="constrained", sharey=True
    )
    labels = []
    for row, job in enumerate(selected):
        k, w = job["teacher_width"], max(job["widths"])
        init_note = f", init {job['output_scale']:g}" if "output_scale" in job else ""
        labels.append(
            f"K={k}, draw {job['teacher_seed']}{init_note}\n{LABELS[job['profile']]} ({k}→{w})"
        )
        for ax, (scope, metric, title) in zip(axes, specs):
            v = "reference" if scope == "fitting" else view
            p = effects[
                (effects.condition == job["id"])
                & (effects.view == v)
                & (effects.scope == scope)
                & (effects.metric == metric)
                & (effects.width == w)
            ]
            if not len(p):
                ax.text(1, row, "no complete pairs", ha="center", fontsize=7)
                continue
            r = p.iloc[0]
            color = (
                PALETTE[0]
                if "momentum" in job["profile"]
                else PALETTE[2]
                if job["profile"] == "adam_01"
                else PALETTE[3]
            )
            ax.plot(
                [max(r.ratio_low, 1e-12), max(r.ratio_high, 1e-12)], [row, row], color=color, lw=2
            )
            ax.plot(
                max(r.ratio, 1e-12), row, "o", color=color, ms=5, mfc=color if r.n == 5 else "white"
            )
            if r.n < 5:
                ax.annotate(
                    f" n={int(r.n)}",
                    (max(r.ratio, 1e-12), row),
                    fontsize=7,
                    xytext=(3, 4),
                    textcoords="offset points",
                )
    for ax, (scope, _, title) in zip(axes, specs):
        ax.set_xscale("log")
        ax.axvline(1, color="#666666", lw=1, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Wider / minimal-width mean\nBelow 1 favors wider", fontsize=8)
        ax.grid(axis="y", alpha=0.15)
        if scope != "fitting":
            limits = ax.get_xlim()
            ticks = [
                v for v in [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64] if limits[0] <= v <= limits[1]
            ]
            ax.set_xticks(ticks, labels=[f"{v:g}" for v in ticks])
            ax.set_xlim(limits)
        ax.minorticks_off()
    axes[0].set_yticks(range(len(selected)), labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(
        f"{'Core sweep' if role == 'core' else 'Learning-rate and teacher-realization checks'}: widest versus minimal student\n{view.capitalize()} interaction accounting; paired bootstrap intervals; hollow markers have fewer than five pairs"
    )
    suffix = "" if show_loss else "_compact"
    save(fig, out, f"robustness_{role}_effects_{view}{suffix}")


def trajectories(cumulative, progress, phases, losses, jobs, out):
    for job in jobs:
        condition = job["id"]
        init_note = f", output init {job['output_scale']:g}" if "output_scale" in job else ""
        c = cumulative[(cumulative.condition == condition) & (cumulative.view == "reference")]
        p = progress[(progress.condition == condition) & (progress.view == "reference")]
        loss = losses[losses.condition == condition]
        bins = phases[
            (phases.condition == condition)
            & (phases.view == "reference")
            & (phases.bin_set == "overview")
        ]
        coverage = bins.groupby(["bin", "seed"]).width.nunique()
        shared = coverage[coverage == len(job["widths"])].reset_index()
        full = shared.groupby("bin").seed.nunique()
        full = full[full == 5].index
        bins = bins.merge(shared[["bin", "seed"]], on=["bin", "seed"])
        bins = bins[bins.bin.isin(full)]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for color, width in zip(PALETTE, job["widths"]):
            label = f"width {width}"
            band(
                axes[0, 0],
                loss[loss.width == width],
                "updates",
                "train_mse",
                color,
                label,
                log=True,
            )
            band(axes[0, 1], c[c.width == width], "updates", "penalty", color, label)
            band(axes[0, 2], c[c.width == width], "updates", "net", color, label)
            band(axes[1, 0], c[c.width == width], "updates", "S_plus", color, label, scale=100.0)
            band(axes[1, 1], p[(p.width == width) & (p.U > 0)], "U", "penalty", color, label)
            b = bins[bins.width == width]
            if len(b):
                band(axes[1, 2], b, "mse", "S_plus", color, label, scale=100.0)
        titles = [
            "Full training MSE",
            "Accumulated unfavorable penalty",
            "Accumulated net penalty",
            "Cumulative penalty / gains (%)",
            "Penalty at equal standalone gains U",
            "Penalty / gains at matched loss (%)",
        ]
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.15)
        for ax in [*axes[0], axes[1, 0]]:
            ax.set_xlabel("Training updates")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axes[1, 1].set_xlabel("Accumulated standalone helpful gains U")
        axes[1, 2].set_xscale("log")
        axes[1, 2].invert_xaxis()
        axes[1, 2].set_xlabel("Fixed-reference MSE; bins shared by all five seeds")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"Teacher {job['teacher_width']}, draw {job['teacher_seed']}{init_note}: {LABELS[job['profile']]}\nFixed 256-example reference; five paired seeds; all actual updates measured"
        )
        save(fig, out, condition.replace("/", "_") + "_trajectory")


def plot_coverage(phases, jobs, out):
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), layout="constrained")
    centers = np.sqrt(BINS["overview"][:-1] * BINS["overview"][1:])
    for job in jobs:
        p = phases[
            (phases.condition == job["id"])
            & (phases.view == "reference")
            & (phases.bin_set == "overview")
            & np.isfinite(phases.S_plus)
        ]
        counts = p.groupby(["bin", "seed"]).width.nunique()
        common = counts[counts == len(job["widths"])].reset_index().groupby("bin").seed.nunique()
        n = common.reindex(range(len(centers)), fill_value=0).to_numpy()
        rows.extend(
            dict(condition=job["id"], bin=i, mse=x, paired_seeds_all_widths=int(v))
            for i, (x, v) in enumerate(zip(centers, n))
        )
        if job["role"] == "core":
            ax = axes[[8, 16].index(job["teacher_width"]), PROFILES.index(job["profile"])]
            ax.step(centers, n, where="mid", color="#555555")
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_ylim(-0.1, 5.3)
            ax.set_xlabel("Fixed-reference MSE")
            ax.set_ylabel("Paired seeds visiting bin at all widths")
            ax.set_title(f"Teacher {job['teacher_width']}: {LABELS[job['profile']]}", fontsize=10)
            ax.grid(alpha=0.15)
    pd.DataFrame(rows).to_csv(out / "loss_bin_coverage.csv", index=False)
    fig.suptitle(
        "Coverage of matched-loss comparisons\nMissing bins are not zero interference; some losses are not attained or are crossed between updates"
    )
    save(fig, out, "robustness_loss_coverage")


def initialization_plot(finals, out):
    path = out / "neuron_structure.csv"
    if not path.exists():
        return
    structural = pd.read_csv(path)
    p = finals[
        (finals.teacher_width == 8)
        & (finals.teacher_seed == 2027)
        & (finals.profile == "adam_001")
        & (finals.view == "reference")
        & finals.width.isin([8, 16, 32])
    ]
    structural = structural.merge(
        p[["condition", "width", "seed", "output_scale"]], on=["condition", "width", "seed"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for color, scale in zip(PALETTE, sorted(p.output_scale.unique())):
        label = f"Outgoing init std {scale:g}"
        band(axes[0], p[p.output_scale == scale], "width", "window_mse", color, label, log=True)
        band(
            axes[1],
            structural[structural.output_scale == scale],
            "width",
            "positive_active",
            color,
            label,
        )
    axes[0].set_title("Mean training MSE over final 10k updates")
    axes[0].set_ylabel("MSE")
    axes[1].set_title("Active neurons with positive outgoing weight")
    axes[1].set_ylabel("Number of neurons")
    for ax in axes:
        ax.set_xticks([8, 16, 32])
        ax.set_xlabel("Student width")
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Adam lr 0.001: targeted outgoing-initialization control\nSame incoming weights and minibatch sequences; all layers train; initial predictions differ"
    )
    save(fig, out, "adam_initialization_control")


def make_plots(finals, phases, progress, cumulative, losses, effects, jobs, out):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    core_plots(finals, phases, progress, jobs, out)
    for view in ["reference", "source"]:
        forest(effects, jobs, out, "core", view)
        forest(effects, jobs, out, "controls", view)
    forest(effects, jobs, out, "core", "reference", show_loss=False)
    trajectories(cumulative, progress, phases, losses, jobs, out)
    plot_coverage(phases, jobs, out)
    initialization_plot(finals, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/pier_study/robustness"))
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads((args.root / "plan.json").read_text())
    if (args.root / "extension_plan.json").exists():
        plan["jobs"] += json.loads((args.root / "extension_plan.json").read_text())["jobs"]
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)
    if args.plots_only:
        finals = pd.read_csv(out / "per_seed_results.csv")
        phases = pd.read_csv(out / "loss_bins.csv")
        progress = pd.read_csv(out / "equal_activity.csv")
        cumulative = pd.read_csv(out / "cumulative.csv")
        losses = pd.read_csv(out / "loss_history.csv")
        effects = pd.read_csv(out / "effects.csv")
        jobs = [j for j in plan["jobs"] if j["id"] in set(finals.condition)]
        make_plots(finals, phases, progress, cumulative, losses, effects, jobs, out)
        return
    gathered = [[] for _ in TABLES]
    fits = []
    losses = []
    jobs = []
    for job in plan["jobs"]:
        folder = args.root / job["id"]
        state = folder / "state.json"
        if not state.exists() or json.loads(state.read_text())["status"] != "complete":
            if args.completed_only:
                continue
            raise RuntimeError(f"Incomplete: {job['id']}")
        frames = cached_collect(folder, args.refresh)
        fit, history = fitting(folder)
        meta = dict(
            condition=job["id"],
            teacher_width=job["teacher_width"],
            teacher_seed=job["teacher_seed"],
            profile=job["profile"],
            role=job["role"],
            output_scale=job.get("output_scale", 0.0),
        )
        for collection, frame in zip(gathered, frames):
            collection.append(frame.assign(**meta))
        fits.append(fit.assign(**meta))
        losses.append(history.assign(**meta))
        jobs.append(job)
    if not jobs:
        print("No completed groups yet")
        return
    cumulative, windows, phases, progress, finals, audits = [
        pd.concat(parts, ignore_index=True) for parts in gathered
    ]
    fits = pd.concat(fits, ignore_index=True)
    losses = pd.concat(losses, ignore_index=True)
    finals = finals.merge(
        fits[
            [
                "condition",
                "width",
                "seed",
                "window_mse",
                "window_test_mse",
                "final_mse",
                "best_mse",
                "best_test_mse",
            ]
        ],
        on=["condition", "width", "seed"],
    )
    primary = equal_bins(phases, "primary")
    early = equal_bins(phases, "early")
    last_progress = progress.groupby(["condition", "view", "width", "seed"]).tail(1)
    effects = pd.concat(
        [
            contrast(finals, jobs, "fixed_budget", ["penalty", "net", "S_plus", "S_net"]),
            contrast(
                finals[finals.view == "reference"],
                jobs,
                "fitting",
                ["window_mse", "window_test_mse", "best_mse"],
            ),
            contrast(last_progress, jobs, "equal_activity", ["penalty", "net", "S_plus", "S_net"]),
            contrast(primary, jobs, "primary_loss", ["S_plus", "S_net"]),
            contrast(early, jobs, "early_loss", ["S_plus", "S_net"]),
        ],
        ignore_index=True,
    )
    aggregate = summary(finals)
    for name, frame in [
        ("per_seed_results", finals),
        ("summary", aggregate),
        ("effects", effects),
        ("loss_bins", phases),
        ("equal_loss_averages", primary),
        ("early_loss_averages", early),
        ("equal_activity", progress),
        ("cumulative", cumulative),
        ("windows", windows),
        ("audit", audits),
        ("fitting", fits),
        ("loss_history", losses),
    ]:
        frame.to_csv(out / f"{name}.csv", index=False)
    if audits.max_local_error.max() > 1e-9 or audits.telescope_error.abs().max() > 1e-8:
        raise AssertionError("Interaction accounting failed")
    make_plots(finals, phases, progress, cumulative, losses, effects, jobs, out)
    print(
        aggregate[(aggregate.view == "reference")][
            ["condition", "width", "window_mse", "stable_successes", "penalty", "S_plus"]
        ].to_string(index=False)
    )
    print("Completed groups:", len(jobs), "of", len(plan["jobs"]))


if __name__ == "__main__":
    main()
