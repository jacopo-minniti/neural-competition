#!/usr/bin/env python3
"""Recover strict-pair frequency and severity from deterministic cohort replays.

Run from the repository root with .venv/bin/python. Original runs are read-only;
only compact sufficient statistics, checks, and figures are written. Replays use
the original initialization, model batch layout, minibatches, and optimizers.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from analyze_robustness import BINS
from analyze_trajectory import interval
from matplotlib.ticker import NullLocator

from neural_competition.model import ShallowReLU
from neural_competition.trajectory import FIELDS, VIEWS, adam_step, components, momentum_step

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "runs/retained"
OUTPUT = ROOT / "docs/theory/checks/strict_matched_loss"
FIGURES = ROOT / "docs/figures/strict_matched_loss"
CONDITIONS = [
    "k8_teacher2027/adam_001",
    "k8_teacher2027/adam_01",
    "k8_teacher2027_init01/adam_001",
    "k8_teacher2027_init1/adam_001",
    "k8_teacher2027/momentum_01",
]
EXTRA_FIELDS = ["eligible_benefit", "event_benefit"]
ALL_FIELDS = FIELDS + EXTRA_FIELDS
METRICS = [
    "strict_frequency",
    "aggregate_severity",
    "event_severity",
    "harm_per_event",
    "harm_per_update",
    "events_per_update",
]
PHASE_EDGES = np.array([0.0, 1e-4, BINS["primary"][7], 0.2, np.inf])
BIN_SETS = {**BINS, "phase": PHASE_EDGES}
LABELS = {
    CONDITIONS[0]: "Adam 0.001, zero output",
    CONDITIONS[1]: "Adam 0.01, zero output",
    CONDITIONS[2]: "Adam 0.001, output std 0.1",
    CONDITIONS[3]: "Adam 0.001, output std 1",
    CONDITIONS[4]: "Momentum 0.1, zero output",
}
COLORS = {8: "#2563eb", 10: "#12a085", 16: "#dd7b16", 32: "#9358bd"}


@torch.jit.script
def detailed_metrics(before, after, y, upper):
    """Original 14 fields, plus exact strict-severity benefit denominators."""
    changes = after - before
    residual = before.sum(-1) - y
    baseline = 0.5 * residual.square().mean(1)
    alone = (residual[:, :, None] * changes + 0.5 * changes.square()).mean(1)
    gram = torch.bmm(changes.transpose(1, 2), changes) / changes.shape[1]
    tolerance = 1e-10 * baseline[:, None, None]
    helpful = alone < -tolerance[:, :, 0]
    eligible = helpful[:, :, None] & helpful[:, None, :] & upper
    directed = upper | upper.transpose(1, 2)
    eligible_ordered = helpful[:, :, None] & helpful[:, None, :] & directed
    marginal = alone[:, None, :] + gram
    strong = eligible_ordered & (marginal > tolerance)
    benefit = -alone[:, None, :].expand_as(gram)
    return torch.stack(
        [
            baseline,
            0.5 * (after.sum(-1) - y).square().mean(1),
            (-alone).clamp_min(0).sum(1),
            alone.clamp_min(0).sum(1),
            (gram.clamp_min(0) * upper).sum((1, 2)),
            ((-gram).clamp_min(0) * upper).sum((1, 2)),
            ((gram > tolerance) & upper).sum((1, 2)).double(),
            upper.sum((1, 2)).double(),
            eligible.sum((1, 2)).double(),
            ((gram > tolerance) & eligible).sum((1, 2)).double(),
            strong.sum((1, 2)).double(),
            torch.where(strong, marginal, 0.0).sum((1, 2)),
            0.5 * changes.square().mean(1).sum(1),
            helpful.sum(1).double(),
            torch.where(eligible_ordered, benefit, 0.0).sum((1, 2)),
            torch.where(strong, benefit, 0.0).sum((1, 2)),
        ],
        dim=-1,
    )


@torch.jit.script
def replay_block(
    p, first, second, x_train, y_train, indices, seeds, widths, xref, yref, upper,
    lr: float, momentum: float, is_adam: bool, start: int,
):
    records = torch.empty((indices.shape[0], p.shape[0], 2, 16), dtype=torch.float64)
    norm64 = widths.double().sqrt()
    norm32 = norm64.float()
    for step in range(indices.shape[0]):
        ids = indices[step].index_select(0, seeds)
        x, y = x_train[ids], y_train[ids]
        if is_adam:
            following, first, second = adam_step(
                p, first, second, x, y, norm32, lr, start + step + 1
            )
        else:
            following, first = momentum_step(p, first, x, y, norm32, lr, momentum)
        combined = torch.cat([x.double(), xref], dim=1)
        before = components(p.double(), combined, norm64)
        after = components(following.double(), combined, norm64)
        n = x.shape[1]
        records[step, :, 0] = detailed_metrics(before[:, :n], after[:, :n], y.double(), upper)
        records[step, :, 1] = detailed_metrics(before[:, n:], after[:, n:], yref, upper)
        p = following
    return p, first, second, records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def divide(a, b):
    return a / b if b > 0 else np.nan


def measured(values, visits):
    s = dict(zip(ALL_FIELDS, values))
    n, h = s["strong_events"], s["strong_harm"]
    return {
        **s,
        "eligible_ordered": 2 * s["helpful_pairs"],
        "strict_frequency": divide(n, 2 * s["helpful_pairs"]),
        "aggregate_severity": divide(h, s["eligible_benefit"]),
        "event_severity": divide(h, s["event_benefit"]),
        "harm_per_event": divide(h, n),
        "harm_per_update": divide(h, visits),
        "events_per_update": divide(n, visits),
    }


def accumulate(records, totals, counts):
    mse = 2 * records[:, :, 1, 0]
    for name, edges in BIN_SETS.items():
        assignments = np.searchsorted(edges, mse, side="right") - 1
        for i in range(records.shape[1]):
            for k in np.unique(assignments[:, i]):
                if k < 0 or k >= len(edges) - 1:
                    continue
                mask = assignments[:, i] == k
                totals[name][i, k] += records[mask, i].sum(0)
                counts[name][i, k] += mask.sum()


def replay(condition, threads):
    torch.set_num_threads(threads)
    hashes = {}

    def read(path, kind="torch"):
        hashes[str(path.relative_to(ROOT))] = digest(path)
        return (
            torch.load(path, weights_only=True, map_location="cpu")
            if kind == "torch" else json.loads(path.read_text())
        )

    config = read(ARCHIVE / "provenance/training/robustness" / condition / "config.json", "json")
    dataset = condition.split("/")[0].split("_init")[0]
    data = read(ARCHIVE / "datasets" / f"{dataset}.pt")
    folder = ARCHIVE / "checkpoints/robustness" / condition
    saved = read(folder / "initial.pt")
    models = config["models"]
    maximum = max(m["width"] for m in models)
    scale = config.get("student_output_scale", 0.0)
    p = torch.zeros(len(models), maximum, 22)
    for i, model in enumerate(models):
        net = ShallowReLU(20, model["width"], model["seed"], zero_output=scale == 0)
        initial = net.parameter_blocks()
        if scale:
            initial[:, -1].mul_(scale)
        p[i, : model["width"]] = initial
    original_indices = saved["original_model_indices"]
    assert torch.equal(p[original_indices], saved["parameters"]), condition
    widths = torch.tensor([m["width"] for m in models])[:, None]
    seeds = torch.tensor([m["seed"] for m in models])
    generators = [torch.Generator().manual_seed(12026 + s) for s in range(5)]
    reference = saved["reference_indices"]
    xref = data["x_train"][reference].double()[None].expand(len(models), -1, -1)
    yref = data["y_train"][reference].double()[None].expand(len(models), -1)
    active = torch.arange(maximum)[None] < widths
    upper = torch.triu(torch.ones(maximum, maximum, dtype=torch.bool), diagonal=1)[None]
    upper = upper & active[:, :, None] & active[:, None, :]
    first, second = torch.zeros_like(p), torch.zeros_like(p)
    totals = {
        name: np.zeros((len(models), len(edges) - 1, 2, len(ALL_FIELDS)))
        for name, edges in BIN_SETS.items()
    }
    counts = {name: np.zeros(v.shape[:2], dtype=np.int64) for name, v in totals.items()}
    all_totals = np.zeros((len(models), 2, len(ALL_FIELDS)))
    raw_path = ARCHIVE / "raw_updates" / f"robustness_{condition.replace('/', '_')}.npz"
    raw = None
    if raw_path.exists():
        hashes[str(raw_path.relative_to(ROOT))] = digest(raw_path)
        raw = np.load(raw_path)["records"]
    seed0 = [i for i, m in enumerate(models) if m["seed"] == 0]
    max_raw_error = 0.0
    checks = []
    started = time.perf_counter()
    for start in range(0, config["steps"], 1000):
        count = min(1000, config["steps"] - start)
        indices = torch.stack(
            [torch.randint(len(data["x_train"]), (count, 64), generator=g) for g in generators],
            dim=1,
        )
        p, first, second, stats = replay_block(
            p, first, second, data["x_train"], data["y_train"], indices, seeds,
            widths, xref, yref, upper, config["lr"], config["momentum"],
            config["optimizer_spec"]["name"] == "adam", start,
        )
        records = stats.numpy()
        assert np.isfinite(records).all(), (condition, start)
        if raw is not None:
            difference = np.max(np.abs(raw[start:start + count] - records[:, seed0, :, :14]))
            max_raw_error = max(max_raw_error, float(difference))
            assert difference == 0, (condition, start, difference)
        accumulate(records, totals, counts)
        all_totals += records.sum(0)
        completed = start + count
        if completed in [10_000, 50_000, 100_000]:
            name = "final" if completed == 100_000 else f"step_{completed:06d}"
            checkpoint = read(folder / f"{name}.pt")
            assert torch.equal(p[original_indices], checkpoint["parameters"]), (condition, name)
            assert torch.equal(first[original_indices], checkpoint["velocity"]), (condition, name)
            if "second" in checkpoint:
                assert torch.equal(second[original_indices], checkpoint["second"]), (condition, name)
            for seed, state in zip(checkpoint["rng_seed_order"], checkpoint["rng_states"]):
                assert torch.equal(generators[seed].get_state(), state), (condition, name, seed)
            checks.append({"updates": completed, "retained_states_bitwise_equal": True})
        if completed % 10_000 == 0:
            print(f"{condition}: {completed:,}/100,000; {time.perf_counter() - started:.1f}s", flush=True)
    rows = []
    for i, model in enumerate(models):
        for v, view in enumerate(VIEWS):
            base = dict(condition=condition, **model, view=view)
            rows.append(dict(
                **base, bin_set="full", bin=-1, lower=0, upper=np.inf, mse=np.nan,
                visits=config["steps"], **measured(all_totals[i, v], config["steps"]),
            ))
            for name, edges in BIN_SETS.items():
                for k, n in enumerate(counts[name][i]):
                    if n == 0:
                        continue
                    rows.append(dict(
                        **base, bin_set=name, bin=k, lower=edges[k], upper=edges[k + 1],
                        mse=np.sqrt(edges[k] * edges[k + 1]) if np.isfinite(edges[k + 1]) else np.nan,
                        visits=int(n), **measured(totals[name][i, k, v], n),
                    ))
    frame = pd.DataFrame(rows)
    old = pd.read_csv(ARCHIVE / "aggregates/robustness/loss_bins.csv.gz")
    old = old[old.condition == condition]
    joined = frame.merge(old, on=["condition", "view", "width", "seed", "bin_set", "bin"])
    assert len(joined) == len(old), (condition, len(joined), len(old))
    frequency_error = np.max(np.abs(joined.strict_frequency - joined.strong_frequency))
    assert frequency_error < 1e-12, frequency_error
    assert (joined.visits_x == joined.visits_y).all()
    full = pd.read_csv(ROOT / "docs/tables/robustness_per_seed_results.csv")
    joined_full = frame[frame.bin_set == "full"].merge(
        full[full.condition == condition], on=["condition", "view", "width", "seed"]
    )
    full_error = np.max(np.abs(joined_full.strict_frequency - joined_full.strong_frequency))
    assert full_error < 1e-12, full_error
    audit = dict(
        condition=condition, models=len(models), seeds=5, updates=config["steps"],
        checkpoints=checks, original_seed0_raw_max_error=max_raw_error if raw is not None else None,
        original_bin_frequency_max_error=float(frequency_error),
        original_full_frequency_max_error=float(full_error),
        input_sha256=hashes, elapsed_seconds=time.perf_counter() - started,
        torch_version=str(torch.__version__), numpy_version=np.__version__, threads=threads,
        replay_is_new_independent_evidence=False,
        replay_producer_sha256=digest(Path(__file__)),
        replay_engine_sha256=digest(ROOT / "src/neural_competition/trajectory.py"),
    )
    stem = condition.replace("/", "_")
    frame.to_csv(OUTPUT / f"{stem}.csv.gz", index=False, float_format="%.17g")
    (OUTPUT / f"{stem}.json").write_text(json.dumps(audit, indent=2) + "\n")
    return frame


def summaries(frame):
    # Equal weight for each of the same five log bins at every width/seed/setting.
    shared = frame[(frame.bin_set == "primary") & frame.bin.between(7, 11)]
    identity = ["condition", "view", "width", "seed"]
    assert shared.groupby(identity).bin.nunique().eq(5).all()
    # Event-conditional magnitudes are undefined in bins with no events. Pool
    # their numerators/denominators separately instead of silently dropping bins.
    unconditional = [
        "strict_frequency", "aggregate_severity", "harm_per_update", "events_per_update"
    ]
    averages = shared.groupby(identity)[unconditional].mean().reset_index()
    averages["scope"] = "shared_five_bins"
    pooled_rows = []
    for keys, group in shared.groupby(identity):
        pooled_rows.append(dict(
            **dict(zip(identity, keys)), scope="shared_pooled",
            **measured(group[ALL_FIELDS].sum().to_numpy(), group.visits.sum()),
        ))
    pooled = pd.DataFrame(pooled_rows)[[*identity, "scope", *METRICS]]
    full = frame[frame.bin_set == "full"][[*identity, *METRICS]].assign(scope="full_budget")
    per_seed = pd.concat([averages, pooled, full], ignore_index=True)
    per_seed.to_csv(OUTPUT / "comparison_per_seed.csv", index=False)
    rows = []
    for (condition, view, scope), group in per_seed.groupby(["condition", "view", "scope"]):
        for width in sorted(set(group.width) - {8}):
            for metric in METRICS:
                p = group.pivot(index="seed", columns="width", values=metric)[[8, width]].dropna()
                if p.empty:
                    continue
                base, wide = p[8].to_numpy(), p[width].to_numpy()
                mean, low, high = interval(wide - base)
                rows.append(dict(
                    condition=condition, view=view, scope=scope, width=width, metric=metric,
                    n=len(p), baseline_mean=base.mean(), wide_mean=wide.mean(),
                    difference=mean, low=low, high=high,
                    ratio=divide(wide.mean(), base.mean()),
                ))
    pd.DataFrame(rows).to_csv(OUTPUT / "comparison_effects.csv", index=False)
    # Fully covered bins only for curves; report sparse coverage separately.
    primary = frame[frame.bin_set == "primary"]
    coverage = primary.groupby(["condition", "view", "bin", "width"]).agg(
        n_seeds=("seed", "nunique"), min_visits=("visits", "min"), max_visits=("visits", "max"),
        lower=("lower", "first"), upper=("upper", "first"), mse=("mse", "first"),
    ).reset_index()
    coverage.to_csv(OUTPUT / "coverage.csv", index=False)
    curve_rows = []
    for keys, group in primary.groupby(["condition", "view", "bin", "width"]):
        condition, view, k, width = keys
        for metric in METRICS:
            mean, low, high = interval(group[metric])
            curve_rows.append(dict(
                condition=condition, view=view, bin=k, width=width, metric=metric,
                n=group.seed.nunique(), n_defined=int(np.isfinite(group[metric]).sum()),
                mse=group.mse.iloc[0], lower=group.lower.iloc[0], upper=group.upper.iloc[0],
                mean=mean, low=low, high=high,
            ))
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUTPUT / "loss_curves.csv", index=False)
    return curves


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_curves(curves):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    layouts = [("adam", CONDITIONS[:4]), ("adam_default", CONDITIONS[:2])]
    for view, (prefix, conditions) in [(v, item) for v in VIEWS for item in layouts]:
        fig, axes = plt.subplots(
            2, len(conditions), figsize=(3.75 * len(conditions), 6.8), layout="constrained"
        )
        for col, condition in enumerate(conditions):
            part = curves[(curves.condition == condition) & (curves.view == view)]
            for row, metric in enumerate(["strict_frequency", "aggregate_severity"]):
                ax = axes[row, col]
                for width, group in part[part.metric == metric].groupby("width"):
                    # Do not connect across bins missing from any seed at this width.
                    group = group.sort_values("mse")
                    ok = (group.n == 5) & (group.n_defined == 5)
                    x = group.mse.to_numpy()
                    mean = np.where(ok, 100 * group["mean"], np.nan)
                    low = np.where(ok, 100 * group.low, np.nan)
                    high = np.where(ok, 100 * group.high, np.nan)
                    ax.plot(x, mean, "o-", ms=3, lw=1.4, color=COLORS[width], label=f"width {width}")
                    ax.fill_between(x, low, high, color=COLORS[width], alpha=0.12, linewidth=0)
                ax.axvspan(BINS["primary"][7], 0.2, color="#777777", alpha=0.07)
                ax.set_xscale("log")
                ax.set_xlim(0.2, 0.02)
                ax.set_xticks([0.2, 0.1, 0.05, 0.02], ["0.2", "0.1", "0.05", "0.02"])
                ax.xaxis.set_minor_locator(NullLocator())
                ax.grid(alpha=0.15)
                ax.set_xlabel("Reference MSE (lower loss to the right)", fontsize=8)
                ax.set_ylabel("Strict-event frequency (%)" if row == 0 else "Aggregate severity (%)")
                if row == 0:
                    ax.set_title(LABELS[condition])
                    ax.legend(fontsize=7)
        fig.suptitle(
            f"Strict pair interference at matched reference loss — {view} evaluation\n"
            "Five original seeds; pointwise 95% seed bootstrap; shaded loss range is common to every run"
        )
        save(fig, f"{prefix}_matched_loss_{view}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        path = OUTPUT / f"{condition.replace('/', '_')}.csv.gz"
        if not args.summarize_only and (args.refresh or not path.exists()):
            replay(condition, args.threads)
    if all((OUTPUT / f"{c.replace('/', '_')}.csv.gz").exists() for c in CONDITIONS):
        frame = pd.concat([
            pd.read_csv(OUTPUT / f"{c.replace('/', '_')}.csv.gz") for c in CONDITIONS
        ], ignore_index=True)
        curves = summaries(frame)
        plot_curves(curves)
        manifest = {
            "purpose": "Strict frequency and harm at matched reference loss in original teacher-8 cohorts",
            "conditions": CONDITIONS,
            "independent_seeds_per_condition": 5,
            "shared_bins": list(range(7, 12)),
            "shared_mse_range": [float(BINS["primary"][7]), 0.2],
            "matching": "pre-update fixed-reference MSE; all visits within each original log bin",
            "aggregation": "ratios of sums per seed/bin; equal bin weights; then equal seed weights",
            "bootstrap": "4000 paired-seed resamples, seed 741; pointwise, exploratory",
            "new_training_conditions": False,
            "new_independent_runs": False,
            "original_archives_modified": False,
            "script_sha256": digest(Path(__file__)),
            "engine_sha256": digest(ROOT / "src/neural_competition/trajectory.py"),
            "replay_producers_sha256": {
                c: json.loads((OUTPUT / f"{c.replace('/', '_')}.json").read_text())[
                    "replay_producer_sha256"
                ] for c in CONDITIONS
            },
        }
        (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Wrote loss-matched strict-frequency/severity comparisons and figures.", flush=True)


if __name__ == "__main__":
    main()
