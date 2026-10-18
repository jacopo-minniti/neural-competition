#!/usr/bin/env python3
"""Summarize sampled peer-update interventions, with seed-level uncertainty."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_temporal_branches import ARMS, CACHE, LONG_PROBE_IDS, ROOT, conditions

OUTPUT = ROOT / "docs/theory/checks/temporal_branches"
RAW = CACHE / "matched_dose"
FIGURES = ROOT / "docs/figures/temporal_branches"
KEYS = ["condition", "width", "seed", "probe_id", "probe_sender", "completed", "sender", "view", "lag"]
BENEFITS = [
    "peer_harm_benefit", "peer_quality_benefit", "peer_residual_benefit", "peer_anti_benefit",
    "peer_gain_benefit", "peer_signed_alone_benefit", "peer_loss_benefit", "peer_delayed_loss_benefit",
    "peer_immediate_loss_benefit", "selective_harm_saving", "selective_loss_saving",
]
CORE = [
    "trajectory/sgd", "robustness/k8_teacher2027/momentum_01",
    "robustness/k8_teacher2027/adam_001", "robustness/k8_teacher2027/adam_01",
    "robustness/k8_teacher2027_init01/adam_001", "robustness/k8_teacher2027_init1/adam_001",
]
LABELS = ["SGD 1.0, teacher 8", "Momentum .1, teacher 8", "Adam .001, zero output",
          "Adam .01, zero output", "Adam .001, output std .1", "Adam .001, output std 1"]
COLORS = {8: "#2563eb", 10: "#12a085", 16: "#df6b20", 18: "#997729", 32: "#9358bd", 64: "#c34a79"}


def interval(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(741)
    sample = a[rng.integers(len(a), size=(4000, len(a)))].mean(1)
    low, high = np.quantile(sample, [.025, .975])
    return float(a.mean()), float(low), float(high), len(a)


def contrast(frame):
    arms = [frame[frame.arm == arm].set_index(KEYS).sort_index() for arm in ARMS]
    a, b, c = arms
    assert a.index.equals(b.index) and a.index.equals(c.index)
    out = a[["ordinary_origin_loss", "removed_prediction_rms", "relative_match_error"]].copy()
    out["relative_cross_width_match_error"] = (
        a.relative_cross_width_match_error if "relative_cross_width_match_error" in a else 0.0
    )
    for name in ["gain", "harm", "anti_work"]:
        out["ordinary_" + name] = a["cumulative_" + name]
    out["ordinary_progress"] = a.origin_loss - a.loss_after
    out["ordinary_point_gain"] = a.gain
    out["ordinary_point_harm"] = a.harm
    out["peer_point_harm_benefit"] = b.harm - a.harm
    out["peer_harm_benefit"] = b.cumulative_harm - a.cumulative_harm
    out["peer_quality_benefit"] = b.cumulative_quality_harm
    out["peer_residual_benefit"] = b.cumulative_residual_harm
    out["peer_anti_benefit"] = b.cumulative_anti_work - a.cumulative_anti_work
    out["peer_gain_benefit"] = a.cumulative_gain - b.cumulative_gain
    out["peer_signed_alone_benefit"] = out.peer_harm_benefit + out.peer_gain_benefit
    out["peer_loss_benefit"] = b.loss_after - a.loss_after
    out["peer_immediate_loss_benefit"] = b.origin_loss - a.origin_loss
    out["peer_delayed_loss_benefit"] = out.peer_loss_benefit - out.peer_immediate_loss_benefit
    out["selective_harm_saving"] = c.cumulative_harm - b.cumulative_harm
    out["selective_loss_saving"] = c.loss_after - b.loss_after
    assert np.allclose(out.peer_harm_benefit,
                       out.peer_quality_benefit + out.peer_residual_benefit, atol=1e-11)
    out = out.reset_index()
    # Both outcome views are assigned the ordinary branch's reference loss.
    keys = [k for k in KEYS if k not in ["view", "lag"]]
    phase = out[(out.view == "reference") & (out.lag == 32)][keys + ["ordinary_origin_loss"]]
    phase = phase.rename(columns={"ordinary_origin_loss": "reference_origin_loss"})
    out = out.merge(phase, on=keys, validate="many_to_one")
    out["reference_mse"] = 2*out.reference_origin_loss
    return out


def scopes(data):
    shared = data[data.reference_mse.between(.0766, .2)].copy()
    first = shared.groupby(["condition", "width", "seed"]).completed.transform("min")
    if "relative_match_error" in data:
        error = np.maximum(data.relative_match_error, data.relative_cross_width_match_error)
        # Drop a whole seed/checkpoint/sender-rank comparison if ANY width has
        # >5% calibration error. Selection uses the intervention, not its outcome.
        grouped = pd.Series(error, index=data.index).groupby([
            data[k] for k in ["condition", "seed", "probe_id", "probe_sender"]
        ]).transform("max")
        well_matched = data[grouped <= .05]
    else:
        well_matched = data
    return {
        "all_sampled": data,
        "long_sampled": data[data.probe_id.isin(LONG_PROBE_IDS)],
        "shared_loss": shared,
        "first_shared_loss": shared[shared.completed == first],
        "late_sampled": data[data.completed >= 1000],
        "well_matched": well_matched,
    }


def summarize(data):
    rows = []
    for scope, part in scopes(data).items():
        for identity, group in part.groupby(["condition", "width", "seed", "view", "lag"]):
            base = dict(zip(["condition", "width", "seed", "view", "lag"], identity))
            g = group.ordinary_gain.sum()
            h = group.ordinary_harm.sum()
            pulse = group.peer_immediate_loss_benefit.abs().sum()
            row = dict(**base, scope=scope, episodes=len(group), checkpoints=group.probe_id.nunique(),
                       ordinary_gain=g, ordinary_harm=h, absolute_initial_loss_perturbation=pulse,
                       removed_prediction_rms=group.removed_prediction_rms.mean(),
                       max_match_error=group.relative_match_error.max())
            for metric in BENEFITS:
                row[metric + "_mean"] = group[metric].mean()
                row[metric + "_per100G"] = 100*group[metric].sum()/g if g > 1e-20 else np.nan
                row[metric + "_per100Pulse"] = 100*group[metric].sum()/pulse if pulse > 1e-20 else np.nan
            row["peer_harm_benefit_per100H"] = (
                100*group.peer_harm_benefit.sum()/h if h > 1e-20 else np.nan
            )
            row["peer_point_harm_benefit_per100G"] = (
                100*group.peer_point_harm_benefit.sum()/group.ordinary_point_gain.sum()
                if group.ordinary_point_gain.sum() > 1e-20 else np.nan
            )
            rows.append(row)
    seed = pd.DataFrame(rows)
    seed.to_csv(OUTPUT / "summary_per_seed.csv.gz", index=False)
    metrics = [c for c in seed if c.endswith(("_mean", "_per100G", "_per100H", "_per100Pulse"))]
    rows = []
    for identity, group in seed.groupby(["condition", "width", "view", "lag", "scope"]):
        base = dict(zip(["condition", "width", "view", "lag", "scope"], identity))
        for metric in metrics:
            mean, low, high, n = interval(group[metric])
            rows.append(dict(**base, metric=metric, mean=mean, low=low, high=high, n=n,
                             min_checkpoints=int(group.checkpoints.min()),
                             max_checkpoints=int(group.checkpoints.max())))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "summary.csv.gz", index=False)
    rows = []
    for identity, group in seed.groupby(["condition", "view", "lag", "scope"]):
        base = dict(zip(["condition", "view", "lag", "scope"], identity))
        widths = sorted(group.width.unique())
        for width in widths[1:]:
            for metric in metrics:
                paired = group.pivot(index="seed", columns="width", values=metric)[[widths[0], width]].dropna()
                if paired.empty:
                    continue
                mean, low, high, n = interval(paired[width] - paired[widths[0]])
                rows.append(dict(**base, narrow=widths[0], width=width, metric=metric,
                                 difference=mean, low=low, high=high, n=n))
    pd.DataFrame(rows).to_csv(OUTPUT / "width_contrasts.csv.gz", index=False)
    coverage = seed[["condition", "width", "seed", "view", "lag", "scope", "episodes", "checkpoints"]]
    coverage.to_csv(OUTPUT / "coverage.csv.gz", index=False)
    return seed, summary


def summarize_neurons():
    rows = []
    for condition in conditions():
        path = RAW / f"{condition.replace('/', '_')}_neurons.csv.gz"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["benefit"] = frame.damped_alone.clip(lower=0) - frame.ordinary_alone.clip(lower=0)
        frame["gross_benefit"] = frame.benefit.clip(lower=0)
        frame["gross_harm"] = (-frame.benefit).clip(lower=0)
        both = (frame.initial_sender_alone < 0) & (frame.initial_recipient_alone < 0)
        strict = both & (frame.initial_recipient_alone + frame.initial_pair_cost > 0)
        frame["pair_class"] = np.select(
            [strict, both & (frame.initial_pair_cost > 0), frame.initial_pair_cost < 0],
            ["strict", "helpful_positive_overlap", "negative_pair_cost"], default="other",
        )
        for label, part in [("all", frame), *list(frame.groupby("pair_class"))]:
            for identity, group in part.groupby(["condition", "width", "seed", "view", "lag"]):
                rows.append(dict(
                    **dict(zip(["condition", "width", "seed", "view", "lag"], identity)),
                    pair_class=label, recipient_cases=len(group), net_benefit=group.benefit.sum(),
                    gross_benefit=group.gross_benefit.sum(), gross_harm=group.gross_harm.sum(),
                    ordinary_harm=group.ordinary_alone.clip(lower=0).sum(),
                    ordinary_gain=(-group.ordinary_alone).clip(lower=0).sum(),
                ))
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / "recipient_effects_per_seed.csv.gz", index=False)
    rows = []
    for identity, group in out.groupby(["condition", "width", "view", "lag", "pair_class"]):
        base = dict(zip(["condition", "width", "view", "lag", "pair_class"], identity))
        for name in ["net_benefit", "gross_benefit", "gross_harm"]:
            values = 100*group[name]/group.ordinary_gain.replace(0, np.nan)
            mean, low, high, n = interval(values)
            rows.append(dict(**base, metric=name + "_per100G", mean=mean, low=low, high=high, n=n))
    pd.DataFrame(rows).to_csv(OUTPUT / "recipient_effects.csv.gz", index=False)
    return out


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_widths(summary, view="heldout", scope="all_sampled", pulse=False):
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), layout="constrained")
    metric = "peer_harm_benefit_per100Pulse" if pulse else "peer_harm_benefit_per100G"
    for ax, condition, label in zip(axes.flat, CORE, LABELS):
        part = summary[(summary.condition == condition) & (summary.view == view)
                       & (summary.scope == scope) & (summary.lag == 32) & (summary.metric == metric)]
        for row in part.itertuples():
            ax.errorbar(row.width, row.mean, yerr=[[max(0, row.mean-row.low)], [max(0, row.high-row.mean)]],
                        fmt="o", color=COLORS[row.width], capsize=4)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(sorted(part.width.unique()))
        ax.set_title(label)
        ax.set_xlabel("Student width")
        ax.set_ylabel("Later harm benefit / initial pulse impact (%)" if pulse
                      else "Peer benefit per 100 recipient helpful units")
        ax.grid(alpha=.15)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    fig.suptitle("Does an earlier sender update reduce later harm at other neurons?\n"
                 f"{view}; cumulative next 32 updates; {scope.replace('_', ' ')}\n"
                 "Positive: ordinary sender prevents harm; negative: damping the sender reduces harm")
    save(fig, f"peer_harm_{view}_{scope}" + ("_pulse" if pulse else ""))


def plot_lags(summary):
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), layout="constrained")
    for ax, condition, label in zip(axes.flat, CORE, LABELS):
        part = summary[(summary.condition == condition) & (summary.view == "heldout")
                       & (summary.scope == "long_sampled")
                       & (summary.metric == "peer_harm_benefit_per100G")]
        for width, group in part.groupby("width"):
            group = group.sort_values("lag")
            ax.plot(group.lag, group["mean"], "o-", color=COLORS[width], label=f"width {width}")
            ax.fill_between(group.lag, group.low, group.high, color=COLORS[width], alpha=.12)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xscale("log")
        ax.set_xticks([1, 4, 16, 32, 128], ["1", "4", "16", "32", "128"])
        ax.set_title(label)
        ax.set_xlabel("Updates after the intervention")
        ax.set_ylabel("Peer benefit per 100 recipient helpful units")
        ax.grid(alpha=.15)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.legend(fontsize=7)
    fig.suptitle("Cumulative temporal peer effects at increasing horizons\n"
                 "Held-out examples; same five checkpoints at every horizon; pointwise 95% seed intervals")
    save(fig, "peer_harm_lags_heldout")


def main():
    global OUTPUT, RAW, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=["matched_dose", "fixed_fraction"], default="matched_dose")
    args = parser.parse_args()
    RAW = CACHE / args.design
    if args.design == "fixed_fraction":
        OUTPUT = OUTPUT / "fixed_fraction"
        FIGURES = FIGURES / "fixed_fraction"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "provenance").mkdir(exist_ok=True)
    files = []
    provenance = []
    for condition in conditions():
        stem = condition.replace("/", "_")
        path = RAW / f"{stem}.json"
        if path.exists():
            meta = json.loads(path.read_text())
            if meta["pilot_limit"] is not None:
                continue
            provenance.append(meta)
            files.append(RAW / f"{stem}.csv.gz")
            (OUTPUT / "provenance" / path.name).write_bytes(path.read_bytes())
    if len(files) != len(conditions()):
        raise RuntimeError(f"Need all {len(conditions())} completed conditions; found {len(files)}")
    executed = OUTPUT / "provenance/executed_producers"
    executed.mkdir(exist_ok=True)
    for source_hash in {p["producer_sha256"] for p in provenance}:
        source = CACHE / "executed_sources" / f"{source_hash}.py"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        (executed / source.name).write_bytes(source.read_bytes())
    data = contrast(pd.concat([pd.read_csv(p) for p in files], ignore_index=True))
    data.to_csv(OUTPUT / "episode_contrasts.csv.gz", index=False, float_format="%.17g")
    seed, summary = summarize(data)
    summarize_neurons()
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    for view in ["heldout", "reference"]:
        for scope in ["all_sampled", "first_shared_loss"]:
            plot_widths(summary, view, scope)
    plot_lags(summary)
    plot_widths(summary, "heldout", "all_sampled", pulse=True)
    manifest = dict(
        purpose="Finite causal peer-update pulse responses, not observational lag correlations",
        conditions=[p["condition"] for p in provenance], independent_original_runs=sum(len(p["original_models"]) for p in provenance),
        checkpoint_episodes=sum(len(p["original_models"])*len(p["checkpoints"])*2 for p in provenance),
        aggregation="pool episode numerators/ordinary recipient gains within seed; then equal seed means",
        bootstrap="4000 paired seed resamples, RNG 741; exploratory pointwise intervals",
        positive_peer_benefit="sender-damped harm minus ordinary harm: positive means ordinary sender prevented harm",
        scopes=list(scopes(data)),
        shared_loss="ordinary fixed-reference MSE .0766 to .2; first_shared_loss retains earliest sampled checkpoint in band",
        long_horizon="128-step results use five checkpoint indices; lag plots restrict all horizons to those same indices",
        producer_sha256={p["condition"]: p["producer_sha256"] for p in provenance},
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in RAW.iterdir() if p.suffix in [".gz", ".csv", ".json"]},
        design=args.design,
        pulse_normalization="sum absolute ordinary-vs-local loss gap immediately after pulse; ratio of sums within seed",
        ordinary_layout_checks=sum(p["normal_branch_layout_audits"] for p in provenance),
        original_checkpoint_checks=sum(len(p["original_checkpoint_audits"]) for p in provenance),
        max_budget_error=max(p["max_budget_error"] for p in provenance),
        max_endpoint_error=max(p["all_seed_endpoint_error"] for p in provenance),
        measured_elapsed_seconds=sum(p["elapsed_seconds"] for p in provenance),
        measured_branch_seconds=sum(p["branch_seconds"] for p in provenance),
    )
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Summarized {len(files)} conditions, {len(seed)} seed summaries.")


if __name__ == "__main__":
    main()
