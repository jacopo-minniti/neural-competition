#!/usr/bin/env python3
"""Objective-scale accounting and exploratory forecasts from existing cohorts.

No optimization is performed. Uses retained tables plus the verified strict
replay caches. Prediction features stop at the declared prefix; held-out seed
outcomes never enter model fitting. Run from the repository root.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_trajectory import interval

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "runs/retained/aggregates"
OUTPUT = ROOT / "docs/theory/checks/effect_magnitude"
FIGURES = ROOT / "docs/figures/effect_magnitude"
HASHES = {}
PREFIXES = [1000, 5000, 10000]
MAIN_CONDITIONS = [
    "k8_teacher2027/momentum_01",
    "k8_teacher2027/adam_001",
    "k8_teacher2027/adam_01",
    "k8_teacher2027_init01/adam_001",
    "k8_teacher2027_init1/adam_001",
]
FEATURES = ["raw_Cplus", "raw_Cnet", "net_fraction", "strict_F", "solo_harm", "rho", "credit"]
LABELS = {
    "raw_Cplus": "Accumulated C+", "raw_Cnet": "Accumulated net pair cost",
    "net_fraction": "Net pair cost / G", "strict_F": "Strict frequency",
    "solo_harm": "Standalone harm / G", "rho": "Total quadratic cost / linear work",
    "credit": "Favorable pair credit",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, **kwargs):
    path = Path(path)
    HASHES[str(path.relative_to(ROOT))] = digest(path)
    return pd.read_csv(path, **kwargs)


def save_frame(frame, name):
    frame.to_csv(OUTPUT / f"{name}.csv", index=False, float_format="%.17g")


def objective_budget(frame):
    """Recover the decomposition-independent total-step budget from old sums."""
    frame = frame.copy()
    assert (frame.penalty > 0).all() and (frame.penalty_per_energy > 0).all()
    frame["individual_energy"] = frame.penalty / frame.penalty_per_energy
    frame["net_solo_gain"] = frame.gain - frame.harm
    frame["linear_work"] = frame.net_solo_gain + frame.individual_energy
    frame["quadratic_cost"] = frame.individual_energy + frame.net
    frame["rho"] = frame.quadratic_cost / frame.linear_work.where(frame.linear_work > 0)
    # s_r=-<e,d_r>, A_r=-s_r+I_r. Thus sum[-s_r]+ lies between
    # max(sum[A_r]+ - sum I_r, 0) and sum[A_r]+, also after pooling time.
    frame["antiwork_lower"] = (frame.harm - frame.individual_energy).clip(lower=0)
    frame["antiwork_upper"] = frame.harm
    for side in ["lower", "upper"]:
        frame["anti_fraction_" + side] = frame["antiwork_" + side] / (
            frame.linear_work + frame["antiwork_" + side]
        ).where(frame.linear_work > 0)
    if "initial_loss" in frame:
        frame["progress"] = frame.initial_loss - frame.final_loss
        error = frame.progress - (frame.net_solo_gain - frame.net)
        assert error.abs().max() < 1e-10, error.abs().max()
        assert np.allclose(frame.progress, frame.linear_work - frame.quadratic_cost, atol=1e-10)
        for numerator in ["penalty", "credit", "net", "harm", "quadratic_cost"]:
            frame[numerator + "_per_progress"] = frame[numerator] / frame.progress
    return frame


def budgets():
    robust = read(ROOT / "docs/tables/robustness_per_seed_results.csv")
    robust = robust[robust.view == "reference"].assign(study="robustness")
    old = read(ROOT / "docs/tables/cumulative_interference_seeds.csv")
    old = old[old.view == "reference"].assign(study="trajectory", teacher_width=8)
    old["condition"] = "teacher8/" + old.optimizer
    frame = objective_budget(pd.concat([robust, old], ignore_index=True))
    save_frame(frame, "cumulative_per_seed")
    rows = []
    for (study, condition), group in frame.groupby(["study", "condition"]):
        k = int(group.teacher_width.iloc[0])
        small = group[group.width == k].set_index("seed")
        for width, large in group[group.width != k].groupby("width"):
            large = large.set_index("seed").loc[small.index]
            for seed in small.index:
                n, w = small.loc[seed], large.loc[seed]
                parts = dict(
                    initial_contribution=n.initial_loss - w.initial_loss,
                    standalone_contribution=w.net_solo_gain - n.net_solo_gain,
                    interaction_contribution=n.net - w.net,
                )
                gap = n.final_loss - w.final_loss
                assert abs(sum(parts.values()) - gap) < 1e-10
                rows.append(dict(
                    study=study, condition=condition, baseline=k, width=width, seed=seed,
                    reference_loss_gap=gap, **parts,
                    narrow_penalty=n.penalty, wide_penalty=w.penalty,
                    narrow_credit=n.credit, wide_credit=w.credit,
                    narrow_net=n.net, wide_net=w.net,
                    narrow_progress=n.progress, wide_progress=w.progress,
                    narrow_initial=n.initial_loss, wide_initial=w.initial_loss,
                    antiwork_reduction_lower=n.antiwork_lower - w.antiwork_upper,
                    antiwork_reduction_upper=n.antiwork_upper - w.antiwork_lower,
                    anti_fraction_reduction_lower=n.anti_fraction_lower - w.anti_fraction_upper,
                    anti_fraction_reduction_upper=n.anti_fraction_upper - w.anti_fraction_lower,
                ))
    contrasts = pd.DataFrame(rows)
    save_frame(contrasts, "budget_contrasts_per_seed")
    save_frame(aggregate(contrasts, ["study", "condition", "baseline", "width"]), "budget_contrasts")
    return robust, frame


def aggregate(frame, keys, metrics=None):
    if metrics is None:
        metrics = [k for k in frame.select_dtypes(include=np.number) if k not in [*keys, "seed"]]
    rows = []
    for key, group in frame.groupby(keys):
        base = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for metric in metrics:
            mean, low, high = interval(group[metric])
            rows.append(dict(**base, metric=metric, mean=mean, low=low, high=high,
                             n=int(np.isfinite(group[metric]).sum())))
    return pd.DataFrame(rows)


def matched_efficiency():
    robust = read(ARCHIVE / "robustness/loss_bins.csv.gz")
    robust = robust[
        robust.condition.isin(MAIN_CONDITIONS) & (robust.bin_set == "primary")
        & robust.bin.between(7, 11)
    ].assign(study="robustness", baseline=8, target_width=32, expected_bins=5)
    old = read(ARCHIVE / "trajectory/loss_bins.csv.gz")
    old = old[old.bin_set == "primary"].assign(
        study="trajectory", baseline=8, target_width=16, expected_bins=12
    )
    old["condition"] = "teacher8/" + old.optimizer
    raw = pd.concat([robust, old], ignore_index=True)
    identity = ["study", "condition", "view", "width", "seed"]
    counts = raw.groupby(identity).bin.nunique()
    expected = raw.groupby(identity).expected_bins.first()
    assert counts.eq(expected).all()
    metrics = ["S_plus", "S_minus", "S_net", "solo_harm_ratio", "retained_gain"]
    assert raw[metrics].notna().all().all()
    per_seed = raw.groupby(identity)[metrics].mean().reset_index()
    assert np.allclose(
        per_seed.retained_gain,
        1 - per_seed.solo_harm_ratio - per_seed.S_plus + per_seed.S_minus, atol=1e-12,
    )
    save_frame(per_seed, "matched_efficiency_per_seed")
    save_frame(aggregate(per_seed, identity[:-1], metrics), "matched_efficiency_summary")
    rows = []
    for (study, condition, view), group in per_seed.groupby(["study", "condition", "view"]):
        small = group[group.width == 8].set_index("seed")
        target = 16 if study == "trajectory" else 32
        large = group[group.width == target].set_index("seed").loc[small.index]
        for seed in small.index:
            n, w = small.loc[seed], large.loc[seed]
            parts = dict(
                less_standalone_harm=n.solo_harm_ratio - w.solo_harm_ratio,
                less_positive_pair_cost=n.S_plus - w.S_plus,
                more_pair_credit=w.S_minus - n.S_minus,
            )
            difference = w.retained_gain - n.retained_gain
            assert abs(sum(parts.values()) - difference) < 1e-12
            rows.append(dict(
                study=study, condition=condition, view=view, seed=seed, width=target,
                retention_difference=difference, **parts,
                net_pair_difference=n.S_net - w.S_net,
                narrow_retention=n.retained_gain, wide_retention=w.retained_gain,
            ))
    contrasts = pd.DataFrame(rows)
    save_frame(contrasts, "matched_contrasts_per_seed")
    save_frame(aggregate(contrasts, ["study", "condition", "view", "width"]), "matched_contrasts")
    return per_seed


def strict_scale(robust):
    parts = []
    folder = ROOT / "docs/theory/checks/strict_matched_loss"
    for condition in MAIN_CONDITIONS:
        frame = read(folder / f"{condition.replace('/', '_')}.csv.gz")
        parts.append(frame[(frame.bin_set == "full") & (frame.view == "reference")])
    strict = pd.concat(parts, ignore_index=True)
    strict = strict[["condition", "width", "seed", "strong_harm", "strong_events"]]
    strict = strict.merge(robust, on=["condition", "width", "seed"], validate="one_to_one")
    strict["progress"] = strict.initial_loss - strict.final_loss
    strict["strict_harm_per_progress"] = strict.strong_harm / strict.progress
    save_frame(strict, "strict_cumulative_scale")
    save_frame(aggregate(strict, ["condition", "width"],
                         ["strong_harm", "strong_events", "progress", "strict_harm_per_progress"]),
               "strict_cumulative_summary")


def direct_progress():
    raw = read(ARCHIVE / "momentum/actual_step_seeds.csv.gz")
    raw = raw[
        raw.optimizer.isin(["momentum_01", "sgd_1"])
        & (((raw.comparison == "mse_0.2") & (raw.probes == 512))
           | ((raw.comparison == "mse_0.01") & (raw.probes == 128)))
    ]
    rows = []
    for (optimizer, comparison, scale), group in raw.groupby(["optimizer", "comparison", "scale"]):
        n = group[group.width == 8].set_index("seed")
        w = group[group.width == 16].set_index("seed").loc[n.index]
        base, wide = -n.loss_change.to_numpy(), -w.loss_change.to_numpy()
        mean, low, high = interval(wide - base)
        rows.append(dict(
            optimizer=optimizer, comparison=comparison, scale=scale, n=len(n),
            narrow_gain=base.mean(), wide_gain=wide.mean(), difference=mean, low=low, high=high,
            relative_gain=wide.mean() / base.mean() - 1,
        ))
    save_frame(pd.DataFrame(rows), "direct_objective_progress")
    adam = read(ROOT / "docs/tables/adam_cross_batch_per_seed.csv")
    adam = adam[adam.comparison == "mse_0.2"]
    rows = []
    for profile, group in adam.groupby("profile"):
        n = group[group.width == 8].set_index("seed")
        w = group[group.width == 32].set_index("seed").loc[n.index]
        difference = w.expected_gain - n.expected_gain
        tail = n.expected_harm - w.expected_harm
        rows.append(dict(
            profile=profile, narrow_first_order_gain=n.expected_gain.mean(),
            wide_first_order_gain=w.expected_gain.mean(),
            first_order_gain_difference=difference.mean(),
            harmful_tail_reduction=tail.mean(), tail_fraction_of_gain_difference=tail.mean()/difference.mean(),
            narrow_harm=n.expected_harm.mean(), wide_harm=w.expected_harm.mean(),
        ))
    save_frame(pd.DataFrame(rows), "adam_first_order_scale")


def prediction_pairs(robust):
    chunks = []
    for frame in read(ARCHIVE / "robustness/cumulative.csv.gz", chunksize=50_000):
        chunks.append(frame[(frame.view == "reference") & frame.updates.isin(PREFIXES)])
    prefix = objective_budget(pd.concat(chunks, ignore_index=True))
    history = read(ARCHIVE / "robustness/loss_history.csv.gz")
    keys = ["condition", "width", "seed"]
    initial = history[history.updates == 0][[*keys, "train_mse"]].rename(
        columns={"train_mse": "initial_train_mse"}
    )
    zero = history[(history.updates == 0) & ~history.condition.str.contains("_init")]
    scale = zero.groupby(["teacher_width", "teacher_seed"]).train_mse.first() / 2
    prefix = prefix.merge(history[[*keys, "updates", "train_mse"]], on=[*keys, "updates"])
    prefix = prefix.merge(initial, on=keys)
    prefix = prefix.merge(robust[[*keys, "window_mse"]], on=keys)
    prefix["objective_scale"] = [scale.loc[k, s] for k, s in zip(prefix.teacher_width, prefix.teacher_seed)]
    assert len(prefix) == 205 * len(PREFIXES)
    rows = []
    for t in PREFIXES:
        auc = {}
        for key, part in history[history.updates >= t].groupby(keys):
            part = part.sort_values("updates")
            assert part.updates.iloc[0] == t and part.updates.iloc[-1] == 100_000
            auc[key] = np.trapezoid(part.train_mse.to_numpy() / 2, part.updates.to_numpy()) / (100_000 - t)
        for condition, group in prefix[prefix.updates == t].groupby("condition"):
            baseline = int(group.teacher_width.iloc[0])
            narrow = group[group.width == baseline].set_index("seed")
            for width, wide in group[group.width != baseline].groupby("width"):
                wide = wide.set_index("seed")
                for seed in range(5):
                    n, w = narrow.loc[seed], wide.loc[seed]
                    z = n.objective_scale
                    row = dict(
                        condition=condition, width=width, seed=seed, prefix=t,
                        group=f"{condition}:{width}", profile=n.profile,
                        future_auc=(auc[condition, baseline, seed] - auc[condition, width, seed]) / z,
                        late_loss=(n.window_mse - w.window_mse) / (2*z),
                        loss_n=n.train_mse / (2*z), loss_w=w.train_mse / (2*z),
                        initial_gap=(n.initial_train_mse - w.initial_train_mse) / (2*z),
                        raw_Cplus=(n.penalty - w.penalty)/z, raw_Cnet=(n.net - w.net)/z,
                        net_fraction=n.S_net - w.S_net,
                        strict_F=n.strong_frequency - w.strong_frequency,
                        solo_harm=n.solo_harm_ratio - w.solo_harm_ratio,
                        rho=n.rho - w.rho, credit=(w.credit - n.credit)/z,
                    )
                    rows.append(row)
    frame = pd.DataFrame(rows)
    assert np.isfinite(frame.select_dtypes(include=np.number)).all().all()
    save_frame(frame, "prediction_pairs")
    return frame


def fit_predict(design_groups, x, y, train, test):
    """Standardization and coefficients use the training seeds only."""
    mean, std = x[train].mean(0), x[train].std(0)
    std = np.where(std > 1e-12, std, 1)
    design = np.column_stack([design_groups, (x - mean)/std])
    coefficients = np.linalg.lstsq(design[train], y[train], rcond=1e-10)[0]
    return design[test] @ coefficients


def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan


def forecasts(pairs):
    records, scores = [], []
    for t, frame in pairs.groupby("prefix"):
        frame = frame.reset_index(drop=True)
        groups = pd.get_dummies(frame.group, dtype=float).to_numpy()
        base = frame[["loss_n", "loss_w", "initial_gap"]].to_numpy()
        for target in ["future_auc", "late_loss"]:
            y = frame[target].to_numpy()
            predictions = {k: np.empty(len(frame)) for k in ["baseline", *FEATURES]}
            for seed in range(5):
                train = frame.seed.to_numpy() != seed
                test = ~train
                predictions["baseline"][test] = fit_predict(groups, base, y, train, test)
                for metric in FEATURES:
                    x = np.column_stack([base, frame[metric].to_numpy()])
                    predictions[metric][test] = fit_predict(groups, x, y, train, test)
            baseline_error = (predictions["baseline"] - y)**2
            for metric, pred in predictions.items():
                errors = (pred - y)**2
                row = dict(
                    prefix=t, target=target, feature=metric, n=len(frame), seed_folds=5,
                    baseline_rmse=np.sqrt(baseline_error.mean()), rmse=np.sqrt(errors.mean()),
                    relative_rmse_reduction=1-np.sqrt(errors.mean()/baseline_error.mean()),
                    improved_seed_folds=sum(
                        errors[frame.seed == s].mean() < baseline_error[frame.seed == s].mean()
                        for s in range(5)
                    ),
                )
                if metric != "baseline":
                    x = frame[metric] - frame.groupby("group")[metric].transform("mean")
                    centered_y = frame[target] - frame.groupby("group")[target].transform("mean")
                    means = frame.groupby("group")[[metric, target]].mean()
                    row.update(within_comparison_r=correlation(x, centered_y),
                               between_comparison_r=correlation(means[metric], means[target]))
                scores.append(row)
                for i, part in frame.iterrows():
                    records.append(dict(
                        condition=part.condition, width=part.width, seed=part.seed,
                        prefix=t, target=target, feature=metric,
                        observed=y[i], predicted=pred[i], squared_error=errors[i],
                    ))
    scores = pd.DataFrame(scores)
    save_frame(scores, "prediction_scores")
    save_frame(pd.DataFrame(records), "held_out_predictions")
    return scores


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plots(efficiency, scores):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for study, conditions in [
        ("trajectory", ["teacher8/momentum", "teacher8/sgd"]),
        ("robustness", MAIN_CONDITIONS),
    ]:
        fig, axes = plt.subplots(1, len(conditions), figsize=(3*len(conditions), 4.4), layout="constrained")
        for ax, condition in zip(np.atleast_1d(axes), conditions):
            part = efficiency[(efficiency.study == study) & (efficiency.condition == condition)
                              & (efficiency.view == "reference")]
            target = 16 if study == "trajectory" else 32
            part = part[part.width.isin([8, target])].groupby("width").mean(numeric_only=True)
            bottom = np.zeros(2)
            for metric, color, label in [
                ("retained_gain", "#2563eb", "Actual objective progress / G"),
                ("solo_harm_ratio", "#d97706", "Standalone harm / G"),
                ("S_net", "#dc2626", "Net pair cost / G"),
            ]:
                values = part[metric].to_numpy()*100
                ax.bar([0, 1], values, bottom=bottom, color=color, width=.62, label=label)
                bottom += values
            for i, value in enumerate(part.retained_gain*100):
                ax.text(i, value/2, f"{value:.1f}%", ha="center", va="center", color="white")
            title = condition.split("/")[-1].replace("momentum_01", "Momentum 0.1")
            title = title.replace("adam_001", "Adam 0.001").replace("adam_01", "Adam 0.01")
            title = title.replace("momentum", "Momentum").replace("sgd", "SGD")
            if "_init" in condition:
                title += "\noutput std " + ("0.1" if "_init01" in condition else "1")
            ax.set_title(title)
            ax.set_xticks([0, 1], ["width 8", f"width {target}"])
            ax.set_ylim(0, 103)
            ax.set_ylabel("Units per 100 standalone helpful units")
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
        window = "0.02–0.2; 10 seeds" if study == "trajectory" else "0.0766–0.2; 5 seeds"
        fig.suptitle(f"Objective accounting at matched reference loss\nMSE {window}; equal log-bin weights, then equal seed weights")
        save_figure(fig, f"{study}_matched_efficiency")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, target in zip(axes, ["future_auc", "late_loss"]):
        for feature in ["raw_Cplus", "net_fraction", "strict_F", "solo_harm"]:
            part = scores[(scores.feature == feature) & (scores.target == target)]
            ax.plot(range(3), 100*part.relative_rmse_reduction, "o-", label=LABELS[feature])
        ax.axhline(0, color="black", lw=.8)
        ax.set_xticks(range(3), ["1k", "5k", "10k"])
        ax.set_xlabel("Cumulative features stop at update")
        ax.set_ylabel("Reduction in prediction RMSE (%)")
        ax.set_title("Later average training loss" if target == "future_auc" else "Late training loss (91k–100k)")
        ax.grid(alpha=.15)
    axes[1].legend(fontsize=8)
    fig.suptitle("Do early measures add information beyond early loss and the width setting?\n205 original runs; hold out one entire paired seed; prediction error is not loss benefit")
    save_figure(fig, "predictive_increment")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    robust, _ = budgets()
    efficiency = matched_efficiency()
    strict_scale(robust)
    direct_progress()
    pairs = prediction_pairs(robust)
    scores = forecasts(pairs)
    plots(efficiency, scores)
    provenance = dict(
        date="2026-10-18", training_performed=False,
        loss_units="half-MSE; reference budgets telescope; source budgets do not",
        matched_windows={"trajectory": [.02, .2], "robustness": [.07662373699114575, .2]},
        prediction_prefixes=PREFIXES, prediction_runs=205, width_contrasts_per_prefix=140,
        independent_paired_seed_folds=5,
        predictive_baseline="comparison fixed effects plus current narrow/wide full-train loss and initial full-train loss gap",
        predictive_features=FEATURES,
        predictive_targets="future mean full-training half-MSE gap (trapezoid, 1000-update grid), and mean half-MSE gap over 91k–100k",
        target_normalization="teacher's zero-predictor full-training half-MSE, common across initializations/widths",
        leakage_control="fit coefficients and feature standardization using only other initialization seeds",
        caveat="exploratory; held-out seeds share one fixed experimental design; comparison effects control away the average width benefit",
        script_sha256=digest(Path(__file__)), inputs_sha256=HASHES,
        helper_sha256={"scripts/studies/analyze_trajectory.py": digest(
            ROOT / "scripts/studies/analyze_trajectory.py"
        )},
        numpy_version=np.__version__, pandas_version=pd.__version__,
    )
    (OUTPUT / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Wrote cumulative budgets, matched efficiencies, direct progress and held-out-seed forecasts.")


if __name__ == "__main__":
    main()
