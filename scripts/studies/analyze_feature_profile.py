#!/usr/bin/env python3
"""Audit the profiled-feature intervention and publish all paired comparisons."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neural_competition.tasks import paired_gaussian_task
from neural_competition.training import components

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/tables/feature_profile"
FIG = ROOT / "docs/figures/feature_profile"
KEY = ["collection", "degrees", "ratio", "bias", "optimizer", "data_seed"]


def interval(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        return np.nan, np.nan
    rng = np.random.default_rng(845)
    return np.quantile(rng.choice(values, (5000, len(values))).mean(1), [0.025, 0.975])


def extension_tables():
    frames, audits, provenance = [], [], []
    for path in sorted((ROOT / "runs/feature_profile/extension").glob("*.json")):
        meta = json.loads(path.read_text())
        opt = path.stem
        for digest in meta["sources"].values():
            snap = path.parent / "sources" / f"{digest}.txt"
            assert hashlib.sha256(snap.read_bytes()).hexdigest() == digest
            (OUT / "sources" / snap.name).write_bytes(snap.read_bytes())
        original = ROOT / f"runs/feature_profile/main/d9027_a30_r-3_{opt}_none.pt"
        assert hashlib.sha256(original.read_bytes()).hexdigest() == meta["initial_sha256"]
        frame = pd.read_csv(path.with_suffix(".csv"), float_precision="round_trip")
        saved = torch.load(path.with_suffix(".pt"), weights_only=True)
        d = paired_gaussian_task(30, -3, 9027)
        n = len(saved["models"])
        norm = torch.tensor([[math.sqrt(m["width"])] for m in saved["models"]], dtype=torch.float64)
        pred = components(saved["parameters"], d["xtest"][None].expand(n, -1, -1), norm).sum(-1)
        actual = 0.5 * (pred - d["ytest"]).square().mean(1)
        error = float(
            np.max(abs(actual.numpy() - frame[frame.step == 100000].test_loss.to_numpy()))
        )
        assert error < 1e-12
        audits.append(
            dict(
                file=str(path.relative_to(ROOT)),
                models=n,
                max_loss_error=error,
                diverged_models=0,
                source_hashes=True,
            )
        )
        provenance.append(
            dict(
                **meta,
                collection="extension",
                optimizer=opt,
                curve_sha256=hashlib.sha256(path.with_suffix(".csv").read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(path.with_suffix(".pt").read_bytes()).hexdigest(),
            )
        )
        frames.append(frame)
    if frames:
        pd.concat(frames).to_csv(OUT / "extension.csv", index=False)
    return audits, provenance


def main():
    torch.set_num_threads(1)
    frames, audits, provenance = [], [], []
    for path in sorted((ROOT / "runs/feature_profile").glob("*/*.json")):
        if path.parent.name == "extension":
            continue
        meta = json.loads(path.read_text())
        args = meta["args"]
        if args["collection"] == "validation":
            continue
        table = path.with_suffix(".csv.gz")
        frame = pd.read_csv(table, float_precision="round_trip")
        if "diverged" not in frame:
            frame["diverged"] = False
        for key in KEY:
            frame[key] = args[key]
        for source, digest in meta["sources"].items():
            snap = path.parent / "sources" / f"{digest}.txt"
            assert hashlib.sha256(snap.read_bytes()).hexdigest() == digest
            dest = OUT / "sources" / snap.name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(snap.read_bytes())
        statepath = path.with_suffix(".pt")
        saved = torch.load(statepath, weights_only=True)
        d = paired_gaussian_task(args["degrees"], args["ratio"], args["data_seed"])
        assert (
            meta["dataset_hash"]
            == hashlib.sha256(
                d["xtrain"].numpy().tobytes() + d["ytrain"].numpy().tobytes()
            ).hexdigest()
        )
        state = saved["states"][args["steps"]]["parameters"]
        n = len(state)
        norm = torch.tensor([[math.sqrt(m["width"])] for m in saved["models"]], dtype=torch.float64)
        pred = components(state, d["xtest"][None].expand(n, -1, -1), norm).sum(-1)
        actual = 0.5 * (pred - d["ytest"]).square().mean(1)
        final = frame[frame.step == args["steps"]]
        valid = ~final.diverged.to_numpy()
        error = float(np.max(abs(actual.numpy()[valid] - final.test_loss.to_numpy()[valid])))
        assert np.isnan(actual.numpy()[~valid]).all()
        assert error < 1e-12
        # Paired early/late causal branches must have identical untreated histories.
        for seed in sorted(set(m["seed"] for m in saved["models"])):
            ids = {m["arm"]: i for i, m in enumerate(saved["models"]) if m["seed"] == seed}
            for t in [0, args["steps"] - 1000]:
                s = saved["states"][t]
                for k in ["parameters", "first", "second"]:
                    torch.testing.assert_close(
                        s[k][ids["ordinary8"]], s[k][ids["late"]], atol=0, rtol=0
                    )
        audits.append(
            dict(
                file=str(path.relative_to(ROOT)),
                models=n,
                max_loss_error=error,
                diverged_models=int((~valid).sum()),
                source_hashes=True,
                late_branch_identity=True,
            )
        )
        provenance.append(
            dict(
                **meta,
                curve_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(statepath.read_bytes()).hexdigest(),
            )
        )
        frames.append(frame)
    f = pd.concat(frames, ignore_index=True)
    f.to_csv(OUT / "curves.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    repeats = []
    original_arms = [
        "ordinary8",
        "ordinary32",
        "profile",
        "diagonal",
        "early",
        "late",
        "fast_readout",
    ]
    for collection in ["initialization_controls", "history_controls"]:
        repeated = f[(f.collection == collection) & f.arm.isin(original_arms)].merge(
            f[f.collection == "main"],
            on=["optimizer", "seed", "arm", "step"],
            suffixes=("_repeat", "_main"),
        )
        error = (repeated.test_loss_repeat - repeated.test_loss_main).abs().max()
        assert error < 1e-12
        repeats.append(dict(collection=collection, rows=len(repeated), max_loss_error=error))
    pd.DataFrame(repeats).to_csv(OUT / "repeat_audit.csv", index=False)
    final = f[f.step == 10000]
    final.to_csv(OUT / "endpoints.csv", index=False)
    final.groupby(KEY + ["arm"]).agg(
        seeds=("seed", "size"),
        mean_loss=("test_loss", "mean"),
        median_loss=("test_loss", "median"),
        mean_profiled_loss=("profiled_test_loss", "mean"),
        mean_positive=("positive", "mean"),
        diverged=("diverged", "sum"),
    ).reset_index().to_csv(OUT / "summary.csv", index=False)
    effects = []
    for key, group in final.groupby(KEY):
        wide = group.pivot(index="seed", columns="arm", values="test_loss")
        gap = (wide.ordinary8 - wide.ordinary32).mean()
        for arm in [
            a
            for a in [
                "profile",
                "diagonal",
                "early",
                "late",
                "fast_readout",
                "initial_only",
                "reset_only",
                "profile_keep",
                "early_keep",
            ]
            if a in wide
        ]:
            delta = wide[arm] - wide.ordinary8
            lo, hi = interval(delta)
            row = dict(zip(KEY, key))
            row.update(
                arm=arm,
                seeds=len(delta),
                difference=delta.mean(),
                low=lo,
                high=hi,
                wins=int((delta < 0).sum()),
                ordinary_width_gap=gap,
                fraction_mean_gap_closed=float(-delta.mean() / gap) if gap > 1e-8 else np.nan,
            )
            effects.append(row)
        for control in [
            a
            for a in [
                "diagonal",
                "early",
                "late",
                "fast_readout",
                "initial_only",
                "reset_only",
                "profile_keep",
                "early_keep",
            ]
            if a in wide
        ]:
            delta = wide.profile - wide[control]
            lo, hi = interval(delta)
            effects.append(
                dict(zip(KEY, key))
                | dict(
                    arm="profile_minus_" + control,
                    seeds=len(delta),
                    difference=delta.mean(),
                    low=lo,
                    high=hi,
                    wins=int((delta < 0).sum()),
                )
            )
    pd.DataFrame(effects).to_csv(OUT / "paired_effects.csv", index=False)
    contrasts = []
    for opt in ["sgd", "momentum", "adam"]:
        init = final[
            (final.collection == "initialization_controls") & (final.optimizer == opt)
        ].pivot(index="seed", columns="arm", values="test_loss")
        history = final[(final.collection == "history_controls") & (final.optimizer == opt)].pivot(
            index="seed", columns="arm", values="test_loss"
        )
        for label, delta in [
            ("early_minus_initial_only", init.early - init.initial_only),
            ("early_keep_minus_initial_only", history.early_keep - init.initial_only),
            ("early_keep_minus_reset_only", history.early_keep - init.reset_only),
        ]:
            lo, hi = interval(delta)
            contrasts.append(
                dict(
                    optimizer=opt,
                    contrast=label,
                    seeds=len(delta),
                    difference=delta.mean(),
                    low=lo,
                    high=hi,
                    wins=int((delta < 0).sum()),
                )
            )
    pd.DataFrame(contrasts).to_csv(OUT / "causal_controls.csv", index=False)
    # Prospective endpoint prediction using only ordinary width-8 states at step 1000.
    # Hold out whole task settings and fit separately within each optimizer.
    ordinary = f[(f.arm == "ordinary8") & f.collection.isin(["main", "controls"])]
    early = ordinary[ordinary.step == 1000].merge(
        ordinary[ordinary.step == 10000][KEY + ["seed", "test_loss"]],
        on=KEY + ["seed"],
        suffixes=("_early", "_final"),
    )
    forecasts = []
    for opt, g in early.groupby("optimizer"):
        g = g.copy()
        g["setting"] = g.degrees.astype(str) + "_" + g.ratio.astype(str) + "_" + g.bias
        for setting in g.setting.unique():
            train = g[g.setting != setting]
            test = g[g.setting == setting]
            if len(train) < 10:
                continue
            for model in ["loss_only", "loss_and_profile"]:

                def design(h):
                    cols = [np.ones(len(h)), np.log10(h.train_loss.clip(lower=1e-12))]
                    if model == "loss_and_profile":
                        cols += [
                            np.log10(h.profiled_train_loss.clip(lower=1e-12)),
                            np.log10(h.equilibration_gap.clip(lower=1e-12)),
                        ]
                    return np.stack(cols, 1)

                coef = np.linalg.lstsq(
                    design(train), np.log10(train.test_loss_final.clip(lower=1e-12)), rcond=None
                )[0]
                pred = design(test) @ coef
                truth = np.log10(test.test_loss_final.clip(lower=1e-12)).to_numpy()
                for seed, a, b in zip(test.seed, pred, truth):
                    forecasts.append(
                        dict(
                            optimizer=opt,
                            setting=setting,
                            seed=seed,
                            model=model,
                            predicted_log10_loss=a,
                            actual_log10_loss=b,
                            squared_error=(a - b) ** 2,
                        )
                    )
    forecasts = pd.DataFrame(forecasts)
    forecasts.to_csv(OUT / "forecasts.csv", index=False)
    if len(forecasts):
        forecasts.groupby(["optimizer", "model"]).squared_error.mean().reset_index().to_csv(
            OUT / "forecast_summary.csv", index=False
        )
    extension_audits, extension_provenance = extension_tables()
    audits.extend(extension_audits)
    provenance.extend(extension_provenance)
    pd.DataFrame(audits).to_csv(OUT / "audit.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (OUT / "analysis.json").write_text(
        json.dumps(
            dict(
                sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                bootstrap_seed=845,
                bootstrap_draws=5000,
                scope="Exploratory paired seed intervals, no multiplicity correction",
            ),
            indent=2,
        )
        + "\n"
    )
    plt.rcParams.update({"font.size": 9, "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    main = final[final.collection == "main"]
    arms = ["ordinary8", "diagonal", "fast_readout", "profile", "early", "late", "ordinary32"]
    for ax, opt in zip(axes, ["sgd", "momentum", "adam"]):
        g = main[main.optimizer == opt]
        for i, arm in enumerate(arms):
            values = g[g.arm == arm].test_loss * 200
            ax.scatter(np.full(len(values), i), values.clip(lower=1e-6), alpha=0.3, s=12)
            below = values < 1e-6
            if below.any():
                ax.scatter([i], [1e-6], marker="v", s=35, color="black", clip_on=False)
            ax.plot([i - 0.25, i + 0.25], [values.mean()] * 2, color="black", lw=2)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 20)
        ax.set_xticks(
            range(len(arms)),
            ["Ord. 8", "Diagonal", "Fast out", "Profile", "Early", "Late", "Ord. 32"],
            rotation=40,
            ha="right",
        )
        ax.set_title(opt)
        ax.set_ylabel("Test MSE / target variance (%)")
    fig.suptitle(
        "Coupled readout interventions: fresh paired starts; triangles mark values below 10⁻⁶%"
    )
    fig.tight_layout()
    fig.savefig(FIG / "interventions.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, opt in zip(axes, ["sgd", "momentum", "adam"]):
        group = f[(f.collection == "main") & (f.optimizer == opt)]
        for arm in ["ordinary8", "profile", "early", "late", "ordinary32"]:
            curve = group[group.arm == arm].groupby("step").test_loss.mean()
            ax.plot(curve.index, 200 * curve, label=arm)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("log")
        ax.set_title(opt)
        ax.set_xlabel("Updates")
        ax.set_ylabel("Test MSE / target variance (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "trajectories.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(effects).query('collection=="main"').to_string(index=False))
    print(
        "Manifests:",
        len(audits),
        "models:",
        len(final),
        "max loss error:",
        max(a["max_loss_error"] for a in audits),
    )


if __name__ == "__main__":
    main()
