#!/usr/bin/env python3
"""Plot exact running G/H/net-pair sums from the original every-update studies.

The archive saves exact prefix sums at selected plotting steps: nothing is
integrated from sparse diagnostic samples. No optimization is performed.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "runs/retained"
OUT = ROOT / "docs/theory/checks/cumulative_budget"
FIGURES = ROOT / "docs/figures/cumulative_budget"
CONDITIONS = [
    "k8_teacher2027/momentum_01", "k8_teacher2027/adam_001", "k8_teacher2027/adam_01",
    "k8_teacher2027_init01/adam_001", "k8_teacher2027_init1/adam_001",
]
COLORS = {8: "#2563eb", 16: "#df6b20", 32: "#9358bd"}
METRICS = {
    "gain": "Helpful standalone gains: G(T)",
    "harm": "Standalone harm: H(T)",
    "net": "Net pair cost: C_net(T)",
    "progress": "Actual loss decrease: L(0) − L(T)",
}
HASHES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, **kwargs):
    HASHES[str(path.relative_to(ROOT))] = digest(path)
    return pd.read_csv(path, **kwargs)


def collect():
    parts = []
    for chunk in read(ARCHIVE / "aggregates/robustness/cumulative.csv.gz", chunksize=50_000):
        parts.append(chunk[(chunk.view == "reference") & chunk.condition.isin(CONDITIONS)])
    robust = pd.concat(parts, ignore_index=True).assign(study="robustness")
    old = read(ARCHIVE / "aggregates/trajectory/cumulative.csv.gz")
    old = old[old.view == "reference"].copy()
    old["condition"] = "teacher8/" + old.optimizer
    old["study"] = "trajectory"
    frame = pd.concat([robust, old], ignore_index=True)
    final = read(ROOT / "docs/tables/robustness_per_seed_results.csv")
    final = final[(final.view == "reference") & final.condition.isin(CONDITIONS)]
    old_final = read(ROOT / "docs/tables/cumulative_interference_seeds.csv")
    old_final = old_final[old_final.view == "reference"].copy()
    old_final["condition"] = "teacher8/" + old_final.optimizer
    final = pd.concat([final, old_final], ignore_index=True)
    keys = ["condition", "width", "seed"]
    assert not final.duplicated(keys).any()
    frame = frame.merge(final[[*keys, "initial_loss", "final_loss"]], on=keys, validate="many_to_one")
    frame["progress"] = frame.gain - frame.harm - frame.net
    frame["reference_loss_after"] = frame.initial_loss - frame.progress
    assert frame.reference_loss_after.min() >= -1e-10
    assert np.allclose(frame.net, frame.penalty - frame.credit, atol=1e-11)
    frame = frame.sort_values([*keys, "updates"])
    # Helpful gains and standalone harm are nonnegative at each actual update.
    # Net pair cost and actual objective progress need not be monotone.
    for _, group in frame.groupby(keys):
        assert (group.gain.diff().dropna() >= -1e-11).all()
        assert (group.harm.diff().dropna() >= -1e-11).all()
    last = frame.groupby(keys).tail(1)
    error = np.max(np.abs(last.reference_loss_after - last.final_loss))
    assert error < 1e-10, error
    joined = last.merge(final, on=keys, suffixes=("_curve", "_original"))
    for metric in ["gain", "harm", "net"]:
        assert np.allclose(joined[metric + "_curve"], joined[metric + "_original"], atol=1e-10)
    # Explicit zero prefixes make the origin clear. There is no training update 0.
    zero = frame.groupby(keys).head(1).copy()
    zero["updates"] = 0
    zero[["gain", "harm", "net", "penalty", "credit", "progress"]] = 0.0
    zero["reference_loss_after"] = zero.initial_loss
    frame = pd.concat([zero, frame], ignore_index=True).sort_values([*keys, "updates"])
    return frame, float(error)


def raw_loss_audit(frame):
    """Compare reconstructed loss curves to independent raw after-step forwards."""
    path = ARCHIVE / "manifest.json"
    HASHES[str(path.relative_to(ROOT))] = digest(path)
    manifest = json.loads(path.read_text())
    audits = []
    for archive in manifest["update_archives"]:
        name = Path(archive["path"]).stem
        if name.startswith("trajectory_"):
            condition = "teacher8/" + name.removeprefix("trajectory_")
        elif name.startswith("robustness_"):
            condition = name.removeprefix("robustness_").rsplit("_", 2)
            condition = condition[0] + "/" + "_".join(condition[1:])
        else:
            continue
        part = frame[(frame.condition == condition) & (frame.updates > 0)]
        if part.empty:
            continue
        path = ARCHIVE / archive["path"]
        HASHES[str(path.relative_to(ROOT))] = digest(path)
        records = np.load(path)["records"]
        for index, model in enumerate(archive["models"]):
            group = part[(part.width == model["width"]) & (part.seed == model["seed"])]
            if group.empty:
                continue
            actual = records[group.updates.to_numpy(dtype=int) - 1, index, 1, 1]
            error = np.max(np.abs(actual - group.reference_loss_after.to_numpy()))
            assert error < 1e-10, (condition, model, error)
            audits.append(dict(condition=condition, **model, points=len(group), max_loss_error=float(error)))
    return audits


def curve(ax, frame, metric):
    for width, group in frame.groupby("width"):
        if width not in COLORS:
            continue
        values = group.pivot(index="seed", columns="updates", values=metric).sort_index(axis=1)
        assert values.notna().all().all()
        x, a = values.columns.to_numpy(), values.to_numpy()
        rng = np.random.default_rng(741)
        ids = rng.integers(len(a), size=(1500, len(a)))
        # Same bootstrap draws at every time point and across metric panels.
        boot = a[ids].mean(1)
        low, high = np.quantile(boot, [.025, .975], axis=0)
        mean = a.mean(0)
        ax.plot(x, mean, color=COLORS[width], lw=1.7, label=f"width {width}")
        ax.fill_between(x, low, high, color=COLORS[width], alpha=.13, linewidth=0)


def plot(frame, conditions, labels, name):
    fig, axes = plt.subplots(len(conditions), 4, figsize=(13.8, 2.8*len(conditions)),
                             squeeze=False, layout="constrained")
    for row, (condition, label) in enumerate(zip(conditions, labels)):
        part = frame[frame.condition == condition]
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            curve(ax, part, metric)
            ax.set_title(METRICS[metric], fontsize=10)
            ax.set_xlim(0, part.updates.max())
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t/1000:g}k" if t else "0"))
            ax.set_xlabel("Actual training updates")
            ax.grid(alpha=.16)
            ax.axhline(0, color="black", lw=.5, alpha=.3)
            if col == 0:
                ax.set_ylabel(f"{label}\n\nCumulative half-MSE units")
                ax.legend(fontsize=8)
    fig.suptitle("Running objective budget on the fixed reference set\n"
                 "Exact every-update sums; curves are seed means; bands are pointwise 95% seed intervals")
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame, final_error = collect()
    audits = raw_loss_audit(frame)
    columns = ["study", "condition", "width", "seed", "updates", "gain", "harm", "net",
               "penalty", "credit", "progress", "reference_loss_after", "initial_loss"]
    frame[columns].to_csv(OUT / "budget_curves.csv.gz", index=False, float_format="%.17g")
    last = frame.groupby(["condition", "width", "seed"]).tail(1)
    last[columns].to_csv(OUT / "endpoints_per_seed.csv", index=False, float_format="%.17g")
    summary = last.groupby(["study", "condition", "width"])[
        ["gain", "harm", "net", "progress", "reference_loss_after", "initial_loss"]
    ].mean().reset_index()
    summary.to_csv(OUT / "endpoints_mean.csv", index=False, float_format="%.17g")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    plot(frame, CONDITIONS[1:3], ["Adam lr 0.001", "Adam lr 0.01"], "adam_cumulative_budget")
    plot(frame, ["teacher8/momentum", "teacher8/sgd"],
         ["Momentum: 200k, 10 seeds", "SGD: 200k, 10 seeds"], "sgd_momentum_cumulative_budget")
    plot(frame, [CONDITIONS[0]], ["Momentum: 100k, 5 seeds"], "momentum_100k_cumulative_budget")
    plot(frame, [CONDITIONS[1], *CONDITIONS[3:]],
         ["Adam .001: output std 0", "Adam .001: output std .1", "Adam .001: output std 1"],
         "adam_initialization_cumulative_budget")
    provenance = dict(
        training_performed=False, loss_units="raw half-MSE on the fixed training-reference subset",
        normalization="none", loss_binning=False,
        prefix_sums="each original actual update contributes; only plotted time points are subsampled",
        displayed_widths={"trajectory": [8, 16], "robustness": [8, 16, 32]},
        additional_exported_widths=[10], independent_runs=len(last),
        progress_identity="G(T)-H(T)-C_net(T) = L_ref(0)-L_ref(T)",
        reference_mse_field_note="original cumulative reference_mse is pre-update; after-step loss is reconstructed and audited",
        max_endpoint_error=final_error, independent_raw_forward_checks=audits,
        input_sha256=HASHES, script_sha256=digest(Path(__file__)),
        numpy_version=np.__version__, pandas_version=pd.__version__,
    )
    (OUT / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote running budget plots for {len(last)} original runs; endpoint error {final_error:.3g}.")


if __name__ == "__main__":
    main()
