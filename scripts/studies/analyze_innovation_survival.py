"""Summarize and audit the innovation-survival probe, without fitting its theory."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from explore_innovation_survival import exception_margin, projected, train


def paired_interval(difference, rng):
    difference = np.asarray(difference)
    boot = difference[rng.integers(len(difference), size=(4000, len(difference)))].mean(axis=1)
    return dict(mean=float(difference.mean()),
                bootstrap_95=np.quantile(boot, [0.025, 0.975]).tolist())


def main():
    root = Path(__file__).resolve().parents[2]
    target = root / "docs/theory"
    summary = json.loads((target / "innovation_survival_results.json").read_text())
    raw = np.load(target / "innovation_survival_results.npz")
    rng = np.random.default_rng(8123)
    audit = dict(paired_effects=[], common_discretization=[])
    for k in (1, 4):
        narrow = raw[f"k{k}_m1_staged_loss"]
        for name, comparator in [
            ("width_1_minus_16", raw[f"k{k}_m16_staged_loss"]),
            ("width_1_relu_minus_leaky", raw[f"k{k}_leak_after_erosion_loss"]),
            ("width_1_staged_minus_stationary", raw[f"k{k}_m1_stationary_loss"]),
        ]:
            audit["paired_effects"].append(dict(k=k, name=name,
                **paired_interval(narrow - comparator, rng)))
        init_rng = np.random.default_rng(summary["protocol"]["seed"] + 100 + k)
        initial = init_rng.normal(size=(summary["protocol"]["seeds"], 16, k + 1))
        initial *= 0.2 / np.linalg.norm(initial, axis=-1, keepdims=True)
        initial = initial[:32, :4]
        expected = projected(initial, k)
        for dt in (0.2, 0.02):
            fitted, loss, _ = train(initial, k, 1.0, 0, 1000, dt)
            audit["common_discretization"].append(dict(k=k, dt=dt,
                max_projection_error=float(np.max(np.abs(fitted - expected))),
                loss=float(loss.mean()), differing_exception_signs=int(np.sum(
                    (exception_margin(fitted, k, 1) > 0) !=
                    (exception_margin(expected, k, 1) > 0)))))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    colors = {1: "#2369a1", 4: "#b44d24"}
    for k in (1, 4):
        rows = [r for r in summary["training"] if r["k"] == k and r["schedule"] == "staged"]
        widths = np.array([r["width"] for r in rows])
        axes[0].plot(widths, [r["predicted_extinction"] for r in rows],
                     color=colors[k], label=f"k={k}: predicted")
        axes[0].errorbar(widths, [r["projected_extinction"]["mean"] for r in rows],
            yerr=[1.96 * r["projected_extinction"]["se"] for r in rows],
            fmt="o", color=colors[k], capsize=2, label=f"k={k}: 512 starts")
        axes[1].plot(widths, [r["loss"]["mean"] for r in rows], "o-",
                     color=colors[k], label=f"k={k}: staged")
        stationary = [r for r in summary["training"]
                      if r["k"] == k and r["schedule"] == "stationary"]
        axes[1].plot(widths, [r["loss"]["mean"] for r in stationary], "s--",
                     color=colors[k], label=f"k={k}: stationary q=.1")
        rare = [r for r in summary["stationary_rare"] if r["k"] == k and r["width"] == 1]
        q = [r["mass"] for r in rare]
        axes[2].plot(q, [r["no_exception_activation"]["mean"] for r in rare], "o-",
                     color=colors[k], label=f"k={k}: stationary, m=1")
        axes[2].axhline(rows[0]["predicted_extinction"], color=colors[k], linestyle=":")
    axes[0].set(title="Predicted loss of all exception activation", xlabel="Width", ylabel="Probability")
    axes[1].set(title="Early exception exposure prevents damage", xlabel="Width", ylabel="Population half-MSE")
    axes[2].set(title="Rare demand approaches erosion limit", xlabel="Exception probability q", ylabel="Extinction frequency", xscale="log")
    axes[2].axhline(0.5, color="gray", linestyle="--", label="Static absence, m=1")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.savefig(target / "innovation_survival.png", dpi=180)
    plt.close(fig)
    (target / "innovation_survival_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
