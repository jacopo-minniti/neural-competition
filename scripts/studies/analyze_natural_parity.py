"""Analyze natural parity study: compute bootstrap statistics, AUCs, and generate figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(np.mean((pos[:, None] > neg) + 0.5 * (pos[:, None] == neg)))


def bootstrap_interval(values, rng, n_boot=4000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return dict(mean=0.0, low=0.0, high=0.0)
    boot_means = values[rng.integers(len(values), size=(n_boot, len(values)))].mean(axis=1)
    return dict(
        mean=float(values.mean()),
        low=float(np.quantile(boot_means, 0.025)),
        high=float(np.quantile(boot_means, 0.975)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/natural_parity/results.json")
    parser.add_argument("--output", default="docs/evidence/natural_parity")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text())
    protocol = data["protocol"]
    rows = data["rows"]

    rng = np.random.default_rng(70731)

    settings = sorted({r["setting"] for r in rows})
    widths = protocol["widths"]

    analysis = dict(protocol=protocol, summaries={})

    for setting in settings:
        setting_rows = [r for r in rows if r["setting"] == setting]
        setting_summary = {}

        for width in widths:
            w_rows = [r for r in setting_rows if r["width"] == width]
            if not w_rows:
                continue

            n_seeds = len(w_rows)
            n_factors = len(w_rows[0]["q"])

            # Extract arrays
            beta0 = np.array([r["beta0"] for r in w_rows])          # (seeds, factors)
            actual_p1 = np.array([r["actual_p1"] for r in w_rows])  # (seeds, factors)
            pred_det = np.array([r["pred_det"] for r in w_rows])    # (seeds, factors)
            pred_ora = np.array([r["pred_ora"] for r in w_rows])    # (seeds, factors)
            losses_end = np.array([r["loss_end"] for r in w_rows])  # (seeds,)
            accs_end = np.array([r["acc_end"] for r in w_rows])      # (seeds,)
            coherences = np.array([r["u_coherence"] for r in w_rows])
            cosines = np.array([r["u_cosine_to_oracle"] for r in w_rows])

            # Events
            event_half = actual_p1 < 0.5 * beta0
            pred_det_half = pred_det < 0.5 * beta0
            pred_ora_half = pred_ora < 0.5 * beta0

            event_sev = actual_p1 < 0.7 * beta0
            pred_det_sev = pred_det < 0.7 * beta0

            # AUCs across all factor probes
            ratio_det = pred_det / np.maximum(beta0, 1e-12)
            ratio_ora = pred_ora / np.maximum(beta0, 1e-12)

            auc_det = auc(event_half.flatten(), (1 - ratio_det).flatten())
            auc_ora = auc(event_half.flatten(), (1 - ratio_ora).flatten())
            auc_base = auc(event_half.flatten(), (-beta0).flatten())

            # Seed rates
            seed_event_rates = event_half.mean(axis=1)
            seed_pred_det_rates = pred_det_half.mean(axis=1)

            setting_summary[str(width)] = dict(
                seeds=n_seeds,
                factors=n_factors,
                coherence=bootstrap_interval(coherences, rng),
                cosine_to_oracle=bootstrap_interval(cosines, rng),
                loss_end=bootstrap_interval(losses_end, rng),
                acc_end=bootstrap_interval(accs_end, rng),
                events_observed=int(event_half.sum()),
                events_pred_det=int(pred_det_half.sum()),
                events_pred_ora=int(pred_ora_half.sum()),
                total_probes=int(event_half.size),
                observed_rate=bootstrap_interval(seed_event_rates, rng),
                predicted_rate_det=bootstrap_interval(seed_pred_det_rates, rng),
                auc_detected=auc_det,
                auc_oracle=auc_ora,
                auc_baseline=auc_base,
                weakened_70_observed=int(event_sev.sum()),
                weakened_70_pred=int(pred_det_sev.sum()),
            )

        analysis["summaries"][setting] = setting_summary

    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    # Generate Figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Episode Coherence and Alignment vs Width (primary_d4)
    ax1 = axes[0, 0]
    p_sum = analysis["summaries"].get("primary_d4", {})
    if p_sum:
        ws = [int(w) for w in p_sum.keys()]
        cohs = [p_sum[str(w)]["coherence"]["mean"] for w in ws]
        coss = [p_sum[str(w)]["cosine_to_oracle"]["mean"] for w in ws]
        ax1.plot(ws, cohs, "o-", color="#1f77b4", label=r"Coherence $\rho = \sigma_1^2 / \sum \sigma_i^2$")
        ax1.plot(ws, coss, "s--", color="#2ca02c", label=r"Cosine $\cos(\hat{u}, u_{\rm oracle})$")
        ax1.set_xlabel("Network Width ($m$)")
        ax1.set_ylabel("Metric")
        ax1.set_title("A. Label-Free Episode SVD Alignment")
        ax1.set_ylim(0.5, 1.05)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

    # 2. Response Weakening Rate (< 0.5 beta0) vs Width
    ax2 = axes[0, 1]
    if p_sum:
        obs_rates = [p_sum[str(w)]["observed_rate"]["mean"] * 100 for w in ws]
        obs_err_low = [obs_rates[i] - p_sum[str(w)]["observed_rate"]["low"] * 100 for i, w in enumerate(ws)]
        obs_err_high = [p_sum[str(w)]["observed_rate"]["high"] * 100 - obs_rates[i] for i, w in enumerate(ws)]
        pred_rates = [p_sum[str(w)]["predicted_rate_det"]["mean"] * 100 for w in ws]

        ax2.errorbar(ws, obs_rates, yerr=[obs_err_low, obs_err_high], fmt="o-", color="#d62728",
                     capsize=4, label="Observed Half-Response Loss")
        ax2.plot(ws, pred_rates, "s--", color="#ff7f0e", label="Online SVD Predicted")
        ax2.set_xlabel("Network Width ($m$)")
        ax2.set_ylabel("Weakening Event Rate (%)")
        ax2.set_title("B. Weakening Probability vs Width")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

    # 3. AUC Comparison
    ax3 = axes[1, 0]
    if p_sum:
        auc_dets = [p_sum[str(w)]["auc_detected"] or 0.5 for w in ws]
        auc_oras = [p_sum[str(w)]["auc_oracle"] or 0.5 for w in ws]
        auc_bases = [p_sum[str(w)]["auc_baseline"] or 0.5 for w in ws]

        x_indices = np.arange(len(ws))
        bar_w = 0.25
        ax3.bar(x_indices - bar_w, auc_dets, width=bar_w, label="Online SVD AUC", color="#1f77b4")
        ax3.bar(x_indices, auc_oras, width=bar_w, label="Oracle AUC", color="#2ca02c")
        ax3.bar(x_indices + bar_w, auc_bases, width=bar_w, label="Initial Capability AUC", color="#7f7f7f")
        ax3.set_xticks(x_indices)
        ax3.set_xticklabels([str(w) for w in ws])
        ax3.set_xlabel("Network Width ($m$)")
        ax3.set_ylabel("AUC")
        ax3.set_title("C. Option Vulnerability Ranking AUC")
        ax3.set_ylim(0.4, 1.05)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend()

    # 4. Final Loss vs Width Across Settings
    ax4 = axes[1, 1]
    color_map = {
        "primary_d4": "#1f77b4",
        "dimension_d6": "#9467bd",
        "flatter_alpha05": "#ff7f0e",
        "uncentered_control": "#d62728",
        "zero_label_control": "#8c564b",
    }
    label_map = {
        "primary_d4": "Primary (4-bit Bipolar)",
        "dimension_d6": "Scaled Dimension (6-bit)",
        "flatter_alpha05": "Flatter Profile (alpha=0.5)",
        "uncentered_control": "Uncentered Control (no opp.)",
        "zero_label_control": "Zero Label Control (no res.)",
    }
    for setting, s_dict in analysis["summaries"].items():
        s_ws = [int(w) for w in s_dict.keys()]
        losses = [s_dict[str(w)]["loss_end"]["mean"] for w in s_ws]
        ax4.plot(s_ws, losses, "o-", color=color_map.get(setting, "black"), label=label_map.get(setting, setting))

    ax4.set_xlabel("Network Width ($m$)")
    ax4.set_ylabel("Final Population Loss")
    ax4.set_title("D. Final Loss vs Width Across Settings")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "natural_parity_study.png", dpi=300)
    plt.savefig(out_dir / "natural_parity_study.svg")
    figures_dir = Path(__file__).resolve().parents[2] / "figures"
    if figures_dir.is_dir():
        plt.savefig(figures_dir / "natural_parity_study.png", dpi=300)
        plt.savefig(figures_dir / "natural_parity_study.svg")
    plt.close()

    print(f"Analysis complete. Figures and summary saved to {out_dir} and {figures_dir}")


if __name__ == "__main__":
    main()
