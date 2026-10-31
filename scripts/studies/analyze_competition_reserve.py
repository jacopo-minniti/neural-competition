"""Summarize state measures; resample seed banks, never individual refinements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auc(event, score):
    event, score = np.asarray(event).ravel(), np.asarray(score).ravel()
    if not event.any() or event.all():
        return None
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    edges = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1, len(score)]
    ranks = np.empty(len(score))
    for start, end in zip(edges[:-1], edges[1:]):
        ranks[order[start:end]] = (start + 1 + end) / 2
    n = int(event.sum())
    return float((ranks[event].sum() - n * (n + 1) / 2) / (n * (len(score) - n)))


def summarize(rows):
    arrays = {key: np.array([r[key] for r in rows]) for key in
              ["current", "actual", "completion", "protected", "diversity", "linear"]}
    current = np.maximum(arrays["current"], 1e-12)
    event = arrays["actual"] < .5 * arrays["current"]
    predicted = arrays["completion"] < .5 * arrays["current"]
    certified = arrays["protected"] >= .5 * arrays["current"]
    q = np.arange(1, current.shape[1] + 1, dtype=float)**-1.15
    q /= q.sum()
    scores = dict(current=-current, participation=-arrays["diversity"],
                  protected=1 - arrays["protected"] / current,
                  completion=1 - arrays["completion"] / current,
                  instantaneous=1 - arrays["linear"] / current)
    return dict(states=len(rows), factors=current.shape[1], loss_fraction=float(event.mean()),
                predicted_loss_fraction=float(predicted.mean()),
                linear_loss_fraction=float((arrays["linear"] < .5 * arrays["current"]).mean()),
                task_weighted_loss=float(np.mean(event @ q)),
                task_weighted_predicted_loss=float(np.mean(predicted @ q)),
                certified_fraction=float(certified.mean()),
                full_training_certificate_violation=float((event & certified).mean()),
                violation_given_certified=float((event & certified).sum() / max(certified.sum(), 1)),
                completion_precision=float((event & predicted).sum() / max(predicted.sum(), 1)),
                completion_recall=float((event & predicted).sum() / max(event.sum(), 1)),
                auc={name: auc(event, score) for name, score in scores.items()},
                mean_other_to_common=float(np.mean([
                    r["diagnostics"]["integrated_other_to_common"] for r in rows])),
                completed=all(r["diagnostics"]["completed"] for r in rows))


def analyze(path):
    result = json.loads(path.read_text())
    rows = result["rows"]
    grouped = []
    for spread in [0., .15]:
        for mass in [.04, .15]:
            for stage in [0, 1]:
                for width in [4, 8, 16]:
                    subset = [r for r in rows if (r["spread"], r["rare_mass"], r["stage"], r["width"])
                              == (spread, mass, stage, width)]
                    grouped.append(dict(spread=spread, rare_mass=mass, stage=stage,
                                        width=width, **summarize(subset)))
    # Within-bank mean over all declared environments/initializations/stages.
    bank_rates = []
    for seed in sorted({r["seed"] for r in rows}):
        bank_rates.append([summarize([r for r in rows if r["seed"] == seed and r["width"] == w])
                           ["loss_fraction"] for w in [4, 8, 16]])
    bank_rates = np.array(bank_rates)
    rng = np.random.default_rng(98723)
    sampled = bank_rates[rng.integers(len(bank_rates), size=(10000, len(bank_rates)))].mean(1)
    gap = sampled[:, 0] - sampled[:, 2]
    stratified = [dict(stage=stage, width=width, **summarize([
        r for r in rows if r["stage"] == stage and r["width"] == width]))
        for stage in [0, 1] for width in [4, 8, 16]]
    return dict(overall=summarize(rows), by_width_stage=stratified, by_width={str(w): summarize([
        r for r in rows if r["width"] == w]) for w in [4, 8, 16]}, by_condition=grouped,
        paired_width4_minus16=dict(estimate=float(np.mean(bank_rates[:, 0] - bank_rates[:, 2])),
                                  seed_bootstrap_95=np.quantile(gap, [.025, .975]).tolist(),
                                  seed_banks=len(bank_rates)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs/evidence/competition_reserve"))
    args = parser.parse_args()
    summary = {name: analyze(args.root / name / "results.json")
               for name in ["confirmation", "transfer32"]}
    coarse = json.loads((args.root / "confirmation/results.json").read_text())["rows"]
    fine = json.loads((args.root / "step_sensitivity/results.json").read_text())["rows"]
    keys = ["spread", "rare_mass", "seed", "initialization", "width", "stage"]
    by_key = {tuple(r[k] for k in keys): r for r in coarse}
    differences, disagreements = [], []
    for r in fine:
        other = by_key[tuple(r[k] for k in keys)]
        differences.extend(np.abs(np.array(r["actual"]) - other["actual"]))
        disagreements.extend((np.array(r["actual"]) < .5*np.array(r["current"])) !=
                             (np.array(other["actual"]) < .5*np.array(other["current"])))
    summary["step_sensitivity"] = dict(states=len(fine), mean_absolute_response_difference=float(
        np.mean(differences)), max_absolute_response_difference=float(np.max(differences)),
        loss_classification_disagreement=float(np.mean(disagreements)))
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in ["confirmation", "transfer32"]:
        print(name, json.dumps(summary[name]["overall"], indent=2))
        print("width gap", summary[name]["paired_width4_minus16"])
    print("step", summary["step_sensitivity"])


if __name__ == "__main__":
    main()
