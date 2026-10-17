#!/usr/bin/env python3
"""Verify all-pair formulas against direct independent neuron-subset forwards."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from audit_neurons import assign

from neural_competition.analysis import load_checkpoint
from neural_competition.model import ShallowReLU, loss
from neural_competition.neurons import candidate_descent
from neural_competition.pairs import (
    pair_loss_effects,
    pair_statistics,
    source_neuron_changes,
    summarize_batches,
)


def rounding_check(output, selected, data):
    """Repeat early full steps after constructing parameter updates in float32."""
    indices = data["probe_indices"]
    x, y = data["x_train"][indices], data["y_train"][indices].double()
    rows = []
    for row in selected[selected.comparison == "mse_0.2"].itertuples():
        snapshot = load_checkpoint(row.run_path, row.step)
        gradients = snapshot["probe_gradients"].clone()
        if row.optimizer == "momentum_01":
            previous = snapshot["optimizer_before"]["momentum_buffer"].clone()
            gradients = previous.mul_(snapshot["optimizer"]["momentum"])[None] + gradients
        before = snapshot["parameters_before"]
        after = torch.add(before[None], gradients, alpha=-snapshot["optimizer"]["lr"])
        prediction, changes = source_neuron_changes(
            before, after.double() - before.double()[None], x
        )
        alone, interaction = pair_loss_effects(prediction - y, changes)
        values, _ = pair_statistics(alone, interaction, 0.5 * (prediction - y).square().mean(1))
        rows.append(
            {
                "optimizer": row.optimizer,
                "width": row.width,
                "seed": row.seed,
                **summarize_batches(values),
            }
        )
    rounded = pd.DataFrame(rows)
    rounded.to_csv(output / "rounding_sensitivity.csv", index=False)
    original = pd.read_csv(output / "pair_seeds.csv")
    original = original[
        (original.comparison == "mse_0.2")
        & (original.scale == 1)
        & (original.probe_set == "original128")
    ]
    joined = rounded.merge(
        original, on=["optimizer", "width", "seed"], suffixes=("_rounded", "_main")
    )
    return {
        "max_early_rounding_frequency_difference": (
            joined.frequency_rounded - joined.frequency_main
        )
        .abs()
        .max(),
        "max_early_rounding_severity_difference": (joined.severity_rounded - joined.severity_main)
        .abs()
        .max(),
    }


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    output = root / "runs/pier_study/pairs"
    manifest = json.loads((output / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    data = torch.load(root / "runs/pier_distributions/data.pt", weights_only=True)
    selected = pd.read_csv(output / "selected_states.csv")
    # Prespecified seed; cover both loss stages and all three optimizer profiles.
    samples = selected[(selected.seed == 0) & (selected.comparison_kind == "matched_loss")]
    batch_frame = pd.read_csv(output / "pair_batches.csv")
    max_error = 0.0
    direct_pairs = 0
    direct_events = 0
    for row in samples.itertuples():
        snapshot = load_checkpoint(row.run_path, row.step)
        before = snapshot["parameters_before"].double()
        model = ShallowReLU(20, row.width).double()
        # Include a real positive event when available, strictly for verification.
        observed = batch_frame[
            (batch_frame.optimizer == row.optimizer)
            & (batch_frame.comparison == row.comparison)
            & (batch_frame.width == row.width)
            & (batch_frame.seed == 0)
            & (batch_frame.scale == 1)
            & (batch_frame.probe_set == "original128")
            & (batch_frame.event_count > 0)
        ]
        positive_batch = int(observed.source_batch.min()) if len(observed) else 1
        source_ids = [0, positive_batch] if positive_batch != 0 else [0, 1]
        indices = data["probe_indices"][source_ids]
        x, y = data["x_train"][indices].double(), data["y_train"][indices].double()
        for scale in [0.01, 1.0]:
            delta = (
                -scale
                * snapshot["optimizer"]["lr"]
                * candidate_descent(snapshot, snapshot["probe_gradients"][source_ids])
            )
            prediction, changes = source_neuron_changes(before, delta, x)
            standalone, interaction = pair_loss_effects(prediction - y, changes)
            baseline = 0.5 * (prediction - y).square().mean(1)
            stats, _ = pair_statistics(standalone, interaction, baseline)
            for b in range(2):
                assign(model, before)
                base_loss = loss(model, x[b], y[b]).item()
                solo = []
                for r in range(row.width):
                    isolated = before.clone()
                    isolated[r] += delta[b, r]
                    assign(model, isolated)
                    value = loss(model, x[b], y[b]).item() - base_loss
                    solo.append(value)
                    max_error = max(max_error, abs(value - standalone[b, r].item()))
                count = 0
                for r in range(row.width):
                    for s in range(r):
                        combined = before.clone()
                        combined[r] += delta[b, r]
                        combined[s] += delta[b, s]
                        assign(model, combined)
                        joint = loss(model, x[b], y[b]).item() - base_loss
                        expected = standalone[b, r] + standalone[b, s] + interaction[b, r, s]
                        max_error = max(max_error, abs(joint - expected.item()))
                        tol = 1e-10 * base_loss
                        if solo[r] < -tol and solo[s] < -tol:
                            count += int(joint - solo[r] > tol) + int(joint - solo[s] > tol)
                        direct_pairs += 1
                assert count == stats["event_count"][b].item()
                direct_events += count
    assert max_error < 1e-12
    # Check raw-file reconstruction and aggregate counts without torch helpers.
    frame = pd.read_csv(output / "pair_seeds.csv")
    numpy_checks = 0
    for row in samples.itertuples():
        for scale in [0.01, 1.0]:
            name = f"{row.collection}_w{row.width}_seed{row.seed}_{row.optimizer}_step{row.step}_original128_scale{scale:g}.npz"
            raw = np.load(output / "raw" / name)
            alone, upper = raw["standalone"], raw["interaction"]
            r, s = raw["r"], raw["s"]
            tol = 1e-10 * raw["baseline_loss"][:, None]
            eligible = (alone[:, r] < -tol) & (alone[:, s] < -tol)
            first = alone[:, s] + upper
            second = alone[:, r] + upper
            events = eligible * ((first > tol).astype(int) + (second > tol).astype(int))
            harm = (
                np.where(eligible & (first > tol), first, 0).sum()
                + np.where(eligible & (second > tol), second, 0).sum()
            )
            saved = frame[
                (frame.optimizer == row.optimizer)
                & (frame.comparison == row.comparison)
                & (frame.seed == 0)
                & (frame.width == row.width)
                & (frame.scale == scale)
                & (frame.probe_set == "original128")
            ].iloc[0]
            assert int(events.sum()) == saved.event_count
            assert int(2 * eligible.sum()) == saved.eligible_count
            assert abs(harm - saved.harm_sum) < 1e-12
            numpy_checks += 1
    audit = {
        "immutable_sources": len(manifest["source_sha256"]),
        "independent_states": len(samples),
        "direct_pair_evaluations": direct_pairs,
        "direct_interference_events": direct_events,
        "max_absolute_loss_error": max_error,
        "independent_numpy_checks": numpy_checks,
    }
    audit.update(rounding_check(output, selected, data))
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(audit)


if __name__ == "__main__":
    main()
