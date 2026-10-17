#!/usr/bin/env python3
"""Verify the curated archive without treating selected cases as a new cohort."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from neural_competition.model import ShallowReLU, loss
from neural_competition.neurons import candidate_descent, step_transfer_masses
from neural_competition.trajectory import components


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=root / "runs/retained")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    manifest = json.loads((archive / "manifest.json").read_text())
    torch.set_num_threads(4)
    for item in manifest["files"]:
        path = (archive / item["path"]).resolve()
        assert path.is_relative_to(archive) and path.is_file(), item["path"]
        assert path.stat().st_size == item["bytes"], item["path"]
        assert sha256(path) == item["sha256"], item["path"]

    maximum_accounting, maximum_checkpoint = 0.0, 0.0
    model_updates = 0
    for meta in manifest["update_archives"]:
        with np.load(archive / meta["path"]) as saved:
            records = saved["records"]
        assert list(records.shape) == meta["shape"] and np.isfinite(records).all()
        fields = {name: i for i, name in enumerate(meta["fields"])}
        terms = sum(
            sign * records[..., fields[name]]
            for sign, name in [(1, "solo_harm"), (-1, "solo_gain"), (1, "penalty"), (-1, "credit")]
        )
        observed = records[..., fields["loss_after"]] - records[..., fields["loss_before"]]
        local_error = np.max(np.abs(terms - observed))
        reference = meta["views"].index("reference")
        total_error = np.max(
            np.abs(
                terms[:, :, reference].sum(0)
                - (
                    records[-1, :, reference, fields["loss_after"]]
                    - records[0, :, reference, fields["loss_before"]]
                )
            )
        )
        maximum_accounting = max(maximum_accounting, float(local_error), float(total_error))
        source = Path(meta["source"])
        if "trajectory" in source.parts:
            checkpoint_folder = archive / "checkpoints/trajectory" / source.parts[-2]
            teacher, ref = "k8_teacher2027", "reference512"
        else:
            teacher, profile = source.parts[-3:-1]
            checkpoint_folder = archive / "checkpoints/robustness" / teacher / profile
            ref = "reference256"
        data = torch.load(archive / "datasets" / f"{teacher}.pt", weights_only=True)
        ids = torch.load(archive / "datasets" / f"{ref}.pt", weights_only=True)["indices"]
        for label, expected in [
            ("initial", records[0, :, reference, fields["loss_before"]]),
            ("final", records[-1, :, reference, fields["loss_after"]]),
        ]:
            state = torch.load(checkpoint_folder / f"{label}.pt", weights_only=True)
            keep = [state["models"].index(m) for m in meta["models"]]
            p = state["parameters"][keep].double()
            widths = torch.tensor([m["width"] for m in meta["models"]])[:, None]
            x = data["x_train"][ids].double()[None].expand(len(p), -1, -1)
            prediction = components(p, x, widths.double().sqrt()).sum(-1)
            actual = 0.5 * (prediction - data["y_train"][ids].double()).square().mean(1)
            maximum_checkpoint = max(
                maximum_checkpoint, float(np.max(np.abs(actual.numpy() - expected)))
            )
        model_updates += records.shape[0] * records.shape[1]
    assert maximum_accounting < 2e-9 and maximum_checkpoint < 1e-10

    # Each retained Adam dot study keeps its raw gradients and frozen moments.
    data = torch.load(archive / "datasets/k8_teacher2027.pt", weights_only=True)
    ids = torch.load(archive / "datasets/adam_cross_batch_probe_indices.pt", weights_only=True)[0]
    cross_count, maximum_gradient, maximum_direction = 0, 0.0, 0.0
    for path in sorted((archive / "cross_batch").glob("*.pt")):
        state = torch.load(path, weights_only=True)
        spec = dict(name="adam", lr=0.001 if state["profile"] == "adam_001" else 0.01)
        snapshot = dict(
            optimizer=spec,
            step=state["updates"],
            optimizer_before={"exp_avg": state["first"], "exp_avg_sq": state["second"]},
        )
        h = candidate_descent(snapshot, state["gradients"])
        maximum_direction = max(maximum_direction, float((h - state["descent"]).abs().max()))
        q = h.flatten(1) @ state["gradients"].flatten(1).T
        torch.testing.assert_close(q, state["adam_dots"], atol=1e-11, rtol=1e-11)
        for name, actual in step_transfer_masses(state["gradients"], h).items():
            assert abs(actual - state[name]) < 1e-11
        model = ShallowReLU(20, state["width"]).double()
        p = state["parameters"]
        with torch.no_grad():
            model.weight.copy_(p[:, :20])
            model.bias.copy_(p[:, 20])
            model.output.copy_(p[:, -1])
        loss(model, data["x_train"][ids].double(), data["y_train"][ids].double()).backward()
        g = model.blocks(tuple(v.grad for v in model.parameters()))
        maximum_gradient = max(maximum_gradient, float((g - state["gradients"][0]).abs().max()))
        cross_count += 1
    assert maximum_gradient < 1e-11 and maximum_direction < 1e-11
    pair_count = 0
    for path in (archive / "pair_interactions").glob("*.npz"):
        with np.load(path) as state:
            a, j = state["standalone"], state["interaction"]
            assert a.ndim == j.ndim == 2 and len(a) == len(j)
            assert j.shape[1] == a.shape[1] * (a.shape[1] - 1) // 2
            assert np.isfinite(a).all() and np.isfinite(j).all()
        pair_count += 1

    result = dict(
        verified_files=len(manifest["files"]),
        representative_model_updates=model_updates,
        update_archives=len(manifest["update_archives"]),
        maximum_accounting_error=maximum_accounting,
        maximum_checkpoint_loss_error=maximum_checkpoint,
        adam_raw_states=cross_count,
        maximum_autograd_error=maximum_gradient,
        maximum_adam_direction_error=maximum_direction,
        pair_array_files=pair_count,
        full_cohort_tables_preserved=manifest["all_cohort_tables_preserved"],
        scope="Curated raw subset; historical full-cohort audits are archived separately",
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
