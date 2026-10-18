#!/usr/bin/env python3
"""Sample finite peer-update interventions in the original optimizer cohorts.

Replay baselines in their original batched layout. At prespecified checkpoints,
damp two uniformly sampled sender updates, separately, for one step only. Every
recipient is observed in the same short branch. Original archives are read-only.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_competition.model import ShallowReLU
from neural_competition.trajectory import adam_step, components, momentum_step

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "runs/retained"
CACHE = ROOT / "runs/temporal_branches"
OUTPUT = CACHE / "fixed_fraction"
FIELDS = [
    "loss_before", "loss_after", "gain", "harm", "anti_work", "energy",
    "harmful_count", "recipient_count", "all_gain", "all_harm", "net_pair",
    "quality_harm", "residual_harm", "linear_gain", "signed_alone",
]
LAGS = [1, 4, 16, 32, 128]
LONG_PROBE_IDS = [5, 9, 13, 17, 21]
ARMS = ["ordinary", "sender_damped", "uniform_matched"]
VIEWS = ["reference", "heldout"]
HASHES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, kind="torch"):
    HASHES[str(path.relative_to(ROOT))] = digest(path)
    return (torch.load(path, weights_only=True, map_location="cpu")
            if kind == "torch" else json.loads(path.read_text()))


@torch.jit.script
def advance(p, first, second, xtrain, ytrain, indices, seeds, norm,
            lr: float, momentum: float, is_adam: bool, completed: int):
    for k in range(indices.shape[0]):
        ids = indices[k].index_select(0, seeds)
        x, y = xtrain[ids], ytrain[ids]
        if is_adam:
            p, first, second = adam_step(p, first, second, x, y, norm, lr, completed + k + 1)
        else:
            p, first = momentum_step(p, first, x, y, norm, lr, momentum)
    return p, first, second


@torch.jit.script
def receiver_statistics(before, after, y, baseline_before, baseline_after, mask):
    """Actual finite-update quantities; baseline is the ordinary branch.

    The symmetric prediction-space split separates changed residual from changed
    neuron prediction movements. Only at lag 1 do untouched recipient parameters
    coincide across branches; later this split is descriptive, not a parameter
    transplant into a common model.
    """
    d = after - before
    d0 = baseline_after - baseline_before
    e = before.sum(-1) - y
    e0 = baseline_before.sum(-1) - y
    a = (e[:, :, None] * d + 0.5 * d.square()).mean(1)
    a0 = (e0[:, :, None] * d0 + 0.5 * d0.square()).mean(1)
    cross_d = (e0[:, :, None] * d + 0.5 * d.square()).mean(1)
    cross_0 = (e[:, :, None] * d0 + 0.5 * d0.square()).mean(1)
    h, h0 = a.clamp_min(0), a0.clamp_min(0)
    hd, hc = cross_d.clamp_min(0), cross_0.clamp_min(0)
    quality = 0.5 * (hd - h0 + h - hc)
    residual = 0.5 * (h - hd + hc - h0)
    linear = -(e[:, :, None] * d).mean(1)
    energy = 0.5 * d.square().mean(1)
    loss = 0.5 * e.square().mean(1)
    end_loss = 0.5 * (after.sum(-1) - y).square().mean(1)
    # Padded neurons have exactly zero changes, so all-neuron totals are safe.
    stats = torch.stack([
        loss, end_loss, ((-a).clamp_min(0) * mask).sum(1),
        (h * mask).sum(1), ((-linear).clamp_min(0) * mask).sum(1),
        (energy * mask).sum(1),
        (((a > 1e-10 * loss[:, None]).double()) * mask).sum(1), mask.sum(1),
        (-a).clamp_min(0).sum(1), h.sum(1),
        0.5 * d.sum(-1).square().mean(1) - energy.sum(1),
        (quality * mask).sum(1), (residual * mask).sum(1),
        (linear.clamp_min(0) * mask).sum(1), (a * mask).sum(1),
    ], dim=-1)
    return stats, a


def make_interventions(p, following, senders, widths, xref, dose, match_seeds=None):
    """One sender damping and one prediction-norm-matched uniform control each."""
    count, n, maximum = len(senders), len(p), p.shape[1]
    norm = widths.double().sqrt()
    normal = components(following.double(), xref, norm).sum(-1)
    delta = following - p
    variants = [following]
    info = []
    for j in range(count):
        sender = senders[j]
        local = following.clone()
        rows = torch.arange(n)
        local[rows, sender] = p[rows, sender] + (1.0 - dose) * delta[rows, sender]
        moved = components(local.double(), xref, norm).sum(-1) - normal
        target = moved.square().mean(1).sqrt()
        requested = target.clone()
        local_fraction = torch.full((n,), dose)
        if match_seeds is not None:
            # The same sampled sender-rank, training seed and checkpoint get
            # the same absolute prediction perturbation at every width. The
            # smallest feasible <=20% pulse sets the target; a zero target
            # produces a no-op at ALL widths, rather than selecting active units.
            for seed in match_seeds.unique():
                selected = match_seeds == seed
                requested[selected] = target[selected].min()
            left, right = torch.zeros(n), torch.full((n,), dose)
            for _ in range(22):
                fraction = (left + right) / 2
                candidate = following.clone()
                candidate[rows, sender] = p[rows, sender] + (1.0 - fraction[:, None])*delta[rows, sender]
                distance = (components(candidate.double(), xref, norm).sum(-1) - normal)
                distance = distance.square().mean(1).sqrt()
                below = distance < requested
                left = torch.where(below, fraction, left)
                right = torch.where(below, right, fraction)
            candidates, distances = [], []
            for fraction in [left, right]:
                candidate = following.clone()
                candidate[rows, sender] = p[rows, sender] + (1.0 - fraction[:, None])*delta[rows, sender]
                candidate[requested == 0] = following[requested == 0]
                distance = (components(candidate.double(), xref, norm).sum(-1) - normal)
                candidates.append(candidate)
                distances.append(distance.square().mean(1).sqrt())
            choose_right = (distances[1] - requested).abs() < (distances[0] - requested).abs()
            local = torch.where(choose_right[:, None, None], candidates[1], candidates[0])
            local_fraction = torch.where(choose_right, right, left)
            local_fraction[requested == 0] = 0
            target = torch.where(choose_right, distances[1], distances[0])
        total = (components(p.double(), xref, norm).sum(-1) - normal).square().mean(1).sqrt()
        lo, hi = torch.zeros(n), torch.ones(n)
        # Close to the ordinary step this is usually monotone. Achieved mismatch
        # is exported, including quantization/dead-neuron exceptions, not hidden.
        for _ in range(22):
            fraction = (lo + hi) / 2
            candidate = following - fraction[:, None, None] * delta
            distance = (components(candidate.double(), xref, norm).sum(-1) - normal)
            distance = distance.square().mean(1).sqrt()
            below = distance < target
            lo = torch.where(below, fraction, lo)
            hi = torch.where(below, hi, fraction)
        candidates = [following - f[:, None, None] * delta for f in [lo, hi]]
        distances = [
            (components(q.double(), xref, norm).sum(-1) - normal).square().mean(1).sqrt()
            for q in candidates
        ]
        choose_hi = (distances[1] - target).abs() < (distances[0] - target).abs()
        fraction = torch.where(choose_hi, hi, lo)
        fraction = torch.where(target == 0, torch.zeros_like(fraction), fraction)
        uniform = following - fraction[:, None, None] * delta
        uniform[target == 0] = following[target == 0]
        achieved = (components(uniform.double(), xref, norm).sum(-1) - normal)
        achieved = achieved.square().mean(1).sqrt()
        variants.extend([local, uniform])
        for i in range(n):
            info.append(dict(
                probe_sender=j, model=i, sender=int(sender[i]), dose=dose,
                removed_prediction_rms=float(target[i]),
                ordinary_prediction_rms=float(total[i]),
                uniform_fraction=float(fraction[i]),
                uniform_removed_rms=float(achieved[i]),
                local_fraction=float(local_fraction[i]),
                requested_common_rms=float(requested[i]),
                relative_cross_width_match_error=(
                    float(abs(target[i] - requested[i]) / requested[i]) if requested[i] > 0 else 0.0
                ),
                relative_match_error=(float(abs(achieved[i] - target[i]) / target[i])
                                      if target[i] > 0 else 0.0),
            ))
        for r in range(n):
            unchanged = torch.arange(maximum) != sender[r]
            assert torch.equal(local[r, unchanged], following[r, unchanged])
    return torch.cat(variants), info


@torch.jit.script
def branch_rollout(p, first, second, xtrain, ytrain, indices, seeds, widths,
                   xeval, yeval, senders, nref: int, lr: float, momentum: float,
                   is_adam: bool, completed: int):
    n = widths.shape[0]
    ns = senders.shape[0]
    variants = 1 + 2 * ns
    maximum = p.shape[1]
    norm64 = widths.double().sqrt()
    norm32 = norm64.float()
    active = (torch.arange(maximum)[None] < widths).double()
    records = torch.empty((indices.shape[0], ns, 3, n, 2, 15), dtype=torch.float64)
    # Retain individual recipient measurements at a few lags, as well as the
    # every-step aggregate. No all-pairs matrix of branch trajectories is needed.
    sampled_alone = torch.zeros((3, ns, 3, n, 2, maximum), dtype=torch.float64)
    for k in range(indices.shape[0]):
        ids = indices[k].index_select(0, seeds).repeat(variants, 1)
        x, y = xtrain[ids], ytrain[ids]
        if is_adam:
            following, first, second = adam_step(
                p, first, second, x, y, norm32.repeat(variants, 1), lr, completed + k + 1
            )
        else:
            following, first = momentum_step(
                p, first, x, y, norm32.repeat(variants, 1), lr, momentum
            )
        before = components(p.double(), xeval.repeat(variants, 1, 1), norm64.repeat(variants, 1))
        after = components(following.double(), xeval.repeat(variants, 1, 1), norm64.repeat(variants, 1))
        for s in range(ns):
            mask = active.clone()
            mask[torch.arange(n), senders[s]] = 0
            for arm in range(3):
                a = 0 if arm == 0 else 1 + 2 * s + arm - 1
                for view in range(2):
                    left = 0 if view == 0 else nref
                    right = nref if view == 0 else xeval.shape[1]
                    stats, alone = receiver_statistics(
                        before[a*n:(a+1)*n, left:right],
                        after[a*n:(a+1)*n, left:right], yeval[:, left:right],
                        before[:n, left:right], after[:n, left:right], mask,
                    )
                    records[k, s, arm, :, view] = stats
                    if k == 0 or k == 31 or k == 127:
                        slot = 0 if k == 0 else (1 if k == 31 else 2)
                        sampled_alone[slot, s, arm, :, view] = alone
        p = following
    return p[:n], first[:n], second[:n], records, sampled_alone


def conditions():
    groups = sorted(
        str(p.parent.relative_to(ARCHIVE / "provenance/training/robustness"))
        for p in (ARCHIVE / "provenance/training/robustness").glob("*/*/config.json")
    )
    return ["trajectory/sgd", "trajectory/momentum", *["robustness/" + g for g in groups]]


def initialize(condition):
    family, label = condition.split("/", 1)
    config = read(ARCHIVE / "provenance/training" / condition / "config.json", "json")
    dataset = "k8_teacher2027" if family == "trajectory" else label.split("/")[0].split("_init")[0]
    data = read(ARCHIVE / "datasets" / f"{dataset}.pt")
    saved = read(ARCHIVE / "checkpoints" / condition / "initial.pt")
    models = config["models"]
    maximum = max(m["width"] for m in models)
    p = torch.zeros(len(models), maximum, 22)
    scale = config.get("student_output_scale", 0.0)
    for i, model in enumerate(models):
        net = ShallowReLU(20, model["width"], model["seed"], zero_output=scale == 0)
        initial = net.parameter_blocks()
        if scale:
            initial[:, -1].mul_(scale)
        p[i, : model["width"]] = initial
    assert torch.equal(p[saved["original_model_indices"]], saved["parameters"])
    return config, data, saved, p


def probe_times(steps):
    return np.unique(np.rint(np.r_[0, np.geomspace(2, steps - 257, 24)]).astype(int))


def future_indices(indices, offset, generators, count, ntrain):
    available = indices[offset:offset + count]
    if len(available) == count:
        return available
    cloned = [torch.Generator().set_state(g.get_state()) for g in generators]
    following = torch.stack([
        torch.randint(ntrain, (1000, 64), generator=g) for g in cloned
    ], dim=1)
    return torch.cat([available, following[:count - len(available)]])


def audit_checkpoint(condition, completed, p, first, second, generators, saved):
    family = condition.split("/")[0]
    if completed != (200_000 if family == "trajectory" else 100_000):
        if family == "trajectory" or completed not in [10_000, 50_000]:
            return None
        name = f"step_{completed:06d}"
    else:
        name = "final"
    checkpoint = read(ARCHIVE / "checkpoints" / condition / f"{name}.pt")
    selected = saved["original_model_indices"]
    for key, value in [("parameters", p), ("velocity", first), ("second", second)]:
        if key in checkpoint:
            assert torch.equal(value[selected], checkpoint[key]), (condition, completed, key)
    for seed, rng in zip(checkpoint["rng_seed_order"], checkpoint["rng_states"]):
        assert torch.equal(generators[seed].get_state(), rng), (condition, completed, seed)
    return dict(completed=completed, parameters_moments_rng_bitwise_equal=True)


def run(condition, dose=0.2, limit=None, match_width_dose=False):
    global HASHES
    HASHES = {}
    started = time.perf_counter()
    config, data, saved, p = initialize(condition)
    n = len(p)
    widths = torch.tensor([m["width"] for m in config["models"]])[:, None]
    seeds = torch.tensor([m["seed"] for m in config["models"]])
    norm = widths.double().sqrt().float()
    generators = [torch.Generator().manual_seed(12026 + s) for s in range(int(seeds.max()) + 1)]
    first, second = torch.zeros_like(p), torch.zeros_like(p)
    ref = saved["reference_indices"]
    test = torch.randperm(len(data["x_test"]), generator=torch.Generator().manual_seed(20261018))[:256]
    xeval = torch.cat([data["x_train"][ref], data["x_test"][test]]).double()[None].expand(n, -1, -1)
    yeval = torch.cat([data["y_train"][ref], data["y_test"][test]]).double()[None].expand(n, -1)
    is_adam = config.get("optimizer_spec", {}).get("name") == "adam"
    total = config["steps"]
    times = probe_times(total)
    if limit:
        times = times[:limit]
    rows, neuron_rows, dose_rows, audits = [], [], [], []
    branch_steps = 0
    branch_seconds = 0.0
    max_budget_error = 0.0
    for block in range(0, total, 1000):
        indices = torch.stack([
            torch.randint(len(data["x_train"]), (min(1000, total - block), 64), generator=g)
            for g in generators
        ], dim=1)
        position = 0
        for probe_id in np.flatnonzero((times >= block) & (times < block + len(indices))):
            t = int(times[probe_id])
            offset = t - block
            p, first, second = advance(
                p, first, second, data["x_train"], data["y_train"], indices[position:offset],
                seeds, norm, config["lr"], config["momentum"], is_adam, block + position,
            )
            position = offset
            horizon = 128 if probe_id in LONG_PROBE_IDS else 32
            future = future_indices(indices, offset, generators, horizon + 1, len(data["x_train"]))
            began = time.perf_counter()
            following, f1, v1 = advance(
                p, first, second, data["x_train"], data["y_train"], future[:1], seeds, norm,
                config["lr"], config["momentum"], is_adam, t,
            )
            rng = np.random.default_rng(7141 + 1009 * probe_id)
            senders = torch.tensor(np.array([
                rng.choice(int(w), size=2, replace=False) for w in widths[:, 0]
            ]).T.copy())
            variants, doses = make_interventions(
                p, following, senders, widths, xeval[:, :len(ref)], dose,
                seeds if match_width_dose else None,
            )
            nv = 1 + 2 * len(senders)
            fp, ff, fv, records, alone = branch_rollout(
                variants, f1.repeat(nv, 1, 1), v1.repeat(nv, 1, 1),
                data["x_train"], data["y_train"], future[1:], seeds, widths, xeval, yeval,
                senders, len(ref), config["lr"], config["momentum"], is_adam, t + 1,
            )
            # A normal branch inside the enlarged batch must reproduce the same
            # trajectory as the original-size batch, including rounded updates.
            expected = advance(
                following, f1, v1, data["x_train"], data["y_train"], future[1:], seeds,
                norm, config["lr"], config["momentum"], is_adam, t + 1,
            )
            assert all(torch.equal(a, b) for a, b in zip((fp, ff, fv), expected)), (condition, t)
            rec = records.numpy()
            assert np.isfinite(rec).all()
            # Every arm's whole-network budget is exact at every measured update.
            err = np.max(np.abs(rec[..., 1] - rec[..., 0] + rec[..., 8] - rec[..., 9] - rec[..., 10]))
            assert err < 1e-11, (condition, t, err)
            max_budget_error = max(max_budget_error, float(err))
            for arm in [1, 2]:
                assert np.max(np.abs(
                    rec[:, :, arm, :, :, 3] - rec[:, :, 0, :, :, 3]
                    - rec[:, :, arm, :, :, 11] - rec[:, :, arm, :, :, 12]
                )) < 1e-11
            cumulative = rec.cumsum(axis=0)
            origin = components(variants.double(), xeval.repeat(nv, 1, 1),
                                widths.double().sqrt().repeat(nv, 1))
            original_before = components(p.double(), xeval, widths.double().sqrt())
            original_change = origin[:n] - original_before
            for info in doses:
                i, s = info["model"], info["probe_sender"]
                metadata = dict(condition=condition, **config["models"][i], probe_id=int(probe_id),
                                completed=t, sender=info["sender"], probe_sender=s)
                dose_rows.append({**metadata, **info})
                for view, v in enumerate(VIEWS):
                    left, right = (0, len(ref)) if view == 0 else (len(ref), len(xeval[0]))
                    ordinary_loss = 0.5 * (origin[i, left:right].sum(-1) - yeval[i, left:right]).square().mean()
                    for arm, label in enumerate(ARMS):
                        index = i if arm == 0 else (1 + 2*s + arm - 1)*n + i
                        start_loss = 0.5 * (origin[index, left:right].sum(-1) - yeval[i, left:right]).square().mean()
                        for lag in [k for k in LAGS if k <= horizon]:
                            values, sums = rec[lag - 1, s, arm, i, view], cumulative[lag - 1, s, arm, i, view]
                            row = dict(**metadata, view=v, arm=label, lag=lag,
                                       origin_loss=float(start_loss), ordinary_origin_loss=float(ordinary_loss),
                                       removed_prediction_rms=info["removed_prediction_rms"],
                                       relative_match_error=info["relative_match_error"],
                                       relative_cross_width_match_error=info["relative_cross_width_match_error"],
                                       **dict(zip(FIELDS, values)))
                            row.update({"cumulative_" + f: sums[k] for k, f in enumerate(FIELDS) if k >= 2})
                            rows.append(row)
                    e0 = original_before[i, left:right].sum(-1) - yeval[i, left:right]
                    d0 = original_change[i, left:right]
                    initial_a = (e0[:, None]*d0 + 0.5*d0.square()).mean(0)
                    initial_j = (d0[:, info["sender"], None]*d0).mean(0)
                    for recipient in range(config["models"][i]["width"]):
                        if recipient == info["sender"]:
                            continue
                        for slot, lag in enumerate([1, 32, 128]):
                            if lag > horizon:
                                continue
                            neuron_rows.append(dict(
                                **metadata, view=v, recipient=recipient, lag=lag,
                                initial_recipient_alone=float(initial_a[recipient]),
                                initial_sender_alone=float(initial_a[info["sender"]]),
                                initial_pair_cost=float(initial_j[recipient]),
                                ordinary_alone=float(alone[slot, s, 0, i, view, recipient]),
                                damped_alone=float(alone[slot, s, 1, i, view, recipient]),
                                uniform_alone=float(alone[slot, s, 2, i, view, recipient]),
                            ))
            branch_steps += (nv - 1) * horizon
            branch_seconds += time.perf_counter() - began
            if probe_id % 5 == 0:
                print(f"{condition}: probe {probe_id+1}/{len(times)}, t={t}, horizon={horizon}; "
                      f"{time.perf_counter()-started:.1f}s", flush=True)
        p, first, second = advance(
            p, first, second, data["x_train"], data["y_train"], indices[position:], seeds, norm,
            config["lr"], config["momentum"], is_adam, block + position,
        )
        check = audit_checkpoint(condition, block + len(indices), p, first, second, generators, saved)
        if check:
            audits.append(check)
            print(f"{condition}: original {check['completed']} checkpoint matches bitwise", flush=True)
        if limit and block + len(indices) > times[-1]:
            break
    endpoint_error = None
    if not limit:
        family, label = condition.split("/", 1)
        source = ("robustness_per_seed_results.csv" if family == "robustness"
                  else "cumulative_interference_seeds.csv")
        path = ROOT / "docs/tables" / source
        HASHES[str(path.relative_to(ROOT))] = digest(path)
        table = pd.read_csv(path)
        table = table[table.view == "reference"]
        table = table[table.condition == label] if family == "robustness" else table[table.optimizer == label]
        final_loss = 0.5*(components(p.double(), xeval[:, :len(ref)], widths.double().sqrt()).sum(-1)
                          - yeval[:, :len(ref)]).square().mean(1)
        values = table.set_index(["width", "seed"]).final_loss
        expected = np.array([values.loc[m["width"], m["seed"]] for m in config["models"]])
        endpoint_error = float(np.max(np.abs(final_loss.numpy() - expected)))
        assert endpoint_error < 1e-10, (condition, endpoint_error)
    stem = condition.replace("/", "_")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / f"{stem}.csv.gz", index=False, float_format="%.17g")
    pd.DataFrame(neuron_rows).to_csv(OUTPUT / f"{stem}_neurons.csv.gz", index=False, float_format="%.17g")
    pd.DataFrame(dose_rows).to_csv(OUTPUT / f"{stem}_doses.csv", index=False, float_format="%.17g")
    provenance = dict(
        condition=condition, original_models=config["models"], independent_new_runs=False,
        counterfactual_training_performed=True, dose=dose, checkpoints=times.tolist(),
        match_width_dose=match_width_dose,
        long_probe_ids=LONG_PROBE_IDS, lags=LAGS, fields=FIELDS,
        sender_selection="two neurons uniformly without replacement per model/checkpoint; RNG 7141+1009*probe_id",
        intervention="one actual rounded parameter update damped; all just-computed moments retained; later updates normal",
        control="uniform parameter-step reduction calibrated to removed prediction RMS on original fixed reference",
        views=dict(reference_size=len(ref), heldout_size=len(test), heldout_seed=20261018),
        original_checkpoint_audits=audits, normal_branch_layout_audits=len(times),
        branch_update_groups=branch_steps, baseline_update_groups=total,
        branch_seconds=branch_seconds, elapsed_seconds=time.perf_counter() - started,
        max_budget_error=max_budget_error, all_seed_endpoint_error=endpoint_error,
        input_sha256=HASHES, producer_sha256=digest(Path(__file__)),
        engine_sha256=digest(ROOT / "src/neural_competition/trajectory.py"),
        torch_version=str(torch.__version__), threads=torch.get_num_threads(), pilot_limit=limit,
    )
    (OUTPUT / f"{stem}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"DONE {condition}: {len(rows)} rows; {time.perf_counter()-started:.1f}s", flush=True)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=conditions(), default=conditions())
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dose", type=float, default=0.2)
    parser.add_argument("--limit-probes", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--match-width-dose", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    OUTPUT = CACHE / ("matched_dose" if args.match_width_dose else "fixed_fraction")
    (CACHE / "executed_sources").mkdir(parents=True, exist_ok=True)
    frozen = CACHE / "executed_sources" / (digest(Path(__file__)) + ".py")
    frozen.write_bytes(Path(__file__).read_bytes())
    if not 0 < args.dose <= 1:
        raise ValueError("Dose must be in (0, 1]")
    for condition in args.conditions:
        path = OUTPUT / (condition.replace("/", "_") + ".json")
        if path.exists() and not args.refresh:
            old = json.loads(path.read_text())
            if (old["dose"] == args.dose and old["pilot_limit"] == args.limit_probes
                    and old.get("match_width_dose", False) == args.match_width_dose):
                print(f"SKIP {condition}: existing completed result", flush=True)
                continue
        run(condition, args.dose, args.limit_probes, args.match_width_dose)


if __name__ == "__main__":
    main()
