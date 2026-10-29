"""Additive ten-output MNIST model and finite optimizer-update diagnostics.

The MSE convention sums the ten output coordinates, then averages examples.
Every hidden neuron owns its incoming weights, bias and all ten readout weights.
"""

import gzip
import hashlib
import math
import struct
from pathlib import Path

import numpy as np
import torch

RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def prepare_data(root: Path):
    """Read checksummed official IDX archives; reserve 500/class for validation."""
    destination = root / "prepared.pt"
    if destination.exists():
        return torch.load(destination, weights_only=True, mmap=True)
    raw = {}
    for name, expected in RESOURCES.items():
        payload = (root / "raw" / name).read_bytes()
        assert hashlib.md5(payload).hexdigest() == expected, name
        payload = gzip.decompress(payload)
        if "images" in name:
            magic, count, height, width = struct.unpack(">IIII", payload[:16])
            assert magic == 2051 and (height, width) == (28, 28)
            raw[name] = torch.from_numpy(np.frombuffer(payload[16:], np.uint8).copy())
            raw[name] = raw[name].reshape(count, 784)
        else:
            magic, count = struct.unpack(">II", payload[:8])
            assert magic == 2049
            raw[name] = torch.from_numpy(np.frombuffer(payload[8:], np.uint8).copy()).long()
            assert len(raw[name]) == count
    labels = raw["train-labels-idx1-ubyte.gz"]
    rng = torch.Generator().manual_seed(20261019)
    validation, training = [], []
    for digit in range(10):
        ids = torch.where(labels == digit)[0]
        ids = ids[torch.randperm(len(ids), generator=rng)]
        validation.append(ids[:500])
        training.append(ids[500:])
    train_ids, val_ids = torch.cat(training).sort().values, torch.cat(validation).sort().values
    images = raw["train-images-idx3-ubyte.gz"].float() / 255
    mean, std = images[train_ids].mean(), images[train_ids].std(unbiased=False)
    result = {
        "xtrain": ((images[train_ids] - mean) / std).contiguous(),
        "xval": ((images[val_ids] - mean) / std).contiguous(),
        "xtest": ((raw["t10k-images-idx3-ubyte.gz"].float() / 255 - mean) / std)
        .contiguous(),
        "train_ids": train_ids, "val_ids": val_ids,
        "train_labels": labels[train_ids], "val_labels": labels[val_ids],
        "test_labels": raw["t10k-labels-idx1-ubyte.gz"],
        "mean": float(mean), "std": float(std), "archive_md5": RESOURCES,
    }
    for split in ["train", "val", "test"]:
        result["y" + split] = torch.nn.functional.one_hot(result[split + "_labels"], 10).float()
    # Reference draws are fixed, roughly class balanced, and never used for tuning.
    rng = torch.Generator().manual_seed(20261020)
    for split in ["train", "val"]:
        ids = []
        for digit in range(10):
            candidates = torch.where(result[split + "_labels"] == digit)[0]
            ids.append(candidates[torch.randperm(len(candidates), generator=rng)[:26 if digit < 6
                                                                             else 25]])
        result[split + "_reference_ids"] = torch.cat(ids)
    torch.save(result, destination)
    return result


def initialize(width, seed, input_dim=784, classes=10, dtype=torch.float32):
    incoming_rng = torch.Generator().manual_seed(50000 + seed)
    outgoing_rng = torch.Generator().manual_seed(60000 + seed)
    return [
        torch.randn(width, input_dim, generator=incoming_rng, dtype=dtype) / math.sqrt(input_dim),
        torch.zeros(width, dtype=dtype),
        0.1 * torch.randn(width, classes, generator=outgoing_rng, dtype=dtype),
    ]


@torch.jit.script
def scores(p: list[torch.Tensor], x: torch.Tensor):
    return torch.relu(x @ p[0].T + p[1]) @ p[2] / math.sqrt(p[0].shape[0])


@torch.jit.script
def loss_rows(z: torch.Tensor, y: torch.Tensor, ce: bool):
    if ce:
        return z.logsumexp(-1) - (z * y).sum(-1)
    return 0.5 * (z - y).square().sum(-1)


@torch.jit.script
def gradients(p: list[torch.Tensor], x: torch.Tensor, y: torch.Tensor, ce: bool):
    norm = math.sqrt(p[0].shape[0])
    hidden = torch.relu(x @ p[0].T + p[1])
    z = hidden @ p[2] / norm
    derivative = z.softmax(-1) - y if ce else z - y
    derivative = derivative / (x.shape[0] * norm)
    dh = (derivative @ p[2].T) * (hidden > 0)
    return [dh.T @ x, dh.sum(0), hidden.T @ derivative]


@torch.jit.script
def propose(p: list[torch.Tensor], g: list[torch.Tensor], first: list[torch.Tensor],
            second: list[torch.Tensor], kind: int, lr: float, completed: int):
    q = torch.jit.annotate(list[torch.Tensor], [])
    new_first = torch.jit.annotate(list[torch.Tensor], [])
    new_second = torch.jit.annotate(list[torch.Tensor], [])
    for i in range(len(p)):
        if kind == 2:
            m = 0.9 * first[i] + 0.1 * g[i]
            v = 0.999 * second[i] + 0.001 * g[i].square()
            direction = (m / (1.0 - 0.9 ** (completed + 1))) / (
                (v / (1.0 - 0.999 ** (completed + 1))).sqrt() + 1e-8)
        else:
            m = g[i] if kind == 0 else 0.9 * first[i] + g[i]
            v = second[i]
            direction = m
        q.append(p[i] - lr * direction)
        new_first.append(m)
        new_second.append(v)
    return q, new_first, new_second


@torch.jit.script
def advance(p: list[torch.Tensor], first: list[torch.Tensor], second: list[torch.Tensor],
            x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor, ce: bool,
            kind: int, lr: float, completed: int):
    for i in range(indices.shape[0]):
        g = gradients(p, x[indices[i]], y[indices[i]], ce)
        p, first, second = propose(p, g, first, second, kind, lr, completed + i)
    return p, first, second


def pack(p):
    return torch.cat([p[0], p[1][:, None], p[2]], dim=1)


def components(p, x):
    hidden = torch.relu(x @ p[0].T + p[1])
    return hidden[:, :, None] * p[2][None, :, :] / math.sqrt(p[0].shape[0])


@torch.no_grad()
def evaluate(p, x, y, ce, batch=2048):
    total, correct = 0.0, 0
    for start in range(0, len(x), batch):
        z = scores(p, x[start:start + batch])
        total += float(loss_rows(z, y[start:start + batch], ce).double().sum())
        correct += int((z.argmax(-1) == y[start:start + batch].argmax(-1)).sum())
    return {"loss": total / len(x), "accuracy": correct / len(x)}


@torch.no_grad()
def interaction_statistics(p, q, x, y, ce, rng, pair_samples=256):
    """All isolated effects; all MSE pairs, uniform sampled exact CE pairs.

    Evaluate actual rounded float32 before/after states in float64. Pair sums
    use inverse-inclusion weights. Frequency is a ratio of estimated counts.
    """
    before = components([v.double() for v in p], x.double())
    after = components([v.double() for v in q], x.double())
    y = y.double()
    z, d = before.sum(1), after - before
    initial = loss_rows(z, y, ce).mean()
    final = loss_rows(after.sum(1), y, ce).mean()
    if ce:
        alone = (loss_rows(z[:, None, :] + d, y[:, None, :], True)
                 - loss_rows(z, y, True)[:, None]).mean(0)
    else:
        alone = ((z - y)[:, None, :] * d + 0.5 * d.square()).sum(-1).mean(0)
    width = len(alone)
    i, j = torch.triu_indices(width, width, 1)
    population = len(i)
    sampled = min(pair_samples, population) if ce else population
    if sampled < population:
        selected = torch.randperm(population, generator=rng)[:sampled]
        i, j = i[selected], j[selected]
    if ce:
        parts = []
        for start in range(0, sampled, 64):
            a, b = i[start:start + 64], j[start:start + 64]
            together = loss_rows(z[:, None, :] + d[:, a] + d[:, b], y[:, None, :], True)
            parts.append(together.mean(0) - initial - alone[a] - alone[b])
        pair = torch.cat(parts)
    else:
        vectors = d.permute(1, 0, 2).reshape(width, -1)
        pair = (vectors @ vectors.T / len(x))[i, j]
    weight = population / sampled
    gain = float((-alone).clamp_min(0).sum())
    harm = float(alone.clamp_min(0).sum())
    net_all = float(final - initial - alone.sum())
    tolerance = 1e-10 * float(initial)
    eligible = (alone[i] < -tolerance) & (alone[j] < -tolerance)
    marginal_i, marginal_j = alone[i] + pair, alone[j] + pair
    events_i = eligible & (marginal_i > tolerance)
    events_j = eligible & (marginal_j > tolerance)
    plus, minus = pair.clamp_min(0), (-pair).clamp_min(0)
    fraction = 1 - sampled / population
    plus_variance = (population ** 2 * fraction * float(plus.var()) / sampled
                     if 1 < sampled < population else 0.0)
    result = dict(
        loss_before=float(initial), loss_after=float(final), progress=float(initial - final),
        gain=gain, harm=harm, net_all=net_all,
        pair_plus=weight * float(plus.sum()), pair_minus=weight * float(minus.sum()),
        pair_net=weight * float(pair.sum()), pair_plus_sampling_variance=plus_variance,
        pair_samples=sampled, pair_population=population,
        eligible=2 * weight * int(eligible.sum()),
        strict_events=weight * (int(events_i.sum()) + int(events_j.sum())),
        strict_harm=weight * float((marginal_i * events_i + marginal_j * events_j).sum()),
        eligible_gain=weight * float(((-alone[i] - alone[j]) * eligible).sum()),
        harmful_neurons=int((alone > tolerance).sum()), neurons=width,
        budget_error=abs(float(final - initial) - (harm - gain + net_all)),
        logit_additivity_error=float((after.sum(1) - z - d.sum(1)).abs().max()),
    )
    if not ce:
        result["pair_identity_error"] = abs(net_all - result["pair_net"])
    else:
        result["pair_identity_error"] = None
    return result, alone, (i, j, pair)


@torch.no_grad()
def batch_transfer(p, first, second, xtrain, ytrain, ce, kind, lr, completed, seed,
                   probes=32, batch_size=128):
    """Frozen prior moments for every candidate source; ordered target/source pairs."""
    p = [v.double() for v in p]
    first, second = [v.double() for v in first], [v.double() for v in second]
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(xtrain), (probes, batch_size), generator=rng)
    gs, hs, candidates, losses = [], [], [], []
    for ids in indices:
        x, y = xtrain[ids].double(), ytrain[ids].double()
        g = gradients(p, x, y, ce)
        q, _, _ = propose(p, g, first, second, kind, lr, completed)
        gs.append(pack(g))
        hs.append(pack([(a - b) / lr for a, b in zip(p, q)]))
        candidates.append(q)
        losses.append(float(loss_rows(scores(p, x), y, ce).mean()))
    g, h = torch.stack(gs), torch.stack(hs)
    transfer = g.flatten(1) @ h.flatten(1).T
    raw = g.flatten(1) @ g.flatten(1).T
    off = ~torch.eye(probes, dtype=torch.bool)
    energy_g, energy_h = g.square().sum((1, 2)).mean(), h.square().sum((1, 2)).mean()
    scale = float((energy_g * energy_h).sqrt())
    raw_scale = float(energy_g)
    pos, neg = torch.zeros(probes, probes, dtype=torch.float64), torch.zeros_like(transfer)
    raw_pos, raw_neg = torch.zeros_like(transfer), torch.zeros_like(transfer)
    reconstructed = torch.zeros_like(transfer)
    for start in range(0, p[0].shape[0], 8):
        t = torch.einsum("crd,brd->cbr", g[:, start:start + 8], h[:, start:start + 8])
        r = torch.einsum("crd,brd->cbr", g[:, start:start + 8], g[:, start:start + 8])
        pos += t.clamp_min(0).sum(-1)
        neg += (-t).clamp_min(0).sum(-1)
        raw_pos += r.clamp_min(0).sum(-1)
        raw_neg += (-r).clamp_min(0).sum(-1)
        reconstructed += t.sum(-1)
    finite_gains, predicted_gains = [], []
    for source in range(min(4, probes // 2)):
        for target in range(4, min(8, probes)):
            ids = indices[target]
            actual = loss_rows(scores(candidates[source], xtrain[ids].double()),
                               ytrain[ids].double(), ce).mean()
            finite_gains.append(losses[target] - float(actual))
            predicted_gains.append(lr * float(transfer[target, source]))
    return dict(
        probe_loss=float(np.mean(losses)), probes=probes, ordered_pairs=int(off.sum()),
        harmful_frequency=float((transfer[off] < 0).double().mean()),
        harmful_gain=lr * float((-transfer[off]).clamp_min(0).mean()),
        helpful_gain=lr * float(transfer[off].clamp_min(0).mean()),
        net_first_order_gain=lr * float(transfer[off].mean()),
        p_step=float(pos[off].mean()) / scale, n_step=float(neg[off].mean()) / scale,
        raw_harmful_frequency=float((raw[off] < 0).double().mean()),
        raw_p=float(raw_pos[off].mean()) / raw_scale,
        raw_n=float(raw_neg[off].mean()) / raw_scale,
        gradient_energy=float(energy_g), direction_energy=float(energy_h), scale=scale,
        attribution_error=float((transfer - reconstructed).abs().max()),
        finite_probe_gain=float(np.mean(finite_gains)),
        finite_probe_prediction=float(np.mean(predicted_gains)),
        finite_probe_harmful_frequency=float(np.mean(np.array(finite_gains) < 0)),
        finite_probe_count=len(finite_gains),
    )
