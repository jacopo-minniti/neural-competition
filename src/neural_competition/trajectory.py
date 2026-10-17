"""Small batched momentum runner and exact per-update interaction accounting.

The padded neurons stay identically zero. Training uses float32; diagnostics
evaluate the actual rounded before/after parameters in float64.
"""

import torch

FIELDS = [
    "loss_before",
    "loss_after",
    "solo_gain",
    "solo_harm",
    "penalty",
    "credit",
    "positive_pairs",
    "pairs",
    "helpful_pairs",
    "positive_helpful_pairs",
    "strong_events",
    "strong_harm",
    "individual_energy",
    "helpful_neurons",
]
VIEWS = ["source", "reference"]


@torch.jit.script
def momentum_step(p, velocity, x, y, norm, lr: float, momentum: float):
    """Analytic half-MSE gradient; equivalent to ordinary PyTorch SGD momentum."""
    dim = x.shape[-1]
    z = torch.bmm(x, p[:, :, :dim].transpose(1, 2)) + p[:, None, :, dim]
    hidden = z.relu()
    prediction = (hidden * p[:, None, :, -1]).sum(-1) / norm
    error = (prediction - y) / (x.shape[1] * norm)
    incoming = error[:, :, None] * p[:, None, :, -1] * (z > 0)
    gradient = torch.cat(
        [
            torch.bmm(incoming.transpose(1, 2), x),
            incoming.sum(1)[:, :, None],
            (error[:, :, None] * hidden).sum(1)[:, :, None],
        ],
        dim=-1,
    )
    velocity = torch.add(velocity * momentum, gradient)
    return torch.add(p, velocity, alpha=-lr), velocity


@torch.jit.script
def components(p, x, norm):
    dim = x.shape[-1]
    z = torch.bmm(x, p[:, :, :dim].transpose(1, 2)) + p[:, None, :, dim]
    return z.relu() * p[:, None, :, -1] / norm[:, :, None]


@torch.jit.script
def interaction_metrics(before, after, y, upper):
    """[models, examples, neurons] -> [models, fields]; each pair counted once."""
    changes = after - before
    residual = before.sum(-1) - y
    ending = after.sum(-1) - y
    baseline = 0.5 * residual.square().mean(1)
    loss_after = 0.5 * ending.square().mean(1)
    alone = (residual[:, :, None] * changes + 0.5 * changes.square()).mean(1)
    gram = torch.bmm(changes.transpose(1, 2), changes) / changes.shape[1]
    tolerance = 1e-10 * baseline[:, None, None]
    helpful = alone < -tolerance[:, :, 0]
    eligible = helpful[:, :, None] & helpful[:, None, :] & upper
    directed = upper | upper.transpose(1, 2)
    eligible_ordered = helpful[:, :, None] & helpful[:, None, :] & directed
    marginal = alone[:, None, :] + gram
    strong = eligible_ordered & (marginal > tolerance)
    return torch.stack(
        [
            baseline,
            loss_after,
            (-alone).clamp_min(0).sum(1),
            alone.clamp_min(0).sum(1),
            (gram.clamp_min(0) * upper).sum((1, 2)),
            ((-gram).clamp_min(0) * upper).sum((1, 2)),
            ((gram > tolerance) & upper).sum((1, 2)).double(),
            upper.sum((1, 2)).double(),
            eligible.sum((1, 2)).double(),
            ((gram > tolerance) & eligible).sum((1, 2)).double(),
            strong.sum((1, 2)).double(),
            torch.where(strong, marginal, 0.0).sum((1, 2)),
            0.5 * changes.square().mean(1).sum(1),
            helpful.sum(1).double(),
        ],
        dim=-1,
    )


@torch.jit.script
def run_block(
    p,
    velocity,
    x_train,
    y_train,
    indices,
    seed_map,
    norm,
    x_reference,
    y_reference,
    upper,
    lr: float,
    momentum: float,
):
    """All actual updates are measured, including steps between plotted points."""
    records = torch.empty((indices.shape[0], p.shape[0], 2, 14), dtype=torch.float64)
    norm64 = norm.double().sqrt()
    norm32 = norm64.float()
    for step in range(indices.shape[0]):
        ids = indices[step].index_select(0, seed_map)
        x, y = x_train[ids], y_train[ids]
        following, velocity = momentum_step(p, velocity, x, y, norm32, lr, momentum)
        combined = torch.cat([x.double(), x_reference], dim=1)
        before = components(p.double(), combined, norm64)
        after = components(following.double(), combined, norm64)
        n = x.shape[1]
        records[step, :, 0] = interaction_metrics(before[:, :n], after[:, :n], y.double(), upper)
        records[step, :, 1] = interaction_metrics(before[:, n:], after[:, n:], y_reference, upper)
        p = following
    return p, velocity, records
