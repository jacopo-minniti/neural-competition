"""Finite-update loss accounting and finite-population response estimation.

These are retrospective identities/estimators, not forecasts or causal scores.
All output losses are means; optimizer metrics must be absorbed into gradients
before using ``panel_signal``.
"""

from __future__ import annotations

import torch


def dot(left, right):
    return sum((a * b).sum() for a, b in zip(left, right, strict=True))


def panel_signal(gradient_mean, mean_example_norm2, size, population):
    """Unbiased squared full-mean gradient for uniform sampling WITHOUT replacement.

    The estimate can be negative; clipping would introduce bias. ``population``
    denotes the finite training set, not an inferred underlying distribution.
    """
    if not 1 < size <= population:
        raise ValueError("Need 1 < panel size <= population size")
    own = dot(gradient_mean, gradient_mean)
    variance = (mean_example_norm2 - own) * size / (size - 1)
    bias = (1 / size - 1 / population) * variance
    return own, own - bias, bias


@torch.no_grad()
def cross_entropy_ledger(before, after, labels, tangent_change):
    """Exact gain = tangent work + nonlinear work - convex Bregman cost.

    Logits and tangent changes use the same examples and output coordinates.
    The Bregman cost equals KL(softmax(before) || softmax(after)).
    """
    logp, logq = before.log_softmax(-1), after.log_softmax(-1)
    p = logp.exp()
    demand = torch.nn.functional.one_hot(labels, before.shape[-1]) - p
    delta = after - before
    tangent = (demand * tangent_change).sum(-1).mean()
    nonlinear = (demand * (delta - tangent_change)).sum(-1).mean()
    cost = (p * (logp - logq)).sum(-1).mean()
    loss0 = -logp.gather(1, labels[:, None]).mean()
    loss1 = -logq.gather(1, labels[:, None]).mean()
    return dict(
        loss=float(loss0),
        next_loss=float(loss1),
        tangent=float(tangent),
        nonlinear=float(nonlinear),
        cost=float(cost),
        gain=float(loss0 - loss1),
        closure=float(loss0 - loss1 - tangent - nonlinear + cost),
    )
