"""Batched scalar ReLU training shared by the residual-growth studies.

Parameters have shape [models, padded_neurons, input_dim + 2], with incoming
weights, hidden bias, then readout. ``norm`` is sqrt(actual width), not padded
width. Zero padding remains zero. SGD/momentum/Adam train both layers with
explicit per-parameter rate multipliers; ``completed`` counts earlier updates.
"""

import math

import torch


@torch.jit.script
def batch_gradient(p, x, y, norm):
    """Analytic half-MSE gradient for each independent model in the batch."""
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
    return gradient


@torch.jit.script
def components(p, x, norm):
    dim = x.shape[-1]
    z = torch.bmm(x, p[:, :, :dim].transpose(1, 2)) + p[:, None, :, dim]
    return z.relu() * p[:, None, :, -1] / norm[:, :, None]


@torch.jit.script
def advance(p, first, second, xtrain, ytrain, indices, seeds, norm, rates,
            lr: float, kind: int, train_bias: bool, completed: int):
    """Float64 joint-layer SGD/momentum/Adam, with explicit per-layer rates."""
    for k in range(indices.shape[0]):
        ids = indices[k].index_select(0, seeds)
        gradient = batch_gradient(p, xtrain[ids], ytrain[ids], norm)
        if not train_bias:
            gradient[:, :, -2] = 0.
        if kind == 2:
            first = .9*first+.1*gradient
            second = .999*second+.001*gradient.square()
            step = completed+k+1
            direction = (first/(1-.9**step))/(second.sqrt()/math.sqrt(1-.999**step)+1e-8)
        else:
            first = gradient if kind == 0 else .9*first+gradient
            direction = first
        p = p-lr*rates*direction
    return p, first, second
