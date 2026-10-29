"""Ordinary optimizer proposals with optional projection onto readout half-lines."""

import math

import torch

from neural_competition.training import batch_gradient


@torch.jit.script
def advance_protected(p, first, second, xtrain, ytrain, indices, seeds, norm,
                      rates, lr: float, kind: int, completed: int,
                      signs, protect, damping_sources):
    counts = torch.zeros(p.shape[0], dtype=torch.int64)
    correction_l1 = torch.zeros(p.shape[0], dtype=p.dtype)
    damping_caps = torch.zeros(p.shape[0], dtype=torch.int64)
    correction_l2_sum = torch.zeros(p.shape[0], dtype=p.dtype)
    for k in range(indices.shape[0]):
        ids = indices[k].index_select(0, seeds)
        gradient = batch_gradient(p, xtrain[ids], ytrain[ids], norm)
        gradient[:, :, -2] = 0.
        if kind == 2:
            first = .9*first+.1*gradient
            second = .999*second+.001*gradient.square()
            step = completed+k+1
            direction = (first/(1-.9**step))/(second.sqrt()/math.sqrt(1-.999**step)+1e-8)
        else:
            first = gradient if kind == 0 else .9*first+gradient
            direction = first
        old_readout = p[:, :, -1].clone()
        p = p-lr*rates*direction
        rejected = (p[:, :, -1]*signs < 0) & protect[:, None]
        correction_l2 = torch.where(rejected, p[:, :, -1], 0.).square().sum(1).sqrt()
        counts += rejected.sum(1)
        correction_l1 += torch.where(rejected, p[:, :, -1].abs(), 0.).sum(1)
        p[:, :, -1] = torch.where(rejected, 0., p[:, :, -1])
        damp = damping_sources >= 0
        budget = correction_l2.index_select(0, damping_sources.clamp_min(0))*damp
        update = p[:, :, -1]-old_readout
        length = update.square().sum(1).sqrt()
        amount = torch.minimum(budget, length)
        p[:, :, -1] -= (amount/length.clamp_min(1e-30))[:, None]*update
        correction_l2 = torch.where(damp, amount, correction_l2)
        damping_caps += (budget > length).to(torch.int64)
        correction_l2_sum += correction_l2
    return p, first, second, counts, correction_l1, correction_l2_sum, damping_caps
