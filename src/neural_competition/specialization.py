"""A population ReLU specialization construction and ordinary training engine."""

import math

import torch

from neural_competition.trajectory import batch_gradient


def relu_kernel(angle):
    """E[relu(u.X)relu(v.X)] for unit u,v and isotropic Gaussian X."""
    return (math.sin(angle)+(math.pi-angle)*math.cos(angle))/(2*math.pi)


def merged_coefficients(angle):
    """Two unit teacher coefficients; ordinary/split quadratic loss coefficients.

    Ordinary motion profiles out the single outgoing coefficient. The split
    keeps total outgoing mass fixed and uses c +/- epsilon*d (not unit vectors).
    """
    a = 4*relu_kernel(angle)
    ordinary = a*((math.pi-angle)*math.cos(angle)-math.sin(angle))/(2*math.pi)
    split = -a*math.sin(angle)/math.pi
    return dict(output=a, ordinary_quadratic=ordinary, split_quadratic=split)


def pair_innovation(xc, xd, angle):
    return (math.sin(angle)*xd.abs()-math.cos(angle)*xc.abs()).relu()


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


def resolved_features(p, teacher, angle, norm):
    """Maximum matching: distinct positive-output student for each target ray."""
    if angle == 0:
        return float('nan')
    incoming = p[:, :-2]
    directions = incoming/incoming.norm(dim=1,keepdim=True).clamp_min(1e-30)
    cosine = teacher @ directions.T
    valid = (p[:, -1]/norm > 1e-4) & (incoming.norm(dim=1) > 1e-8)
    neighbors = [torch.where((row > math.cos(angle/2)) & valid)[0].tolist() for row in cosine]
    owners = {}
    def assign(target, visited):
        for unit in neighbors[target]:
            if unit in visited:
                continue
            visited.add(unit)
            if unit not in owners or assign(owners[unit],visited):
                owners[unit] = target
                return True
        return False
    return sum(assign(t,set()) for t in range(len(teacher)))
