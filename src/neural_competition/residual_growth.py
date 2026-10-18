"""Residual feature potentials and conditional Gram geometry for Gaussian ReLUs."""

import math

import torch


def gaussian_relu_gram(left, right):
    """Population E[relu(left.X)relu(right.X)^T], including vector norms."""
    ln = left.norm(dim=-1)
    rn = right.norm(dim=-1)
    scales = ln[:, None]*rn[None, :]
    cosine = ((left @ right.T)/scales.clamp_min(1e-30)).clamp(-1, 1)
    theta = cosine.acos()
    return scales*(theta.sin()+(math.pi-theta)*cosine)/(2*math.pi)


def conditional_gram(teacher, centers):
    tt = gaussian_relu_gram(teacher, teacher)
    tc = gaussian_relu_gram(teacher, centers)
    cc = gaussian_relu_gram(centers, centers)
    return tt-tc @ torch.linalg.solve(cc, tc.T)


def residual_potential(directions, teacher, beta, incoming, output):
    """Exact population correlations, with output in effective (unscaled) units."""
    return (gaussian_relu_gram(directions, teacher) @ beta
            -gaussian_relu_gram(directions, incoming) @ output)


def relu_kernel(angle):
    """E[relu(u.X)relu(v.X)] for unit u,v and isotropic Gaussian X."""
    return (math.sin(angle)+(math.pi-angle)*math.cos(angle))/(2*math.pi)
