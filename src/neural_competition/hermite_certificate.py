"""Gaussian second-Hermite obstruction to approximating signed ReLU teachers.

These are population quantities for standard isotropic Gaussian inputs.
Coefficients multiply unnormalized ReLU(incoming @ x), without width scaling.
"""

import math

import torch

from neural_competition.residual_growth import gaussian_relu_gram


def moments(incoming, coefficients):
    radii = incoming.norm(dim=1)
    mass = coefficients*radii
    directions = incoming/radii.clamp_min(1e-30)[:, None]
    mean = mass.sum()/math.sqrt(2*math.pi)
    linear = .5*(coefficients[:, None]*incoming).sum(0)
    quadratic = (directions.T*mass)@directions/math.sqrt(2*math.pi)
    return mean, linear, quadratic


def inertia_bound(target_quadratic, positive: int, negative: int):
    """Population MSE lower bound from available signed neuron counts.

    A rank-one signed ReLU contributes a same-sign semidefinite second moment.
    No fitted matrix with these inertia limits can retain all target eigenvalues.
    """
    eigenvalues = torch.linalg.eigvalsh(target_quadratic)
    pos = eigenvalues.clamp_min(0).flip(0)
    neg = (-eigenvalues).clamp_min(0)
    return .5*(pos[positive:].square().sum()+neg[negative:].square().sum())


def population_certificate(teacher, beta, incoming, coefficients):
    tm, tl, tq = moments(teacher, beta)
    sm, sl, sq = moments(incoming, coefficients)
    all_w = torch.cat([teacher, incoming])
    all_c = torch.cat([beta, -coefficients])
    mse = all_c @ gaussian_relu_gram(all_w, all_w) @ all_c
    radii = incoming.norm(dim=1)
    positive = int(((coefficients>0) & (radii>0)).sum())
    negative = int(((coefficients<0) & (radii>0)).sum())
    bound = inertia_bound(tq, positive, negative)
    mean_error = (tm-sm).square()
    linear_error = (tl-sl).square().sum()
    quadratic_error = .5*(tq-sq).square().sum()
    return dict(population_mse=float(mse), sign_bound=float(bound),
                mean_error=float(mean_error), linear_error=float(linear_error),
                quadratic_error=float(quadratic_error),
                higher_error=float(mse-mean_error-linear_error-quadratic_error),
                positive=positive, negative=negative)
