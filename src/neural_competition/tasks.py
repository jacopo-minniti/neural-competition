"""Deterministic paired Gaussian teacher task, normalized by training variance."""

import math

import torch

from .residual_growth import conditional_gram


def paired_gaussian_task(degrees, ratio, data_seed):
    """Four feature pairs in 20D; 4096 train/test examples, no target centering.

    ``degrees`` is the pair half-angle, and each pair has coefficients (1, ratio).
    QR basis, data RNG order, and sample-standard-deviation normalization match
    the original executed study. Returned coarse fits are analytical probes,
    not teacher-informed student initialization.
    """
    rng = torch.Generator().manual_seed(data_seed)
    basis = torch.linalg.qr(torch.randn(20, 8, generator=rng, dtype=torch.float64)).Q.T
    centers, contrasts = basis[::2], basis[1::2]
    theta = math.radians(degrees)
    teacher = torch.stack([math.cos(theta)*centers+math.sin(theta)*contrasts,
                           math.cos(theta)*centers-math.sin(theta)*contrasts], 1).flatten(0, 1)
    beta = torch.tensor([1., ratio]*4, dtype=torch.float64)/math.sqrt(8)
    x = torch.randn(8192, 20, generator=rng, dtype=torch.float64)
    y = (x @ teacher.T).relu() @ beta
    scale = y[:4096].std()
    y, beta = y/scale, beta/scale
    cg = conditional_gram(teacher, centers)
    phi = (x @ centers.T).relu()
    coarse_beta = torch.linalg.lstsq(phi[:4096], y[:4096], driver='gelsd').solution
    return dict(xtrain=x[:4096], xtest=x[4096:], ytrain=y[:4096], ytest=y[4096:],
                teacher=teacher, beta=beta, centers=centers, conditional_gram=cg,
                predicted_growth=beta.sign()*(cg @ beta),
                coarse_beta=coarse_beta,
                coarse_test_loss=float(.5*((phi[4096:] @ coarse_beta)-y[4096:]).square().mean()))
