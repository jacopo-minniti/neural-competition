"""Residual-weighted coupling forecasts for a fixed empirical feature matrix."""

import math

import torch


def fit_modes(phi, targets, readout, relative_cutoff=1e-10):
    n = len(phi)
    diagonal = phi.square().mean(0)
    active = diagonal > 0
    scale = diagonal[active].sqrt()
    z = phi[:, active]/scale
    c = z.T@z/n
    eigenvalues, vectors = torch.linalg.eigh(c)
    keep = eigenvalues > relative_cutoff*eigenvalues.max()
    lam, v = eigenvalues[keep], vectors[:, keep]
    b = scale[:, None]*readout[active]
    work = v.T@(z.T@(z@b-targets)/n)
    energy = work.square().sum(1)/(2*lam)
    optimum = v@((v.T@(z.T@targets/n))/lam[:, None])
    floor = .5*(z@optimum-targets).square().sum(1).mean()
    inverse = (v/lam)@v.T
    metric_diagonal = torch.zeros_like(diagonal)
    metric_diagonal[active] = 1/diagonal[active]
    metric_full = torch.zeros(len(diagonal), len(diagonal), dtype=phi.dtype)
    metric_full[active[:, None] & active[None, :]] = (
        inverse/scale[:, None]/scale[None, :]).flatten()
    return dict(active=active, scale=scale, z=z, c=c, b=b, eigenvalues=lam,
                vectors=v, energy=energy, floor=floor, inverse=inverse,
                metric_diagonal=metric_diagonal, metric_full=metric_full,
                eta=.5/max(1., float(lam.max())))


def forecast(model, time):
    eta = model['eta']
    steps = math.ceil(time/eta-1e-10)
    lam, energy = model['eigenvalues'], model['energy']
    decay = (1-eta*lam).pow(2*steps)
    reference = (1-eta)**(2*steps)
    signed = energy*(decay-reference)
    return dict(steps=steps, time=steps*eta,
                predicted_diagonal=float(model['floor']+(energy*decay).sum()),
                predicted_full=float(model['floor']+energy.sum()*reference),
                predicted_damage=float(signed.sum()),
                slow_cost=float(signed.clamp_min(0).sum()),
                fast_benefit=float((-signed).clamp_min(0).sum()))


def ordinary_metric(model, phi, targets, readout):
    """Remove normalized inter-column coupling while retaining block speeds.

    q are orthonormal loss coordinates in the positive-curvature subspace.
    A is ordinary Euclidean readout GD; B is the stated decoupling benchmark.
    """
    lam, v = model['eigenvalues'], model['vectors']
    d = model['scale'].square()
    b = v.T@(d[:, None]*v)
    a = lam.sqrt()[:, None]*b*lam.sqrt()[None, :]
    inverse_sqrt = v/lam.sqrt()
    active_metric = inverse_sqrt@b@inverse_sqrt.T
    active_metric = active_metric/model['scale'][:, None]/model['scale'][None, :]
    metric = torch.zeros(phi.shape[1],phi.shape[1],dtype=phi.dtype)
    active=model['active']
    metric[active[:,None]&active[None,:]]=active_metric.flatten()
    work=v.T@(model['z'].T@(phi@readout-targets)/len(phi))
    q0=work/lam.sqrt()[:,None]
    return dict(a=a,b=b,q0=q0,metric=metric,inverse_sqrt=inverse_sqrt,
                rate=min(1.,.5/max(float(torch.linalg.eigvalsh(a).max()),
                                  float(torch.linalg.eigvalsh(b).max()))))


def matrix_decay(matrix, rate, steps, q):
    lam,v=torch.linalg.eigh(matrix)
    return v@((1-rate*lam).pow(steps)[:,None]*(v.T@q))
