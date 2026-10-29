"""Numerical, matrix-free, and changing-option audits for the MNIST study."""

from __future__ import annotations

import json

import numpy as np
import torch
from explore_conditional_editability import (
    OUT,
    ROOT,
    Net,
    contrasts,
    kernel,
    shift,
    solve,
    transform,
)
from torch.func import functional_call, jvp, vjp


def conjugate_gradient(action, rhs, tolerance=1e-7, limit=200):
    x = torch.zeros_like(rhs)
    residual, direction = rhs.clone(), rhs.clone()
    squared = residual @ residual
    for iteration in range(limit):
        response = action(direction)
        alpha = squared / (direction @ response)
        x += alpha * direction
        residual -= alpha * response
        updated = residual @ residual
        if updated.sqrt() < tolerance * rhs.norm():
            return x, iteration+1, float(updated.sqrt()/rhs.norm())
        direction = residual + (updated/squared) * direction
        squared = updated
    return x, limit, float(updated.sqrt()/rhs.norm())


def main():
    data = torch.load(ROOT/'data/mnist/prepared.pt', weights_only=True)
    ids = torch.randperm(5000, generator=torch.Generator().manual_seed(281500))[:96]
    original = data['xval'][ids[:32]].double()
    x = torch.cat([original, shift(original, fill=float(data['xval'].min()))])
    u, t = contrasts(), transform(32)
    model = Net(32).double()
    model.load_state_dict(torch.load(OUT/'model_281501_32_600.pt', weights_only=True))
    # Independent autodiff audit, including biases and both trainable layers.
    small = x[:3]
    rows = []
    for scalar in (model(small) @ u).flatten():
        derivatives = torch.autograd.grad(scalar, tuple(model.parameters()), retain_graph=True)
        rows.append(torch.cat([d.flatten() for d in derivatives]))
    jacobian = torch.stack(rows)
    analytic = kernel(model, small, u)
    autodiff_error = float((analytic-jacobian@jacobian.T).norm()/analytic.norm())
    # Matched-cost exact duplicates preserve functional options, despite doubling columns.
    cloned = torch.cat([jacobian, jacobian], dim=1)/np.sqrt(2)
    clone_error = float((cloned@cloned.T-analytic).norm()/analytic.norm())
    names = tuple(dict(model.named_parameters()))
    parameters = tuple(model.parameters())

    def function(parameters):
        return t @ ((functional_call(model, dict(zip(names, parameters)), (x,))) @ u).flatten()

    _, pullback = vjp(function, parameters)

    def multiply(vector):
        # No parameter Jacobian or kernel matrix is formed here.
        return jvp(function, (parameters,), (pullback(vector)[0],))[1].detach()

    k = t @ kernel(model, x, u) @ t.T
    ridge = .1*float(k.trace()/len(k))
    q = 144
    r, _, _ = solve(k, q, ridge)
    generator = torch.Generator().manual_seed(281503)
    random = torch.randn(len(k), generator=generator, dtype=torch.float64)
    action_error = float((multiply(random)-k@random).norm()/(k@random).norm())
    estimates, iterations, residuals = [], [], []
    for _ in range(16):
        z = (2*torch.randint(2, (q,), generator=generator)-1).double()
        rhs = torch.cat([z, torch.zeros(len(k)-q)])
        alpha, count, residual = conjugate_gradient(lambda v: multiply(v)+ridge*v, rhs)
        estimates.append(float(z @ multiply(alpha)[:q]))
        iterations.append(count)
        residuals.append(residual)
    operators = np.load(OUT/'operators.npz')
    turnover = []
    for seed in (281501, 281502):
        for width in (32, 128):
            for suffix in ('R', 'R_fixed'):
                initial = operators[f'{seed}_{width}_0_{suffix}']
                final = operators[f'{seed}_{width}_600_{suffix}']
                eigen = np.linalg.eigvalsh(final-initial)
                turnover.append(dict(seed=seed, width=width, scale=suffix,
                    lost=float(-eigen[eigen<0].sum()), gained=float(eigen[eigen>0].sum()),
                    net=float(np.trace(final-initial))))
    # Random-subset audit at the final states; roles are never chosen by class or correctness.
    split_rows, protection_rows, heldout_rows = [], [], []
    for width in (32, 128):
        model = Net(width).double()
        model.load_state_dict(torch.load(OUT/f'model_281501_{width}_600.pt', weights_only=True))
        full = t @ kernel(model, x, u) @ t.T
        ridge = .1*float(full.trace()/len(full))
        fresh = data['xval'][ids[32:]].double()
        fresh_views = torch.cat([fresh, shift(fresh, fill=float(data['xval'].min()))])
        combined = kernel(model, torch.cat([x, fresh_views]), u)
        cross = combined[len(x)*9:, :len(x)*9]
        cross = (cross[:len(fresh)*9] + cross[len(fresh)*9:])/(2*np.sqrt(len(fresh)))
        cross = cross @ t.T
        for multiplier in (.01, .1, 1., 10.):
            rr, _, dual = solve(full, q, multiplier*float(full.trace()/len(full)))
            fit_response = (full @ dual)[q:2*q]
            unseen_response = cross @ dual
            excess = float(unseen_response.square().sum()-fit_response.square().sum())
            heldout_rows.append(dict(width=width, ridge_multiplier=multiplier,
                in_sample_option_value=float(rr.trace()),
                heldout_background_option_value=float(rr.trace())-excess,
                excess_collateral=excess,
                note='Background-only cross-fit correction; focal views remain in-sample.'))
        for gamma in (.1, 1., 10., 100.):
            weights = torch.ones(len(full), dtype=torch.float64)
            weights[q:2*q] *= np.sqrt(gamma)
            protected = weights[:, None]*full*weights[None, :]
            _, eig, _ = solve(protected, q, ridge)
            protection_rows.append(dict(width=width, gamma=gamma, options=float(eig.sum())))
        for split_seed in range(4):
            order = torch.randperm(32, generator=torch.Generator().manual_seed(281600+split_seed))
            index = torch.cat([order, order+32])
            k = t @ kernel(model, x[index], u) @ t.T
            _, eig, _ = solve(k, q, ridge)
            split_rows.append(dict(width=width, split_seed=split_seed, options=float(eig.sum())))
    result = dict(analytic_autodiff_relative_error=autodiff_error,
        matched_cost_clone_relative_error=clone_error,
        matrix_free_action_relative_error=action_error,
        matrix_free_trace=dict(probes=16, estimate=float(np.mean(estimates)),
            standard_error=float(np.std(estimates, ddof=1)/4), exact=float(r.trace()),
            max_cg_iterations=max(iterations), max_relative_residual=max(residuals)),
        turnover=turnover, random_splits=split_rows, protection_curve=protection_rows,
        heldout_background=heldout_rows)
    (OUT/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
