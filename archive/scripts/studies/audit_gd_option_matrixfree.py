"""Trace and spectral breadth without a Jacobian, kernel, or parameter Hessian."""

import json

import numpy as np
import torch
from audit_conditional_editability import conjugate_gradient
from explore_conditional_editability import OUT as SOURCE
from explore_conditional_editability import ROOT, Net, contrasts
from explore_gd_option_dynamics import OUT
from torch.func import functional_call, jvp, vjp


def main():
    data = torch.load(ROOT/'data/mnist/prepared.pt', weights_only=True)
    ids = torch.randperm(8192, generator=torch.Generator().manual_seed(281710))[:64]
    x = data['xtrain'][ids].double()
    model = Net(32).double()
    model.load_state_dict(torch.load(SOURCE/'model_281501_32_600.pt', weights_only=True))
    u = contrasts()
    named = list(model.named_parameters())
    lengths = [p.numel() for _, p in named]
    flat = torch.cat([p.detach().flatten() for _, p in named])

    def function(flat):
        parts = flat.split(lengths)
        parameters = {name: part.reshape(p.shape) for (name, p), part in zip(named, parts)}
        return (functional_call(model, parameters, (x,)) @ u).flatten()/np.sqrt(len(x))

    _, pullback = vjp(function, flat)
    probability = model(x).softmax(-1).detach()

    def output_curvature(vector):
        z = vector.reshape(len(x), 9) @ u.T
        hz = probability*(z-(probability*z).sum(1, keepdim=True))
        return (hz @ u).flatten()

    def action(vector):
        output = jvp(function, (flat,), (vector,))[1]
        return (pullback(output_curvature(output))[0]+.5*vector).detach()

    generator = torch.Generator().manual_seed(281712)
    traces, squares, counts, residuals = [], [], [], []
    for _ in range(16):
        z = (2*torch.randint(2, (len(x)*9,), generator=generator)-1).double()
        rhs = pullback(z)[0].detach()
        solution, steps, residual = conjugate_gradient(action, rhs, tolerance=1e-7)
        response = jvp(function, (flat,), (solution,))[1].detach()
        traces.append(float(z@response))
        squares.append(float(response.square().sum()))
        counts.append(steps)
        residuals.append(residual)
    exact = np.load(OUT/'operators.npz')['281501_32_600_benefit']
    result = dict(probes=16, mass_estimate=float(np.mean(traces)),
        mass_standard_error=float(np.std(traces, ddof=1)/4), exact_mass=float(np.trace(exact)),
        participation_estimate=float(np.mean(traces)**2/np.mean(squares)),
        exact_participation=float(np.trace(exact)**2/np.sum(exact**2)),
        max_cg_steps=max(counts), max_relative_residual=max(residuals),
        note='Participation is a ratio estimator; the reported mass SE does not give its confidence interval.')
    (OUT/'matrixfree.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
