"""Derive option filters from ordinary CE gradient descent; audit both approximations."""

import copy
import json

import numpy as np
import torch
from explore_conditional_editability import OUT as SOURCE
from explore_conditional_editability import ROOT, Net, contrasts, kernel
from torch.nn import functional as F

OUT = ROOT/'docs/evidence/gd_options'
torch.set_num_threads(2)


def geometry(model, x, y, u):
    logits = model(x).detach()
    probability = logits.softmax(-1)
    h = torch.diag_embed(probability)-probability[:, :, None]*probability[:, None, :]
    h = torch.einsum('ac,nab,bd->ncd', u, h, u)
    values, vectors = torch.linalg.eigh(h)
    root = (vectors*values.clamp_min(1e-14).sqrt()[:, None, :]) @ vectors.transpose(1, 2)
    inverse = (vectors/values.clamp_min(1e-14).sqrt()[:, None, :]) @ vectors.transpose(1, 2)
    gradient = (probability-F.one_hot(y, 10)) @ u
    residual = -torch.einsum('nij,nj->ni', inverse, gradient).flatten()/np.sqrt(len(x))
    raw = kernel(model, x, u)
    blockroot = torch.block_diag(*root)
    weighted = blockroot @ raw @ blockroot/len(x)
    return logits, residual, raw, weighted, float(values.min()), blockroot


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(seeds=[281501, 281502], widths=[32, 128], checkpoints=[0, 100, 600],
        audit_seed=281710, examples=64, heldout_examples=128, rates=[.005, .05], horizons=[1, 5, 20],
        count_horizon=1., count_metric='current CE curvature; uniform-probability curvature control',
        note='Stored before new branch outcomes. Saved checkpoints were trained by ordinary SGD.')
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    data = torch.load(ROOT/'data/mnist/prepared.pt', weights_only=True)
    ids = torch.randperm(8192, generator=torch.Generator().manual_seed(281710))[:64]
    x, y = data['xtrain'][ids].double(), data['train_labels'][ids]
    hx, hy = data['xval'][:128].double(), data['val_labels'][:128]
    u = contrasts()
    rows, branches, saved = [], [], {}
    for seed in protocol['seeds']:
        for width in protocol['widths']:
            for checkpoint in protocol['checkpoints']:
                model = Net(width).double()
                model.load_state_dict(torch.load(SOURCE/f'model_{seed}_{width}_{checkpoint}.pt', weights_only=True))
                logits, residual, raw, weighted, minimum, blockroot = geometry(model, x, y, u)
                eigen, vectors = torch.linalg.eigh(weighted)
                eigen = eigen.clamp_min(0)
                coefficients = vectors.T @ residual
                geometric = .1*raw/len(x)
                ge, gv = torch.linalg.eigh(geometric)
                ge = ge.clamp_min(0)
                discounted = 2*eigen/(1+2*eigen)
                discounted_geometry = 2*ge/(1+2*ge)
                # Fixed residual coordinates: discounted ordinary-GD loss-benefit operator.
                # Woodbury form avoids inverse CE curvature at highly confident examples.
                gkernel = raw/len(x)
                crossroot = gkernel @ blockroot
                benefit = 2*(gkernel-(crossroot @ (vectors/(eigen+.5))) @ (vectors.T @ crossroot.T))
                benefit = (benefit+benefit.T)/2
                be = torch.linalg.eigvalsh(benefit).clamp_min(0)
                error = ((F.one_hot(y, 10)-logits.softmax(-1)) @ u).flatten()
                base_loss = float(F.cross_entropy(logits, y))
                row = dict(seed=seed, width=width, checkpoint=checkpoint, loss=base_loss,
                    current_curvature_options=float((1-torch.exp(-2*eigen)).sum()),
                    fixed_curvature_options=float((1-torch.exp(-2*ge)).sum()),
                    discounted_options=float(discounted.sum()),
                    discounted_geometry_options=float(discounted_geometry.sum()),
                    current_loss_weighted_value=float(.5*(discounted*coefficients.square()).sum()),
                    option_mass=float(be.sum()), option_participation=float(be.sum()**2/be.square().sum()),
                    benefit_identity_error=float(abs((error@benefit@error)/(2*len(x))-
                        .5*(discounted*coefficients.square()).sum())),
                    mean_curvature_rate=float(eigen.mean()), min_output_curvature=minimum,
                    max_curvature_rate=float(eigen.max()))
                saved[f'{seed}_{width}_{checkpoint}_loss_R'] = ((vectors*discounted)@vectors.T).numpy()
                saved[f'{seed}_{width}_{checkpoint}_geometry_R'] = ((gv*discounted_geometry)@gv.T).numpy()
                saved[f'{seed}_{width}_{checkpoint}_eigen'] = eigen.numpy()
                saved[f'{seed}_{width}_{checkpoint}_benefit'] = benefit.numpy()
                rows.append(row)
                print(json.dumps(row), flush=True)
                if seed == protocol['seeds'][0]:
                    combined = kernel(model, torch.cat([x, hx]), u)
                    cross = combined[len(x)*9:, :len(x)*9]/len(x)
                    for rate in protocol['rates']:
                        branch = copy.deepcopy(model)
                        z = logits @ u
                        held_z = model(hx).detach() @ u
                        cumulative = torch.zeros(len(x)*9, dtype=torch.float64)
                        initial_held_loss = float(F.cross_entropy(model(hx), hy).detach())
                        initial_rows = F.cross_entropy(logits, y, reduction='none')
                        for step in range(1, 21):
                            branch.zero_grad()
                            F.cross_entropy(branch(x), y).backward()
                            with torch.no_grad():
                                for parameter in branch.parameters():
                                    parameter.add_(parameter.grad, alpha=-rate)
                                grad = ((z @ u.T).softmax(-1)-F.one_hot(y, 10)) @ u
                                cumulative -= rate*grad.flatten()
                                z -= rate*(raw @ grad.flatten()).reshape(-1, 9)/len(x)
                            if step in protocol['horizons']:
                                predicted_gain = .5*((1-(1-rate*eigen)**(2*step))*coefficients.square()).sum()
                                actual_rows = F.cross_entropy(branch(x), y, reduction='none').detach()
                                frozen_rows = F.cross_entropy(z @ u.T, y, reduction='none')
                                actual = float(actual_rows.mean())
                                frozen = float(F.cross_entropy(z @ u.T, y))
                                frozen_held = float(F.cross_entropy((held_z+(cross@cumulative).reshape(-1, 9)) @ u.T, hy))
                                branches.append(dict(seed=seed, width=width, checkpoint=checkpoint,
                                    rate=rate, steps=step, stable=float(rate*eigen.max())<2,
                                    base_loss=base_loss, predicted_quadratic_loss=base_loss-float(predicted_gain),
                                    frozen_jacobian_ce_loss=frozen, actual_loss=actual,
                                    predicted_gain=float(predicted_gain), actual_gain=base_loss-actual,
                                    actual_collateral=float((actual_rows-initial_rows).clamp_min(0).mean()),
                                    frozen_collateral=float((frozen_rows-initial_rows).clamp_min(0).mean()),
                                    initial_heldout_loss=initial_held_loss,
                                    actual_heldout_loss=float(F.cross_entropy(branch(hx), hy).detach()),
                                    frozen_jacobian_heldout_loss=frozen_held))
                (OUT/'results.json').write_text(json.dumps(dict(protocol=protocol,
                    measurements=rows, branches=branches), indent=2)+'\n')
                np.savez_compressed(OUT/'operators.npz', **saved)


if __name__ == '__main__':
    main()
