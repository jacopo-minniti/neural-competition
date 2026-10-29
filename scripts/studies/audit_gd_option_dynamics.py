"""Audit loss-derived coherence, residual accounting, and SGD noise corrections."""

import copy
import json

import numpy as np
import torch
from explore_conditional_editability import OUT as SOURCE
from explore_conditional_editability import ROOT, Net, contrasts, shift
from explore_gd_option_dynamics import OUT, geometry
from torch.nn import functional as F


def main():
    data = torch.load(ROOT/'data/mnist/prepared.pt', weights_only=True)
    ids = torch.randperm(8192, generator=torch.Generator().manual_seed(281710))[:64]
    x, y = data['xtrain'][ids].double(), data['train_labels'][ids]
    u = contrasts()
    rows, identities = [], []
    for width in (32, 128):
        for checkpoint in (0, 600):
            model = Net(width).double()
            model.load_state_dict(torch.load(SOURCE/f'model_281501_{width}_{checkpoint}.pt', weights_only=True))
            logits, residual, raw, weighted, _, _ = geometry(model, x, y, u)
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            gradient_norm = sum(p.grad.square().sum() for p in model.parameters())
            residual_rate = residual @ weighted @ residual
            view = model(shift(x, fill=float(data['xval'].min()))).detach()
            mean = (logits+view)/2
            pmean = mean.softmax(-1)
            kl = .5*((pmean*(pmean.log()-logits.log_softmax(-1))).sum(-1).mean()+
                      (pmean*(pmean.log()-view.log_softmax(-1))).sum(-1).mean())
            augmented = .5*(F.cross_entropy(logits, y)+F.cross_entropy(view, y))
            identity = F.cross_entropy(mean, y)+kl
            identities.append(dict(width=width, checkpoint=checkpoint,
                gradient_rate_relative_error=float(abs(gradient_norm-residual_rate)/gradient_norm),
                augmentation_identity_error=float(abs(augmented-identity))))
            probability = logits.softmax(-1)
            h = torch.diag_embed(probability)-probability[:, :, None]*probability[:, None, :]
            h = torch.einsum('ac,nab,bd->ncd', u, h, u)
            hblock = torch.block_diag(*h)
            g = (probability-F.one_hot(y, 10)) @ u
            responses = torch.einsum('aic,ic->ai', raw.reshape(len(x)*9, len(x), 9), g)
            average = responses.mean(1)
            mean_curvature = (average @ hblock @ average)/len(x)
            per_example_curvature = (responses*(hblock@responses)).sum(0).mean()/len(x)
            covariance_curvature = per_example_curvature-mean_curvature
            base = float(F.cross_entropy(logits, y))
            for rate in (.005, .05):
                for batch_size in (16, 64):
                    noise_cost = .5*rate**2*float(covariance_curvature)/batch_size
                    forecast = base-rate*float(gradient_norm)+.5*rate**2*float(mean_curvature)+noise_cost
                    samples = []
                    generator = torch.Generator().manual_seed(281711)
                    for _ in range(32):
                        batch = torch.randint(len(x), (batch_size,), generator=generator)
                        branch = copy.deepcopy(model)
                        branch.zero_grad()
                        F.cross_entropy(branch(x[batch]), y[batch]).backward()
                        with torch.no_grad():
                            for p in branch.parameters():
                                p.add_(p.grad, alpha=-rate)
                            samples.append(float(F.cross_entropy(branch(x), y)))
                    rows.append(dict(width=width, checkpoint=checkpoint, rate=rate,
                        batch_size=batch_size, base_loss=base, predicted_loss=forecast,
                        observed_mean=float(np.mean(samples)), observed_se=float(np.std(samples, ddof=1)/np.sqrt(32)),
                        predicted_noise_cost=noise_cost))
    saved = np.load(OUT/'operators.npz')
    turnover, benefit_turnover = [], []
    for seed in (281501, 281502):
        for width in (32, 128):
            delta = saved[f'{seed}_{width}_600_geometry_R']-saved[f'{seed}_{width}_0_geometry_R']
            eigen = np.linalg.eigvalsh(delta)
            turnover.append(dict(seed=seed, width=width,
                lost=float(-eigen[eigen<0].sum()), gained=float(eigen[eigen>0].sum())))
            delta = saved[f'{seed}_{width}_600_benefit']-saved[f'{seed}_{width}_0_benefit']
            eigen = np.linalg.eigvalsh(delta)
            benefit_turnover.append(dict(seed=seed, width=width,
                lost=float(-eigen[eigen<0].sum()), gained=float(eigen[eigen>0].sum())))
    result = dict(identities=identities, sgd=rows, fixed_metric_turnover=turnover,
        residual_response_turnover=benefit_turnover,
        caveat='SGD forecast uses frozen generalized Gauss-Newton curvature, not the full Hessian.')
    (OUT/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
