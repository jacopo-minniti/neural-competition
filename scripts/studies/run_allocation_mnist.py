#!/usr/bin/env python3
"""Cross-fitted neuron allocation at existing matched-loss MNIST checkpoints.

MSE uses its exact logit-space quadratic; CE uses the local softmax Hessian.
All reported interventions are evaluated with the actual loss and forward pass.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from neural_competition.allocation import allocation_rules, objective
from neural_competition.mnist import gradients, loss_rows, prepare_data, propose, scores

ROOT = Path(__file__).resolve().parents[2]
KINDS = dict(sgd=0, momentum=1, adam=2)


def linearized(p, delta, x):
    z = x @ p[0].T + p[1]
    dz = x @ delta[0].T + delta[1]
    return (z.relu()[:, :, None] * delta[2][None]
            + ((z > 0) * dz)[:, :, None] * p[2][None]) / math.sqrt(len(p[0]))


def terms(logits, y, ell, ce):
    probability = logits.softmax(-1)
    residual = probability-y if ce else logits-y
    b = (residual[:, None, :] * ell).sum(-1).mean(0)
    if ce:
        weighted = ell * probability[:, None, :]
        means = weighted.sum(-1)
        gram = (torch.einsum('nrc,nsc->rs', ell, weighted) - means.T @ means)/len(y)
    else:
        gram = torch.einsum('nrc,nsc->rs', ell, ell)/len(y)
    return b[None], gram[None]


def apply(p, delta, alpha):
    return [p[0]+alpha[:, None]*delta[0], p[1]+alpha*delta[1],
            p[2]+alpha[:, None]*delta[2]]


def one(path, data, cal, test, sources):
    state = torch.load(path, weights_only=True)
    task = state['task']
    ce = task['loss'] == 'ce'
    p = state['parameters']
    pd = [a.double() for a in p]
    xc, yc = data['xtrain'][cal].double(), data['ytrain'][cal].double()
    xt, yt = data['xtest'][test].double(), data['ytest'][test].double()
    zc, zt = scores(pd, xc), scores(pd, xt)
    lc, lt = float(loss_rows(zc, yc, ce).mean()), float(loss_rows(zt, yt, ce).mean())
    rows = []
    for source in range(sources):
        rng = torch.Generator().manual_seed(79300 + 1000*task['seed'] + source)
        ids = torch.randint(len(data['xtrain']), (task['batch_size'],), generator=rng)
        g = gradients(p, data['xtrain'][ids], data['ytrain'][ids], ce)
        q, _, _ = propose(p, g, state['first'], state['second'], KINDS[task['optimizer']],
                          task['lr'], state['completed'])
        for fraction in [1., .01]:
            delta = [fraction*(b.double()-a) for a, b in zip(pd, q)]
            ell_c = linearized(pd, delta, xc)
            ell_t = linearized(pd, delta, xt)
            b, gram = terms(zc, yc, ell_c, ce)
            bt, gt = terms(zt, yt, ell_t, ce)
            rules, cert = allocation_rules(b, gram, torch.tensor([[task['width']]]),
                                           seed=79400 + source)
            best, best_gamma = float('inf'), 0.
            for gamma in [*torch.linspace(0, 1, 33).tolist(), float(rules['uniform'][0, 0])]:
                moved = [a + gamma*d for a, d in zip(pd, delta)]
                loss = float(loss_rows(scores(moved, xc), yc, ce).mean())
                if loss < best:
                    best, best_gamma = loss, gamma
            rules['uniform_grid'] = torch.full_like(b, best_gamma)
            rules['ordinary'] = torch.ones_like(b)
            # Exact per-neuron logit movements retain simultaneous layer changes.
            full = [a+d for a, d in zip(pd, delta)]
            before = (xt @ pd[0].T+pd[1]).relu()[:, :, None]*pd[2][None]
            after = (xt @ full[0].T+full[1]).relu()[:, :, None]*full[2][None]
            moves = (after-before)/math.sqrt(task['width'])
            residual = zt.softmax(-1)-yt if ce else zt-yt
            work = -(residual[:, None, :]*moves).sum(-1).mean(0)
            total = moves.sum(1)
            total_energy = float(total.square().sum(-1).mean())
            residual_energy = float(residual.square().sum(-1).mean())
            progress = lt-float(loss_rows(zt+total, yt, ce).mean())
            cosine = max(float(work.sum()), 0)**2/(residual_energy*total_energy)
            q_uniform = float(objective(b, gram, rules['uniform']))
            q_neuron = float(objective(b, gram, rules['neuron']))
            assert q_neuron <= q_uniform + 1e-11
            scale = b.abs().sum()+gram.abs().sum()
            for arm, alpha in rules.items():
                moved = apply(pd, delta, alpha[0])
                row = dict(dataset='mnist', loss_kind=task['loss'], optimizer=task['optimizer'],
                           width=task['width'], seed=task['seed'], lr=task['lr'],
                           target_loss=state['target_loss'], completed=state['completed'],
                           achieved_loss=state['achieved_loss'], source=source, fraction=fraction,
                           arm=arm, calibration_before=lc, loss=lt,
                           calibration_after=float(loss_rows(scores(moved, xc), yc, ce).mean()),
                           heldout_after=float(loss_rows(scores(moved, xt), yt, ce).mean()),
                           predicted_calibration_change=float(objective(b, gram, alpha)),
                           predicted_heldout_change=float(objective(bt, gt, alpha)),
                           coefficient_mean=float(alpha.mean()),
                           qp_relative_gap=float(cert['gap'].max()/scale.clamp_min(1e-300)),
                           group_relative_gap=float(cert['group_gap'].max()/scale.clamp_min(1e-300)),
                           oracle_allocation_gain=q_uniform-q_neuron,
                           positive_work=float(work.clamp_min(0).sum()),
                           negative_work=float((-work).clamp_min(0).sum()),
                           work=float(work.sum()), curvature=float(work.sum())-progress,
                           corrective_cosine=cosine, progress=progress)
                rows.append(row)
    return rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sources', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = prepare_data(ROOT/'data/mnist')
    cal = torch.randperm(len(data['xtrain']), generator=torch.Generator().manual_seed(78191))[:512]
    test = torch.randperm(len(data['xtest']), generator=torch.Generator().manual_seed(78192))[:1024]
    paths = [p for p in sorted((ROOT/'runs/mnist/matched').glob('*/target_*.pt'))
             if '_w16_' in str(p) or '_w128_' in str(p)]
    assert len(paths) == 119
    if args.limit:
        paths = paths[:args.limit]
    output = ROOT/'runs/allocation_mnist'/('pilot' if args.limit else 'full')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for i, path in enumerate(paths):
        destination = output/(path.parent.name+'_'+path.stem+'.csv.gz')
        if not destination.exists():
            pd.DataFrame(one(path, data, cal, test, args.sources)).to_csv(
                destination, index=False, float_format='%.17g')
        print(f'{i+1}/{len(paths)} {path.parent.name} {path.stem} '
              f'{time.monotonic()-started:.1f}s', flush=True)
    provenance = dict(checkpoints=len(paths), sources=args.sources, calibration_size=len(cal),
                      heldout_size=len(test), calibration_ids=cal.tolist(), test_ids=test.tolist(),
                      source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [
                          Path(__file__), ROOT/'src/neural_competition/allocation.py',
                          ROOT/'src/neural_competition/mnist.py']},
                      checkpoint_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
                      data_sha256=digest(ROOT/'data/mnist/prepared.pt'),
                      elapsed_seconds=time.monotonic()-started, torch_version=str(torch.__version__))
    (output/'manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
