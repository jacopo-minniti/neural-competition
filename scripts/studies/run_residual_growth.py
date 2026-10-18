#!/usr/bin/env python3
"""Ordinary random-init training across predicted residual-growth sign changes."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from neural_competition.residual_growth import (
    gaussian_relu_gram,
    residual_potential,
)
from neural_competition.tasks import paired_gaussian_task
from neural_competition.training import advance, components

ROOT = Path(__file__).resolve().parents[2]


def run(degrees, ratio, optimizer, args):
    started = time.monotonic()
    tag = f'd{args.data_seed}_a{degrees}_r{ratio:g}_{optimizer}_bias{int(args.train_bias)}'
    folder = ROOT/'runs/residual_growth'/args.collection
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/f'{tag}.json').exists():
        print('SKIP', tag, flush=True)
        return
    data = paired_gaussian_task(degrees, ratio, args.data_seed)
    models = [dict(seed=seed, width=width, convention=convention)
              for seed in range(args.seeds) for width, convention in
              [(8, 'fixed'), (16, 'fixed'), (16, 'mapped'), (32, 'fixed'), (32, 'mapped')]]
    n = len(models)
    p = torch.zeros(n, 32, 22, dtype=torch.float64)
    for i, model in enumerate(models):
        seed, width = model['seed'], model['width']
        rng = torch.Generator().manual_seed(93000+seed)
        p[i, :width, :20] = torch.randn(width, 20, generator=rng, dtype=torch.float64)/math.sqrt(20)
        rng = torch.Generator().manual_seed(94000+seed)
        p[i, :width, -1] = args.out_init*torch.randn(width, generator=rng, dtype=torch.float64)
    norm = torch.tensor([math.sqrt(m['width']) for m in models], dtype=torch.float64)[:, None]
    seed_map = torch.tensor([m['seed'] for m in models])
    rates = torch.ones(n, 1, 22, dtype=torch.float64)
    kind = ['sgd', 'momentum', 'adam'].index(optimizer)
    for i, model in enumerate(models):
        if model['convention'] == 'mapped':
            if kind == 2:
                rates[i, 0, -1] = math.sqrt(8/model['width'])
            else:
                rates[i, 0, :21] = model['width']/8
    first, second = torch.zeros_like(p), torch.zeros_like(p)
    rngs = [torch.Generator().manual_seed(95000+s) for s in range(args.seeds)]
    checkpoints = sorted(t for t in set([0, 1, 10, 100, 300, 1000, 3000, 10000,
                                         30000, args.steps]) if t <= args.steps)
    rows, probes = [], []
    coarse_test = (data['xtest'] @ data['centers'].T).relu() @ data['coarse_beta']
    last = 0
    base_lr = dict(sgd=.3, momentum=.03, adam=.003)[optimizer]*args.rate_factor
    for step in checkpoints:
        while last < step:
            count = min(1000, step-last)
            indices = torch.stack([torch.randint(4096, (count, 128), generator=g) for g in rngs], 1)
            p, first, second = advance(p, first, second, data['xtrain'], data['ytrain'],
                                       indices, seed_map, norm, rates, base_lr, kind,
                                       args.train_bias, last)
            last += count
        predictions = {}
        for view in ['train', 'test']:
            predictions[view] = components(p, data['x'+view][None].expand(n, -1, -1), norm).sum(-1)
        for i, model in enumerate(models):
            row = dict(degrees=degrees, ratio=ratio, optimizer=optimizer, completed=step,
                       data_seed=args.data_seed, train_bias=args.train_bias, **model)
            for view in ['train', 'test']:
                row[view+'_loss'] = float(.5*(predictions[view][i]-data['y'+view]).square().mean())
            row['coarse_test_loss'] = data['coarse_test_loss']
            row['distance_to_coarse'] = float((predictions['test'][i]-coarse_test).square().mean().sqrt())
            row['predicted_min_growth'] = float(data['predicted_growth'].min())
            rows.append(row)
            phi_t = (data['xtest'] @ data['teacher'].T).relu()
            empirical = phi_t.T @ (data['ytest']-predictions['test'][i])/len(phi_t)
            if not args.train_bias:
                actual = residual_potential(data['teacher'], data['teacher'], data['beta'],
                                             p[i, :, :20], p[i, :, -1]/norm[i, 0])
                # Neuron potential sign: teacher-only growth versus actual residual growth.
                w = p[i, :model['width'], :20]
                u = w/w.norm(dim=1, keepdim=True).clamp_min(1e-30)
                teacher_signal = gaussian_relu_gram(u, data['teacher']) @ data['beta']
                own = (p[i, :model['width'], -1]/norm[i, 0])*w.norm(dim=1)*.5
                potential = residual_potential(u, data['teacher'], data['beta'],
                                               p[i, :, :20], p[i, :, -1]/norm[i, 0])
                signs = p[i, :model['width'], -1].sign()
                # Holding each neuron itself fixed, peers reverse its amplitude growth.
                no_peers = signs*(teacher_signal-own)
                actual_growth = signs*potential
                row['peer_reversed_fraction'] = float(((no_peers > 1e-5) &
                                                       (actual_growth < -1e-5)).double().mean())
            else:
                actual = torch.full((8,), float('nan'))
            for feature in range(8):
                probes.append(dict(**row, feature=feature,
                                   predicted_growth=float(data['predicted_growth'][feature]),
                                   actual_growth=float(actual[feature]*data['beta'][feature].sign()),
                                   empirical_growth=float(empirical[feature]*data['beta'][feature].sign())))
    pd.DataFrame(rows).to_csv(folder/f'{tag}.csv.gz', index=False, float_format='%.17g')
    pd.DataFrame(probes).to_csv(folder/f'{tag}_probes.csv.gz', index=False, float_format='%.17g')
    torch.save(dict(parameters=p, first=first, second=second, models=models, data=data),
               folder/f'{tag}.pt')
    sources = [Path(__file__), ROOT/'src/neural_competition/residual_growth.py',
               ROOT/'src/neural_competition/training.py', ROOT/'src/neural_competition/tasks.py',
               ROOT/'docs/theory/residual_growth_protocol.md']
    manifest = dict(args=vars(args), degrees=degrees, ratio=ratio, optimizer=optimizer,
                    elapsed_seconds=time.monotonic()-started,
                    sources={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest()
                             for s in sources})
    (folder/f'{tag}.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(tag, 'DONE', time.monotonic()-started, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degrees', type=int, nargs='+', default=[30, 60])
    parser.add_argument('--ratios', type=float, nargs='+', default=[1, 3, -3])
    parser.add_argument('--optimizer', nargs='+', default=['sgd', 'momentum', 'adam'])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--data-seed', type=int, default=9027)
    parser.add_argument('--train-bias', action='store_true')
    parser.add_argument('--rate-factor', type=float, default=1.)
    parser.add_argument('--out-init', type=float, default=.1)
    parser.add_argument('--collection', default='main')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for degrees in args.degrees:
        for ratio in args.ratios:
            for optimizer in args.optimizer:
                run(degrees, ratio, optimizer, args)


if __name__ == '__main__':
    main()
