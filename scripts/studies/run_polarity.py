#!/usr/bin/env python3
"""Test mean and covariance interference with an ordinary learned output bias."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from run_residual_growth import dataset

from neural_competition.residual_growth import gaussian_relu_gram
from neural_competition.specialization import advance
from neural_competition.trajectory import components

ROOT = Path(__file__).resolve().parents[2]


def run(degrees, ratio, optimizer, args):
    start = time.monotonic()
    tag = f'd{args.data_seed}_a{degrees}_r{ratio:g}_{optimizer}'
    folder = ROOT/'runs/polarity'/args.collection
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/f'{tag}.json').exists():
        print('SKIP', tag, flush=True)
        return
    data = dataset(degrees, ratio, args.data_seed)
    modes = ['no_bias', 'free_bias', 'centered_bias']
    models = [dict(seed=s, width=w, mode=mode) for s in range(args.seeds)
              for w in [8, 16, 32] for mode in modes]
    n = len(models)
    p = torch.zeros(n, 33, 22, dtype=torch.float64)
    norm = torch.tensor([math.sqrt(m['width']) for m in models], dtype=torch.float64)[:, None]
    kind = ['sgd', 'momentum', 'adam'].index(optimizer)
    rates = torch.ones(n, 33, 22, dtype=torch.float64)
    for i, model in enumerate(models):
        seed, width = model['seed'], model['width']
        rng = torch.Generator().manual_seed(93000+seed)
        p[i, :width, :20] = torch.randn(width, 20, generator=rng, dtype=torch.float64)/math.sqrt(20)
        rng = torch.Generator().manual_seed(94000+seed)
        p[i, :width, -1] = args.out_init*torch.randn(width, generator=rng, dtype=torch.float64)
        if args.balanced_signs:
            signs = torch.ones(width, dtype=torch.float64)
            signs[:width//2] = -1
            permutation = torch.randperm(width, generator=torch.Generator().manual_seed(96000+seed))
            p[i, :width, -1] = p[i, :width, -1].abs()*signs[permutation]
        rates[i, width:] = 0
        # A constant feature encodes a scalar output bias. Its incoming
        # parameters are frozen; the readout rate gives an ordinary bias step.
        if model['mode'] != 'no_bias':
            p[i, 32, -2] = 1.
            rates[i, 32, -1] = math.sqrt(width) if kind == 2 else width
            if model['mode'] == 'centered_bias':
                p[i, 32, -1] = norm[i, 0]*data['ytrain'].mean()
    first, second = torch.zeros_like(p), torch.zeros_like(p)
    seeds = torch.tensor([m['seed'] for m in models])
    rngs = [torch.Generator().manual_seed(95000+s) for s in range(args.seeds)]
    steps = sorted(t for t in set([0, 1, 10, 30, 100, 300, 1000, 3000, 10000,
                                    30000, args.steps]) if t <= args.steps)
    mu = torch.full((8,), 1/math.sqrt(2*math.pi), dtype=torch.float64)
    gram = gaussian_relu_gram(data['teacher'], data['teacher'])
    raw_evidence = data['beta'].sign()*(gram @ data['beta'])
    centered_evidence = data['beta'].sign()*((gram-mu[:, None]*mu[None, :]) @ data['beta'])
    lr = dict(sgd=.3, momentum=.03, adam=.003)[optimizer]*args.rate_factor
    rows, last = [], 0
    for step in steps:
        while last < step:
            count = min(1000, step-last)
            ids = torch.stack([torch.randint(4096, (count, 128), generator=g) for g in rngs], 1)
            p, first, second = advance(p, first, second, data['xtrain'], data['ytrain'],
                                       ids, seeds, norm, rates, lr, kind, args.train_hidden_bias, last)
            last += count
        predictions = {v:components(p, data['x'+v][None].expand(n, -1, -1), norm).sum(-1)
                       for v in ['train', 'test']}
        for i, model in enumerate(models):
            w, a = p[i, :model['width'], :20], p[i, :model['width'], -1]
            row = dict(degrees=degrees, ratio=ratio, optimizer=optimizer, completed=step,
                       data_seed=args.data_seed, **model,
                       positive_outputs=int((a > 0).sum()),
                       output_bias=float(p[i, 32, -1]/norm[i, 0]),
                       raw_min_evidence=float(raw_evidence.min()),
                       centered_min_evidence=float(centered_evidence.min()),
                       positive_mass=float((a*w.norm(dim=1)/norm[i, 0]).clamp_min(0).sum()))
            for view in ['train', 'test']:
                row[view+'_loss'] = float(.5*(predictions[view][i]-data['y'+view]).square().mean())
            rows.append(row)
    pd.DataFrame(rows).to_csv(folder/f'{tag}.csv.gz', index=False, float_format='%.17g')
    torch.save(dict(parameters=p, first=first, second=second, models=models, data=data), folder/f'{tag}.pt')
    sources = [Path(__file__), ROOT/'scripts/studies/run_residual_growth.py',
               ROOT/'src/neural_competition/residual_growth.py', ROOT/'src/neural_competition/specialization.py',
               ROOT/'src/neural_competition/trajectory.py', ROOT/'docs/theory/polarity_followup_protocol.md']
    manifest = dict(args=vars(args), degrees=degrees, ratio=ratio, optimizer=optimizer,
                    elapsed_seconds=time.monotonic()-start,
                    sources={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest()
                             for s in sources})
    (folder/f'{tag}.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(tag, 'DONE', time.monotonic()-start, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degrees', type=int, nargs='+', default=[30, 60])
    parser.add_argument('--ratios', type=float, nargs='+', default=[-3])
    parser.add_argument('--optimizer', nargs='+', default=['sgd', 'momentum', 'adam'])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--data-seed', type=int, default=9027)
    parser.add_argument('--rate-factor', type=float, default=1.)
    parser.add_argument('--out-init', type=float, default=.1)
    parser.add_argument('--balanced-signs', action='store_true')
    parser.add_argument('--train-hidden-bias', action='store_true')
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
