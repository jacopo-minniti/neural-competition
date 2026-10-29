#!/usr/bin/env python3
"""Direct test of whether preserving output sign eligibility rescues feature learning."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from neural_competition.feature_profile import profile_readout
from neural_competition.sign_survival import advance_protected
from neural_competition.tasks import paired_gaussian_task
from neural_competition.training import components

ROOT = Path(__file__).resolve().parents[2]
ARMS = ['balanced8', 'protect_early', 'protect_all', 'protect_late',
        'profile_early', 'ordinary8', 'ordinary32']


def run(args, optimizer):
    start = time.monotonic()
    folder = ROOT / 'runs/sign_survival' / args.collection
    folder.mkdir(parents=True, exist_ok=True)
    tag = f'd{args.data_seed}_a{args.degrees}_r{args.ratio:g}_{optimizer}'
    marker = folder / f'{tag}.json'
    if marker.exists():
        previous = json.loads(marker.read_text())['args']
        previous.setdefault('movement_control', False)
        assert previous == vars(args), 'Use a new collection'
        print('SKIP', tag, flush=True)
        return
    sources = [Path(__file__), ROOT/'src/neural_competition/sign_survival.py',
               ROOT/'src/neural_competition/training.py', ROOT/'src/neural_competition/tasks.py',
               ROOT/'src/neural_competition/residual_growth.py',
               ROOT/'src/neural_competition/feature_profile.py',
               ROOT/'docs/theory/sign_survival_protocol.md']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    snapshots = folder / 'sources'
    snapshots.mkdir(exist_ok=True)
    for path in sources:
        (snapshots / (hashes[str(path.relative_to(ROOT))]+'.txt')).write_bytes(path.read_bytes())
    data = paired_gaussian_task(args.degrees, args.ratio, args.data_seed)
    x, y = data['xtrain'], data['ytrain']
    models = [dict(seed=s, width=32 if arm == 'ordinary32' else 8, arm=arm)
              for s in range(args.seed_start, args.seed_start+args.seeds)
              for arm in ARMS+(['damp_early'] if args.movement_control else [])]
    n = len(models)
    p = torch.zeros(n, 32, 22, dtype=torch.float64)
    norm = torch.tensor([[math.sqrt(m['width'])] for m in models], dtype=torch.float64)
    rates = torch.ones_like(p)
    for i, model in enumerate(models):
        s, w = model['seed'], model['width']
        p[i, :w, :20] = torch.randn(w, 20, generator=torch.Generator().manual_seed(93000+s),
                                    dtype=torch.float64)/math.sqrt(20)
        a = .1*torch.randn(w, generator=torch.Generator().manual_seed(94000+s),
                          dtype=torch.float64)
        if model['arm'] not in ['ordinary8', 'ordinary32']:
            sign = torch.ones(w, dtype=torch.float64)
            sign[:w//2] = -1
            perm = torch.randperm(w, generator=torch.Generator().manual_seed(96000+s))
            a = a.abs()*sign[perm]
        p[i, :w, -1] = a
        rates[i, w:] = 0
    signs = p[:, :, -1].sign()
    first, second = torch.zeros_like(p), torch.zeros_like(p)
    seeds = torch.tensor([m['seed']-args.seed_start for m in models])
    rngs = [torch.Generator().manual_seed(95000+s)
            for s in range(args.seed_start, args.seed_start+args.seeds)]
    kind = ['sgd', 'momentum', 'adam'].index(optimizer)
    lr = [.3, .03, .003][kind]*args.rate_factor
    checkpoints = {0, 10, 30, 100, 300, 1000, 3000, args.steps-1000, args.steps}
    rows, states = [], {}
    counts = torch.zeros(n, dtype=torch.int64)
    corrections = torch.zeros(n, dtype=torch.float64)
    correction_norms = torch.zeros(n, dtype=torch.float64)
    damping_caps = torch.zeros(n, dtype=torch.int64)
    damping_sources = torch.tensor([
        next(j for j, peer in enumerate(models)
             if peer['seed']==m['seed'] and peer['arm']=='protect_early')
        if m['arm']=='damp_early' else -1 for m in models])
    failures = {}
    for step in range(0, args.steps+1, 10):
        if step in checkpoints:
            if step in {0, 1000, args.steps-1000, args.steps}:
                states[step] = dict(parameters=p.clone(), first=first.clone(), second=second.clone())
            for i, m in enumerate(models):
                w = m['width']
                finite = torch.isfinite(p[i]).all().item()
                row = dict(**m, step=step, optimizer=optimizer, degrees=args.degrees,
                           ratio=args.ratio, data_seed=args.data_seed, diverged=not finite,
                           projection_count=int(counts[i]), correction_l1=float(corrections[i]),
                           correction_l2_sum=float(correction_norms[i]),
                           damping_caps=int(damping_caps[i]),
                           positive=int((p[i, :w, -1]>0).sum()),
                           zero=int((p[i, :w, -1]==0).sum()))
                for split in ['train', 'test']:
                    pred = components(p[i:i+1], data['x'+split][None], norm[i:i+1]).sum(-1)[0]
                    row[split+'_loss'] = float(.5*(pred-data['y'+split]).square().mean())
                if finite:
                    phi = (x @ p[i, :w, :20].T).relu()/math.sqrt(w)
                    astar, ploss = profile_readout(phi, y)
                    phit = (data['xtest'] @ p[i, :w, :20].T).relu()/math.sqrt(w)
                    row['profiled_train_loss'] = float(ploss)
                    row['profiled_test_loss'] = float(.5*(phit @ astar-data['ytest']).square().mean())
                rows.append(row)
        if step == args.steps:
            break
        if step == args.steps-1000:
            for i, m in enumerate(models):
                if m['arm'] == 'protect_late':
                    signs[i] = p[i, :, -1].sign()
        if step < 1000:
            for i, m in enumerate(models):
                if m['arm'] == 'profile_early' and i not in failures:
                    phi = (x @ p[i, :8, :20].T).relu()/math.sqrt(8)
                    p[i, :8, -1] = profile_readout(phi, y)[0]
        protect = torch.tensor([m['arm']=='protect_all'
                                or (m['arm']=='protect_early' and step<1000)
                                or (m['arm']=='protect_late' and step>=args.steps-1000)
                                for m in models])
        ids = torch.stack([torch.randint(4096, (10, 128), generator=g) for g in rngs], 1)
        p, first, second, c, delta, delta_norms, caps = advance_protected(
            p, first, second, x, y, ids, seeds, norm, rates, lr, kind, step, signs, protect,
            damping_sources)
        counts += c
        corrections += delta
        correction_norms += delta_norms
        damping_caps += caps
        unstable = ~torch.isfinite(p).all((1, 2)) | (p.abs().amax((1, 2))>1e50)
        for i in torch.where(unstable)[0].tolist():
            failures.setdefault(i, step+10)
            p[i] = float('nan')
    frame = pd.DataFrame(rows)
    frame.to_csv(folder/f'{tag}.csv.gz', index=False, float_format='%.17g',
                 compression={'method':'gzip', 'mtime':0})
    torch.save(dict(states=states, models=models), folder/f'{tag}.pt')
    marker.write_text(json.dumps(dict(args=vars(args), optimizer=optimizer, sources=hashes,
                                     elapsed_seconds=time.monotonic()-start,
                                     failures=[dict(**models[i], first_detected_step=t)
                                               for i,t in failures.items()],
                                     dataset_hash=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()),
                                indent=2)+'\n')
    print(tag, round(time.monotonic()-start, 1),
          (200*frame.query('step==@args.steps').groupby('arm').test_loss.mean()).to_dict(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', default='main')
    parser.add_argument('--optimizer', nargs='+', choices=['sgd', 'momentum', 'adam'],
                        default=['sgd', 'momentum', 'adam'])
    parser.add_argument('--degrees', type=int, default=30)
    parser.add_argument('--ratio', type=float, default=-3)
    parser.add_argument('--data-seed', type=int, default=9027)
    parser.add_argument('--seed-start', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--rate-factor', type=float, default=1.)
    parser.add_argument('--movement-control', action='store_true')
    args = parser.parse_args()
    assert args.steps>=2000 and args.steps%10==0
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for optimizer in args.optimizer:
        run(args, optimizer)


if __name__ == '__main__':
    main()
