#!/usr/bin/env python3
"""Prospective correlated-feature specialization experiment; both layers train."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from neural_competition.specialization import advance, resolved_features
from neural_competition.trajectory import components

ROOT = Path(__file__).resolve().parents[2]
ARMS = ['crowded8','balanced8','split16','clean8','clone16',
        'random8','random16_fixed','random16_mapped']
RATES = {'sgd': .3, 'momentum': .03, 'adam': .003}


def dataset(degrees, data_seed):
    rng = torch.Generator().manual_seed(data_seed)
    basis = torch.linalg.qr(torch.randn(20,8,generator=rng,dtype=torch.float64)).Q.T
    centers, contrasts = basis[::2], basis[1::2]
    angle = math.radians(degrees)
    teacher = torch.stack([math.cos(angle)*centers+math.sin(angle)*contrasts,
                           math.cos(angle)*centers-math.sin(angle)*contrasts],dim=1).flatten(0,1)
    x = torch.randn(8192,20,generator=rng,dtype=torch.float64)
    raw = (x @ teacher.T).relu().sum(1)/math.sqrt(8)
    y = raw/raw[:4096].std()
    phi = (x @ centers.T).relu()
    beta = torch.linalg.lstsq(phi[:4096],y[:4096],driver='gelsd',rcond=1e-12).solution
    coarse = phi @ beta
    return dict(xtrain=x[:4096],ytrain=y[:4096],xtest=x[4096:],ytest=y[4096:],
                centers=centers,contrasts=contrasts,teacher=teacher,coarse_beta=beta,
                coarse_train_loss=float(.5*(coarse[:4096]-y[:4096]).square().mean()),
                coarse_test_loss=float(.5*(coarse[4096:]-y[4096:]).square().mean()))


def initialize(data, seed, arm, jitter):
    width = 16 if '16' in arm else 8
    rng = torch.Generator().manual_seed(83000+seed)
    p = torch.zeros(width,22,dtype=torch.float64)
    if arm.startswith('random'):
        # Separate streams make the first eight incoming AND outgoing values
        # identical across widths before width normalization.
        p[:,:20] = torch.randn(width,20,generator=rng,dtype=torch.float64)/math.sqrt(20)
        out = torch.Generator().manual_seed(84000+seed)
        p[:,-1] = .1*torch.randn(width,generator=out,dtype=torch.float64)
        return p
    counts = [2,2,2,2] if arm == 'balanced8' else [1,1,3,3]
    order = torch.randperm(4,generator=torch.Generator().manual_seed(85000+seed)).tolist()
    parts = []
    for g,count in zip(order,counts):
        number = count*(2 if width == 16 else 1)
        local = torch.zeros(number,22,dtype=torch.float64)
        local[:,:20] = data['centers'][g]
        local[:,-1] = math.sqrt(width)*data['coarse_beta'][g]/number
        if arm not in ['clean8','clone16']:
            noise = torch.randn(number,20,generator=rng,dtype=torch.float64)/math.sqrt(20)
            noise -= noise.mean(0)
            local[:,:20] += jitter*noise
        parts.append(local)
    return torch.cat(parts)


def run(degrees, optimizer, args):
    started = time.monotonic()
    tag = f'd{args.data_seed}_a{degrees}_{optimizer}_bias{int(args.train_bias)}'
    output = ROOT/'runs/specialization'/args.collection
    output.mkdir(parents=True,exist_ok=True)
    if (output/f'{tag}.json').exists():
        print('SKIP',tag,flush=True)
        return
    data = dataset(degrees,args.data_seed)
    models = [dict(seed=seed,arm=arm,width=16 if '16' in arm else 8)
              for seed in range(args.seeds) for arm in ARMS]
    n = len(models)
    p = torch.zeros(n,16,22,dtype=torch.float64)
    for i,model in enumerate(models):
        p[i,:model['width']] = initialize(data,model['seed'],model['arm'],args.jitter)
    seeds = torch.tensor([m['seed'] for m in models])
    widths = torch.tensor([m['width'] for m in models],dtype=torch.float64)[:,None]
    norm = widths.sqrt()
    rates = torch.ones(n,1,22,dtype=torch.float64)
    kind = ['sgd','momentum','adam'].index(optimizer)
    for i,model in enumerate(models):
        if model['width'] == 16 and model['arm'] != 'random16_fixed':
            if kind == 2:
                rates[i,0,-1] = 1/math.sqrt(2)
            else:
                rates[i,0,:21] = 2
    first,second = torch.zeros_like(p),torch.zeros_like(p)
    rngs = [torch.Generator().manual_seed(86000+s) for s in range(args.seeds)]
    checkpoints = sorted(set([0,1,10,100,300,1000,3000,args.steps]))
    checkpoints = [t for t in checkpoints if t <= args.steps]
    rows = []
    last = 0
    for completed in checkpoints:
        while last < completed:
            count = min(1000,completed-last)
            indices = torch.stack([torch.randint(4096,(count,128),generator=g) for g in rngs],dim=1)
            p,first,second = advance(p,first,second,data['xtrain'],data['ytrain'],indices,
                                     seeds,norm,rates,RATES[optimizer]*args.rate_factor,kind,
                                     args.train_bias,last)
            last += count
        losses = {}
        for view in ['train','test']:
            x,y = data['x'+view],data['y'+view]
            prediction = components(p,x[None].expand(n,-1,-1),norm).sum(-1)
            losses[view] = .5*(prediction-y).square().mean(1)
        for i,model in enumerate(models):
            row = dict(data_seed=args.data_seed,degrees=degrees,optimizer=optimizer,
                       train_bias=args.train_bias,completed=completed,**model,
                       train_loss=float(losses['train'][i]),test_loss=float(losses['test'][i]),
                       coarse_test_loss=data['coarse_test_loss'],
                       resolved=resolved_features(p[i,:model['width']],data['teacher'],
                                                  math.radians(degrees),float(norm[i,0])))
            rows.append(row)
        if completed == args.steps:
            print(tag,'DONE',time.monotonic()-started,flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/f'{tag}.csv.gz',index=False,float_format='%.17g')
    torch.save(dict(parameters=p,first=first,second=second,models=models,data=data),output/f'{tag}.pt')
    protocol = ROOT/'docs/theory/specialization_protocol.md'
    source = ROOT/'src/neural_competition/specialization.py'
    provenance = dict(args=vars(args),degrees=degrees,optimizer=optimizer,models=models,
                      elapsed_seconds=time.monotonic()-started,
                      sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [Path(__file__),source,protocol,
                                            ROOT/'src/neural_competition/trajectory.py']})
    (output/f'{tag}.json').write_text(json.dumps(provenance,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degrees',type=int,nargs='+',default=[0,15,30,60])
    parser.add_argument('--optimizer',nargs='+',default=['sgd','momentum','adam'])
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--data-seed',type=int,default=8027)
    parser.add_argument('--jitter',type=float,default=.02)
    parser.add_argument('--train-bias',action='store_true')
    parser.add_argument('--rate-factor',type=float,default=1.)
    parser.add_argument('--collection',default='main')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for degrees in args.degrees:
        for optimizer in args.optimizer:
            run(degrees,optimizer,args)


if __name__ == '__main__':
    main()
