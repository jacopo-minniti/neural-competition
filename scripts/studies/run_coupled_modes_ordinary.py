#!/usr/bin/env python3
"""Ordinary-SGD metric follow-up to the normalized MNIST coupling probe."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch

from neural_competition.coupled_modes import fit_modes, matrix_decay, ordinary_metric
from neural_competition.mnist import gradients, loss_rows, prepare_data, scores

ROOT=Path(__file__).resolve().parents[2]


def loss(p,x,y):
    return float(loss_rows(scores(p,x),y,False).mean())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=int,default=429)
    parser.add_argument('--collection',default='ordinary4096')
    parser.add_argument('--tangent',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    folder=ROOT/'runs/coupled_modes'/args.collection
    folder.mkdir(parents=True,exist_ok=True)
    data=prepare_data(ROOT/'data/mnist')
    rng=torch.Generator().manual_seed(231023)
    ids=torch.randperm(len(data['xtrain']),generator=rng)[:4096]
    vi=torch.randperm(len(data['xval']),generator=rng)[:1024]
    x,y=data['xtrain'][ids].double(),data['ytrain'][ids].double()
    xv,yv=data['xval'][vi].double(),data['yval'][vi].double()
    rows=[]
    for seed in range(5):
        for width in [16,128]:
            path=ROOT/f'runs/mnist/main/mse_sgd_w{width}_s{seed}/checkpoint_{args.checkpoint}.pt'
            p=[v.double() for v in torch.load(path,weights_only=True)['parameters']]
            phi=(x@p[0].T+p[1]).relu()/math.sqrt(width)
            phiv=(xv@p[0].T+p[1]).relu()/math.sqrt(width)
            model=fit_modes(phi,y,p[2])
            ordinary=ordinary_metric(model,phi,y,p[2])
            active=model['active']
            predictions={}
            for step in [1,10,100]:
                train,val=[],[]
                for matrix in [ordinary['a'],ordinary['b']]:
                    q=matrix_decay(matrix,ordinary['rate'],step,ordinary['q0'])
                    a=p[2].clone()
                    a[active]+=ordinary['inverse_sqrt']@(q-ordinary['q0'])/model['scale'][:,None]
                    train.append(float(.5*(phi@a-y).square().sum(1).mean()))
                    val.append(float(.5*(phiv@a-yv).square().sum(1).mean()))
                predictions[step]=dict(predicted_diagonal=train[0],predicted_full=train[1],
                                       predicted_damage=train[0]-train[1],
                                       predicted_validation_damage=val[0]-val[1])
            if args.tangent:
                gate=(x@p[0].T+p[1])>0
                gatev=(xv@p[0].T+p[1])>0
                base=phi@p[2]
                basev=phiv@p[2]
                delta=[[torch.zeros_like(a) for a in p] for _ in range(2)]
                for step in range(1,101):
                    for j in range(2):
                        dp=delta[j]
                        z=base+phi@dp[2]+((x@dp[0].T+dp[1])*gate)@p[2]/math.sqrt(width)
                        e=(z-y)/len(x)
                        back=(e@p[2].T)*gate/math.sqrt(width)
                        grads=[back.T@x,back.sum(0),phi.T@e]
                        if j==1:
                            grads[2]=ordinary['metric']@grads[2]
                        delta[j]=[a-ordinary['rate']*b for a,b in zip(dp,grads)]
                    if step in predictions:
                        lt,lv=[],[]
                        for dp in delta:
                            z=base+phi@dp[2]+((x@dp[0].T+dp[1])*gate)@p[2]/math.sqrt(width)
                            zv=basev+phiv@dp[2]+((xv@dp[0].T+dp[1])*gatev)@p[2]/math.sqrt(width)
                            lt.append(float(.5*(z-y).square().sum(1).mean()))
                            lv.append(float(.5*(zv-yv).square().sum(1).mean()))
                        predictions[step].update(tangent_damage=lt[0]-lt[1],
                            tangent_validation_damage=lv[0]-lv[1],
                            tangent_diagonal=lt[0],tangent_full=lt[1])
            (folder/f'forecast_w{width}_s{seed}.json').write_text(json.dumps(predictions,indent=2))
            branches=[[v.clone() for v in p] for _ in range(2)]
            frozen=[p[2].clone() for _ in range(2)]
            for step in range(1,101):
                for j in range(2):
                    g=gradients(branches[j],x,y,False)
                    if j==1:
                        g[2]=ordinary['metric']@g[2]
                    branches[j]=[a-ordinary['rate']*b for a,b in zip(branches[j],g)]
                    gf=phi.T@(phi@frozen[j]-y)/len(x)
                    frozen[j]-=ordinary['rate']*(gf if j==0 else ordinary['metric']@gf)
                if step in predictions:
                    actual=[loss(q,x,y) for q in branches]
                    validation=[loss(q,xv,yv) for q in branches]
                    fixed=[float(.5*(phi@a-y).square().sum(1).mean()) for a in frozen]
                    rows.append(dict(seed=seed,width=width,checkpoint=args.checkpoint,
                        steps=step,time=step*ordinary['rate'],
                        initial_loss=loss(p,x,y),profile_floor=float(model['floor']),
                        **predictions[step],observed_diagonal=actual[0],observed_full=actual[1],
                        observed_damage=actual[0]-actual[1],
                        validation_damage=validation[0]-validation[1],
                        validation_diagonal=validation[0],validation_full=validation[1],
                        frozen_diagonal=fixed[0],frozen_full=fixed[1]))
            torch.save(dict(branches=branches,train_ids=ids,val_ids=vi),
                       folder/f'final_w{width}_s{seed}.pt')
            print(seed,width,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(folder/'results.csv',index=False)
    sources=[Path(__file__),ROOT/'src/neural_competition/coupled_modes.py',
             ROOT/'src/neural_competition/mnist.py',ROOT/'docs/theory/coupled_modes_protocol.md']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (folder/'sources').mkdir(exist_ok=True)
    for path in sources:
        (folder/'sources'/(hashes[str(path.relative_to(ROOT))]+'.txt')).write_bytes(path.read_bytes())
    (folder/'manifest.json').write_text(json.dumps(dict(sources=hashes,args=vars(args)),indent=2))


if __name__=='__main__':
    main()
