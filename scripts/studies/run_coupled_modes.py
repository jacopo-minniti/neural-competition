#!/usr/bin/env python3
"""Predict and test coupled-mode loss damage on saved MNIST representations."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from neural_competition.coupled_modes import fit_modes, forecast
from neural_competition.mnist import gradients, loss_rows, prepare_data, scores

ROOT = Path(__file__).resolve().parents[2]


def loss(p, x, y):
    return float(loss_rows(scores(p, x), y, False).mean())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-size',type=int,default=256)
    parser.add_argument('--collection',default='pilot256')
    parser.add_argument('--checkpoint',type=int,default=429)
    args=parser.parse_args()
    torch.set_num_threads(2)
    start=time.monotonic()
    folder=ROOT/'runs/coupled_modes'/args.collection
    folder.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__), ROOT/'src/neural_competition/coupled_modes.py',
             ROOT/'src/neural_competition/mnist.py',ROOT/'docs/theory/coupled_modes_protocol.md']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (folder/'sources').mkdir(exist_ok=True)
    for p in sources:
        (folder/'sources'/(hashes[str(p.relative_to(ROOT))]+'.txt')).write_bytes(p.read_bytes())
    data=prepare_data(ROOT/'data/mnist')
    rng=torch.Generator().manual_seed(231023)
    train_ids=torch.randperm(len(data['xtrain']),generator=rng)[:args.train_size]
    val_ids=torch.randperm(len(data['xval']),generator=rng)[:1024]
    x,y=data['xtrain'][train_ids].double(),data['ytrain'][train_ids].double()
    xv,yv=data['xval'][val_ids].double(),data['yval'][val_ids].double()
    rows=[]
    for seed in range(5):
        for width in [16,128]:
            source=ROOT/f'runs/mnist/main/mse_sgd_w{width}_s{seed}/checkpoint_{args.checkpoint}.pt'
            ckpt=torch.load(source,weights_only=True)
            p=[v.double().clone() for v in ckpt['parameters']]
            phi=(x@p[0].T+p[1]).relu()/math.sqrt(width)
            model=fit_modes(phi,y,p[2])
            predictions=[forecast(model,t) for t in [.1,.5,2.]]
            zval=(xv@p[0].T+p[1]).relu()[:,model['active']]/math.sqrt(width)/model['scale']
            work=model['vectors'].T@(model['z'].T@(model['z']@model['b']-y)/len(x))
            for pred in predictions:
                val_losses=[]
                for full in [False,True]:
                    decay=((1-model['eta'])**pred['steps'] if full else
                           (1-model['eta']*model['eigenvalues']).pow(pred['steps']))
                    fitted=model['b']-model['vectors']@(
                        ((1-decay)/model['eigenvalues'])[:,None]*work)
                    val_losses.append(float(.5*(zval@fitted-yv).square().sum(1).mean()))
                pred['predicted_validation_damage']=val_losses[0]-val_losses[1]
            (folder/f'forecast_w{width}_s{seed}.json').write_text(json.dumps(predictions,indent=2))
            # These predictions are saved before observing either continuation.
            branches=[[v.clone() for v in p] for _ in range(2)]
            frozen=[model['b'].clone() for _ in range(2)]
            initial=loss(p,x,y)
            terminal={r['steps']:r for r in predictions}
            for step in range(1,max(terminal)+1):
                for j in range(2):
                    residual=model['z']@frozen[j]-y
                    g=model['z'].T@residual/len(x)
                    frozen[j]-=model['eta']*(g if j==0 else model['inverse']@g)
                    g_live=gradients(branches[j],x,y,False)
                    delta_a=(model['metric_diagonal'][:,None]*g_live[2] if j==0
                             else model['metric_full']@g_live[2])
                    branches[j]=[branches[j][0]-model['eta']*g_live[0],
                                 branches[j][1]-model['eta']*g_live[1],
                                 branches[j][2]-model['eta']*delta_a]
                if step in terminal:
                    pred=terminal[step]
                    fixed=[float(.5*(model['z']@b-y).square().sum(1).mean()) for b in frozen]
                    observed=[loss(q,x,y) for q in branches]
                    validation=[loss(q,xv,yv) for q in branches]
                    rows.append(dict(seed=seed,width=width,checkpoint=args.checkpoint,
                                     train_size=args.train_size,initial_loss=initial,
                                     profile_floor=float(model['floor']),**pred,
                                     frozen_diagonal=fixed[0],frozen_full=fixed[1],
                                     observed_diagonal=observed[0],observed_full=observed[1],
                                     observed_damage=observed[0]-observed[1],
                                     validation_damage=validation[0]-validation[1],
                                     validation_diagonal=validation[0],validation_full=validation[1]))
            torch.save(dict(branches=branches,train_ids=train_ids,val_ids=val_ids),
                       folder/f'final_w{width}_s{seed}.pt')
            print(seed,width,round(time.monotonic()-start,1),rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(folder/'results.csv',index=False)
    (folder/'manifest.json').write_text(json.dumps(dict(sources=hashes,args=vars(args),
        elapsed_seconds=time.monotonic()-start,torch_version=torch.__version__,
        train_ids=train_ids.tolist(),val_ids=val_ids.tolist()),indent=2))


if __name__=='__main__':
    main()
