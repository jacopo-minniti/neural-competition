"""Task-derived specialization basins and ordinary fixed-width loss forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task
from explore_transported_gates import gate_floor, initialize, losses, train

ROOT=Path(__file__).resolve().parents[2]/'docs/theory'


def forecast():
    x,p,y=task(1,0.,.1)
    state=initialize(281110,1,32768)
    w,a,_=state
    w,a=w[0],a[0]
    initial_w=w.copy()
    norm=np.sqrt(np.sum(w*w,1)+a*a)
    w/=norm[:,None]
    a/=norm
    snapshots={}
    for step in range(3200):
        force=((w@x.T>0)*(p*y))@x
        da=np.sum(w*force,axis=1)
        dw=a[:,None]*force
        a+=.02*da
        w+=.02*dw
        # Homogeneity makes this identical to normalizing the unbounded forced trajectory.
        norm=np.sqrt(np.sum(w*w,1)+a*a)
        w/=norm[:,None]
        a/=norm
        if step+1 in (800,1600,3200):
            snapshots[(step+1)*.02]=w.copy()
    signatures=np.sum((w@x.T>0)*(1<<np.arange(4)),axis=1)
    probabilities=np.bincount(signatures,minlength=16)/len(w)
    rng=np.random.default_rng(281111)
    rows=[]
    for m in (4,8,16):
        ids=rng.integers(len(w),size=(8192,m))
        costs=gate_floor(w[ids],x,p,y)
        before=gate_floor(initial_w[ids],x,p,y)
        rows.append(dict(width=m,predicted_mean_loss=float(costs.mean()),
                         mc_se=float(costs.std(ddof=1)/np.sqrt(len(costs))),
                         initial_floor=float(before.mean())))
    result=dict(protocol=dict(bank=32768,dt=.02,alignment_time=64,heldout_seed=281199,
        heldout_seeds=96,scales=[.2,.02,.002],widths=[4,8,16],endpoint=2000),
        gate_probabilities=probabilities.tolist(),forecasts=rows,
        convergence={str(t):float(np.mean((snapshots[t]@x.T>0)!=(w@x.T>0))) for t in snapshots},
        status='Saved before held-out finite-network training.')
    (ROOT/'alignment_extinction_forecasts.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/'alignment_extinction_bank.npz',w=w,a=a,initial_w=initial_w)
    print(json.dumps(result,indent=2),flush=True)


def validate():
    result=json.loads((ROOT/'alignment_extinction_forecasts.json').read_text())
    x,p,y=task(1,0.,.1)
    rows,raw=[],{}
    initial=initialize(281199,96,16)
    for scale in (.2,.02,.002):
        for m in (4,8,16):
            state=(initial[0][:,:m].copy()*scale/.2,initial[1][:,:m].copy()*scale/.2,initial[2].copy())
            end=train(state,x,p,y,2000)
            loss=losses(end,x,p,y)
            row=dict(scale=scale,width=m,mean_loss=float(loss.mean()),se=float(loss.std(ddof=1)/np.sqrt(len(loss))))
            rows.append(row)
            raw[f's{scale}_m{m}_loss']=loss
            for label,value in zip(('w','a','c'),end):
                raw[f's{scale}_m{m}_{label}']=value
            print(json.dumps(row),flush=True)
    result['validation']=rows
    (ROOT/'alignment_extinction_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/'alignment_extinction_validation.npz',**raw)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['forecast','validate'])
    args=parser.parse_args()
    forecast() if args.phase=='forecast' else validate()
