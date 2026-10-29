"""Forecast ordinary-width failures from a fluctuating shared early residual field.

Only short pilot training at width 32 is used. Forecasts precede held-out
width-4/6/8/12 endpoints. This explicitly tests the conditional-independence
closure rather than assuming it succeeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task
from explore_transported_gates import gate_floor, initialize, losses, train

ROOT=Path(__file__).resolve().parents[2]/'docs/theory'
CASES=[('discrete',1,0.),('clusters',3,.15)]


def residual(state,x,y):
    w,a,c=state
    return c[:,None]+(a[...,None]*np.maximum(w@x.T,0)).mean(1)-y


def forecast():
    rng=np.random.default_rng(281026)
    rows,raw=[],{}
    for name,nodes,spread in CASES:
        x,p,y=task(nodes,spread,.1)
        pilot=initialize(281001,32,32)
        reference=initialize(281002,1,4096)
        fields,means=[],[]
        for _ in range(400):
            fields.append(residual(pilot,x,y))
            means.append(residual(reference,x,y)[0])
            pilot=train(pilot,x,p,y,.05)
            reference=train(reference,x,p,y,.05)
        fields,means=np.array(fields),np.array(means)
        fluctuations=fields-fields.mean(1,keepdims=True)
        # A finite Gaussian process with the pilot covariance, not a fitted endpoint law.
        mix=rng.normal(size=(24,32))/np.sqrt(31)
        gaussian=np.einsum('ds,tsn->tdn',mix,fluctuations)
        raw[f'{name}_pilot_residual']=fields
        raw[f'{name}_reference_residual']=means
        for m in (4,6,8,12):
            bank=initialize(281003,24,512)
            w,a,_=bank
            for t in range(400):
                forcing=means[t]+np.sqrt(32/m)*gaussian[t]
                pre=w@x.T
                act=np.maximum(pre,0)
                da=-(act*(p*forcing)[:,None]).sum(2)
                dw=-a[...,None]*(((pre>0)*(p*forcing)[:,None])@x)
                a+=.05*da
                w+=.05*dw
            ids=rng.integers(512,size=(24,128,m))
            sampled=np.stack([w[d,ids[d]] for d in range(24)]).reshape(-1,m,3)
            costs=gate_floor(sampled,x,p,y).reshape(24,128)
            ids0=rng.integers(4096,size=(3072,m))
            iid=gate_floor(reference[0][0,ids0],x,p,y)
            positive=np.all(w@x[[i for i in range(len(x)) if x[i,1]>0]].T<=0,axis=2).mean(1)
            negative=np.all(w@x[[i for i in range(len(x)) if x[i,1]<0]].T<=0,axis=2).mean(1)
            row=dict(case=name,width=m,shared_field_forecast=float(costs.mean()),
                driver_se=float(costs.mean(1).std(ddof=1)/np.sqrt(24)),
                deterministic_field_forecast=float(iid.mean()),
                shared_blind_probability=float(np.mean(positive**m+negative**m)),
                iid_blind_probability=float(positive.mean()**m+negative.mean()**m))
            rows.append(row)
            print(json.dumps(row),flush=True)
    result=dict(protocol=dict(pilot_width=32,pilot_seeds=32,reference_width=4096,
        horizon=20,dt=.05,drivers=24,bank_per_driver=512,draws_per_driver=128,
        heldout_seed=281099,heldout_seeds=96,endpoint=2000),forecasts=rows,
        status='Saved before held-out fixed-width training endpoints.')
    (ROOT/'shared_pressure_forecasts.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/'shared_pressure_pilot.npz',**raw)


def validate():
    result=json.loads((ROOT/'shared_pressure_forecasts.json').read_text())
    rows,raw=[],{}
    for name,nodes,spread in CASES:
        x,p,y=task(nodes,spread,.1)
        initial=initialize(281099,96,12)
        for m in (4,6,8,12):
            state=tuple(v[:,:m].copy() if v.ndim>1 else v.copy() for v in initial)
            initial_floor=gate_floor(state[0],x,p,y)
            early=train(state,x,p,y,20)
            early_floor=gate_floor(early[0],x,p,y)
            end=train(early,x,p,y,1980)
            loss=losses(end,x,p,y)
            row=dict(case=name,width=m,mean_loss=float(loss.mean()),
                se=float(loss.std(ddof=1)/np.sqrt(len(loss))),
                early_floor=float(early_floor.mean()),initial_floor=float(initial_floor.mean()))
            rows.append(row)
            raw[f'{name}_m{m}_loss']=loss
            raw[f'{name}_m{m}_early_floor']=early_floor
            print(json.dumps(row),flush=True)
    result['validation']=rows
    (ROOT/'shared_pressure_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/'shared_pressure_validation.npz',**raw)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['forecast','validate'])
    args=parser.parse_args()
    forecast() if args.phase=='forecast' else validate()
