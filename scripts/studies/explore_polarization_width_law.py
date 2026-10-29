"""Prospective width law from task-induced polarization, including dynamic death.

Coarse families are stable even when individual ReLU boundary gates chatter.
No finite-network endpoint is used to estimate the family probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task
from explore_transported_gates import initialize, losses, train

ROOT=Path(__file__).resolve().parents[2]/'docs/theory'
CASES=[('base',1,0.,.1,2.),('near',1,0.,.1,1.),
       ('rare',1,0.,.02,2.),('clusters',3,.15,.1,2.)]


def data(case):
    _,nodes,spread,q,separation=case
    x,p,y=task(nodes,spread,q)
    x[:,0]*=separation/2
    return x,p,y


def labels(w,a,x):
    alive=np.any(w@x.T>0,axis=-1)
    return np.where(alive,np.where(a>0,1,-1),0)


def polarize(w,a,x,p,y,dt=.02,horizon=64.):
    w,a=w.copy(),a.copy()
    initial=labels(w,a,x)
    snapshots={}
    checkpoints={round(t/dt):t for t in (8.,16.,32.,horizon)}
    for step in range(round(horizon/dt)):
        force=((w@x.T>0)*(p*y))@x
        da=np.sum(w*force,axis=-1)
        dw=a[...,None]*force
        a+=dt*da
        w+=dt*dw
        norm=np.sqrt(np.sum(w*w,axis=-1)+a*a)
        w/=norm[...,None]
        a/=norm
        if step+1 in checkpoints:
            snapshots[checkpoints[step+1]]=labels(w,a,x)
    final=snapshots[horizon]
    return final,dict(initial_dead=float(np.mean(initial==0)),final_dead=float(np.mean(final==0)),
        new_dead=float(np.mean((initial!=0)&(final==0))),
        family_change={str(t):float(np.mean(v!=final)) for t,v in snapshots.items()})


def forecast():
    rows,raw,details=[],{},{}
    for case in CASES:
        name,nodes,spread,q,separation=case
        x,p,y=data(case)
        bank=initialize(281210,1,16384)
        family,diagnostics=polarize(bank[0][0],bank[1][0],x,p,y)
        d=float(np.mean(family==0))
        # Sign symmetry is a property of this task and isotropic initialization.
        positive=(1-d)/2
        half_damage=q*(1-q)+(1+3*q)*spread**2/3
        constant_damage=.5*(p@y**2-(p@y)**2)
        initial=initialize(281299,256,12)
        assignments,_=polarize(initial[0],initial[1],x,p,y)
        raw[f'{name}_families']=assignments
        details[name]=dict(diagnostics=diagnostics,positive_probability=positive,
                           missing_half_damage=half_damage,constant_damage=constant_damage)
        for m in (4,6,8,12):
            pos=np.any(assignments[:,:m]==1,axis=1)
            neg=np.any(assignments[:,:m]==-1,axis=1)
            predicted=np.where(pos&neg,0,np.where(pos|neg,half_damage,constant_damage))
            raw[f'{name}_m{m}_predicted']=predicted
            law=2*half_damage*((1+d)/2)**m+(constant_damage-2*half_damage)*d**m
            rows.append(dict(case=name,width=m,law=float(law),
                predicted_realized_mean=float(predicted.mean()),
                predicted_missing_fraction=float(np.mean(~(pos&neg))),
                static_dead_only_law=float(2*half_damage*((1+diagnostics['initial_dead'])/2)**m+
                    (constant_damage-2*half_damage)*diagnostics['initial_dead']**m)))
        print(json.dumps(dict(case=name,details=details[name],forecasts=rows[-4:])),flush=True)
    result=dict(protocol=dict(seed=281299,seeds=256,widths=[4,6,8,12],scale=.002,
        endpoint=2000,dt=.05,bank=16384,forced_dt=.02,forced_time=64),
        details=details,forecasts=rows,status='Saved before these held-out full-network continuations.')
    (ROOT/'polarization_width_forecasts.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/'polarization_width_predictions.npz',**raw)


def validate():
    result=json.loads((ROOT/'polarization_width_forecasts.json').read_text())
    predictions=np.load(ROOT/'polarization_width_predictions.npz')
    rows,raw=[],{}
    for case in CASES:
        name=case[0]
        x,p,y=data(case)
        initial=initialize(281299,256,12)
        for m in (4,6,8,12):
            state=(initial[0][:,:m]*.01,initial[1][:,:m]*.01,initial[2].copy())
            end=train(state,x,p,y,2000)
            loss=losses(end,x,p,y)
            predicted=predictions[f'{name}_m{m}_predicted']
            positive_half=x[:,1]>0
            gates=end[0]@x.T>0
            blind=np.all(~gates[:,:,positive_half],axis=(1,2)) | np.all(~gates[:,:,~positive_half],axis=(1,2))
            row=dict(case=name,width=m,mean_loss=float(loss.mean()),
                se=float(loss.std(ddof=1)/np.sqrt(len(loss))),
                forecast_mean=float(predicted.mean()),mae=float(abs(predicted-loss).mean()),
                blind_fraction=float(blind.mean()),
                predicted_missing_but_not_blind=int(np.sum((predicted>0)&~blind)),
                predicted_complete_but_blind=int(np.sum((predicted==0)&blind)))
            rows.append(row)
            raw[f'{name}_m{m}_loss']=loss
            raw[f'{name}_m{m}_blind']=blind
            for label,value in zip(('w','a','c'),end):
                raw[f'{name}_m{m}_{label}']=value
            print(json.dumps(row),flush=True)
            (ROOT/'polarization_width_validation.json').write_text(json.dumps(
                dict(**result,validation=rows),indent=2)+'\n')
            np.savez_compressed(ROOT/'polarization_width_validation.npz',**raw)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['forecast','validate'])
    args=parser.parse_args()
    forecast() if args.phase=='forecast' else validate()
