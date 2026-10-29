"""Prospective transfer of the checkpoint renewal law across task geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task
from explore_transported_gates import losses, train

ROOT=Path(__file__).resolve().parents[2]/'docs/theory'


def forecast():
    previous=json.loads((ROOT/'option_renewal_forecasts.json').read_text())
    raw=np.load(ROOT/'conditional_options_trajectories.npz')
    selected=previous['selected_seed_index']
    original=tuple(raw[f'discrete_m4_t100_{s}'][selected:selected+1].copy() for s in ('w','a','c'))
    rng=np.random.default_rng(241030)
    rows,arrays=[],{}
    for separation in (1.,4.):
        x,p,y=task(1,0.,.1)
        x[:,0]*=separation/2
        w,a,c=(s.copy() for s in original)
        w[:,:,0]*=2/separation
        rho=(2-separation**2)/(2+separation**2)
        miss=.25+np.arcsin(rho)/(2*np.pi)
        for k in (1,2):
            fresh=rng.normal(size=(128,k,3))
            scale=np.sqrt((4+k)/4)
            state=(np.concatenate([np.repeat(scale*w,128,axis=0),fresh],axis=1),
                   np.concatenate([np.repeat(scale*a,128,axis=0),np.zeros((128,k))],axis=1),np.repeat(c,128))
            assert np.allclose(losses(state,x,p,y),.09)
            # Both relevant examples have the same positive attribute; old gates are off on both.
            dead=np.all((fresh@x[[1,3]].T)<=0,axis=(1,2))
            key=f'c{int(separation)}_k{k}'
            for label,value in zip(('w','a','c'),state):
                arrays[f'{key}_{label}']=value
            arrays[f'{key}_linear_loss']=.09*dead
            rows.append(dict(separation=separation,added=k,miss_probability=float(miss),
                predicted_mean=.09*float(miss**k),realized_linear_mean=float(.09*dead.mean())))
    np.savez_compressed(ROOT/'option_renewal_geometry_checkpoints.npz',**arrays)
    result=dict(seed=241030,draws_per_setting=128,forecasts=rows,
                status='Saved before new geometry continuations.')
    (ROOT/'option_renewal_geometry_forecasts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def validate():
    result=json.loads((ROOT/'option_renewal_geometry_forecasts.json').read_text())
    raw=np.load(ROOT/'option_renewal_geometry_checkpoints.npz')
    rows,endpoints=[],{}
    for separation in (1.,4.):
        x,p,y=task(1,0.,.1)
        x[:,0]*=separation/2
        for k in (1,2):
            key=f'c{int(separation)}_k{k}'
            state=tuple(raw[f'{key}_{s}'] for s in ('w','a','c'))
            end=train(state,x,p,y,1900)
            loss=losses(end,x,p,y)
            linear=raw[f'{key}_linear_loss']
            endpoints[f'{key}_loss']=loss
            row=dict(separation=separation,added=k,mean_loss=float(loss.mean()),
                se=float(loss.std(ddof=1)/np.sqrt(len(loss))),mae_from_linear=float(abs(loss-linear).mean()),
                repair_disagreements=int(np.sum((loss>1e-6)!=(linear>1e-6))))
            rows.append(row)
            print(json.dumps(row),flush=True)
    result['validation']=rows
    (ROOT/'option_renewal_geometry_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/'option_renewal_geometry_endpoints.npz',**endpoints)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['forecast','validate'])
    args=parser.parse_args()
    forecast() if args.phase=='forecast' else validate()
