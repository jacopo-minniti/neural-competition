"""Held-out forecasts of task damage from early gate spans on bounded clusters.

Three Gauss nodes per nuisance coordinate train one fully learned ReLU network.
Nine nodes per coordinate independently audit integration and generalization.
Targets have an explicit four-neuron representation on the entire support.
All early forecasts are written before any continuation to final time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from explore_transported_gates import initialize, losses, train


def task(nodes, spread, q):
    grid, weights = np.polynomial.legendre.leggauss(nodes)
    inputs, masses, targets = [], [], []
    for context, pc in [(-2, 1-q), (2, q)]:
        for attribute in (-1, 1):
            for i, u in enumerate(grid):
                for j, v in enumerate(grid):
                    c, z = context+spread*u, attribute+spread*v
                    inputs.append([c, z, 1.])
                    masses.append(pc*.5*weights[i]*weights[j]/4)
                    targets.append(2*z+2*(context>0)*z)
    return np.array(inputs), np.array(masses), np.array(targets)


def linear_span(w, x, mass, y, evaluation=None):
    gates = (w@x.T)>0
    f = (gates[..., None]*x[None, None]).transpose(0, 2, 1, 3)
    f = f.reshape(len(w), len(x), -1)
    f = np.concatenate([np.ones((len(w),len(x),1)), f], axis=2)
    weighted = f*np.sqrt(mass)[None, :, None]
    # Truncated SVD tolerance is numerical rank tolerance only.
    coef = np.linalg.pinv(weighted, rcond=1e-10) @ (np.sqrt(mass)*y)
    pred = np.einsum('sni,si->sn', f, coef)
    floor = .5*np.sum(mass*(pred-y)**2, axis=1)
    if evaluation is None:
        return floor
    xe, pe, ye = evaluation
    ge = (w@xe.T)>0
    fe = (ge[..., None]*xe[None, None]).transpose(0, 2, 1, 3)
    fe = fe.reshape(len(w), len(xe), -1)
    fe = np.concatenate([np.ones((len(w),len(xe),1)), fe], axis=2)
    prediction = np.einsum('sni,si->sn', fe, coef)
    # This is extrapolation of the coarse-grid fitted function, not a lower bound.
    return floor, .5*np.sum(pe*(prediction-ye)**2, axis=1)


CASES = [('discrete', 1, 0., .1), ('clusters', 3, .15, .1),
         ('rare_clusters', 3, .15, .02)]
ROOT = Path(__file__).resolve().parents[2]/'docs/evidence/option_transport_resilience'


def forecast():
    raw, rows = {}, []
    for name, nodes, spread, q in CASES:
        x, p, y = task(nodes, spread, q)
        evaluate = task(9, spread, q)
        # Exact representation: 2z + ReLU(c+z)-ReLU(c-z).
        witness_w = np.array([[0,1,0],[0,-1,0],[1,1,0],[1,-1,0.]])
        witness_a = np.array([8.,-8.,4.,-4.])
        witness = (witness_a[:,None]*np.maximum(witness_w@evaluate[0].T,0)).mean(0)
        assert np.allclose(witness, evaluate[2])
        init = initialize(661902, 96, 16)
        for m in (4, 8, 16):
            state = tuple(v[:, :m].copy() if v.ndim>1 else v.copy() for v in init)
            previous = 0
            for t in (0,20,100):
                state = train(state,x,p,y,t-previous)
                floor, extrapolated = linear_span(state[0],x,p,y,evaluate)
                key=f'{name}_m{m}_t{t}'
                for label,val in zip(('w','a','c'),state):
                    raw[f'{key}_{label}']=val
                raw[f'{key}_floor']=floor
                raw[f'{key}_dense_extrapolation']=extrapolated
                rows.append(dict(case=name,width=m,time=t,mean_floor=float(floor.mean()),
                    dense_extrapolation=float(extrapolated.mean())))
                previous=t
    np.savez_compressed(ROOT/'stable_distinctions_checkpoints.npz',**raw)
    result=dict(protocol=dict(seed=661902,seeds=96,dt=.05,checkpoints=[0,20,100],
        endpoint=2000,cases=CASES,forecast_status='Saved before final continuations.'),
        forecasts=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'stable_distinctions_forecasts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def validate():
    result=json.loads((ROOT/'stable_distinctions_forecasts.json').read_text())
    assert result['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw=np.load(ROOT/'stable_distinctions_checkpoints.npz')
    rows, endpoints=[],{}
    for name,nodes,spread,q in CASES:
        x,p,y=task(nodes,spread,q)
        evaluate=task(9,spread,q)
        for m in (4,8,16):
            key=f'{name}_m{m}_t100'
            state=tuple(raw[f'{key}_{s}'] for s in ('w','a','c'))
            end=train(state,x,p,y,1900)
            loss=losses(end,x,p,y)
            dense=losses(end,*evaluate)
            final_floor=linear_span(end[0],x,p,y)
            row=dict(case=name,width=m,mean_loss=float(loss.mean()),
                dense_loss=float(dense.mean()),se=float(dense.std(ddof=1)/np.sqrt(len(dense))),
                final_floor=float(final_floor.mean()),forecasts={})
            for t in (0,20,100):
                pred=raw[f'{name}_m{m}_t{t}_floor']
                row['forecasts'][t]=dict(mean=float(pred.mean()),
                    mae=float(abs(pred-loss).mean()),
                    bias=float((pred-loss).mean()))
            rows.append(row)
            endpoints[f'{name}_m{m}_loss']=loss
            endpoints[f'{name}_m{m}_dense_loss']=dense
            print(json.dumps(row),flush=True)
    result['validation']=rows
    (ROOT/'stable_distinctions_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/'stable_distinctions_endpoints.npz',**endpoints)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['forecast','validate'])
    args=parser.parse_args()
    forecast() if args.phase=='forecast' else validate()
