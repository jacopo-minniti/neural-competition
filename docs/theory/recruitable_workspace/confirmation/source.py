"""Finite nonlinear correction assays, separate from response-spectrum diagnostics.

Run a pilot before a frozen fresh-seed confirmation. Output dirs never overwritten.
All losses are half MSE. Neuron lr=m*rate; affine skip lr=rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')


def task(seed, n, validation=False, affine=False):
    rng = np.random.default_rng(seed + (100000 if validation else 0))
    x = rng.uniform(-1, 1, n)
    x = np.column_stack([x, np.ones(n)])
    # Data-level local sensor responses, not neuron roles or a neural teacher.
    r = np.random.default_rng(seed + 70000)
    centers = np.array([-.60, 0., .57]) + r.uniform(-.035, .035, 3)
    widths = np.array([.25, .28, .23])
    heights = np.array([1., -.8, .7]) * r.uniform(.9, 1.1, 3)
    y = .2*x[:, 0]
    if not affine:
        for c, w, h in zip(centers, widths, heights):
            y += h*np.maximum(1-np.abs((x[:, 0]-c)/w), 0)
    return x, y


def initial(seed, m):
    rng = np.random.default_rng(seed+50000)
    bank = rng.normal(size=(36, 3))
    return bank[:m, :2].copy(), .3*bank[:m, 2].copy(), np.zeros(2)


def copy(s):
    return tuple(a.copy() for a in s)


def predict(s, x):
    w, a, c = s
    return np.maximum(x@w.T, 0)@a/len(a) + x@c


def direction(s, x, y):
    w, a, c = s
    z = x@w.T
    h = np.maximum(z, 0)
    e = h@a/len(a) + x@c-y
    # The m multiplier is already included in these neuron velocities.
    da = -(e@h)/len(x)
    dw = -((e[:, None]*(z>0)).T@x)*a[:, None]/len(x)
    dc = -(e@x)/len(x)
    power = (np.sum(dw*dw)+np.sum(da*da))/len(a) + np.sum(dc*dc)
    return (dw, da, dc), power


def evolve(s, x, y, n, rate, vx=None, vy=None, checkpoints=(), offset=None, voffset=None):
    s = copy(s)
    rows = []
    off = np.zeros(len(x)) if offset is None else offset
    voff = np.zeros(len(vx)) if vx is not None and voffset is None else voffset
    arc = 0.
    for step in range(n+1):
        if step in checkpoints:
            tr = predict(s,x)-off-y
            va = predict(s,vx)-voff-vy
            rows.append(dict(step=step, train=float(np.mean(tr**2)/2),
                             validation=float(np.mean(va**2)/2),
                             train_regions=[float(np.mean(tr[k]**2)/2) for k in regions(x)],
                             validation_regions=[float(np.mean(va[k]**2)/2) for k in regions(vx)],
                             parameter_arc=arc))
        if step == n:
            break
        vel,power = direction(s,x,y+off)
        arc += rate*np.sqrt(power)
        s = tuple(t+rate*d for t,d in zip(s,vel))
    return s, rows


def regions(x):
    return [x[:,0]<-1/3, np.abs(x[:,0])<=1/3, x[:,0]>1/3]


def expand(s, kind, seed):
    w,a,c = s
    if kind == 'clone':
        return np.tile(w,(4,1)), np.tile(a,4), c.copy()
    rng = np.random.default_rng(seed+90000)
    # m -> 4m, old homogeneous params times sqrt(4): same function AND old metric.
    nw = rng.normal(size=(3*len(a),2))
    return np.concatenate([2*w,nw]), np.concatenate([2*a,np.zeros(3*len(a))]), c.copy()


def audit():
    import torch
    x,y = task(3,37)
    s = initial(4,9)
    w,a,c = [torch.tensor(t, dtype=torch.float64,requires_grad=True) for t in s]
    xt = torch.tensor(x)
    loss = ((torch.relu(xt@w.T)@a/9+xt@c-torch.tensor(y))**2).mean()/2
    loss.backward()
    d,p = direction(s,x,y)
    errors = [np.max(np.abs(d[i]+t.grad.numpy()*(9 if i<2 else 1))) for i,t in enumerate([w,a,c])]
    assert max(errors)<1e-12
    clone=expand(s,'clone',4)
    fresh=expand(s,'fresh',4)
    z,_=evolve(s,x,y,50,.05)
    zc,_=evolve(clone,x,y,50,.05)
    err=float(np.max(np.abs(predict(z,x)-predict(zc,x))))
    assert err<1e-12
    assert np.max(np.abs(predict(s,x)-predict(fresh,x)))<1e-12
    # Exact nine-knot representability of every tent task is checked separately.
    return dict(autodiff_velocity_error=max(errors), clone_50_step_error=err, power=p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--affine',action='store_true')
    parser.add_argument('--ordinary-steps',type=int,default=3000)
    args=parser.parse_args()
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        raise RuntimeError('Choose a new output directory; old records are immutable.')
    protocol=dict(created=stamp(), seeds=args.seeds, affine=args.affine,
        train_n=256, validation_n=2048, widths=[9,36], rate=.06,
        ordinary_steps=args.ordinary_steps, checkpoint=200, pressure_steps=400,
        recovery_steps=1200, pressure_region='seed modulo 3; all other regions withheld',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forecasts=[
            'Replacement hypothesis: peer-only exposure increases fixed-common-demand recovery cost more at width 9 than 36.',
            'Recruitable-space hypothesis: fresh zero-head additions improve recovery more than exact clones, including after initial gain matching.',
            'Initial-coverage alternative: width advantage already present at initialization; pressure may add no deficit.',
            'Evidence suppression: full-distribution continuation outperforms peer-only exposure, but equalized residual assays may erase the difference.',
            'Affine negative control: local-refinement obstruction and benefit of new routes should be small.',
            'Exact clones must match narrow dynamics to floating point.'],
        interpretation='All predictions qualitative and recorded before outcomes; no fit to endpoints. Paired seed/task is independent unit.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    (out/'audit.json').write_text(json.dumps(audit(),indent=2)+'\n')
    for seed in args.seeds:
        x,y=task(seed,256,affine=args.affine)
        vx,vy=task(seed,2048,True,args.affine)
        starts={m:initial(seed,m) for m in [9,36]}
        ck={}
        for m in [9,36]:
            ck[m], _=evolve(starts[m],x,y,200,.06)
            _, trace=evolve(starts[m],x,y,args.ordinary_steps,.06,vx,vy,
                           sorted({0,200,600,1200,args.ordinary_steps}))
            for row in trace:
                append(out/'ordinary.jsonl',dict(seed=seed,width=m,**row))
        # Demand is the SAME residual function for both widths and all histories.
        ref=(predict(ck[9],x)+predict(ck[36],x))/2
        vref=(predict(ck[9],vx)+predict(ck[36],vx))/2
        demand,vdemand=y-ref,vy-vref
        pressure=regions(x)[seed%3]
        states={}
        for m in [9,36]:
            states[m,'initial']=starts[m]
            states[m,'pause']=ck[m]
            states[m,'peer'],_=evolve(ck[m],x[pressure],y[pressure],400,.06)
            states[m,'full'],_=evolve(ck[m],x,y,400,.06)
            for history in ['initial','pause','peer','full']:
                s=states[m,history]
                append(out/'states.jsonl',dict(seed=seed,width=m,history=history,
                    train=float(np.mean((predict(s,x)-y)**2)/2),
                    validation=float(np.mean((predict(s,vx)-vy)**2)/2)))
                # Reset ONLY output origin, not trainable state or target correction.
                off,voff=predict(s,x),predict(s,vx)
                _, power=direction(s,x,demand+off)
                append(out/'forecasts.jsonl',dict(recorded=stamp(),seed=seed,width=m,
                    history=history,initial_gain_rate=float(power),
                    common_train_loss=float(np.mean(demand**2)/2)))
                _, rows=evolve(s,x,demand,1200,.06,vx,vdemand,
                              [0,20,100,300,1200],off,voff)
                for row in rows:
                    append(out/'assays.jsonl',dict(seed=seed,width=m,history=history,**row))
        # Interventions start at the same learned, peer-exposed function.
        s=states[9,'peer']
        off,voff=predict(s,x),predict(s,vx)
        _,basepower=direction(s,x,demand+off)
        for arm in ['clone','fresh','fresh_gain_matched','fast','slow_time_matched']:
            armstate=expand(s,'clone' if arm=='clone' else 'fresh',seed) if arm in ['clone','fresh','fresh_gain_matched'] else s
            _,power=direction(armstate,x,demand+off)
            multiplier=basepower/power if arm=='fresh_gain_matched' else (2 if arm=='fast' else (.5 if arm=='slow_time_matched' else 1))
            steps=2400 if arm=='slow_time_matched' else 1200
            marks=[0,40,200,600,2400] if steps==2400 else [0,20,100,300,1200]
            append(out/'forecasts.jsonl',dict(recorded=stamp(),seed=seed,arm=arm,
                initial_gain_rate=float(power),rate_multiplier=float(multiplier),
                function_error=float(np.max(np.abs(predict(armstate,x)-off)))))
            _, rows=evolve(armstate,x,demand,steps,.06*multiplier,vx,vdemand,marks,off,voff)
            for row in rows:
                append(out/'interventions.jsonl',dict(seed=seed,arm=arm,rate_multiplier=float(multiplier),**row))
        np.savez_compressed(out/f'states_{seed}.npz',x=x,y=y,vx=vx,vy=vy,demand=demand,vdemand=vdemand,
            **{f'{m}_{history}_{name}':a for (m,history),s in states.items() for name,a in zip(['w','a','c'],s)})
        print(json.dumps(dict(seed=seed,done=True)),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(completed=stamp()))+'\n')


if __name__=='__main__':
    main()
