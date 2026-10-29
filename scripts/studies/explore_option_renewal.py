"""Task-derived renewal law from a damaged training checkpoint, not initialization.

One added feature opens the last missing contrast when the existing edit span
has codimension one. Compare exact linear repair with unrestricted continuation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task
from explore_transported_gates import losses, train
from trace_conditional_options import basis, probe

ROOT = Path(__file__).resolve().parents[2]/'docs/theory'


def forecast():
    raw = np.load(ROOT/'conditional_options_trajectories.npz')
    x, p, y = task(1, 0., .1)
    checkpoint = tuple(raw[f'discrete_m4_t100_{s}'] for s in ('w', 'a', 'c'))
    current = probe(checkpoint, x, p, y, basis(x, p))
    # Select using checkpoint information only, before the new continuations.
    eligible = np.flatnonzero((current[:, 2] < 1e-10) & (current[:, 0] > 1e-4))
    selected = int(eligible[0])
    w, a, c = (s[selected:selected+1] for s in checkpoint)
    gates = w @ x.T > 0
    design = (gates[..., None]*x[None, None]).transpose(0, 2, 1, 3).reshape(1, 4, -1)[0]
    design = np.column_stack([np.ones(4), design])
    u, singular, _ = np.linalg.svd(np.sqrt(p)[:, None]*design, full_matrices=True)
    assert np.sum(singular > 1e-10) == 3
    head = np.column_stack([np.ones(4), np.maximum(w @ x.T, 0)[0].T])
    assert np.linalg.matrix_rank(head, tol=1e-10) == 3
    null = u[:, -1]
    support = np.flatnonzero(abs(null) > 1e-8)
    assert len(support) == 2
    rho = np.dot(x[support[0]], x[support[1]]) / np.prod(np.linalg.norm(x[support], axis=1))
    p_miss = .25 + np.arcsin(rho)/(2*np.pi)
    damage = .5*(null @ (np.sqrt(p)*y))**2
    rng = np.random.default_rng(241029)
    arrays, rows = {}, []
    for k in (1, 2, 4):
        new_w = rng.normal(size=(128, k, 3))
        m = a.shape[1]
        scale = np.sqrt((m+k)/m)
        grown = (np.concatenate([np.repeat(scale*w,128,axis=0),new_w],axis=1),
                 np.concatenate([np.repeat(scale*a,128,axis=0),np.zeros((128,k))],axis=1),
                 np.repeat(c,128))
        assert np.allclose(losses(grown,x,p,y),losses((w,a,c),x,p,y))
        features = np.maximum(new_w @ x.T, 0)
        contrast = np.einsum('smi,i->sm',features,np.sqrt(p)*null)
        missed = np.all(abs(contrast)<1e-10,axis=1)
        arrays[f'k{k}_linear_loss'] = damage*missed
        for label,value in zip(('w','a','c'),grown):
            arrays[f'k{k}_{label}'] = value
        rows.append(dict(added=k,predicted_loss=float(damage*p_miss**k),
                         predicted_failure_probability=float(p_miss**k),
                         linear_repair_mean=float(damage*missed.mean()),linear_failures=int(missed.sum())))
    result=dict(checkpoint_time=100,selected_seed_index=selected,draws=128,seed=241029,
                task_support=support.tolist(),input_correlation=float(rho),damage=float(damage),
                miss_probability=float(p_miss),forecasts=rows,
                status='Forecasts saved before unrestricted continuation of these expanded networks.')
    np.savez_compressed(ROOT/'option_renewal_checkpoints.npz',**arrays)
    (ROOT/'option_renewal_forecasts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def validate():
    result=json.loads((ROOT/'option_renewal_forecasts.json').read_text())
    raw=np.load(ROOT/'option_renewal_checkpoints.npz')
    x,p,y=task(1,0.,.1)
    endpoints,rows={},[]
    for k in (1,2,4):
        state=tuple(raw[f'k{k}_{label}'] for label in ('w','a','c'))
        end=train(state,x,p,y,1900)
        final=losses(end,x,p,y)
        linear=raw[f'k{k}_linear_loss']
        endpoints[f'k{k}_loss']=final
        row=dict(added=k,actual_mean_loss=float(final.mean()),se=float(final.std(ddof=1)/np.sqrt(len(final))),
                 linear_mean_loss=float(linear.mean()),mae_from_linear=float(abs(linear-final).mean()),
                 predicted_fail_but_recovered=int(np.sum((linear>1e-6)&(final<1e-6))),
                 predicted_repair_but_failed=int(np.sum((linear<1e-6)&(final>1e-6))))
        rows.append(row)
        print(json.dumps(row),flush=True)
    result['validation']=rows
    np.savez_compressed(ROOT/'option_renewal_endpoints.npz',**endpoints)
    (ROOT/'option_renewal_validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['forecast','validate'])
    args=parser.parse_args()
    forecast() if args.phase=='forecast' else validate()
