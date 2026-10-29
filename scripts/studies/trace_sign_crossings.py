#!/usr/bin/env python3
"""Post-result attribution of ordinary positive-to-negative sign crossings.

Fields are measured on the actual source minibatch, immediately before the
update. This is descriptive accounting, not an independent causal intervention.
"""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch

from neural_competition.tasks import paired_gaussian_task
from neural_competition.training import advance

ROOT = Path(__file__).resolve().parents[2]


@torch.jit.script
def trace(p, first, second, x, y, ids, norm, lr: float, kind: int):
    n=p.shape[0]
    counts=torch.zeros(n, 7, dtype=torch.int64)
    seeds=torch.arange(n)
    for t in range(ids.shape[0]):
        xb=x[ids[t]]
        yb=y[ids[t]]
        h=torch.bmm(xb,p[:,:,:20].transpose(1,2)).relu()/norm[:,:,None]
        parts=h*p[:,None,:,-1]
        pred=parts.sum(-1)
        full=((yb-pred)[:,:,None]*h).mean(1)
        nopeer=((yb[:,:,None]-parts)*h).mean(1)
        proposed,first,second=advance(p,first,second,x,y,ids[t:t+1],seeds,norm,
                                      torch.ones_like(p),lr,kind,False,t)
        crossed=(p[:,:,-1]>0)&(proposed[:,:,-1]<0)
        counts[:,0]+=crossed.sum(1)
        counts[:,1]+=(crossed&(full<0)).sum(1)
        counts[:,2]+=(crossed&(nopeer<0)).sum(1)
        counts[:,3]+=(crossed&(nopeer>=0)&(full<0)).sum(1)
        counts[:,4]+=(crossed&(full>=0)).sum(1)
        counts[:,5]+=(crossed&(full>nopeer)).sum(1)
        counts[:,6]+=((p[:,:,-1]<0)&(proposed[:,:,-1]>0)).sum(1)
        p=proposed
    return p,first,second,counts


def main():
    torch.set_num_threads(1)
    data=paired_gaussian_task(30,-3,9027)
    rows=[]
    for kind,opt in enumerate(['sgd','momentum','adam']):
        bundle=torch.load(ROOT/f'runs/sign_survival/main/d9027_a30_r-3_{opt}.pt',weights_only=False)
        selected=[i for i,m in enumerate(bundle['models']) if m['arm']=='balanced8']
        p=bundle['states'][0]['parameters'][selected,:8].clone()
        ids=torch.stack([torch.randint(4096,(1000,128),
                         generator=torch.Generator().manual_seed(95000+s))
                         for s in range(200,220)],1)
        norm=torch.full((20,1),math.sqrt(8),dtype=torch.float64)
        p,first,second,counts=trace(p,torch.zeros_like(p),torch.zeros_like(p),
                                  data['xtrain'],data['ytrain'],ids,norm,[.3,.03,.003][kind],kind)
        for actual,key in [(p,'parameters'),(first,'first'),(second,'second')]:
            # Different padded widths can affect matrix-kernel reduction roundoff.
            torch.testing.assert_close(actual,bundle['states'][1000][key][selected,:8],
                                       rtol=1e-10,atol=1e-12)
        columns=['positive_to_negative','gradient_supports_crossing','nopeer_already_adverse',
                 'peer_induced_gradient_reversal','gradient_opposes_crossing',
                 'peers_relieve_adverse_force','negative_to_positive']
        for seed,values in zip(range(200,220),counts.tolist()):
            rows.append(dict(optimizer=opt,seed=seed,**dict(zip(columns,values))))
    folder=ROOT/'docs/tables/sign_survival'
    pd.DataFrame(rows).to_csv(folder/'crossings.csv',index=False)
    sources=[Path(__file__),ROOT/'src/neural_competition/training.py',
             ROOT/'src/neural_competition/tasks.py',ROOT/'src/neural_competition/residual_growth.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sources:
        (folder/'sources'/(hashes[str(p.relative_to(ROOT))]+'.txt')).write_bytes(p.read_bytes())
    (folder/'crossings_provenance.json').write_text(json.dumps(dict(
        sources=hashes,steps=1000,seeds=list(range(200,220)),data_seed=9027,
        note='Post-result source-minibatch accounting; seeds shared with main experiment.'),indent=2)+'\n')
    print(pd.DataFrame(rows).groupby('optimizer').sum(numeric_only=True).drop(columns='seed').to_string())


if __name__=='__main__':
    main()
