"""Check the width comparison with a halved optimizer step."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task
from explore_transported_gates import initialize, losses, train


def main():
    rows=[]
    for name,nodes,spread in [('discrete',1,0.),('clusters',3,.15)]:
        x,p,y=task(nodes,spread,.1)
        initial=initialize(997031,48,16)
        for m in (4,8):
            state=tuple(v[:,:m].copy() if v.ndim>1 else v.copy() for v in initial)
            final=train(state,x,p,y,2000,dt=.025)
            loss=losses(final,x,p,y)
            row=dict(case=name,width=m,dt=.025,mean_loss=float(loss.mean()),
                     se=float(loss.std(ddof=1)/np.sqrt(len(loss))),losses=loss.tolist())
            rows.append(row)
            print({k:v for k,v in row.items() if k!='losses'},flush=True)
    output=Path(__file__).resolve().parents[2]/'docs/theory/conditional_options_step_audit.json'
    output.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    main()
