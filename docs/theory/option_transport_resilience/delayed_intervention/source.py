"""Early versus late implementation slack, with matched function and full kernel.

Fresh nested seeds. Original contextual loss throughout. Fixed metric changes
must be reported: this is a mechanism intervention, not standard-width SGD.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task

ROOT=Path(__file__).resolve().parents[2]/"docs/theory/option_transport_resilience/delayed_intervention"


def append(name,row):
    with (ROOT/name).open("a") as f:
        f.write(json.dumps(row,allow_nan=False)+"\n")


def pred(s,x):
    w,a,c,m,_,_=s
    return c[:,None]+np.sum(a[...,None]*np.maximum(w@x.T,0),axis=1)/m


def gram(s,x,p):
    w,a,_,m,ma,mw=s
    z=w@x.T
    h=z.clip(0)
    g=(z>0)*a[...,None]
    k=(ma*np.einsum("smi,smj->sij",h,h)+mw*np.einsum("smi,smj->sij",g,g)*(x@x.T))/m**2+1
    return k*np.sqrt(p)[None,:,None]*np.sqrt(p)[None,None,:]


def expand(s,uniform=False):
    w,a,c,m,_,_=s
    k=4
    aa=np.tile(a/k,(1,k)) if uniform else np.concatenate([a,np.zeros((len(a),3*m))],axis=1)
    return np.tile(w,(1,k,1)),aa,c.copy(),m,m/k,m*k if uniform else m


def step(s,x,p,y,dt):
    w,a,c,m,ma,mw=s
    z=w@x.T
    h=z.clip(0)
    e=p*(pred(s,x)-y)
    da=-(ma/m)*np.sum(h*e[:,None],axis=2)
    dw=-(mw/m)*a[...,None]*(((z>0)*e[:,None])@x)
    c-=dt*e.sum(axis=1)
    a+=dt*da
    w+=dt*dw


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/"protocol.json").exists():
        raise RuntimeError("Preserving recorded experiment")
    protocol=dict(seed=282942,seeds=64,widths=[4,8],scale=.002,dt=.05,end=2000,
        arms=["ordinary","clone_at_2","reserve_at_2","reserve_at_20","double_rate_at_2"],
        cases=["delayed","continuous","no_refinement"],
        predictions=[
            "Uniform clones reproduce ordinary learning.",
            "Early reserve could preserve an implementation until delayed demand develops; late reserve may fail once all useful gates close.",
            "Matching function and kernel excludes added initial coverage and initial loss derivative as explanations of reserve effects.",
            "The intervention changes the fixed metric and subsequent feature-learning speed; neither later speed nor standard-width mediation is thereby controlled.",
            "No-refinement task should have little persistent absolute benefit.",
        ],
        chronology="Designed after delayed-demand and response-matched sensor results; all new branch outcomes follow this protocol.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    (ROOT/"source.py").write_bytes(Path(__file__).read_bytes())
    raw={}
    for name,nodes,spread,detail in [("delayed",1,0.,2.),("continuous",3,.15,2.),("no_refinement",1,0.,0.)]:
        x,p,_=task(nodes,spread,.1)
        y=2*x[:,1]+detail*(x[:,0]>0)*x[:,1]
        vx,vp,_=task(9,spread,.1)
        vy=2*vx[:,1]+detail*(vx[:,0]>0)*vx[:,1]
        rng=np.random.default_rng(282942)
        iw=.002*rng.normal(size=(64,8,3))
        ia=.002*rng.normal(size=(64,8))
        for m in [4,8]:
            for arm in protocol["arms"]:
                s=(iw[:,:m].copy(),ia[:,:m].copy(),np.zeros(64),m,m,m)
                snapshots={}
                for i in range(40001):
                    t=i*.05
                    change=(t==2 and arm in ["clone_at_2","reserve_at_2"]) or (t==20 and arm=="reserve_at_20")
                    if change:
                        old=s
                        s=expand(s,uniform=arm=="clone_at_2")
                        append("matching.jsonl",dict(case=name,width=m,arm=arm,time=t,
                            function_error=float(np.max(abs(pred(old,vx)-pred(s,vx)))),
                            kernel_error=float(np.max(abs(gram(old,x,p)-gram(s,x,p))))))
                    if t in [0,2,4,6,8,10,20,50,100,200,1000,2000]:
                        e=pred(s,x)-y
                        l=.5*np.sum(p*e**2,axis=1)
                        dense=.5*np.sum(vp*(pred(s,vx)-vy)**2,axis=1)
                        k=gram(s,x,p)
                        eig,vec=np.linalg.eigh(k)
                        load=np.einsum("sij,si->sj",vec,np.sqrt(p)*e)**2
                        null=.5*np.sum(load*(eig<1e-11*np.maximum(1,eig[:,-1:])),axis=1)
                        for seed in range(64):
                            append("trajectory.jsonl",dict(case=name,width=m,arm=arm,time=t,seed=seed,
                                loss=float(l[seed]),dense_loss=float(dense[seed]),null_energy=float(null[seed])))
                        if t in [2,20,2000]:
                            snapshots[t]=tuple(v.copy() for v in s[:3])
                    if i<40000:
                        step(s,x,p,y,.1 if arm=="double_rate_at_2" and t>=2 else .05)
                for t,ss in snapshots.items():
                    for key,val in zip(["w","a","c"],ss):
                        raw[f"{name}_{m}_{arm}_{t}_{key}"]=val
                np.savez_compressed(ROOT/"states.npz",**raw)
                print(json.dumps(dict(case=name,width=m,arm=arm,loss=float(l.mean()))),flush=True)
    (ROOT/"complete.json").write_text("{}\n")


if __name__=="__main__":
    main()
