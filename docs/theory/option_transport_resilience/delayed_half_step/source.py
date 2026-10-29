"""Delayed contextual demand and a prospective finite-budget gate-exit bound.

All training uses the full stationary task. Task contrasts only audit temporal
ordering; options and forecasts use the complete model-discovered response span.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task
from explore_transported_gates import losses, train

ROOT = Path(__file__).resolve().parents[2]


def write(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, allow_nan=False)+"\n")


def measure(state, x, p, y):
    w,a,c=state
    m=a.shape[1]
    z=w@x.T
    gate=z>0
    h=z.clip(0)
    pred=c[:,None]+np.mean(a[...,None]*h,axis=1)
    e=y-pred
    root=np.sqrt(p)
    ga=gate*a[...,None]
    k=(np.einsum("smi,smj->sij",h,h)+
       np.einsum("smi,smj->sij",ga,ga)*(x@x.T))/m+1
    k*=root[None,:,None]*root[None,None,:]
    eig,vec=np.linalg.eigh(k)
    eig=np.maximum(eig,0)
    loading=np.einsum("sij,si->sj",vec,root*e)**2
    tol=1e-11*np.maximum(1,eig[:,-1:])
    null_energy=.5*np.sum(loading*(eig<tol),axis=1)
    # Relax factorized gate-cell coefficients; this is a lower bound on cell loss.
    design=(gate[...,None]*x[None,None]).transpose(0,2,1,3).reshape(len(w),len(x),-1)
    design=np.concatenate([np.ones((len(w),len(x),1)),design],axis=2)*root[None,:,None]
    coef=np.linalg.pinv(design,rcond=1e-12)@(root*y)
    residual=root*y-np.einsum("sij,sj->si",design,coef)
    floor=.5*np.sum(residual**2,axis=1)
    loss=.5*np.sum(p*e**2,axis=1)
    # Fixed optimizer metric: M_w=m I, M_a=m I, M_c=1.
    distance2=np.min(z**2/(m*np.sum(x*x,axis=1)),axis=(1,2))
    # Conservative floating-point allowance, not an extinction threshold.
    floor_lower=np.maximum(0,floor-1e-10*(1+.5*np.sum(p*y*y)))
    surplus=np.maximum(loss-floor_lower,1e-300)
    clock=distance2/surplus
    # Functional invariant: all currently blind examples share the intercept.
    # Crossings outside this discovered set cannot repair its within-set variance.
    blind=np.all(~gate,axis=1)
    bp=np.sum(p*blind,axis=1)
    by=np.sum(p*y*blind,axis=1)/np.maximum(bp,1e-300)
    blind_floor=.5*np.sum(p*blind*(y-by[:,None])**2,axis=1)
    blind_lower=np.maximum(0,blind_floor-1e-10*(1+.5*np.sum(p*y*y)))
    bdist=np.min(np.where(blind[:,None,:],z**2/(m*np.sum(x*x,axis=1)),np.inf),axis=(1,2))
    bdist=np.where(bp>0,bdist,0.)
    blind_clock=bdist/np.maximum(loss-blind_lower,1e-300)
    # Orthogonality follows the task's context/attribute independence.
    r=(x[:,0]>0).astype(float)
    q=p@r
    znorm=np.sqrt(p@(x[:,1]**2))
    main=x[:,1]/znorm
    detail=(r-q)*x[:,1]/np.sqrt(q*(1-q))/znorm
    main_e=e@(p*main)
    detail_e=e@(p*detail)
    return dict(loss=loss,main_energy=.5*main_e**2,detail_energy=.5*detail_e**2,
        null_energy=null_energy,cell_floor=floor,cell_floor_lower=floor_lower,
        distance2=distance2,recovery_clock=clock,
        blind_floor=blind_floor,blind_floor_lower=blind_lower,
        blind_distance2=bdist,blind_recovery_clock=blind_clock,
        option_value=.5*np.sum(loading*eig/(eig+.5),axis=1),
        gradient_power=np.sum(loading*eig,axis=1),
        rank=np.sum(eig>tol,axis=1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--count",type=int,default=64)
    parser.add_argument("--dt",type=float,default=.05)
    parser.add_argument("--seed",type=int,default=282940)
    args=parser.parse_args()
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/"protocol.json").exists():
        raise RuntimeError("Preserving prior records")
    cases=[("delayed",1,0.,.1,2.,.002),
           ("rarer",1,0.,.02,2.,.002),
           ("continuous",3,.15,.1,2.,.002),
           ("no_refinement",1,0.,.1,0.,.002),
           ("larger_init",1,0.,.1,2.,.2)]
    protocol=dict(seed=args.seed,count=args.count,widths=[4,8],cases=cases,dt=args.dt,
        forecast_times=[20,100],end=2000,
        hypotheses=[
            "The shared rule fits before contextual detail; detail is present throughout ordinary training.",
            "Some narrow paths lose a residual-loaded functional response before detail becomes dominant.",
            "A positive cell floor with an exit clock above the remaining horizon predicts persistent floor loss.",
            "Continuous variation and unfinished main fitting can permit recovery; a zero option alone does not predict permanence.",
            "Removing contextual refinement removes its loss cost; smaller q need not increase extinction probability.",
            "The blind-set clock ignores irrelevant outside gate crossings and should cover more persistent coarse errors than the all-gates clock.",
        ],
        chronology="Blind-set refinement added after original seed-bank 282940 results; new seed banks validate it prospectively.",
        bound_scope="Gradient flow with fixed metric; numerical GD tests are validation, not a discrete-time theorem.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    (out/"source.py").write_bytes(Path(__file__).read_bytes())
    raw={}
    for name,nodes,spread,q,detail,scale in cases:
        x,p,y=task(nodes,spread,q)
        y=2*x[:,1]+detail*(x[:,0]>0)*x[:,1]
        # Capacity witness on the full bounded support, including dense quadrature.
        vx,vp,_=task(9,spread,q)
        vy=2*vx[:,1]+detail*(vx[:,0]>0)*vx[:,1]
        ww=np.array([[0,1,0],[0,-1,0],[1,1,0],[1,-1,0.]])
        aa=np.array([2,-2,detail/2,-detail/2])
        witness=np.maximum(vx@ww.T,0)@aa
        assert np.max(abs(witness-vy))<1e-12
        rng=np.random.default_rng(args.seed)
        iw=scale*rng.normal(size=(args.count,8,3))
        ia=scale*rng.normal(size=(args.count,8))
        for m in [4,8]:
            state=iw[:,:m].copy(),ia[:,:m].copy(),np.zeros(args.count)
            key=f"{name}_{m}"
            traces=[]
            times=[]
            forecasts={}
            initial_gate=state[0]@x.T>0
            first_null=np.full(args.count,-1.)
            first_crossover=np.full(args.count,-1.)
            first_exit={}
            first_blind_exit={}
            for i in range(round(2000/args.dt)+1):
                t=i*args.dt
                observed=i%max(1,round(.25/args.dt))==0 if t<=20 else any(abs(t-v)<1e-8 for v in [50,100,200,1000,2000])
                if observed:
                    obs=measure(state,x,p,y)
                    traces.append(np.stack(list(obs.values()),axis=1))
                    times.append(t)
                    first_null[(first_null<0)&(obs["null_energy"]>1e-8)]=t
                    first_crossover[(first_crossover<0)&(obs["detail_energy"]>=obs["main_energy"])]=t
                    if t in [20,100]:
                        forecasts[t]={k:v.copy() for k,v in obs.items()}
                        forecasts[t]["gate"]=(state[0]@x.T>0).copy()
                        forecasts[t]["blind"]=np.all(~forecasts[t]["gate"],axis=1)
                        first_exit[t]=np.full(args.count,-1.)
                        first_blind_exit[t]=np.full(args.count,-1.)
                        for seed in range(args.count):
                            write(out/"forecasts.jsonl",dict(case=name,width=m,seed=seed,time=t,
                                **{k:float(v[seed]) for k,v in obs.items()},
                                floor_for_100=bool(obs["recovery_clock"][seed]>100),
                                floor_to_2000=bool(obs["recovery_clock"][seed]>2000-t),
                                blind_floor_to_2000=bool(obs["blind_recovery_clock"][seed]>2000-t)))
                        for label,val in zip(["w","a","c"],state):
                            raw[f"{key}_t{int(t)}_{label}"]=val.copy()
                for ft,f in forecasts.items():
                    changed=np.any((state[0]@x.T>0)!=f["gate"],axis=(1,2))
                    first_exit[ft][(first_exit[ft]<0)&changed]=t-ft
                    bchanged=np.any((state[0]@x.T>0)&f["blind"][:,None,:],axis=(1,2))
                    first_blind_exit[ft][(first_blind_exit[ft]<0)&bchanged]=t-ft
                if i<round(2000/args.dt):
                    state=train(state,x,p,y,args.dt,dt=args.dt)
            final=measure(state,x,p,y)
            dense=losses(state,vx,vp,vy)
            raw[key+"_trace"]=np.stack(traces)
            raw[key+"_times"]=np.array(times)
            raw[key+"_initial_gate"]=initial_gate
            raw[key+"_columns"]=np.array(list(final))
            for label,val in zip(["w","a","c"],state):
                raw[f"{key}_final_{label}"]=val
            for seed in range(args.count):
                write(out/"outcomes.jsonl",dict(case=name,width=m,seed=seed,
                    **{k:float(v[seed]) for k,v in final.items()},dense_loss=float(dense[seed]),
                    first_null_time=float(first_null[seed]),crossover_time=float(first_crossover[seed]),
                    first_gate_exit={str(ft):float(v[seed]) for ft,v in first_exit.items()},
                    first_blind_exit={str(ft):float(v[seed]) for ft,v in first_blind_exit.items()}))
            np.savez_compressed(out/"trajectories.npz",**raw)
            print(json.dumps(dict(case=name,width=m,mean_loss=float(final["loss"].mean()),
                failures=int(np.sum(final["loss"]>1e-7)))),flush=True)
    (out/"complete.json").write_text("{}\n")


if __name__=="__main__":
    main()
