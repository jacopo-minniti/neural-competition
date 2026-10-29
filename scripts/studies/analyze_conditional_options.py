"""Quantitative trajectory budgets, option transitions, and growth controls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]/'docs/theory'


def main():
    raw=np.load(ROOT/'conditional_options_trajectories.npz')
    results=json.loads((ROOT/'conditional_options_results.json').read_text())
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for case in ('discrete','clusters'):
        for m in (4,8,16):
            key=f'{case}_m{m}'
            time,h=raw[f'{key}_time'],raw[f'{key}_history']
            zero=h[:,:,2]<1e-10
            lost=np.any(~zero[:-1]&zero[1:],axis=0)
            recovered=np.any(zero[:-1]&~zero[1:],axis=0)
            # Absolute energy budget, with no logarithm near numerical zero.
            mask=time<=100
            released=h[mask][-1,:,17]
            transfer=h[mask][-1,:,18]
            finite_step=h[mask][-1,:,19]
            energy_change=h[mask][-1,:,11]-h[0,:,11]
            discrepancy=energy_change+released-transfer-finite_step
            snapshots=[]
            for t in (0,5,20,100,2000):
                j=np.argmin(abs(time-t))
                snapshots.append(dict(time=t,loss=float(h[j,:,0].mean()),
                    kappa=float(h[j,:,2].mean()),effective_blocks=float(h[j,:,3].mean()),
                    effective_gate_families=float(h[j,:,10].mean()),
                    conditional_rate=float(h[j,:,12].mean()),zero_options=int(zero[j].sum())))
            row=dict(case=case,width=m,ever_lost=int(lost.sum()),ever_recovered=int(recovered.sum()),
                checkpoints=snapshots,energy_budget_to100=dict(initial=float(h[0,:,11].mean()),
                    final=float(h[mask][-1,:,11].mean()),direct_reduction=float(released.mean()),
                    main_transfer=float(transfer.mean()),finite_step_correction=float(finite_step.mean()),
                    mean_absolute_discrepancy=float(abs(discrepancy).mean())),
                kappa_change_ledger_error=float(np.max(abs(h[-1,:,2]-h[0,:,2]-h[1:,:,15:17].sum(axis=(0,2))))))
            certificate=next(r['stationary_projection'] for r in results['results'] if r['case']==case and r['width']==m)
            certified=np.array(certificate['feasible'])&(np.array(certificate['margin'])>1e-8)&(np.array(certificate['normal_error'])<1e-8)&(np.array(certificate['loss'])>1e-6)
            row['positive_loss_stationary_cells']=int(certified.sum())
            if certified.any():
                row['minimum_certified_margin']=float(np.min(np.array(certificate['margin'])[certified]))
            rows.append(row)
            if case=='clusters':
                axes[0].plot(time,h[:,:,0].mean(1),label=f'width {m}')
                axes[1].plot(time,np.mean(zero,axis=1),label=f'width {m}')
    axes[0].set(xscale='symlog',yscale='log',xlabel='Training time',ylabel='Mean loss',title='Continuous task')
    axes[1].set(xscale='symlog',xlabel='Training time',ylabel='Fraction with zero contrast option',title='Loss and recovery of options')
    axes[0].legend()
    axes[1].legend(fontsize=8,loc='upper left')
    growth=[]
    for case in ('discrete','clusters'):
        h=raw[f'{case}_m4_history']
        t=raw[f'{case}_m4_time']
        idx=np.argmin(abs(t-100))
        damaged=h[idx,:,2]<1e-10
        for kind in ('clone','dormant','common','rare'):
            p=raw[f'{case}_growth_{kind}_probe']
            loss=raw[f'{case}_growth_{kind}_loss']
            growth.append(dict(case=case,kind=kind,damaged_seeds=int(damaged.sum()),
                initially_reopened=int(np.sum(damaged&(p[:,2]>1e-10))),
                final_damaged_mean=float(loss[damaged].mean()) if damaged.any() else None,
                contrast_kappa_mean=float(p[:,2].mean()),conditional_rate_mean=float(p[:,12].mean()),
                effective_blocks_mean=float(p[:,3].mean()),effective_gate_families_mean=float(p[:,10].mean())))
    names=['clone','dormant','common','rare']
    values=[next(r['mean_loss'] for r in results['interventions'] if r['case']=='clusters' and r['kind']==name) for name in names]
    axes[2].bar(names,values)
    axes[2].set(yscale='log',ylabel='Mean loss at time 2,000',title='Grow width 4 → 8 at time 100')
    fig.savefig(ROOT/'conditional_options.png',dpi=180)
    audit=dict(trajectories=rows,growth=growth)
    (ROOT/'conditional_options_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
