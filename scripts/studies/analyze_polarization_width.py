"""Analyze the prospective polarization law and audit its numerical basin map."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from explore_polarization_width_law import CASES, data, labels, polarize
from explore_transported_gates import initialize

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]/'docs/theory'


def main():
    result=json.loads((ROOT/'polarization_width_validation.json').read_text())
    pred=np.load(ROOT/'polarization_width_predictions.npz')
    actual=np.load(ROOT/'polarization_width_validation.npz')
    rows,audits=[],[]
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    for case,ax in zip(CASES,axes.flat):
        name=case[0]
        x,p,y=data(case)
        initial=initialize(281299,256,12)
        initial_labels=labels(initial[0],initial[1],x)
        for m in (4,6,8,12):
            old=(np.any(initial_labels[:,:m]==1,axis=1)&np.any(initial_labels[:,:m]==-1,axis=1))
            forecast=pred[f'{name}_m{m}_predicted']
            loss=actual[f'{name}_m{m}_loss']
            failure=loss>1e-8
            predicted_fates=pred[f'{name}_families'][:,:m]
            actual_fates=labels(actual[f'{name}_m{m}_w'],actual[f'{name}_m{m}_a'],x)
            rows.append(dict(case=name,width=m,dynamically_predicted_family_extinctions=int(np.sum(old&(forecast>0))),
                initialization_family_misclassifications=int(np.sum((~old)!=failure)),
                dynamic_family_misclassifications=int(np.sum((forecast>0)!=failure)),
                predicted_dead_fraction=float(np.mean(predicted_fates==0)),
                actual_dead_fraction=float(np.mean(actual_fates==0)),
                neuron_fate_disagreement=float(np.mean(predicted_fates!=actual_fates))))
        forecast=[r for r in result['forecasts'] if r['case']==name]
        validation=[r for r in result['validation'] if r['case']==name]
        widths=[r['width'] for r in forecast]
        ax.plot(widths,[r['law'] for r in forecast],'-o',label='Task-derived law')
        ax.plot(widths,[r['static_dead_only_law'] for r in forecast],'--',label='Initial death only')
        ax.errorbar(widths,[r['mean_loss'] for r in validation],yerr=[r['se'] for r in validation],
                     fmt='s',capsize=3,label='Observed ± SE')
        ax.set(yscale='log',ylim=(1e-5,.1),xticks=widths,xlabel='Width',ylabel='Half squared loss',title=name)
        ax.legend(fontsize=8)
        bank=initialize(281410,1,2048)
        original,_=polarize(bank[0][0],bank[1][0],x,p,y)
        refined,_=polarize(bank[0][0],bank[1][0],x,p,y,dt=.01)
        extended,_=polarize(bank[0][0],bank[1][0],x,p,y,horizon=128.)
        audits.append(dict(case=name,halved_step_disagreement=float(np.mean(original!=refined)),
            doubled_horizon_disagreement=float(np.mean(original!=extended)),
            death_original=float(np.mean(original==0)),death_refined=float(np.mean(refined==0))))
    fig.suptitle('Zero losses fall below the logarithmic plotting range',fontsize=10)
    fig.savefig(ROOT/'polarization_width_law.png',dpi=180)
    out=dict(classification=rows,basin_audit=audits,
        caveat='Nonzero continuous-task loss includes within-cluster errors; binary loss classification there is not a pure extinction test.')
    (ROOT/'polarization_width_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
