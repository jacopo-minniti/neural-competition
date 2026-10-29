#!/usr/bin/env python3
"""Audit local MNIST coupling forecasts and retain unsuccessful probes."""

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neural_competition.mnist import loss_rows, prepare_data, scores

ROOT=Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(2)
    dest=ROOT/'docs/tables/coupled_modes'
    (dest/'sources').mkdir(parents=True,exist_ok=True)
    data=prepare_data(ROOT/'data/mnist')
    frames,manifests,audits=[],[],[]
    for folder in sorted((ROOT/'runs/coupled_modes').iterdir()):
        if not (folder/'results.csv').exists():
            continue
        frame=pd.read_csv(folder/'results.csv',float_precision='round_trip')
        frame['collection']=folder.name
        frame['horizon_index']=frame.groupby(['seed','width']).cumcount()
        # Frozen forecasts are an implementation audit, not independent validation.
        for name in ['diagonal','full']:
            np.testing.assert_allclose(frame['predicted_'+name],frame['frozen_'+name],
                                       rtol=1e-10,atol=1e-12)
        manifest=json.loads((folder/'manifest.json').read_text())
        manifests.append(dict(collection=folder.name,**manifest))
        for source,digest in manifest['sources'].items():
            path=folder/'sources'/f'{digest}.txt'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,source
            shutil.copyfile(path,dest/'sources'/path.name)
        for path in folder.glob('final_*.pt'):
            bundle=torch.load(path,weights_only=True)
            width=int(path.stem.split('_w')[1].split('_')[0])
            seed=int(path.stem.split('_s')[1])
            saved=frame.query('width==@width and seed==@seed').iloc[-1]
            maximum=0.
            for branch,p in enumerate(bundle['branches']):
                name='diagonal' if branch==0 else 'full'
                for split,ids,prefix in [('train',bundle['train_ids'],'observed'),
                                         ('val',bundle['val_ids'],'validation')]:
                    x,y=data['x'+split][ids].double(),data['y'+split][ids].double()
                    measured=float(loss_rows(scores(p,x),y,False).mean())
                    expected=float(saved[prefix+'_'+name])
                    np.testing.assert_allclose(measured,expected,rtol=1e-10,atol=1e-12)
                    maximum=max(maximum,abs(measured-expected)/max(1,abs(expected)))
            audits.append(dict(collection=folder.name,width=width,seed=seed,
                               max_scaled_error=maximum))
        frames.append(frame)
    frame=pd.concat(frames,ignore_index=True)
    frame.to_csv(dest/'forecasts.csv',index=False)
    metrics=[]
    for (collection,horizon),g in frame.groupby(['collection','horizon_index']):
        for target in ['training','validation']:
            actual=g['observed_damage' if target=='training' else 'validation_damage'].to_numpy()
            candidates={'readout':'predicted_damage' if target=='training'
                        else 'predicted_validation_damage'}
            if 'tangent_damage' in g and g.tangent_damage.notna().any():
                candidates['full_tangent']='tangent_damage' if target=='training' else 'tangent_validation_damage'
            for model,column in candidates.items():
                if column not in g or g[column].isna().all():
                    continue
                prediction=g[column].to_numpy()
                metrics.append(dict(collection=collection,horizon_index=horizon,target=target,
                    model=model,cases=len(g),rmse=float(np.sqrt(np.mean((actual-prediction)**2))),
                    zero_rmse=float(np.sqrt(np.mean(actual**2))),
                    sign_correct=int(np.sum(actual*prediction>0)),
                    mean_prediction=float(prediction.mean()),mean_observed=float(actual.mean())))
    pd.DataFrame(metrics).to_csv(dest/'prediction_metrics.csv',index=False)
    pd.DataFrame(audits).to_csv(dest/'audit.csv',index=False)
    (dest/'provenance.json').write_text(json.dumps(manifests,indent=2)+'\n')
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (dest/'sources'/f'{digest}.txt').write_bytes(Path(__file__).read_bytes())
    (dest/'analysis_provenance.json').write_text(json.dumps({str(Path(__file__).relative_to(ROOT)):digest}))
    figure=ROOT/'docs/figures/coupled_modes'
    figure.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(9,7),constrained_layout=True)
    for row,(actual,pred,label) in enumerate([
        ('observed_damage','tangent_damage','Training'),
        ('validation_damage','tangent_validation_damage','Validation')]):
        for col,horizon in enumerate([10,100]):
            ax=axes[row,col]
            g=frame.query('collection=="tangent_epoch5" and steps==@horizon')
            for width,color in [(16,'#177d99'),(128,'#b46a42')]:
                a=g[g.width==width]
                ax.scatter(a[pred],a[actual],label=f'Width {width}',color=color,s=35)
            lim=max(g[pred].abs().max(),g[actual].abs().max())*1.15
            ax.plot([-lim,lim],[-lim,lim],color='#888888',ls='--',lw=1)
            ax.axhline(0,color='#aaaaaa',lw=.6)
            ax.axvline(0,color='#aaaaaa',lw=.6)
            ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='Forecast loss difference',
                   ylabel='Observed loss difference',title=f'{label}, {horizon} updates')
            ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Coupling damage forecasts: accurate locally, drift at longer horizons\nMNIST, five paired starts; positive means the coupling intervention helps',fontsize=12)
    fig.savefig(figure/'forecast_validation.png',dpi=180)
    print(pd.DataFrame(metrics).query('collection=="tangent_epoch5"').to_string(index=False))
    print('Audited',len(audits),'final bundles;',len(frame),'horizon rows')


if __name__=='__main__':
    main()
