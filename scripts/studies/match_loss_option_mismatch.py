"""Exploratory loss-matched comparison of saved option/residual trajectories."""

import json

import matplotlib
import numpy as np
from track_option_mismatch import OUT, SEEDS

matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGETS = [.8, .6, .4, .3, .25]
FIELDS = ['strongest_half_residual', 'strongest_quarter_residual', 'alignment',
          'residual_strength_percentile', 'breadth', 'mass', 'step']


def match(rows, arrays, width, target, field):
    name = 'narrow' if width == 32 else 'wide'
    for before, after in zip(rows, rows[1:]):
        high, low = before[name][field], after[name][field]
        if high >= target >= low and high > low:
            fraction = (high-target)/(high-low)

            def extract(row):
                loading = arrays[f'{width}_{row["step"]}_residual_loading']
                loading = loading/loading.sum()
                metrics = {k:row[name][k] for k in FIELDS if k not in
                           ['strongest_half_residual', 'strongest_quarter_residual', 'step']}
                metrics.update(strongest_half_residual=float(loading[len(loading)//2:].sum()),
                    strongest_quarter_residual=float(loading[3*len(loading)//4:].sum()), step=row['step'])
                return metrics

            a, b = extract(before), extract(after)
            return dict(metrics={k:(1-fraction)*a[k]+fraction*b[k] for k in FIELDS},
                        bracket_steps=[before['step'],after['step']], bracket_losses=[high,low],
                        interpolation_fraction=fraction)
    return None


def main():
    forecasts = [json.loads(line) for line in (OUT/'forecasts.jsonl').read_text().splitlines()]
    details, summaries = [], []
    for field in ('full_loss','panel_loss'):
        for target in TARGETS:
            pairs = []
            for seed in SEEDS:
                rows = sorted((r for r in forecasts if r['seed']==seed),key=lambda r:r['step'])
                with np.load(OUT/f'spectra_{seed}.npz') as arrays:
                    narrow = match(rows,arrays,32,target,field)
                    wide = match(rows,arrays,128,target,field)
                if narrow is not None and wide is not None:
                    pairs.append(dict(seed=seed,matching_loss=field,target=target,narrow=narrow,wide=wide))
            details.extend(pairs)
            metrics = {}
            for name in FIELDS:
                a = np.array([r['narrow']['metrics'][name] for r in pairs])
                b = np.array([r['wide']['metrics'][name] for r in pairs])
                difference = b-a
                metrics[name] = dict(narrow_mean=float(a.mean()),wide_mean=float(b.mean()),
                    paired_difference_mean=float(difference.mean()),
                    paired_difference_se=float(difference.std(ddof=1)/np.sqrt(len(difference))),
                    wide_greater_count=int((difference>0).sum()))
            summaries.append(dict(matching_loss=field,target=target,pairs=len(pairs),metrics=metrics))
    result = dict(method='First descending loss crossing per seed/width; linear interpolation of metrics between adjacent 20-step checkpoints.',
        status='Exploratory post-training analysis; no new runs and no revision of held-out forecast scores.',
        targets=TARGETS,primary_match='full_loss',sensitivity_match='panel_loss',summaries=summaries,pairs=details,
        caveats=['Strength ranks are relative to each network spectrum, not a common absolute strength threshold.',
                 'Equal scalar loss does not equate residual patterns or training histories.',
                 'Interpolation approximates checkpoint statistics; it does not reconstruct an exact intermediate model.',
                 'Paired SE uses eight seeds; loss targets reuse the same seeds and are not independent experiments.'])
    (OUT/'matched_loss.json').write_text(json.dumps(result,indent=2)+'\n')
    fig, axes = plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    primary = [r for r in summaries if r['matching_loss']=='full_loss']
    for ax,name,title in zip(axes,['strongest_half_residual','breadth','alignment'],
        ['Residual in strongest half of modes','Effective response breadth','Strength-weighted relative alignment']):
        for prefix,color,label in [('narrow','#2969a2','Width 32'),('wide','#be6925','Width 128')]:
            ax.plot(TARGETS,[r['metrics'][name][prefix+'_mean'] for r in primary],'-o',color=color,label=label)
        ax.invert_xaxis()
        ax.set(title=title,xlabel='Matched full-training CE (decreasing)')
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Eight paired seeds; interpolated first loss crossings',fontsize=10)
    fig.savefig(OUT/'matched_loss.png',dpi=180)
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
