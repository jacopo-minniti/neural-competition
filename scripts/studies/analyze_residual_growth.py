#!/usr/bin/env python3
"""Audit and summarize the residual-potential and polarity experiments."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

from neural_competition.residual_growth import gaussian_relu_gram
from neural_competition.tasks import paired_gaussian_task
from neural_competition.training import components

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/tables/residual_growth'
FIG = ROOT/'docs/figures/residual_growth'
KEYS = ['study', 'collection', 'data_seed', 'degrees', 'ratio', 'optimizer', 'completed']


def interval(x, rng):
    return np.quantile(rng.choice(x, (4000, len(x))).mean(1), [.025, .975])


def read_and_audit():
    frames, probes, provenance, audits, roles = [], [], [], [], []
    for study in ['residual_growth', 'polarity']:
        for manifest_path in sorted((ROOT/'runs'/study).glob('*/*.json')):
            if manifest_path.parent.name == 'pilot':
                continue
            meta = json.loads(manifest_path.read_text())
            args = meta['args']
            path = manifest_path.with_suffix('.csv.gz')
            frame = pd.read_csv(path)
            frame['study'], frame['collection'] = study, manifest_path.parent.name
            frame['budget'], frame['rate_factor'] = args['steps'], args['rate_factor']
            frame['out_init'] = args['out_init']
            frame['hidden_bias'] = args.get('train_hidden_bias', args.get('train_bias', False))
            frame['balanced_signs'] = args.get('balanced_signs', False)
            frame['arm'] = ['w'+str(w)+'_'+str(mode) for w, mode in
                            zip(frame.width, frame['mode' if study == 'polarity' else 'convention'])]
            frames.append(frame)
            if study == 'residual_growth':
                probes.append(pd.read_csv(manifest_path.with_name(manifest_path.stem+'_probes.csv.gz'))
                              .assign(collection=manifest_path.parent.name))
            snapshot_count = 0
            for source, expected in meta['sources'].items():
                actual = (hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
                          if (ROOT/source).exists() else None)
                if actual != expected:
                    snapshot = OUT/'source_snapshots'/f'{expected}.txt'
                    assert snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected
                    snapshot_count += 1
            state_path = manifest_path.with_suffix('.pt')
            state = torch.load(state_path, weights_only=False)
            norm = torch.tensor([math.sqrt(m['width']) for m in state['models']],
                                dtype=torch.float64)[:, None]
            prediction = components(state['parameters'],
                                    state['data']['xtest'][None].expand(len(norm), -1, -1), norm).sum(-1)
            losses = .5*(prediction-state['data']['ytest']).square().mean(1)
            final = frame[frame.completed == args['steps']]
            difference = float(np.max(np.abs(losses.numpy()-final.test_loss.to_numpy())))
            assert difference < 1e-12
            for i, model in enumerate(state['models']):
                width = model['width']
                a = state['parameters'][i, :width, -1]
                w = state['parameters'][i, :width, :20]
                rng = torch.Generator().manual_seed(94000+model['seed'])
                initial = args['out_init']*torch.randn(width, generator=rng, dtype=torch.float64)
                initial_positive = width//2 if args.get('balanced_signs', False) else int((initial > 0).sum())
                roles.append(dict(study=study, collection=manifest_path.parent.name,
                                  data_seed=meta['args']['data_seed'], degrees=meta['degrees'],
                                  ratio=meta['ratio'], optimizer=meta['optimizer'],
                                  budget=args['steps'], seed=model['seed'], width=width,
                                  arm=final.iloc[i]['arm'], initial_positive=initial_positive,
                                  final_positive=int(((a > 0) & (w.norm(dim=1) > 0)).sum()),
                                  final_negative=int(((a < 0) & (w.norm(dim=1) > 0)).sum()),
                                  required_positive=int((state['data']['beta'] > 0).sum()),
                                  required_negative=int((state['data']['beta'] < 0).sum()),
                                  test_loss=float(losses[i])))
            audits.append(dict(study=study, collection=manifest_path.parent.name,
                               file=manifest_path.name, models=len(final), source_hashes_match=True,
                               archived_source_versions=snapshot_count,
                               reconstructed_loss_max_error=difference))
            provenance.append(dict(study=study, collection=manifest_path.parent.name, **meta,
                                   curve_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   parameters_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest()))
    pd.DataFrame(audits).to_csv(OUT/'audit.csv', index=False)
    pd.DataFrame(roles).to_csv(OUT/'roles.csv', index=False)
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    pd.concat(probes, ignore_index=True).to_csv(OUT/'feature_probes.csv.gz', index=False)
    return pd.concat(frames, ignore_index=True)


def geometry(frame):
    rows = []
    keys = frame[['data_seed', 'degrees', 'ratio']].drop_duplicates()
    for seed, degree, ratio in keys.itertuples(index=False, name=None):
        data = paired_gaussian_task(degree, ratio, seed)
        g = gaussian_relu_gram(data['teacher'], data['teacher'])
        centered = g-1/(2*math.pi)
        for j in range(8):
            sign = data['beta'][j].sign()
            rows.append(dict(data_seed=seed, degrees=degree, ratio=ratio, feature=j,
                             beta=float(data['beta'][j]),
                             raw_evidence=float(sign*(g @ data['beta'])[j]),
                             centered_evidence=float(sign*(centered @ data['beta'])[j]),
                             conditional_evidence=float(data['predicted_growth'][j])))
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'geometry.csv', index=False)
    return result


def figures(frame, geo):
    plt.rcParams.update({'font.size': 10, 'axes.spines.right': False, 'axes.spines.top': False})
    colors = {8:'#b84c38', 16:'#356dc2', 32:'#268266'}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for row, ratio in enumerate([3, -3]):
        for col, opt in enumerate(['sgd', 'momentum', 'adam']):
            ax = axes[row, col]
            part = frame[(frame.study == 'residual_growth') & (frame.collection == 'replication20') &
                         (frame.ratio == ratio) & (frame.optimizer == opt) &
                         (frame.convention == 'fixed') & (frame.completed > 0)]
            for width in [8, 16, 32]:
                stats = part[part.width == width].groupby('completed').test_loss.mean()
                ax.plot(stats.index, stats.clip(lower=1e-7), color=colors[width], lw=2, label=f'Width {width}')
            ax.set(xscale='log', yscale='log', title=f'{opt.upper()} · outputs (1,{ratio})')
            ax.grid(alpha=.15)
            if col == 0:
                ax.set_ylabel('Test half-MSE')
            if row == 1:
                ax.set_xlabel('Minibatch updates')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Ordinary random initialization · same feature directions, changed output signs\n'
                 '20 paired starts · both layers train · unchanged rates across widths')
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(FIG/'ordinary_training.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, opt in zip(axes, ['sgd', 'momentum', 'adam']):
        part = frame[(frame.study == 'residual_growth') & (frame.collection == 'longer') &
                     (frame.optimizer == opt) & (frame.convention == 'fixed') & (frame.completed > 0)]
        for width in [8, 16, 32]:
            stats = part[part.width == width].groupby('completed').test_loss.mean()
            ax.plot(stats.index, stats, color=colors[width], lw=2, marker='o', markersize=3,
                    label=f'Width {width}')
        ax.set(xscale='log', yscale='log', title=opt.upper(), xlabel='Minibatch updates')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Test half-MSE')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Opposing outputs (1,−3) · longer training · five existing starts')
    fig.tight_layout(rect=(0, .1, 1, .94))
    fig.savefig(FIG/'longer.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    subset = geo[(geo.degrees == 30) & (geo.data_seed == 9027) & (geo.feature == 0) & (geo.ratio < 0)]
    subset = subset.assign(strength=-subset.ratio).sort_values('strength')
    axes[0].plot(subset.strength, subset.raw_evidence, 'o-', label='Initial signed feature evidence')
    axes[0].axhline(0, color='black', lw=.8)
    axes[0].set_ylabel('Population residual correlation')
    axes[0].legend(frameon=False)
    for opt in ['sgd', 'momentum', 'adam']:
        part = frame[(frame.study == 'residual_growth') & frame.collection.isin(['main', 'sign_scan']) &
                     (frame.degrees == 30) & (frame.ratio < 0) & (frame.optimizer == opt) &
                     (frame.convention == 'fixed') & (frame.completed == 10000)]
        means = part.groupby(['ratio', 'width']).test_loss.mean().unstack('width')
        means.index = -means.index
        means = means.sort_index()
        axes[1].plot(means.index, means[8]-means[32], 'o-', label=opt.upper())
    axes[1].axhline(0, color='black', lw=.8)
    axes[1].set(xlabel='Opposing component strength r in outputs (1,−r)',
                ylabel='Width-8 loss minus width-32 loss')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('The local sign change is not a phase boundary for width benefit\n'
                 'Five paired starts; unchanged rates; half-angle 30°')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(FIG/'sign_test.png', dpi=160)
    plt.close(fig)

    roles = pd.read_csv(OUT/'roles.csv')
    roles = roles[(roles.study == 'residual_growth') & (roles.collection == 'replication20') &
                  (roles.ratio == -3) & roles.arm.str.endswith('_fixed')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, opt in zip(axes, ['sgd', 'momentum', 'adam']):
        part = roles[roles.optimizer == opt].groupby('width')[['initial_positive', 'final_positive']].mean()
        positions = np.arange(len(part))
        ax.bar(positions-.18, part.initial_positive, .36, label='Initialization', color='#929aa4')
        ax.bar(positions+.18, part.final_positive, .36, label='After 10k updates', color='#356dc2')
        ax.axhline(4, color='#b84c38', linestyle='--', label='Four required positive features')
        ax.set(xticks=positions, xticklabels=part.index, xlabel='Hidden width', title=opt.upper())
    axes[0].set_ylabel('Mean number of positive-output hidden neurons')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Positive-output roles are lost during ordinary training · 20 starts')
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.savefig(FIG/'roles.png', dpi=160)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame = read_and_audit()
    frame.to_csv(OUT/'curves.csv.gz', index=False, float_format='%.17g')
    final = frame[frame.completed == frame.budget].copy()
    final.to_csv(OUT/'endpoints.csv', index=False, float_format='%.17g')
    summary = final.groupby(KEYS+['arm']).agg(seeds=('seed', 'size'),
                 mean_loss=('test_loss', 'mean'), median_loss=('test_loss', 'median'),
                 max_loss=('test_loss', 'max')).reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    rng = np.random.default_rng(742)
    rows = []
    for key, group in final.groupby(KEYS):
        pivot = group.pivot(index='seed', columns='arm', values='test_loss')
        if key[0] == 'residual_growth':
            pairs = [(a, 'w8_fixed') for a in ['w16_fixed', 'w16_mapped', 'w32_fixed', 'w32_mapped']]
        else:
            pairs = [(f'w{w}_{mode}', f'w8_{mode}') for w in [16, 32]
                     for mode in ['no_bias', 'free_bias', 'centered_bias']]
            pairs += [(f'w8_{mode}', 'w8_no_bias') for mode in ['free_bias', 'centered_bias']]
        for treatment, control in pairs:
            delta = (pivot[treatment]-pivot[control]).to_numpy()
            low, high = interval(delta, rng)
            rows.append(dict(zip(KEYS, key), treatment=treatment, control=control, seeds=len(delta),
                             mean_difference=delta.mean(), ci_low=low, ci_high=high,
                             treatment_lower=int((delta < 0).sum())))
    effects = pd.DataFrame(rows)
    effects.to_csv(OUT/'paired_effects.csv', index=False)
    interactions = []
    rep = final[(final.study == 'residual_growth') & (final.collection == 'replication20')]
    for opt, group in rep.groupby('optimizer'):
        pivot = group.pivot(index='seed', columns=['ratio', 'arm'], values='test_loss')
        for wide in ['w16_fixed', 'w32_fixed']:
            diff = (pivot[(-3., 'w8_fixed')]-pivot[(-3., wide)]-
                    pivot[(3., 'w8_fixed')]+pivot[(3., wide)]).to_numpy()
            low, high = interval(diff, rng)
            interactions.append(dict(optimizer=opt, wide=wide, seeds=len(diff),
                                     extra_width_benefit_opposing=diff.mean(), ci_low=low, ci_high=high))
    pd.DataFrame(interactions).to_csv(OUT/'sign_width_interaction.csv', index=False)
    geo = geometry(frame)
    figures(frame, geo)
    metadata = dict(analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    bootstrap_draws=4000, bootstrap_seed=742, intervals='pointwise paired seed bootstrap',
                    executed_model_endpoints=len(final),
                    repeat_warning='Collections overlap; longer runs extend existing starts. Do not pool seeds.')
    (OUT/'analysis.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(summary[summary.collection.isin(['replication20', 'longer'])].to_string(index=False))
    print(effects[(effects.collection == 'replication20') & (effects.treatment == 'w32_fixed')].to_string(index=False))


if __name__ == '__main__':
    main()
