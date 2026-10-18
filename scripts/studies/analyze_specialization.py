#!/usr/bin/env python3
"""Summarize every non-pilot specialization run and audit its provenance."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_specialization import initialize

from neural_competition.specialization import merged_coefficients, relu_kernel
from neural_competition.trajectory import components

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT/'docs/tables/specialization'
FIGURES = ROOT/'docs/figures/specialization'
KEYS = ['collection', 'data_seed', 'degrees', 'optimizer', 'train_bias', 'completed']
CONTRASTS = [('split16', 'crowded8'), ('balanced8', 'crowded8'),
             ('clone16', 'clean8'), ('random16_fixed', 'random8'),
             ('random16_mapped', 'random8')]
COLORS = dict(crowded8='#c54e36', balanced8='#268266', split16='#386ac1',
              clone16='#8c8790', random8='#c54e36', random16_fixed='#bf8a18',
              random16_mapped='#386ac1')
LABELS = dict(crowded8='8: uneven allocation', balanced8='8: balanced allocation',
              split16='16: perturbed expansion', clone16='16: unperturbed copies',
              random8='8: random', random16_fixed='16: random, same rates',
              random16_mapped='16: random, mapped rates')


def bootstrap(values, rng):
    samples = rng.choice(values, (4000, len(values)), replace=True).mean(1)
    return np.quantile(samples, [.025, .975])


def audit(paths):
    rows = []
    provenance = []
    for path in paths:
        manifest = json.loads(path.with_suffix('').with_suffix('.json').read_text())
        provenance.append(dict(collection=path.parent.name, **manifest,
                               curve_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        state = torch.load(path.with_suffix('').with_suffix('.pt'), weights_only=False)
        data, models = state['data'], state['models']
        hash_ok = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == sha
                      for p, sha in manifest['sources'].items())
        max_initial_rmse, max_clone_rmse = 0., 0.
        for seed in sorted({m['seed'] for m in models}):
            initial = []
            for arm in ['crowded8', 'balanced8', 'split16', 'clean8', 'clone16']:
                p = initialize(data, seed, arm, manifest['args']['jitter'])
                norm = torch.tensor([[math.sqrt(len(p))]], dtype=torch.float64)
                pred = components(p[None], data['xtest'][None], norm).sum(-1)
                coarse = (data['xtest'] @ data['centers'].T).relu() @ data['coarse_beta']
                initial.append(float((pred-coarse).square().mean().sqrt()))
            max_initial_rmse = max(max_initial_rmse, *initial)
            ids = [i for i, m in enumerate(models)
                   if m['seed'] == seed and m['arm'] in ['clean8', 'clone16']]
            p = state['parameters'][ids]
            norm = torch.tensor([math.sqrt(models[i]['width']) for i in ids],
                                dtype=torch.float64)[:, None]
            pred = components(p, data['xtest'][None].expand(2, -1, -1), norm).sum(-1)
            max_clone_rmse = max(max_clone_rmse, float((pred[0]-pred[1]).square().mean().sqrt()))
        rows.append(dict(file=str(path.relative_to(ROOT)), hashes_match=hash_ok,
                         models=len(models), initial_perturbation_rmse_max=max_initial_rmse,
                         final_clean_clone_rmse_max=max_clone_rmse))
    frame = pd.DataFrame(rows)
    assert frame.hashes_match.all(), 'Producer or protocol changed since execution'
    frame.to_csv(TABLES/'audit.csv', index=False)
    (TABLES/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return frame


def plots(frame):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 130})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for row, (collection, angle) in enumerate([('replication20', 15), ('main', 30)]):
        for col, optimizer in enumerate(['sgd', 'momentum', 'adam']):
            ax = axes[row, col]
            subset = frame[(frame.collection == collection) & (frame.degrees == angle) &
                           (frame.optimizer == optimizer) & (frame.completed > 0)]
            for arm in ['crowded8', 'balanced8', 'split16', 'clone16']:
                stats = subset[subset.arm == arm].groupby('completed').fraction_remaining.mean()
                ax.plot(stats.index, stats.clip(lower=1e-7), color=COLORS[arm],
                        label=LABELS[arm], lw=2)
            ax.set(xscale='log', yscale='log', ylim=(1e-7, 3),
                   title=f'{optimizer.upper()} · half-angle {angle}°')
            ax.grid(alpha=.15)
            if col == 0:
                ax.set_ylabel('Test loss / initial shared-feature loss')
            if row == 1:
                ax.set_xlabel('Minibatch updates')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Specialization depends on available local degrees of freedom\n'
                 'Both layers train; means over 20 paired starts at 15°, 5 at 30°')
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(FIGURES/'controlled.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, optimizer in zip(axes, ['sgd', 'momentum', 'adam']):
        subset = frame[(frame.collection == 'replication20') &
                       (frame.optimizer == optimizer) & (frame.completed > 0)]
        for arm in ['random8', 'random16_fixed', 'random16_mapped']:
            stats = subset[subset.arm == arm].groupby('completed').test_loss.mean()
            ax.plot(stats.index, stats, color=COLORS[arm], label=LABELS[arm], lw=2)
        ax.set(xscale='log', yscale='log', title=optimizer.upper(), xlabel='Minibatch updates')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Test half-MSE')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Ordinary random initialization · half-angle 15° · 20 paired starts')
    fig.tight_layout(rect=(0, .1, 1, .94))
    fig.savefig(FIGURES/'random.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    matched = frame[((frame.collection == 'main') & (frame.degrees == 15)) |
                    ((frame.collection == 'longer') & (frame.completed == 100000))]
    for ax, optimizer in zip(axes, ['sgd', 'momentum', 'adam']):
        subset = matched[(matched.optimizer == optimizer) & (matched.completed > 0)]
        for arm in ['crowded8', 'split16', 'random8']:
            stats = subset[subset.arm == arm].groupby('completed').test_loss.mean()
            ax.plot(stats.index, stats.clip(lower=1e-12), color=COLORS[arm],
                    linestyle='--' if arm == 'random8' else '-', marker='o', markersize=3,
                    label=LABELS[arm], lw=2)
        ax.set(xscale='log', yscale='log', title=optimizer.upper(), xlabel='Minibatch updates')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Test half-MSE')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('The narrow networks catch up · half-angle 15° · same five paired starts')
    fig.tight_layout(rect=(0, .1, 1, .94))
    fig.savefig(FIGURES/'longer.png')
    plt.close(fig)

    angles = np.linspace(0, 89, 400)
    values = [merged_coefficients(math.radians(t)) for t in angles]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label, color in [('ordinary_quadratic', 'Move one shared neuron (refit its output)',
                               COLORS['crowded8']),
                              ('split_quadratic', 'Separate two copies', COLORS['split16'])]:
        ax.plot(angles, [v[key] for v in values], label=label, color=color, lw=2)
    ax.axhline(0, color='black', lw=.8)
    ax.set(xlabel='Teacher pair half-angle θ (degrees)',
           ylabel='Quadratic coefficient of population loss',
           title='One shared feature: ordinary motion versus specialization')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(FIGURES/'curvature.png')
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in (ROOT/'runs/specialization').glob('*/*.csv.gz')
                   if p.parent.name != 'pilot')
    frames = []
    for path in paths:
        args = json.loads(path.with_suffix('').with_suffix('.json').read_text())['args']
        frames.append(pd.read_csv(path).assign(collection=path.parent.name,
                                               rate_factor=args['rate_factor'],
                                               jitter=args['jitter'], budget=args['steps']))
    frame = pd.concat(frames, ignore_index=True)
    frame['fraction_remaining'] = np.where(frame.degrees > 0,
                                           frame.test_loss/frame.coarse_test_loss, np.nan)
    frame.to_csv(TABLES/'curves.csv.gz', index=False, float_format='%.17g')
    final = frame[frame.completed == frame.budget].copy()
    final['success'] = np.where(final.degrees > 0, final.fraction_remaining < .01, np.nan)
    final.to_csv(TABLES/'endpoints.csv', index=False, float_format='%.17g')
    summary = final.groupby(KEYS+['arm']).agg(
        seeds=('seed', 'size'), test_loss=('test_loss', 'mean'),
        test_loss_median=('test_loss', 'median'),
        coarse_test_loss=('coarse_test_loss', 'first'),
        fraction_remaining=('fraction_remaining', 'mean'),
        resolved=('resolved', 'mean'), successes=('success', 'sum')).reset_index()
    summary.loc[summary.degrees == 0, 'successes'] = np.nan
    summary.to_csv(TABLES/'summary.csv', index=False, float_format='%.17g')
    rng = np.random.default_rng(741)
    rows = []
    for key, group in frame.groupby(KEYS):
        for metric in ['test_loss', 'fraction_remaining']:
            if metric == 'fraction_remaining' and key[2] == 0:
                continue
            pivot = group.pivot(index='seed', columns='arm', values=metric)
            for treatment, control in CONTRASTS:
                differences = (pivot[treatment]-pivot[control]).to_numpy()
                lo, hi = bootstrap(differences, rng)
                rows.append(dict(zip(KEYS, key), metric=metric, treatment=treatment,
                                 control=control, seeds=len(differences),
                                 mean_difference=differences.mean(), ci_low=lo, ci_high=hi,
                                 treatment_lower=int((differences < 0).sum())))
    pd.DataFrame(rows).to_csv(TABLES/'paired_contrasts.csv', index=False, float_format='%.17g')
    rows = []
    for degrees in [0, 15, 30, 60, 75]:
        theta = math.radians(degrees)
        innovation = (1+2*relu_kernel(2*theta)+2*math.cos(theta)**2
                      -8*math.cos(theta)*relu_kernel(theta))
        rows.append(dict(degrees=degrees, **merged_coefficients(theta),
                         wedge_probability=2*theta/math.pi, innovation_energy=innovation))
    pd.DataFrame(rows).to_csv(TABLES/'population.csv', index=False)
    low, high = 0., math.pi/2
    for _ in range(60):
        mid = (low+high)/2
        if merged_coefficients(mid)['ordinary_quadratic'] > 0:
            low = mid
        else:
            high = mid
    audits = audit(paths)
    # First five starts of the replication are repeats, not extra independent evidence.
    old = frame[(frame.collection == 'main') & (frame.degrees == 15)]
    new = frame[(frame.collection == 'replication20') & (frame.seed < 5)]
    sort = ['optimizer', 'completed', 'seed', 'arm']
    duplicate_delta = float(np.max(np.abs(old.sort_values(sort).test_loss.to_numpy()-
                                         new.sort_values(sort).test_loss.to_numpy())))
    assert duplicate_delta < 1e-14
    final['effective_jitter'] = np.where(final.arm.isin(['crowded8', 'balanced8', 'split16']),
                                        final.jitter, 0.)
    identity = ['data_seed', 'degrees', 'optimizer', 'train_bias', 'rate_factor',
                'effective_jitter', 'seed', 'arm', 'budget']
    duplicates = final.groupby(identity).test_loss.agg(['min', 'max'])
    assert (duplicates['max']-duplicates['min']).max() < 1e-14
    metadata = dict(ordinary_stability_boundary_degrees=math.degrees((low+high)/2),
                    executed_nonpilot_model_endpoints=len(final),
                    distinct_endpoint_configurations=len(duplicates),
                    distinct_training_configurations=len(final.drop_duplicates(identity[:-1])),
                    repeated_first_five_max_loss_delta=duplicate_delta,
                    max_initial_perturbation_rmse=float(audits.initial_perturbation_rmse_max.max()),
                    resamples=4000, bootstrap_seed=741, independent_units='paired training seeds',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (TABLES/'analysis.json').write_text(json.dumps(metadata, indent=2)+'\n')
    plots(frame)
    print(json.dumps(metadata, indent=2))
    print(summary[summary.collection == 'replication20'].to_string(index=False))


if __name__ == '__main__':
    main()
