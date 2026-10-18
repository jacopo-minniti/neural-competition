#!/usr/bin/env python3
"""Seed-level summaries of local cross-fitted allocation and residual work."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'docs/tables/allocation'
FIGURES = ROOT/'docs/figures/allocation'


def interval(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(741)
    draws = a[rng.integers(len(a), size=(4000, len(a)))].mean(1)
    lo, hi = np.quantile(draws, [.025, .975])
    return a.mean(), lo, hi


def episodes(data, keys):
    arms = {name: group.set_index(keys).sort_index() for name, group in data.groupby('arm')}
    ordinary = arms['ordinary']
    assert all(arm.index.equals(ordinary.index) for arm in arms.values())
    columns = ['loss', 'calibration_before', 'positive_work', 'negative_work', 'work',
               'curvature', 'corrective_cosine', 'progress', 'oracle_allocation_gain',
               'qp_relative_gap', 'group_relative_gap']
    frame = ordinary[columns].copy()
    for name in ['neuron', 'sign', 'group4', 'shuffled', 'ordinary']:
        frame[name+'_benefit'] = arms['uniform_grid'].heldout_after-arms[name].heldout_after
    frame['beyond_sign'] = arms['sign'].heldout_after-arms['neuron'].heldout_after
    frame['beyond_shuffled'] = arms['shuffled'].heldout_after-arms['neuron'].heldout_after
    frame['uniform_gain'] = ordinary.loss-arms['uniform_grid'].heldout_after
    frame['neuron_gain'] = ordinary.loss-arms['neuron'].heldout_after
    frame['linearization_error'] = (arms['neuron'].heldout_after-ordinary.loss
                                    - arms['neuron'].predicted_heldout_change)
    frame['calibration_neuron_benefit'] = (arms['uniform_grid'].calibration_after
                                         - arms['neuron'].calibration_after)
    return frame.reset_index()


def pooled(part, scope):
    rows = []
    for key, group in part.groupby(['condition', 'width', 'seed', 'fraction']):
        row = dict(zip(['condition', 'width', 'seed', 'fraction'], key), scope=scope,
                   episodes=len(group), checkpoints=group.completed.nunique())
        positive = group.positive_work.sum()
        row['positive_work_sum'] = positive
        row['negative_work_sum'] = group.negative_work.sum()
        row['mean_loss'] = group.loss.mean()
        row['anti_work_pct'] = 100*group.negative_work.sum()/positive if positive > 0 else np.nan
        row['curvature_pct'] = 100*group.curvature.sum()/positive if positive > 0 else np.nan
        row['corrective_cosine_pct'] = 100*group.corrective_cosine.mean()
        for name in ['neuron_benefit', 'sign_benefit', 'group4_benefit', 'shuffled_benefit',
                     'beyond_sign', 'beyond_shuffled', 'uniform_gain', 'neuron_gain',
                     'calibration_neuron_benefit', 'oracle_allocation_gain', 'progress',
                     'linearization_error']:
            row[name+'_raw'] = group[name].mean()
            row[name+'_work_pct'] = 100*group[name].sum()/positive if positive > 0 else np.nan
            row[name+'_loss_pct'] = 100*(group[name]/group.loss).mean()
        row['neuron_win_fraction'] = (group.neuron_benefit > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_seed):
    metrics = [c for c in per_seed if c.endswith(('_pct', '_raw', '_fraction'))]
    rows, contrasts = [], []
    for key, group in per_seed.groupby(['condition', 'scope', 'fraction', 'width']):
        meta = dict(zip(['condition', 'scope', 'fraction', 'width'], key))
        for metric in metrics:
            mean, lo, hi = interval(group[metric])
            rows.append(dict(**meta, metric=metric, mean=mean, low=lo, high=hi,
                             seeds=group.seed.nunique(), episodes=int(group.episodes.sum())))
    for key, group in per_seed.groupby(['condition', 'scope', 'fraction']):
        low_width, high_width = group.width.min(), group.width.max()
        if low_width == high_width:
            continue
        low = group[group.width == low_width].set_index('seed')
        high = group[group.width == high_width].set_index('seed')
        common = low.index.intersection(high.index)
        for metric in metrics:
            values = high.loc[common, metric]-low.loc[common, metric]
            mean, lo, hi = interval(values)
            contrasts.append(dict(zip(['condition', 'scope', 'fraction'], key), metric=metric,
                                  narrow=int(low_width), wide=int(high_width), mean=mean,
                                  low=lo, high=hi, paired_seeds=len(common)))
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def plot(summary, prefix, scope, conditions, labels):
    fig, axes = plt.subplots(1, 3, figsize=(13, max(3.8, len(conditions)*.55)), sharey=True)
    metrics = ['anti_work_pct', 'neuron_benefit_work_pct', 'beyond_sign_work_pct']
    titles = ['Anti-residual work / useful work', 'Allocation gain beyond uniform',
              'Allocation gain beyond sign filter']
    for ax, metric, title in zip(axes, metrics, titles):
        for j, condition in enumerate(conditions):
            part = summary[(summary.condition == condition) & (summary.scope == scope)
                           & (summary.fraction == 1) & (summary.metric == metric)].sort_values('width')
            if part.empty:
                continue
            selected = part.iloc[[0, -1]]
            for k, (_, row) in enumerate(selected.iterrows()):
                ax.errorbar(row['mean'], j+(k-.5)*.22,
                            xerr=[[max(0,row['mean']-row.low)], [max(0,row.high-row['mean'])]],
                            fmt='o', color=['#2563eb', '#e36b27'][k], capsize=3,
                            label=['Narrow', 'Widest'][k] if j == 0 else None)
        ax.axvline(0, color='#999', lw=.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Units per 100 useful work units')
        ax.grid(axis='x', alpha=.15)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False)
    fig.suptitle(f'{prefix}: held-out exact one-step outcomes ({scope})', fontsize=12)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(FIGURES/f'{prefix}_{scope}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    teacher_paths = sorted((ROOT/'runs/allocation/full').glob('*.csv.gz'))
    mnist_paths = sorted((ROOT/'runs/allocation_mnist/full').glob('*.csv.gz'))
    all_seed = []
    manifest = {}
    if teacher_paths:
        data = pd.concat([pd.read_csv(p) for p in teacher_paths], ignore_index=True)
        keys = ['condition', 'width', 'seed', 'completed', 'probe_id', 'fraction']
        frame = episodes(data, keys)
        scopes = dict(all_sampled=frame, after_100=frame[frame.completed >= 100],
                      late=frame[frame.completed >= 1000],
                      shared_loss_band=frame[(2*frame.calibration_before).between(.0766, .2)])
        for scope, part in scopes.items():
            all_seed.append(pooled(part, scope))
        frame.to_csv(OUTPUT/'teacher_episodes.csv.gz', index=False, float_format='%.17g')
        manifests = [json.loads(p.with_suffix('').with_suffix('.json').read_text()) for p in teacher_paths]
        manifest['teacher'] = dict(conditions=len(teacher_paths),
                                   trajectories=sum(len(m['models']) for m in manifests),
                                   max_endpoint_error=max(m['endpoint_error'] for m in manifests),
                                   checkpoint_audits=sum(len(m['baseline_audits']) for m in manifests),
                                   max_qp_relative_gap=float(frame.qp_relative_gap.max()),
                                   max_group_relative_gap=float(frame.group_relative_gap.max()))
        (OUTPUT/'teacher_provenance.json').write_text(json.dumps(manifests, indent=2)+'\n')
        full_paths = sorted((ROOT/'runs/allocation/full_calibration').glob('*.csv.gz'))
        if full_paths:
            full = pd.concat([pd.read_csv(p) for p in full_paths], ignore_index=True)
            full = episodes(full, keys)
            # Sensitivity uses identical state membership, selected by the
            # original 512-example calibration loss, not by new outcomes.
            selected = frame[keys+['calibration_before']].rename(
                columns={'calibration_before': 'selection_calibration_before'})
            full = full.merge(selected, on=keys, validate='one_to_one')
            for scope, part in dict(
                fullcal_all_sampled=full,
                fullcal_late=full[full.completed >= 1000],
                fullcal_shared_loss_band=full[(2*full.selection_calibration_before).between(.0766,.2)],
            ).items():
                all_seed.append(pooled(part, scope))
            full.to_csv(OUTPUT/'teacher_fullcal_episodes.csv.gz', index=False, float_format='%.17g')
            full_manifests = [json.loads(p.with_suffix('').with_suffix('.json').read_text())
                              for p in full_paths]
            (OUTPUT/'teacher_fullcal_provenance.json').write_text(json.dumps(full_manifests, indent=2)+'\n')
            manifest['full_calibration'] = dict(conditions=len(full_paths),
                                                max_qp_relative_gap=float(full.qp_relative_gap.max()))
    if mnist_paths:
        data = pd.concat([pd.read_csv(p) for p in mnist_paths], ignore_index=True)
        data['condition'] = 'mnist/'+data.loss_kind+'/'+data.optimizer
        keys = ['condition', 'width', 'seed', 'completed', 'target_loss', 'source', 'fraction']
        frame = episodes(data, keys)
        # Keep only paired target coverage; one known narrow SGD target is missing.
        counts = frame.groupby(['condition', 'seed', 'target_loss', 'source', 'fraction']).width.transform('nunique')
        paired = frame[counts == 2]
        all_seed.append(pooled(paired, 'matched_targets'))
        all_seed.append(pooled(frame, 'all_available_targets'))
        for target in sorted(frame.target_loss.unique()):
            all_seed.append(pooled(paired[paired.target_loss == target], f'target_{target:g}'))
        frame.to_csv(OUTPUT/'mnist_episodes.csv.gz', index=False, float_format='%.17g')
        manifest['mnist'] = dict(checkpoints=len(mnist_paths),
                                 episodes=len(frame), paired_episodes=len(paired),
                                 max_qp_relative_gap=float(frame.qp_relative_gap.max()),
                                 max_group_relative_gap=float(frame.group_relative_gap.max()))
        original = ROOT/'runs/allocation_mnist/full/manifest.json'
        if original.exists():
            (OUTPUT/'mnist_provenance.json').write_bytes(original.read_bytes())
    matched_paths = sorted((ROOT/'runs/allocation/matched').glob('*.csv.gz'))
    if matched_paths:
        data = pd.concat([pd.read_csv(p) for p in matched_paths], ignore_index=True)
        keys = ['condition','width','seed','completed','target_loss','source','fraction']
        frame = episodes(data, keys)
        manifests = [json.loads(p.with_suffix('').with_suffix('.json').read_text()) for p in matched_paths]
        expected = {m['condition']: len(set(c['width'] for c in m['coverage'])) for m in manifests}
        counts = frame.groupby(['condition','seed','target_loss','source','fraction']).width.transform('nunique')
        paired = frame[counts == frame.condition.map(expected)]
        all_seed.append(pooled(paired,'teacher_matched_targets'))
        for target in sorted(frame.target_loss.unique()):
            all_seed.append(pooled(paired[paired.target_loss == target],f'teacher_target_{target:g}'))
        frame.to_csv(OUTPUT/'teacher_matched_episodes.csv.gz',index=False,float_format='%.17g')
        coverage = pd.DataFrame([dict(condition=m['condition'],**c) for m in manifests for c in m['coverage']])
        coverage.to_csv(OUTPUT/'teacher_matched_coverage.csv',index=False)
        (OUTPUT/'teacher_matched_provenance.json').write_text(json.dumps(manifests,indent=2)+'\n')
        manifest['teacher_matched'] = dict(conditions=len(matched_paths),
                                           found_targets=int(coverage.found.sum()), intended_targets=len(coverage),
                                           episodes=len(frame), paired_episodes=len(paired),
                                           max_qp_relative_gap=float(frame.qp_relative_gap.max()))
    per_seed = pd.concat(all_seed, ignore_index=True)
    summary, contrasts = summarize(per_seed)
    per_seed.to_csv(OUTPUT/'per_seed.csv.gz', index=False, float_format='%.17g')
    summary.to_csv(OUTPUT/'summary.csv.gz', index=False, float_format='%.17g')
    contrasts.to_csv(OUTPUT/'width_contrasts.csv.gz', index=False, float_format='%.17g')
    manifest['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw_paths = [*teacher_paths, *mnist_paths, *matched_paths,
                 *sorted((ROOT/'runs/allocation/full_calibration').glob('*.csv.gz'))]
    manifest['raw_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in raw_paths}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    teacher = ['trajectory/sgd', 'robustness/k8_teacher2027/momentum_01',
               'robustness/k8_teacher2027/adam_001', 'robustness/k8_teacher2027/adam_01',
               'robustness/k8_teacher2027_init01/adam_001',
               'robustness/k8_teacher2027_init1/adam_001']
    labels = ['SGD 1', 'Momentum .1', 'Adam .001 (zero)', 'Adam .01 (zero)',
              'Adam .001 (init .1)', 'Adam .001 (init 1)']
    plot(summary, 'teacher', 'shared_loss_band', teacher, labels)
    if matched_paths:
        plot(summary, 'teacher', 'teacher_matched_targets', teacher, labels)
    mnist = [f'mnist/{loss}/{opt}' for loss in ['mse', 'ce'] for opt in ['sgd', 'momentum', 'adam']]
    plot(summary, 'MNIST', 'matched_targets', mnist,
         [f'{loss.upper()} {opt}' for loss in ['mse', 'ce'] for opt in ['sgd', 'momentum', 'adam']])
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
