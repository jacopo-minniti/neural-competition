"""Summarize the exploratory pressure-law cohorts and check paired step refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from probe_probabilistic_natural_training import train  # noqa: E402


def step_comparison(coarse_dir, fine_dir):
    coarse = json.loads((coarse_dir / 'summary.json').read_text())
    fine = json.loads((fine_dir / 'summary.json').read_text())
    for name in ['w', 'a']:
        np.testing.assert_array_equal(np.load(coarse_dir / 'initial_bank.npz')[name],
                                      np.load(fine_dir / 'initial_bank.npz')[name])
    rows = []
    for setting in fine['protocol']['settings']:
        key = setting['key']
        a = np.load(coarse_dir / f'{key}_actual.npz')
        b = np.load(fine_dir / f'{key}_actual.npz')
        for i, horizon in enumerate(coarse['protocol']['times']):
            rows.append(dict(
                key=key, time=horizon, count=len(a[f't{i}_blind']),
                closure_label_disagreements=int(np.sum(a[f't{i}_blind'] != b[f't{i}_blind'])),
                response_mae=float(np.mean(abs(a[f't{i}_response'] - b[f't{i}_response']))),
                rare_loss_mae=float(np.mean(abs(a[f't{i}_rare_loss'] - b[f't{i}_rare_loss']))),
            ))
    return rows


def long_step_check(folder, count=512):
    """Paired first-seed subset; no independent confirmation claim."""
    protocol = json.loads((folder / 'protocol.json').read_text())
    bank = np.load(folder / 'initial_bank.npz')
    rows = []
    for setting in protocol['settings']:
        width, scale = setting['width'], setting['scale']
        fine = train(scale * bank['w'][:count, :width], scale * bank['a'][:count, :width],
                     np.pi * setting['angle_over_pi'], setting['q'], [0, 20, 200], .005)
        coarse = np.load(folder / f"{setting['key']}_actual.npz")
        for horizon in [20, 200]:
            index = protocol['times'].index(horizon)
            base, check = coarse[f't{index}_blind'][:count], fine[horizon]['blind']
            rows.append(dict(
                **setting, time=horizon, count=count,
                coarse_closed=int(base.sum()), fine_closed=int(check.sum()),
                closure_disagreements=int(np.sum(base != check)),
                coarse_mean_loss=float(coarse[f't{index}_loss'][:count].mean()),
                fine_mean_loss=float(fine[horizon]['loss'].mean()),
                response_mae=float(np.mean(abs(coarse[f't{index}_response'][:count]
                                               - fine[horizon]['response']))),
            ))
        print('Long step check:', setting['key'], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('docs/evidence/probabilistic_competition'))
    parser.add_argument('--long-step-check', action='store_true')
    args = parser.parse_args()
    root = args.root
    primary = root / 'natural_training_pilot_20261030'
    longer = root / 'natural_training_long_followup_20261030'
    data = json.loads((primary / 'summary.json').read_text())
    long_data = json.loads((longer / 'summary.json').read_text())
    rows = data['rows']
    selected = [r for r in rows if r['time'] == 20 and r['scale'] == .2]
    short_checks = []
    for folder in ['natural_training_half_step_20261030',
                   'natural_training_smallq_half_step_20261030']:
        short_checks.extend(step_comparison(primary, root / folder))
    check_path = primary / 'long_step_check.json'
    if args.long_step_check:
        if check_path.exists():
            raise RuntimeError('Preserve existing long-step check')
        check_path.write_text(json.dumps(dict(
            status='Post-pilot numerical sensitivity audit on first 512 paired seeds.',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            rows=long_step_check(longer)), indent=2)+'\n')

    long_traps = []
    for setting in long_data['protocol']['settings']:
        actual = np.load(longer / f"{setting['key']}_actual.npz")
        certified = actual['t4_trap']
        long_traps.append(dict(
            **setting, certified_at20=int(certified.sum()),
            still_closed_at200=int(np.sum(actual['t7_blind'] & certified)),
            minimum_loss_at200_among_certified=float(actual['t7_loss'][certified].min())
            if certified.any() else None,
            theoretical_loss_floor=setting['q']/2,
            mean_loss_at200=float(actual['t7_loss'].mean()),
            mean_rare_loss_at200=float(actual['t7_rare_loss'].mean()),
            newly_closed_at200=int(np.sum(actual['t7_blind'] & ~actual['t0_blind'])),
        ))
    result = dict(
        status='Exploratory analysis, same initial seed bank reused across conditions and follow-ups.',
        primary_full_task_runs=2048 * len(data['protocol']['settings']), independent_seed_banks=2048,
        forecast='Initial-state scalar-clock bridge; not b(mean observed pressure)^m.',
        endpoint_selected=selected, short_step_checks=short_checks, long_traps=long_traps,
        long_followup_selection='q=.1, scale=.2 and T=200 selected after the short-pilot outcomes.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (primary / 'analysis.json').write_text(json.dumps(result, indent=2)+'\n')

    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    blue, orange, green = '#2466A6', '#BD5B32', '#41866b'
    ax = axes[0, 0]
    for width, color in [(4, blue), (8, orange)]:
        group = sorted([r for r in selected if r['angle_over_pi'] > .3 and r['width'] == width],
                       key=lambda r: r['q'])
        x = [r['q'] for r in group]
        y = np.array([r['observed_closed']/r['count'] for r in group])
        lo, hi = np.array([r['observed_closed_ci95'] for r in group]).T
        ax.errorbar(x, y, yerr=[y-lo, hi-y], color=color, marker='o', capsize=3,
                    label=f'Actual, width {width}')
        ax.plot(x, [r['predicted_closed']/r['count'] for r in group], color=color,
                marker='x', linestyle='--', label=f'Forecast, width {width}')
    ax.set(xscale='log', xlabel='Rare-example probability q', ylabel='Closed at time 20',
           title='Closure counts agree when rare pressure is small', ylim=(-.001, .031))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for angle, label, color in [(1/12, '15 degrees', green), (1/3, '60 degrees', blue)]:
        group = [r for r in rows if r['scale'] == .2 and r['width'] == 4
                 and r['q'] == .001 and r['angle_over_pi'] == angle]
        ax.plot([r['time'] for r in group], [r['observed_closed']/r['count'] for r in group],
                color=color, marker='o', label=f'Actual, {label}')
        ax.plot([r['time'] for r in group], [r['predicted_closed']/r['count'] for r in group],
                color=color, linestyle='--', label=f'Forecast, {label}')
    ax.set(xscale='symlog', xlabel='Training time', ylabel='Closed response fraction',
           title='Geometry permits both net reopening and net closure')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for q, color in [(.001, blue), (.01, green), (.1, orange)]:
        group = [r for r in rows if r['scale'] == .2 and r['width'] == 4
                 and r['angle_over_pi'] > .3 and r['q'] == q and r['time'] > 0]
        ax.plot([r['time'] for r in group], [r['response_rmse_ratio'] for r in group],
                color=color, marker='o', label=f'q={q}')
    ax.axhline(1, color='#666666', linestyle='--', linewidth=1)
    ax.set(xscale='log', xlabel='Forecast horizon', ylabel='Response RMSE / no-change RMSE',
           title='Continuous-strength forecasts lose accuracy later', ylim=(0, 1.15))
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    for width, color in [(4, blue), (8, orange)]:
        group = [r for r in long_data['rows'] if r['angle_over_pi'] > .3 and r['width'] == width]
        ax.plot([r['time'] for r in group], [r['mean_rare_loss'] for r in group],
                color=color, marker='o', label=f'Width {width}')
    ax.set(xscale='symlog', yscale='log', xlabel='Training time',
           ylabel='Mean loss conditional on rare examples',
           title='Longer full-task follow-up: q=0.1, 60 degrees')
    ax.legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=.16)
    fig.suptitle('Pressure-model pilot under ordinary full-task training', fontsize=15)
    fig.supxlabel('2,048 paired Gaussian seed banks; scale 0.2 shown. '
                  'Upper-left bars: binomial 95% intervals.\n'
                  'Exploratory cohorts; the longer follow-up was selected after the short pilot.', fontsize=9)
    fig.savefig(primary / 'summary.png', dpi=170)
    fig.savefig(primary / 'summary.svg')
    plt.close(fig)
    print(primary / 'summary.png', flush=True)


if __name__ == '__main__':
    main()
