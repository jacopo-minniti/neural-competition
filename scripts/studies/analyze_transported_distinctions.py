"""Audit analytic costs, dynamics, and numerical accuracy; plot held-out forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from explore_distinction_dynamics import spectral_forecast
from explore_stable_distinctions import linear_span, task
from explore_transported_gates import initialize, losses, train

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / 'docs/theory'


def main():
    x, p, y = task(9, .15, .1)
    w = np.array([[[0., 1., 0.], [0., -1., 0.]]])
    s, v = .15**2/3, 16*.1*.9
    formula = .5*(v*s/(4*(v+s)) + 4*.1*.9*s)
    projection = float(linear_span(w, x, p, y)[0])
    assert abs(formula-projection) < 1e-12
    state = initialize(10, 1, 4)
    predicted = spectral_forecast(state, x, p, y, 1e-6)
    actual = losses(train(state, x, p, y, 1e-6, dt=1e-6), x, p, y)
    derivative_error = float(abs(predicted-actual).max())
    assert derivative_error < 1e-9

    cp = np.load(ROOT/'distinction_spectral_checkpoints.npz')
    ep = np.load(ROOT/'distinction_spectral_endpoints.npz')
    state = tuple(cp[f'm16_{name}'][:8] for name in ('w', 'a', 'c'))
    xc, pc, yc = task(3, .15, .1)
    refined = losses(train(state, xc, pc, yc, 1900, dt=.025), xc, pc, yc)
    default = ep['m16_loss'][:8]

    raw = np.load(ROOT/'transported_gates_validation.npz')
    dynamics = []
    # This tolerance distinguishes floating-point zero in the audit, not a role definition.
    for key in ('context_2_m4', 'attribute_2_m8', 'attribute_5_m8'):
        initial, early, final = (raw[f'{key}_{name}'] for name in ('initial', 'early', 'loss'))
        dynamics.append(dict(case=key, newly_positive=int(np.sum((initial<1e-8)&(early>1e-8))),
            recovered_after_early=int(np.sum((early>1e-8)&(final<early/10))),
            early_mean=float(early.mean()), final_mean=float(final.mean())))
    result = dict(regional_affine_formula=formula, regional_affine_projection=projection,
        infinitesimal_kernel_error=derivative_error,
        continuation_step_audit=dict(seeds=8, width=16, default_dt=.05, refined_dt=.025,
            default_mean=float(default.mean()), refined_mean=float(refined.mean()),
            mean_absolute_difference=float(abs(default-refined).mean())), dynamics=dynamics)
    (ROOT/'transported_distinctions_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

    rows = json.loads((ROOT/'distinction_spectral_validation.json').read_text())['validation']
    widths = [r['width'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), layout='constrained')
    for name, label, marker in [('floor', 'Early span only', 's'),
                                ('spectral', 'Early spectrum', '^')]:
        axes[0].plot(widths, [r[name]['mean'] for r in rows], marker=marker, label=label)
    axes[0].errorbar(widths, [r['mean_loss'] for r in rows],
                     yerr=[r['se'] for r in rows], marker='o', label='Observed ± SE')
    axes[0].set(yscale='log', xlabel='Width (4 suffice for exact representation)',
                ylabel='Half mean squared error', xticks=widths,
                title='Continuous clusters: prospective forecasts')
    axes[0].legend(fontsize=8)
    axes[1].scatter(ep['m16_loss'], cp['m16_floor'], label='Early span only', alpha=.7)
    axes[1].scatter(ep['m16_loss'], cp['m16_spectral'], label='Early spectrum', alpha=.7)
    bound = max(float(ep['m16_loss'].max()), float(cp['m16_spectral'].max()))*1.05
    axes[1].plot([0, bound], [0, bound], color='gray', linestyle='--')
    axes[1].set(xlabel='Observed loss', ylabel='Predicted loss',
                title='Width 16: 48 held-out initializations')
    axes[1].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    axes[1].legend(fontsize=8)
    fig.savefig(ROOT/'transported_distinctions.png', dpi=180)


if __name__ == '__main__':
    main()
