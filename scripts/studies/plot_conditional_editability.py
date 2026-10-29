"""Plot the option spectrum, turnover and finite-buffer counterexample."""

import json

import matplotlib
import numpy as np
from explore_conditional_editability import OUT

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    result = json.loads((OUT/'results.json').read_text())
    audit = json.loads((OUT/'audit.json').read_text())
    buffers = json.loads((OUT/'buffer_audit.json').read_text())
    operators = np.load(OUT/'operators.npz')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    for width, color in [(32, '#2969a2'), (128, '#be6925')]:
        spectrum = operators[f'281501_{width}_600_spectrum'][::-1]
        axes[0].plot(np.arange(1, 145), spectrum, color=color, label=f'Width {width}')
        values = [np.mean([r['options_shape_0.1'] for r in result['measurements']
                          if r['width']==width and r['step']==step]) for step in (0, 100, 600)]
        axes[1].plot([0, 100, 600], values, '-o', color=color, label=f'Width {width}')
        items = [r for r in buffers if r['width']==width]
        axes[2].plot([r['anchors'] for r in items], [r['options'] for r in items],
                     '--o', color=color, label=f'{width}: in-sample')
        axes[2].plot([r['anchors'] for r in items], [r['validated_background_value'] for r in items],
                     '-s', color=color, label=f'{width}: held-out value')
    axes[0].set(title='Many conditional response modes', xlabel='Mode (sorted)',
                ylabel='Realizable response fraction', ylim=(0, 1.02))
    axes[1].set(title='Training need not consume options', xlabel='SGD step',
                ylabel='Effective count / 144 possible', ylim=(100, 145))
    axes[2].set(title='More protection data improves validity', xlabel='Background examples',
                ylabel='Count or validated value', xticks=[16, 64])
    for axis in axes:
        axis.legend(fontsize=8)
        axis.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT/'summary.png', dpi=180)
    assert audit['analytic_autodiff_relative_error'] < 1e-10


if __name__ == '__main__':
    main()
