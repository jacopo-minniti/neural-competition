"""Plot the rejected curvature count, corrected spectrum, and local loss audit."""

import json

import matplotlib
import numpy as np
from explore_gd_option_dynamics import OUT

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    data = json.loads((OUT/'results.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    metrics = [('current_curvature_options', 'A curvature-count drop is ambiguous', 'Loss-whitened count'),
               ('option_mass', 'Total residual response grows', 'Response mass'),
               ('option_participation', 'Response breadth contracts', 'Effective breadth')]
    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        for width, color in ((32, '#2969a2'), (128, '#be6925')):
            values = [np.mean([r[key] for r in data['measurements'] if r['width']==width and
                              r['checkpoint']==step]) for step in (0, 100, 600)]
            ax.plot([0, 100, 600], values, '-o', color=color, label=f'Width {width}')
        ax.set(title=title, xlabel='Training step', ylabel=ylabel)
        ax.legend()
    ax = axes[1, 1]
    for rate, color in ((.005, '#2969a2'), (.05, '#be6925')):
        rows = [r for r in data['branches'] if r['rate']==rate]
        ax.scatter([r['actual_gain'] for r in rows], [r['predicted_gain'] for r in rows],
                   color=color, alpha=.7, label=f'Step size {rate}')
    ax.plot([.001, 3], [.001, 3], '--', color='gray')
    ax.set(xscale='log', yscale='log', title='Frozen-quadratic loss prediction',
           xlabel='Observed loss reduction', ylabel='Predicted loss reduction')
    ax.legend()
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('MNIST: ordinary SGD checkpoints; response diagnostics at discounted horizon 1', fontsize=11)
    fig.savefig(OUT/'summary.png', dpi=180)


if __name__ == '__main__':
    main()
