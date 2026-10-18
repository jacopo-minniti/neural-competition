#!/usr/bin/env python3
"""Full-cohort late-state readout refits, with held-out evaluation.

Use the final sampled state from each original trajectory, not a selection of
failures. Least squares is solved in float64 on all original training examples.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    output = ROOT/'docs/tables/allocation'
    output.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for path in sorted((ROOT/'runs/allocation/full').glob('*.pt')):
        hashes[str(path.relative_to(ROOT))] = digest(path)
        saved = torch.load(path, weights_only=True)
        config, state = saved['config'], saved['snapshots'][-1]
        condition = json.loads(path.with_suffix('.json').read_text())['condition']
        family, label = condition.split('/', 1)
        dataset = 'k8_teacher2027' if family == 'trajectory' else label.split('/')[0].split('_init')[0]
        dp = ROOT/'runs/retained/datasets'/f'{dataset}.pt'
        hashes[str(dp.relative_to(ROOT))] = digest(dp)
        data = torch.load(dp, weights_only=True)
        x, y = data['x_train'].double(), data['y_train'].double()
        xt, yt = data['x_test'].double(), data['y_test'].double()
        for i, model in enumerate(config['models']):
            width = model['width']
            p = state['parameters'][i, :width].double()
            phi = (x @ p[:, :20].T + p[:, 20]).relu()/width**.5
            phit = (xt @ p[:, :20].T + p[:, 20]).relu()/width**.5
            ordinary = phi @ p[:, -1]
            ordinary_t = phit @ p[:, -1]
            fit = torch.linalg.lstsq(phi, y, rcond=1e-10, driver='gelsd')
            residual = phi @ fit.solution-y
            loss = float(.5*(ordinary-y).square().mean())
            floor = float(.5*residual.square().mean())
            assert floor <= loss + 1e-10
            rows.append(dict(condition=condition, **model, completed=state['completed'],
                             train_loss=loss, readout_train_floor=floor,
                             readout_gap=loss-floor, floor_fraction=floor/loss if loss else float('nan'),
                             test_loss=float(.5*(ordinary_t-yt).square().mean()),
                             readout_test_loss=float(.5*(phit @ fit.solution-yt).square().mean()),
                             rank=int(fit.rank), max_singular=float(fit.singular_values.max()),
                             min_singular=float(fit.singular_values.min())))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'readout_per_seed.csv', index=False, float_format='%.17g')
    frame.groupby(['condition','width']).mean(numeric_only=True).reset_index().to_csv(
        output/'readout_summary.csv',index=False,float_format='%.17g')
    manifest = dict(models=len(frame), rcond=1e-10, solver='torch.linalg.lstsq gelsd float64',
                    input_sha256=hashes, source_sha256=digest(Path(__file__)))
    (output/'readout_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(frame.groupby(['condition','width'])[['train_loss','readout_train_floor','floor_fraction']]
          .mean().round(6).to_string())


if __name__ == '__main__':
    main()
