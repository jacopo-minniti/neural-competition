#!/usr/bin/env python3
"""Repeat all teacher probes using all training and the identical test examples.

This is a calibration-size sensitivity, with the same states and proposals.
It does not add independent training runs or new optimization trajectories.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import run_allocation as study
import run_temporal_branches as replay
import torch

from neural_competition.trajectory import adam_step, momentum_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', action='append')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = study.ROOT/'runs/allocation/full_calibration'
    output.mkdir(parents=True, exist_ok=True)
    for condition in args.condition or replay.conditions():
        stem = condition.replace('/', '_')
        path = study.OUTPUT/'full'/f'{stem}.pt'
        if not path.exists():
            raise FileNotFoundError(f'Baseline replay must finish first: {path}')
        if (output/f'{stem}.json').exists():
            continue
        saved = torch.load(path, weights_only=True)
        config = saved['config']
        family, label = condition.split('/', 1)
        dataset = 'k8_teacher2027' if family == 'trajectory' else label.split('/')[0].split('_init')[0]
        dp = replay.ARCHIVE/'datasets'/f'{dataset}.pt'
        data = torch.load(dp, weights_only=True)
        widths = torch.tensor([m['width'] for m in config['models']])[:, None]
        norm = widths.double().sqrt().float()
        n = len(widths)
        xc = data['x_train'].double()[None].expand(n, -1, -1)
        yc = data['y_train'].double()[None].expand(n, -1)
        test = saved['test_indices']
        xt = data['x_test'][test].double()[None].expand(n, -1, -1)
        yt = data['y_test'][test].double()[None].expand(n, -1)
        is_adam = config.get('optimizer_spec', {}).get('name') == 'adam'
        rows = []
        started = time.monotonic()
        for j, state in enumerate(saved['snapshots']):
            p, first, second = state['parameters'], state['first'], state['second']
            ids, t = state['source_indices'], state['completed']
            if is_adam:
                following, _, _ = adam_step(p, first, second, data['x_train'][ids],
                                            data['y_train'][ids], norm, config['lr'], t+1)
            else:
                following, _ = momentum_step(p, first, data['x_train'][ids], data['y_train'][ids],
                                              norm, config['lr'], config['momentum'])
            measured = study.measure(p, following, widths, xc, yc, xt, yt, 79193+j)
            for row in measured:
                i = row.pop('model')
                rows.append(dict(condition=condition, **config['models'][i], completed=t,
                                 probe_id=j, **row))
            if j % 5 == 0:
                print(condition, j+1, time.monotonic()-started, flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(output/f'{stem}.csv.gz', index=False, float_format='%.17g')
        manifest = dict(condition=condition, calibration_size=len(data['x_train']),
                        heldout_size=len(test), elapsed_seconds=time.monotonic()-started,
                        source_sha256={str(p.relative_to(study.ROOT)): replay.digest(p) for p in [
                            Path(__file__), Path(study.__file__),
                            study.ROOT/'src/neural_competition/allocation.py']},
                        input_sha256={str(p.relative_to(study.ROOT)): replay.digest(p) for p in [path, dp]},
                        max_qp_relative_gap=float(frame.qp_relative_gap.max()))
        (output/f'{stem}.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print('DONE', condition, flush=True)


if __name__ == '__main__':
    main()
