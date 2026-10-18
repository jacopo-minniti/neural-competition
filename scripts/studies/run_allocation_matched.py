#!/usr/bin/env python3
"""Teacher diagnostics at first sampled states within 5% of fixed MSE targets."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import run_allocation as study
import run_temporal_branches as replay
import torch

from neural_competition.trajectory import adam_step, components, momentum_step


def run(condition):
    output = study.ROOT/'runs/allocation/matched'
    output.mkdir(parents=True, exist_ok=True)
    stem = condition.replace('/', '_')
    if (output/f'{stem}.json').exists():
        return
    started = time.monotonic()
    config, data, saved, p = replay.initialize(condition)
    n = len(p)
    widths = torch.tensor([m['width'] for m in config['models']])[:, None]
    seeds = torch.tensor([m['seed'] for m in config['models']])
    norm = widths.double().sqrt().float()
    rngs = [torch.Generator().manual_seed(12026+s) for s in range(int(seeds.max())+1)]
    first, second = torch.zeros_like(p), torch.zeros_like(p)
    cal = torch.randperm(len(data['x_train']), generator=torch.Generator().manual_seed(78191))[:512]
    test = torch.randperm(len(data['x_test']), generator=torch.Generator().manual_seed(78192))[:1024]
    xphase = data['x_train'][cal].double()[None].expand(n, -1, -1)
    yphase = data['y_train'][cal].double()[None]
    targets = [.2, .1]
    found, states, rows = set(), [], []
    adam = config.get('optimizer_spec', {}).get('name') == 'adam'
    for block in range(0, config['steps'], 1000):
        indices = torch.stack([torch.randint(len(data['x_train']), (1000, 64), generator=g)
                               for g in rngs], dim=1)
        stride = 1 if block < 1000 else 10
        for offset in range(0, 1000, stride):
            t = block+offset
            mse = (components(p.double(), xphase, widths.double().sqrt()).sum(-1)-yphase).square().mean(1)
            for ti, target in enumerate(targets):
                selected = [i for i in range(n) if (i,ti) not in found
                            and abs(float(mse[i])/target-1) <= .05]
                if selected:
                    ids = torch.tensor(selected)
                    sp, sf, ss, sw = p[ids], first[ids], second[ids], widths[ids]
                    states.append(dict(completed=t, target_loss=target, model_indices=selected,
                                       parameters=sp.clone(), first=sf.clone(), second=ss.clone()))
                    xc = data['x_train'].double()[None].expand(len(ids), -1, -1)
                    yc = data['y_train'].double()[None].expand(len(ids), -1)
                    xt = data['x_test'][test].double()[None].expand(len(ids), -1, -1)
                    yt = data['y_test'][test].double()[None].expand(len(ids), -1)
                    # Candidate streams are paired across widths, independent of
                    # training and coefficient calibration; frozen moments.
                    for source in range(4):
                        batch = torch.stack([torch.randint(len(data['x_train']), (64,),
                                            generator=torch.Generator().manual_seed(
                                                79600+1000*int(seeds[i])+100*ti+source))
                                             for i in selected])
                        if adam:
                            q, _, _ = adam_step(sp, sf, ss, data['x_train'][batch],
                                               data['y_train'][batch], norm[ids], config['lr'], t+1)
                        else:
                            q, _ = momentum_step(sp, sf, data['x_train'][batch], data['y_train'][batch],
                                                  norm[ids], config['lr'], config['momentum'])
                        measured = study.measure(sp, q, sw, xc, yc, xt, yt, 79700+100*ti+source)
                        for row in measured:
                            i = selected[row.pop('model')]
                            rows.append(dict(condition=condition, **config['models'][i], completed=t,
                                             target_loss=target, source=source,
                                             achieved_mse=float(mse[i]), **row))
                    found.update((i,ti) for i in selected)
            if len(found) == n*len(targets):
                break
            p, first, second = replay.advance(p, first, second, data['x_train'], data['y_train'],
                                              indices[offset:offset+stride], seeds, norm,
                                              config['lr'], config['momentum'], adam, t)
        if len(found) == n*len(targets):
            break
        if block % 10000 == 0:
            print(condition, block, len(found), '/', n*len(targets), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/f'{stem}.csv.gz', index=False, float_format='%.17g')
    torch.save(dict(config=config, states=states), output/f'{stem}.pt')
    coverage = [dict(**m, target_loss=target, found=(i,j) in found)
                for i,m in enumerate(config['models']) for j,target in enumerate(targets)]
    manifest = dict(condition=condition, calibration_size=len(data['x_train']), heldout_size=len(test),
                    match_reference_size=len(cal), mse_targets=targets, tolerance=.05,
                    check_every='each update through 1000, then every 10', coverage=coverage, sources=4,
                    elapsed_seconds=time.monotonic()-started,
                    max_qp_relative_gap=float(frame.qp_relative_gap.max()),
                    source_sha256={str(path.relative_to(study.ROOT)): replay.digest(path)
                                   for path in [Path(__file__),Path(study.__file__),Path(replay.__file__),
                                                study.ROOT/'src/neural_competition/allocation.py']})
    (output/f'{stem}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DONE',condition,len(found),'/',n*len(targets),time.monotonic()-started,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', action='append')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for condition in args.condition or replay.conditions():
        run(condition)


if __name__ == '__main__':
    main()
