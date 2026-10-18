#!/usr/bin/env python3
"""Replay complete teacher cohorts and cross-fit one-step allocation diagnostics."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_temporal_branches as replay
import torch

from neural_competition.allocation import (
    allocation_rules,
    linearized_changes,
    objective,
    quadratic_terms,
    work_statistics,
)
from neural_competition.trajectory import adam_step, components, momentum_step

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/allocation"


def measure(p, following, widths, xc, yc, xt, yt, seed):
    """Choose coefficients on training calibration; evaluate exact held-out loss."""
    norm = widths.double().sqrt()
    p64, delta = p.double(), (following.double() - p.double())
    base_c = components(p64, xc, norm)
    base_t = components(p64, xt, norm)
    ec, et = base_c.sum(-1) - yc, base_t.sum(-1) - yt
    rows = []
    for fraction in [1.0, 0.01]:
        dp = fraction * delta
        ell = linearized_changes(p64, dp, xc, norm)
        b, gram = quadratic_terms(ec, ell)
        rules, cert = allocation_rules(b, gram, widths, seed=seed)
        scale = b.abs().sum(-1) + gram.abs().sum((-1, -2))
        # Discrete exact-forward scalar control includes both endpoints and the
        # analytic uniform coefficient. It is grid-best, not a global optimum.
        best = torch.full((len(p),), float('inf'), dtype=torch.float64)
        grid_gamma = torch.zeros_like(best)
        for gamma in torch.linspace(0, 1, 33, dtype=torch.float64):
            prediction = components(p64 + gamma * dp, xc, norm).sum(-1)
            loss = 0.5 * (prediction - yc).square().mean(1)
            improve = loss < best
            best = torch.minimum(best, loss)
            grid_gamma[improve] = gamma
        uq = p64 + rules['uniform'][:, :, None] * dp
        uloss = 0.5 * (components(uq, xc, norm).sum(-1) - yc).square().mean(1)
        improve = uloss < best
        grid_gamma[improve] = rules['uniform'][improve, 0]
        rules['uniform_grid'] = grid_gamma[:, None].expand_as(b)
        rules['ordinary'] = torch.ones_like(b)
        finite = components(p64 + dp, xt, norm) - base_t
        stats = work_statistics(et, finite)
        ell_t = linearized_changes(p64, dp, xt, norm)
        bt, gt = quadratic_terms(et, ell_t)
        q_uniform = objective(b, gram, rules['uniform'])
        q_neuron = objective(b, gram, rules['neuron'])
        assert torch.all(q_neuron <= q_uniform + 1e-12)
        for arm, alpha in rules.items():
            q = p64 + alpha[:, :, None] * dp
            after_c = components(q, xc, norm).sum(-1)
            after_t = components(q, xt, norm).sum(-1)
            exact_c = 0.5 * (after_c - yc).square().mean(1)
            exact_t = 0.5 * (after_t - yt).square().mean(1)
            predicted_c = objective(b, gram, alpha)
            predicted_t = objective(bt, gt, alpha)
            for i in range(len(p)):
                row = dict(model=i, fraction=fraction, arm=arm,
                           calibration_before=float(0.5 * ec[i].square().mean()),
                           calibration_after=float(exact_c[i]), heldout_after=float(exact_t[i]),
                           predicted_calibration_change=float(predicted_c[i]),
                           predicted_heldout_change=float(predicted_t[i]),
                           coefficient_mean=float(alpha[i, :int(widths[i, 0])].mean()),
                           qp_gap=float(cert['gap'][i]),
                           qp_relative_gap=float(cert['gap'][i] / scale[i].clamp_min(1e-300)),
                           group_relative_gap=float(cert['group_gap'][i] / scale[i].clamp_min(1e-300)),
                           qp_sweeps=cert['sweeps'],
                           oracle_allocation_gain=float(q_uniform[i] - q_neuron[i]))
                row.update({key: float(value[i]) for key, value in stats.items()})
                rows.append(row)
    return rows


def run(condition, limit=None):
    started = time.monotonic()
    replay.HASHES = {}
    config, data, saved, p = replay.initialize(condition)
    n = len(p)
    widths = torch.tensor([m['width'] for m in config['models']])[:, None]
    seeds = torch.tensor([m['seed'] for m in config['models']])
    norm = widths.double().sqrt().float()
    generators = [torch.Generator().manual_seed(12026 + s) for s in range(int(seeds.max()) + 1)]
    first, second = torch.zeros_like(p), torch.zeros_like(p)
    cal = torch.randperm(len(data['x_train']), generator=torch.Generator().manual_seed(78191))[:512]
    test = torch.randperm(len(data['x_test']), generator=torch.Generator().manual_seed(78192))[:1024]
    xc = data['x_train'][cal].double()[None].expand(n, -1, -1)
    yc = data['y_train'][cal].double()[None].expand(n, -1)
    xt = data['x_test'][test].double()[None].expand(n, -1, -1)
    yt = data['y_test'][test].double()[None].expand(n, -1)
    is_adam = config.get('optimizer_spec', {}).get('name') == 'adam'
    total = config['steps']
    times = replay.probe_times(total)
    if limit:
        times = times[:limit]
    rows, audits = [], []
    stem = condition.replace('/', '_')
    destination = OUTPUT / ('pilot' if limit else 'full')
    destination.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for block in range(0, total, 1000):
        indices = torch.stack([
            torch.randint(len(data['x_train']), (min(1000, total - block), 64), generator=g)
            for g in generators
        ], dim=1)
        position = 0
        for probe_id in np.flatnonzero((times >= block) & (times < block + len(indices))):
            t = int(times[probe_id])
            offset = t - block
            p, first, second = replay.advance(
                p, first, second, data['x_train'], data['y_train'], indices[position:offset],
                seeds, norm, config['lr'], config['momentum'], is_adam, block + position)
            position = offset
            ids = indices[offset].index_select(0, seeds)
            if is_adam:
                following, _, _ = adam_step(p, first, second, data['x_train'][ids],
                                            data['y_train'][ids], norm, config['lr'], t + 1)
            else:
                following, _ = momentum_step(p, first, data['x_train'][ids],
                                              data['y_train'][ids], norm,
                                              config['lr'], config['momentum'])
            snapshots.append(dict(completed=t, parameters=p.clone(), first=first.clone(),
                                  second=second.clone(), source_indices=ids))
            records = measure(p, following, widths, xc, yc, xt, yt, 79193 + int(probe_id))
            for row in records:
                i = row.pop('model')
                rows.append(dict(condition=condition, **config['models'][i],
                                 completed=t, probe_id=int(probe_id), **row))
            print(f'{condition}: {probe_id + 1}/{len(times)} t={t} '
                  f'{time.monotonic()-started:.1f}s', flush=True)
        p, first, second = replay.advance(
            p, first, second, data['x_train'], data['y_train'], indices[position:], seeds, norm,
            config['lr'], config['momentum'], is_adam, block + position)
        check = replay.audit_checkpoint(condition, block + len(indices), p, first, second,
                                        generators, saved)
        if check:
            audits.append(check)
        if limit and block + len(indices) > times[-1]:
            break
    endpoint_error = None
    if not limit:
        family, label = condition.split('/', 1)
        source = ('robustness_per_seed_results.csv' if family == 'robustness'
                  else 'cumulative_interference_seeds.csv')
        path = ROOT / 'docs/tables' / source
        replay.HASHES[str(path.relative_to(ROOT))] = replay.digest(path)
        table = pd.read_csv(path)
        table = table[table.view == 'reference']
        table = (table[table.condition == label] if family == 'robustness'
                 else table[table.optimizer == label])
        ref = saved['reference_indices']
        xr = data['x_train'][ref].double()[None].expand(n, -1, -1)
        yr = data['y_train'][ref].double()[None]
        actual = 0.5 * (components(p.double(), xr, widths.double().sqrt()).sum(-1)-yr).square().mean(1)
        expected = table.set_index(['width', 'seed']).final_loss
        endpoint_error = max(abs(float(actual[i])-expected.loc[m['width'], m['seed']])
                             for i, m in enumerate(config['models']))
        assert endpoint_error < 1e-10
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / f'{stem}.csv.gz', index=False, float_format='%.17g')
    torch.save(dict(config=config, calibration_indices=cal, test_indices=test, snapshots=snapshots),
               destination / f'{stem}.pt')
    provenance = dict(condition=condition, models=config['models'], checkpoints=times.tolist(),
                      calibration_size=512, heldout_size=1024, pilot_limit=limit,
                      baseline_audits=audits, endpoint_error=endpoint_error,
                      elapsed_seconds=time.monotonic()-started,
                      input_sha256=replay.HASHES,
                      source_sha256={str(p.relative_to(ROOT)): replay.digest(p) for p in [
                          Path(__file__), ROOT / 'src/neural_competition/allocation.py',
                          ROOT / 'src/neural_competition/trajectory.py', Path(replay.__file__)]},
                      max_qp_relative_gap=float(frame.qp_relative_gap.max()),
                      max_group_relative_gap=float(frame.group_relative_gap.max()),
                      dtype='float32 original training; float64 diagnostic interventions',
                      torch_version=str(torch.__version__))
    (destination / f'{stem}.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'DONE {condition}: {len(rows)} rows', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', action='append')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for condition in args.condition or replay.conditions():
        destination = OUTPUT / ('pilot' if args.limit else 'full')
        if (destination / (condition.replace('/', '_') + '.json')).exists():
            print('SKIP completed', condition, flush=True)
            continue
        run(condition, args.limit)


if __name__ == '__main__':
    main()
