"""Exploratory full-task test of the three-prototype pressure approximation.

Forecasts use initial parameters and a scalar common-only clock. No full-task
future is used in them. Observed-clock reconstructions are explicitly diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np


def transported(w, a, exposure):
    """Existing exact common-pressure flow, allowing one clock per network."""
    u = np.asarray(exposure)[..., None]
    active = w[..., 0] > 0
    along = w[..., 0] * np.cosh(u) + a * np.sinh(u)
    readout = a * np.cosh(u) + w[..., 0] * np.sinh(u)
    hit = active & (along <= 0)
    result = w.copy()
    result[..., 0] = np.where(active, np.maximum(along, 0), w[..., 0])
    coefficient = np.where(active, readout, a)
    coefficient = np.where(
        hit, -np.sqrt(np.maximum(a * a - w[..., 0] ** 2, 0)), coefficient
    )
    return result, coefficient


def clock_velocity(w, a, exposure, q):
    wt, at = transported(w, a, exposure)
    prediction = exposure + np.mean(at * np.maximum(wt[..., 0], 0), axis=-1)
    return (1 - q) * (4 - prediction)


def forecast(w, a, q, times, step):
    """RK4 integration of the scalar pressure clock, not network training."""
    exposure = np.zeros(len(w))
    snapshots = {0.0: exposure.copy()}
    indices = {round(t / step): t for t in times}
    for k in range(1, round(max(times) / step) + 1):
        first = clock_velocity(w, a, exposure, q)
        second = clock_velocity(w, a, exposure + step * first / 2, q)
        third = clock_velocity(w, a, exposure + step * second / 2, q)
        fourth = clock_velocity(w, a, exposure + step * third, q)
        exposure += step * (first + 2 * second + 2 * third + fourth) / 6
        if k in indices:
            snapshots[indices[k]] = exposure.copy()
    return snapshots


def fields(w, a, c, angle, q, common_only=False):
    cosine, sine = np.cos(angle), np.sin(angle)
    pre = np.stack(
        [w[..., 0], -cosine * w[..., 0] + sine * w[..., 1],
         -cosine * w[..., 0] - sine * w[..., 1]], axis=-1
    )
    activation = np.maximum(pre, 0)
    residual = np.array([4.0, 1.0, -1.0]) - c[..., None] - np.mean(
        a[..., None] * activation, axis=-2
    )
    pressure = residual * np.array([1 - q, q / 2, q / 2])
    if common_only:
        pressure[..., 1:] = 0
    gated = (pre > 0) * pressure[..., None, :]
    dw = np.stack(
        [a * (gated[..., 0] - cosine * (gated[..., 1] + gated[..., 2])),
         a * sine * (gated[..., 1] - gated[..., 2])], axis=-1
    )
    da = np.sum(activation * pressure[..., None, :], axis=-1)
    return dw, da, pressure.sum(-1), residual, pre


def measure(w, a, c, angle, q, tau=20.0):
    _, _, _, residual, pre = fields(w, a, c, angle, q)
    x = np.array([[1.0, 0.0], [-np.cos(angle), np.sin(angle)],
                  [-np.cos(angle), -np.sin(angle)]])
    root = np.sqrt([1 - q, q / 2, q / 2])
    h, ga = np.maximum(pre, 0), (pre > 0) * a[..., None]
    kernel = (
        np.swapaxes(h, -1, -2) @ h
        + (np.swapaxes(ga, -1, -2) @ ga) * (x @ x.T)
    ) / a.shape[-1] + 1
    kernel *= root[:, None] * root[None, :]
    eigenvalue, eigenvector = np.linalg.eigh(kernel)
    direction = np.array([0.0, 1.0, -1.0]) / np.sqrt(2)
    loading = (np.swapaxes(eigenvector, -1, -2) @ direction) ** 2
    eigenvalue = np.maximum(eigenvalue, 0)
    strength = np.sum(loading * eigenvalue / (eigenvalue + .5 / tau), axis=-1)
    blind = np.all(pre[..., 1:] <= 0, axis=(-2, -1))
    margin = -np.max(pre[..., 1:], axis=(-2, -1))
    trap = (blind & np.all(w[..., 0] > 0, axis=-1)
            & np.all(a >= 0, axis=-1) & (c >= 0) & (residual[..., 0] >= 0))
    return dict(
        response=strength, blind=blind, trap=trap, rare_inactive_margin=margin,
        loss=.5 * np.sum(residual ** 2 * root ** 2, axis=-1),
        rare_loss=.25 * np.sum(residual[..., 1:] ** 2, axis=-1),
        contrast_gap=np.abs((residual[..., 1] - residual[..., 2]) / 2),
        common_residual=residual[..., 0],
    )


def train(w, a, angle, q, times, step, common_only=False):
    w, a, c = w.copy(), a.copy(), np.zeros(len(w))
    exposure = np.zeros(len(w))
    initial = measure(w, a, c, angle, q)
    ever_closed = np.zeros(len(w), dtype=bool)
    ever_reopened = np.zeros(len(w), dtype=bool)
    previously_closed = initial['blind'].copy()
    negative_common = np.zeros(len(w), dtype=bool)
    snapshots = {0.0: {**initial, 'exposure': exposure.copy(),
                       'ever_closed': ever_closed.copy(), 'ever_reopened': ever_reopened.copy(),
                       'negative_common_seen': negative_common.copy()}}
    indices = {round(t / step): t for t in times}
    for k in range(1, round(max(times) / step) + 1):
        dw, da, dc, residual, _ = fields(w, a, c, angle, q, common_only)
        exposure += step * (1 - q) * residual[..., 0]
        negative_common |= residual[..., 0] < 0
        w += step * dw
        a += step * da
        c += step * dc
        cosine, sine = np.cos(angle), np.sin(angle)
        # All rare gates off iff -cos(angle) W + sin(angle) |V| <= 0 for every unit.
        closed = np.all(-cosine * w[..., 0] + sine * np.abs(w[..., 1]) <= 0, axis=-1)
        ever_closed |= closed & ~initial['blind']
        ever_reopened |= previously_closed & ~closed
        previously_closed = closed
        if k in indices:
            record = measure(w, a, c, angle, q)
            record.update(exposure=exposure.copy(), ever_closed=ever_closed.copy(),
                          ever_reopened=ever_reopened.copy(),
                          negative_common_seen=negative_common.copy())
            snapshots[indices[k]] = record
    return snapshots


def save_records(path, rows):
    np.savez_compressed(path, **{
        f't{index}_{name}': values for index, row in enumerate(rows) for name, values in row.items()
    })


def wilson(count, total):
    z, p = 1.95996398454, count / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    rows = []
    for setting in protocol['settings']:
        key = setting['key']
        pred, actual = np.load(out / f'{key}_forecast.npz'), np.load(out / f'{key}_actual.npz')
        control = np.load(out / f'{key}_common.npz')
        initial = actual['t0_blind']
        for index, horizon in enumerate(protocol['times']):
            prefix = f't{index}_'
            y, f = actual[prefix + 'response'], pred[prefix + 'response']
            b, pb = actual[prefix + 'blind'], pred[prefix + 'blind']
            initial_response = actual['t0_response']
            persistence = np.sqrt(np.mean((y - initial_response) ** 2))
            count = len(y)
            ncontrol = len(control[prefix + 'response'])
            mse = np.mean((y - f) ** 2)
            rows.append(dict(
                **setting, time=horizon, count=count, observed_closed=int(b.sum()),
                predicted_closed=int(pb.sum()), observed_closed_ci95=wilson(int(b.sum()), count),
                newly_closed=int(np.sum(b & ~initial)), predicted_newly_closed=int(np.sum(pb & ~initial)),
                initial_closed=int(initial.sum()), reopened_from_initial=int(np.sum(~b & initial)),
                ever_closed=int(actual[prefix + 'ever_closed'].sum()),
                recovered_after_new_closure=int(np.sum(actual[prefix + 'ever_closed'] & ~b)),
                closure_disagreement=float(np.mean(pb != b)),
                response_rmse=float(np.sqrt(mse)),
                response_rmse_ratio=float(np.sqrt(mse) / persistence) if persistence > 1e-12 else None,
                mean_actual_response=float(y.mean()), mean_predicted_response=float(f.mean()),
                observed_clock_response_rmse=float(np.sqrt(np.mean(
                    (actual[prefix + 'observed_clock_response'] - y) ** 2))),
                clock_rmse=float(np.sqrt(np.mean((actual[prefix + 'exposure'] - pred[prefix + 'exposure']) ** 2))),
                common_response_rmse=float(np.sqrt(np.mean(
                    (control[prefix + 'response'] - f[:ncontrol]) ** 2))),
                common_closure_disagreement=float(np.mean(control[prefix + 'blind'] != pb[:ncontrol])),
                trap_count=int(actual[prefix + 'trap'].sum()),
                mean_loss=float(actual[prefix + 'loss'].mean()),
                mean_rare_loss=float(actual[prefix + 'rare_loss'].mean()),
                mean_contrast_gap=float(actual[prefix + 'contrast_gap'].mean()),
                mastery_count=int(np.sum(actual[prefix + 'contrast_gap'] <= .1)),
                blind_mean_contrast_gap=float(actual[prefix + 'contrast_gap'][b].mean()) if b.any() else None,
                negative_common_seen=int(actual[prefix + 'negative_common_seen'].sum()),
            ))
    (out / 'summary.json').write_text(json.dumps(dict(protocol=protocol, rows=rows), indent=2) + '\n')
    for r in rows:
        if r['time'] == max(protocol['times']):
            print({k: r[k] for k in ['key', 'count', 'observed_closed', 'predicted_closed',
                                    'newly_closed', 'trap_count', 'response_rmse_ratio',
                                    'common_closure_disagreement', 'mean_rare_loss']}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=2048)
    parser.add_argument('--control-count', type=int, default=256)
    parser.add_argument('--seed', type=int, default=301031)
    parser.add_argument('--step', type=float, default=.01)
    parser.add_argument('--times', nargs='+', type=float, default=[0, .25, .5, 1, 2, 5, 20])
    parser.add_argument('--scales', nargs='+', type=float, default=[.02, .2])
    parser.add_argument('--rare-mass', nargs='+', type=float, default=[.001, .01, .1])
    parser.add_argument('--widths', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.output)
        return
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Preserve existing evidence: choose a new output directory')
    assert 0 in args.times and all(abs(t / args.step - round(t / args.step)) < 1e-8 for t in args.times)
    start = time.monotonic()
    settings = [dict(key=f'a{angle_index}_s{scale:g}_q{q:g}_m{width}',
                     angle_over_pi=angle, scale=scale, q=q, width=width)
                for width in args.widths for scale in args.scales for q in args.rare_mass
                for angle_index, angle in enumerate([1 / 12, 1 / 3])]
    source = Path(__file__)
    protocol = dict(status='Exploratory pilot; no fitted forecast parameters; paired seed bank across settings.',
                    count=args.count, control_count=min(args.control_count, args.count), seed=args.seed,
                    step=args.step, clock_step=args.step, times=args.times, tau=20,
                    target=[4, 1, -1], settings=settings,
                    optimizer='Full-batch Euler GD, M_w=M_a=m and M_c=1; physical ReLU gates.',
                    forecast='Initial-state common-only scalar RK4 clock and exact transported features.',
                    diagnostic='Observed-clock response is retrospective, not a forecast.',
                    event='All units inactive on both rare inputs. New closure excludes initial absence.',
                    dependence='Settings and widths share initial seed banks; seeds are independent.',
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    shutil.copyfile(source, out / source.name)
    rng = np.random.default_rng(args.seed)
    bank_w = rng.normal(size=(args.count, max(args.widths), 2))
    bank_a = rng.normal(size=(args.count, max(args.widths)))
    np.savez_compressed(out / 'initial_bank.npz', w=bank_w, a=bank_a)

    # Write every forecast before starting any full-task outcome run.
    for width in args.widths:
        for scale in args.scales:
            w0, a0 = scale * bank_w[:, :width], scale * bank_a[:, :width]
            for q in args.rare_mass:
                clocks = forecast(w0, a0, q, args.times, args.step)
                for setting in settings:
                    if (setting['width'], setting['scale'], setting['q']) != (width, scale, q):
                        continue
                    angle = np.pi * setting['angle_over_pi']
                    predicted = []
                    for horizon in args.times:
                        u = clocks[horizon]
                        wt, at = transported(w0, a0, u)
                        predicted.append({**measure(wt, at, u, angle, q), 'exposure': u})
                    save_records(out / f"{setting['key']}_forecast.npz", predicted)
                print(f'Forecasts saved: m={width}, scale={scale}, q={q}; {time.monotonic()-start:.1f}s', flush=True)

    for setting in settings:
        width, scale, q = setting['width'], setting['scale'], setting['q']
        angle = np.pi * setting['angle_over_pi']
        w0, a0 = scale * bank_w[:, :width], scale * bank_a[:, :width]
        actual = train(w0, a0, angle, q, args.times, args.step)
        for row in actual.values():
            wt, at = transported(w0, a0, np.maximum(row['exposure'], 0))
            row['observed_clock_response'] = measure(wt, at, row['exposure'], angle, q)['response']
        save_records(out / f"{setting['key']}_actual.npz", [actual[t] for t in args.times])
        ncontrol = min(args.control_count, args.count)
        control = train(w0[:ncontrol], a0[:ncontrol], angle, q, args.times, args.step, True)
        save_records(out / f"{setting['key']}_common.npz", [control[t] for t in args.times])
        print(f"Outcomes saved: {setting['key']}; {time.monotonic()-start:.1f}s", flush=True)
    summarize(out)
    (out / 'complete.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-start))+'\n')


if __name__ == '__main__':
    main()
