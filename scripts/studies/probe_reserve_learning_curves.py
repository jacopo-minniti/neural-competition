"""Illustrative learning curves with an initial-state reserve closure forecast.

Loss curves are measured full-task training, not reserve-model forecasts.
The loss floor additionally uses the prototype invariant-trap theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from competition_reserve import (
    common_flow,
    directional_lifetimes,
    exposure_for_gain,
    head_prediction,
    protected_reserve,
)

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


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



def setup(angle, q):
    x = np.array([[1., 0.], [-np.cos(angle), np.sin(angle)],
                  [-np.cos(angle), -np.sin(angle)]])
    audit = dict(x=x, p=np.array([1-q, q/2, q/2]),
                 directions=np.array([[0.], [1.], [-1.]])/np.sqrt(2))
    head = dict(head_center=np.array([1., 0.]), head_radius=np.zeros(2))
    return audit, head


def response_at(state, audit, tau=20.):
    w, a, c = state
    angle = np.arccos(-audit['x'][1, 0])
    q = float(2 * audit['p'][1])
    return float(measure(w[None], a[None], np.array([c]), angle, q, tau)['response'][0])


def closure_time(state, exposure, q, nodes=128):
    grid, weights = np.polynomial.legendre.leggauss(nodes)
    u = exposure * (grid + 1) / 2
    residual = np.array([4-head_prediction(common_flow(state, value)) for value in u])
    assert np.min(residual) > 0, 'Predicted closure must be reachable on the positive branch'
    return float(exposure / 2 * np.sum(weights / ((1-q)*residual)))


def select_pair(seed, angle, q):
    """Select using only initialization and reserve, before any training outcome."""
    rng = np.random.default_rng(seed)
    bank_w, bank_a = .2*rng.normal(size=(4096, 8, 2)), .2*rng.normal(size=(4096, 8))
    audit, head = setup(angle, q)
    eligible = np.flatnonzero(np.all(bank_w[:, :4, 0] > 0, axis=1)
                             & np.all(bank_a[:, :4] > 0, axis=1))
    for index in eligible:
        narrow = (bank_w[index, :4].copy(), bank_a[index, :4].copy(), np.array(0.))
        wide = (bank_w[index].copy(), bank_a[index].copy(), np.array(0.))
        exposure = float(directional_lifetimes(narrow, np.ones((1, 1)), angle).max())
        gain = head_prediction(common_flow(narrow, exposure)) - head_prediction(narrow)
        safe, mask = protected_reserve(wide, head, audit)
        initial_active = np.any(narrow[0] @ audit['x'][1:].T > 0)
        if (initial_active and exposure > 0 and gain < .8*(4-head_prediction(narrow))
                and safe[0] > .001):
            return narrow, wide, dict(
                bank_index=int(index), closing_exposure=exposure, closing_gain=gain,
                wide_protected_response=float(safe[0]), wide_protected_units=int(mask.sum()),
                closing_time=closure_time(narrow, exposure, q),
            )
    raise RuntimeError('No initially eligible pair in the declared bank')


def train_curve(state, angle, q, times, step):
    w, a = state[0][None].copy(), state[1][None].copy()
    c = np.array([float(state[2])])
    baseline = head_prediction(state)
    indices = {round(t/step): float(t) for t in times}
    records = []
    first_closure, first_trap = None, None
    first_closure_gain, first_trap_gain = None, None
    for k in range(round(max(times)/step)+1):
        pre_rare_max = -np.cos(angle)*w[..., 0] + np.sin(angle)*abs(w[..., 1])
        blind = bool(np.all(pre_rare_max <= 0))
        common_residual = float(4-c[0]-np.mean(a*np.maximum(w[..., 0], 0)))
        trapped = (blind and np.all(w[..., 0] > 0) and np.all(a >= 0)
                   and c[0] >= 0 and common_residual >= 0)
        if blind and first_closure is None:
            first_closure = k*step
            first_closure_gain = 4-common_residual-baseline
        if trapped and first_trap is None:
            first_trap = k*step
            first_trap_gain = 4-common_residual-baseline
        if k in indices:
            values = measure(w, a, c, angle, q)
            records.append(dict(time=indices[k],
                                common_gain=4-common_residual-baseline,
                                **{name: float(v[0]) for name, v in values.items()}))
        if k == round(max(times)/step):
            break
        dw, da, dc, _, _ = fields(w, a, c, angle, q)
        w += step*dw
        a += step*da
        c += step*dc
    return dict(records=records, first_closure=first_closure, first_trap=first_trap,
                first_closure_gain=first_closure_gain, first_trap_gain=first_trap_gain)


def plot(out):
    forecast = json.loads((out/'forecast.json').read_text())
    actual = json.loads((out/'outcomes.json').read_text())
    colors = {'4':'#e05252', '8':'#35b5cf'}
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), layout='constrained')
    ax = axes[0]
    for width in ['4', '8']:
        rows = actual['coarse'][width]['records']
        ax.plot([r['time'] for r in rows], [r['loss'] for r in rows],
                color=colors[width], linewidth=2.4, label=f'Actual loss, width {width}')
    closing_time = forecast['selection']['closing_time']
    ax.axvline(closing_time, color='#333333', linestyle='--', linewidth=1.2,
               label=f'Predicted closure, t={closing_time:.3f}')
    first_trap = actual['coarse']['4']['first_trap']
    if first_trap is not None:
        ax.hlines(forecast['q']/2, first_trap, forecast['end'], color=colors['4'],
                  linestyle=':', linewidth=2, label=r'Trap loss floor $q/2$')
    ax.set(xscale='symlog', yscale='log', xlabel='Training time t', ylabel='Full-task squared loss',
           title='Loss separates after a predicted loss of response', ylim=(1e-8, 12))
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.text(.03, .04, 'Solid curves: actual nonlinear training\n'
            'The marker and conditional floor are predictions;\n'
            'the full loss curves are not predicted by reserve alone.',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    ax = axes[1]
    gains = np.array(forecast['gains'])
    for width in ['4', '8']:
        pred = forecast['responses'][width]
        ax.plot(gains, pred, color=colors[width], linestyle='--', linewidth=2,
                label=f'Reserve prediction, width {width}')
        rows = actual['coarse'][width]['records']
        stop = next((i+1 for i,r in enumerate(rows) if r['common_gain'] >= gains[-1]), len(rows))
        rows = rows[:stop]
        ax.plot([r['common_gain'] for r in rows],
                [0 if r['blind'] else r['response'] for r in rows],
                color=colors[width], linewidth=1.8, alpha=.8,
                label=f'Actual response, width {width}')
    ax.axvline(forecast['selection']['closing_gain'], color='#333333', linestyle='--', linewidth=1)
    ax.axhline(forecast['selection']['wide_protected_response'], color=colors['8'],
               linestyle=':', linewidth=1.4, label='Protected floor, width 8')
    ax.set(xlabel='Common prediction improvement g', ylabel=r'Rare-contrast response $d^TC_\tau d$',
           title='What the reserve model directly forecasts', xlim=(0, gains[-1]))
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    for ax in axes:
        ax.grid(alpha=.16)
    fig.suptitle('A needed correction closes at width 4 and remains available at width 8', fontsize=14)
    fig.supxlabel('Illustrative pair selected by initial-state criteria before outcomes; '
                  'three prototypes, q=0.01, Gaussian scale 0.2.\n'
                  'Time uses a symmetric log scale. The loss floor additionally uses the invariant-trap theorem.',
                  fontsize=9)
    fig.savefig(out/'learning_and_reserve.png', dpi=180)
    fig.savefig(out/'learning_and_reserve.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=305731)
    parser.add_argument('--end', type=float, default=1000.)
    parser.add_argument('--step', type=float, default=.01)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    if args.plot_only:
        plot(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise RuntimeError('Preserve recorded examples; choose a new directory')
    angle, q = np.pi/3, .01
    narrow, wide, selection = select_pair(args.seed, angle, q)
    source = Path(__file__)
    protocol = dict(seed=args.seed, bank_size=4096, q=q, angle_over_pi=1/3, scale=.2,
                    widths=[4,8], step=args.step, end=args.end, tau=20,
                    status='Single illustrative pair; initially selected, not a prevalence estimate.',
                    selection_rule='First bank entry: positive narrow readouts/common gates, initially open, '
                              'closure before 80% common correction, wide protected response > .001.',
                    loss_curve='Measured full-task GD; reserve predicts response, not entire loss curve.',
                    floor='Conditional invariant-trap theorem q/2, additional to protected reserve.',
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copyfile(source, out/source.name)
    np.savez(out/'initial_state.npz', w=wide[0], a=wide[1])
    audit, _ = setup(angle, q)
    max_gain=.95*min(4-head_prediction(s) for s in [narrow, wide])
    gains=np.unique(np.r_[np.linspace(0,max_gain,180), selection['closing_gain'],
                         selection['closing_gain']+1e-7])
    predictions={}
    for width,state in [('4',narrow),('8',wide)]:
        predictions[width]=[response_at(common_flow(state,exposure_for_gain(state,g)),audit)
                            for g in gains]
    fine_time=closure_time(narrow,selection['closing_exposure'],q,nodes=64)
    assert abs(fine_time-selection['closing_time'])<1e-10
    closed=common_flow(narrow,selection['closing_exposure']+1e-7)
    assert response_at(closed,audit)<1e-10
    forecast=dict(**protocol,selection=selection,gains=gains.tolist(),responses=predictions)
    (out/'forecast.json').write_text(json.dumps(forecast,indent=2)+'\n')
    print('Initial-state forecast saved:',selection,flush=True)

    times=np.unique(np.r_[0,np.round(np.geomspace(args.step,args.end,190)/args.step)*args.step])
    actual={}
    for label,step in [('coarse',args.step),('fine',args.step/2)]:
        actual[label]={}
        for width,state in [('4',narrow),('8',wide)]:
            actual[label][width]=train_curve(state,angle,q,times,step)
            result=actual[label][width]
            print(label,width,'closure:',result['first_closure'],'trap:',result['first_trap'],
                  'final loss:',result['records'][-1]['loss'],flush=True)
    (out/'outcomes.json').write_text(json.dumps(actual,indent=2)+'\n')
    plot(out)


if __name__=='__main__':
    main()
