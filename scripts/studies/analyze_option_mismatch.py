"""Held-seed-out comparison of forecast differences during ordinary SGD."""

import json

import matplotlib
import numpy as np
from track_option_mismatch import OUT, SEEDS

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(name):
    return [json.loads(line) for line in (OUT/name).read_text().splitlines()]


def features(row):
    a, b = row['narrow'], row['wide']
    fields = ['full_loss', 'gradient_norm_squared', 'scalar_curvature_prediction',
              'value', 'alignment', 'residual_strength_percentile', 'breadth', 'mass']
    z = {'time':np.log1p(row['step'])}
    for field in fields:
        z[field+'_gap'] = b[field]-a[field]
        z[field+'_mean'] = (a[field]+b[field])/2
    z['spectral_prediction_gap'] = b['spectral_noise_prediction']-a['spectral_noise_prediction']
    z.update(row['cross'])
    z['common_response_gap'] = row['cross']['common_value_wide']-row['cross']['common_value_narrow']
    z['excess_response_gap'] = row['cross']['excess_value_wide']-row['cross']['excess_value_narrow']
    return z


def score(actual, prediction):
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    correlation = np.corrcoef(actual, prediction)[0,1] if prediction.std()>1e-15 else None
    return dict(mae=float(np.mean(np.abs(actual-prediction))),
        rmse=float(np.sqrt(np.mean((actual-prediction)**2))), correlation=None if correlation is None else float(correlation),
        direction_accuracy=float(np.mean((actual>0)==(prediction>0))),
        n=len(actual), mean_actual=float(actual.mean()), mean_prediction=float(prediction.mean()))


def fit_predict(x, y, calibration, test):
    mean, scale = x[calibration].mean(0), x[calibration].std(0)
    scale = np.maximum(scale, 1e-12)
    z = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    penalty = np.eye(z.shape[1])
    penalty[0,0] = 0
    coef = np.linalg.solve(z[calibration].T@z[calibration]+penalty,
                           z[calibration].T@y[calibration])
    return z[test]@coef


def main():
    base = ['time', 'full_loss_gap', 'full_loss_mean', 'gradient_norm_squared_gap', 'gradient_norm_squared_mean']
    curvature = base+['scalar_curvature_prediction_gap', 'scalar_curvature_prediction_mean']
    options = ['value_gap', 'alignment_gap', 'residual_strength_percentile_gap', 'breadth_gap', 'mass_gap']
    models = {'loss_time':base[:3], 'loss_gradient':base, 'loss_gradient_curvature':curvature,
              'loss_gradient_plus_options':base+options,
              'loss_gradient_curvature_plus_options':curvature+options,
              'loss_gradient_plus_spectral':base+['spectral_prediction_gap'],
              'loss_gradient_plus_matched_demand':base+['capability_component','common_response_gap','excess_response_gap']}
    protocol = dict(target='wide gain minus narrow gain over next 20 ordinary SGD updates',
        models=models, ridge_penalty=1., calibration_seeds=SEEDS[:4], test_seeds=SEEDS[4:],
        periods=dict(all='start>=0', late='start>=100'),
        note='Initial feature sets fixed before inspecting target scores. Phase-conditioned fits and matched-demand additions are exploratory follow-ups. Intervals within a seed are correlated.')
    (OUT/'analysis_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    forecasts = read('forecasts.jsonl')
    by_id = {(r['seed'],r['step']):r for r in forecasts}
    outcomes = read('outcomes.jsonl')
    y = np.array([r['gain_difference'] for r in outcomes])
    rows = [by_id[(r['seed'],r['start'])] for r in outcomes]
    f = [features(r) for r in rows]
    calibration = np.array([r['seed'] in SEEDS[:4] for r in rows])
    test = ~calibration
    prediction = {'zero':np.zeros(test.sum())}
    for field in ['first_order_prediction','scalar_curvature_prediction','spectral_prediction','spectral_noise_prediction']:
        prediction[field] = np.array([r['wide'][field]-r['narrow'][field] for r in rows])[test]
    for name, fields in models.items():
        x = np.array([[r[field] for field in fields] for r in f])
        prediction[name] = fit_predict(x, y, calibration, test)
    times = np.array([r['step'] for r in rows])[test]
    seeds = np.array([r['seed'] for r in rows])[test]
    truth = y[test]
    scores = {}
    for period, selection in [('all',np.ones(len(truth),bool)), ('late',times>=100)]:
        scores[period] = {name:score(truth[selection], p[selection]) for name,p in prediction.items()}
    per_seed = {str(seed):{name:score(truth[seeds==seed], p[seeds==seed]) for name,p in prediction.items()}
                for seed in SEEDS[4:]}
    # Same predeclared late phase for fitting and evaluation: avoids letting the
    # initial transient dominate a predictor of small late residual differences.
    all_times = np.array([r['step'] for r in rows])
    late_calibration = calibration & (all_times>=100)
    late_test = test & (all_times>=100)
    late_prediction = {}
    for name, fields in models.items():
        x = np.array([[r[field] for field in fields] for r in f])
        late_prediction[name] = fit_predict(x, y, late_calibration, late_test)
    phase_calibrated = {name:score(y[late_test],p) for name,p in late_prediction.items()}
    phase_seeds = np.array([r['seed'] for r in rows])[late_test]
    phase_per_seed = {str(seed):{name:score(y[late_test][phase_seeds==seed],p[phase_seeds==seed])
                     for name,p in late_prediction.items()} for seed in SEEDS[4:]}
    # Width-specific mismatch and common/excess residual diagnostics remain descriptive.
    trajectories = []
    for step in sorted({r['step'] for r in forecasts}):
        group = [r for r in forecasts if r['step']==step]
        row = dict(step=step)
        for name in ('narrow','wide'):
            for field in ['full_loss','validation_loss','mass','breadth','alignment','value',
                          'residual_strength_percentile','lost_from_initial']:
                row[name+'_'+field] = float(np.mean([r[name][field] for r in group]))
        row['excess_response_ratio'] = float(np.mean([r['cross']['excess_value_wide']/max(r['cross']['excess_value_narrow'],1e-20) for r in group]))
        row['capability_component'] = float(np.mean([r['cross']['capability_component'] for r in group]))
        row['demand_component'] = float(np.mean([r['cross']['demand_component'] for r in group]))
        trajectories.append(row)
    demand_cdf = []
    fig_cdf, cdf_axes = plt.subplots(1,4,figsize=(13,3.4),layout='constrained')
    for step, ax in zip((0,100,400,800),cdf_axes):
        for width,color in [(32,'#2969a2'),(128,'#be6925')]:
            curves = []
            for seed in SEEDS:
                with np.load(OUT/f'spectra_{seed}.npz') as arrays:
                    load = arrays[f'{width}_{step}_residual_loading']
                curves.append(np.cumsum(load)/load.sum())
            curve = np.mean(curves,axis=0)
            grid = np.arange(1,len(curve)+1)/len(curve)
            ax.plot(grid,curve,color=color,label=f'Width {width}')
            demand_cdf.append(dict(step=step,width=width,
                residual_fraction_in_weakest_quarter=float(curve[len(curve)//4-1]),
                residual_fraction_in_weakest_half=float(curve[len(curve)//2-1])))
        ax.plot([0,1],[0,1],'--',color='gray',label='Uniform residual')
        ax.set(title=f'Training step {step}',xlabel='Fraction of modes, weakest first',
               ylabel='Cumulative residual energy',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=7)
    fig_cdf.savefig(OUT/'residual_loading.png',dpi=180)
    result = dict(protocol=protocol,scores=scores,per_test_seed=per_seed,
        late_phase_calibrated=phase_calibrated,late_phase_per_test_seed=phase_per_seed,trajectories=trajectories,
        residual_loading=demand_cdf,
        forecast_audit=dict(rows=len(forecasts), intervals=len(outcomes),
            unstable=sum(not r[name]['stable'] for r in forecasts for name in ['narrow','wide']),
            max_projection_fraction=max(r[name]['gradient_projection_fraction'] for r in forecasts for name in ['narrow','wide'])),
        caveat='Raw forecasts retain an unused panel_spectral_prediction auxiliary field from the producer prototype; it is excluded from every analysis and removed from the current producer.')
    (OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    (OUT/'heldout_predictions.json').write_text(json.dumps(dict(actual=truth.tolist(),times=times.tolist(),
        seeds=seeds.tolist(),predictions={k:v.tolist() for k,v in prediction.items()}),indent=2)+'\n')
    fig, axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
    time = [r['step'] for r in trajectories]
    for name,color,label in [('narrow','#2969a2','Width 32'),('wide','#be6925','Width 128')]:
        for ax,field,title in zip(axes.flat[:4], ['full_loss','breadth','residual_strength_percentile','alignment'],
            ['Ordinary training loss','Effective response breadth','Residual-weighted strength percentile','Alignment relative to mean strength']):
            ax.plot(time,[r[name+'_'+field] for r in trajectories],color=color,label=label)
            ax.set(title=title,xlabel='SGD step')
            ax.legend(fontsize=8)
    axes[0,0].set_yscale('log')
    select = times>=100
    for name,label,color in [('loss_gradient','Loss + gradient','#2969a2'),
                             ('loss_gradient_plus_options','Plus option diagnostics','#be6925')]:
        axes[1,1].scatter(truth[select],late_prediction[name],s=13,alpha=.55,label=label,color=color)
    lo,hi = min(truth[select]),max(truth[select])
    axes[1,1].plot([lo,hi],[lo,hi],'--',color='gray')
    axes[1,1].set(title='Held-out seeds: next-interval gain difference',xlabel='Observed: wide minus narrow',ylabel='Predicted')
    axes[1,1].legend(fontsize=8)
    for seed in SEEDS[4:]:
        selection = (seeds==seed)&(times>=100)
        axes[1,2].plot(times[selection],truth[selection],alpha=.5,label=str(seed))
    axes[1,2].axhline(0,color='gray',linestyle='--')
    axes[1,2].set(title='Later-phase width difference in learning progress',xlabel='SGD step',ylabel='Wide gain minus narrow gain')
    axes[1,2].set_ylim(-.015,.015)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'summary.png',dpi=180)
    print(json.dumps(dict(scores=scores,per_test_seed=per_seed),indent=2))


if __name__ == '__main__':
    main()
