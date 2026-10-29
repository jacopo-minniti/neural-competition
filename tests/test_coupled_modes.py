import torch

from neural_competition.coupled_modes import (
    fit_modes,
    forecast,
    matrix_decay,
    ordinary_metric,
)


def test_forecast_matches_iterated_coupled_and_preconditioned_updates():
    rng=torch.Generator().manual_seed(52)
    phi=torch.randn(60,7,generator=rng,dtype=torch.float64)
    phi[:,1]=phi[:,0]+.1*phi[:,1]
    phi[:,6]=0
    target=torch.randn(60,3,generator=rng,dtype=torch.float64)
    a=torch.randn(7,3,generator=rng,dtype=torch.float64)
    m=fit_modes(phi,target,a)
    pred=forecast(m,2)
    for full in [False,True]:
        b=m['b'].clone()
        for _ in range(pred['steps']):
            g=m['z'].T@(m['z']@b-target)/len(phi)
            b-=m['eta']*(m['inverse']@g if full else g)
        actual=float(.5*(m['z']@b-target).square().sum(1).mean())
        assert abs(actual-pred['predicted_full' if full else 'predicted_diagonal'])<1e-10


def test_column_rescaling_and_orthogonal_no_competition():
    q=torch.linalg.qr(torch.randn(40,5,dtype=torch.float64)).Q
    y=torch.randn(40,2,dtype=torch.float64)
    a=torch.randn(5,2,dtype=torch.float64)
    model=fit_modes(q,y,a)
    assert abs(forecast(model,1)['predicted_damage'])<1e-13
    scales=torch.tensor([.1,2,3,4,10],dtype=torch.float64)
    changed=fit_modes(q*scales,y,a/scales[:,None])
    for key in ['predicted_damage','predicted_diagonal','predicted_full']:
        assert abs(forecast(model,1)[key]-forecast(changed,1)[key])<1e-12


def test_ordinary_metric_preserves_trace_and_forecasts_its_actual_intervention():
    rng=torch.Generator().manual_seed(24)
    phi=torch.randn(50,6,generator=rng,dtype=torch.float64)
    phi[:,1]=phi[:,0]+.3*phi[:,1]
    phi*=torch.tensor([1.,2.,3.,.5,.2,1.],dtype=torch.float64)
    y=torch.randn(50,2,generator=rng,dtype=torch.float64)
    a=torch.randn(6,2,generator=rng,dtype=torch.float64)
    model=fit_modes(phi,y,a)
    om=ordinary_metric(model,phi,y,a)
    torch.testing.assert_close(om['a'].trace(),om['b'].trace())
    assert torch.linalg.eigvalsh(om['metric']).min()>0
    for full in [False,True]:
        trained=a.clone()
        for _ in range(12):
            g=phi.T@(phi@trained-y)/len(phi)
            trained-=om['rate']*(om['metric']@g if full else g)
        q=matrix_decay(om['b'] if full else om['a'],om['rate'],12,om['q0'])
        predicted=model['floor']+.5*q.square().sum()
        actual=.5*(phi@trained-y).square().sum(1).mean()
        torch.testing.assert_close(predicted,actual)
