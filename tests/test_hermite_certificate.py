import math

import torch

from neural_competition.hermite_certificate import inertia_bound, moments, population_certificate


def test_one_relu_has_known_orthogonal_hermite_energies():
    w = torch.tensor([[1., 0.]], dtype=torch.float64)
    a = torch.ones(1, dtype=torch.float64)
    mean, linear, quadratic = moments(w, a)
    torch.testing.assert_close(mean, torch.tensor(1/math.sqrt(2*math.pi), dtype=w.dtype))
    assert float(linear.square().sum()) == .25
    assert abs(float(.5*quadratic.square().sum())-1/(4*math.pi)) < 1e-14
    result = population_certificate(w, a, w, torch.zeros_like(a))
    assert abs(result['population_mse']-.5) < 1e-14
    assert abs(result['sign_bound']-1/(4*math.pi)) < 1e-14
    assert result['higher_error'] > 0


def test_inertia_bound_matches_best_signed_spectral_truncation():
    rng = torch.Generator().manual_seed(11)
    q = torch.linalg.qr(torch.randn(5, 5, generator=rng, dtype=torch.float64)).Q
    eig = torch.tensor([-5., -2., 0., 1., 3.], dtype=torch.float64)
    target = (q*eig)@q.T
    fitted = (q*torch.tensor([-5., 0., 0., 0., 3.]))@q.T
    torch.testing.assert_close(inertia_bound(target, 1, 1), .5*(target-fitted).square().sum())


def test_random_population_certificates_and_rescaling():
    rng = torch.Generator().manual_seed(29)
    for _ in range(20):
        teacher = torch.randn(8, 6, generator=rng, dtype=torch.float64)
        beta = torch.randn(8, generator=rng, dtype=torch.float64)
        incoming = torch.randn(5, 6, generator=rng, dtype=torch.float64)
        a = -torch.rand(5, generator=rng, dtype=torch.float64)
        out = population_certificate(teacher, beta, incoming, a)
        assert out['population_mse']+1e-10 >= out['quadratic_error']
        assert out['quadratic_error']+1e-10 >= out['sign_bound']
        assert out['higher_error'] >= -1e-10
        rescaled = population_certificate(teacher, beta, 3*incoming, a/3)
        for key in out:
            assert abs(out[key]-rescaled[key]) < 1e-10
