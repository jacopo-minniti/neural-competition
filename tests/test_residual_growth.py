import math

import pytest
import torch

from neural_competition.residual_growth import (
    conditional_gram,
    gaussian_relu_gram,
    relu_kernel,
    residual_potential,
)
from neural_competition.training import advance, batch_gradient, components


def test_gaussian_gram_norms_and_opposite_directions():
    vectors = torch.tensor([[1., 0.], [-2., 0.], [0., 3.]], dtype=torch.float64)
    gram = gaussian_relu_gram(vectors, vectors)
    torch.testing.assert_close(gram.diag(), torch.tensor([.5, 2., 4.5], dtype=torch.float64))
    assert abs(float(gram[0, 1])) < 1e-14
    assert abs(float(gram[0, 2])-3/(2*math.pi)) < 1e-14


def test_conditional_gram_predicts_growth_sign_switch():
    theta = math.pi/6
    teacher = torch.tensor([[math.cos(theta), math.sin(theta)],
                            [math.cos(theta), -math.sin(theta)]], dtype=torch.float64)
    center = torch.tensor([[1., 0.]], dtype=torch.float64)
    s = conditional_gram(teacher, center)
    assert float(torch.linalg.eigvalsh(s).min()) > 0
    assert abs(float(s[0, 0])-(.5-2*relu_kernel(theta)**2)) < 1e-14
    assert abs(float(s[0, 1])-(relu_kernel(2*theta)-2*relu_kernel(theta)**2)) < 1e-14
    assert float((s @ torch.tensor([1., 1.], dtype=torch.float64))[0]) > 0
    assert float((s @ torch.tensor([1., 3.], dtype=torch.float64))[0]) < 0
    assert float((s @ torch.tensor([1., -3.], dtype=torch.float64))[0]) > 0
    beta = torch.tensor([1., 3.], dtype=torch.float64)
    coarse_beta = torch.linalg.solve(gaussian_relu_gram(center, center),
                                     gaussian_relu_gram(center, teacher) @ beta)
    torch.testing.assert_close(s @ beta, residual_potential(teacher, teacher, beta,
                                                           center, coarse_beta))


def test_signed_amplitude_dynamics_without_balanced_initialization():
    rng = torch.Generator().manual_seed(76)
    p = torch.randn(1, 4, 5, generator=rng, dtype=torch.float64)
    p[:, :, -2] = 0
    x = torch.randn(1, 40, 3, generator=rng, dtype=torch.float64)
    y = torch.randn(1, 40, generator=rng, dtype=torch.float64)
    norm = torch.tensor([[2.]], dtype=torch.float64)
    gradient = batch_gradient(p, x, y, norm)
    w, a = p[0, :, :3], p[0, :, -1]
    radius = w.norm(dim=1)
    d_radius = -(w*gradient[0, :, :3]).sum(1)/radius
    actual = (-radius*gradient[0, :, -1]+a*d_radius)/2
    error = y[0]-components(p, x, norm).sum(-1)[0]
    potential = (x[0] @ (w/radius[:, None]).T).relu().T @ error/40
    predicted = (a.square()+radius.square())/4*potential
    torch.testing.assert_close(actual, predicted, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('kind', [0, 1, 2])
def test_constant_feature_is_a_scalar_output_bias(kind):
    rng = torch.Generator().manual_seed(77)
    width = 8
    p = torch.randn(1, width+1, 5, generator=rng, dtype=torch.float64)*.1
    p[:, :, -2] = 0
    p[:, -1] = torch.tensor([0., 0., 0., 1., .3*math.sqrt(width)])
    x = torch.randn(40, 3, generator=rng, dtype=torch.float64)
    y = torch.randn(40, generator=rng, dtype=torch.float64)
    norm = torch.tensor([[math.sqrt(width)]], dtype=torch.float64)
    bias = p[0, -1, -1]/norm[0, 0]
    gb = (components(p, x[None], norm).sum(-1)[0]-y).mean()
    rates = torch.ones_like(p)
    rates[:, -1] = 0
    rates[:, -1, -1] = math.sqrt(width) if kind == 2 else width
    end, _, _ = advance(p, torch.zeros_like(p), torch.zeros_like(p), x, y,
                        torch.arange(40)[None, None], torch.tensor([0]), norm, rates,
                        .003, kind, False, 0)
    expected = bias-.003*(gb/(gb.abs()+math.sqrt(width)*1e-8) if kind == 2 else gb)
    torch.testing.assert_close(end[0, -1, -1]/norm[0, 0], expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(end[0, -1, :-1], p[0, -1, :-1])
