import math

import pytest
import torch

from neural_competition.specialization import (
    advance,
    merged_coefficients,
    pair_innovation,
    relu_kernel,
)
from neural_competition.trajectory import components


def test_relu_pair_innovation_identity():
    rng = torch.Generator().manual_seed(825)
    x = torch.randn(1000,2,generator=rng,dtype=torch.float64)
    for angle in [0.,.1,.5,1.]:
        u,v = math.cos(angle)*x[:,0],math.sin(angle)*x[:,1]
        actual = (u+v).relu()+(u-v).relu()-2*u.relu()
        torch.testing.assert_close(actual,pair_innovation(x[:,0],x[:,1],angle))


@pytest.mark.parametrize('degrees',[15,30,60,75])
def test_population_curvatures_against_exact_kernel_loss(degrees):
    theta = math.radians(degrees)
    co = merged_coefficients(theta)
    a = co['output']
    yy = 1+2*relu_kernel(2*theta)
    baseline = .5*yy-4*relu_kernel(theta)**2
    eps = 1e-4
    cross = relu_kernel(theta-eps)+relu_kernel(theta+eps)
    ordinary = .5*yy-cross**2
    phi = math.atan(eps)
    ff = a*a/4*(1+eps*eps)*(1+2*relu_kernel(2*phi))
    fy = a*math.sqrt(1+eps*eps)*(relu_kernel(theta-phi)+relu_kernel(theta+phi))
    split = .5*ff-fy+.5*yy
    assert (ordinary-baseline)/eps**2 == pytest.approx(co['ordinary_quadratic'],abs=2e-7)
    assert (split-baseline)/eps**2 == pytest.approx(co['split_quadratic'],abs=8e-5)


@pytest.mark.parametrize('kind',[0,1,2])
def test_identical_copy_layer_rate_mapping(kind):
    rng = torch.Generator().manual_seed(826)
    p = torch.randn(1,4,5,generator=rng,dtype=torch.float64)*.2
    x = torch.randn(100,3,generator=rng,dtype=torch.float64)
    y = torch.randn(100,generator=rng,dtype=torch.float64)
    clone = p.repeat_interleave(2,dim=1)
    clone[:,:,-1] /= math.sqrt(2)
    padded = torch.zeros(2,8,5,dtype=torch.float64)
    padded[0,:4] = p[0]
    padded[1] = clone[0]
    norm = torch.tensor([[2.],[math.sqrt(8)]],dtype=torch.float64)
    rates = torch.ones(2,1,5,dtype=torch.float64)
    if kind == 2:
        rates[1,0,-1] = 1/math.sqrt(2)
    else:
        rates[1,0,:4] = 2
    indices = torch.randint(100,(25,1,16),generator=rng)
    end,_,_ = advance(padded,torch.zeros_like(padded),torch.zeros_like(padded),
                      x,y,indices,torch.tensor([0,0]),norm,rates,.003,kind,True,0)
    pred = components(end,x[None].expand(2,-1,-1),norm).sum(-1)
    # Adam's fixed epsilon breaks exact gradient-scale homogeneity; the SGD
    # mappings are exact, while this nondegenerate Adam example is approximate.
    torch.testing.assert_close(pred[0],pred[1],rtol=1e-7,atol=1e-6 if kind==2 else 1e-14)
