import torch
import torch.nn.functional as F

from cs_dfm.flow.inference import three_term_rates, two_term_rates
from cs_dfm.flow.schedulers import PowerScheduler, PowerUniformBumpScheduler


def _rate_matrix_three(k, t, scheduler, p0, q):
    rows=[]
    w=scheduler(t);u=torch.full((k,),1/k,dtype=t.dtype);pt=w.kappa1*q+w.kappa2*u+w.kappa3*p0
    for y in range(k):
        likelihood=w.kappa1*F.one_hot(torch.tensor(y),k).to(t)+(w.kappa2/k+w.kappa3*p0[y])
        posterior=(q*likelihood/pt[y]).float()[None,:,None,None]
        row=three_term_rates(posterior,torch.tensor([[[y]]]),t,scheduler,p0.float()[None,:,None,None])[0,0,0]
        row=row.clone();row[y]=-row.sum();rows.append(row)
    return torch.stack(rows)


def test_three_term_numerical_continuity_equation():
    k=5;s=PowerUniformBumpScheduler(2.,.3);p0=torch.tensor([.1,.2,.25,.3,.15],dtype=torch.float64);target=torch.tensor([.3,.1,.2,.15,.25],dtype=torch.float64);u=torch.full((k,),1/k,dtype=torch.float64)
    for value in [.2,.7]:
        t=torch.tensor([value],dtype=torch.float64);w=s(t);p=w.kappa1*target+w.kappa2*u+w.kappa3*p0;dp=w.d_kappa1*target+w.d_kappa2*u+w.d_kappa3*p0
        matrix=_rate_matrix_three(k,t,s,p0,target).double();assert torch.allclose(p@matrix,dp,atol=3e-7,rtol=3e-7)
        eps=1e-6;wp=s(t+eps);wm=s(t-eps);finite=((wp.kappa1-wm.kappa1)*target+(wp.kappa2-wm.kappa2)*u+(wp.kappa3-wm.kappa3)*p0)/(2*eps)
        assert torch.allclose(finite,dp,atol=2e-6,rtol=2e-6)


def test_uniform_strength_zero_reduces_to_two_term_rates():
    k=4;t=torch.tensor([.2,.6]);p1=torch.softmax(torch.randn(2,k,2,3),1);zt=torch.randint(k,(2,2,3))
    source=torch.softmax(torch.randn(2,k,2,3),1)
    three=three_term_rates(p1,zt,t,PowerUniformBumpScheduler(2.,0.),source)
    two=two_term_rates(p1,zt,t,PowerScheduler(2.))
    assert torch.allclose(three,two,atol=1e-6,rtol=1e-6)
