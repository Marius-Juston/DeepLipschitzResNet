"""Original-QC, inverse, gradient, and prior-work checks for the research extension."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "revision_acyclic", Path(__file__).resolve().parents[2]/"research"/"acyclic.py")
acyclic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acyclic
SPEC.loader.exec_module(acyclic)
DTYPE = torch.float64


def explicit(x, cs, output, biases, activation=torch.tanh):
    states = x
    for c, bias in zip(cs, biases):
        states = torch.cat([states, activation(states @ c.T + bias)], dim=-1)
    return states @ output.T


@pytest.mark.parametrize("dims,p,bound", [([2,3],1,.7), ([3,2,4],2,1.3), ([2,4,3,2],4,2.)])
def test_soundness_exact_inverse_and_original_lben_identity(dims,p,bound):
    torch.manual_seed(121)
    raw = [torch.randn(d,sum(dims[:i]),dtype=DTYPE)*.3 for i,d in enumerate(dims[1:],1)]
    ro = torch.randn(p,sum(dims),dtype=DTYPE)*.2
    lam = [torch.exp(torch.randn(d,dtype=DTYPE)*.2) for d in dims[1:]]
    factors = acyclic.parameterize(raw,ro,bound,lam)
    cs,o = factors.weights()
    m = acyclic.assemble_qc(cs,o,bound,lam)
    assert torch.linalg.eigvalsh(m).min() > 0
    f,q = factors.dense_factor(), factors.output
    torch.testing.assert_close(m,f@(torch.eye(sum(dims),dtype=DTYPE)-q.T@q)@f.T)
    back,bo = acyclic.inverse(cs,o,bound,lam)
    for v,b in zip(raw,back):
        torch.testing.assert_close(v,b,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(ro,bo,atol=1e-11,rtol=1e-11)
    # Published LBEN Condition 2 on the identity-copy lift, independently built.
    w=torch.zeros_like(m); start=dims[0]
    for c in cs:
        w[start:start+c.shape[0],:start]=c
        start+=c.shape[0]
    u=torch.zeros(sum(dims),dims[0],dtype=DTYPE);u[:dims[0]]=torch.eye(dims[0],dtype=DTYPE)
    xi=torch.diag(torch.cat([torch.full((dims[0],),bound,dtype=DTYPE),*[v/bound for v in lam]]))
    r=2*xi-xi@w-w.T@xi-(o.T@o+xi@u@u.T@xi)/bound
    torch.testing.assert_close(bound*r,m)


def test_completeness_for_independently_chosen_feasible_weights():
    torch.manual_seed(512)
    dims=[3,2,4,2]; bound=.8
    lam=[torch.exp(torch.randn(d,dtype=DTYPE)) for d in dims[1:]]
    cs=[torch.randn(d,sum(dims[:i]),dtype=DTYPE)*.01 for i,d in enumerate(dims[1:],1)]
    o=torch.randn(2,sum(dims),dtype=DTYPE)*.01
    assert torch.linalg.eigvalsh(acyclic.assemble_qc(cs,o,bound,lam)).min()>0
    raw,ro=acyclic.inverse(cs,o,bound,lam)
    recovered,orec=acyclic.parameterize(raw,ro,bound,lam).weights()
    for a,b in zip(cs,recovered):
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(o,orec,atol=1e-12,rtol=1e-12)


def test_factor_evaluation_matches_physical_weights_outputs_and_gradients():
    torch.manual_seed(16)
    dims=[2,3,2]
    raw=[(torch.randn(d,sum(dims[:i]),dtype=DTYPE)*.2).requires_grad_()
         for i,d in enumerate(dims[1:],1)]
    ro=(torch.randn(2,sum(dims),dtype=DTYPE)*.2).requires_grad_()
    x=torch.randn(5,2,dtype=DTYPE,requires_grad=True)
    lam=[torch.tensor([.5,1.2,1.8],dtype=DTYPE),torch.tensor([.7,2.],dtype=DTYPE)]
    biases=[torch.randn(d,dtype=DTYPE) for d in dims[1:]]
    factors=acyclic.parameterize(raw,ro,1.4,lam)
    cs,o=factors.weights()
    direct=explicit(x,cs,o,biases)
    fast=factors.forward(x,biases)
    torch.testing.assert_close(fast,direct,atol=1e-12,rtol=1e-12)
    left=torch.autograd.grad(fast.square().sum(),[*raw,ro,x],retain_graph=True)
    right=torch.autograd.grad(direct.square().sum(),[*raw,ro,x])
    for a,b in zip(left,right):
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
    jac=torch.autograd.functional.jacobian(lambda t:factors.forward(t.unsqueeze(0),biases)[0],x[0])
    assert torch.linalg.matrix_norm(jac,ord=2)<1.4


def test_row_complement_avoids_subtraction_and_has_finite_difference_gradients():
    torch.manual_seed(15)
    v=(torch.randn(2,4,dtype=DTYPE)*.3).requires_grad_()
    assert torch.autograd.gradcheck(lambda t:acyclic.row_contraction(t)[0],(v,))
    q,j=acyclic.row_contraction(v.detach()*1000)
    torch.testing.assert_close(q@q.T+j@j.T,torch.eye(2,dtype=DTYPE),atol=1e-12,rtol=1e-12)
    assert (j.diag()>0).all()


def test_strict_wang_feedthrough_network_is_recovered_with_all_skips():
    torch.manual_seed(43)
    mu,nu=.3,1.7
    dw=torch.tensor([.7,1.2,1.5,1.1,1.6],dtype=DTYPE)
    u=torch.randn(5,2,dtype=DTYPE)*.1
    w=torch.zeros(5,5,dtype=DTYPE);w[3:,:3]=torch.randn(2,3,dtype=DTYPE)*.1
    y=u.T*dw
    rw=2*torch.diag(dw)-dw[:,None]*w-w.T*dw-2/(nu-mu)*y.T@y
    assert torch.linalg.eigvalsh(rw).min()>0
    cs=[u[:3],torch.cat([u[3:],w[3:,:3]],dim=1)]
    output=torch.cat([mu*torch.eye(2,dtype=DTYPE),y],dim=1)
    lam=[nu*dw[:3],nu*dw[3:]]
    raw,ro=acyclic.inverse(cs,output,nu,lam)
    back,o=acyclic.parameterize(raw,ro,nu,lam).weights()
    for a,b in zip(cs,back):torch.testing.assert_close(a,b)
    torch.testing.assert_close(o,output)


def test_masking_lben_to_a_chain_can_destroy_the_same_certificate():
    w=torch.tril(torch.full((3,3),-1.9,dtype=DTYPE),diagonal=-1)
    certificate=2*torch.eye(3,dtype=DTYPE)-w-w.T
    torch.testing.assert_close(torch.linalg.eigvalsh(certificate),torch.tensor([.1,.1,5.8],dtype=DTYPE))
    w[2,0]=0
    assert torch.linalg.eigvalsh(2*torch.eye(3,dtype=DTYPE)-w-w.T).min()<0
    # This is loss of the fixed-multiplier certificate, not a function-level
    # non-Lipschitz counterexample (here the output is zero).
