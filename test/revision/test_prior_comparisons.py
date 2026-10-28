"""Independent checks of the published LBEN/Wang certificate mappings.

These tests supplement the analytic appendix; no experiment implementation is
changed. Formulas are assembled independently of the recursive parameterizer.
"""
import pytest
import torch
from models.certified_ldl import assemble_lmi, inverse_residual_weights, residual_weights

DTYPE = torch.float64


def lift(a, cs, b):
    dims = [a.shape[1], *[c.shape[0] for c in cs]]
    offsets = [0]
    for size in dims:
        offsets.append(offsets[-1] + size)
    w = torch.zeros(offsets[-1], offsets[-1], dtype=DTYPE)
    for i, c in enumerate(cs):
        w[offsets[i+1]:offsets[i+2], offsets[i]:offsets[i+1]] = c
    u = torch.zeros(offsets[-1], dims[0], dtype=DTYPE)
    u[:dims[0]] = torch.eye(dims[0], dtype=DTYPE)
    o = torch.zeros(a.shape[0], offsets[-1], dtype=DTYPE)
    o[:, :dims[0]] = a
    o[:, offsets[-2]:] = b
    return w, u, o


def lben_residual(w, u, o, xi, bound):
    return 2*xi - xi@w - w.T@xi - (o.T@o + xi@u@u.T@xi)/bound


@pytest.mark.parametrize('dims,p,bound', [([2,3],1,.7), ([3,2,4],2,1.), ([2,4,3,2],4,2.3)])
def test_exact_lben_matrix_and_optimal_copy_multiplier(dims, p, bound):
    torch.manual_seed(507)
    a = torch.randn(p, dims[0], dtype=DTYPE)
    cs = [torch.randn(y, x, dtype=DTYPE) for x, y in zip(dims, dims[1:])]
    b = torch.randn(p, dims[-1], dtype=DTYPE)
    lam = [torch.exp(torch.randn(d, dtype=DTYPE)) for d in dims[1:]]
    w, u, o = lift(a, cs, b)
    matrix = assemble_lmi(a, cs, b, bound, lam)
    xi = torch.diag(torch.cat([torch.full((dims[0],), bound, dtype=DTYPE), *[v/bound for v in lam]]))
    torch.testing.assert_close(bound*lben_residual(w,u,o,xi,bound), matrix)
    copy = torch.exp(torch.randn(dims[0], dtype=DTYPE))
    xi[:dims[0], :dims[0]] = torch.diag(copy)
    penalty = torch.zeros_like(matrix)
    penalty[:dims[0], :dims[0]] = torch.diag((copy-bound)**2)
    torch.testing.assert_close(bound*lben_residual(w,u,o,xi,bound), matrix-penalty)
    assert torch.linalg.eigvalsh(penalty).min() >= 0


def test_lben_free_parameter_reconstruction_and_function_mapping():
    torch.manual_seed(19)
    bound=1.4
    lam=[torch.tensor([.8,1.2,1.5],dtype=DTYPE),torch.tensor([.7,1.6],dtype=DTYPE)]
    weights=residual_weights(torch.randn(2,2,dtype=DTYPE)*.2,
                             [torch.randn(3,2,dtype=DTYPE)*.2, torch.randn(2,3,dtype=DTYPE)*.2],
                             torch.randn(2,2,dtype=DTYPE)*.2, bound=bound, multipliers=lam)
    w,u,o=lift(weights.a,weights.cs,weights.b)
    xi=torch.diag(torch.cat([torch.full((2,),bound,dtype=DTYPE),*[v/bound for v in lam]]))
    r=lben_residual(w,u,o,xi,bound)
    epsilon=torch.linalg.eigvalsh(r).min()/4
    assert epsilon>0
    identity=torch.eye(len(w),dtype=DTYPE)
    v=torch.linalg.cholesky(r/2-epsilon*identity,upper=True)
    skew=(w.T@xi-xi@w)/2
    reconstructed=identity-torch.linalg.solve(xi,(o.T@o+xi@u@u.T@xi)/(2*bound)+v.T@v+epsilon*identity+skew)
    torch.testing.assert_close(reconstructed,w,atol=1e-12,rtol=1e-12)
    # Mixed identity/tanh channels show the lift does not require ReLU scaling.
    x=torch.randn(7,2,dtype=DTYPE)
    bias=torch.randn(len(w),dtype=DTYPE);bias[:2]=0
    hidden=torch.tanh(x@weights.cs[0].T+bias[2:5])
    hidden=torch.tanh(hidden@weights.cs[1].T+bias[5:])
    target=x@weights.a.T+hidden@weights.b.T
    state=torch.zeros(7,len(w),dtype=DTYPE)
    for _ in range(3):
        value=state@reconstructed.T+x@u.T+bias
        state=torch.cat([value[:,:2],torch.tanh(value[:,2:])],dim=1)
    torch.testing.assert_close(state@o.T,target,atol=1e-12,rtol=1e-12)


def test_wang_tied_condition_is_exact_schur_complement():
    torch.manual_seed(43)
    mu,nu=.3,1.7
    d,m=2,5
    w=torch.zeros(m,m,dtype=DTYPE)
    w[3:,:3]=torch.randn(2,3,dtype=DTYPE)*.1
    u=torch.randn(m,d,dtype=DTYPE)*.1
    dw=torch.diag(torch.exp(torch.randn(m,dtype=DTYPE)))
    y=u.T@dw
    dl=nu*dw
    a=mu*torch.eye(d,dtype=DTYPE)
    leading=nu**2*torch.eye(d,dtype=DTYPE)-a.T@a
    off=-a.T@y-u.T@dl
    bottom=2*dl-dl@w-w.T@dl-y.T@y
    schur=bottom-off.T@torch.linalg.solve(leading,off)
    wang=2*dw-dw@w-w.T@dw-2/(nu-mu)*y.T@y
    torch.testing.assert_close(schur,nu*wang,atol=1e-12,rtol=1e-12)
    assert torch.linalg.eigvalsh(wang).min()>0


def test_noninjective_function_is_in_our_strict_parameterized_class():
    a=torch.zeros(1,1,dtype=DTYPE)
    c=torch.tensor([[.5]],dtype=DTYPE)
    b=torch.ones(1,1,dtype=DTYPE)
    matrix=assemble_lmi(a,[c],b)
    torch.testing.assert_close(torch.linalg.eigvalsh(matrix),torch.tensor([.5,1.5],dtype=DTYPE))
    recovered=residual_weights(*inverse_residual_weights(a,[c],b))
    for actual,expected in [(recovered.a,a),(recovered.cs[0],c),(recovered.b,b)]:
        torch.testing.assert_close(actual,expected)
    x=torch.tensor([[-1.],[-2.]],dtype=DTYPE)
    f=x@recovered.a.T+torch.relu(x@recovered.cs[0].T)@recovered.b.T
    assert torch.equal(f[0],f[1]) and not torch.equal(x[0],x[1])


def test_lben_recurrent_witness_and_wang_common_subset():
    scalar=lambda v:torch.tensor([[v]],dtype=DTYPE)
    w=torch.tensor([[0.,0.],[.25,.5]],dtype=DTYPE)
    u=torch.tensor([[1.],[0.]],dtype=DTYPE)
    o=torch.tensor([[0.,.25]],dtype=DTYPE)
    r=lben_residual(w,u,o,torch.eye(2,dtype=DTYPE),1.)
    torch.testing.assert_close(torch.linalg.det(r),torch.tensor(7/8,dtype=DTYPE))
    assert torch.linalg.eigvalsh(r).min()>0
    assert torch.trace(w)==.5
    # A nontrivial one-hidden-state network satisfying both strict conditions.
    m=assemble_lmi(scalar(.25),[scalar(.5)],scalar(.5))
    assert torch.linalg.eigvalsh(m).min()>0
    wang=2-2/(1-.25)*.5**2
    assert wang==pytest.approx(4/3)


def test_wang_bound_saturating_map_is_on_our_boundary():
    mu,nu=.25,1.0
    gamma=nu-mu
    scalar=lambda v:torch.tensor([[v]],dtype=DTYPE)
    c=scalar(gamma**.5)
    assert 2-2/gamma*float(c.T@c)==pytest.approx(0,abs=1e-14)
    matrix=assemble_lmi(scalar(mu),[c],c,bound=nu,
                        multipliers=[torch.tensor([nu],dtype=DTYPE)])
    eig=torch.linalg.eigvalsh(matrix)
    assert abs(float(eig[0]))<1e-14 and eig[1]>0
    # Positive inputs saturate the upper bound; no strict-LMI realization
    # at this bound can represent this function (as proved in the appendix).
    assert mu+float(c@c)==pytest.approx(nu)
