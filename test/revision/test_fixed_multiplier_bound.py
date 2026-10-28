"""Check the new Schur bound against independently assembled QCs and bisection."""
import importlib.util
from pathlib import Path

import pytest
import torch
from models.certified_ldl import residual_weights

path = Path(__file__).resolve().parents[2] / "research/fixed_multiplier_bound.py"
spec = importlib.util.spec_from_file_location("fixed_multiplier_bound", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fixed_multiplier_bound = module.fixed_multiplier_bound


def original_qc(a, cs, b, multipliers, bound):
    dims = [a.shape[1], *[c.shape[0] for c in cs]]
    offsets = [0]
    for d in dims:
        offsets.append(offsets[-1] + d)
    out = torch.zeros(a.shape[0], offsets[-1], dtype=a.dtype)
    out[:, :dims[0]], out[:, offsets[-2]:] = a, b
    diagonal = torch.cat([torch.full((dims[0],), bound**2, dtype=a.real.dtype),
                          *[2 * v for v in multipliers]])
    matrix = torch.diag(diagonal).to(a.dtype) - out.mH @ out
    for i, (c, lam) in enumerate(zip(cs, multipliers)):
        row = slice(offsets[i+1], offsets[i+2])
        col = slice(offsets[i], offsets[i+1])
        term = -lam[:, None] * c
        matrix[row, col] += term
        matrix[col, row] += term.mH
    return matrix


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("dims,p", [([2, 3], 1), ([3, 2, 4], 2), ([2, 3, 2, 4], 3)])
def test_minimum_against_direct_lmi_bisection(dtype, dims, p):
    torch.manual_seed(823)
    prescribed = 1.3
    lam = [torch.exp(torch.randn(d, dtype=torch.float64) * .2) for d in dims[1:]]
    weights = residual_weights(torch.randn(p, dims[0], dtype=dtype) * .2,
                               [torch.randn(y, x, dtype=dtype) * .2 for x, y in zip(dims, dims[1:])],
                               torch.randn(p, dims[-1], dtype=dtype) * .2,
                               bound=prescribed, multipliers=lam)
    def matrix(bound):
        return original_qc(weights.a, weights.cs, weights.b, lam, bound)
    tightened = fixed_multiplier_bound(matrix(prescribed), dims[0], prescribed)
    lo, hi = 0., prescribed
    for _ in range(55):
        mid = (lo + hi) / 2
        if torch.linalg.eigvalsh(matrix(mid))[0] >= 0:
            hi = mid
        else:
            lo = mid
    assert abs(tightened.item() - hi) < 1e-10
    assert torch.linalg.eigvalsh(matrix(tightened.item()))[0].abs() < 1e-10
    assert torch.linalg.eigvalsh(matrix(tightened.item() * .99))[0] < 0


def test_scalar_closed_form_and_actual_relu_slope():
    # a=0, c=1/2, b=1, lambda=1: fixed-QC optimal bound is 1/2.
    matrix = torch.tensor([[1., -.5], [-.5, 1.]], dtype=torch.float64)
    torch.testing.assert_close(fixed_multiplier_bound(matrix, 1, 1.), torch.tensor(.5, dtype=torch.float64))
    # This equals the exact slope on the positive half-line in this example.
    with pytest.raises(torch.linalg.LinAlgError):
        fixed_multiplier_bound(torch.tensor([[1., 2.], [2., 1.]], dtype=torch.float64), 1, 1.)


def test_zero_output_constant_function():
    matrix = torch.diag(torch.tensor([1., 2.], dtype=torch.float64))
    assert fixed_multiplier_bound(matrix, 1, 1.).item() == 0.
