"""Strictly feasible residual LMI parameterization, version 2.

The activation must be componentwise slope restricted in [0, 1]. All matrix
operations also support leading batch dimensions and complex Hermitian factors
(for frequency-domain linear operators). Nonlinear activations remain spatial.
No jitter, eigenvalue clipping, or detached factorization is used. A failed
Cholesky factorization is an error, not a certificate for a modified matrix.
"""

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor

PARAMETERIZATION_VERSION = 2


def eye_like(x: Tensor, n: int) -> Tensor:
    return torch.eye(n, device=x.device, dtype=x.dtype)


def hermitian(x: Tensor) -> Tensor:
    """Remove roundoff asymmetry without changing the exact expression."""
    return (x + x.mH) * 0.5


def spd_solve(a: Tensor, b: Tensor) -> Tensor:
    """Solve a Hermitian positive definite system without explicit inversion."""
    return torch.cholesky_solve(b, torch.linalg.cholesky(hermitian(a)))


def contraction(w: Tensor, alpha: float | Tensor = 1.0) -> Tensor:
    """Map a finite matrix to the open operator-norm unit ball (alpha > 0)."""
    r = torch.linalg.cholesky(
        hermitian(w.mH @ w) + alpha * eye_like(w, w.shape[-1]), upper=True
    )
    return torch.linalg.solve_triangular(r, w, upper=True, left=False)


def inverse_contraction(q: Tensor, alpha: float | Tensor = 1.0) -> Tensor:
    """Exact inverse of ``contraction`` on the open unit ball.

    If I-Q*Q = L L*, set W = sqrt(alpha) Q L^{-*}. Then the upper
    Cholesky factor of alpha I + W*W is sqrt(alpha) L^{-*}.
    """
    l = torch.linalg.cholesky(hermitian(eye_like(q, q.shape[-1]) - q.mH @ q))
    return alpha**0.5 * torch.linalg.solve_triangular(l.mH, q, upper=True, left=False)


@dataclass
class ResidualWeights:
    a: Tensor
    cs: tuple[Tensor, ...]
    b: Tensor
    multipliers: tuple[Tensor, ...]
    leading_pivots: tuple[Tensor, ...]
    output_center: Tensor
    output_metric: Tensor
    output_slack: Tensor


def residual_weights(
    raw_a: Tensor,
    raw_cs: Sequence[Tensor],
    raw_b: Tensor,
    bound: float = 1.0,
    multipliers: Sequence[Tensor] | None = None,
    alphas: Sequence[float | Tensor] | None = None,
) -> ResidualWeights:
    """Construct all strictly feasible weights for fixed positive multipliers.

    Shapes are A=(p,d0), C_i=(di,dprev), B=(p,dn). Multipliers are
    positive real vectors of length di. ``alphas`` has n+2 positive entries
    ordered A, C_1,...,C_n,B. The returned leading pivots are positive
    definite in exact arithmetic. Biases are unrestricted and external.

    Complexity is cubic in block widths and linear in depth for fixed widths;
    the full hidden-state LMI is never materialized by this construction.
    """
    if not raw_cs or bound <= 0:
        raise ValueError("At least one hidden layer and a positive bound are required")
    n = len(raw_cs)
    if multipliers is None:
        multipliers = tuple(torch.ones(c.shape[-2], device=c.device, dtype=c.real.dtype)
                            for c in raw_cs)
    if len(multipliers) != n or any(torch.any(v <= 0) for v in multipliers):
        raise ValueError("Provide one strictly positive diagonal multiplier per hidden layer")
    if alphas is None:
        alphas = (1.0,) * (n + 2)
    if len(alphas) != n + 2 or any(torch.any(torch.as_tensor(v) <= 0) for v in alphas):
        raise ValueError("All n+2 normalization parameters must be positive")
    a = bound * contraction(raw_a, alphas[0])
    d = hermitian(bound**2 * eye_like(a, a.shape[-1]) - a.mH @ a)
    e = a.mH
    s = eye_like(a, a.shape[-2])
    cs, pivots = [], []
    for i, (raw_c, lam) in enumerate(zip(raw_cs, multipliers)):
        pivots.append(d)
        de = spd_solve(d, e)
        last = i == n - 1
        # The last layer uses the completed-square slack, not the stricter
        # condition obtained by discarding cancellation through A and B.
        z = hermitian(d + e @ spd_solve(s, e.mH)) if last else d
        lz = torch.linalg.cholesky(z)
        q = contraction(raw_c, alphas[i + 1])
        c = 2.0**0.5 * lam.rsqrt().unsqueeze(-1) * (q @ lz.mH)
        cs.append(c)
        t = hermitian(s + e.mH @ de)
        if last:
            u = de.mH @ (c.mH * lam.unsqueeze(-2))
            center = -spd_solve(t, u)
            root_lam = lam.sqrt()
            k = hermitian(2 * root_lam.unsqueeze(-1)
                          * (eye_like(q, q.shape[-2]) - q @ q.mH)
                          * root_lam.unsqueeze(-2))
            lt = torch.linalg.cholesky(t)
            lk = torch.linalg.cholesky(k)
            qb = contraction(raw_b, alphas[-1])
            b = center + torch.linalg.solve_triangular(lt.mH, qb @ lk.mH, upper=True)
        else:
            e = lam.unsqueeze(-1) * (c @ de)
            d = hermitian(2 * lam.sqrt().unsqueeze(-1)
                          * (eye_like(q, q.shape[-2]) - q @ q.mH)
                          * lam.sqrt().unsqueeze(-2))
            s = t
    return ResidualWeights(a, tuple(cs), b, tuple(multipliers), tuple(pivots), center, t, k)


def inverse_residual_weights(
    a: Tensor, cs: Sequence[Tensor], b: Tensor, bound: float = 1.0,
    multipliers: Sequence[Tensor] | None = None,
) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
    """Recover free parameters from a strictly feasible real/complex LMI.

    Returns the inverse with all alpha=1. Fails if strict feasibility fails.
    This is a coverage witness, not an optimization or an SDP projection.
    """
    if multipliers is None:
        multipliers = tuple(torch.ones(c.shape[-2], device=c.device, dtype=c.real.dtype) for c in cs)
    raw_a = inverse_contraction(a / bound)
    d = hermitian(bound**2 * eye_like(a, a.shape[-1]) - a.mH @ a)
    e, s = a.mH, eye_like(a, a.shape[-2])
    raw_cs = []
    for i, (c, lam) in enumerate(zip(cs, multipliers)):
        de = spd_solve(d, e)
        last = i == len(cs) - 1
        z = hermitian(d + e @ spd_solve(s, e.mH)) if last else d
        lz = torch.linalg.cholesky(z)
        q = torch.linalg.solve_triangular(lz.mH, lam.sqrt().unsqueeze(-1) * c / 2**0.5,
                                          upper=True, left=False)
        raw_cs.append(inverse_contraction(q))
        t = hermitian(s + e.mH @ de)
        if last:
            u = de.mH @ (c.mH * lam.unsqueeze(-2))
            center = -spd_solve(t, u)
            lc = lam.unsqueeze(-1) * c
            k = hermitian(2 * torch.diag_embed(lam) - lc @ spd_solve(z, lc.mH))
            lt, lk = torch.linalg.cholesky(t), torch.linalg.cholesky(k)
            qb = torch.linalg.solve_triangular(lk.mH, lt.mH @ (b - center), upper=True, left=False)
            raw_b = inverse_contraction(qb)
        else:
            lc = lam.unsqueeze(-1) * c
            next_d = hermitian(2 * torch.diag_embed(lam) - lc @ spd_solve(d, lc.mH))
            e, d, s = lc @ de, next_d, t
    return raw_a, tuple(raw_cs), raw_b


def assemble_lmi(a: Tensor, cs: Sequence[Tensor], b: Tensor, bound: float = 1.0,
                 multipliers: Sequence[Tensor] | None = None) -> Tensor:
    """Independently assemble the negated [0,1]-sector LMI for validation.

    Unlike the constructor this uses the original quadratic forms, with no LDL
    recursions. Suitable for small examples; all leading batch axes are kept.
    """
    if multipliers is None:
        multipliers = tuple(torch.ones(c.shape[-2], device=c.device, dtype=c.real.dtype) for c in cs)
    dims = [a.shape[-1]] + [c.shape[-2] for c in cs]
    offsets = [0]
    for size in dims:
        offsets.append(offsets[-1] + size)
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2], *(c.shape[:-2] for c in cs))
    total = offsets[-1]
    out = a.new_zeros(*batch, a.shape[-2], total)
    out[..., :dims[0]] = a
    out[..., offsets[-2]:] = b
    m = -(out.mH @ out)
    m[..., :dims[0], :dims[0]] += bound**2 * eye_like(a, dims[0])
    for i, (c, lam) in enumerate(zip(cs, multipliers)):
        prev = slice(offsets[i], offsets[i+1])
        cur = slice(offsets[i+1], offsets[i+2])
        lc = lam.unsqueeze(-1) * c
        m[..., cur, cur] += 2 * torch.diag_embed(lam)
        m[..., cur, prev] -= lc
        m[..., prev, cur] -= lc.mH
    return m
