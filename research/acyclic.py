"""Research extension: complete strict QC parameterization on a full acyclic graph.

Real matrices only. This is separate from the frozen revision campaign.
Every hidden block can read all earlier states; arbitrary edge masks are NOT
supported. Exact arithmetic guarantees do not replace numerical LMI audits.
"""
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def row_contraction(v: Tensor) -> tuple[Tensor, Tensor]:
    """Return Q and its lower complementary factor J: QQ^T + JJ^T = I."""
    identity = torch.eye(v.shape[0], dtype=v.dtype, device=v.device)
    lower = torch.linalg.cholesky(identity + v @ v.T)
    return (torch.linalg.solve_triangular(lower, v, upper=False),
            torch.linalg.solve_triangular(lower, identity, upper=False))


def inverse_row_contraction(q: Tensor) -> Tensor:
    identity = torch.eye(q.shape[0], dtype=q.dtype, device=q.device)
    lower = torch.linalg.cholesky(identity - q @ q.T)
    return torch.linalg.solve_triangular(lower, q, upper=False)


@dataclass
class AcyclicFactors:
    rows: tuple[Tensor, ...]  # F_i,0:i, without trailing zeros; row zero is L I.
    output: Tensor          # Q, with physical output O = Q F^T.
    multipliers: tuple[Tensor, ...]
    bound: float

    @property
    def dims(self):
        return tuple(row.shape[0] for row in self.rows)

    def dense_factor(self) -> Tensor:
        total = sum(self.dims)
        return torch.cat([torch.nn.functional.pad(row, (0, total-row.shape[1]))
                          for row in self.rows])

    def weights(self) -> tuple[tuple[Tensor, ...], Tensor]:
        """Materialize physical weights for independent checks / frozen inference."""
        factor = self.dense_factor()
        gram = factor @ factor.T
        cs, start = [], self.dims[0]
        for dim, lam in zip(self.dims[1:], self.multipliers):
            cs.append(-gram[start:start+dim, :start] / lam[:, None])
            start += dim
        return tuple(cs), self.output @ factor.T

    def forward(self, x: Tensor, biases: Sequence[Tensor] | None = None,
                output_bias: Tensor | None = None,
                activation: Callable[[Tensor], Tensor] = torch.tanh) -> Tensor:
        """Evaluate without forming F F^T, physical weights, or an equilibrium solve.

        At step i the accumulator is sum_{j<i} F_j^T z_j, restricted
        to the first sum_{j<i} d_j coordinates. The caller must supply
        componentwise slope-[0,1] activations for the certificate to apply.
        """
        if biases is None:
            biases = [row.new_zeros(row.shape[0]) for row in self.rows[1:]]
        if len(biases) != len(self.multipliers):
            raise ValueError("One bias is required per hidden block")
        acc = x @ self.rows[0]
        for row, lam, bias in zip(self.rows[1:], self.multipliers, biases):
            prefix = acc.shape[-1]
            prior, diagonal = row[:, :prefix], row[:, prefix:]
            state = activation(-(acc @ prior.T)/lam + bias)
            acc = torch.cat([acc + state @ prior, state @ diagonal], dim=-1)
        value = acc @ self.output.T
        return value if output_bias is None else value + output_bias


def parameterize(raw_rows: Sequence[Tensor], raw_output: Tensor,
                 bound: float = 1., multipliers: Sequence[Tensor] | None = None
                 ) -> AcyclicFactors:
    if not raw_rows or bound <= 0:
        raise ValueError("A positive bound and at least one hidden block are required")
    if any(v.ndim != 2 or v.is_complex() for v in [*raw_rows, raw_output]):
        raise ValueError("This research implementation supports real matrices only")
    if multipliers is None:
        multipliers = [v.new_ones(v.shape[0]) for v in raw_rows]
    if len(multipliers) != len(raw_rows):
        raise ValueError("One positive diagonal multiplier is required per block")
    dims = [raw_rows[0].shape[1]]
    rows = [bound * torch.eye(dims[0], device=raw_output.device, dtype=raw_output.dtype)]
    for v, lam in zip(raw_rows, multipliers):
        if v.shape[1] != sum(dims) or lam.shape != (v.shape[0],) or not bool((lam > 0).all()):
            raise ValueError("Invalid prefix dimensions or nonpositive multiplier")
        q, complement = row_contraction(v)
        rows.append((2*lam).sqrt()[:, None] * torch.cat([q, complement], dim=1))
        dims.append(v.shape[0])
    if raw_output.shape[1] != sum(dims):
        raise ValueError("The output must read every state, including the input")
    qout, _ = row_contraction(raw_output)
    return AcyclicFactors(tuple(rows), qout, tuple(multipliers), bound)


def assemble_qc(cs: Sequence[Tensor], output: Tensor, bound: float,
                multipliers: Sequence[Tensor]) -> Tensor:
    """Assemble the original activation QC from physical weights, independently."""
    dims = [cs[0].shape[1], *[c.shape[0] for c in cs]]
    diagonal = torch.cat([output.new_full((dims[0],), bound**2),
                          *[2*lam for lam in multipliers]])
    lower = torch.zeros_like(output.T @ output)
    start = dims[0]
    for c, lam in zip(cs, multipliers):
        lower[start:start+c.shape[0], :start] = -lam[:, None]*c
        start += c.shape[0]
    return torch.diag(diagonal) + lower + lower.T - output.T @ output


def inverse(cs: Sequence[Tensor], output: Tensor, bound: float,
            multipliers: Sequence[Tensor]) -> tuple[tuple[Tensor, ...], Tensor]:
    """Recover free parameters from any strictly QC-feasible physical weights."""
    matrix = assemble_qc(cs, output, bound, multipliers)
    # Check the domain, not merely positivity of G=M+O^T O.
    torch.linalg.cholesky(matrix)
    factor = torch.linalg.cholesky(matrix + output.T @ output)
    start, raw = cs[0].shape[1], []
    for c, lam in zip(cs, multipliers):
        size = c.shape[0]
        row = factor[start:start+size, :start+size] / (2*lam).sqrt()[:, None]
        raw.append(torch.linalg.solve_triangular(row[:, start:], row[:, :start], upper=False))
        start += size
    qout = torch.linalg.solve_triangular(factor.T, output, upper=True, left=False)
    return tuple(raw), inverse_row_contraction(qout)
