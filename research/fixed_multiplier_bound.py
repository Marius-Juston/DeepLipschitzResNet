"""Post-training Schur bound for a fixed QC and fixed multipliers.

Research diagnostic only: the result is not outward-rounded. Reported campaign
certificates continue to use their prescribed bounds. This module is outside
the frozen training-source digest and does not alter trained models.
"""
import torch
from torch import Tensor


def fixed_multiplier_bound(matrix: Tensor, input_dim: int, prescribed_bound: float) -> Tensor:
    """Minimize the bound in an already assembled strict residual QC.

``matrix`` is M(L) with input coordinates first. For M(l), only its leading
input block changes, by (l**2-L**2) I. The hidden principal block must be SPD.
Real symmetric and complex Hermitian matrices are supported; batching is not.
This diagnostic uses a dense solve, unlike the optional block implementation
described in the proof's complexity statement. It never changes the weights.
"""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected one square certificate matrix")
    if not 0 < input_dim < matrix.shape[0] or prescribed_bound <= 0:
        raise ValueError("Need nonempty input/hidden blocks and a positive bound")
    if not torch.isfinite(matrix).all() or not torch.allclose(matrix, matrix.mH):
        raise ValueError("Certificate must be finite and Hermitian")
    m = (matrix + matrix.mH) / 2
    # Reject invalid input certificates rather than silently repairing them.
    torch.linalg.cholesky(m)
    hidden = m[input_dim:, input_dim:]
    coupling = m[input_dim:, :input_dim]
    identity = torch.eye(input_dim, dtype=m.dtype, device=m.device)
    a_gram = prescribed_bound**2 * identity - m[:input_dim, :input_dim]
    hidden_factor = torch.linalg.cholesky(hidden)
    optimal_gram = a_gram + coupling.mH @ torch.cholesky_solve(coupling, hidden_factor)
    squared = torch.linalg.eigvalsh((optimal_gram + optimal_gram.mH) / 2)[-1]
    if squared < 0 or squared >= prescribed_bound**2:
        raise RuntimeError("Numerical bound is inconsistent with a strict valid QC")
    return torch.sqrt(squared)
