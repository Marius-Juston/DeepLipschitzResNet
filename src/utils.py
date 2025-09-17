from typing import Tuple, Optional, Any

import torch
from torch import Tensor, cholesky_inverse
from torch.autograd import Function
from torch.linalg import eigh
import torch.nn.functional as F

# @torch.compile
def matrix_inv_square_root(x: Tensor) -> Tensor:
    """
    Returns the Matrix's square root inverse x**(-1/2), detaches matrix so that gradients do not cause problems close to 0
    :param x: Matrix to square root inverse
    :return: The inverse square root of x
    """
    eigvals, eigvecs = eigh(x.detach())
    return (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T


def matrix_inv_square_root_triangle(x: Tensor, B: Tensor = None, upper: bool = False, left: bool = True) -> Tensor:
    """
    Returns the Matrix's square root inverse x**(-1/2), detaches matrix so that gradients do not cause problems close to 0.
    THIS IS NOT THE "UNIQUE" SYMMETRIC MATRIX INV SQUARE ROOT!!!
    :param B:
    :param upper:
    :param left:
    :param x: Matrix to square root inverse
    :return: The inverse square root of x
    """
    L = torch.linalg.cholesky_ex(x, upper=upper).L

    if B is None:
        B = torch.eye(L.shape[0], dtype=x.dtype, device=x.device)
    L_inv = torch.linalg.solve_triangular(L, B, upper=upper, left=left)
    return L_inv


def symmetrize_(x: Tensor) -> Tensor:
    return (x + x.T) / 2.0


# @torch.compile
def fast_symmetric_positive_definitive_matrix_inverse(x: Tensor) -> Tensor:
    """
    With the assumption that x is a symmetric positive definite (PSD) matrix, take the inverse of the matrix
    :param x: The matrix to invert
    :return: The inverse of the matrix
    """
    L = torch.linalg.cholesky(x)

    return cholesky_inverse(L)


def get_largest_singular_value_power_iteration(matrix: Tensor, num_iterations: int = 25, v: Optional[Tensor] = None) -> \
        Tuple[Tensor, Tensor]:
    """
    Computes the largest singular value of a matrix using the Power Iteration method.
    This is a classic and reliable iterative method.

    Args:
        matrix (torch.Tensor): The input matrix.
        num_iterations (int): The number of iterations to run.

    Returns:
        float: The largest singular value.
    """
    if v is None:
        v = torch.randn(matrix.shape[1], 1, device=matrix.device)

    with torch.no_grad():
        ata = matrix.T @ matrix
        for _ in range(num_iterations):
            v_new = ata @ v
            v = v_new / torch.norm(v_new)

    largest_eigenvalue = (v.T @ ata @ v) / (v.T @ v)
    return torch.sqrt(largest_eigenvalue.maximum(torch.tensor(0.0, device=matrix.device))), v

# Stollen from https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html
class SpectralNorm(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[-{ndim}, {ndim - 1}] but got {dim})"
            )

        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                f"got n_power_iterations={n_power_iterations}"
            )
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if ndim > 1:
            # For ndim == 1 we do not need to approximate anything (see _SpectralNorm.forward)
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer("_u", F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer("_v", F.normalize(v, dim=0, eps=self.eps))

            # Start with u, v initialized to some reasonable values by performing a number
            # of iterations of the power method
            self.initialize(weight_mat)

    def initialize(self, weight_mat: torch.Tensor ) -> None:
        self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # Precondition
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(
                self.dim, *(d for d in range(weight.dim()) if d != self.dim)
            )

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        # See original note at torch/nn/utils/spectral_norm.py
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallelized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.

        # Precondition
        assert weight_mat.ndim > 1

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            self._u = F.normalize(
                torch.mv(weight_mat, self._v),  # type: ignore[has-type]
                dim=0,
                eps=self.eps,
                out=self._u,  # type: ignore[has-type]
            )
            self._v = F.normalize(
                torch.mv(weight_mat.H, self._u),  # type: ignore[has-type]
                dim=0,
                eps=self.eps,
                out=self._v,  # type: ignore[has-type]
            )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.vdot(u, torch.mv(weight_mat, v))
            return sigma


class CholeskyRankKFunction(Function):
    """
    Inputs:
      L : (n,n) lower-triangular Cholesky factor (A = LL^T)
      M : (n,k) update matrix; A' = A + MM^T
    Output:
      Lp : (n,n) lower-triangular Cholesky factor of A'
    """

    @staticmethod
    def forward(L, M):
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError("L must be square matrix (n,n)")
        if M.ndim != 2 or M.shape[0] != L.shape[0]:
            raise ValueError("M must have shape (n,k)")
        k = M.shape[1]
        with torch.no_grad():
            Lp = L.clone().contiguous()
            M_c = M.contiguous()
            for i in range(k):
                col = M_c[:, i]
                Lp = CholeskyRankKFunction._rank_1_update(Lp, col)
            Lp = torch.tril(Lp)  # enforce lower triangular
        return Lp

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        L, M = inputs
        Lp = output

        ctx.L = L
        ctx.M = M
        ctx.Lp = Lp

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None

        L = ctx.L
        M = ctx.M
        Lp = ctx.Lp

        needs_L, needs_M = ctx.needs_input_grad
        if not (needs_L or needs_M):
            return None, None

        G = grad_output
        P = Lp.transpose(-1, -2).matmul(G)
        diagP = torch.diagonal(P, dim1=-2, dim2=-1)
        S = torch.tril(P) - 0.5 * torch.diag_embed(diagP)
        n = Lp.shape[-1]
        I = torch.eye(n, device=Lp.device, dtype=Lp.dtype)
        Lp_inv = torch.linalg.solve_triangular(Lp, I, left=True, upper=False)
        tmp = S.matmul(Lp_inv)
        W = torch.linalg.solve_triangular(Lp.transpose(-1, -2), tmp, left=True, upper=True)
        W = 0.5 * (W + W.transpose(-1, -2))
        grad_L = None
        grad_M = None
        if needs_L:
            grad_L = torch.tril(2.0 * W.matmul(L))
        if needs_M:
            grad_M = 2.0 * W.matmul(M)
        return grad_L, grad_M

    @staticmethod
    def _rank_1_update(L: torch.Tensor, x: torch.Tensor):
        n = x.shape[0]

        L = L.contiguous()
        x = x.contiguous()

        for k in range(n):
            r = torch.hypot(L[k, k], x[k])

            c = r / L[k, k]
            s = x[k] / L[k, k]
            L[k, k] = r

            if k + 1 < n:
                lk = L[k + 1:n, k]
                xk = x[k + 1:n]

                L[k + 1:n, k] = (lk + s * xk) / c
                x[k + 1:n] = c * xk - s * L[k + 1:n, k]

        return L


def cholesky_rank_k(L: torch.Tensor, M: torch.Tensor):
    """
    Wrapper to apply the custom Function; this is the user-facing call.
    """
    return CholeskyRankKFunction.apply(L, M)
