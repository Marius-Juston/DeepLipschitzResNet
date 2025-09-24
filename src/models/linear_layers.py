from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torch.nn.utils import parametrize

from utils import PositiveExp


class AbstractLipschitzLinearLayer(nn.Linear):
    identity: torch.Tensor
    identity_out: torch.Tensor
    constant_sq: torch.Tensor
    constant: torch.Tensor
    R: torch.Tensor
    omega_r: torch.Tensor

    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
                 constant_sq: float = 1.0,
                 enable_alpha: bool = True,
                 device=None,
                 dtype=None):
        super(AbstractLipschitzLinearLayer, self).__init__(in_features, out_features, bias=bias, device=device,
                                                           dtype=dtype)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.register_buffer(
            "identity", torch.eye(in_features, **factory_kwargs)
        )

        self.register_buffer(
            "identity_out", torch.eye(out_features, **factory_kwargs)
        )

        self.register_buffer(
            "constant_sq", torch.tensor(constant_sq, **factory_kwargs)
        )

        self.register_buffer(
            "constant", self.constant_sq.sqrt()
        )

        initial_alpha = torch.tensor(1.0, **factory_kwargs)

        if enable_alpha:
            self.alpha = Parameter(initial_alpha)
            parametrize.register_parametrization(self, "alpha", PositiveExp())
        else:
            self.register_buffer('alpha', initial_alpha)

    def scale_w(self):
        omega = self.sum_weight(transpose=True)
        R = torch.linalg.cholesky(omega, upper=True)

        W = torch.linalg.solve_triangular(R, self.weight, upper=True, left=False)

        return omega, W, R

    def sum_weight(self, transpose: bool = False):
        if transpose:
            return self.weight.T @ self.weight + self.identity * self.alpha
        else:
            return self.weight @ self.weight.T + self.identity_out * self.alpha


class FirstLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        omega, W, R = self.scale_w()

        W = W * self.constant

        d_inv = omega / (self.constant_sq * self.alpha)
        self.omega_r = R

        return torch.nn.functional.linear(input, W, self.bias), W, d_inv


class InnerWeightLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    def forward(self, input: Tensor, prev_layer: Union[FirstLipschitzLinearLayer, "InnerWeightLipschitzLinearLayer"]) -> \
            Tuple[Tensor, Tensor, Tensor]:
        _, W, _ = self.scale_w()

        # with torch.no_grad():
        #     assert  torch.linalg.eigvalsh(W @ W.T).max() <= 1.0
        #     assert  torch.linalg.eigvalsh(W.T @ W).max() <= 1.0

        # assert torch.allclose(prev_layer.omega_r.T.tril(), prev_layer.omega_r.T), "This is not lower triangular"
        W = torch.linalg.solve_triangular(prev_layer.omega_r.T, W, upper=False, left=False)

        # with torch.no_grad():
        #     W_temp = W @ prev_layer.omega_r.T
        #
        #     assert torch.allclose(W_temp, self.scale_w()[1])

        # with torch.no_grad():
        #     omega = prev_layer.omega_r.T @ prev_layer.omega_r
        #
        #     value = W @ omega @ W.T
        #
        #     assert torch.linalg.svdvals(value).abs().max() <= 1.0
        #     assert torch.linalg.eigvalsh(value).abs().max() <= 1.0

        scale = prev_layer.alpha.sqrt() * prev_layer.constant * self.constant
        div_scale = self.constant_sq * self.alpha

        # with torch.no_grad():
        #     square_scale = scale ** 2
        #
        #     div_scale_prev = prev_layer.constant_sq * prev_layer.alpha
        #
        #     value = square_scale / div_scale_prev
        #
        #     # tmep = torch.isclose(value, torch.full_like(value, 1.0))
        #     tmep2 = torch.isclose(value, torch.full_like(value, 2.0))
        #
        #     assert  tmep2, value

        W = scale * W

        omega = self.sum_weight()

        d_inv = omega / div_scale

        self.omega_r = torch.linalg.cholesky(omega, upper=True)

        # with torch.no_grad():
        #     assert torch.allclose(omega, self.omega_r.T @ self.omega_r)
        #     R_inv = torch.linalg.solve_triangular(self.omega_r, self.identity_out, upper=True)
        #
        #     inverse_val = R_inv.T @ omega @ R_inv
        #
        #     assert torch.allclose(self.identity_out, inverse_val)
        #
        #     scale = self.alpha.sqrt() * self.constant
        #
        #     temp =  scale * R_inv
        #     result = temp.T @ d_inv @ temp
        #
        #     assert torch.allclose(self.identity_out, result)

        # with torch.no_grad():
        #     M = torch.randn_like(self.identity)
        #
        #     c = self.constant
        #
        #     q, _ = torch.linalg.qr(M)
        #
        #     q *= c
        #
        #     q_2 = torch.linalg.solve_triangular(prev_layer.omega_r.T, q, upper=False, left=False)
        #
        #     omega = prev_layer.omega_r.T @ prev_layer.omega_r
        #
        #     inverse_val = q_2 @ omega @ q_2.T
        #
        #     assert torch.allclose(self.identity * self.constant_sq, inverse_val)

        return torch.nn.functional.linear(input, W, self.bias), W, d_inv


class LastLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    def forward(self, input: Tensor, sigma_r: Tensor, last_layer: InnerWeightLipschitzLinearLayer) -> Tuple[
        Tensor, Tensor]:
        _, W, _ = self.scale_w()

        prev_alpha = last_layer.alpha
        prev_r = last_layer.omega_r

        # assert torch.allclose(prev_r.triu(), prev_r), "This is not upper triangular"
        W = torch.linalg.solve_triangular(prev_r.T, W, upper=False, left=False)
        W = self.constant * prev_alpha.sqrt() * torch.linalg.solve_triangular(sigma_r, W, upper=True, left=True)

        return torch.nn.functional.linear(input, W, self.bias), W


class LinearLipschitz(AbstractLipschitzLinearLayer):
    def __init__(self, in_features: int, out_features: int, activation=None, bias: bool = True,
                 constant_sq: float = 1.0,
                 enable_alpha=True,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias, constant_sq, enable_alpha, device, dtype)
        self.activation = activation

    def forward(self, input: Tensor, prev_d_sqrt: Tensor) -> Tuple[Tensor, Tensor]:
        _, W, _ = self.scale_w()

        W = self.constant * torch.linalg.solve_triangular(prev_d_sqrt.T, W, upper=False, left=False)
        d_sqrt = self.constant * torch.linalg.cholesky(self.sum_weight(), upper=True)

        out = torch.nn.functional.linear(input, W, self.bias)

        if self.activation:
            out = self.activation(out)

        return out, d_sqrt


# Taken from https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html
# Taken but removed weight output, only keeping eigenvalue https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/parametrizations.py#L401
class SpectralNorm(torch.nn.Module):
    def __init__(
            self,
            weight: torch.Tensor,
            n_power_iterations: int = 1,
            dim: int = 0,
            eps: float = 1e-12,
            default_n_iterations: int = 15
    ) -> None:
        super().__init__()
        self.default_n_iterations = default_n_iterations
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

        if default_n_iterations < 1:
            raise ValueError(
                "Expected default_n_iterations to be greater than 1, but "
                f"got n_power_iterations={default_n_iterations}"
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

        self.initialized = False

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

        if not self.initialized:
            n_power_iterations = self.default_n_iterations

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

        self.initialized = True

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
