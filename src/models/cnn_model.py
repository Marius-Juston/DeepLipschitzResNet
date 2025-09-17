from typing import Sequence, Union, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.nn import ModuleList, ReLU
from torch.nn.common_types import _size_2_t

from models.cnn_layers import InnerWeightLipschitzConvLayer, LastLipschitzConvLayer, FirstLipschitzConvLayer


class DeepLipschitzConvResNet(nn.Module):
    A: FirstLipschitzConvLayer
    B: LastLipschitzConvLayer
    layers: Sequence[InnerWeightLipschitzConvLayer]

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            layers_widths: Sequence[int],
            kernel_size: _size_2_t = 3,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            activation: Union[nn.Module, List[nn.Module]] = None,
            lipschitz_constant: float = 1.0,
            device: Union[str, torch.device] = 'cuda',
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        assert len(layers_widths) >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_widths = list(layers_widths)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.lipschitz_constant = lipschitz_constant
        self.lipschitz_constant_sq = lipschitz_constant * lipschitz_constant

        self.activation = ReLU() if activation is None else activation

        # Build layers: first -> inner(s) -> last
        self.A = FirstLipschitzConvLayer(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
            constant_sq=self.lipschitz_constant_sq,
            **self.factory_kwargs
        )

        # inner layers (note: layer_widths[0] is the first hidden out)
        layers = []
        prev = self.in_channels
        for i, width in enumerate(self.layer_widths):
            if i == 1:
                constant = 2.0 * self.lipschitz_constant_sq
            else:
                constant = 4.0
            layers.append(
                InnerWeightLipschitzConvLayer(
                    in_channels=prev,
                    out_channels=width,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=True,
                    constant_sq=constant,
                    **self.factory_kwargs
                )
            )
            prev = width
        self.layers = ModuleList(layers)

        self.B = LastLipschitzConvLayer(
            in_channels=self.layer_widths[-1],
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=False,
            **self.factory_kwargs
        )

        # placeholder for spectral_norm helper (user can supply a batched, complex-aware spectral norm)
        self.spectral_norm = BatchedConvSpectralNorm()

    def _cholesky_rank_k_freq(self, prev_sigma_lower: Optional[Tensor], T: Tensor) -> Tensor:
        if prev_sigma_lower is None:
            return T
        else:
            prev = prev_sigma_lower
            S_prev = torch.matmul(prev, prev.mH)
            S_t = torch.matmul(T, T.mH)
            S = S_prev + S_t
            L = torch.linalg.cholesky(S, upper=False)
            return L

    def compute_b(self, x: Tensor, sigma_lower: Tensor, a_weight_freq: Tensor, prev_r: Tensor) -> Tuple[
        Tensor, Tensor]:
        # inner_inverse = prev_r^{-1}
        inner_inverse = torch.cholesky_inverse(prev_r, upper=True).real
        # spectral norm scaling
        max_singular = self.spectral_norm(inner_inverse)
        max_singular = max_singular.max()

        left = torch.matmul(a_weight_freq, sigma_lower)

        scaling = self.B.identity_out + torch.matmul(left, left.mH)

        R_sigma = torch.linalg.cholesky(scaling, upper=True)

        second, b_weight_spatial = self.B.forward(x, R_sigma)
        second_s = second * max_singular
        b_weight_s = b_weight_spatial * max_singular

        return second_s, b_weight_s

    def forward(self, x: Tensor) -> Tensor:
        # First layer -> get normalized freq matrices once
        first, W_spatial_A, Wfreq_A, d_sqrt, d_inv = self.A.forward(x)
        # initial gamma (identity per-frequency)
        gamma = self.A.identity.to(dtype=d_inv.dtype)

        current_input = x
        sigma_lower = None

        # iterate inner layers; each inner returns its Wfreq_norm so we can use it for gamma update
        for layer in self.layers:
            T = torch.matmul(gamma, d_inv)  # (F, n, n)
            current_input, c_spatial, c_freq_norm, d_sqrt, d_inv = layer.forward(current_input, d_sqrt)
            # update sigma_lower via rank-k cholesky update in freq domain
            sigma_lower = self._cholesky_rank_k_freq(sigma_lower, T)
            # update gamma using the frequency-normalized inner weight returned by the layer (no recompute)
            # c_freq_norm has shape (F, out, in) where in matches gamma's second dim
            # we want gamma = T @ c_freq_norm^H
            gamma = torch.matmul(T, c_freq_norm.mH)
            current_input = self.activation(current_input)

        # obtain prev_r from last inner layer (omega_r stored on that layer in forward)
        prev_r = self.layers[-1].omega_r
        second, b_spatial = self.compute_b(current_input, sigma_lower, Wfreq_A, prev_r)
        return first + second


class BatchedConvSpectralNorm(nn.Module):
    """
    Batched Spectral Norm for (F, n, n) matrices.

    - Call initialize(weight_mat) with weight_mat shape (F, n, n) to create internal buffers.
      If forward receives a batched matrix whose batch-size differs from buffers, initialize()
      will be called automatically (costly).
    - forward(weight) accepts:
        * weight: (F, n, n) complex or real -> returns scalar max singular value (real)
        * weight: (n, n) -> returns spectral norm (scalar)
    - The module stores batched estimates of left/right singular vectors in buffers:
        _u: (F, n), _v: (F, n)
    """

    def __init__(
            self,
            n_power_iterations: int = 1,
            dim: int = 1,
            eps: float = 1e-12,
            default_n_iterations: int = 15
    ) -> None:
        super().__init__()
        self.dim = dim
        self.default_n_iterations = default_n_iterations

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

        self.eps = eps
        self.n_power_iterations = n_power_iterations

        self.initialized = False

    def initialize(self, weight_mat):
        ndim = weight_mat.ndim
        if self.dim >= ndim or self.dim < -ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[-{ndim}, {ndim - 1}] but got {self.dim})"
            )

        self.dim = self.dim if self.dim >= 0 else self.dim + ndim

        if ndim != 3:
            raise ValueError(f"Expected weight to have 3 dimensions, but got {ndim}")

        b, h, w = weight_mat.size()

        u = weight_mat.new_empty(b, h).normal_(0, 1)
        v = weight_mat.new_empty(b, w).normal_(0, 1)
        self.register_buffer("_u", F.normalize(u, dim=1, eps=self.eps))
        self.register_buffer("_v", F.normalize(v, dim=1, eps=self.eps))

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
            self.initialize(weight_mat)
            n_power_iterations = self.default_n_iterations

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            self._u = F.normalize(
                torch.einsum('bij,bj->bi', weight_mat, self._v),
                dim=1,
                eps=self.eps,
                out=self._u,
            )
            self._v = F.normalize(
                torch.einsum('bji,bj->bi', weight_mat.transpose(-2, -1), self._u),
                dim=1,
                eps=self.eps,
                out=self._v,
            )

        self.initialized = True

    def forward(self, weight_mat: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._power_method(weight_mat, self.n_power_iterations)
        # See above on why we need to clone
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)
        # The proper way of computing this should be through F.bilinear, but
        # it seems to have some efficiency issues:
        # https://github.com/pytorch/pytorch/issues/58093
        sigma = torch.linalg.vecdot(u, torch.einsum('bij,bj->bi', weight_mat, v))
        return sigma
