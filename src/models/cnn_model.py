from typing import Sequence, Union, Optional, Tuple, List

import torch
from torch import nn, Tensor
from torch.nn import ModuleList, ReLU
from torch.nn.common_types import _size_2_t

from models.cnn_layers import InnerWeightLipschitzConvLayer, LastLipschitzConvLayer, FirstLipschitzConvLayer
from utils import conjt


class DeepLipschitzConvResNet(nn.Module):
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
        self.spectral_norm = None

    def _triangular_inverse_batch(self, R: Tensor, upper: bool = True) -> Tensor:
        F, n, _ = R.shape
        I = torch.eye(n, dtype=R.dtype, device=R.device).unsqueeze(0).expand(F, -1, -1)
        return torch.linalg.solve_triangular(R, I, upper=upper, left=True)

    def _cholesky_rank_k_freq(self, prev_sigma_lower: Optional[Tensor], T: Tensor) -> Tensor:
        if prev_sigma_lower is None:
            return T
        else:
            prev = prev_sigma_lower
            S_prev = torch.matmul(prev, conjt(prev))
            S_t = torch.matmul(T, conjt(T))
            S = S_prev + S_t
            L = torch.linalg.cholesky(S, upper=False)
            return L

    def compute_b(self, x: Tensor, sigma_lower: Optional[Tensor], a_weight_freq: Tensor, prev_r: Tensor) -> Tuple[
        Tensor, Tensor]:
        # inner_inverse = prev_r^{-1}
        inner_inverse = self._triangular_inverse_batch(prev_r, upper=True)
        # spectral norm scaling
        if self.spectral_norm is None:
            max_singular = 1.0
        else:
            max_singular = self.spectral_norm(inner_inverse)

        if sigma_lower is None:
            left = a_weight_freq
        else:
            left = torch.matmul(a_weight_freq, sigma_lower)

        scaling = torch.matmul(left, conjt(left))

        Iout = torch.eye(scaling.shape[-1], dtype=scaling.dtype, device=scaling.device)

        scaling = scaling + Iout

        L_sigma = torch.linalg.cholesky(scaling, upper=False)
        R_sigma = conjt(L_sigma)  # upper triangular

        second, b_weight_spatial = self.B.forward(x, R_sigma)
        if isinstance(max_singular, torch.Tensor):
            second_s = second * max_singular
            b_weight_s = b_weight_spatial * max_singular
        else:
            second_s = second * float(max_singular)
            b_weight_s = b_weight_spatial * float(max_singular)
        return second_s, b_weight_s

    def forward(self, x: Tensor) -> Tensor:
        B, C_in, H, W = x.shape
        # First layer -> get normalized freq matrices once
        first, W_spatial_A, Wfreq_A, d_sqrt, d_inv = self.A.forward(x)
        # initial gamma (identity per-frequency)
        Ffreq = d_inv.shape[0]
        n_inA = d_inv.shape[-1]
        gamma = torch.eye(n_inA, dtype=d_inv.dtype, device=d_inv.device).unsqueeze(0).expand(Ffreq, -1, -1)

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
            gamma = torch.matmul(T, conjt(c_freq_norm))
            current_input = self.activation(current_input)

        # obtain prev_r from last inner layer (omega_r stored on that layer in forward)
        prev_r = self.layers[-1].omega_r
        second, b_spatial = self.compute_b(current_input, sigma_lower, Wfreq_A, prev_r)
        return first + second


import torch
import torch.nn.functional as F
from torch import nn, Tensor


class SpectralNorm(nn.Module):
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

    def __init__(self, n_power_iterations: int = 1, eps: float = 1e-12) -> None:
        super().__init__()
        if n_power_iterations <= 0:
            raise ValueError("n_power_iterations must be >= 1")
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        # buffers will be created in initialize()
        self.register_buffer("_u", None)
        self.register_buffer("_v", None)

    def initialize(self, weight_mat: Tensor, n_init_steps: int = 15) -> None:
        """
        Initialize _u and _v with random vectors and run power method n_init_steps times.
        weight_mat: (F, n, n) or (n, n)
        """
        with torch.no_grad():
            if weight_mat.ndim == 2:
                # single matrix -> unsqueeze a batch dim of size 1
                W = weight_mat.unsqueeze(0)
            elif weight_mat.ndim == 3:
                W = weight_mat
            else:
                raise ValueError("weight_mat must be 2D or 3D")

            F, n, _ = W.shape
            # create random initial u, v
            u = W.new_empty(F, n).normal_(0, 1)
            v = W.new_empty(F, n).normal_(0, 1)
            u = F.normalize(u, dim=-1, eps=self.eps)
            v = F.normalize(v, dim=-1, eps=self.eps)

            # register buffers (replace any existing)
            self._u = u
            self._v = v

            # do a few power iterations to initialize
            self._power_method(W, n_init_steps)

    @torch.no_grad()
    def _power_method(self, W: Tensor, n_iters: int) -> None:
        """
        Batched power iteration on W shape (F, n, n).
        Updates self._u and self._v in-place (buffers).
        """
        assert W.ndim == 3
        Fbatch = W.shape[0]
        # Ensure buffers exist and have matching shapes
        if self._u is None or self._v is None:
            raise RuntimeError("Buffers not initialized; call initialize(weight_mat) first.")

        # loop n_iters (usually small)
        for _ in range(n_iters):
            # u <- normalize( W @ v )
            # batched matvec: (F, n, n) @ (F, n) -> (F, n)
            v = self._v  # shape (F, n)
            u_new = torch.einsum("fij,fj->fi", W, v)
            F.normalize(u_new, dim=-1, eps=self.eps, out=self._u)  # in-place update _u

            # v <- normalize( W^H @ u )
            u = self._u
            # W^H: conj transpose along last two dims
            v_new = torch.einsum("fji,fj->fi", W.conj(), u)  # uses conj transpose
            F.normalize(v_new, dim=-1, eps=self.eps, out=self._v)  # in-place update _v

    def forward(self, weight: Tensor) -> Tensor:
        """
        Compute spectral norm(s) for weight.
        - If weight is (n,n): returns spectral norm scalar (real)
        - If weight is (F,n,n): returns scalar max singular value across batch (real)
          (if you want per-frequency values, change the final line to return sigma_per_freq)
        """
        if weight.ndim == 2:
            # Single matrix path: convert to batched form for reuse
            W = weight.unsqueeze(0)
            auto_init = True
        elif weight.ndim == 3:
            W = weight
            auto_init = False
        else:
            raise ValueError("weight must be 2D or 3D")

        Fbatch, n, _ = W.shape

        # initialize buffers if missing or batch-size changed
        if self._u is None or self._v is None or self._u.shape[0] != Fbatch:
            # cost: this runs an initialization power-method (15 iters) -> relatively expensive
            # but it ensures buffers have correct batch-size
            self.initialize(W)

        # run configured number of power iterations if training
        if self.training and self.n_power_iterations > 0:
            # small number of power iterations; updates _u and _v in-place
            self._power_method(W, self.n_power_iterations)

        # clone u,v for safe use (same reason as PyTorch implementation)
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)

        # compute W @ v (batched matvec)
        Wv = torch.einsum("fij,fj->fi", W, v)  # (F, n)
        # compute sigma per frequency: <u, W v>
        # note: u can be complex; use conj for left vector
        sigma_per_freq = torch.sum(u.conj() * Wv, dim=-1)  # (F,) complex scalars
        # absolute value -> real singular values
        sigma_vals = sigma_per_freq.abs()

        # return the maximum singular value across frequencies (scalar)
        sigma_max = sigma_vals.max()
        # If you prefer per-frequency values, return sigma_vals instead.
        return sigma_max
