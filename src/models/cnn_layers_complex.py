from __future__ import annotations

from typing import Optional, Sequence, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.nn.utils import parametrize

from utils import PositiveExp


def safe_cholesky(A: Tensor, *, upper: bool, jitter: float = 1e-6) -> Tensor:
    """Hermitian Cholesky with a tiny diagonal jitter if needed (complex-safe)."""
    try:
        return torch.linalg.cholesky(A, upper=upper)
    except RuntimeError:
        I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
        return torch.linalg.cholesky(A + jitter * I, upper=upper)


def _center_and_pad_kernel(weight: Tensor, H: int, W: int) -> Tensor:
    """Center the spatial kernel and pad to (H, W) so FFT represents linear conv."""
    out_ch, in_ch, kH, kW = weight.shape
    padded = F.pad(weight, (0, W - kW, 0, H - kH))
    # move the kernel center to the origin
    return torch.roll(padded, shifts=(-kH // 2, -kW // 2), dims=(-2, -1))


def _freq_batchify_complex(K_fft: Tensor) -> Tuple[Tensor, int, int]:
    """
    (out, in, H, Wf) complex -> (F, out, in) complex, where F = H * Wf.
    Frequency is just a batch axis.
    """
    o, i, H, Wf = K_fft.shape
    F = H * Wf
    K_flat = K_fft.permute(2, 3, 0, 1).reshape(F, o, i)
    return K_flat, H, Wf


def _unbatchify_complex(M_flat: Tensor, H: int, Wf: int, out: int, inn: int) -> Tensor:
    """(F, out, in) -> (out, in, H, Wf)"""
    return M_flat.reshape(H, Wf, out, inn).permute(2, 3, 0, 1)


def _ifft_to_spatial(Wf_norm: Tensor, H: int, W: int, kernel_size: _size_2_t) -> Tensor:
    """
    Wf_norm: (out, in, H, Wf) complex -> (out, in, kH, kW) real spatial kernel.
    """
    kH, kW = _pair(kernel_size)
    pad_spatial = torch.fft.irfft2(Wf_norm, s=(H, W), norm="backward").real
    # undo centering and crop
    pad_spatial = torch.roll(pad_spatial, shifts=(kH // 2, kW // 2), dims=(-2, -1))
    return pad_spatial[..., :kH, :kW]


# ----------------- core: sequential, spatial, complex math -------------------

class _AbstractLipConvSeqSpatial(nn.Conv2d):
    """
    Sequential (feedforward) Lipschitz-constrained 2D conv layer:
    - Uses complex FFT to normalize a conv kernel per frequency.
    - Maps the normalized filter back to spatial kernel and applies F.conv2d.
    - Supports exact conv2d semantics (stride/dilation/padding) for groups=1.
    The LDLᵀ-style normalization mirrors the linear case on a per-frequency matrix.

    Returns from forward (for debugging/verification):
      y, Wf_norm, Dinv, alpha, Omega_out_lower, const
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 activation: Optional[nn.Module] = None,
                 *,
                 constant_sq: float = 1.0,
                 enable_alpha: bool = True,
                 device=None, dtype=None) -> None:

        if isinstance(stride, tuple):  # make scalar
            assert all(s == 1 for s in stride), "Normalized conv requires stride=1; downsample with AvgPool."
        else:
            assert stride == 1, "Normalized conv requires stride=1; downsample with AvgPool."
        if isinstance(dilation, tuple):
            assert all(d == 1 for d in dilation), "Normalized conv requires dilation=1."
        else:
            assert dilation == 1, "Normalized conv requires dilation=1."

        if groups != 1:
            raise NotImplementedError("groups>1 not yet supported in normalized complex formulation.")

        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, "zeros", device, dtype)

        self.activation = activation

        # buffers & parameters
        factory = dict(device=device, dtype=dtype)
        self.register_buffer("I_in", torch.eye(in_channels, **factory))
        self.register_buffer("I_out", torch.eye(out_channels, **factory))

        self.register_buffer("c2", torch.tensor(float(constant_sq), **factory))
        self.register_buffer("c", torch.sqrt(torch.tensor(float(constant_sq), **factory)))

        alpha0 = torch.tensor(1.0, **factory)
        if enable_alpha:
            self.alpha = nn.Parameter(alpha0)
            parametrize.register_parametrization(self, "alpha", PositiveExp())
        else:
            self.register_buffer("alpha", alpha0)

    # ---- complex frequency helpers ----

    def _weight_fft(self, H: int, W: int) -> Tuple[Tensor, int, int]:
        """
        Returns Wf: (F, out, in) complex, plus (H, Wf) to reconstruct.
        """
        Kpad = _center_and_pad_kernel(self.weight, H, W)  # (o,i,H,W)
        Kfft = torch.fft.rfft2(Kpad, s=(H, W), norm="backward")  # (o,i,H,Wf) complex
        Wf, Hf, Wff = _freq_batchify_complex(Kfft)  # (F,o,i)
        return Wf, Hf, Wff

    def _Omega_in(self, Wf: Tensor) -> Tensor:
        # (F,in,in) Hermitian: Wfᴴ Wf + alpha I_in
        return Wf.mH @ Wf + self.alpha * self.I_in

    def _Omega_out(self, Wf: Tensor) -> Tensor:
        # (F,out,out) Hermitian: Wf Wfᴴ + alpha I_out
        return Wf @ Wf.mH + self.alpha * self.I_out

    # ---- spatial conv with normalized complex filters ----

    def _conv2d_with_spatial_weight(self,
                                    x: Tensor,
                                    Wf_norm: Tensor,
                                    Hf: int, Wff: int,
                                    H: int, W: int) -> Tensor:
        """
        Map normalized frequency weights to spatial kernel and apply conv2d.
        """
        Wf_norm_full = _unbatchify_complex(Wf_norm, Hf, Wff, self.out_channels, self.in_channels)  # (o,i,H,Wf)
        Ksp = _ifft_to_spatial(Wf_norm_full, H, W, self.kernel_size)  # (o,i,kH,kW)
        y = F.conv2d(x, Ksp, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


class FirstSequentialLipConv(_AbstractLipConvSeqSpatial):
    """
    First layer in sequential chain:
      - Right-normalize by R_f from Omega_in (input Gram).
      - Scale by c (layer budget).
      - For Dinv/Fisher-like bookkeeping, also form Omega_out.
    """

    @torch.compiler.disable(reason='Using complex matrix multiplication.')
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        B, Cin, H, W = x.shape
        Wf, Hf, Wff = self._weight_fft(H, W)  # (F,o,i) complex
        Omega_in = self._Omega_in(Wf)  # (F,i,i)
        R = safe_cholesky(Omega_in, upper=True)  # (F,i,i) upper
        # right solve: X R = Wf  =>  X = Wf R^{-1}
        Wn = torch.linalg.solve_triangular(R, Wf, upper=True, left=False)  # (F,o,i)
        Wn = self.c * Wn

        # bookkeeping for sequential composition (lives on outputs)
        Omega_out = self._Omega_out(Wf)  # (F,o,o)
        Dinv = Omega_out / (self.c2 * self.alpha)  # (F,o,o)
        L_lower = safe_cholesky(Omega_out, upper=False)  # (F,o,o)

        y = self._conv2d_with_spatial_weight(x, Wn, Hf, Wff, H, W)
        if self.activation is not None:
            y = self.activation(y)

        return y, Wn, Dinv, self.alpha, L_lower, self.c


class InnerSequentialLipConv(_AbstractLipConvSeqSpatial):
    """
    Inner (and last) layers in the sequential chain:
      - Right-normalize by R_f from Omega_in.
      - Project through previous output factor: right-solve by prev L_lower.
      - Scale by sqrt(prev_alpha) * prev_c * c (the standard 2-Lipschitz composition).
      - Update Omega_out, Dinv, and new L_lower.
    """

    @torch.compiler.disable(reason='Using complex matrix multiplication.')
    def forward(self,
                x: Tensor,
                prev_alpha: Optional[Tensor],
                prev_L_lower: Optional[Tensor],
                prev_c: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        B, _, H, W = x.shape
        Wf, Hf, Wff = self._weight_fft(H, W)  # (F,o,i) complex
        F = Wf.shape[0]

        # Right-normalize columns by R_f from Omega_in
        Omega_in = self._Omega_in(Wf)  # (F,i,i)
        R = safe_cholesky(Omega_in, upper=True)
        Wn = torch.linalg.solve_triangular(R, Wf, upper=True, left=False)  # (F,o,i)

        # Decide whether we can project through L_prev
        use_projection = (
                prev_L_lower is not None
                and prev_alpha is not None
                and prev_c is not None
                and prev_L_lower.shape[0] == F
                and prev_L_lower.shape[1] == self.in_channels
                and prev_L_lower.shape[2] == self.in_channels
        )

        if use_projection:
            # Project through previous output basis (same F, no size change)
            Wn = torch.linalg.solve_triangular(prev_L_lower, Wn, upper=False, left=False)
            scale = torch.sqrt(prev_alpha) * prev_c * self.c
        else:
            # RESET: size changed (e.g., AvgPool or stride>1) or state absent
            scale = self.c

        Wn = scale * Wn

        # Update out-side Gram and its inverse factorization for the next layer (at current resolution)
        Omega_out = self._Omega_out(Wf)  # (F,o,o)
        Dinv = Omega_out / (self.c2 * self.alpha)
        L_lower = safe_cholesky(Omega_out, upper=False)

        # Map normalized weights back to spatial and apply conv2d
        y = self._conv2d_with_spatial_weight(x, Wn, Hf, Wff, H, W)
        if self.activation is not None:
            y = self.activation(y)

        return y, Wn, Dinv, self.alpha, L_lower, self.c


class LipschitzBatchNorm2d(nn.Module):
    """
    BatchNorm whose forward is always *samplewise affine* with slope s_c = s_max*tanh(rho_c),
    using running statistics. This keeps the forward ℓ2-Lipschitz constant ≤ s_max.
    We still update running stats (EMA) in training for adaptation.
    """

    def __init__(self, num_features: int,
                 eps: float = 1e-5, momentum: float = 0.1,
                 s_max: float = 1.0,
                 affine_bias: bool = True,
                 track_running_stats: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.s_max = float(s_max)
        self.track_running_stats = track_running_stats

        factory = dict(device=device, dtype=dtype)
        # trainable per-channel slope (reparam via tanh)
        self.rho = nn.Parameter(torch.zeros(num_features, **factory))
        # trainable bias
        self.bias = nn.Parameter(torch.zeros(num_features, **factory)) if affine_bias else None

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features, **factory))
            self.register_buffer("running_var", torch.ones(num_features, **factory))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long, device=device))
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    @torch.no_grad()
    def _update_running(self, x: torch.Tensor):
        if not self.track_running_stats or not self.training:
            return
        self.num_batches_tracked += 1
        momentum = self.momentum
        # per-channel moments over N,H,W
        mean = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)
        self.running_mean.mul_(1 - momentum).add_(momentum * mean)
        self.running_var.mul_(1 - momentum).add_(momentum * var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update running stats from the batch (no gradient)
        self._update_running(x)

        # Use *running* stats for the forward to keep it samplewise affine
        if self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        else:
            # Fallback: use per-batch stats but still keep samplewise affine
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)

        s = self.s_max * torch.tanh(self.rho)  # per-channel slope bound in [-s_max, s_max]
        inv_std = torch.rsqrt(var + self.eps)  # 1/sqrt(var+eps)
        # Compute *gamma* so that effective slope s = gamma / sqrt(var + eps)
        gamma = s * (1.0 / inv_std)
        beta = self.bias if self.bias is not None else 0.0

        # Affine map per channel (samplewise)
        y = (x - mean[None, :, None, None]) * (gamma[None, :, None, None] * inv_std[None, :, None, None]) + beta
        # Note: gamma*inv_std simplifies to s, but writing it this way makes the logic explicit
        # and keeps gradients flowing to rho through s.

        return y

    def lipschitz(self) -> float:
        # Exact ℓ2 Lipschitz of this layer (samplewise) is max|s_c| ≤ s_max
        with torch.no_grad():
            return float((self.s_max * torch.tanh(self.rho)).abs().max().item())


class LipschitzAvgPool2d(nn.Module):
    """AvgPool2d; ℓ2-Lipschitz ≤ 1 (see notes)."""

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad)

    def lipschitz(self) -> float:
        return 1.0


class DeepLipschitzConvSequentialSpatial(nn.Module):
    """
    Sequential Lipschitz CNN (spatial convs), with optional BN (samplewise, bounded)
    and AvgPool (≤1 Lipschitz). Ordering per block: Conv -> BN (opt) -> ReLU (opt) -> Pool (opt).
    The first and inner convs implement LDLᵀ-style normalization in the complex domain,
    then convert to spatial kernels for convolution.

    NOTE: BN uses running stats in forward; this preserves samplewise Lipschitz bounds.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layers_widths: Sequence[int],
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 1,
                 dilation: _size_2_t = 1,
                 activation: Optional[nn.Module] = nn.ReLU(),
                 lipschitz_constant: float = 1.0,
                 enable_alpha: bool = True,
                 use_bn: bool = True,
                 bn_smax: float = 1.0,
                 use_pool: bool = False,
                 pool_kernel: int = 2,
                 pool_stride: Optional[int] = None,
                 device: Union[str, torch.device] = "cuda",
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        assert len(layers_widths) >= 1

        self.activation = activation
        self.use_bn = use_bn
        self.use_pool = use_pool

        factory = dict(device=device, dtype=dtype)
        c2_first = float(lipschitz_constant) ** 2

        # --- Build conv stack (no activation inside convs) ---
        convs: List[nn.Module] = []
        bns: List[nn.Module] = []
        pools: List[nn.Module] = []

        # First layer
        convs.append(
            FirstSequentialLipConv(in_channels, layers_widths[0],
                                   kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   activation=None, constant_sq=c2_first, enable_alpha=enable_alpha, **factory)
        )
        if use_bn:
            bns.append(LipschitzBatchNorm2d(layers_widths[0], s_max=bn_smax, **factory))
        if use_pool:
            pools.append(LipschitzAvgPool2d(pool_kernel, stride=pool_stride))

        # Middle layers (constant_sq = 2)
        for i in range(1, len(layers_widths)):
            convs.append(
                InnerSequentialLipConv(layers_widths[i - 1], layers_widths[i],
                                       kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                       activation=None, constant_sq=2.0, enable_alpha=enable_alpha, **factory)
            )
            if use_bn:
                bns.append(LipschitzBatchNorm2d(layers_widths[i], s_max=bn_smax, **factory))
            if use_pool:
                pools.append(LipschitzAvgPool2d(pool_kernel, stride=pool_stride))

        # Final conv to out_channels (constant_sq = 1, no activation)
        convs.append(
            InnerSequentialLipConv(layers_widths[-1], out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   activation=None, constant_sq=1.0, enable_alpha=enable_alpha, **factory)
        )
        # Optionally a last BN with s_max ≤ 1 (safe). No ReLU afterward unless you want it.
        if use_bn:
            bns.append(LipschitzBatchNorm2d(out_channels, s_max=bn_smax, **factory))
        # Pool after last layer is typically unusual; keep it off by default.
        # (add if you really need it)

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.pools = nn.ModuleList(pools)

    def forward(self, x: Tensor) -> Tensor:
        # State carried across convs for LDLᵀ recursion
        y, Wn, Dinv, alpha, L_lower, c = self.convs[0](x)  # no activation inside
        idx_bn, idx_pool = 0, 0

        if self.use_bn and idx_bn < len(self.bns):
            y = self.bns[idx_bn](y)
            idx_bn += 1
        if self.activation is not None:
            y = self.activation(y)
        if self.use_pool and idx_pool < len(self.pools):
            y = self.pools[idx_pool](y)
            idx_pool += 1

        # Inner and final
        for conv_idx in range(1, len(self.convs)):
            y, Wn, Dinv, alpha, L_lower, c = self.convs[conv_idx](y, alpha, L_lower, c)

            # For the very last conv (to out_channels), you might skip BN/activation/pool
            is_last = (conv_idx == len(self.convs) - 1)

            if self.use_bn and idx_bn < len(self.bns):
                y = self.bns[idx_bn](y)
                idx_bn += 1
            if not is_last and self.activation is not None:
                y = self.activation(y)
            if self.use_pool and idx_pool < len(self.pools) and not is_last:
                y = self.pools[idx_pool](y)
                idx_pool += 1

        return y
