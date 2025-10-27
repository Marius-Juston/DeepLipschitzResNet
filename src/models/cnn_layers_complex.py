from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
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
                prev_alpha: Tensor,
                prev_L_lower: Tensor,
                prev_c: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        B, _, H, W = x.shape
        Wf, Hf, Wff = self._weight_fft(H, W)  # (F,o,i)
        Omega_in = self._Omega_in(Wf)  # (F,i,i)
        R = safe_cholesky(Omega_in, upper=True)  # (F,i,i)

        # Normalize columns and project through previous output basis:
        Wn = torch.linalg.solve_triangular(R, Wf, upper=True, left=False)  # (F,o,i) : Wf R^{-1}
        Wn = torch.linalg.solve_triangular(prev_L_lower, Wn, upper=False, left=False)  # (F,o,i) : Wn L_prev^{-1}

        scale = torch.sqrt(prev_alpha) * prev_c * self.c
        Wn = scale * Wn

        # Update for next layer:
        Omega_out = self._Omega_out(Wf)  # (F,o,o)
        Dinv = Omega_out / (self.c2 * self.alpha)  # (F,o,o)
        L_lower = safe_cholesky(Omega_out, upper=False)

        y = self._conv2d_with_spatial_weight(x, Wn, Hf, Wff, H, W)
        if self.activation is not None:
            y = self.activation(y)

        return y, Wn, Dinv, self.alpha, L_lower, self.c


class DeepLipschitzConvSequentialSpatial(nn.Module):
    """
    A full sequential network:
      [FirstSequentialLipConv] + [InnerSequentialLipConv x (L-1)],
    with the last inner layer having c^2 = 1 and no activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layers_widths: Sequence[int],
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 activation: Optional[nn.Module] = nn.ReLU(),
                 lipschitz_constant: float = 1.0,
                 enable_alpha: bool = True,
                 device: Union[str, torch.device] = "cuda",
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        assert len(layers_widths) >= 1, "Need at least one hidden layer."

        self.factory = dict(device=device, dtype=dtype)
        c2_first = float(lipschitz_constant) ** 2

        self.layers = nn.ModuleList()

        # First layer: from in_channels -> layers_widths[0]
        self.layers.append(
            FirstSequentialLipConv(in_channels, layers_widths[0],
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation,
                                   activation=activation,
                                   constant_sq=c2_first, enable_alpha=enable_alpha,
                                   **self.factory)
        )

        # Middle layers
        for i in range(1, len(layers_widths)):
            self.layers.append(
                InnerSequentialLipConv(layers_widths[i - 1], layers_widths[i],
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding, dilation=dilation,
                                       activation=activation,
                                       constant_sq=2.0, enable_alpha=enable_alpha,
                                       **self.factory)
            )

        # Last layer: hidden -> out_channels, c^2 = 1, no activation
        self.layers.append(
            InnerSequentialLipConv(layers_widths[-1], out_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation,
                                   activation=None,
                                   constant_sq=1.0, enable_alpha=enable_alpha,
                                   **self.factory)
        )

    def forward(self, x: Tensor) -> Tensor:
        # carry the recursion state (alpha, L_lower, c) across layers
        y, Wn, Dinv, alpha, L_lower, c = self.layers[0](x)
        prev_alpha, prev_L_lower, prev_c = alpha, L_lower, c

        for layer in self.layers[1:]:
            y, Wn, Dinv, alpha, L_lower, c = layer(y, prev_alpha, prev_L_lower, prev_c)
            prev_alpha, prev_L_lower, prev_c = alpha, L_lower, c

        return y
