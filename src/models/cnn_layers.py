from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.modules.utils import _pair

from utils import conjt


def _center_and_pad_kernel(kernel: Tensor, H: int, W: int) -> Tensor:
    out_ch, in_ch, kH, kW = kernel.shape
    device = kernel.device
    dtype = kernel.dtype
    pad = torch.zeros((out_ch, in_ch, H, W), dtype=dtype, device=device)
    pad[..., :kH, :kW] = kernel
    shift_h = - (kH // 2)
    shift_w = - (kW // 2)
    pad = torch.roll(pad, shifts=(shift_h, shift_w), dims=(-2, -1))
    return pad


def _unroll_and_crop_spatial(pad_spatial: Tensor, kH: int, kW: int) -> Tensor:
    shift_h = (kH // 2)
    shift_w = (kW // 2)
    pad_spatial = torch.roll(pad_spatial, shifts=(shift_h, shift_w), dims=(-2, -1))
    return pad_spatial[..., :kH, :kW].contiguous()


def _freq_batchify(K_fft: Tensor) -> Tuple[Tensor, int, int]:
    # input K_fft: (out, in, H, Wf) complex -> returns (F, out, in)
    out_ch, in_ch, H, Wf = K_fft.shape
    F = H * Wf
    K_flat = K_fft.permute(2, 3, 0, 1).reshape(F, out_ch, in_ch)
    return K_flat, H, Wf


def _unbatchify_freq(M_flat: Tensor, H: int, Wf: int) -> Tensor:
    F, out_ch, in_ch = M_flat.shape
    assert F == H * Wf
    M = M_flat.reshape(H, Wf, out_ch, in_ch).permute(2, 3, 0, 1).contiguous()
    return M  # (out, in, H, Wf)


def _compute_cholesky_and_normalize(Wf: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    # Wf: (F, out, in) complex
    F, out_ch, in_ch = Wf.shape
    device = Wf.device
    dtype = Wf.dtype

    W_conj_t = Wf.conj().permute(0, 2, 1)  # (F, in, out)
    A = torch.matmul(W_conj_t, Wf)  # (F, in, in)
    I = torch.eye(in_ch, dtype=dtype, device=device)
    A = A + I

    # L lower triangular: A = L @ L^*
    L = torch.linalg.cholesky(A)  # (F, in, in)
    R = conjt(L)  # upper triangular such that A = R^* R

    # Solve W @ R^{-1}: solve R X = W  -> X = W @ R^{-1}
    # torch.linalg.solve_triangular supports batched solves: here left=False solves X R = W -> X = W R^{-1}
    Wnorm = torch.linalg.solve_triangular(R, Wf, upper=True, left=False)
    return Wnorm, R, A


class AbstractLipschitzConvLayer(nn.Conv2d):
    constant_sq: torch.Tensor
    constant: torch.Tensor

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, constant_sq: float = 1.0, **conv_kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias,
                         **conv_kwargs)
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}
        self.register_buffer("constant_sq", torch.tensor(constant_sq, **factory_kwargs))
        self.register_buffer("constant", torch.sqrt(self.constant_sq))

    def _weight_fft_for_input(self, H: int, W: int) -> Tensor:
        pad = _center_and_pad_kernel(self.weight, H, W)
        K_fft = torch.fft.rfft2(pad, s=(H, W), norm="ortho")
        return K_fft

    def _ifft_from_freq(self, M_freq: Tensor, H: int, W: int) -> Tensor:
        pad_spatial = torch.fft.irfft2(M_freq, s=(H, W), norm="ortho")
        return pad_spatial

    def conv_with_normalized_weight(self, input: Tensor, W_spatial: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                W_spatial,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(input, W_spatial, bias, self.stride, self.padding, self.dilation, self.groups)

    # A convenience method used by forward to compute normalized frequency matrices and spatial kernel
    def spectral_normalize_weight_for_input(self, H: int, W: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # returns: W_spatial (real), Wfreq_norm (F,out,in) complex, d_sqrt (F,in,in) complex, d_inv (F,in,in) complex
        K_fft = self._weight_fft_for_input(H, W)  # (out,in,H,Wf)
        Wf, H_, Wf_ = _freq_batchify(K_fft)  # (F,out,in)
        Wnorm_f, R, A = _compute_cholesky_and_normalize(Wf)
        Wnorm_f = Wnorm_f * self.constant
        M_freq = _unbatchify_freq(Wnorm_f, H_, Wf_)
        pad_spatial = self._ifft_from_freq(M_freq, H_, W)
        kH, kW = _pair(self.kernel_size)
        W_spatial = _unroll_and_crop_spatial(pad_spatial, kH, kW)
        d_sqrt = R * self.constant
        d_inv = A / self.constant_sq
        return W_spatial, Wnorm_f, d_sqrt, d_inv


class FirstLipschitzConvLayer(AbstractLipschitzConvLayer):
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
          out: spatial conv result
          W_spatial: normalized spatial kernel (out,in,kH,kW) real
          Wfreq_norm: normalized freq matrices (F, out, in) complex
          d_sqrt: (F, in, in) upper triangular complex
          d_inv: (F, in, in) complex
        """
        B, C_in, H, W = input.shape
        W_spatial, Wfreq_norm, d_sqrt, d_inv = self.spectral_normalize_weight_for_input(H, W)
        out = self.conv_with_normalized_weight(input, W_spatial, self.bias)
        # return everything so the ResNet can reuse Wfreq_norm without recomputing
        return out, W_spatial, Wfreq_norm, d_sqrt, d_inv


class InnerWeightLipschitzConvLayer(AbstractLipschitzConvLayer):
    """
    Inner layer: uses prev_d_sqrt to normalize; now returns Wfreq_norm so the ResNet can use it.
    """

    def forward(self, input: Tensor, prev_d_sqrt: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
          input: spatial input (B, in_ch, H, W)
          prev_d_sqrt: (F, in_ch, in_ch) upper triangular complex from previous layer

        Returns:
          out: spatial conv output (B, out_ch, H', W')
          W_spatial: normalized spatial kernel (out,in,kH,kW) real
          Wfreq_norm: (F, out, in) complex normalized freq matrices (already scaled by constant)
          d_sqrt: (F, in, in) complex upper triangular for this layer
          d_inv: (F, in, in) complex A matrix divided by constant_sq
        Side effect:
          self.omega_r is set to the R (upper triangular) for use by the ResNet.
        """
        B, Cin, H, W = input.shape
        # compute raw kernel FFT (out,in,H,Wf)
        K_fft = self._weight_fft_for_input(H, W)
        Wf, H_, Wf_ = _freq_batchify(K_fft)  # (F,out,in)

        # normalize with prev_d_sqrt: solve X * prev_d_sqrt = Wf  => X = Wf @ prev_d_sqrt^{-1}
        prev = prev_d_sqrt.to(Wf.dtype).to(Wf.device)
        # left=False: X * prev = Wf -> X = Wf @ prev^{-1}
        Wf_normed = torch.linalg.solve_triangular(prev, Wf, upper=True, left=False)

        # scale by constant
        const = self.constant.to(Wf_normed.dtype)
        Wf_normed = Wf_normed * const

        # compute this layer's cholesky + normalization
        Wnorm_f, R, A = _compute_cholesky_and_normalize(Wf_normed)
        # Wnorm_f is already Wf_normed @ R^{-1}, and we keep it as the per-frequency normalized matrix
        # reconstruct spatial kernel
        M_freq = _unbatchify_freq(Wnorm_f, H_, Wf_)
        pad_spatial = self._ifft_from_freq(M_freq, H_, W)
        kH, kW = _pair(self.kernel_size)
        W_spatial = _unroll_and_crop_spatial(pad_spatial, kH, kW)

        # store upper triangular factor (omega_r) for use by the network (and B)
        self.omega_r = R  # (F, in, in), upper triangular

        d_sqrt = R * const
        d_inv = A / self.constant_sq.to(A.dtype)

        out = self.conv_with_normalized_weight(input, W_spatial, self.bias)
        # return the frequency normalized matrix so ResNet can update gamma without recomputing
        return out, W_spatial, Wnorm_f, d_sqrt, d_inv


class LastLipschitzConvLayer(AbstractLipschitzConvLayer):
    def forward(self, input: Tensor, prev_r: Tensor) -> Tuple[Tensor, Tensor]:
        """
        prev_r: (F, in, in) upper triangular complex from previous layer
        returns: out spatial, W_spatial (normalized)
        """
        B, Cin, H, W = input.shape
        # compute raw kernel FFT
        K_fft = self._weight_fft_for_input(H, W)
        Wf, H_, Wf_ = _freq_batchify(K_fft)  # (F,out,in)

        prev_r = prev_r.to(Wf.dtype).to(Wf.device)
        Wf_norm = torch.linalg.solve_triangular(prev_r, Wf, upper=True, left=False)
        const = self.constant.to(Wf_norm.dtype)
        Wf_norm = Wf_norm * const

        M_freq = _unbatchify_freq(Wf_norm, H_, Wf_)
        pad_spatial = self._ifft_from_freq(M_freq, H_, W)
        kH, kW = _pair(self.kernel_size)
        W_spatial = _unroll_and_crop_spatial(pad_spatial, kH, kW)

        out = self.conv_with_normalized_weight(input, W_spatial, self.bias)
        return out, W_spatial
