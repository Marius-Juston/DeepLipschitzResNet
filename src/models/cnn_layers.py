from abc import abstractmethod, ABC
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Parameter
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.nn.utils import parametrize

from utils import PositiveExp


def _center_and_pad_kernel(kernel: Tensor, H: int, W: int) -> Tensor:
    out_ch, in_ch, kH, kW = kernel.shape
    padded = F.pad(kernel, (0, W - kW, 0, H - kH))
    return torch.roll(padded, shifts=(- (kH // 2), - (kW // 2)), dims=(-2, -1))


def _unroll_and_crop_spatial(pad_spatial: Tensor, kH: int, kW: int) -> Tensor:
    shift_h = (kH // 2)
    shift_w = (kW // 2)
    pad_spatial = torch.roll(pad_spatial, shifts=(shift_h, shift_w), dims=(-2, -1))
    return pad_spatial[..., :kH, :kW]


def _freq_batchify(K_fft: Tensor) -> Tuple[Tensor, int, int]:
    # input K_fft: (out, in, H, Wf) complex -> returns (F, out, in)
    out_ch, in_ch, H, Wf = K_fft.shape
    F = H * Wf
    K_flat = K_fft.permute(2, 3, 0, 1).reshape(F, out_ch, in_ch)
    return K_flat, H, Wf


def _unbatchify_freq(M_flat: Tensor, H: int, Wf: int) -> Tensor:
    F, out_ch, in_ch = M_flat.shape
    # assert F == H * Wf
    M = M_flat.reshape(H, Wf, out_ch, in_ch).permute(2, 3, 0, 1).contiguous()
    return M  # (out, in, H, Wf)


class AbstractLipschitzConvLayer(nn.Conv2d, ABC):
    identity: torch.Tensor
    identity_out: torch.Tensor
    constant_sq: torch.Tensor
    constant: torch.Tensor
    R: torch.Tensor
    omega_r: torch.Tensor

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 constant_sq: float = 1.0,
                 enable_alpha: bool = True,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 device=None,
                 dtype=None, ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode,
                         device=device,
                         dtype=dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.register_buffer(
            "identity", torch.eye(in_channels, **factory_kwargs)
        )

        self.register_buffer(
            "identity_out", torch.eye(out_channels, **factory_kwargs)
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

    def sum_weight(self, Wf, transpose: bool = False):
        if transpose:
            return self.alpha * self.identity + torch.matmul(Wf.mH, Wf)
        else:
            return self.alpha * self.identity_out + torch.matmul(Wf, Wf.mH)

    def scale_w(self, Wf: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        omega = self.sum_weight(Wf, transpose=True)
        R = torch.linalg.cholesky(omega, upper=True)

        W = torch.linalg.solve_triangular(R, Wf, upper=True, left=False)

        return omega, W, R

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

    def spectral_norm_frequency(self, H: int, W: int) -> Tuple[Tensor, int, int]:
        K_fft = self._weight_fft_for_input(H, W)  # (out,in,H,Wf)
        Wf, H_, Wf_ = _freq_batchify(K_fft)  # (F,out,in)

        return Wf, H_, Wf_

    def convert_to_spatial(self, Wnorm_f, H_, Wf_, W):
        M_freq = _unbatchify_freq(Wnorm_f, H_, Wf_)
        pad_spatial = self._ifft_from_freq(M_freq, H_, W)
        kH, kW = _pair(self.kernel_size)
        return _unroll_and_crop_spatial(pad_spatial, kH, kW)

    @abstractmethod
    def process(self, Wf, *args, **kwargs):
        pass

    def forward(self, input: Tensor, *args) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B, C_in, H, W = input.shape

        Wf, H_, Wf_ = self.spectral_norm_frequency(H, W)

        Wf, d_inv = self.process(Wf, *args)

        W_spatial = self.convert_to_spatial(Wf, H_, Wf_, W)

        output = self.conv_with_normalized_weight(input, W_spatial, self.bias)

        return output, W_spatial, Wf, d_inv


class FirstLipschitzConvLayer(AbstractLipschitzConvLayer):
    def process(self, Wf):
        omega, W, R = self.scale_w(Wf)

        W = self.constant * W

        d_inv = omega / (self.constant_sq * self.alpha)

        self.omega_r = R

        return W, d_inv


class InnerWeightLipschitzConvLayer(AbstractLipschitzConvLayer):
    def process(self, Wf: Tensor, prev_layer: Union[FirstLipschitzConvLayer, "InnerWeightLipschitzConvLayer"]) -> \
            Tuple[Tensor, Tensor]:
        _, W, _ = self.scale_w(Wf)

        W = torch.linalg.solve_triangular(prev_layer.omega_r.mH, W, upper=False, left=False)

        scale = prev_layer.alpha.sqrt() * prev_layer.constant * self.constant
        div_scale = self.constant_sq * self.alpha

        W = scale * W

        omega = self.sum_weight(Wf)

        d_inv = omega / div_scale

        self.omega_r = torch.linalg.cholesky(omega, upper=True)

        return W, d_inv


class LastLipschitzConvLayer(AbstractLipschitzConvLayer):
    def process(self, Wf: Tensor, sigma_r: Tensor, prev_r_sqrt: torch.Tensor, prev_alpha: torch.Tensor):
        _, W, _ = self.scale_w(Wf)

        W = torch.linalg.solve_triangular(prev_r_sqrt.mH, W, upper=False, left=False)
        W = self.constant * prev_alpha.sqrt() * torch.linalg.solve_triangular(sigma_r, W, upper=True, left=True)

        return W

    def forward(self, input: Tensor, sigma_r: Tensor, prev_layer: InnerWeightLipschitzConvLayer) -> Tuple[
        Tensor, Tensor]:
        B, C_in, H, W = input.shape

        Wf, H_, Wf_ = self.spectral_norm_frequency(H, W)

        prev_alpha = prev_layer.alpha
        prev_r = prev_layer.omega_r

        Wf = self.process(Wf, sigma_r, prev_r, prev_alpha)

        W_spatial = self.convert_to_spatial(Wf, H_, Wf_, W)

        output = self.conv_with_normalized_weight(input, W_spatial, self.bias)

        return output, W_spatial, Wf
