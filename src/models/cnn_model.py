from typing import Sequence, Union, Optional, Tuple, List

import torch
from torch import Tensor
from torch import nn
from torch.nn import ModuleList, ReLU
from torch.nn.common_types import _size_2_t

from models.cnn_layers import InnerWeightLipschitzConvLayer, LastLipschitzConvLayer, FirstLipschitzConvLayer, \
    FirstSequentialLipschitzConvLayer
from utils import safe_cholesky


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
            enable_alpha: bool = True,
            use_safe_cholesky: bool = False,
            wrong_normalization: bool = False,
            device: Union[str, torch.device] = 'cuda',
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_channels > 0, "The input width needs to be larger than 0"
        assert out_channels > 0, "The output width needs to be larger than 0"

        self.cholesky_function = safe_cholesky if use_safe_cholesky else torch.linalg.cholesky
        self.enable_alpha = enable_alpha
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

        self.generate_layers()

    def generate_layers(self):
        self.A = FirstLipschitzConvLayer(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
            enable_alpha=self.enable_alpha,
            constant_sq=self.lipschitz_constant_sq,
            **self.factory_kwargs
        )

        layers = []

        prev = self.in_channels

        for i, out_features in enumerate(self.layer_widths):
            layers.append(
                InnerWeightLipschitzConvLayer(prev, out_features,
                                              enable_alpha=self.enable_alpha, constant_sq=2.0,
                                              kernel_size=self.kernel_size,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation=self.dilation,
                                              activation=self.activation,
                                              groups=self.groups, **self.factory_kwargs))
            prev = out_features

        self.layers = ModuleList(layers)

        self.B = LastLipschitzConvLayer(
            in_channels=self.layer_widths[-1],
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            # Don't need bias on both A and B since they would be doing the same thing
            bias=False,
            **self.factory_kwargs
        )

    def compute_b(self, x: Tensor, sigma_lower, a_wf, prev_alpha: Tensor, prev_omega_r: Tensor) -> Tuple[
        Tensor, Tensor]:

        # left = a_wf @ sigma_lower
        # scaling = self.B.identity_out + left @ left.mT

        scaling = self.B.identity_out + torch.einsum("fir,fjs,frk,fsk->fij", a_wf, a_wf, sigma_lower, sigma_lower)

        R_sigma = self.cholesky_function(scaling, upper=True)

        second, b_wf = self.B(x, R_sigma, prev_alpha, prev_omega_r)

        return second, b_wf

    # @torch.compiler.disable
    def forward(self, x: Tensor) -> Tensor:
        first, a_wf, d_inv, prev_alpha, prev_omega_r, prev_constant = self.A(x)

        gamma = torch.eye(d_inv.shape[1], device=d_inv.device, dtype=d_inv.dtype).expand(d_inv.shape[0], -1, -1)
        sigma_lower = None

        current_input = x

        for layer in self.layers:
            T = gamma @ d_inv

            current_input, c_wf, d_inv, prev_alpha, prev_omega_r, prev_constant = layer(current_input, prev_alpha,
                                                                                        prev_omega_r, prev_constant)

            if sigma_lower is None:
                sigma_lower = T
            else:
                sigma_lower = self.cholesky_function(sigma_lower @ sigma_lower.mT + T @ T.mT, upper=False)

            gamma = T @ c_wf.mT

        second, b_wf = self.compute_b(current_input, sigma_lower, a_wf, prev_alpha, prev_omega_r)

        return first + second


class DeepLipschitzConvSequential(nn.Module):
    layers: ModuleList

    def __init__(self, in_channels: int,
                 out_channels: int,
                 layers_widths: Sequence[int],
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 activation: Union[nn.Module, List[nn.Module]] = None,
                 lipschitz_constant: float = 1.0,
                 enable_alpha: bool = True,
                 use_safe_cholesky: bool = False,
                 wrong_normalization: bool = False,
                 dropout: bool = False,
                 use_averaging: bool = False,
                 device: Union[str, torch.device] = 'cuda',
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()

        super().__init__()
        self.use_averaging = use_averaging
        self.dropout = dropout
        self.wrong_normalization = wrong_normalization
        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_channels > 0, "The input width needs to be larger than 0"
        assert out_channels > 0, "The output width needs to be larger than 0"

        self.cholesky_function = safe_cholesky if use_safe_cholesky else torch.linalg.cholesky
        self.enable_alpha = enable_alpha
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

        self.generate_layers()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def generate_layers(self) -> None:
        layer = []

        prev = self.in_channels

        for i, out_features in enumerate(self.layer_widths):
            activation = self.activation
            constant = 2.0

            if i == 0:
                class_ = FirstSequentialLipschitzConvLayer
            else:
                class_ = InnerWeightLipschitzConvLayer

            if isinstance(out_features, Sequence):
                out_channels, kernel_size = out_features
                padding = kernel_size // 2
            else:
                out_channels = out_features
                kernel_size = self.kernel_size
                padding = self.padding

            layer.append(
                class_(prev, out_channels,
                       kernel_size=kernel_size,
                       enable_alpha=self.enable_alpha,
                       constant_sq=constant,
                       stride=self.stride,
                       padding=padding,
                       dilation=self.dilation,
                       activation=activation,
                       groups=self.groups, **self.factory_kwargs))

            if self.use_averaging and i % 2 == 0:
                layer.append(nn.AvgPool2d(2, divisor_override=2))

            if self.dropout:
                layer.append(nn.Dropout())

            prev = out_channels

        layer.append(
            InnerWeightLipschitzConvLayer(prev, self.out_channels, kernel_size=self.kernel_size, constant_sq=1.0,
                                          activation=None,
                                          **self.factory_kwargs))

        self.layers = ModuleList(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_input = x

        first = True

        for layer in self.layers:
            if isinstance(layer, nn.Dropout) or isinstance(layer, nn.AvgPool2d):
                current_input = layer(current_input)
            else:
                if first:
                    current_input, _, d_inv, prev_alpha, prev_omega_r, prev_constant = layer(current_input)
                    first = False
                else:
                    current_input, _, d_inv, prev_alpha, prev_omega_r_temp, prev_constant = layer(current_input,
                                                                                                  prev_alpha,
                                                                                                  prev_omega_r,
                                                                                                  prev_constant)

                    if not self.wrong_normalization:
                        prev_omega_r = prev_omega_r_temp

            # current_input, _, d_inv, prev_alpha, prev_omega_r, prev_constant = layer(current_input)

        return current_input
