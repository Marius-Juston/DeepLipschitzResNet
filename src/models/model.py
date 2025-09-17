import math
from typing import Union, List, Sequence

import torch
from torch import nn
from torch.nn import ModuleList, Module, ReLU, Parameter, init

from models.layers import FirstLipschitzLinearLayer, LastLipschitzLinearLayer, InnerWeightLipschitzLinearLayer, \
    LinearLipschitz
from utils import SpectralNorm


class DeepLipschitzLinearResNet(nn.Module):
    A: FirstLipschitzLinearLayer
    B: LastLipschitzLinearLayer
    layers: ModuleList

    def __init__(self, in_features: int, out_features: int, layers_widths: Sequence[int],
                 activation: Union[Module, List[Module]] = None, lipschitz_constant: float = 1.0, bias: bool = True,
                 num_iterations:int = 10,
                 device: Union[str, torch.device] = 'cuda', dtype=None) -> None:
        super().__init__()

        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_features > 0, "The input width needs to be larger than 0"
        assert out_features > 0, "The output width needs to be larger than 0"

        self.num_iterations = num_iterations
        self.lipschitz_constant = lipschitz_constant
        self.lipschitz_constant_sq = lipschitz_constant * lipschitz_constant

        self.out_features = out_features
        self.in_features = in_features
        self.layer_widths = layers_widths

        self.dtype = dtype
        self.device = device

        self.factory_kwargs = {"device": device, "dtype": dtype}

        if activation is None:
            self.activation = ReLU()
        else:
            self.activation = activation

        if bias:
            self.bias = Parameter(torch.empty(out_features, **self.factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.generate_layers()

        self.reset_bias()

        self.first = False

    def reset_bias(self) -> None:
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.A.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self) -> None:
        self.reset_bias()

        # Find a better initialization scheme for the current formulation of the system
        self.A.reset_parameters()
        self.B.reset_parameters()

        for layer in self.layers:
            layer.reset_parameters()

    def generate_layers(self) -> None:
        self.A = FirstLipschitzLinearLayer(in_features=self.in_features, out_features=self.out_features,
                                           constant_sq=self.lipschitz_constant_sq, **self.factory_kwargs)

        self.B = LastLipschitzLinearLayer(in_features=self.layer_widths[-1], out_features=self.out_features,
                                          **self.factory_kwargs)

        layer = []

        prev = self.in_features

        for i, out_features in enumerate(self.layer_widths):
            if i == 0:
                constant = 2.0 * self.lipschitz_constant_sq
            else:
                constant = 4.0

            layer.append(
                InnerWeightLipschitzLinearLayer(prev, out_features, constant_sq=constant, **self.factory_kwargs))
            prev = out_features

        self.layers = ModuleList(layer)

        self.spectral_norm = SpectralNorm(self.B.identity)

    def compute_b(self, x, sigma_lower, a_weight, prev_r):
        inner_inverse = torch.cholesky_inverse(prev_r, upper=False)

        if self.first:
            self.spectral_norm.initialize(inner_inverse)

        max_singular = self.spectral_norm(inner_inverse)

        left = a_weight @ sigma_lower

        scaling = self.B.identity_out + left @ left.T

        R_sigma = torch.linalg.cholesky(scaling, upper=True)

        second, b_weight = self.B(x, R_sigma)

        second_s = second * max_singular
        b_weight_s = b_weight * max_singular

        return second_s, b_weight_s


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, a_weight, d_sqrt, d_inv = self.A(x)

        gamma = self.A.identity
        sigma_lower = None

        current_input = x

        for layer in self.layers:
            T = gamma @ d_inv

            current_input, c_weight, d_sqrt, d_inv = layer(current_input, d_sqrt)

            if sigma_lower is None:
                sigma_lower = T
            else:
                # Instead of using custom CholeskyRankK update this is computationally much faster
                sigma_lower = torch.linalg.cholesky(sigma_lower @ sigma_lower.T + T @ T.T, upper=False)
                # sigma_lower = cholesky_rank_k(sigma_lower, T)

            gamma = T @ c_weight.T

            current_input = self.activation(current_input)

        second, b_weight = self.compute_b(current_input, sigma_lower, a_weight, self.layers[-1].omega_r)

        self.first = False

        return first + second + self.bias


class DeepLipschitzSequential(nn.Module):
    layers: ModuleList

    def __init__(self, in_features: int, out_features: int, layers_widths: Sequence[int],
                 activation: Union[Module, List[Module]] = None, lipschitz_constant: float = 1.0,
                 device: Union[str, torch.device] = 'cuda', dtype=None) -> None:
        super().__init__()

        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_features > 0, "The input width needs to be larger than 0"
        assert out_features > 0, "The output width needs to be larger than 0"

        self.lipschitz_constant = lipschitz_constant
        self.lipschitz_constant_sq = lipschitz_constant * lipschitz_constant

        self.out_features = out_features
        self.in_features = in_features
        self.layer_widths = layers_widths

        self.dtype = dtype
        self.device = device

        self.factory_kwargs = {"device": device, "dtype": dtype}

        if activation is None:
            self.activation = ReLU()
        else:
            self.activation = activation

        self.generate_layers()


    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def generate_layers(self) -> None:
        layer = []

        prev = self.in_features

        for i, out_features in enumerate(self.layer_widths):
            activation = self.activation
            constant = 2.0

            layer.append(
                LinearLipschitz(prev, out_features, constant_sq=constant, activation=activation, **self.factory_kwargs))
            prev = out_features

        layer.append(
            LinearLipschitz(prev, self.out_features, constant_sq=1.0, activation=None, **self.factory_kwargs))

        self.layers = ModuleList(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_input = x
        d_sqrt = self.lipschitz_constant * self.layers[0].identity

        for layer in self.layers:
            current_input, d_sqrt = layer(current_input, d_sqrt)

        return current_input