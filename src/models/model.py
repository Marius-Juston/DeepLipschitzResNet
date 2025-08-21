import math
from typing import Sequence, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor, device
from torch.nn import Module, ModuleList, ReLU, Parameter, init

from models.layers import LipschitzInnerLayer
from utils import matrix_inv_square_root


class DeepLipschitzResNet(Module):
    A: LipschitzInnerLayer
    B: LipschitzInnerLayer
    layers: ModuleList

    def __init__(self, in_features: int, out_features: int, layers_widths: Sequence[int],
                 activation: Union[Module, List[Module]] = None, lipschitz_constant: float = 1.0, bias: bool = True,
                 device: Union[str, device] = 'cuda', dtype=None) -> None:
        super().__init__()

        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_features > 0, "The input width needs to be larger than 0"
        assert out_features > 0, "The output width needs to be larger than 0"

        self.lipschitz_constant = lipschitz_constant
        self.out_features = out_features
        self.in_features = in_features
        self.layer_widths = layers_widths

        self.dtype = dtype
        self.device = device

        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.sqrt_2 = 1.4142135623730950  # the value of sqrt(2) to 16 decimal places

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

    def reset_bias(self):
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
        self.A = LipschitzInnerLayer(in_features=self.in_features, out_features=self.out_features,
                                     constant=self.lipschitz_constant, **self.factory_kwargs)

        self.B = LipschitzInnerLayer(in_features=self.layer_widths[-1], out_features=self.out_features,
                                     **self.factory_kwargs)

        layer = []

        prev = self.in_features

        for out_features in self.layer_widths:
            layer.append(LipschitzInnerLayer(prev, out_features, constant=self.sqrt_2, **self.factory_kwargs))
            prev = out_features

        self.layers = ModuleList(layer)

    def check_weights(self):
        a_weight = self.A.compute_weight()
        assert torch.linalg.matrix_norm(a_weight.T @ a_weight, ord=2) < self.lipschitz_constant

    def forward(self, input: Tensor) -> Tensor:
        first, a_weight = self.A(input)

        second = input

        ds = [self.A.compute_d1_sqrt()]

        total_sum = self.A.compute_d1_inv()

        current_product = total_sum

        for i, layer in enumerate(self.layers):
            layer: LipschitzInnerLayer

            second, cn = layer(second, ds[-1])

            ds.append(layer.compute_d_sqrt())

            di_inv = layer.compute_d_inv()

            current_product = current_product @ cn.T

            total_sum = total_sum + (current_product @ di_inv @ current_product.T)

            current_product = current_product @ di_inv

            if isinstance(self.activation, Sequence):
                second = self.activation[i](second)
            else:
                second = self.activation(second)

        smallest_singular_value_cn = torch.linalg.matrix_norm(self.layers[-1].weight, ord=-2)

        B_factor = torch.sqrt(2.0 / (1 + (smallest_singular_value_cn * smallest_singular_value_cn)))

        a_sigma_a_t = B_factor * matrix_inv_square_root(self.B.identity_out + a_weight @ total_sum @ a_weight.T)

        second, b_weight = self.B(second)

        return F.linear(second, a_sigma_a_t, first + self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, inner_widths=({','.join(map(str, self.layer_widths))})"
