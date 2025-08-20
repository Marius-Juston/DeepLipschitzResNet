from typing import Sequence, Union, Optional, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Linear, ModuleList, ReLU


def matrix_inv_square_root(A: Tensor) -> Tensor:
    eigvals, eigvecs = torch.linalg.eigh(A)
    return (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T


def fast_symmetric_positive_definitive_matrix_inverse(A):
    L = torch.linalg.cholesky_ex(A).L
    return torch.cholesky_inverse(L)


class LipschitzInnerLayer(Linear):
    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
                 constant: float = 0.0,
                 device=None,
                 dtype=None):
        super(LipschitzInnerLayer, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.register_buffer(
            "identity", torch.eye(in_features, **factory_kwargs)
        )

        self.register_buffer(
            "identity_out", torch.eye(out_features, **factory_kwargs)
        )

        self.register_buffer(
            "constant", torch.tensor(constant, **factory_kwargs)
        )

        self.identity: Optional[Tensor]
        self.constant: Optional[Tensor]

    def compute_weight(self, scaling: Optional[Tensor] = None) -> Tensor:
        inner = self.identity + self.weight.T @ self.weight
        inner_inv_sqrt = matrix_inv_square_root(inner)

        weight = self.constant * self.weight @ inner_inv_sqrt

        if scaling is not None:
            weight = weight @ scaling

        return weight

    def compute_d(self):
        return fast_symmetric_positive_definitive_matrix_inverse(self.identity_out + self.weight @ self.weight.T)

    def compute_d_sqrt(self):
        return matrix_inv_square_root(self.identity_out + self.weight @ self.weight.T)

    def forward(self, input: Tensor, scaling: Optional[Tensor] = None) -> Tensor:
        weight = self.compute_weight(scaling)

        return F.linear(input, weight, self.bias)


class DeepLipschitzResNet(Module):
    A: LipschitzInnerLayer
    B: LipschitzInnerLayer
    layers: ModuleList

    def __init__(self, in_features: int, out_features: int, layers_widths: Sequence[int],
                 activation: Union[Module, List[Module]] = None, lipschitz_constant: float = 1.0,
                 device: Union[str, torch.device] = 'cuda', dtype=None) -> None:
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

        self.generate_layers()

    def reset_parameters(self) -> None:
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

    def forward(self, input: Tensor) -> Tensor:
        first = self.A(input)

        ds = [self.A.compute_d_sqrt()]

        second = input

        for i, layer in enumerate(self.layers):
            layer: LipschitzInnerLayer

            second = layer(second, ds[-1])

            ds.append(layer.compute_d_sqrt())

            if isinstance(self.activation, Sequence):
                second = self.activation[i](second)
            else:
                second = self.activation(second)

        second = self.B(second)

        return first + second

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, inner_widths=({','.join(map(str, self.layer_widths))})"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running on device", device)

    input_features = 10
    output_features = 12

    model = DeepLipschitzResNet(input_features, output_features, (32, 128, 128, 32), device=device)

    print(model)

    torch.manual_seed(0)

    batch = 64

    M = torch.randn(batch, input_features, device=device)

    output = model(M)

    print(output)


if __name__ == '__main__':
    main()
