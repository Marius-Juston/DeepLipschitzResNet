from typing import Tuple

import torch
from torch import nn, Tensor


def normalize(W, R, left=False, upper=True):
    return torch.linalg.solve_triangular(R, W, upper=upper, left=left)


class AbstractLipschitzLinearLayer(nn.Linear):
    identity: torch.Tensor
    identity_out: torch.Tensor
    constant_sq: torch.Tensor
    constant: torch.Tensor
    R: torch.Tensor

    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
                 constant_sq: float = 1.0,
                 device=None,
                 dtype=None):
        super(AbstractLipschitzLinearLayer, self).__init__(in_features, out_features, bias=bias, device=device,
                                                           dtype=dtype)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.register_buffer(
            "identity", torch.eye(in_features, **factory_kwargs)
        )

        self.register_buffer(
            "identity_out", torch.eye(out_features, **factory_kwargs)
        )

        self.register_buffer(
            "constant_sq", torch.tensor(constant_sq, **factory_kwargs)
        )

        self.register_buffer(
            "constant", self.constant_sq.sqrt()
        )

    def scale_w(self):
        omega = self.sum_weight(transpose=True)
        R = torch.linalg.cholesky(omega, upper=True)

        W = normalize(self.weight, R, left=False)

        return omega, W

    def sum_weight(self, transpose: bool = False):
        if transpose:
            return self.identity + self.weight.T @ self.weight
        else:
            return self.identity_out + self.weight @ self.weight.T


class FirstLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        omega, W = self.scale_w()

        W = W * self.constant

        d_inv = omega / self.constant_sq
        d_sqrt = self.constant * omega

        return torch.nn.functional.linear(input, W, self.bias), W, d_sqrt, d_inv


class InnerWeightLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    def forward(self, input: Tensor, prev_d_sqrt: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        _, W = self.scale_w()

        W = self.constant * normalize(W, prev_d_sqrt, upper=True, left=False)

        omega = self.sum_weight()

        d_inv = omega / self.constant_sq

        d_sqrt = torch.linalg.cholesky(omega, upper=True)

        self.omega_r = d_sqrt

        d_sqrt = self.constant * d_sqrt

        return torch.nn.functional.linear(input, W, self.bias), W, d_sqrt, d_inv


class LastLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    def forward(self, input: Tensor, prev_r: Tensor) -> Tuple[Tensor, Tensor]:
        _, W = self.scale_w()

        W = self.constant * normalize(W, prev_r, left=True)

        return torch.nn.functional.linear(input, W, self.bias), W


class LinearLipschitz(AbstractLipschitzLinearLayer):
    def __init__(self, in_features: int, out_features: int, activation=None, bias: bool = True,
                 constant_sq: float = 1.0,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias, constant_sq, device, dtype)
        self.activation = activation

    def forward(self, input: Tensor, prev_d_sqrt: Tensor) -> Tuple[Tensor, Tensor]:
        _, W = self.scale_w()

        W = self.constant * normalize(W, prev_d_sqrt, upper=True, left=False)
        d_sqrt = self.constant * torch.linalg.cholesky(self.sum_weight(), upper=True)

        out = torch.nn.functional.linear(input, W, self.bias)

        if self.activation:
            out = self.activation(out)

        return out, d_sqrt
