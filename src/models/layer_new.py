from typing import Tuple

import torch
from torch import nn, Tensor


class FirstLipschitzLinearLayer(nn.Linear):
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
        super(FirstLipschitzLinearLayer, self).__init__(in_features, out_features, bias=bias, device=device,
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

        self.R = self.identity
        self.omega_ = None
        self.omega_sqrt_t_ = None

    def normalization_constant(self) -> Tensor:
        self.omega_ = self.sum_weight(transpose=True)
        temp = torch.linalg.cholesky(self.omega_, upper=True)
        self.omega_sqrt_t_ = temp.T

        return temp

    def normalize(self, W, R, left=False, upper=True):
        return torch.linalg.solve_triangular(R, W, upper=upper, left=left)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        self.R = self.normalization_constant()

        W = self.constant * self.normalize(self.weight, self.R, left=False)

        return torch.nn.functional.linear(input, W, self.bias), W

    def sum_weight(self, transpose: bool = False):
        if transpose:
            return self.identity + self.weight.T @ self.weight
        else:
            return self.identity_out + self.weight @ self.weight.T

    def omega(self):
        return self.omega_

    def omega_sqrt(self):
        return self.omega_sqrt_t_


class InnerWeightLipschitzLinearLayer(FirstLipschitzLinearLayer):
    def forward(self, input: Tensor, prev_r: Tensor) -> Tuple[Tensor, Tensor]:
        self.R = self.normalization_constant()

        W = self.normalize(self.weight, self.R, left=False)
        W = self.constant * self.normalize(W, prev_r, left=False, upper=False)

        self.omega()

        return torch.nn.functional.linear(input, W, self.bias), W

    def omega(self):
        self.omega_ = self.sum_weight()
        return self.omega_

    def omega_sqrt(self):
        self.omega_sqrt_t_ = torch.linalg.cholesky(self.omega_)
        return self.omega_sqrt_t_


class LastLipschitzLinearLayer(FirstLipschitzLinearLayer):
    def forward(self, input: Tensor, prev_r: Tensor) -> Tuple[Tensor, Tensor]:
        self.R = self.normalization_constant()

        W = self.normalize(self.weight, self.R, left=False)
        W = self.constant * self.normalize(W, prev_r, left=True)

        return torch.nn.functional.linear(input, W, self.bias), W



class LinearLipschitz(nn.Linear):
    identity: torch.Tensor
    identity_out: torch.Tensor
    constant_sq: torch.Tensor
    constant: torch.Tensor
    R: torch.Tensor

    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
                 constant_sq: float = 1.0,
                 activation:nn.Module = None,
                 device=None,
                 dtype=None):
        super(LinearLipschitz, self).__init__(in_features, out_features, bias=bias, device=device,
                                                        dtype=dtype)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.activation = activation
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

    def normalize(self, W, R, left=False, upper=True):
        return torch.linalg.solve_triangular(R, W, upper=upper, left=left)

    def forward(self, input: Tensor, prev_d_sqrt: Tensor) -> Tuple[Tensor, Tensor]:
        norm = self.sum_weight(transpose=True)
        R = torch.linalg.cholesky(norm, upper=True)

        W = self.normalize(self.weight, R)

        W = self.constant * self.normalize(W, prev_d_sqrt, upper=True, left=False)
        d_sqrt = self.constant * torch.linalg.cholesky(self.sum_weight(), upper=True)

        out = torch.nn.functional.linear(input, W, self.bias)

        if self.activation:
            out = self.activation(out)

        return out, d_sqrt

    def sum_weight(self, transpose: bool = False):
        if transpose:
            return self.identity + self.weight.T @ self.weight
        else:
            return self.identity_out + self.weight @ self.weight.T