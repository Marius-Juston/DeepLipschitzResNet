from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

import utils
from utils import fast_symmetric_positive_definitive_matrix_inverse


class LipschitzLinearLayer(Linear):
    __constants__ = ['identity', 'identity_out', 'constant', 'constant_sq']

    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = False,
                 constant: float = 1.0,
                 device=None,
                 dtype=None):
        super(LipschitzLinearLayer, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

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

        self.register_buffer(
            "constant_sq", torch.tensor(constant * constant, **factory_kwargs)
        )

        self.identity: Optional[Tensor]
        self.identity_out: Optional[Tensor]
        self.constant: Optional[Tensor]
        self.constant_sq: Optional[Tensor]

    def apply_inv_sqrt(self, weight, sqrt_matrix, left=False) -> torch.Tensor:
        sqrt_matrix = utils.symmetrize_(sqrt_matrix)

        # L, info = torch.linalg.cholesky_ex(sqrt_matrix, upper=True)
        # L = torch.linalg.cholesky_ex(sqrt_matrix, upper=True)
        # info = 0
        # FIXME There are problems where the matrix input to the system is no longer PD or PSD due to numerical issues
        L = torch.linalg.cholesky(sqrt_matrix, upper=True)

        return torch.linalg.solve_triangular(L, weight, upper=True, left=left)

    def compute_weight(self, scaling: Optional[Tensor] = None, left: bool = False) -> Tensor:
        weight = self.constant * self.apply_inv_sqrt(self.weight, self.sum_weight(transpose=True), left=False)

        if scaling is not None:
            weight = self.apply_inv_sqrt(weight, scaling, left=left)

        return weight

    def sum_weight(self, transpose=False):
        if transpose:
            return self.identity + self.weight.T @ self.weight
        else:
            return self.identity_out + self.weight @ self.weight.T

    def compute_d(self) -> Tensor:
        return self.constant_sq * fast_symmetric_positive_definitive_matrix_inverse(self.sum_weight())

    def compute_d_extra(self, transpose=False) -> Tensor:
        return self.sum_weight(transpose) * self.constant_sq

    def compute_d_inv(self, transpose=False) -> Tensor:
        return self.sum_weight(transpose) / self.constant_sq

    def forward(self, input: Tensor, scaling: Optional[Tensor] = None, left: bool = False) -> Tuple[Tensor, Tensor]:
        weight = self.compute_weight(scaling, left)

        return F.linear(input, weight, self.bias), weight
