from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from utils import matrix_inv_square_root, fast_symmetric_positive_definitive_matrix_inverse


class LipschitzLinearLayer(Linear):
    __constants__ = ['identity', 'identity_out', 'constant', 'constant_sq']

    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
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

    def apply_inv_sqrt(self) -> torch.Tensor:
        S = self.sum_weight(transpose=True)

        L = torch.linalg.cholesky(S, upper=False)

        Y = torch.linalg.solve_triangular(L, self.weight.T, upper=False, left=True)

        return Y.T

    def compute_weight(self, scaling: Optional[Tensor] = None) -> Tensor:
        weight = self.constant * self.apply_inv_sqrt()

        # assert torch.linalg.matrix_norm(weight, ord=2) <= self.constant

        if scaling is not None:
            # assert torch.linalg.matrix_norm(scaling, ord=2) <= self.constant + 1e-3

            weight = weight @ scaling

            # assert torch.linalg.matrix_norm(weight @ utils.fast_symmetric_positive_definitive_matrix_inverse(scaling).T, ord=2) <= self.constant_sq + 1e-3

        return weight

    def sum_weight(self, transpose=False):
        if transpose:
            return self.identity + self.weight.T @ self.weight
        else:
            return self.identity_out + self.weight @ self.weight.T

    def compute_d(self) -> Tensor:
        return self.constant_sq * fast_symmetric_positive_definitive_matrix_inverse(self.sum_weight())

    def compute_d_sqrt(self, transpose=False) -> Tensor:
        # FIXME the problem for the triangular square root arrises in this computation
        return self.constant * matrix_inv_square_root(self.sum_weight(transpose))

    def compute_d_inv(self, transpose=False) -> Tensor:
        return self.sum_weight(transpose) / self.constant_sq

    def forward(self, input: Tensor, scaling: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        weight = self.compute_weight(scaling)

        return F.linear(input, weight, self.bias), weight
