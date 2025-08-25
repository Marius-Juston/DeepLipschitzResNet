from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from utils import fast_symmetric_positive_definitive_matrix_inverse


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

        self.current_weight_version = None
        self.current_weight_svd_decomposition = None
        self._identity_plus_weight_weight_transpose_sqrt = None
        self._identity_plus_weight_transpose_weight_sqrt = None

    @property
    def identity_plus_weight_weight_transpose_sqrt(self):
        self.get_cache()
        return self._identity_plus_weight_weight_transpose_sqrt

    @property
    def identity_plus_weight_transpose_weight_sqrt(self):
        self.get_cache()
        return self._identity_plus_weight_transpose_weight_sqrt

    def get_cache(self):
        if self.weight._version != self.current_weight_version:
            self.current_weight_svd_decomposition = torch.linalg.svd(self.weight, full_matrices=False)

            self.current_weight_version = self.weight._version

            U, S, Vh = self.current_weight_svd_decomposition

            inverse_sqrt = torch.diag((1 + S).sqrt())

            self._identity_plus_weight_transpose_weight_sqrt = Vh.T @ inverse_sqrt @ Vh
            self._identity_plus_weight_weight_transpose_sqrt = U @ inverse_sqrt @ U.T

    @property
    def svd(self):
        self.get_cache()
        return self.current_weight_svd_decomposition

    def compute_weight(self, scaling: Optional[Tensor] = None) -> Tensor:
        U, S, Vh = self.svd

        weight = U @ torch.diag(S * (1 + S).rsqrt()) @ Vh

        if scaling is not None:
            weight = weight @ scaling

        return weight

    @property
    def norm_2(self):
        self.get_cache()
        _, S, _ = self.svd

        return S.max()

    @property
    def norm_neg_2(self):
        self.get_cache()
        _, S, _ = self.svd

        return S.min()

    def compute_d(self) -> Tensor:
        return self.constant_sq * fast_symmetric_positive_definitive_matrix_inverse(
            self.identity_out + self.weight @ self.weight.T)

    def compute_d1(self) -> Tensor:
        return self.constant_sq * fast_symmetric_positive_definitive_matrix_inverse(
            self.identity + self.weight.T @ self.weight)

    def compute_d1_inv(self) -> Tensor:
        return (self.identity + self.weight.T @ self.weight) / self.constant_sq

    def compute_d1_sqrt(self) -> Tensor:
        return self.constant * self.identity_plus_weight_transpose_weight_sqrt

    def compute_d_sqrt(self) -> Tensor:
        return self.constant * self.identity_plus_weight_weight_transpose_sqrt

    def compute_d_inv(self) -> Tensor:
        return (self.identity_out + self.weight @ self.weight.T) / self.constant_sq

    def forward(self, input: Tensor, scaling: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        weight = self.compute_weight(scaling)

        return F.linear(input, weight, self.bias), weight
