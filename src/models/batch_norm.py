from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d


class LipschitzBatchNorm(BatchNorm1d):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: Optional[float] = 0.1,
            track_running_stats: bool = True,
            scale=False,
            lipschitz_constant: float = 1.0,
            device=None,
            dtype=None,
    ) -> None:
        self.scale = scale
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, False, track_running_stats, **factory_kwargs
        )

        self.register_buffer("inv_lipschitz_constant",
                             torch.tensor(lipschitz_constant, **factory_kwargs).sqrt().reciprocal())

        if not self.scale:
            self.running_var = None

    def forward(self, input: Tensor) -> Tensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var.clamp(
                min=self.inv_lipschitz_constant) if not self.training or self.track_running_stats or not self.scale else None,
            self.weight,  # None
            self.bias,  # None
            bn_training,
            exponential_average_factor,
            self.eps,
        )
