r"""
Batch-normalization layers with explicit Lipschitz control.

Parameters
----------
None

Returns
-------
None

Raises
------
RuntimeError
    Raised indirectly if PyTorch detects invalid tensor shapes when invoking
    the batch-normalization kernels.

Notes
-----
Wrapping :class:`torch.nn.BatchNorm1d` allows experiments to either disable
running statistics (``scale=False``) or clamp the variance to a target Lipschitz
constant. The layer can therefore be inserted between Lipschitz-aware blocks
without loosening the overall bound.

Mathematical Sketch
-------------------
The Lipschitz bound is maintained by scaling the running variance so that the
normalized activations satisfy \(\|y\|_2 \leq L \|x\|_2\). When ``scale`` is
``False`` the transformation reduces to centering plus optional learned gain.

Examples
--------
>>> import torch
>>> from models.batch_norm import LipschitzBatchNorm
>>> layer = LipschitzBatchNorm(16, scale=True, lipschitz_constant=0.5)
>>> out = layer(torch.randn(4, 16))
>>> out.shape
torch.Size([4, 16])
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d


class LipschitzBatchNorm(BatchNorm1d):
    r"""
    1-D batch normalization that respects a user-defined Lipschitz limit.

    Parameters
    ----------
    num_features : int
        Number of channels/features in the incoming activation vector.
    eps : float, default=1e-5
        Epsilon added to the denominator for numerical stability.
    momentum : float, optional
        Exponential moving average factor for running statistics. When ``None``,
        cumulative moving averages are used.
    track_running_stats : bool, default=True
        Whether to maintain moving averages of batch statistics in eval mode.
    scale : bool, default=False
        When ``False`` no variance scaling is applied in eval mode.
    lipschitz_constant : float, default=1.0
        Target upper bound \(L\) on the induced \(\ell_2\) operator norm.
    device, dtype : Any, optional
        Factory kwargs passed to the underlying tensors.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Propagated from :class:`torch.nn.BatchNorm1d` when input tensors do not
        match ``num_features`` or have invalid shapes.

    Notes
    -----
    The layer rescales (or removes) the running variance buffer so that evaluation
    time statistics cannot enlarge the Lipschitz constant. When ``scale`` is
    ``False`` the layer behaves like a mean-only normalization in eval mode.

    Mathematical Sketch
    -------------------
    During evaluation with scaling, the running variance ``v`` is clamped so that
    the affine transform ``(x - μ) / sqrt(v + eps)`` has gain bounded by
    \(L = \text{lipschitz_constant}\). Setting ``scale=False`` effectively
    enforces \(L = 1\) by omitting the variance term.

    Examples
    --------
    >>> import torch
    >>> layer = LipschitzBatchNorm(16, scale=True, lipschitz_constant=0.5)
    >>> x = torch.randn(8, 16)
    >>> out = layer(x)
    >>> out.std(dim=0).max().item() >= 0
    True
    """

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: Optional[float] = 0.1,
            track_running_stats: bool = True,
            scale: bool = False,
            lipschitz_constant: float = 1.0,
            device: Optional[torch.device | str] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the Lipschitz-aware batch-normalization layer.

        Parameters
        ----------
        num_features : int
            Feature dimension of the incoming activations.
        eps : float, default=1e-5
            Small constant added for numerical stability.
        momentum : float, optional
            Momentum for exponential moving averages. ``None`` switches to a
            cumulative moving average.
        track_running_stats : bool, default=True
            Whether to maintain running statistics in eval mode.
        scale : bool, default=False
            When ``False`` the running variance buffer is disabled.
        lipschitz_constant : float, default=1.0
            Target Lipschitz upper bound for the affine transform.
        device : str or torch.device, optional
            Device hosting the parameters and buffers.
        dtype : torch.dtype, optional
            Desired floating-point precision.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated from :class:`torch.nn.BatchNorm1d` when the provided
            arguments are invalid (e.g., ``num_features <= 0``).

        Notes
        -----
        Registers ``inv_lipschitz_constant`` as a buffer holding
        ``1 / sqrt(lipschitz_constant)`` so that constraints can be enforced in
        ``forward`` without recomputation.

        Examples
        --------
        >>> LipschitzBatchNorm(8, scale=False, lipschitz_constant=1.0)
        LipschitzBatchNorm(...)
        """
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
        r"""Normalize activations while respecting the Lipschitz constraint.

        Parameters
        ----------
        input : (B, C) tensor
            Mini-batch with ``C = num_features`` to be normalized.

        Returns
        -------
        output : (B, C) tensor
            Normalized activations shaped like ``input``.

        Raises
        ------
        RuntimeError
            Propagated from :func:`torch.nn.functional.batch_norm` when tensor
            shapes or dtypes are invalid.

        Notes
        -----
        Mirrors :func:`torch.nn.functional.batch_norm` semantics while injecting a
        clamp on ``running_var`` to enforce the Lipschitz constraint. Complexity
        is \(O(B C)\).

        Mathematical Sketch
        -------------------
        The forward pass implements
        \(y = \text{BN}(x) = \gamma \odot (x - \mu) / \sqrt{\sigma^2 + \varepsilon}
        + \beta\) with \(\sigma^2\) clamped below ``inv_lipschitz_constant`` so
        that \(\|y\|_2 \leq L \|x\|_2\).

        Examples
        --------
        >>> import torch
        >>> layer = LipschitzBatchNorm(4, scale=True)
        >>> layer(torch.randn(2, 4)).shape
        torch.Size([2, 4])
        """
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
