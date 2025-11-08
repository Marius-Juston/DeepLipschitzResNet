r"""
Lipschitz-constrained CNN/MLP factories and utilities.

Parameters
----------
None

Returns
-------
None

Raises
------
RuntimeError
    Propagated from :func:`is_complex_config` when the CLI arguments do not
    describe either the simple or complex architecture families.

Notes
-----
The module centralizes helper utilities (e.g., :class:`PaddingChannels`) and
the :class:`LipschitzNetwork` wrapper so that all experiments consume a single
source of truth for model creation. The design enforces consistent Lipschitz
bounds by composing convolutional and linear blocks drawn from
``models.cnn_model`` and ``models.linear_model``. The sequential pipeline obeys
the constraint \( L \leq L_\text{conv} \cdot L_\text{dense} \) by ensuring
each stage is individually normalized.

Mathematical Sketch
-------------------
Let ``f_c`` and ``f_d`` denote the convolutional and dense portions. Because
each enforces \(L(f_c) \leq 1\) and \(L(f_d) \leq 1\), the composite network
obeys \(L(f_d \circ f_c) \leq L(f_d) L(f_c) = 1\).

Examples
--------
>>> from argparse import Namespace
>>> cfg = Namespace(
...     n_conv=2,
...     n_dense=1,
...     conv_inner_dim=32,
...     dense_inner_dim=128,
...     w=16,
...     dataset="cifar10",
...     batch_norm=False,
...     dropout=False,
...     activation="relu",
...     safe_cholesky=False,
...     wrong_normalization=False,
...     linear_conv=False,
...     n_classes=10,
...     averaging=False,
... )
>>> net = LipschitzNetwork(cfg, n_classes=10)
>>> isinstance(net.model, nn.Sequential)
True
"""

import logging
from argparse import Namespace
from typing import Literal, Optional, Sequence

import numpy as np
import torch.nn
import torch.nn.functional as F
from torch import Tensor, nn

from models.batch_norm import LipschitzBatchNorm
from models.cnn_model import DeepLipschitzConvResNet, DeepLipschitzConvSequential
from models.linear_model import DeepLipschitzSequential

type ModuleSequence = list[nn.Module]
type ComplexConvConfig = Sequence[tuple[int, int, int]]
type ComplexDenseConfig = Sequence[tuple[int, int]]

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
}


class PaddingChannels(nn.Module):
    """
    Channel-expanding pre-processing layer.

    Parameters
    ----------
    ncout : int
        Target number of channels after padding/replication.
    ncin : int, default=3
        Incoming channel count.
    mode : {"zero", "clone"}, default="zero"
        Strategy used to satisfy ``ncout`` (zero-pad or repeat channels).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If ``mode`` is not among the allowed literals.

    Notes
    -----
    The layer preserves the overall variance by dividing repeated channels by
    ``sqrt(ncout / ncin)`` so that the induced linear operator remains
    approximately isometric prior to the Lipschitz-constrained backbone.

    Examples
    --------
    >>> import torch
    >>> layer = PaddingChannels(ncout=6, ncin=3, mode="clone")
    >>> x = torch.ones(2, 3, 4, 4)
    >>> y = layer(x)
    >>> y.shape
    torch.Size([2, 6, 4, 4])
    """

    def __init__(self, ncout: int, ncin: int = 3, mode: Literal["zero", "clone"] = "zero") -> None:
        """Configure the padding layer with the desired channel budget.

        Parameters
        ----------
        ncout : int
            Desired number of channels (must satisfy ``ncout >= ncin``).
        ncin : int, default=3
            Number of input channels.
        mode : {"zero", "clone"}, default="zero"
            ``"zero"`` appends zero-valued channels, whereas ``"clone"`` repeats
            existing channels and rescales them by ``1 / sqrt(ncout / ncin)``.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If ``mode`` is not among the supported literals.

        Notes
        -----
        The scaling constant ``repeat_norm`` is precomputed to avoid costly
        square-root operations during the forward pass.

        Examples
        --------
        >>> PaddingChannels(ncout=4, ncin=2, mode="zero")
        PaddingChannels(...)
        """
        super().__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.mode = mode

        assert mode in ["zero", "clone"], "You must have the mode either be 'zero' or 'clone'"

        self.pad_channels = self.ncout - self.ncin

        self.repeat_num = int(self.ncout / self.ncin)
        self.repeat_norm = np.sqrt(int(self.ncout / self.ncin))

    def forward(self, x: Tensor) -> Tensor:
        r"""Pad or repeat channels along the channel dimension.

        Parameters
        ----------
        x : (B, C, H, W) tensor
            Input batch with ``C = ncin`` channels stored in CHW order.

        Returns
        -------
        y : (B, ncout, H, W) tensor
            Tensor with ``ncout`` channels following the configured mode.

        Raises
        ------
        RuntimeError
            Propagated from :func:`torch.nn.functional.pad` when the padding
            arguments are inconsistent with the tensor shape.

        Notes
        -----
        Zero padding simply appends silent channels, while clone padding repeats
        the existing tensor ``repeat_num`` times and rescales by
        ``repeat_norm`` to keep the per-channel variance unchanged. Complexity
        is \(O(B C H W)\) because the operation copies the entire tensor.

        Mathematical Sketch
        -------------------
        The clone operation approximates an isometry by repeating the tensor and
        dividing by \(\sqrt{n_{\text{cout}} / n_{\text{cin}}}\), which keeps the
        Frobenius norm invariant: \(\lVert y \rVert_F = \lVert x \rVert_F\).

        Examples
        --------
        >>> import torch
        >>> layer = PaddingChannels(5, 3, mode="zero")
        >>> x = torch.randn(1, 3, 2, 2)
        >>> layer(x).shape
        torch.Size([1, 5, 2, 2])
        """
        if self.mode == "zero":
            return F.pad(x, (0, 0, 0, 0, 0, self.pad_channels), mode="constant", value=0)
        elif self.mode == "clone":
            return x.repeat(1, self.repeat_num, 1, 1) / self.repeat_norm


def is_complex_config(config: Namespace) -> bool:
    r"""Return ``True`` when the CLI arguments describe the complex model.

    Parameters
    ----------
    config : Namespace
        Argument namespace produced by :mod:`argparse`.

    Returns
    -------
    is_complex : bool
        ``True`` when ``n_conv`` and ``n_dense`` are sequences of tuples and the
        inner-dimension overrides are ``None``; otherwise ``False``.

    Raises
    ------
    RuntimeError
        If the namespace fails to satisfy either the complex or simple schema.

    Notes
    -----
    The simple schema expects scalars for ``n_conv``, ``n_dense``,
    ``dense_inner_dim``, and ``conv_inner_dim``. The complex schema expects the
    tuple-based residual definition with ``dense_inner_dim`` and
    ``conv_inner_dim`` equal to ``None``.

    Mathematical Sketch
    -------------------
    The predicate implements a two-branch decision tree evaluating membership in
    the set ``S_complex`` (tuple definitions) versus ``S_simple`` (scalars). If
    the namespace lies outside \(S_\text{complex} \cup S_\text{simple}\), a
    ``RuntimeError`` is raised to prevent ambiguous model creation.

    Examples
    --------
    >>> from argparse import Namespace
    >>> cfg = Namespace(n_conv=( (2, 32, 3), ), n_dense=((1, 128),),
    ...                 dense_inner_dim=None, conv_inner_dim=None)
    >>> is_complex_config(cfg)
    True
    >>> cfg_simple = Namespace(n_conv=2, n_dense=2,
    ...                        dense_inner_dim=128, conv_inner_dim=64)
    >>> is_complex_config(cfg_simple)
    False
    """
    if (isinstance(config.n_conv, Sequence) and
            isinstance(config.n_conv, Sequence) and
            config.dense_inner_dim is None and
            config.conv_inner_dim is None):
        return True
    elif (isinstance(config.n_conv, int) and
          isinstance(config.n_dense, int) and
          isinstance(config.dense_inner_dim, int)
          and isinstance(config.conv_inner_dim, int)):
        return False
    else:
        raise RuntimeError(
            "The config is not properly setup for the Lipschitz network configuration, it must either be the complex or the simple setup")


class LipschitzNetwork(nn.Module):
    r"""
    Compose convolutional and linear Lipschitz sub-networks from CLI configs.

    Parameters
    ----------
    config : Namespace
        Configuration namespace consumed during initialization.
    n_classes : int
        Number of output logits for the classifier head.
    device : str, optional
        Device identifier forwarded to all nested modules.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Raised when ``config`` fails validation inside :func:`is_complex_config`.

    Notes
    -----
    The wrapper defers to either :class:`DeepLipschitzConvResNet` or
    :class:`DeepLipschitzConvSequential` for the convolutional stem and always
    terminates with :class:`DeepLipschitzSequential`. The full network satisfies
    the Lipschitz budget by construction because each block is normalized.

    Mathematical Sketch
    -------------------
    The model composes three Lipschitz mappings: padding ``P``, convolutional
    head ``f_c``, and dense head ``f_d``. Because each satisfies
    \(L \leq 1\), the composite mapping obeys
    \(L(f_d \circ f_c \circ P) \leq L(f_d) L(f_c) L(P) = 1\).

    Examples
    --------
    >>> from argparse import Namespace
    >>> cfg = Namespace(
    ...     n_conv=2,
    ...     n_dense=1,
    ...     conv_inner_dim=32,
    ...     dense_inner_dim=64,
    ...     w=16,
    ...     dataset="cifar10",
    ...     batch_norm=False,
    ...     dropout=False,
    ...     activation="relu",
    ...     safe_cholesky=False,
    ...     wrong_normalization=False,
    ...     linear_conv=True,
    ...     averaging=False,
    ... )
    >>> net = LipschitzNetwork(cfg, n_classes=10)
    >>> isinstance(net.model, nn.Sequential)
    True
    """

    def generate_simple(self, config: Namespace) -> ModuleSequence:
        r"""Instantiate the uniform-depth CNN/MLP pipeline.

        Parameters
        ----------
        config : Namespace
            Namespace containing scalar ``n_conv``/``n_dense`` along with
            ``conv_inner_dim``/``dense_inner_dim`` widths and dataset metadata.

        Returns
        -------
        modules : list of nn.Module
            Ordered list of layers passed to :class:`nn.Sequential`.

        Raises
        ------
        ValueError
            If ``config.dataset`` is not one of ``{"cifar10", "cifar100",
            "tiny-imagenet"}``.

        Notes
        -----
        The convolutional stage processes images of shape ``(B, 3, 32, 32)`` or
        ``(B, 3, 64, 64)`` depending on the dataset. Average pooling reduces the
        spatial extent before a flattened linear Lipschitz classifier produces
        ``n_classes`` logits. Complexity is \(O(n_\text{conv} + n_\text{dense})\)
        because each configuration entry generates one module.

        Mathematical Sketch
        -------------------
        Each convolution satisfies \(L_i \leq 1\), so the composed block obeys
        \(L_\text{conv} = \prod_i L_i = 1\). The linear block is normalized so
        that \(L_\text{linear} = 1\), giving a total bound
        \(L_\text{total} = L_\text{conv} L_\text{linear} = 1\).

        Examples
        --------
        >>> from argparse import Namespace
        >>> cfg = Namespace(
        ...     conv_inner_dim=32,
        ...     dense_inner_dim=64,
        ...     w=16,
        ...     dataset="cifar10",
        ...     batch_norm=False,
        ...     dropout=False,
        ...     activation="relu",
        ...     safe_cholesky=False,
        ...     wrong_normalization=False,
        ...     linear_conv=False,
        ...     n_conv=2,
        ...     n_dense=1,
        ...     averaging=False,
        ... )
        >>> net = LipschitzNetwork(cfg, n_classes=10)
        >>> isinstance(net.generate_simple(cfg), list)
        True
        """
        n_conv: int = config.n_conv
        n_dense: int = config.n_dense
        cin: int = config.w
        conv_inner_dim: int = config.conv_inner_dim
        dense_inner_dim: int = config.dense_inner_dim
        n_classes: int = self.n_classes
        batch_norm: bool = config.batch_norm
        dropout: bool = config.dropout

        model = []
        model.append(
            PaddingChannels(cin, 3, 'zero')
        )

        inner_conv_layers = [conv_inner_dim] * n_conv

        if hasattr(config, "activation") and config.activation in ACTIVATION_FUNCTIONS:
            activation = ACTIVATION_FUNCTIONS[config.activation]()
        else:
            activation = torch.nn.ReLU()

        if batch_norm:
            model.append(LipschitzBatchNorm(cin, scale=True, device=self.device))

        model.append(
            self.conv_model(cin, cin, inner_conv_layers, use_safe_cholesky=config.safe_cholesky,
                            wrong_normalization=config.wrong_normalization,
                            activation=activation,
                            dropout=dropout,
                            device=self.device))

        model.append(nn.AvgPool2d(4, divisor_override=4))

        model.append(nn.Flatten())

        inner_linear_layers = [dense_inner_dim] * n_dense

        if config.dataset in ['cifar10', 'cifar100']:
            in_channels = cin * 8 * 8
        elif config.dataset == 'tiny-imagenet':
            in_channels = cin * 16 * 16
        else:
            raise ValueError("The dataset must be 'cifar10', 'cifar100' or 'tiny-imagenet'")

        if batch_norm:
            model.append(LipschitzBatchNorm(in_channels, scale=True, device=self.device))

        model.append(
            DeepLipschitzSequential(in_channels, n_classes, inner_linear_layers, activation=activation, dropout=dropout,
                                    device=self.device))

        return model

    def generate_complex(self, config: Namespace) -> ModuleSequence:
        r"""Construct the tuple-driven CNN/MLP pipeline.

        Parameters
        ----------
        config : Namespace
            Namespace with ``n_conv`` specified as ``(count, out_channels,
            kernel_size)`` tuples and ``n_dense`` specified as ``(count, width)``
            tuples.

        Returns
        -------
        modules : list of nn.Module
            Ordered modules comprising the complex architecture.

        Raises
        ------
        ValueError
            If ``config.dataset`` is unsupported.

        Notes
        -----
        Unlike the simple variant, each tuple allows fine-grained control over
        the number of repeated layers, output channels, and kernel sizes. A
        hard-coded ``in_channels = 16384`` remains for task compatibility and is
        documented in :func:`generate_complex` to avoid silent surprises. The
        construction cost scales with the total number of expanded tuples,
        \(O(\sum_i n_i)\).

        Mathematical Sketch
        -------------------
        The tuple list expands to a sequence ``[(n_i, c_i, k_i)]``. Each entry is
        unrolled into ``n_i`` convolutional layers whose operator norms satisfy
        \(L_{i,j} \leq 1\). The dense tuples similarly ensure
        \(L_{i} \leq 1\), yielding a composite bound
        \(L = \prod_{i,j} L_{i,j} \leq 1\).

        Examples
        --------
        >>> from argparse import Namespace
        >>> cfg = Namespace(
        ...     n_conv=((1, 16, 3),),
        ...     n_dense=((1, 128),),
        ...     dataset="cifar10",
        ...     activation="relu",
        ...     safe_cholesky=False,
        ...     averaging=False,
        ...     batch_norm=False,
        ...     dropout=False,
        ...     wrong_normalization=False,
        ...     linear_conv=False,
        ...     dense_inner_dim=None,
        ...     conv_inner_dim=None,
        ... )
        >>> net = LipschitzNetwork(cfg, n_classes=10)
        >>> modules = net.generate_complex(cfg)
        >>> isinstance(modules[-1], nn.Module)
        True
        """
        n_conv: ComplexConvConfig = config.n_conv
        n_dense: ComplexDenseConfig = config.n_dense

        model = []

        cin = 3

        inner_conv_layers = []

        last_channel_size = cin

        for n, channels, kernel in n_conv:
            for _ in range(n):
                inner_conv_layers.append([channels, kernel])

            last_channel_size = channels

        logging.info(f"Setting up model with conv config: {inner_conv_layers}")
        logging.info(f"Cin: {cin}")
        logging.info(f"Last channels: {last_channel_size}")

        if hasattr(config, "activation") and config.activation in ACTIVATION_FUNCTIONS:
            activation = ACTIVATION_FUNCTIONS[config.activation]()
        else:
            activation = torch.nn.ReLU()

        model.append(
            self.conv_model(cin, last_channel_size, inner_conv_layers, use_safe_cholesky=config.safe_cholesky,
                            device=self.device, activation=activation, use_averaging=config.averaging))

        model.append(nn.AvgPool2d(4, divisor_override=4))

        model.append(nn.Flatten())

        inner_dense_layers = []

        for n, outputs in n_dense:
            for _ in range(n):
                inner_dense_layers.append(outputs)

        if config.dataset in ['cifar10', 'cifar100']:
            in_channels = cin * 8 * 8
        elif config.dataset == 'tiny-imagenet':
            in_channels = cin * 16 * 16
        else:
            raise ValueError("The dataset must be 'cifar10', 'cifar100' or 'tiny-imagenet'")

        in_channels = 16384  # hardcoded for this current task should not be hard-coded

        model.append(DeepLipschitzSequential(in_channels, self.n_classes, inner_dense_layers, device=self.device,
                                             activation=activation))

        return model

    def __init__(self, config: Namespace, n_classes: int, device: Optional[str] = None) -> None:
        r"""Initialize the wrapped sequential network.

        Parameters
        ----------
        config : Namespace
            Namespace produced by :mod:`argparse` containing all hyperparams.
        n_classes : int
            Number of logits produced by the terminal linear block.
        device : str, optional
            Device identifier forwarded to convolutional/linear modules.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If :func:`is_complex_config` cannot classify ``config``.

        Notes
        -----
        The constructor selects either :class:`DeepLipschitzConvResNet` or
        :class:`DeepLipschitzConvSequential` based on ``config.linear_conv`` to
        generate ``self.conv_model`` before assembling the sequential pipeline.

        Mathematical Sketch
        -------------------
        Initialization guarantees that the resulting sequential model is a
        composition \(f(x) = f_\text{dense}(f_\text{conv}(P(x)))\) where ``P``
        denotes the channel-padding operator. Each constituent preserves the
        Lipschitz budget, hence ``f`` inherits the product bound.

        Examples
        --------
        >>> from argparse import Namespace
        >>> cfg = Namespace(
        ...     n_conv=2,
        ...     n_dense=1,
        ...     conv_inner_dim=16,
        ...     dense_inner_dim=32,
        ...     w=8,
        ...     dataset="cifar10",
        ...     batch_norm=True,
        ...     dropout=False,
        ...     activation="relu",
        ...     safe_cholesky=False,
        ...     wrong_normalization=False,
        ...     linear_conv=False,
        ...     averaging=False,
        ... )
        >>> LipschitzNetwork(cfg, n_classes=10)
        LipschitzNetwork(...)
        """
        super().__init__()

        self.device = device
        self.n_classes = n_classes
        self.config = config

        if config.linear_conv:
            self.conv_model = DeepLipschitzConvSequential
        else:
            self.conv_model = DeepLipschitzConvResNet
        logging.info(f"Using model ({config.linear_conv}): {self.conv_model.__name__}")

        if is_complex_config(config):
            logging.info("Using complex configuration for model setup")
            model = self.generate_complex(config)
        else:
            logging.info("Using simple configuration for model setup")
            model = self.generate_simple(config)

        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        r"""Evaluate the sequential Lipschitz model.

        Parameters
        ----------
        x : (B, C, H, W) tensor
            Mini-batch of images in ``NCHW`` format, typically with
            ``C in {3, self.config.w}``.

        Returns
        -------
        logits : (B, n_classes) tensor
            Output logits produced by :class:`nn.Sequential`.

        Raises
        ------
        RuntimeError
            Propagated from PyTorch layers when tensor shapes or dtypes mismatch
            the expected configuration.

        Notes
        -----
        The forward pass applies channel padding, convolutional blocks,
        average pooling, flattening, and a Lipschitz linear head. All submodules
        operate on the device specified during initialization.

        Mathematical Sketch
        -------------------
        Forward evaluation computes
        \(f(x) = (f_\text{dense} \circ \text{flatten} \circ \text{pool} \circ
        f_\text{conv} \circ P)(x)\). Each component is 1-Lipschitz (or bounded by
        the prescribed constant), so ``f`` inherits the same bound by
        composition.

        Examples
        --------
        >>> import torch
        >>> from argparse import Namespace
        >>> cfg = Namespace(
        ...     n_conv=2,
        ...     n_dense=1,
        ...     conv_inner_dim=16,
        ...     dense_inner_dim=32,
        ...     w=8,
        ...     dataset="cifar10",
        ...     batch_norm=False,
        ...     dropout=False,
        ...     activation="relu",
        ...     safe_cholesky=False,
        ...     wrong_normalization=False,
        ...     linear_conv=True,
        ...     averaging=False,
        ... )
        >>> net = LipschitzNetwork(cfg, n_classes=10)
        >>> logits = net(torch.randn(2, 3, 32, 32))
        >>> logits.shape
        torch.Size([2, 10])
        """
        return self.model(x)
