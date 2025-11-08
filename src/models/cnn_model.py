r"""
High-level convolutional architectures built from Lipschitz-aware layers.

Parameters
----------
None

Returns
-------
None

Raises
------
RuntimeError
    Propagated from PyTorch when invalid tensor shapes reach the kernels.

Notes
-----
Exposes residual and sequential CNNs composed from the primitives in
:mod:`models.cnn_layers` to provide globally Lipschitz classifiers.

Mathematical Sketch
-------------------
Global Lipschitz bounds follow from per-layer constraints: ``L_total = Π_i L_i``
with each \(L_i \leq 1\).

Examples
--------
>>> import torch
>>> from models.cnn_model import DeepLipschitzConvResNet
>>> net = DeepLipschitzConvResNet(3, 10, layers_widths=[32, 64])
>>> net(torch.randn(2, 3, 32, 32)).shape
torch.Size([2, 10])
"""

from collections.abc import Sequence
from typing import Optional, Union, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import ModuleList, ReLU
from torch.nn.common_types import _size_2_t

from models.cnn_layers import InnerWeightLipschitzConvLayer, LastLipschitzConvLayer, FirstLipschitzConvLayer, \
    FirstSequentialLipschitzConvLayer
from utils import safe_cholesky

type ActivationSpec = Union[nn.Module, list[nn.Module]]


class DeepLipschitzConvResNet(nn.Module):
    """Residual convolutional backbone with explicit Lipschitz control.

    Parameters
    ----------
    in_channels, out_channels : int
        Input/output channel counts.
    layers_widths : sequence of int
        Hidden channel sizes for each residual block.
    kernel_size, stride, padding, dilation, groups : int or tuple
        Convolutional hyperparameters mirroring :class:`nn.Conv2d`.
    activation : Module or list, optional
        Activation(s) applied after each convolution.
    lipschitz_constant : float, default=1.0
        Target Lipschitz constant.
    enable_alpha : bool, default=True
        Whether to learn relaxation parameters.
    use_safe_cholesky : bool, default=False
        Swaps :func:`torch.linalg.cholesky` with :func:`utils.safe_cholesky`.
    wrong_normalization : bool, default=False
        Debug flag retaining historical behavior.
    device : str or torch.device, default="cuda"
        Device for module initialization.
    dtype : torch.dtype, optional
        Numeric precision.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If invalid widths or channel counts are supplied.

    Notes
    -----
    Composes :class:`FirstLipschitzConvLayer`, repeated
    :class:`InnerWeightLipschitzConvLayer` blocks, and a final
    :class:`LastLipschitzConvLayer`.

    Examples
    --------
    >>> import torch
    >>> net = DeepLipschitzConvResNet(3, 10, layers_widths=[32, 64])
    >>> net(torch.randn(1, 3, 32, 32)).shape
    torch.Size([1, 10])
    """
    A: FirstLipschitzConvLayer
    B: LastLipschitzConvLayer
    layers: Sequence[InnerWeightLipschitzConvLayer]

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            layers_widths: Sequence[int],
            kernel_size: _size_2_t = 3,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            activation: Optional[ActivationSpec] = None,
            lipschitz_constant: float = 1.0,
            enable_alpha: bool = True,
            use_safe_cholesky: bool = False,
            wrong_normalization: bool = False,
            device: Union[str, torch.device] = 'cuda',
            dtype: Optional[torch.dtype] = None
    ):
        """Initialize a Lipschitz-controlled residual CNN.

        Parameters
        ----------
        Same as the class docstring.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If provided widths or channel counts are invalid.

        Notes
        -----
        Stores ``factory_kwargs`` and selects the desired Cholesky routine.

        Examples
        --------
        >>> DeepLipschitzConvResNet(3, 10, layers_widths=[16])
        DeepLipschitzConvResNet(...)
        """
        super().__init__()
        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_channels > 0, "The input width needs to be larger than 0"
        assert out_channels > 0, "The output width needs to be larger than 0"

        self.cholesky_function = safe_cholesky if use_safe_cholesky else torch.linalg.cholesky
        self.enable_alpha = enable_alpha
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer_widths = list(layers_widths)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.lipschitz_constant = lipschitz_constant
        self.lipschitz_constant_sq = lipschitz_constant * lipschitz_constant

        self.activation = ReLU() if activation is None else activation

        self.generate_layers()

    def generate_layers(self) -> None:
        """Instantiate the first, inner, and last convolutional layers.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Populates ``self.A`` (first layer), ``self.layers`` (module list for the
        residual blocks), and ``self.B`` (final layer).

        Mathematical Sketch
        -------------------
        Inner layers use ``constant_sq = 2`` to maintain the residual budget.

        Examples
        --------
        >>> net = DeepLipschitzConvResNet(3, 10, layers_widths=[16])
        >>> len(net.layers)
        1
        """
        self.A = FirstLipschitzConvLayer(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
            enable_alpha=self.enable_alpha,
            constant_sq=self.lipschitz_constant_sq,
            **self.factory_kwargs
        )

        layers = []

        prev = self.in_channels

        for i, out_features in enumerate(self.layer_widths):
            layers.append(
                InnerWeightLipschitzConvLayer(prev, out_features,
                                              enable_alpha=self.enable_alpha, constant_sq=2.0,
                                              kernel_size=self.kernel_size,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation=self.dilation,
                                              activation=self.activation,
                                              groups=self.groups, **self.factory_kwargs))
            prev = out_features

        self.layers = ModuleList(layers)

        self.B = LastLipschitzConvLayer(
            in_channels=self.layer_widths[-1],
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            # Don't need bias on both A and B since they would be doing the same thing
            bias=False,
            **self.factory_kwargs
        )

    def compute_b(self, x: Tensor, sigma_lower, a_wf, prev_alpha: Tensor, prev_omega_r: Tensor) -> Tuple[
        Tensor, Tensor]:
        """
        Propagate the final activations through the last Lipschitz layer.

        Parameters
        ----------
        x : Tensor
            Activation after the residual stack.
        sigma_lower : Tensor
            Accumulated Cholesky factor from previous layers.
        a_wf : Tensor
            Frequency-domain weights from the first layer.
        prev_alpha, prev_omega_r : Tensor
            Relaxation parameter and Cholesky factor from the last inner layer.

        Returns
        -------
        output : Tensor
            Final convolution output.
        weights : Tensor
            Normalized weights returned by :class:`LastLipschitzConvLayer`.

        Raises
        ------
        RuntimeError
            Propagated if Cholesky decomposition fails.

        Notes
        -----
        Combines the accumulated ``sigma_lower`` with the first-layer weights to
        prepare the final block.

        Mathematical Sketch
        -------------------
        Computes ``scaling = I + (A σ)(A σ)^T`` prior to the last call.

        Examples
        --------
        >>> # doctest: +SKIP
        >>> import torch
        >>> net = DeepLipschitzConvResNet(3, 4, layers_widths=[4])
        >>> x = torch.randn(1, 3, 32, 32)
        >>> first, a_wf, d_inv, alpha, omega_r, const = net.A(x)
        >>> sigma = torch.eye(d_inv.shape[1], device=x.device).expand_as(d_inv) @ d_inv
        >>> out, weights = net.compute_b(x, sigma, a_wf, alpha, omega_r)
        >>> out.shape[-1] == x.shape[-1]
        True
        """

        # left = a_wf @ sigma_lower
        # scaling = self.B.identity_out + left @ left.mT

        scaling = self.B.identity_out + torch.einsum("fir,fjs,frk,fsk->fij", a_wf, a_wf, sigma_lower, sigma_lower)

        R_sigma = self.cholesky_function(scaling, upper=True)

        second, b_wf = self.B(x, R_sigma, prev_alpha, prev_omega_r)

        return second, b_wf

    # @torch.compiler.disable
    def forward(self, x: Tensor) -> Tensor:
        """Run the residual CNN end-to-end.

        Parameters
        ----------
        x : (B, C, H, W) tensor
            Input image batch.

        Returns
        -------
        logits : (B, out_channels) tensor
            Output logits (pre-softmax) after enforcing the Lipschitz bound.

        Raises
        ------
        RuntimeError
            Propagated from submodules when tensor shapes are invalid.

        Notes
        -----
        Accumulates ``sigma_lower`` matrices across residual blocks and finally
        combines them via :meth:`compute_b`.

        Mathematical Sketch
        -------------------
        Evaluates ``A(x)`` plus the final block ``B`` acting on the residual
        stack, mirroring the linear architecture in the convolutional domain.

        Examples
        --------
        >>> import torch
        >>> net = DeepLipschitzConvResNet(3, 10, layers_widths=[16])
        >>> net(torch.randn(1, 3, 32, 32)).shape
        torch.Size([1, 10])
        """
        first, a_wf, d_inv, prev_alpha, prev_omega_r, prev_constant = self.A(x)

        gamma = torch.eye(d_inv.shape[1], device=d_inv.device, dtype=d_inv.dtype).expand(d_inv.shape[0], -1, -1)
        sigma_lower = None

        current_input = x

        for layer in self.layers:
            T = gamma @ d_inv

            current_input, c_wf, d_inv, prev_alpha, prev_omega_r, prev_constant = layer(current_input, prev_alpha,
                                                                                        prev_omega_r, prev_constant)

            if sigma_lower is None:
                sigma_lower = T
            else:
                sigma_lower = self.cholesky_function(sigma_lower @ sigma_lower.mT + T @ T.mT, upper=False)

            gamma = T @ c_wf.mT

        second, b_wf = self.compute_b(current_input, sigma_lower, a_wf, prev_alpha, prev_omega_r)

        return first + second


class DeepLipschitzConvSequential(nn.Module):
    """Sequential (non-residual) CNN variant with Lipschitz guarantees.

    Parameters
    ----------
    in_channels, out_channels : int
        Input/output channels.
    layers_widths : sequence
        Hidden channel widths (optionally ``(channels, kernel)`` tuples).
    kernel_size, stride, padding, dilation, groups : int or tuple
        Convolutional hyperparameters.
    activation : Module or list, optional
        Activation applied after each convolution.
    lipschitz_constant : float, default=1.0
        Target Lipschitz constant.
    enable_alpha, use_safe_cholesky, wrong_normalization : bool
        Control normalization behaviors.
    dropout : bool, default=False
        Insert dropout modules between layers.
    use_averaging : bool, default=False
        Insert average pooling layers periodically.
    device : str or torch.device, default="cuda"
        Device for module initialization.
    dtype : torch.dtype, optional
        Numeric precision.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If widths or channel counts are invalid.

    Examples
    --------
    >>> import torch
    >>> net = DeepLipschitzConvSequential(3, 10, layers_widths=[16, 16])
    >>> net(torch.randn(1, 3, 32, 32)).shape
    torch.Size([1, 10])
    """
    layers: ModuleList

    def __init__(self, in_channels: int,
                 out_channels: int,
                 layers_widths: Sequence[int],
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 activation: Optional[ActivationSpec] = None,
                 lipschitz_constant: float = 1.0,
                 enable_alpha: bool = True,
                 use_safe_cholesky: bool = False,
                 wrong_normalization: bool = False,
                 dropout: bool = False,
                 use_averaging: bool = False,
                 device: Union[str, torch.device] = 'cuda',
                 dtype: Optional[torch.dtype] = None) -> None:
        """Initialize the sequential CNN.

        Parameters
        ----------
        Same as the class docstring.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If configuration is invalid (empty layers or nonpositive channels).

        Notes
        -----
        Stores ``use_averaging``/``dropout`` flags for :meth:`generate_layers`.
        """
        super().__init__()
        self.use_averaging = use_averaging
        self.dropout = dropout
        self.wrong_normalization = wrong_normalization
        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_channels > 0, "The input width needs to be larger than 0"
        assert out_channels > 0, "The output width needs to be larger than 0"

        self.cholesky_function = safe_cholesky if use_safe_cholesky else torch.linalg.cholesky
        self.enable_alpha = enable_alpha
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer_widths = list(layers_widths)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.lipschitz_constant = lipschitz_constant
        self.lipschitz_constant_sq = lipschitz_constant * lipschitz_constant

        self.activation = ReLU() if activation is None else activation

        self.generate_layers()

    def reset_parameters(self) -> None:
        """Reset all convolutional layers to their default initialization.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated from child layers during reset.
        """
        for layer in self.layers:
            layer.reset_parameters()

    def generate_layers(self) -> None:
        """Instantiate the sequential stack of Lipschitz convolutional blocks.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Interleaves optional average pooling and dropout modules according to
        ``use_averaging``/``dropout`` flags.

        Mathematical Sketch
        -------------------
        Sets ``constant_sq = 2`` for hidden blocks and ``1`` for the head to
        satisfy the global Lipschitz constraint.

        Examples
        --------
        >>> net = DeepLipschitzConvSequential(3, 10, layers_widths=[16])
        >>> len(net.layers) >= 1
        True
        """
        layer = []

        prev = self.in_channels

        for i, out_features in enumerate(self.layer_widths):
            activation = self.activation
            constant = 2.0

            if i == 0:
                class_ = FirstSequentialLipschitzConvLayer
            else:
                class_ = InnerWeightLipschitzConvLayer

            if isinstance(out_features, Sequence):
                out_channels, kernel_size = out_features
                padding = kernel_size // 2
            else:
                out_channels = out_features
                kernel_size = self.kernel_size
                padding = self.padding

            layer.append(
                class_(prev, out_channels,
                       kernel_size=kernel_size,
                       enable_alpha=self.enable_alpha,
                       constant_sq=constant,
                       stride=self.stride,
                       padding=padding,
                       dilation=self.dilation,
                       activation=activation,
                       groups=self.groups, **self.factory_kwargs))

            if self.use_averaging and i % 2 == 0:
                layer.append(nn.AvgPool2d(2, divisor_override=2))

            if self.dropout:
                layer.append(nn.Dropout())

            prev = out_channels

        layer.append(
            InnerWeightLipschitzConvLayer(prev, self.out_channels, kernel_size=self.kernel_size, constant_sq=1.0,
                                          activation=None,
                                          **self.factory_kwargs))

        self.layers = ModuleList(layer)

    def forward(self, x: Tensor) -> Tensor:
        """Run the sequential CNN.

        Parameters
        ----------
        x : (B, C, H, W) tensor
            Input image batch.

        Returns
        -------
        activations : Tensor
            Output logits or intermediate activations depending on configuration.

        Raises
        ------
        RuntimeError
            Propagated when submodules encounter invalid tensor shapes.

        Notes
        -----
        Maintains normalization factors via ``d_inv``/``prev_alpha``/``prev_omega_r``
        similar to the residual variant.

        Mathematical Sketch
        -------------------
        Sequentially applies normalized convolutions and optional pooling/dropout
        with each block satisfying the Lipschitz constraint.

        Examples
        --------
        >>> import torch
        >>> net = DeepLipschitzConvSequential(3, 10, layers_widths=[16, 16])
        >>> net(torch.randn(1, 3, 32, 32)).shape
        torch.Size([1, 10])
        """
        current_input = x

        first = True

        for layer in self.layers:
            if isinstance(layer, nn.Dropout) or isinstance(layer, nn.AvgPool2d):
                current_input = layer(current_input)
            else:
                if first:
                    current_input, _, d_inv, prev_alpha, prev_omega_r, prev_constant = layer(current_input)
                    first = False
                else:
                    current_input, _, d_inv, prev_alpha, prev_omega_r_temp, prev_constant = layer(current_input,
                                                                                                  prev_alpha,
                                                                                                  prev_omega_r,
                                                                                                  prev_constant)

                    if not self.wrong_normalization:
                        prev_omega_r = prev_omega_r_temp

            # current_input, _, d_inv, prev_alpha, prev_omega_r, prev_constant = layer(current_input)

        return current_input
