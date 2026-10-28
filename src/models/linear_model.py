r"""
High-level Lipschitz linear architectures.

Parameters
----------
None

Returns
-------
None

Raises
------
RuntimeError
    Propagated from PyTorch when tensor shapes or Cholesky factors are invalid.

Notes
-----
This module exposes residual and sequential linear networks whose components
are built from :mod:`models.linear_layers` primitives.

Mathematical Sketch
-------------------
Version 2 jointly certifies each residual or sequential feature module using
the exact structured construction in models.certified_ldl. Individual internal
weights need not have spectral norm at most one. A separate normalized head
is accounted for through its row differences during classification certification.

Examples
--------
>>> import torch
>>> from models.linear_model import DeepLipschitzLinearResNet
>>> net = DeepLipschitzLinearResNet(16, 10, layers_widths=[32, 32])
>>> net(torch.randn(2, 16)).shape
torch.Size([2, 10])
"""

from collections.abc import Sequence
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import ModuleList, ReLU

from lbdn_models.layers import LinearNormalized
from models.linear_layers import FirstLipschitzLinearLayer, LastLipschitzLinearLayer, InnerWeightLipschitzLinearLayer, \
    LinearLipschitz
from models.certified_ldl import residual_weights
from models.certificate_version import CertificateVersion, validate_activation

type ActivationSpec = Union[nn.Module, list[nn.Module]]


class DeepLipschitzLinearResNet(CertificateVersion, nn.Module):
    r"""Residual linear network with explicit Lipschitz control.

    Parameters
    ----------
    in_features, out_features : int
        Input/output feature dimensions.
    layers_widths : sequence of int
        Hidden widths for each residual block.
    activation : Module or list, optional
        Activation(s) applied after each block.
    lipschitz_constant : float, default=1.0
        Global Lipschitz bound enforced by the architecture.
    enable_alpha : bool, default=True
        Whether to learn the relaxation parameters.
    use_safe_cholesky : bool, default=False
        Retained for constructor compatibility; True is rejected by version 2.
    device : str or torch.device, default="cuda"
        Device hosting the layers.
    dtype : torch.dtype, optional
        Desired floating-point precision.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If invalid widths are provided (checked via assertions).

    Notes
    -----
    Composes :class:`FirstLipschitzLinearLayer`, repeated
    :class:`InnerWeightLipschitzLinearLayer` blocks, and a final
    :class:`LastLipschitzLinearLayer`.

    Mathematical Sketch
    -------------------
    Each block enforces \(L_i \leq 1\); thus the residual connection preserves
    \(L \leq 1\) for the entire network.

    Examples
    --------
    >>> import torch
    >>> net = DeepLipschitzLinearResNet(8, 4, layers_widths=[8, 8])
    >>> net(torch.randn(2, 8)).shape
    torch.Size([2, 4])
    """
    A: FirstLipschitzLinearLayer
    B: LastLipschitzLinearLayer
    layers: ModuleList

    def __init__(self, in_features: int, out_features: int, layers_widths: Sequence[int],
                 activation: Optional[ActivationSpec] = None, lipschitz_constant: float = 1.0,
                 enable_alpha: bool = True,
                 use_safe_cholesky: bool = False,
                 device: Union[str, torch.device] = 'cuda', dtype: Optional[torch.dtype] = None) -> None:
        """Initialize the residual Lipschitz network.

        Parameters
        ----------
        in_features, out_features : int
            Input/output feature dimensions.
        layers_widths : sequence of int
            Widths of the hidden residual blocks.
        activation : Module or list, optional
            Activation(s) applied after each hidden block.
        lipschitz_constant : float, default=1.0
            Global Lipschitz bound enforced by the architecture.
        enable_alpha : bool, default=True
            Whether to make the relaxation parameter trainable.
        use_safe_cholesky : bool, default=False
            Switch to the numerically stable Cholesky helper.
        device : str or torch.device, default="cuda"
            Device hosting all parameters.
        dtype : torch.dtype, optional
            Parameter precision.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If ``layers_widths`` is empty or feature counts are nonpositive.

        Notes
        -----
        Stores ``factory_kwargs`` for consistent tensor creation.

        Examples
        --------
        >>> DeepLipschitzLinearResNet(4, 2, layers_widths=[4])
        DeepLipschitzLinearResNet(...)
        """
        super().__init__()
        self.register_buffer("_parameterization_version", torch.tensor(2, dtype=torch.int64))

        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_features > 0, "The input width needs to be larger than 0"
        assert out_features > 0, "The output width needs to be larger than 0"

        if use_safe_cholesky:
            raise ValueError("Certificate version 2 does not permit matrix-changing Cholesky repair")
        self.enable_alpha = enable_alpha
        self.lipschitz_constant = lipschitz_constant
        self.lipschitz_constant_sq = lipschitz_constant * lipschitz_constant

        self.out_features = out_features
        self.in_features = in_features
        self.layer_widths = layers_widths

        self.dtype = dtype
        self.device = device

        self.factory_kwargs = {"device": device, "dtype": dtype}

        if activation is None:
            self.activation = ReLU()
        else:
            self.activation = activation

        validate_activation(self.activation)
        self.generate_layers()

    def reset_parameters(self) -> None:
        """Reset all submodules to their initialization scheme.

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

        Notes
        -----
        Applies each submodule's ``reset_parameters`` helper sequentially.

        Examples
        --------
        >>> import torch
        >>> net = DeepLipschitzLinearResNet(4, 2, layers_widths=[4])
        >>> net.reset_parameters()
        """
        # Find a better initialization scheme for the current formulation of the system
        self.A.reset_parameters()
        self.B.reset_parameters()

        for layer in self.layers:
            layer.reset_parameters()

    def generate_layers(self) -> None:
        """Instantiate the first, residual, and last Lipschitz layers.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated if layer construction fails.

        Notes
        -----
        Creates :class:`ModuleList` of inner layers while storing ``self.A`` and
        ``self.B`` for the first/last blocks.

        Mathematical Sketch
        -------------------
        Inner layers use ``constant_sq = 2`` to maintain the residual Lipschitz
        budget.

        Examples
        --------
        >>> net = DeepLipschitzLinearResNet(4, 2, layers_widths=[4])
        >>> len(net.layers)
        1
        """
        self.A = FirstLipschitzLinearLayer(in_features=self.in_features, out_features=self.out_features,
                                           constant_sq=self.lipschitz_constant_sq, enable_alpha=self.enable_alpha,
                                           **self.factory_kwargs)

        self.B = LastLipschitzLinearLayer(in_features=self.layer_widths[-1], out_features=self.out_features,
                                          bias=False,
                                          enable_alpha=self.enable_alpha,
                                          # Don't need bias on both A and B since they would be doing the same thing
                                          **self.factory_kwargs)

        layer = []

        prev = self.in_features

        for i, out_features in enumerate(self.layer_widths):
            layer.append(
                InnerWeightLipschitzLinearLayer(prev, out_features,
                                                constant_sq=2.0,
                                                enable_alpha=self.enable_alpha,
                                                **self.factory_kwargs))
            prev = out_features

        self.layers = ModuleList(layer)

        # self.spectral_norm = SpectralNorm(self.B.identity)

    def normalized_weights(self):
        """Return the corrected version-2 residual certificate weights."""
        return residual_weights(self.A.weight, [layer.weight for layer in self.layers],
                                self.B.weight, bound=self.lipschitz_constant,
                                alphas=[self.A.alpha, *[layer.alpha for layer in self.layers], self.B.alpha])


    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the corrected residual block; factorization failure is fatal."""
        weights = self.normalized_weights()
        hidden = x
        for i, (layer, c) in enumerate(zip(self.layers, weights.cs)):
            activation = self.activation[i] if isinstance(self.activation, (list, nn.ModuleList)) else self.activation
            hidden = activation(torch.nn.functional.linear(hidden, c, layer.bias))
        return (torch.nn.functional.linear(x, weights.a, self.A.bias)
                + torch.nn.functional.linear(hidden, weights.b, self.B.bias))


    def last_weight(self) -> Tensor:
        """Return the identity head on the certified output feature map.

        Concatenating A and B does not give a head with a 1-Lipschitz feature
        map: the stacked input/hidden state has no such established bound.
        """
        if hasattr(self, 'last'):
            return torch.nn.functional.normalize(self.last.weight, p=2, dim=1)
        return torch.eye(self.out_features, device=self.A.weight.device, dtype=self.A.weight.dtype)



class DeepLipschitzLinearResNetLNN(DeepLipschitzLinearResNet):
    """Residual variant used for the Lipschitz Neural Network (LNN) baseline.

    Parameters
    ----------
    Same as :class:`DeepLipschitzLinearResNet`.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Propagated from base class initializers.

    Notes
    -----
    Keeps the latent dimensionality constant before the final normalized head.

    Examples
    --------
    >>> import torch
    >>> net = DeepLipschitzLinearResNetLNN(8, 4, layers_widths=[8, 8])
    >>> net(torch.randn(2, 8)).shape
    torch.Size([2, 4])
    """

    def generate_layers(self) -> None:
        """Override to keep the latent dimension constant before the final head.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated if layer creation fails.
        """
        self.A = FirstLipschitzLinearLayer(in_features=self.in_features, out_features=self.layer_widths[-1],
                                           constant_sq=self.lipschitz_constant_sq, enable_alpha=self.enable_alpha,
                                           **self.factory_kwargs)

        self.B = LastLipschitzLinearLayer(in_features=self.layer_widths[-1], out_features=self.layer_widths[-1],
                                          bias=False,
                                          enable_alpha=self.enable_alpha,
                                          # Don't need bias on both A and B since they would be doing the same thing
                                          **self.factory_kwargs)

        layer = []

        prev = self.in_features

        for i, out_features in enumerate(self.layer_widths):
            layer.append(
                InnerWeightLipschitzLinearLayer(prev, out_features,
                                                constant_sq=2.0,
                                                enable_alpha=self.enable_alpha,
                                                **self.factory_kwargs))
            prev = out_features

        self.last = LinearNormalized(prev, self.out_features, **self.factory_kwargs)

        self.layers = ModuleList(layer)

    def forward(self, x: Tensor) -> Tensor:
        """Run the residual network followed by the normalized head.

        Parameters
        ----------
        x : (B, in_features) tensor
            Input batch.

        Returns
        -------
        logits : (B, out_features) tensor
            Output produced by the normalized head.

        Raises
        ------
        RuntimeError
            Propagated from submodules when shapes mismatch.

        Examples
        --------
        >>> import torch
        >>> net = DeepLipschitzLinearResNetLNN(4, 2, layers_widths=[4])
        >>> net(torch.randn(1, 4)).shape
        torch.Size([1, 2])
        """
        x = super().forward(x)

        return self.last(x)


class DeepLipschitzSequential(CertificateVersion, nn.Module):
    """Sequential linear network composed of ``LinearLipschitz`` blocks.

    Parameters
    ----------
    in_features, out_features : int
        Input/output feature sizes.
    layers_widths : sequence of int
        Hidden widths for each sequential block.
    activation : Module or list, optional
        Activation after each block.
    lipschitz_constant : float, default=1.0
        Global Lipschitz bound target.
    dropout : bool, default=False
        Whether to interleave dropout modules.
    device : str or torch.device, default="cuda"
        Device for parameter allocation.
    dtype : torch.dtype, optional
        Numeric precision.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If layer widths are invalid or feature counts are nonpositive.

    Notes
    -----
    Chains ``LinearLipschitz`` blocks and optionally dropout to build a simple
    Lipschitz-constrained MLP.

    Examples
    --------
    >>> import torch
    >>> net = DeepLipschitzSequential(8, 2, layers_widths=[8, 8])
    >>> net(torch.randn(3, 8)).shape
    torch.Size([3, 2])
    """
    layers: ModuleList

    def __init__(self, in_features: int, out_features: int, layers_widths: Sequence[int],
                 activation: Optional[ActivationSpec] = None, lipschitz_constant: float = 1.0,
                 dropout: bool = False,
                 device: Union[str, torch.device] = 'cuda', dtype: Optional[torch.dtype] = None) -> None:
        """Initialize the sequential Lipschitz MLP.

        Parameters
        ----------
        in_features, out_features : int
            Input/output feature counts.
        layers_widths : sequence of int
            Hidden widths for the sequential stack.
        activation : Module or list, optional
            Activation applied after each linear block.
        lipschitz_constant : float, default=1.0
            Target Lipschitz constant enforced across the chain.
        dropout : bool, default=False
            Insert dropout modules between layers.
        device : str or torch.device, default="cuda"
            Device for tensors.
        dtype : torch.dtype, optional
            Numeric precision.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If configuration is invalid (empty layers or nonpositive widths).

        Notes
        -----
        Stores ``factory_kwargs`` for later tensor creation.

        Examples
        --------
        >>> DeepLipschitzSequential(4, 2, layers_widths=[4])
        DeepLipschitzSequential(...)
        """
        super().__init__()
        self.register_buffer("_parameterization_version", torch.tensor(2, dtype=torch.int64))

        if dropout:
            raise ValueError("Dropout is not supported by the version-2 training certificate")
        self.dropout = dropout
        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_features > 0, "The input width needs to be larger than 0"
        assert out_features > 0, "The output width needs to be larger than 0"

        self.lipschitz_constant = lipschitz_constant
        self.lipschitz_constant_sq = lipschitz_constant * lipschitz_constant

        self.out_features = out_features
        self.in_features = in_features
        self.layer_widths = layers_widths

        self.dtype = dtype
        self.device = device

        self.factory_kwargs = {"device": device, "dtype": dtype}

        if activation is None:
            self.activation = ReLU()
        else:
            self.activation = activation

        validate_activation(self.activation)
        self.generate_layers()

    def reset_parameters(self) -> None:
        """Reset all linear blocks (and leave dropout unchanged).

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

        Examples
        --------
        >>> net = DeepLipschitzSequential(4, 2, layers_widths=[4])
        >>> net.reset_parameters()
        """
        for layer in self.layers:
            layer.reset_parameters()

    def generate_layers(self) -> None:
        """Build the sequential list of Lipschitz linear layers (plus dropout).

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated when layer instantiation fails.

        Notes
        -----
        Appends a final ``LinearLipschitz`` layer without activation to produce
        logits.

        Mathematical Sketch
        -------------------
        Hidden layers use ``constant_sq = 2`` so that the product of constants
        equals the desired global Lipschitz bound.

        Examples
        --------
        >>> net = DeepLipschitzSequential(4, 2, layers_widths=[4])
        >>> isinstance(net.layers[-1], LinearLipschitz)
        True
        """
        layer = []

        prev = self.in_features

        for i, out_features in enumerate(self.layer_widths):
            activation = self.activation
            constant = 2.0

            layer.append(
                LinearLipschitz(prev, out_features, constant_sq=constant, activation=activation, **self.factory_kwargs))

            if self.dropout:
                layer.append(nn.Dropout())

            prev = out_features

        layer.append(
            LinearLipschitz(prev, self.out_features, constant_sq=1.0, activation=None, **self.factory_kwargs))

        self.layers = ModuleList(layer)

    def normalized_weights(self):
        """Use the same exact certificate with the feed-through fixed to zero."""
        layers = list(self.layers)
        a = layers[0].weight.new_zeros(layers[-1].out_features, self.in_features)
        return residual_weights(a, [layer.weight for layer in layers[:-1]],
                                layers[-1].weight, bound=self.lipschitz_constant,
                                alphas=[1.0, *[layer.alpha for layer in layers]])

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate a jointly certified sequential feature map."""
        weights = self.normalized_weights()
        for layer, c in zip(self.layers[:-1], weights.cs):
            x = torch.nn.functional.linear(x, c, layer.bias)
            if layer.activation is not None:
                x = layer.activation(x)
        return torch.nn.functional.linear(x, weights.b, self.layers[-1].bias)


    def last_weight(self) -> Tensor:
        """Return the actual normalized head, or identity on the certified output."""
        if hasattr(self, 'last'):
            return torch.nn.functional.normalize(self.last.weight, p=2, dim=1)
        last = self.layers[-1].weight
        return torch.eye(self.out_features, device=last.device, dtype=last.dtype)



class DeepLipschitzSequentialLNN(DeepLipschitzSequential):
    """Sequential variant used for Lipschitz Neural Network experiments.

    Parameters
    ----------
    Same as :class:`DeepLipschitzSequential`.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Propagated from base-class initialization when invalid configs are
        supplied.

    Notes
    -----
    Adds an extra normalized head after the shared sequential body.

    Examples
    --------
    >>> net = DeepLipschitzSequentialLNN(8, 2, layers_widths=[8])
    >>> isinstance(net.last, LinearNormalized)
    True
    """

    def generate_layers(self) -> None:
        """Override to insert an additional normalized head layer.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated if layer construction fails.
        """
        layer = []

        prev = self.in_features

        for i, out_features in enumerate(self.layer_widths):
            activation = self.activation
            constant = 2.0

            layer.append(
                LinearLipschitz(prev, out_features, constant_sq=constant, activation=activation, **self.factory_kwargs))

            if self.dropout:
                layer.append(nn.Dropout())

            prev = out_features

        layer.append(
            LinearLipschitz(prev, self.layer_widths[-1], constant_sq=1.0, activation=None, **self.factory_kwargs))

        self.layers = ModuleList(layer)

        self.last = LinearNormalized(self.layer_widths[-1], self.out_features, **self.factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Run the base sequential network followed by the normalized head.

        Parameters
        ----------
        x : (B, in_features) tensor
            Input batch.

        Returns
        -------
        logits : (B, out_features) tensor
            Output after the normalized head.

        Raises
        ------
        RuntimeError
            Propagated from the base network or normalized head.

        Examples
        --------
        >>> import torch
        >>> net = DeepLipschitzSequentialLNN(4, 2, layers_widths=[4])
        >>> net(torch.randn(1, 4)).shape
        torch.Size([1, 2])
        """
        x = super().forward(x)
        x = self.last(x)

        return x
