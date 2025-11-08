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
Each architecture composes Lipschitz-normalized blocks, yielding global bounds
of the form \(L = \prod_i L_i\) with each \(L_i \leq 1\).

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
from utils import safe_cholesky

type ActivationSpec = Union[nn.Module, list[nn.Module]]


class DeepLipschitzLinearResNet(nn.Module):
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
        Use :func:`utils.safe_cholesky` instead of :func:`torch.linalg.cholesky`.
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

        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_features > 0, "The input width needs to be larger than 0"
        assert out_features > 0, "The output width needs to be larger than 0"

        self.cholesky_function = safe_cholesky if use_safe_cholesky else torch.linalg.cholesky
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

    def compute_b(self, x: Tensor, sigma_lower: Tensor, a_weight: Tensor, prev_alpha: Tensor, prev_omega_r: Tensor):
        """
        Compute the residual contribution of the last block.

        Parameters
        ----------
        x : Tensor
            Activation after the residual stack.
        sigma_lower : Tensor
            Accumulated Cholesky factor.
        a_weight : Tensor
            Weight matrix from the first layer.
        prev_alpha, prev_omega_r : Tensor
            Factors from the last inner layer.

        Returns
        -------
        tuple
            Output activations and normalized weights for the final block.

        Raises
        ------
        RuntimeError
            Propagated if Cholesky decompositions fail.

        Notes
        -----
        Solves for the sigma factor by combining the first-layer weights with
        the accumulated ``sigma_lower``.

        Mathematical Sketch
        -------------------
        Computes ``R_sigma = chol(I + A σ σ^T A^T)`` before passing values to the
        last layer.

        Examples
        --------
        >>> net = DeepLipschitzLinearResNet(4, 2, layers_widths=[4])
        >>> x = torch.randn(2, 4)
        >>> first, a_weight, d_inv, alpha, omega_r, constant = net.A(x)
        >>> sigma_lower = net.A.identity @ d_inv
        >>> net.compute_b(x, sigma_lower, a_weight, alpha, omega_r)[0].shape
        torch.Size([2, 2])
        """
        # inner_inverse = torch.cholesky_inverse(prev_r, upper=True)
        # max_singular = torch.sqrt(2 * self.spectral_norm(inner_inverse))

        left = a_weight @ sigma_lower

        scaling = self.B.identity_out + left @ left.T

        R_sigma = self.cholesky_function(scaling, upper=True)

        second, b_weight = self.B(x, R_sigma, prev_alpha, prev_omega_r)

        return second, b_weight

    def forward(self, x: Tensor) -> Tensor:
        r"""Run the residual linear network.

        Parameters
        ----------
        x : (B, in_features) tensor
            Input batch.

        Returns
        -------
        logits : (B, out_features) tensor
            Output logits respecting the Lipschitz bound.

        Raises
        ------
        RuntimeError
            Propagated from submodules if triangular solves or Cholesky
            operations fail.

        Notes
        -----
        Iteratively updates ``sigma_lower`` to aggregate diagonal factors before
        calling :meth:`compute_b`.

        Mathematical Sketch
        -------------------
        Implements \(f(x) = A(x) + B(h_{L-1}(x))\) where ``A`` and ``B`` are the
        first/last Lipschitz blocks and ``h`` denotes the inner residual stack.

        Examples
        --------
        >>> import torch
        >>> net = DeepLipschitzLinearResNet(8, 4, layers_widths=[8])
        >>> net(torch.randn(2, 8)).shape
        torch.Size([2, 4])
        """
        first, a_weight, d_inv, prev_alpha, prev_omega_r, prev_constant = self.A(x)

        gamma = self.A.identity
        sigma_lower = None

        current_input = x

        # for i, layer in enumerate(self.layers):
        for layer in self.layers:
            T = gamma @ d_inv

            # d_inv_prev = d_inv

            current_input, c_weight, d_inv, prev_alpha, prev_omega_r, prev_constant = layer(current_input, prev_alpha,
                                                                                            prev_omega_r, prev_constant)

            # with torch.no_grad():
            #     R_inv = torch.linalg.solve_triangular(prev_layer.omega_r, layer.identity, upper=True)
            #
            #     scale = prev_layer.alpha.sqrt() * prev_layer.constant
            #
            #     temp = scale * R_inv
            #     result = temp.T @ d_inv_prev @ temp
            #
            #     assert torch.allclose(layer.identity, result)

            # with torch.no_grad():
            #     val = c_weight @ d_inv_prev @ c_weight.T
            #     eigenvals = torch.linalg.eigvalsh(val)
            #     max_val = eigenvals.max().item()
            #     min_val = eigenvals.min().item()
            #
            #     print(i, max_val <= 2.0, max_val, min_val)

            if sigma_lower is None:
                sigma_lower = T
            else:
                # Instead of using custom CholeskyRankK update this is computationally much faster
                sigma_lower = safe_cholesky(sigma_lower @ sigma_lower.T + T @ T.T, upper=False)
                # sigma_lower = cholesky_rank_k(sigma_lower, T)

            gamma = T @ c_weight.T

            current_input = self.activation(current_input)

        second, b_weight = self.compute_b(current_input, sigma_lower, a_weight, prev_alpha, prev_omega_r)

        return first + second

    def last_weight(self) -> torch.Tensor:
        """Return the concatenated last-layer weights for certification.

        Parameters
        ----------
        None

        Returns
        -------
        weights : (out_features, in_features + out_features) tensor
            Concatenation of first/last block weights as expected by
            certification utilities.

        Raises
        ------
        RuntimeError
            Propagated if submodules fail during the forward-like pass.

        Notes
        -----
        Recomputes a dummy forward pass with zeros to retrieve normalized
        weights without gradient tracking.

        Mathematical Sketch
        -------------------
        The concatenation corresponds to the certificate structure that maps
        both initial and terminal weights.

        Examples
        --------
        >>> net = DeepLipschitzLinearResNet(4, 2, layers_widths=[4])
        >>> W = net.last_weight()
        >>> W.shape[0] == net.out_features
        True
        """
        x = torch.zeros((1, self.in_features), **self.factory_kwargs)

        first, a_weight, d_inv, prev_alpha, prev_omega_r, prev_constant = self.A(x)

        gamma = self.A.identity
        sigma_lower = None

        current_input = x

        for layer in self.layers:
            T = gamma @ d_inv

            current_input, c_weight, d_inv, prev_alpha, prev_omega_r, prev_constant = layer(current_input, prev_alpha,
                                                                                            prev_omega_r, prev_constant)

            if sigma_lower is None:
                sigma_lower = T
            else:
                sigma_lower = safe_cholesky(sigma_lower @ sigma_lower.T + T @ T.T, upper=False)

            gamma = T @ c_weight.T

            current_input = self.activation(current_input)

        second, b_weight = self.compute_b(current_input, sigma_lower, a_weight, prev_alpha, prev_omega_r)

        return torch.cat([a_weight, b_weight], dim=1)


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


class DeepLipschitzSequential(nn.Module):
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

    def forward(self, x: Tensor) -> Tensor:
        r"""Run the sequential linear network.

        Parameters
        ----------
        x : (B, in_features) tensor
            Input batch.

        Returns
        -------
        logits : (B, out_features) tensor
            Output logits after all layers.

        Raises
        ------
        RuntimeError
            Propagated when submodules encounter invalid shapes.

        Notes
        -----
        Maintains ``d_sqrt`` (diagonal factor), ``prev_alpha``, and
        ``prev_const`` across layers to ensure global Lipschitz control.

        Mathematical Sketch
        -------------------
        Sequentially applies normalized linear blocks so that
        \(L = L_1 L_2 \cdots L_n\) with all \(L_i \leq 1\).

        Examples
        --------
        >>> import torch
        >>> net = DeepLipschitzSequential(8, 2, layers_widths=[8, 8])
        >>> net(torch.randn(1, 8)).shape
        torch.Size([1, 2])
        """
        current_input = x
        d_sqrt = self.lipschitz_constant * self.layers[0].identity
        prev_alpha = torch.tensor(1.0, **self.factory_kwargs)
        prev_const = torch.tensor(1.0, **self.factory_kwargs)

        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                current_input = layer(current_input)
            else:
                layer: LinearLipschitz

                current_input, d_sqrt, _ = layer(current_input, d_sqrt, prev_alpha, prev_const)

                prev_const = layer.constant
                prev_alpha = layer.alpha

        return current_input

    def last_weight(self) -> Tensor:
        """Retrieve the final normalized weights for certification.

        Parameters
        ----------
        None

        Returns
        -------
        weights : (out_features, prev_features) tensor
            Final normalized weight matrix.

        Raises
        ------
        RuntimeError
            Propagated when submodules fail during the dummy forward pass.

        Notes
        -----
        Mirrors the main forward loop but captures the last normalized ``W``
        instead of producing logits.

        Mathematical Sketch
        -------------------
        The output corresponds to the final Lipschitz-normalized affine map.

        Examples
        --------
        >>> net = DeepLipschitzSequential(4, 2, layers_widths=[4])
        >>> net.last_weight().shape[0] == net.out_features
        True
        """
        x = torch.zeros((1, self.in_features), **self.factory_kwargs)

        d_sqrt = self.lipschitz_constant * self.layers[0].identity
        prev_alpha = torch.tensor(1.0, **self.factory_kwargs)
        prev_const = torch.tensor(1.0, **self.factory_kwargs)

        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                x = layer(x)
            else:
                layer: LinearLipschitz
                x, d_sqrt, W = layer(x, d_sqrt, prev_alpha, prev_const)

                prev_const = layer.constant
                prev_alpha = layer.alpha

        return W


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
