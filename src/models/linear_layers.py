r"""
Lipschitz-controlled linear building blocks.

Parameters
----------
None

Returns
-------
None

Raises
------
RuntimeError
    Raised indirectly when PyTorch detects invalid tensor shapes/values during
    Cholesky operations.

Notes
-----
Implements variants of :class:`torch.nn.Linear` whose weights are normalized via
Gram-matrix Cholesky factors to maintain spectral bounds.

Mathematical Sketch
-------------------
Each layer enforces \(L \leq \text{constant}\) by solving
\(W_{\text{norm}} = R^{-1} W\) where ``R`` is a Cholesky factor of
``W W^T + α I``.

Examples
--------
>>> import torch
>>> from models.linear_layers import FirstLipschitzLinearLayer
>>> layer = FirstLipschitzLinearLayer(16, 16)
>>> layer(torch.randn(2, 16))[0].shape
torch.Size([2, 16])
"""

import math
from collections.abc import Callable
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils import parametrize

from utils import PositiveExp

type LinearArtifacts = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class AbstractLipschitzLinearLayer(nn.Linear):
    r"""Linear transform with explicit Lipschitz normalization.

    Parameters
    ----------
    in_features, out_features : int
        Input/output feature sizes (see :meth:`__init__`).
    bias : bool, default=True
        Whether to include an additive bias term.
    constant_sq : float, default=1.0
        Target squared Lipschitz bound.
    enable_alpha : bool, default=True
        Whether to make the relaxation parameter trainable.
    device, dtype : Any, optional
        Tensor factory kwargs.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Propagated from :class:`torch.nn.Linear` when invalid shapes or dtypes
        are supplied.

    Notes
    -----
    Registers identity matrices and Lipschitz constants as buffers to support
    spectral normalization via Cholesky solves.

    Mathematical Sketch
    -------------------
    The layer ensures \(\|W_{\text{norm}} x\|_2 \leq \sqrt{L}\|x\|_2\) by
    computing ``R = chol(W W^T + α I)`` and using
    ``W_norm = constant * R^{-T} W``.

    Examples
    --------
    >>> import torch
    >>> layer = FirstLipschitzLinearLayer(4, 4)
    >>> layer(torch.randn(1, 4))[0].shape
    torch.Size([1, 4])
    """
    identity: torch.Tensor
    identity_out: torch.Tensor
    constant_sq: torch.Tensor
    constant: torch.Tensor
    R: torch.Tensor
    omega_r: torch.Tensor

    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
                 constant_sq: float = 1.0,
                 enable_alpha: bool = True,
                 device: Optional[torch.device | str] = None,
                 dtype: Optional[torch.dtype] = None) -> None:
        """Bootstrap the Lipschitz-aware linear layer.

        Parameters
        ----------
        in_features, out_features : int
            Input/output feature dimensions.
        bias : bool, default=True
            Whether to include an additive bias.
        constant_sq : float, default=1.0
            Target square of the Lipschitz constant.
        enable_alpha : bool, default=True
            Whether to make the relaxation parameter trainable.
        device : str or torch.device, optional
            Device on which to allocate parameters/buffers.
        dtype : torch.dtype, optional
            Parameter precision.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated from :class:`torch.nn.Linear` upon invalid construction
            arguments.

        Notes
        -----
        Registers identity buffers plus ``constant`` and ``alpha`` (optionally
        parametrized via :class:`utils.PositiveExp`).

        Examples
        --------
        >>> AbstractLipschitzLinearLayer(4, 4)  # doctest: +SKIP
        """
        super(AbstractLipschitzLinearLayer, self).__init__(in_features, out_features, bias=bias, device=device,
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

        initial_alpha = torch.tensor(1.0, **factory_kwargs)

        if enable_alpha:
            self.alpha = Parameter(initial_alpha)
            parametrize.register_parametrization(self, "alpha", PositiveExp(**factory_kwargs))
        else:
            self.register_buffer('alpha', initial_alpha)

    def reset_parameters(self) -> None:
        """Initialize weights/biases via Kaiming-uniform defaults.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated if PyTorch encounters invalid shapes during initialization.

        Notes
        -----
        Implements :func:`torch.nn.init.kaiming_uniform_` with ``a = sqrt(5)`` and
        biases sampled from ``U(-1/√fan_in, 1/√fan_in)``.

        Mathematical Sketch
        -------------------
        Choice of ``a`` preserves variance assuming ReLU-like activations under
        He initialization.

        Examples
        --------
        >>> import torch
        >>> layer = FirstLipschitzLinearLayer(8, 8)
        >>> layer.reset_parameters()
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def scale_w(self) -> tuple[Tensor, Tensor, Tensor]:
        r"""Normalize weights to satisfy the Lipschitz constraint.

        Parameters
        ----------
        None

        Returns
        -------
        omega : Tensor
            Gram matrix prior to scaling.
        W : Tensor
            Scaled weight matrix.
        R : Tensor
            Upper-triangular Cholesky factor of ``omega``.

        Raises
        ------
        RuntimeError
            Propagated if :func:`torch.linalg.cholesky` fails (e.g., non-SPD).

        Notes
        -----
        Equivalent to solving ``R R^T = W W^T + α I`` and computing
        ``W_norm = R^{-T} W``.

        Mathematical Sketch
        -------------------
        Ensures \(\|W_{\text{norm}}\|_2 \leq 1\) because ``R`` upper-triangular
        solves whiten the rows.

        Examples
        --------
        >>> layer = FirstLipschitzLinearLayer(4, 4)
        >>> omega, W, R = layer.scale_w()
        >>> omega.shape
        torch.Size([4, 4])
        """
        omega = self.sum_weight(transpose=True)
        R = torch.linalg.cholesky(omega, upper=True)

        W = torch.linalg.solve_triangular(R, self.weight, upper=True, left=False)

        return omega, W, R

    def sum_weight(self, transpose: bool = False) -> Tensor:
        """
        Form \\(W^\\top W + \\alpha I\\) or \\(W W^\\top + \\alpha I\\).

        Parameters
        ----------
        transpose : bool, default=False
            Choose the orientation of the Gram matrix.

        Returns
        -------
        gram : Tensor
            Symmetric positive-definite matrix oriented per ``transpose``.

        Raises
        ------
        RuntimeError
            Propagated when matrix multiplication encounters mismatched shapes.

        Notes
        -----
        Adds ``alpha`` to guarantee positive-definiteness even for degenerate
        weights.

        Mathematical Sketch
        -------------------
        Guarantees eigenvalues \\(\\lambda_i \\geq \\alpha\\), enabling stable
        Cholesky decompositions.

        Examples
        --------
        >>> layer = FirstLipschitzLinearLayer(4, 4)
        >>> gram = layer.sum_weight(transpose=True)
        >>> gram.shape
        torch.Size([4, 4])
        """
        if transpose:
            return self.weight.T @ self.weight + self.identity * self.alpha
        else:
            return self.weight @ self.weight.T + self.identity_out * self.alpha


class FirstLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    r"""Entry linear layer normalized without prior context.

    Parameters
    ----------
    See :class:`AbstractLipschitzLinearLayer`.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Propagated from base class constructors or Cholesky routines.

    Notes
    -----
    Serves as the starting point for sequential Lipschitz linear models by
    normalizing weights independently.

    Mathematical Sketch
    -------------------
    Because no prior sigma factors exist, the layer simply scales weights by its
    own ``constant`` so that \(L = \sqrt{\text{constant\_sq}}\).

    Examples
    --------
    >>> import torch
    >>> layer = FirstLipschitzLinearLayer(16, 16)
    >>> y, *_ = layer(torch.randn(2, 16))
    >>> y.shape
    torch.Size([2, 16])
    """

    def forward(self, input: Tensor) -> LinearArtifacts:
        r"""
        Apply the first linear transform with standalone scaling.

        Parameters
        ----------
        input : (batch, in_features) tensor
            Input activations.

        Returns
        -------
        output : Tensor
            Linear projection.
        W : Tensor
            Normalized weight matrix.
        d_inv : Tensor
            Diagonal inverse used downstream.
        alpha : Tensor
            Relaxation parameter.
        omega_r : Tensor
            Lower Cholesky factor of the Gram matrix.
        constant : Tensor
            Scalar Lipschitz constant for the layer.

        Raises
        ------
        RuntimeError
            Propagated if Cholesky or triangular solves fail.

        Notes
        -----
        Equivalent to calling :meth:`scale_w` followed by a constant rescaling
        because no prior sigma factors exist.

        Mathematical Sketch
        -------------------
        The output weights satisfy \(W_0 = c R^{-T} W\) with
        \(c = \sqrt{\text{constant\_sq}}\).

        Examples
        --------
        >>> layer = FirstLipschitzLinearLayer(8, 8)
        >>> y, *_ = layer(torch.randn(2, 8))
        >>> y.shape
        torch.Size([2, 8])
        """
        omega, W, R = self.scale_w()

        W = W * self.constant

        d_inv = omega / (self.constant_sq * self.alpha)
        omega_r = R.T

        return torch.nn.functional.linear(input, W, self.bias), W, d_inv, self.alpha, omega_r, self.constant


class InnerWeightLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    r"""Hidden layer that depends on scaling from the preceding block.

    Parameters
    ----------
    See :class:`AbstractLipschitzLinearLayer`.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Propagated when triangular solves or Cholesky decompositions fail.

    Notes
    -----
    Incorporates prior Lipschitz constants via ``prev_alpha`` and
    ``prev_constant`` to maintain the global bound.

    Mathematical Sketch
    -------------------
    The effective weight becomes
    \(W = c_{i-1} c_i R_{i-1}^{-T} W_i\), chaining previous sigma factors.

    Examples
    --------
    >>> import torch
    >>> layer = InnerWeightLipschitzLinearLayer(8, 8)
    >>> prev = FirstLipschitzLinearLayer(8, 8)
    >>> _, _, _, alpha, omega_r, constant = prev(torch.randn(2, 8))
    >>> y, *_ = layer(torch.randn(2, 8), alpha, omega_r, constant)
    >>> y.shape
    torch.Size([2, 8])
    """

    def forward(self, input: Tensor, prev_alpha: Tensor, prev_omega_r: Tensor, prev_constant: Tensor) -> \
            LinearArtifacts:
        r"""
        Apply an inner linear block conditioned on previous scaling factors.

        Parameters
        ----------
        input : Tensor
            Incoming activations.
        prev_alpha : Tensor
            Previous relaxation parameter.
        prev_omega_r : Tensor
            Previous lower Cholesky factor.
        prev_constant : Tensor
            Previous Lipschitz constant.

        Returns
        -------
        tuple
            (output, normalized weights, diagonal inverse, alpha, omega_r, constant)

        Raises
        ------
        RuntimeError
            Propagated from triangular/Cholesky solvers when matrices lose
            positive definiteness.

        Notes
        -----
        Reuses ``prev_omega_r`` to whiten current weights before re-scaling by
        ``prev_alpha`` and ``prev_constant``.

        Mathematical Sketch
        -------------------
        Computes \(W_i = c_{i-1} c_i R_{i-1}^{-T} W\) and updates the running
        diagonal inverse ``d_inv`` used downstream.

        Examples
        --------
        >>> import torch
        >>> layer = InnerWeightLipschitzLinearLayer(8, 8)
        >>> prev = FirstLipschitzLinearLayer(8, 8)
        >>> _, _, _, alpha, omega_r, constant = prev(torch.randn(2, 8))
        >>> out = layer(torch.randn(2, 8), alpha, omega_r, constant)
        >>> len(out)
        6
        """
        _, W, _ = self.scale_w()

        # with torch.no_grad():
        #     assert  torch.linalg.eigvalsh(W @ W.T).max() <= 1.0
        #     assert  torch.linalg.eigvalsh(W.T @ W).max() <= 1.0

        # assert torch.allclose(prev_layer.omega_r.T.tril(), prev_layer.omega_r.T), "This is not lower triangular"
        W = torch.linalg.solve_triangular(prev_omega_r, W, upper=False, left=False)

        # with torch.no_grad():
        #     W_temp = W @ prev_layer.omega_r.T
        #
        #     assert torch.allclose(W_temp, self.scale_w()[1])

        # with torch.no_grad():
        #     omega = prev_layer.omega_r.T @ prev_layer.omega_r
        #
        #     value = W @ omega @ W.T
        #
        #     assert torch.linalg.svdvals(value).abs().max() <= 1.0
        #     assert torch.linalg.eigvalsh(value).abs().max() <= 1.0

        scale = prev_alpha.sqrt() * prev_constant * self.constant
        div_scale = self.constant_sq * self.alpha

        # with torch.no_grad():
        #     square_scale = scale ** 2
        #
        #     div_scale_prev = prev_layer.constant_sq * prev_layer.alpha
        #
        #     value = square_scale / div_scale_prev
        #
        #     # tmep = torch.isclose(value, torch.full_like(value, 1.0))
        #     tmep2 = torch.isclose(value, torch.full_like(value, 2.0))
        #
        #     assert  tmep2, value

        W = scale * W

        omega = self.sum_weight()

        d_inv = omega / div_scale

        omega_r = torch.linalg.cholesky(omega, upper=False)

        # with torch.no_grad():
        #     assert torch.allclose(omega, self.omega_r.T @ self.omega_r)
        #     R_inv = torch.linalg.solve_triangular(self.omega_r, self.identity_out, upper=True)
        #
        #     inverse_val = R_inv.T @ omega @ R_inv
        #
        #     assert torch.allclose(self.identity_out, inverse_val)
        #
        #     scale = self.alpha.sqrt() * self.constant
        #
        #     temp =  scale * R_inv
        #     result = temp.T @ d_inv @ temp
        #
        #     assert torch.allclose(self.identity_out, result)

        # with torch.no_grad():
        #     M = torch.randn_like(self.identity)
        #
        #     c = self.constant
        #
        #     q, _ = torch.linalg.qr(M)
        #
        #     q *= c
        #
        #     q_2 = torch.linalg.solve_triangular(prev_layer.omega_r.T, q, upper=False, left=False)
        #
        #     omega = prev_layer.omega_r.T @ prev_layer.omega_r
        #
        #     inverse_val = q_2 @ omega @ q_2.T
        #
        #     assert torch.allclose(self.identity * self.constant_sq, inverse_val)

        return torch.nn.functional.linear(input, W, self.bias), W, d_inv, self.alpha, omega_r, self.constant


class LastLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    r"""Final layer combining accumulated sigma factors.

    Parameters
    ----------
    See :class:`AbstractLipschitzLinearLayer`.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Propagated when triangular solves or Cholesky decompositions fail.

    Notes
    -----
    Consumes the cumulative sigma factor ``sigma_r`` to project the final
    activations while preserving the Lipschitz budget.

    Mathematical Sketch
    -------------------
    Implements \(W_L = c_L \alpha_{L-1}^{1/2} \sigma^{-T} R_{L-1}^{-T} W\).

    Examples
    --------
    >>> import torch
    >>> first = FirstLipschitzLinearLayer(8, 8)
    >>> _, _, _, alpha, omega_r, constant = first(torch.randn(2, 8))
    >>> last = LastLipschitzLinearLayer(8, 8)
    >>> sigma_r = torch.linalg.cholesky(torch.eye(8))
    >>> logits, weights = last(torch.randn(2, 8), sigma_r, alpha, omega_r)
    >>> logits.shape
    torch.Size([2, 8])
    """

    def forward(self, input: Tensor, sigma_r: Tensor, prev_alpha: Tensor, prev_omega_r: Tensor) -> tuple[
        Tensor, Tensor]:
        r"""
        Apply the final linear transform using accumulated sigma factors.

        Parameters
        ----------
        input : Tensor
            Activation from the previous layer.
        sigma_r : Tensor
            Cholesky factor built from previous diagonals.
        prev_alpha : Tensor
            Relaxation of the preceding layer.
        prev_omega_r : Tensor
            Previous Cholesky factor.

        Returns
        -------
        output : Tensor
            Final linear projection.
        W : Tensor
            Normalized weights for certification routines.

        Raises
        ------
        RuntimeError
            Propagated if ``sigma_r`` is not positive-definite or triangular
            solves fail.

        Notes
        -----
        Reuses ``sigma_r`` to combine accumulated diagonal factors, mirroring
        the ``Last`` convolution layer logic.

        Mathematical Sketch
        -------------------
        Computes \(W_L = c_L \alpha_{L-1}^{1/2} \sigma_r^{-T} R_{L-1}^{-T} W\).

        Examples
        --------
        >>> import torch
        >>> first = FirstLipschitzLinearLayer(4, 4)
        >>> _, _, _, alpha, omega_r, constant = first(torch.randn(1, 4))
        >>> last = LastLipschitzLinearLayer(4, 4)
        >>> sigma_r = torch.linalg.cholesky(torch.eye(4))
        >>> logits, weights = last(torch.randn(1, 4), sigma_r, alpha, omega_r)
        >>> logits.shape
        torch.Size([1, 4])
        """
        _, W, _ = self.scale_w()

        prev_alpha = prev_alpha
        prev_r = prev_omega_r

        # assert torch.allclose(prev_r.triu(), prev_r), "This is not upper triangular"
        W = torch.linalg.solve_triangular(prev_r, W, upper=False, left=False)
        W = self.constant * prev_alpha.sqrt() * torch.linalg.solve_triangular(sigma_r, W, upper=True, left=True)

        return torch.nn.functional.linear(input, W, self.bias), W


class LinearLipschitz(AbstractLipschitzLinearLayer):
    """Sequential linear block with optional activation.

    Parameters
    ----------
    in_features, out_features : int
        Projection sizes.
    activation : callable, optional
        Post-linear activation applied in forward.
    bias, constant_sq, enable_alpha, device, dtype :
        Forwarded to :class:`AbstractLipschitzLinearLayer`.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Propagated when base-class initialization fails.

    Notes
    -----
    Stores ``sqrt_2`` buffer for historical experiments (retained for
    completeness) and optionally applies ``activation``.

    Mathematical Sketch
    -------------------
    Shares the same Lipschitz normalization as the base class while optionally
    composing with an activation that must be 1-Lipschitz (e.g., ReLU).

    Examples
    --------
    >>> import torch
    >>> block = LinearLipschitz(8, 8, activation=torch.nn.ReLU())
    >>> out, _, _ = block(torch.randn(2, 8), torch.eye(8), torch.ones(1), torch.tensor(1.0))
    >>> out.shape
    torch.Size([2, 8])
    """
    sqrt_2: Tensor

    def __init__(self, in_features: int, out_features: int,
                 activation: Optional[Callable[[Tensor], Tensor]] = None, bias: bool = True,
                 constant_sq: float = 1.0,
                 enable_alpha: bool = True,
                 device: Optional[torch.device | str] = None, dtype: Optional[torch.dtype] = None) -> None:
        """Configure the sequential linear block.

        Parameters
        ----------
        in_features, out_features : int
            Linear projection dimensions.
        activation : callable, optional
            Activation applied after the linear map.
        bias, constant_sq, enable_alpha, device, dtype :
            Passed directly to :class:`AbstractLipschitzLinearLayer`.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated if base-class initialization fails.

        Notes
        -----
        Stores ``sqrt_2`` for compatibility with older experiments; it does not
        alter current computations but remains as a buffer for reproducibility.

        Examples
        --------
        >>> block = LinearLipschitz(4, 4)
        >>> isinstance(block.activation, type(None))
        True
        """
        super().__init__(in_features, out_features, bias, constant_sq, enable_alpha, device, dtype)
        self.register_buffer('sqrt_2', torch.tensor(2, device=device, dtype=dtype).sqrt())

        self.activation = activation

    def forward(self, input: Tensor, prev_omega_r: Tensor, prev_alpha: Tensor, pre_constant: Tensor) -> tuple[
        Tensor, Tensor, Tensor]:
        r"""
        Apply a sequential linear transform and optional activation.

        Parameters
        ----------
        input : Tensor
            Activations entering the block.
        prev_omega_r : Tensor
            Lower Cholesky factor from the previous block.
        prev_alpha : Tensor
            Previous relaxation parameter.
        pre_constant : Tensor
            Previous Lipschitz constant.

        Returns
        -------
        output : Tensor
            Projected (and optionally activated) tensor.
        omega_r : Tensor
            New lower Cholesky factor.
        W : Tensor
            Normalized weight matrix.

        Raises
        ------
        RuntimeError
            Propagated if triangular solves or Cholesky decompositions fail.

        Notes
        -----
        Equivalent to :class:`InnerWeightLipschitzLinearLayer` followed by an
        optional activation function.

        Mathematical Sketch
        -------------------
        Applies \(W = c_{i-1} c_i R_{i-1}^{-T} W_i\) before evaluating the
        optional activation ``φ``.

        Examples
        --------
        >>> import torch
        >>> block = LinearLipschitz(4, 4, activation=torch.nn.ReLU())
        >>> out, omega_r, W = block(torch.randn(2, 4), torch.eye(4), torch.ones(1), torch.tensor(1.0))
        >>> out.shape
        torch.Size([2, 4])
        """
        _, W, _ = self.scale_w()

        W = torch.linalg.solve_triangular(prev_omega_r, W, upper=False, left=False)

        scale = prev_alpha.sqrt() * pre_constant * self.constant

        W = scale * W

        omega = self.sum_weight()

        omega_r = torch.linalg.cholesky(omega, upper=False)

        out = torch.nn.functional.linear(input, W, self.bias)

        if self.activation:
            out = self.activation(out)

        return out, omega_r, W
