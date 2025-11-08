r"""
Convolutional building blocks that maintain explicit Lipschitz constants.

Parameters
----------
None

Returns
-------
None

Raises
------
RuntimeError
    Propagated from PyTorch FFT/Cholesky routines when tensors are invalid.

Notes
-----
Provides FFT-domain utilities and abstract layers used by both residual and
sequential Lipschitz CNNs.

Mathematical Sketch
-------------------
Weight normalization is implemented via frequency-domain Cholesky factors so
that each convolution satisfies \(\|W\|_2 \leq L\).

Examples
--------
>>> import torch
>>> from models.cnn_layers import FirstLipschitzConvLayer
>>> layer = FirstLipschitzConvLayer(3, 3)
>>> layer(torch.randn(1, 3, 8, 8))[0].shape
torch.Size([1, 3, 8, 8])
"""

import math
from abc import abstractmethod, ABC
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Parameter, init
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.nn.utils import parametrize

from utils import PositiveExp


def _center_and_pad_kernel(kernel: Tensor, H: int, W: int) -> Tensor:
    r"""
    Zero-pad and center a convolution kernel for FFT processing.

    Parameters
    ----------
    kernel : (out_ch, in_ch, kH, kW) tensor
        Spatial convolution weights.
    H, W : int
        Target padded height/width.

    Returns
    -------
    padded : Tensor
        Kernel padded to ``(out_ch, in_ch, H, W)`` and circularly shifted so that
        the impulse is centered.

    Raises
    ------
    RuntimeError
        Propagated from :func:`torch.nn.functional.pad` when target dimensions
        are smaller than the kernel.

    Notes
    -----
        Complexity is \(O(out\_ch \cdot in\_ch \cdot H \cdot W)\) due to tensor copying.

    Mathematical Sketch
    -------------------
    Pads the kernel with zeros and applies a roll so that FFTs treat the impulse
    response as centered.

    Examples
    --------
    >>> import torch
    >>> K = torch.ones(2, 2, 3, 3)
    >>> _center_and_pad_kernel(K, 4, 4).shape
    torch.Size([2, 2, 4, 4])
    """
    out_ch, in_ch, kH, kW = kernel.shape
    padded = F.pad(kernel, (0, W - kW, 0, H - kH))
    return torch.roll(padded, shifts=(- (kH // 2), - (kW // 2)), dims=(-2, -1))


def _unroll_and_crop_spatial(pad_spatial: Tensor, kH: int, kW: int) -> Tensor:
    """
    Undo the padding/rolling performed for FFT convolution.

    Parameters
    ----------
    pad_spatial : Tensor
        Spatial domain tensor after inverse FFT.
    kH, kW : int
        Desired kernel height/width.

    Returns
    -------
    cropped : Tensor
        Tensor with shape ``(..., kH, kW)`` representing the unshifted kernel.

    Raises
    ------
    RuntimeError
        Propagated if tensor slicing is invalid.

    Notes
    -----
    Complements :func:`_center_and_pad_kernel` by reversing the spatial shift.

    Mathematical Sketch
    -------------------
    Applies the inverse roll (shift) and crops to the original kernel size.

    Examples
    --------
    >>> import torch
    >>> pad = torch.zeros(2, 2, 4, 4)
    >>> _unroll_and_crop_spatial(pad, 3, 3).shape
    torch.Size([2, 2, 3, 3])
    """
    shift_h = (kH // 2)
    shift_w = (kW // 2)
    pad_spatial = torch.roll(pad_spatial, shifts=(shift_h, shift_w), dims=(-2, -1))
    return pad_spatial[..., :kH, :kW]


def _freq_batchify(K_fft: Tensor) -> Tuple[Tensor, int, int]:
    """
    Flatten frequency bins into a batch dimension.

    Parameters
    ----------
    K_fft : (out_ch, in_ch, H, Wf) tensor
        Complex-valued FFT of a convolution kernel (real-imag stacked).

    Returns
    -------
    K_flat : (F, out_ch, in_ch) tensor
        View where each spatial frequency is treated as a batch item.
    H : int
        Number of spatial rows.
    Wf : int
        Number of frequency columns (``W // 2 + 1`` for real FFT).

    Raises
    ------
    RuntimeError
        Propagated from tensor reshapes if dimensions mismatch.

    Notes
    -----
    Enables treating each frequency bin independently during normalization.

    Mathematical Sketch
    -------------------
    Reshapes the FFT tensor so that the 2-D frequency grid becomes a batch
    dimension of size ``F = H * Wf``.

    Examples
    --------
    >>> import torch
    >>> K_fft = torch.randn(2, 2, 4, 3)
    >>> flat, H, Wf = _freq_batchify(K_fft)
    >>> flat.shape
    torch.Size([12, 2, 2])
    """
    # input K_fft: (out, in, H, Wf) complex -> returns (F, out, in)
    out_ch, in_ch, H, Wf = K_fft.shape
    F = H * Wf
    # K_flat = K_fft.reshape(out_ch, in_ch, F).permute(2, 0, 1)
    K_flat = K_fft.permute(2, 3, 0, 1).reshape(F, out_ch, in_ch)
    return K_flat, H, Wf


def _unbatchify_freq(M_flat: Tensor, H: int, Wf: int) -> Tensor:
    """
    Reverse :func:`_freq_batchify` by restoring the frequency grid shape.

    Parameters
    ----------
    M_flat : (F, out_ch, in_ch) tensor
        Flattened frequency representation.
    H, Wf : int
        Target spatial and frequency dimensions.

    Returns
    -------
    tensor : (out_ch, in_ch, H, Wf) tensor
        Reshaped frequency-domain tensor.

    Raises
    ------
    RuntimeError
        Propagated if the flat tensor size is incompatible with ``H``/``Wf``.

    Notes
    -----
    Inverse operation of :func:`_freq_batchify`.

    Mathematical Sketch
    -------------------
    Reshapes the batchified frequencies back into a 2-D grid before IFFT.

    Examples
    --------
    >>> import torch
    >>> flat = torch.randn(12, 2, 2)
    >>> _unbatchify_freq(flat, 4, 3).shape
    torch.Size([2, 2, 4, 3])
    """
    F, out_ch, in_ch = M_flat.shape
    # assert F == H * Wf
    # M =  M_flat.permute(1, 2, 0).reshape(out_ch, in_ch, H, Wf)
    M = M_flat.reshape(H, Wf, out_ch, in_ch).permute(2, 3, 0, 1)
    return M  # (out, in, H, Wf)


def make_real(M: Tensor) -> Tensor:
    """
    Convert a complex-valued matrix into its doubled real representation.

    Parameters
    ----------
    M : Tensor
        Complex tensor whose last two dimensions encode output/input channels.

    Returns
    -------
    Tensor
        Real-valued block matrix with doubled channel dimensions so that
        real-only routines can manipulate complex weights.

    Raises
    ------
    RuntimeError
        Propagated when tensor concatenations fail due to mismatched dims.

    Notes
    -----
    Implements the standard real-imaginary block embedding.

    Mathematical Sketch
    -------------------
    Builds the block matrix ``[[Re, -Im], [Im, Re]]`` to represent complex
    multiplication using real-valued algebra.

    Examples
    --------
    >>> import torch
    >>> M = torch.randn(2, 2, dtype=torch.cfloat)
    >>> make_real(M).shape
    torch.Size([4, 4])
    """
    # return M
    X, Y = M.real, M.imag
    top = torch.cat([X, -Y], dim=-1)  # (..., out, 2*in)
    bot = torch.cat([Y, X], dim=-1)  # (..., out, 2*in)
    return torch.cat([top, bot], dim=-2)  # (..., 2*out, 2*in)


def make_complex(M: Tensor) -> Tensor:
    """
    Reverse :func:`make_real` by reconstructing the complex tensor.

    Parameters
    ----------
    M : Tensor
        Real block matrix with doubled channel dimensions.

    Returns
    -------
    Tensor
        Complex tensor where real/imaginary parts are reconstructed from blocks.

    Raises
    ------
    RuntimeError
        Propagated when slicing yields mismatched shapes.

    Notes
    -----
    Inverse of :func:`make_real`.

    Mathematical Sketch
    -------------------
    Reconstructs the complex tensor from the block matrix by treating the top
    block as the real part and the bottom block as the imaginary part.

    Examples
    --------
    >>> import torch
    >>> M = torch.randn(4, 4)
    >>> make_complex(M).dtype == torch.cfloat
    True
    """
    # return M
    out2, in2 = M.shape[-2:]
    out, inn = out2 // 2, in2 // 2
    X = M[..., :out, :inn]  # Re
    Y = M[..., out:, :inn]  # Im
    return torch.complex(X, Y)


class AbstractLipschitzConvLayer(nn.Conv2d, ABC):
    r"""Base class for convolutional layers with explicit Lipschitz control.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel counts.
    kernel_size, stride, padding, dilation, groups : see :class:`nn.Conv2d`.
    constant_sq : float, default=1.0
        Target squared Lipschitz constant.
    enable_alpha : bool, default=True
        Whether to make the relaxation parameter trainable.
    activation : nn.Module, optional
        Optional post-convolution activation.
    device, dtype : Any, optional
        Tensor factory kwargs.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Propagated from :class:`nn.Conv2d` during initialization when arguments
        are invalid.

    Notes
    -----
    Operates in the frequency domain to bound the induced operator norm.

    Mathematical Sketch
    -------------------
    Weights are normalized by computing frequency-domain Gram matrices and
    solving triangular systems to ensure \(\|W\|_2 \leq \sqrt{\text{constant\_sq}}\).

    Examples
    --------
    >>> import torch
    >>> layer = FirstLipschitzConvLayer(3, 3)
    >>> layer(torch.randn(1, 3, 8, 8))[0].shape
    torch.Size([1, 3, 8, 8])
    """
    identity: torch.Tensor
    identity_out: torch.Tensor
    constant_sq: torch.Tensor
    constant: torch.Tensor
    R: torch.Tensor
    omega_r: torch.Tensor

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 3,
                 constant_sq: float = 1.0,
                 enable_alpha: bool = True,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 activation: Optional[nn.Module] = None,
                 device: Optional[torch.device | str] = None,
                 dtype: Optional[torch.dtype] = None, ):
        """Initialize a convolutional layer with Lipschitz-aware parameterization.

        Parameters
        ----------
        Same as the class docstring.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated from :class:`nn.Conv2d` when invalid arguments are given.

        Notes
        -----
        Registers identity buffers plus ``constant`` and ``alpha`` for later use.

        Examples
        --------
        >>> AbstractLipschitzConvLayer(3, 3)  # doctest: +SKIP
        """
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode,
                         device=device,
                         dtype=dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.activation = activation
        self.register_buffer(
            "identity", torch.eye(in_channels, **factory_kwargs)
        )

        self.register_buffer(
            "identity_out", torch.eye(out_channels, **factory_kwargs)
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
        """Initialize weights/biases with Kaiming-uniform defaults.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated when PyTorch initialization helpers encounter invalid
            tensors.

        Notes
        -----
        Mirrors :meth:`nn.Conv2d.reset_parameters` while ensuring the bias range
        matches the selected fan-in.

        Mathematical Sketch
        -------------------
        Uses He initialization with ``a = sqrt(5)`` consistent with ReLU layers.

        Examples
        --------
        >>> layer = FirstLipschitzConvLayer(3, 3)
        >>> layer.reset_parameters()
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def sum_weight(self, Wf: Tensor, transpose: bool = False) -> Tensor:
        r"""
        Accumulate Gram matrices in frequency space.

        Parameters
        ----------
        Wf : Tensor
            Frequency-domain weights shaped ``(F, out, in)``.
        transpose : bool, default=False
            When ``True`` computes \\( W^T W + \\alpha I \\); otherwise computes
            \\( W W^T + \\alpha I \\).

        Returns
        -------
        Tensor
            Symmetric positive-definite matrix used for scaling.

        Raises
        ------
        RuntimeError
            Propagated when matrix multiplications encounter mismatched shapes.

        Notes
        -----
        Adds ``alpha`` to guarantee positive-definiteness, enabling stable
        Cholesky factors.

        Mathematical Sketch
        -------------------
        Forms \(W^\top W + \alpha I\) or \(W W^\top + \alpha I\) depending on
        ``transpose``.

        Examples
        --------
        >>> layer = FirstLipschitzConvLayer(3, 3)
        >>> Wf, _, _ = layer.spectral_norm_frequency(8, 8)
        >>> gram = layer.sum_weight(Wf)
        >>> gram.shape[-2:] == (3, 3)
        True
        """
        if transpose:
            # m = torch.einsum('fim,fin->fmn', Wf, Wf)
            # n = torch.matmul(Wf.mT, Wf)
            # assert torch.allclose(m, n)

            return self.alpha * self.identity + torch.matmul(Wf.mT, Wf)
        else:
            # m = torch.einsum('fni,fmi->fmn', Wf, Wf)
            # n = torch.matmul(Wf, Wf.mT)
            # assert torch.allclose(m, n)

            # return self.alpha * self.identity_out + m
            return self.alpha * self.identity_out + torch.matmul(Wf, Wf.mT)

    def scale_w(self, Wf: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Normalize weights so the spectral norm meets the Lipschitz target.

        Parameters
        ----------
        Wf : Tensor
            Frequency-domain weights.

        Returns
        -------
        omega : Tensor
            Gram matrix before scaling.
        W : Tensor
            Scaled weights with bounded norm.
        R : Tensor
            Upper-triangular Cholesky factor of ``omega``.

        Raises
        ------
        RuntimeError
            Propagated if Cholesky or triangular solves fail.

        Notes
        -----
        Equivalent to solving ``R R^T = omega`` and normalizing ``W`` via
        ``R``.

        Mathematical Sketch
        -------------------
        Ensures \(\|W\|_2 \leq constant\) by whitening frequency-domain weights.

        Examples
        --------
        >>> layer = FirstLipschitzConvLayer(3, 3)
        >>> Wf, _, _ = layer.spectral_norm_frequency(8, 8)
        >>> omega, W, R = layer.scale_w(Wf)
        >>> omega.shape
        torch.Size([3, 3])
        """
        omega = self.sum_weight(Wf, transpose=True)
        R = torch.linalg.cholesky(omega, upper=True)

        W = torch.linalg.solve_triangular(R, Wf, upper=True, left=False)

        return omega, W, R

    def _weight_fft_for_input(self, H: int, W: int) -> Tensor:
        """
        Compute the FFT of the convolution weights for a given input size.

        Parameters
        ----------
        H, W : int
            Height/width of the incoming feature map.

        Returns
        -------
        Tensor
            Real-stacked FFT tensor ready for spectral scaling.

        Raises
        ------
        RuntimeError
            Propagated from FFT routines when dimensions are invalid.

        Notes
        -----
        Uses :func:`torch.fft.rfft2` to obtain the half-spectrum representation.

        Mathematical Sketch
        -------------------
        Pads/centers the kernel before computing the FFT to align impulse
        responses properly.

        Examples
        --------
        >>> layer = FirstLipschitzConvLayer(3, 3)
        >>> layer._weight_fft_for_input(8, 8).shape[-2:]
        (8, 5)
        """
        pad = _center_and_pad_kernel(self.weight, H, W)
        K_fft = torch.fft.rfft2(pad, s=(H, W), norm="backward")

        K_fft_expanded_real = make_real(K_fft)

        return K_fft_expanded_real

    def _ifft_from_freq(self, M_freq: Tensor, H: int, W: int) -> Tensor:
        """
        Convert frequency-domain matrices back to spatial kernels.

        Parameters
        ----------
        M_freq : Tensor
            Frequency-domain tensor shaped like ``(out, in, H, Wf)``.
        H, W : int
            Target spatial resolution for the inverse FFT.

        Returns
        -------
        Tensor
            Spatial-domain convolution kernel.

        Raises
        ------
        RuntimeError
            Propagated from :func:`torch.fft.irfft2` when shapes are invalid.

        Notes
        -----
        Converts real-stacked frequency data back into complex form before
        applying the inverse FFT.

        Mathematical Sketch
        -------------------
        Implements inverse of :meth:`_weight_fft_for_input`.

        Examples
        --------
        >>> layer = FirstLipschitzConvLayer(3, 3)
        >>> Wf = layer._weight_fft_for_input(8, 8)
        >>> layer._ifft_from_freq(_unbatchify_freq(Wf, 8, 5), 8, 8).shape
        torch.Size([3, 3, 8, 8])
        """
        M_freq_complex = make_complex(M_freq)

        pad_spatial = torch.fft.irfft2(M_freq_complex, s=(H, W), norm="backward")
        return pad_spatial

    def conv_with_normalized_weight(self, input: Tensor, W_spatial: Tensor, bias: Optional[Tensor]) -> Tensor:
        """
        Perform a 2-D convolution using the normalized spatial weights.

        Parameters
        ----------
        input : (B, C_in, H, W) tensor
            Feature map to convolve.
        W_spatial : Tensor
            Spatial kernel returned by :meth:`convert_to_spatial`.
        bias : Tensor, optional
            Bias broadcast across spatial locations.

        Returns
        -------
        Tensor
            Convolved output matching :func:`F.conv2d` semantics.

        Raises
        ------
        RuntimeError
            Propagated when :func:`torch.nn.functional.conv2d` encounters invalid
            shapes.

        Notes
        -----
        Handles nonzero padding modes by delegating to :func:`F.pad` manually.

        Mathematical Sketch
        -------------------
        Applies the normalized spatial kernel using standard convolution.

        Examples
        --------
        >>> layer = FirstLipschitzConvLayer(3, 3)
        >>> Wf, H_fft, Wf_fft = layer.spectral_norm_frequency(8, 8)
        >>> _, W_norm, _ = layer.scale_w(Wf)
        >>> spatial = layer.convert_to_spatial(W_norm, H_fft, Wf_fft, 8)
        >>> layer.conv_with_normalized_weight(torch.randn(1, 3, 8, 8), spatial, None).shape
        torch.Size([1, 3, 8, 8])
        """
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                W_spatial,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(input, W_spatial, bias, self.stride, self.padding, self.dilation, self.groups)

    def spectral_norm_frequency(self, H: int, W: int) -> Tuple[Tensor, int, int]:
        """
        Prepare frequency-domain weights for the current input size.

        Parameters
        ----------
        H, W : int
            Spatial height/width of the input tensor.

        Returns
        -------
        Wf : Tensor
            Flattened FFT weights.
        H_ : int
            Number of rows in the frequency grid.
        Wf_ : int
            Number of columns (half-spectrum) in the frequency grid.

        Raises
        ------
        RuntimeError
            Propagated from FFT helpers when shapes are invalid.

        Notes
        -----
        Combines :meth:`_weight_fft_for_input` and :func:`_freq_batchify`.

        Mathematical Sketch
        -------------------
        Returns the half-spectrum representation needed for spectral scaling.

        Examples
        --------
        >>> layer = FirstLipschitzConvLayer(3, 3)
        >>> Wf, H, Wf_ = layer.spectral_norm_frequency(8, 8)
        >>> Wf.shape[0] == H * Wf_
        True
        """
        K_fft = self._weight_fft_for_input(H, W)  # (out,in,H,Wf)
        Wf, H_, Wf_ = _freq_batchify(K_fft)  # (F,out,in)

        return Wf, H_, Wf_

    def convert_to_spatial(self, Wnorm_f: Tensor, H_: int, Wf_: int, W: int) -> Tensor:
        """
        Convert normalized frequency weights back to cropped spatial kernels.

        Parameters
        ----------
        Wnorm_f : Tensor
            Frequency-domain normalized weights.
        H_, Wf_ : int
            Frequency grid dimensions.
        W : int
            Target spatial width (the height is already encoded in ``H_``).

        Returns
        -------
        Tensor
            Spatial kernel cropped to ``kernel_size`` suitable for `conv2d`.

        Raises
        ------
        RuntimeError
            Propagated if tensor reshaping fails.

        Notes
        -----
        Applies inverse FFT and cropping to translate normalized frequencies
        back to spatial kernels.

        Mathematical Sketch
        -------------------
        Inverse of :func:`_freq_batchify` followed by :func:`_unroll_and_crop_spatial`.

        Examples
        --------
        >>> layer = FirstLipschitzConvLayer(3, 3)
        >>> Wf, H_fft, Wf_fft = layer.spectral_norm_frequency(8, 8)
        >>> _, W_norm, _ = layer.scale_w(Wf)
        >>> layer.convert_to_spatial(W_norm, H_fft, Wf_fft, 8).shape
        torch.Size([3, 3, 3, 3])
        """
        M_freq = _unbatchify_freq(Wnorm_f, H_, Wf_)
        pad_spatial = self._ifft_from_freq(M_freq, H_, W)
        kH, kW = _pair(self.kernel_size)
        return _unroll_and_crop_spatial(pad_spatial, kH, kW)

    @abstractmethod
    def process(self, Wf, *args, **kwargs):
        """
        Transform frequency-domain weights according to the layer type.

        Parameters
        ----------
        Wf : Tensor
            Frequency-domain weights.
        *args, **kwargs :
            Additional tensors required by subclasses (e.g., previous factors).

        Returns
        -------
        tuple
            Implementation-defined tensors such as normalized weights and
            Cholesky factors.

        Raises
        ------
        NotImplementedError
            Must be provided by subclasses.

        Examples
        --------
        >>> AbstractLipschitzConvLayer.process  # doctest: +SKIP
        """
        raise NotImplementedError

    def circular_convolution(self, input: Tensor, H: int, W: int, Wf_norm: Tensor,
                             H_fft: int, Wf_fft: int) -> Tensor:
        r"""
        Apply a circular convolution via FFT multiplication.

        Parameters
        ----------
        input : (B, C_in, H, W) tensor
            Feature map to convolve.
        H, W : int
            Spatial resolution (used for FFT sizing).
        Wf_norm : Tensor
            Frequency-domain normalized weights.
        H_fft, Wf_fft : int
            Frequency grid sizes returned by :meth:`spectral_norm_frequency`.

        Returns
        -------
        Tensor
            Convolved activation map in spatial domain with bias added if set.

        Raises
        ------
        RuntimeError
            Propagated from FFT/inverse FFT routines.

        Notes
        -----
        Performs circular convolution via element-wise multiplication in the
        frequency domain.

        Mathematical Sketch
        -------------------
        Applies \(Y(f) = W(f) X(f)\) followed by inverse FFT.

        Examples
        --------
        >>> layer = FirstLipschitzConvLayer(3, 3)
        >>> Wf, H_fft, Wf_fft = layer.spectral_norm_frequency(8, 8)
        >>> _, W_norm, _ = layer.scale_w(Wf)
        >>> layer.circular_convolution(torch.randn(1, 3, 8, 8), 8, 8, W_norm, H_fft, Wf_fft).shape
        torch.Size([1, 3, 8, 8])
        """
        # FFT of the input tensor
        X_fft = torch.fft.rfft2(input, s=(H, W), norm="backward")
        X_fft = make_real(X_fft)

        X_fft, _, _ = _freq_batchify(X_fft)

        # Y_fft_einsum = torch.einsum('foc,fbc->fbo', Wf_norm, X_fft)
        Y_fft_einsum = torch.matmul(X_fft, Wf_norm.mT)

        Y_fft = _unbatchify_freq(Y_fft_einsum, H_fft, Wf_fft)

        output = self._ifft_from_freq(Y_fft, H, W)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output

    def forward(self, input: Tensor, *args) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Run the convolution while returning intermediate scaling quantities.

        Parameters
        ----------
        input : (B, C_in, H, W) tensor
            Activation batch to convolve.
        *args :
            Additional tensors consumed by subclasses (e.g., previous factors).

        Returns
        -------
        output : Tensor
            Convolved activations (after optional activation).
        Wf_norm : Tensor
            Normalized frequency weights.
        d_inv : Tensor
            Diagonal inverse used for cumulative sigma computation.
        alpha : Tensor
            Current relaxation parameter.
        omega_r : Tensor
            Cholesky factor of the Gram matrix.
        constant : Tensor
            Scalar Lipschitz constant associated with this layer.

        Raises
        ------
        RuntimeError
            Propagated when FFT, Cholesky, or convolution operations fail.

        Notes
        -----
        Subclasses supply ``*args`` to incorporate prior normalization factors.

        Mathematical Sketch
        -------------------
        Normalizes ``W`` via :meth:`process`, performs circular convolution, and
        applies the optional activation.

        Examples
        --------
        >>> layer = FirstLipschitzConvLayer(3, 3)
        >>> out = layer(torch.randn(1, 3, 8, 8))
        >>> out[0].shape
        torch.Size([1, 3, 8, 8])
        """
        B, C_in, H, W = input.shape

        Wf_orig, H_fft, Wf_fft = self.spectral_norm_frequency(H, W)

        Wf_norm, d_inv, omega_r = self.process(Wf_orig, *args)

        output = self.circular_convolution(input, H, W, Wf_norm, H_fft, Wf_fft)

        if self.activation:
            output = self.activation(output)

        return output, Wf_norm, d_inv, self.alpha, omega_r, self.constant


class FirstLipschitzConvLayer(AbstractLipschitzConvLayer):
    """
    Entry convolution layer that scales weights without referencing predecessors.
    """

    def process(self, Wf: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Normalize the first-layer weights.

        Parameters
        ----------
        Wf : Tensor
            Frequency-domain weights for the layer.

        Returns
        -------
        tuple
            Normalized weights, diagonal inverse, and lower Cholesky factor.
        """
        omega, W, R = self.scale_w(Wf)

        W = self.constant * W

        d_inv = omega / (self.constant_sq * self.alpha)

        omega_r = torch.linalg.cholesky(omega, upper=False)

        return W, d_inv, omega_r


class InnerWeightLipschitzConvLayer(AbstractLipschitzConvLayer):
    """
    Hidden convolution layer that depends on the previous layer's scaling.
    """

    def process(self, Wf: Tensor, prev_alpha: Tensor, prev_omega_r: Tensor, prev_constant: Tensor) -> \
            Tuple[Tensor, Tensor, Tensor]:
        """
        Normalize weights using the previous layer's factors.

        Parameters
        ----------
        Wf : Tensor
            Frequency-domain weights.
        prev_alpha : Tensor
            Relaxation parameter of the preceding layer.
        prev_omega_r : Tensor
            Lower Cholesky factor from the preceding layer.
        prev_constant : Tensor
            Scalar Lipschitz constant of the preceding layer.

        Returns
        -------
        tuple
            Normalized weights, diagonal inverse, and Cholesky factor.
        """
        _, W, _ = self.scale_w(Wf)

        W = torch.linalg.solve_triangular(prev_omega_r, W, upper=False, left=False)

        scale = prev_alpha.sqrt() * prev_constant * self.constant
        div_scale = self.constant_sq * self.alpha

        W = scale * W

        omega = self.sum_weight(Wf)

        d_inv = omega / div_scale

        omega_r = torch.linalg.cholesky(omega, upper=False)

        return W, d_inv, omega_r


class LastLipschitzConvLayer(AbstractLipschitzConvLayer):
    """
    Final convolution layer that aggregates the sigma factors.
    """

    def process(self, Wf: Tensor, sigma_r: Tensor, prev_r_sqrt: torch.Tensor, prev_alpha: torch.Tensor):
        """
        Normalize the last layer using accumulated sigma factors.

        Parameters
        ----------
        Wf : Tensor
            Frequency-domain weights.
        sigma_r : Tensor
            Cholesky factor of the accumulated sigma matrix.
        prev_r_sqrt : Tensor
            Previous triangular factor.
        prev_alpha : Tensor
            Relaxation from the preceding layer.

        Returns
        -------
        Tensor
            Normalized frequency-domain weights.
        """
        _, W, _ = self.scale_w(Wf)

        W = torch.linalg.solve_triangular(prev_r_sqrt, W, upper=False, left=False)
        W = self.constant * prev_alpha.sqrt() * torch.linalg.solve_triangular(sigma_r, W, upper=True, left=True)

        return W

    def forward(self, input: Tensor, sigma_r: Tensor, prev_alpha: Tensor, prev_omega_r: Tensor) -> Tuple[
        Tensor, Tensor]:
        """
        Apply the last layer convolution.

        Parameters
        ----------
        input : Tensor
            Activation map from the previous layer.
        sigma_r : Tensor
            Cholesky factor of the sigma matrix.
        prev_alpha : Tensor
            Relaxation parameter from the previous layer.
        prev_omega_r : Tensor
            Previous lower Cholesky factor.

        Returns
        -------
        output : Tensor
            Convolved activation map.
        Wf_norm : Tensor
            Normalized weights used for future metrics.
        """
        B, C_in, H, W = input.shape

        Wf_orig, H_fft, Wf_fft = self.spectral_norm_frequency(H, W)

        prev_alpha = prev_alpha
        prev_r = prev_omega_r

        Wf_norm = self.process(Wf_orig, sigma_r, prev_r, prev_alpha)

        output = self.circular_convolution(input, H, W, Wf_norm, H_fft, Wf_fft)

        return output, Wf_norm


class FirstSequentialLipschitzConvLayer(AbstractLipschitzConvLayer):
    """
    First layer used in the sequential variant (no residual blocks).
    """

    def process(self, Wf: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Normalize the first sequential convolution.

        Parameters
        ----------
        Wf : Tensor
            Frequency-domain weights.

        Returns
        -------
        tuple
            Normalized weights, diagonal inverse, and Cholesky factor.
        """
        _, W, _ = self.scale_w(Wf)

        W = self.constant * W

        omega = self.sum_weight(Wf)

        d_inv = omega / (self.constant_sq * self.alpha)

        omega_r = torch.linalg.cholesky(omega, upper=False)

        return W, d_inv, omega_r
