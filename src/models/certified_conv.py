"""Circular residual operators using the same version-2 certificate per frequency.

Real spatial free kernels give conjugate-symmetric frequency symbols. The
normalized symbols are applied directly, never cropped back to a small kernel.
Identity activation multipliers are shared over all pixels and frequencies.
"""
import torch
from torch import nn, Tensor
from models.certified_ldl import residual_weights
from models.certificate_version import CertificateVersion


def symbol(kernel: Tensor, size: tuple[int, int]) -> Tensor:
    """Return (...,out,in) symbols, with leading (height,width) axes."""
    return torch.fft.fft2(kernel, s=size).permute(2, 3, 0, 1)


def circular_apply(x: Tensor, weight: Tensor) -> Tensor:
    """Apply a full-grid symbol via unitary FFTs to a real spatial tensor."""
    spectrum = torch.fft.fft2(x, norm='ortho')
    y = torch.einsum('hwoc,bchw->bohw', weight, spectrum)
    return torch.fft.ifft2(y, norm='ortho').real


class CertifiedCircularResidual(CertificateVersion, nn.Module):
    def __init__(self, channels: int, hidden: int | None = None, depth: int = 2,
                 kernel_size: int = 3, dtype=None, device=None):
        super().__init__()
        if channels <= 0 or depth < 1 or kernel_size < 1:
            raise ValueError('Positive dimensions and at least one hidden layer are required')
        hidden = hidden or channels
        factory = dict(dtype=dtype, device=device)
        self.register_buffer('_parameterization_version', torch.tensor(2, dtype=torch.int64))
        self.raw_a = nn.Parameter(torch.randn(channels, channels, 1, 1, **factory) * .1)
        dims = [channels] + [hidden] * depth
        self.raw_cs = nn.ParameterList([
            nn.Parameter(torch.randn(y, x, kernel_size, kernel_size, **factory)
                         / (x * kernel_size**2)**.5 * .2)
            for x, y in zip(dims, dims[1:])])
        self.raw_b = nn.Parameter(torch.randn(channels, hidden, 1, 1, **factory) * .1)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(hidden, **factory)) for _ in range(depth)])
        self.bias = nn.Parameter(torch.zeros(channels, **factory))
        self._cached = None

    def train(self, mode=True):
        self._cached = None
        return super().train(mode)

    def normalized_weights(self, size):
        if min(size) < self.raw_cs[0].shape[-1]:
            raise ValueError('Grid must be at least as large as the free kernel')
        return residual_weights(symbol(self.raw_a, size), [symbol(w, size) for w in self.raw_cs],
                                symbol(self.raw_b, size))

    def forward(self, x):
        key = (tuple(x.shape[-2:]), x.device, x.dtype,
               tuple(p._version for p in self.parameters()))
        if not self.training and not torch.is_grad_enabled() and self._cached is not None and self._cached[0] == key:
            weights = self._cached[1]
        else:
            weights = self.normalized_weights(tuple(x.shape[-2:]))
            if not self.training and not torch.is_grad_enabled():
                self._cached = (key, weights)
        h = x
        for c, bias in zip(weights.cs, self.biases):
            h = torch.relu(circular_apply(h, c) + bias[None, :, None, None])
        return circular_apply(x, weights.a) + circular_apply(h, weights.b) + self.bias[None, :, None, None]
