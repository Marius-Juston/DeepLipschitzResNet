"""Deterministic classification certificates in the declared input coordinates."""
import torch
from torch import Tensor


def clear_weight_caches(model):
    """Discard derived baseline weights after optimizer steps/checkpoint loads."""
    for module in model.modules():
        for name in ('Q', 'Qfft'):
            value = getattr(module, name, None)
            if isinstance(value, Tensor) and not isinstance(value, torch.nn.Parameter):
                setattr(module, name, None)


@torch.no_grad()
def audit_dense_certificate(model):
    """Check effective weights in float64 against the original dense LMI.

    This is a numerical diagnostic, not an interval-arithmetic proof. It checks
    the float32 weights actually produced, rather than a different float64
    parameterization of the same raw parameters.
    """
    from models.certified_ldl import assemble_lmi
    w = model.normalized_weights()
    a, cs, b = w.a.detach().cpu().double(), [c.detach().cpu().double() for c in w.cs], w.b.detach().cpu().double()
    matrix = assemble_lmi(a, cs, b, bound=model.lipschitz_constant)
    eig = torch.linalg.eigvalsh(matrix)
    minimum, scale = float(eig[0]), max(1., float(eig.abs().max()))
    if minimum <= 0:
        raise RuntimeError(f'Effective weights have no positive numerical certificate margin: {minimum:g}')
    return dict(min_eigenvalue=minimum, scaled_min_eigenvalue=minimum / scale,
                dimension=matrix.shape[-1], diagnostic_dtype='float64', input_weight_dtype=str(w.a.dtype))


def certified_radii(logits: Tensor, labels: Tensor, head_weight: Tensor,
                    feature_bound: float = 1.0) -> Tensor:
    """Return all-class margin radii; wrong or tied predictions receive zero.

    ``head_weight`` must be the actual affine head of an L_h-Lipschitz feature
    map, with L_h including preprocessing. Unit row norms do not imply that
    the vector-valued head has operator norm one. Equal head rows yield an
    infinite pairwise radius only when their (bias-dependent) margin is positive.
    """
    if feature_bound <= 0:
        raise ValueError("feature_bound must be positive")
    labels = labels.long()
    rows = torch.arange(logits.shape[0], device=logits.device)
    margins = logits[rows, labels, None] - logits
    distances = torch.linalg.vector_norm(head_weight[labels, None, :] - head_weight[None, :, :], dim=-1)
    denom = feature_bound * distances
    positive = torch.full_like(margins, torch.inf)
    ratios = torch.where(denom > 0, margins / denom.clamp_min(torch.finfo(logits.dtype).tiny),
                         torch.where(margins > 0, positive, torch.zeros_like(margins)))
    ratios[rows, labels] = torch.inf
    radii = ratios.min(dim=1).values.clamp_min(0)
    return torch.where(logits.argmax(dim=1) == labels, radii, torch.zeros_like(radii))
