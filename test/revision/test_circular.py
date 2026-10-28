import torch
from models.certified_conv import CertifiedCircularResidual, circular_apply
from models.certified_ldl import assemble_lmi


def explicit_operator(weight):
    height, width, out_channels, in_channels = weight.shape
    basis = torch.eye(in_channels * height * width, dtype=torch.float64).reshape(-1, in_channels, height, width)
    return circular_apply(basis, weight).reshape(basis.shape[0], -1).T


def test_frequency_certificate_matches_spatial_operator():
    torch.manual_seed(7)
    model = CertifiedCircularResidual(1, hidden=2, depth=2, kernel_size=2, dtype=torch.float64)
    w = model.normalized_weights((2, 2))
    assert torch.linalg.eigvalsh(assemble_lmi(w.a, w.cs, w.b)).min() > 0
    a, cs, b = explicit_operator(w.a), [explicit_operator(c) for c in w.cs], explicit_operator(w.b)
    assert torch.linalg.eigvalsh(assemble_lmi(a, cs, b)).min() > 0
    x = torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    y = model(x)
    hidden = x.flatten()
    for c, bias in zip(cs, model.biases):
        hidden = torch.relu(c @ hidden + bias.repeat_interleave(4))
    expected = a @ x.flatten() + b @ hidden + model.bias.repeat_interleave(4)
    torch.testing.assert_close(y.flatten(), expected)
    y.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_conjugate_symmetry_and_cache():
    torch.manual_seed(8)
    model = CertifiedCircularResidual(2, kernel_size=2, dtype=torch.float64).eval()
    with torch.no_grad():
        w = model.normalized_weights((3, 4))
        for matrix in [w.a, *w.cs, w.b]:
            for i in range(3):
                for j in range(4):
                    torch.testing.assert_close(matrix[i, j], matrix[-i % 3, -j % 4].conj())
        x = torch.randn(2, 2, 3, 4, dtype=torch.float64)
        first = model(x)
        torch.testing.assert_close(first, model(x))
        model.train()
        assert model._cached is None
