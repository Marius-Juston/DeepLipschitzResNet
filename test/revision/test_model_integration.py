import pytest
import torch
from certification import certified_radii
from models.linear_model import DeepLipschitzLinearResNet, DeepLipschitzSequential, DeepLipschitzLinearResNetLNN, DeepLipschitzSequentialLNN
from models.certified_ldl import assemble_lmi


@pytest.mark.parametrize('cls', [DeepLipschitzLinearResNet, DeepLipschitzSequential,
                                DeepLipschitzLinearResNetLNN, DeepLipschitzSequentialLNN])
def test_integrated_model(cls):
    torch.manual_seed(3)
    model = cls(3, 2, layers_widths=[4, 2], device='cpu', dtype=torch.float64)
    w = model.normalized_weights()
    assert torch.linalg.eigvalsh(assemble_lmi(w.a, w.cs, w.b)).min() > 0
    x = torch.randn(5, 3, dtype=torch.float64)
    y = model(x)
    assert y.shape == (5, 2)
    y.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.load_state_dict(model.state_dict())
    legacy = {k:v for k,v in model.state_dict().items() if k != '_parameterization_version'}
    with pytest.raises(RuntimeError, match='version 2'):
        model.load_state_dict(legacy, strict=False)


def test_certificate_boundaries_and_equal_rows():
    weights = torch.tensor([[1., 0.], [1., 0.], [-1., 0.]])
    logits = torch.tensor([[3., 2., 1.], [2., 2., 0.], [0., 1., 2.]])
    radii = certified_radii(logits, torch.tensor([0, 0, 0]), weights, feature_bound=2.)
    torch.testing.assert_close(radii, torch.tensor([.5, 0., 0.]))


def test_unproved_activation_rejected():
    with pytest.raises(ValueError, match='sector'):
        DeepLipschitzLinearResNet(2, 2, [2], activation=torch.nn.SiLU(), device='cpu')
