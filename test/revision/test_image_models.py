import pytest
import torch
from certification import clear_weight_caches
from revision_image import ImageModel


@pytest.mark.parametrize('algorithm', ['ldlt-resnet','sandwich','sdp'])
def test_image_forward_backward_and_evaluation(algorithm):
    torch.manual_seed(12)
    model=ImageModel(algorithm,channels=3,features=4)
    x=torch.rand(2,3,32,32)
    # Initialization forward materializes any lazy baseline scale parameters.
    model(x)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    optimizer.zero_grad()
    model(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    model.eval();clear_weight_caches(model)
    with torch.no_grad():
        y=model(x)
        assert y.shape==(2,10) and torch.isfinite(y).all()
        torch.testing.assert_close(y,model(x))
