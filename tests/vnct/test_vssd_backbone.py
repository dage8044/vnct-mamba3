import torch

from vnct.models.backbones.vssd import vssd_debug
from vnct.models.layers.nc_ssd import NCSSD


def test_ncssd_forward_and_backward():
    layer = NCSSD(d_model=16, d_state=8, headdim=8)
    tokens = torch.randn(2, 16, 16, requires_grad=True)
    output = layer(tokens, height=4, width=4)
    assert output.shape == tokens.shape
    output.square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_vssd_backbone_returns_multiscale_features():
    model = vssd_debug().eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        features = model(image)
    assert [tuple(feature.shape) for feature in features] == [
        (1, 16, 16, 16),
        (1, 32, 8, 8),
        (1, 64, 4, 4),
        (1, 128, 2, 2),
    ]
