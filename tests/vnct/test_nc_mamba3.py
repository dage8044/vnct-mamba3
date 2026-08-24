import torch

from vnct.models import NCMamba3, vnct_debug
from vnct.models.layers import NCMamba3 as ExportedNCMamba3


def test_trapezoidal_coefficients() -> None:
    layer = NCMamba3(16, d_state=16, expand=2, headdim=16, mimo_rank=2)
    delta = torch.zeros(2, 5, layer.nheads)
    decay = torch.zeros_like(delta)
    trap = torch.zeros_like(delta)

    alpha, beta, gamma = layer._coefficients(delta, decay, trap)

    assert ExportedNCMamba3 is NCMamba3
    assert torch.all((alpha > 0) & (alpha < 1))
    assert torch.all(beta > 0)
    assert torch.all(gamma > 0)
    assert torch.allclose(beta / alpha, gamma, rtol=1e-5, atol=1e-7)


def test_beta_shift_does_not_wrap() -> None:
    beta = torch.tensor([[[1.0], [2.0], [3.0]]])
    shifted = NCMamba3._shift_beta(beta)
    assert torch.equal(shifted, torch.tensor([[[2.0], [3.0], [0.0]]]))


def test_nc_mamba3_forward_backward() -> None:
    layer = NCMamba3(
        16,
        d_state=16,
        expand=2,
        headdim=16,
        mimo_rank=2,
        chunk_size=7,
    )
    tokens = torch.randn(2, 20, 16, requires_grad=True)
    output = layer(tokens, 4, 5)
    output.square().mean().backward()

    assert output.shape == tokens.shape
    assert torch.isfinite(output).all()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_vnct_backbone_multiscale_dynamic_shape() -> None:
    model = vnct_debug()
    image = torch.randn(1, 3, 64, 80)
    outputs = model(image)

    assert [tuple(output.shape) for output in outputs] == [
        (1, 16, 16, 20),
        (1, 32, 8, 10),
        (1, 64, 4, 5),
        (1, 128, 2, 3),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
