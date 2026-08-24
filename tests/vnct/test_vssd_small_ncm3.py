import torch

from vnct.models.backbones.vssd_small_ncm3 import (
    VSSDCheckpointNCM3,
    VSSD_SMALL_CAMERA_READY,
    _VSSD,
    vssd_ncm3_debug,
    vssd_original_debug,
)
from vnct.utils.checkpoint import load_vssd_checkpoint


def _debug_config() -> dict[str, object]:
    config = dict(VSSD_SMALL_CAMERA_READY)
    config.update(
        img_size=32,
        embed_dim=16,
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 4, 8),
        async_state=(8, 8, 16, 16),
        drop_path_rate=0.0,
    )
    return config


def test_camera_ready_small_preset() -> None:
    assert VSSD_SMALL_CAMERA_READY["depths"] == (2, 4, 15, 4)
    assert VSSD_SMALL_CAMERA_READY["mlp_ratio"] == 3.0
    assert VSSD_SMALL_CAMERA_READY["async_state"] == (12, 24, 48, 64)
    assert VSSD_SMALL_CAMERA_READY["attn_types"][-1] == "standard"


def test_checkpoint_bridge_loads_all_original_backbone_keys() -> None:
    torch.manual_seed(7)
    original = _VSSD.Backbone_VMAMBA2(out_indices=(0, 1, 2, 3), **_debug_config()).eval()
    bridge = vssd_ncm3_debug(img_size=32).eval()
    report = load_vssd_checkpoint(bridge, {"model_ema": original.state_dict()})

    assert report.coverage == 1.0
    assert not report.missing_keys
    assert not report.shape_mismatches
    extra = [
        key
        for key in bridge.state_dict()
        if key.endswith(
            (
                "trap_weight",
                "trap_bias",
                "decay_weight",
                "decay_bias",
                "mimo_source_weight",
                "mimo_query_weight",
                "mimo_in",
                "mimo_out",
                "ncm3_scale",
            )
        )
    ]
    assert len(extra) == 27  # three NC-SSD stages, one block each, nine tensors.

    image = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        expected = original(image)
        actual = bridge(image)
    for expected_feature, actual_feature in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_feature, expected_feature)


def test_original_builder_contains_no_ncm3_bridge_and_matches_zero_scale() -> None:
    torch.manual_seed(9)
    original = vssd_original_debug().eval()
    bridge = vssd_ncm3_debug(img_size=32, ncm3_scale_init=0.0).eval()
    report = load_vssd_checkpoint(bridge, {"model_ema": original.state_dict()})

    assert report.coverage == 1.0
    assert not any(
        isinstance(module, VSSDCheckpointNCM3) for module in original.modules()
    )
    image = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        expected = original(image)
        actual = bridge(image)
    for expected_feature, actual_feature in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_feature, expected_feature)


def test_trapezoidal_path_forward_backward() -> None:
    model = vssd_ncm3_debug(img_size=32, ncm3_scale_init=1.0).train()
    image = torch.randn(2, 3, 32, 32, requires_grad=True)
    features = model(image)
    loss = sum(feature.float().mean() for feature in features)
    loss.backward()

    assert [tuple(feature.shape) for feature in features] == [
        (2, 16, 8, 8),
        (2, 32, 4, 4),
        (2, 64, 2, 2),
        (2, 128, 1, 1),
    ]
    assert torch.isfinite(image.grad).all()
    mixers = [module for module in model.modules() if isinstance(module, VSSDCheckpointNCM3)]
    assert len(mixers) == 3
    assert all(module.mimo_rank == 4 for module in mixers)
    assert all(module.trap_weight.grad is not None for module in mixers)
    assert all(module.decay_weight.grad is not None for module in mixers)
    assert all(module.mimo_source_weight.grad is not None for module in mixers)
    assert all(module.mimo_query_weight.grad is not None for module in mixers)
    assert all(module.mimo_in.grad is not None for module in mixers)
    assert all(module.mimo_out.grad is not None for module in mixers)
