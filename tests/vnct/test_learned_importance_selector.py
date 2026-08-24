"""Tests for marginal-coverage routing and unified regional evidence."""

from __future__ import annotations

import torch

from vnct.models import vnct_biqa_debug
from vnct.models.interactions import SpatialReducedCrossAttention
from vnct.models.refinement import SelectedRegionRefiner
from vnct.models.selectors import LearnedImportanceSelector


def test_selector_normalizes_map_and_applies_budget_relative_coverage() -> None:
    torch.manual_seed(11)
    selector = LearnedImportanceSelector(
        16,
        region_size=5,
        max_regions=4,
        coverage_threshold=0.8,
    )
    feature = torch.randn(2, 16, 16, 16, requires_grad=True)

    selection = selector(feature, image_size=(64, 64))

    assert selection.score_map.shape == (2, 1, 16, 16)
    assert selection.boxes.shape == (2, 4, 4)
    assert selection.stage_boxes.shape == (2, 4, 4)
    assert selection.marginal_gains.shape == (2, 4)
    assert selection.gain_shares.shape == (2, 4)
    assert selection.valid_mask.shape == (2, 4)
    assert torch.all((1 <= selection.num_selected) & (selection.num_selected <= 4))
    torch.testing.assert_close(
        selection.score_map.float().sum(dim=(-2, -1)), torch.ones(2, 1)
    )
    torch.testing.assert_close(
        selection.gain_shares.float().sum(dim=1), torch.ones(2)
    )
    assert torch.all(
        selection.marginal_gains[:, 1:] <= selection.marginal_gains[:, :-1] + 1e-7
    )
    cumulative = selection.gain_shares.cumsum(dim=1)
    selected_cumulative = cumulative.gather(
        1, (selection.num_selected - 1).unsqueeze(1)
    ).squeeze(1)
    assert torch.all(selected_cumulative >= 0.8 - 1e-6)
    for batch in range(2):
        count = int(selection.num_selected[batch])
        if count > 1:
            assert cumulative[batch, count - 2] < 0.8

    # The continuous soft map, rather than hard argmax indices, carries the
    # quality gradient used by the model's local-summary token.
    spatial_probe = torch.linspace(0, 1, 16 * 16).view(1, 1, 16, 16)
    (selection.score_map * spatial_probe).sum().backward()
    gradient = selector.decoder[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_refiner_runs_active_rois_and_emits_one_center_token() -> None:
    torch.manual_seed(13)
    refiner = SelectedRegionRefiner(
        16,
        d_state=8,
        num_heads=4,
        roi_size=5,
        residual_scale_init=0.1,
    )
    feature = torch.randn(1, 16, 10, 10, requires_grad=True)
    indices = torch.tensor([[0, 5, 30, 35]])
    stage_boxes = torch.tensor(
        [[[0, 0, 5, 5], [5, 0, 10, 5], [0, 5, 5, 10], [5, 5, 10, 10]]]
    )
    boxes = stage_boxes.float() * 4
    centers = 0.5 * (stage_boxes[..., :2] + stage_boxes[..., 2:]).float()
    centers = centers / 10
    valid_mask = torch.tensor([[True, True, False, False]])

    output = refiner(
        feature,
        boxes,
        image_size=(40, 40),
        valid_mask=valid_mask,
        candidate_indices=indices,
        coordinates=centers,
    )

    assert output.tokens.shape == (1, 4, 16)
    assert output.coordinates.shape == (1, 4, 2)
    assert output.roi_features.shape == (1, 4, 16, 5, 5)
    assert torch.count_nonzero(output.tokens[:, 2:]) == 0
    output.tokens[:, :2].square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


def test_attention_mask_ignores_inactive_regional_slots() -> None:
    torch.manual_seed(15)
    attention = SpatialReducedCrossAttention(16, 16, inner_dim=16, num_heads=4)
    attention.eval()
    query = torch.randn(2, 16, 4, 4)
    source = torch.randn(2, 5, 16)
    position = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)

    reference = attention(query, source, position, source_mask=mask)
    modified = source.clone()
    modified[:, 3:] = 1e4
    actual = attention(query, modified, position, source_mask=mask)

    torch.testing.assert_close(actual, reference)


def test_debug_model_uses_stage_maps_and_soft_summary_gradients() -> None:
    torch.manual_seed(17)
    model = vnct_biqa_debug(
        image_size=64,
        selector_mode="learned_importance",
        importance_max_regions=4,
        importance_coverage_threshold=0.8,
    )
    selector_image = torch.rand(1, 3, 64, 64)
    mean = selector_image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = selector_image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    output = model((selector_image - mean) / std, selector_image, return_details=True)

    assert output.selection is None
    assert len(output.stage_selections) == 3
    assert [item.score_map.shape[-2:] for item in output.stage_selections] == [
        (16, 16),
        (8, 8),
        (4, 4),
    ]
    assert all(item.boxes.shape == (1, 4, 4) for item in output.stage_selections)
    output.score.mean().backward()
    for selector in model.importance_selectors:
        gradient = selector.decoder[-1].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 1e-8
