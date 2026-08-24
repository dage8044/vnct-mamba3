"""Unit and integration tests for the VNCT-BIQA architecture."""

from __future__ import annotations

import pytest
import torch

from vnct.models import vnct_biqa_debug
from vnct.models.heads import JointStageQualityHead, MANIQAPatchWeightedHead
from vnct.models.interactions import UnifiedEvidenceInteraction
from vnct.models.layers import MultiRangeLocalMixer
from vnct.models.refinement import SelectedRegionRefiner
from vnct.models.selectors import MSCNGGDSelector


def test_local_mixer_is_identity_at_initialization() -> None:
    mixer = MultiRangeLocalMixer(12)
    feature = torch.randn(2, 12, 9, 11)

    output = mixer(feature)

    torch.testing.assert_close(output, feature, rtol=0.0, atol=0.0)
    assert output.shape == feature.shape


def test_region_refiner_emits_one_center_token_per_box() -> None:
    refiner = SelectedRegionRefiner(
        16,
        d_state=8,
        num_heads=4,
        roi_size=5,
    )
    feature = torch.randn(2, 16, 16, 16, requires_grad=True)
    boxes = torch.tensor(
        [
            [[0, 0, 16, 16], [32, 32, 48, 48]],
            [[16, 0, 32, 16], [0, 32, 16, 48]],
        ],
        dtype=torch.float32,
    )

    output = refiner(feature, boxes, image_size=(64, 64))

    assert output.tokens.shape == (2, 2, 16)
    assert output.coordinates.shape == (2, 2, 2)
    assert output.roi_features.shape == (2, 2, 16, 5, 5)
    torch.testing.assert_close(
        output.coordinates[0, 0], torch.tensor([0.125, 0.125])
    )
    assert torch.all((output.coordinates >= 0) & (output.coordinates <= 1))
    output.tokens.mean().backward()
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()


def test_unified_interaction_generates_feature_from_masked_evidence() -> None:
    interaction = UnifiedEvidenceInteraction(
        16,
        inner_dim=16,
        num_heads=4,
        local_grid_size=3,
        has_enhancement=True,
    )
    main = torch.randn(2, 16, 8, 8)
    local = torch.randn_like(main)
    enhancement = torch.randn(2, 12, 16)
    coordinates = torch.rand(2, 12, 2)
    mask = torch.ones(2, 12, dtype=torch.bool)
    importance = torch.rand(2, 1, 8, 8).softmax(dim=-1)
    importance = importance / importance.sum(dim=(-2, -1), keepdim=True)

    output = interaction(
        main,
        local,
        importance_map=importance,
        enhancement_tokens=enhancement,
        enhancement_coordinates=coordinates,
        enhancement_mask=mask,
    )

    assert output.shape == main.shape
    assert not torch.equal(output, main)
    assert interaction.last_stats["evidence_tokens_mean"].item() == 22
    assert interaction.last_stats["fusion_change_ratio"].item() > 0
    assert not hasattr(interaction, "local_scale")


def test_joint_head_starts_with_uniform_pooling_weights() -> None:
    head = JointStageQualityHead(
        (8, 16, 24, 32),
        embed_dim=32,
        grid_size=3,
        num_heads=4,
    ).eval()
    features = (
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 24, 4, 4),
        torch.randn(2, 32, 2, 2),
    )

    output = head(features)

    assert output.score.shape == (2,)
    assert output.stage_scores.shape == (2, 4)
    assert output.stage_weights.shape == (2, 4)
    assert output.spatial_weights.shape == (2, 4, 9)
    torch.testing.assert_close(
        output.stage_weights, torch.full_like(output.stage_weights, 0.25)
    )
    torch.testing.assert_close(
        output.spatial_weights, torch.full_like(output.spatial_weights, 1.0 / 9.0)
    )


def test_maniqa_head_normalizes_patch_and_stage_contributions() -> None:
    torch.manual_seed(31)
    head = MANIQAPatchWeightedHead(
        (8, 16, 24, 32),
        embed_dim=32,
        grid_size=3,
        dropout=0.0,
    )
    features = (
        torch.randn(2, 8, 16, 16, requires_grad=True),
        torch.randn(2, 16, 8, 8, requires_grad=True),
        torch.randn(2, 24, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    )

    output = head(features)

    assert output.score.shape == (2,)
    assert output.stage_scores.shape == (2, 4)
    assert output.stage_weights.shape == (2, 4)
    assert output.spatial_weights.shape == (2, 4, 9)
    torch.testing.assert_close(
        output.spatial_weights.sum(dim=-1),
        torch.ones_like(output.stage_scores),
    )
    torch.testing.assert_close(
        output.stage_weights.sum(dim=-1), torch.ones_like(output.score)
    )
    torch.testing.assert_close(
        output.score,
        (output.stage_weights * output.stage_scores).sum(dim=-1),
    )
    assert torch.all(output.score >= 0)
    output.score.sum().backward()
    assert head.quality_predictor[-2].weight.grad is not None
    assert head.weight_predictor[-2].weight.grad is not None
    assert all(feature.grad is not None for feature in features)


def test_debug_model_selects_maniqa_head_without_joint_encoder() -> None:
    model = vnct_biqa_debug(
        image_size=64,
        head_name="maniqa_patch_weighted",
        head_dropout=0.1,
    )

    assert model.head_name == "maniqa_patch_weighted"
    assert isinstance(model.quality_head, MANIQAPatchWeightedHead)
    assert not hasattr(model.quality_head, "joint_encoder")
    assert not hasattr(model.quality_head, "stage_predictor")


def test_debug_model_forward_backward_and_detailed_contract() -> None:
    torch.manual_seed(7)
    model = vnct_biqa_debug(image_size=64)
    selector_image = torch.rand(1, 3, 64, 64)
    mean = selector_image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = selector_image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    backbone_image = (selector_image - mean) / std

    output = model(backbone_image, selector_image, return_details=True)

    assert output.score.shape == (1,)
    assert output.stage_scores.shape == (1, 4)
    assert output.spatial_weights.shape == (1, 4, 16)
    assert output.selection.boxes.shape == (1, 4, 4)
    assert [feature.shape[1] for feature in output.interacted_features] == [
        16,
        32,
        64,
        128,
    ]
    assert torch.isfinite(output.score).all()
    output.score.mean().backward()
    assert model.quality_head.quality_predictor[-1].weight.grad is not None


@pytest.mark.parametrize("selected", (2, 4))
def test_model_accepts_selector_ablation_counts(selected: int) -> None:
    model = vnct_biqa_debug(
        image_size=64,
        selector=MSCNGGDSelector(
            patch_size=16,
            patch_stride=16,
            num_patches=selected,
            alpha_steps=101,
        ),
    ).eval()
    selector_image = torch.rand(1, 3, 64, 64)
    mean = selector_image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = selector_image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    with torch.inference_mode():
        output = model(
            (selector_image - mean) / std,
            selector_image,
            return_details=True,
        )

    assert output.selection.boxes.shape == (1, selected, 4)
    assert output.selection.patch_scores.shape == (1, selected)
    assert torch.isfinite(output.score).all()


def test_integrated_model_rejects_normalized_selector_input() -> None:
    model = vnct_biqa_debug(image_size=64)
    image = torch.full((1, 3, 64, 64), -1.0)

    try:
        model(image, image)
    except ValueError as error:
        assert "[0, 1]" in str(error)
    else:
        raise AssertionError("normalized selector input should have been rejected")


def test_local_only_model_constructs_no_selector_refiner_or_enhancement_path() -> None:
    torch.manual_seed(23)
    model = vnct_biqa_debug(
        image_size=64,
        selector_mode="none",
        enable_refinement=False,
    )
    image = torch.rand(1, 3, 64, 64)
    mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    output = model((image - mean) / std, image, return_details=True)

    assert model.selector is None
    assert len(model.importance_selectors) == 0
    assert len(model.refiners) == 0
    assert all(not interaction.has_enhancement for interaction in model.interactions)
    assert output.selection is None
    assert output.stage_selections == ()
    output.score.mean().backward()
    assert model.interactions[0].fusion[0].weight.grad is not None


def test_zero_residual_profile_still_uses_feature_generating_interaction() -> None:
    torch.manual_seed(19)
    model = vnct_biqa_debug(
        image_size=64,
        ncm3_scale_init=0.0,
        local_residual_scale_init=0.0,
        refinement_residual_scale_init=0.0,
    ).eval()
    selector_image = torch.rand(1, 3, 64, 64)
    mean = selector_image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = selector_image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    backbone_image = (selector_image - mean) / std

    with torch.inference_mode():
        expected = model.backbone(backbone_image)
        actual = model(
            backbone_image, selector_image, return_details=True
        ).interacted_features

    assert all(actual_feature.shape == expected_feature.shape for expected_feature, actual_feature in zip(expected, actual, strict=True))
    assert any(
        not torch.equal(actual_feature, expected_feature)
        for expected_feature, actual_feature in zip(expected, actual, strict=True)
    )
