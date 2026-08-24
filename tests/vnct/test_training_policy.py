"""Tests for the two intentionally supported VNCT training policies."""

from __future__ import annotations

import pytest
import torch

from vnct.engine.policy import (
    apply_training_mode,
    build_optimizer,
    configure_training_policy,
)
from vnct.models import vnct_biqa_debug


NEW_BACKBONE_SUFFIXES = (
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


def test_new_modules_only_freezes_only_inherited_vssd_parameters() -> None:
    model = vnct_biqa_debug(image_size=32)
    report = configure_training_policy(model, "new_modules_only")

    assert report.inherited_trainable == 0
    assert 0 < report.new_trainable == report.trainable < report.total
    for name, parameter in model.named_parameters():
        if name.startswith("backbone.") and not name.endswith(
            NEW_BACKBONE_SUFFIXES
        ):
            assert not parameter.requires_grad, name
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.endswith(NEW_BACKBONE_SUFFIXES)
    )

    apply_training_mode(model, "new_modules_only")
    assert not model.backbone.training
    assert model.quality_head.training


def test_full_policy_and_discriminative_optimizer_groups() -> None:
    model = vnct_biqa_debug(image_size=32)
    report = configure_training_policy(model, "full")
    optimizer = build_optimizer(
        model,
        policy="full",
        new_learning_rate=3e-4,
        inherited_learning_rate=3e-5,
        weight_decay=1e-2,
    )

    assert report.trainable == report.total
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["new_decay"]["lr"] == 3e-4
    assert groups["inherited_decay"]["lr"] == 3e-5
    assert groups["new_no_decay"]["weight_decay"] == 0.0
    assert groups["inherited_no_decay"]["weight_decay"] == 0.0


def test_unified_interaction_has_no_residual_source_gates() -> None:
    model = vnct_biqa_debug(image_size=32)

    assert all(not hasattr(interaction, "local_scale") for interaction in model.interactions)
    assert all(
        not hasattr(interaction, "enhancement_scale") for interaction in model.interactions
    )
    assert all(interaction.fusion[0].weight.requires_grad for interaction in model.interactions)
    assert all(mixer.residual_scale.item() == pytest.approx(0.01) for mixer in model.local_mixers)
    assert all(refiner.residual_scale.item() == pytest.approx(0.01) for refiner in model.refiners)


def test_small_nonzero_ncm3_gives_every_rank_gradient_in_frozen_eval_backbone() -> None:
    model = vnct_biqa_debug(image_size=32, ncm3_scale_init=0.01, mimo_rank=4)
    configure_training_policy(model, "new_modules_only")
    apply_training_mode(model, "new_modules_only")
    selector_image = torch.rand(1, 3, 32, 32)
    mean = selector_image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = selector_image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    model((selector_image - mean) / std, selector_image).mean().backward()

    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(NEW_BACKBONE_SUFFIXES)
    ]
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
