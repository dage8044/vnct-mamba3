"""Composition and public builders for the VNCT-BIQA architecture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vnct.models.backbones.vssd_small_ncm3 import (
    vssd_ncm3_debug,
    vssd_small_original,
    vssd_small_ncm3,
)
from vnct.models.heads import (
    JointStageQualityHead,
    MANIQAPatchWeightedHead,
    QualityHeadOutput,
)
from vnct.models.interactions import UnifiedEvidenceInteraction
from vnct.models.layers import MultiRangeLocalMixer
from vnct.models.refinement import RefinedRegionTokens, SelectedRegionRefiner
from vnct.models.selectors import (
    ImportanceSelection,
    LearnedImportanceSelector,
    MSCNGGDSelector,
    NSSSelection,
)


@dataclass(frozen=True)
class VNCTBIQAOutput:
    """Detailed model output returned when ``return_details=True``."""

    score: torch.Tensor
    stage_scores: torch.Tensor
    stage_weights: torch.Tensor
    spatial_weights: torch.Tensor
    selection: NSSSelection | None
    stage_selections: tuple[ImportanceSelection, ...]
    interacted_features: tuple[torch.Tensor, ...]


class VNCTBIQA(nn.Module):
    """Single-backbone BIQA model with local and sparse refinement sources.

    ``backbone_image`` is the normalized tensor expected by the backbone.  The
    fixed NSS baseline additionally consumes ``selector_image`` in ``[0, 1]``;
    the learned stage selector operates directly on backbone features.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        channels: tuple[int, int, int, int],
        selector: MSCNGGDSelector | None = None,
        selector_mode: str = "mscn_ggd",
        enable_refinement: bool = True,
        importance_max_regions: int = 4,
        importance_region_sizes: tuple[int, int, int] | None = None,
        importance_coverage_threshold: float = 0.8,
        importance_candidate_stride: int = 1,
        importance_decoder_kernel_size: int = 5,
        importance_output_init_std: float = 1e-3,
        refinement_state_dims: tuple[int, int, int] = (12, 24, 48),
        refinement_heads: tuple[int, int, int] = (4, 4, 8),
        roi_sizes: tuple[int, int, int] = (5, 5, 5),
        interaction_dims: tuple[int, int, int, int] = (96, 192, 192, 192),
        interaction_heads: int = 4,
        local_grid_size: int = 7,
        local_dilation: int = 2,
        local_ffn_ratio: float = 3.0,
        local_residual_scale_init: float = 0.01,
        refinement_residual_scale_init: float = 0.01,
        head_name: str = "joint_stage_quality",
        head_embed_dim: int = 192,
        head_grid_size: int = 7,
        head_num_heads: int = 4,
        head_depth: int = 1,
        head_mlp_ratio: float = 3.0,
        head_dropout: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("channels must contain four stage dimensions")
        if not (
            len(refinement_state_dims) == len(refinement_heads) == len(roi_sizes) == 3
        ):
            raise ValueError("refinement settings must contain three stage values")
        if len(interaction_dims) != 4:
            raise ValueError("interaction_dims must contain four stage values")
        if selector_mode not in ("none", "mscn_ggd", "learned_importance"):
            raise ValueError(
                "selector_mode must be 'none', 'mscn_ggd', or 'learned_importance'"
            )
        if enable_refinement != (selector_mode != "none"):
            raise ValueError(
                "enable_refinement must be false exactly when selector_mode='none'"
            )
        if importance_region_sizes is None:
            importance_region_sizes = roi_sizes
        if len(importance_region_sizes) != 3:
            raise ValueError("importance_region_sizes must contain three values")
        if selector_mode == "learned_importance" and selector is not None:
            raise ValueError("a fixed selector cannot be supplied in learned mode")

        self.backbone = backbone
        self.channels = tuple(int(channel) for channel in channels)
        self.selector_mode = selector_mode
        self.enable_refinement = bool(enable_refinement)
        self.selector = (
            (selector or MSCNGGDSelector())
            if selector_mode == "mscn_ggd"
            else None
        )
        self.importance_selectors = (
            nn.ModuleList(
                LearnedImportanceSelector(
                    channels[stage],
                    region_size=importance_region_sizes[stage],
                    max_regions=importance_max_regions,
                    coverage_threshold=importance_coverage_threshold,
                    candidate_stride=importance_candidate_stride,
                    decoder_kernel_size=importance_decoder_kernel_size,
                    output_init_std=importance_output_init_std,
                )
                for stage in range(3)
            )
            if selector_mode == "learned_importance"
            else nn.ModuleList()
        )
        self.local_mixers = nn.ModuleList(
            MultiRangeLocalMixer(
                channel,
                dilation=local_dilation,
                ffn_ratio=local_ffn_ratio,
                residual_scale_init=local_residual_scale_init,
            )
            for channel in channels
        )
        self.refiners = (
            nn.ModuleList(
                SelectedRegionRefiner(
                    channels[stage],
                    d_state=refinement_state_dims[stage],
                    num_heads=refinement_heads[stage],
                    roi_size=roi_sizes[stage],
                    residual_scale_init=refinement_residual_scale_init,
                    candidate_stride=importance_candidate_stride,
                )
                for stage in range(3)
            )
            if enable_refinement
            else nn.ModuleList()
        )
        self.interactions = nn.ModuleList(
            UnifiedEvidenceInteraction(
                channels[stage],
                inner_dim=interaction_dims[stage],
                num_heads=interaction_heads,
                local_grid_size=local_grid_size,
                has_enhancement=enable_refinement and stage < 3,
                dropout=dropout,
            )
            for stage in range(4)
        )
        self.head_name = str(head_name)
        resolved_head_dropout = dropout if head_dropout is None else head_dropout
        if self.head_name == "joint_stage_quality":
            self.quality_head = JointStageQualityHead(
                channels,
                embed_dim=head_embed_dim,
                grid_size=head_grid_size,
                num_heads=head_num_heads,
                depth=head_depth,
                mlp_ratio=head_mlp_ratio,
                dropout=resolved_head_dropout,
            )
        elif self.head_name == "maniqa_patch_weighted":
            self.quality_head = MANIQAPatchWeightedHead(
                channels,
                embed_dim=head_embed_dim,
                grid_size=head_grid_size,
                dropout=resolved_head_dropout,
            )
        else:
            raise ValueError(f"unsupported quality head: {self.head_name!r}")

    def _validate_features(self, features: object) -> tuple[torch.Tensor, ...]:
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise ValueError("the backbone must return four stage feature maps")
        validated: list[torch.Tensor] = []
        for stage, (feature, channels) in enumerate(
            zip(features, self.channels, strict=True)
        ):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise ValueError(f"backbone stage {stage + 1} must be an NCHW tensor")
            if feature.shape[1] != channels:
                raise ValueError(
                    f"backbone stage {stage + 1} has {feature.shape[1]} channels; "
                    f"expected {channels}"
                )
            validated.append(feature)
        return tuple(validated)

    def forward(
        self,
        backbone_image: torch.Tensor,
        selector_image: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | VNCTBIQAOutput:
        if backbone_image.ndim != 4 or selector_image.ndim != 4:
            raise ValueError("both inputs must have shape [B, C, H, W]")
        if backbone_image.shape != selector_image.shape:
            raise ValueError(
                "backbone_image and selector_image must have the same shape"
            )
        if self.selector_mode == "mscn_ggd":
            selector_min = selector_image.detach().amin().item()
            selector_max = selector_image.detach().amax().item()
            if selector_min < 0.0 or selector_max > 1.0:
                raise ValueError("selector_image values must lie in [0, 1]")

        main_features = self._validate_features(self.backbone(backbone_image))
        local_features = tuple(
            mixer(feature)
            for mixer, feature in zip(self.local_mixers, main_features, strict=True)
        )
        image_size = tuple(int(size) for size in selector_image.shape[-2:])
        selection: NSSSelection | None = None
        stage_selections: tuple[ImportanceSelection, ...] = ()
        if self.selector_mode == "mscn_ggd":
            assert self.selector is not None
            selection = self.selector(selector_image)
            enhancements: list[RefinedRegionTokens] = [
                refiner(main_features[stage], selection.boxes, image_size=image_size)
                for stage, refiner in enumerate(self.refiners)
            ]
        elif self.selector_mode == "learned_importance":
            stage_selections = tuple(
                stage_selector(main_features[stage], image_size=image_size)
                for stage, stage_selector in enumerate(self.importance_selectors)
            )
            enhancements = [
                refiner(
                    main_features[stage],
                    stage_selections[stage].boxes,
                    image_size=image_size,
                    valid_mask=stage_selections[stage].valid_mask,
                    candidate_indices=stage_selections[stage].indices,
                    coordinates=stage_selections[stage].centers,
                )
                for stage, refiner in enumerate(self.refiners)
            ]
        else:
            enhancements = []

        interacted: list[torch.Tensor] = []
        for stage, interaction in enumerate(self.interactions):
            if self.enable_refinement and stage < 3:
                refined = enhancements[stage]
                output = interaction(
                    main_features[stage],
                    local_features[stage],
                    importance_map=(
                        stage_selections[stage].score_map
                        if self.selector_mode == "learned_importance"
                        else None
                    ),
                    enhancement_tokens=refined.tokens,
                    enhancement_coordinates=refined.coordinates,
                    enhancement_mask=refined.valid_mask,
                )
            else:
                output = interaction(main_features[stage], local_features[stage])
            interacted.append(output)
        interacted_features = tuple(interacted)
        head_output: QualityHeadOutput = self.quality_head(interacted_features)
        if not return_details:
            return head_output.score
        return VNCTBIQAOutput(
            score=head_output.score,
            stage_scores=head_output.stage_scores,
            stage_weights=head_output.stage_weights,
            spatial_weights=head_output.spatial_weights,
            selection=selection,
            stage_selections=stage_selections,
            interacted_features=interacted_features,
        )


def vssd_small_ncm3_biqa(
    *,
    pretrained: str | Path | None = None,
    ncm3_scale_init: float = 0.0,
    mimo_rank: int = 4,
    adapter_init_std: float = 0.02,
    **model_overrides: Any,
) -> VNCTBIQA:
    """Build the production VSSD-Small/NC-M3 VNCT-BIQA model."""
    backbone = vssd_small_ncm3(
        pretrained=pretrained,
        ncm3_scale_init=ncm3_scale_init,
        mimo_rank=mimo_rank,
        adapter_init_std=adapter_init_std,
    )
    return VNCTBIQA(
        backbone,
        channels=(96, 192, 384, 768),
        **model_overrides,
    )


def vssd_small_original_biqa(
    *,
    pretrained: str | Path | None = None,
    **model_overrides: Any,
) -> VNCTBIQA:
    """Build the original VSSD-Small with the complete VNCT-BIQA modules."""
    backbone = vssd_small_original(pretrained=pretrained)
    return VNCTBIQA(
        backbone,
        channels=(96, 192, 384, 768),
        **model_overrides,
    )


def vnct_biqa_debug(*, image_size: int = 64, **model_overrides: Any) -> VNCTBIQA:
    """Build a reduced model for unit and integration tests."""
    ncm3_scale_init = float(model_overrides.pop("ncm3_scale_init", 0.01))
    mimo_rank = int(model_overrides.pop("mimo_rank", 4))
    adapter_init_std = float(model_overrides.pop("adapter_init_std", 0.02))
    selector_mode = str(model_overrides.get("selector_mode", "mscn_ggd"))
    selector = model_overrides.pop("selector", None)
    if selector is None and selector_mode == "mscn_ggd":
        selector = MSCNGGDSelector(
            patch_size=16,
            patch_stride=16,
            num_patches=4,
        )
    backbone = vssd_ncm3_debug(
        img_size=image_size,
        ncm3_scale_init=ncm3_scale_init,
        mimo_rank=mimo_rank,
        adapter_init_std=adapter_init_std,
    )
    defaults: dict[str, Any] = {
        "refinement_state_dims": (8, 8, 16),
        "refinement_heads": (1, 2, 4),
        "roi_sizes": (3, 3, 3) if selector_mode == "learned_importance" else (4, 2, 1),
        "interaction_dims": (16, 32, 32, 32),
        "interaction_heads": 4,
        "local_grid_size": 4,
        "head_embed_dim": 32,
        "head_grid_size": 4,
        "head_num_heads": 4,
    }
    defaults.update(model_overrides)
    return VNCTBIQA(
        backbone,
        channels=(16, 32, 64, 128),
        selector=selector,
        **defaults,
    )


__all__ = [
    "VNCTBIQA",
    "VNCTBIQAOutput",
    "vnct_biqa_debug",
    "vssd_small_ncm3_biqa",
    "vssd_small_original_biqa",
]
