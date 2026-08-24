"""Independent NC-SSD refinement of dynamically selected feature ROIs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.ops import roi_align

from vnct.models.layers.nc_ssd import NCSSD


@dataclass(frozen=True)
class RefinedRegionTokens:
    """Padded center tokens, coordinates, and active-slot mask."""

    tokens: torch.Tensor
    coordinates: torch.Tensor
    valid_mask: torch.Tensor
    roi_features: torch.Tensor


class SelectedRegionRefiner(nn.Module):
    """Run the shared stage Region NC-SSD independently for every active ROI."""

    def __init__(
        self,
        channels: int,
        *,
        d_state: int,
        num_heads: int,
        roi_size: int = 5,
        residual_scale_init: float = 0.0,
        candidate_stride: int = 1,
    ) -> None:
        super().__init__()
        if channels <= 0 or d_state <= 0:
            raise ValueError("channels and d_state must be positive")
        if num_heads <= 0 or channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if roi_size <= 0:
            raise ValueError("roi_size must be positive")
        if candidate_stride <= 0:
            raise ValueError("candidate_stride must be positive")
        if d_state % 4 != 0:
            raise ValueError("d_state must be divisible by four for 2-D RoPE")

        self.channels = int(channels)
        self.roi_size = int(roi_size)
        self.candidate_stride = int(candidate_stride)
        self.pre_norm = nn.LayerNorm(channels)
        self.mixer = NCSSD(
            d_model=channels,
            d_state=d_state,
            expand=1.0,
            headdim=channels // num_heads,
            d_conv=3,
            use_rope=True,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _gather_stage_windows(
        self,
        feature: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        rows = (height - self.roi_size) // self.candidate_stride + 1
        columns = (width - self.roi_size) // self.candidate_stride + 1
        if rows <= 0 or columns <= 0:
            raise ValueError("feature is smaller than the configured ROI")
        windows = feature.unfold(
            2, self.roi_size, self.candidate_stride
        ).unfold(3, self.roi_size, self.candidate_stride)
        windows = windows.contiguous().view(
            batch, channels, rows * columns, self.roi_size, self.roi_size
        ).permute(0, 2, 1, 3, 4)
        gather_index = indices[:, :, None, None, None].expand(
            -1, -1, channels, self.roi_size, self.roi_size
        )
        return windows.gather(1, gather_index)

    def _align_image_boxes(
        self,
        feature: torch.Tensor,
        boxes: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, region_count = boxes.shape[:2]
        image_height, image_width = image_size
        feature_height, feature_width = feature.shape[-2:]
        scale = boxes.new_tensor(
            (
                feature_width / image_width,
                feature_height / image_height,
                feature_width / image_width,
                feature_height / image_height,
            ),
            dtype=torch.float32,
        )
        feature_boxes = boxes.float() * scale
        batch_indices = torch.arange(batch, device=boxes.device).view(batch, 1, 1)
        batch_indices = batch_indices.expand(-1, region_count, -1).float()
        rois = torch.cat((batch_indices, feature_boxes), dim=-1).reshape(-1, 5)
        aligned = roi_align(
            feature,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )
        return aligned.view(
            batch,
            region_count,
            self.channels,
            self.roi_size,
            self.roi_size,
        )

    @staticmethod
    def _normalized_centers(
        boxes: torch.Tensor,
        *,
        image_size: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        image_height, image_width = image_size
        centers = 0.5 * (boxes[..., :2] + boxes[..., 2:]).to(dtype=dtype)
        return centers / centers.new_tensor((image_width, image_height))

    def forward(
        self,
        feature: torch.Tensor,
        boxes: torch.Tensor,
        *,
        image_size: tuple[int, int],
        valid_mask: torch.Tensor | None = None,
        candidate_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> RefinedRegionTokens:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(f"feature must have shape [B, {self.channels}, H, W]")
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape [B, K, 4]")
        batch, region_count = boxes.shape[:2]
        if feature.shape[0] != batch:
            raise ValueError("feature and boxes batch dimensions must match")
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch, region_count), device=feature.device, dtype=torch.bool
            )
        if valid_mask.shape != (batch, region_count):
            raise ValueError("valid_mask must have shape [B, K]")
        if not bool(valid_mask.any()):
            raise ValueError("at least one ROI must be active")

        if candidate_indices is not None:
            if candidate_indices.shape != (batch, region_count):
                raise ValueError("candidate_indices must have shape [B, K]")
            aligned = self._gather_stage_windows(feature, candidate_indices)
        else:
            aligned = self._align_image_boxes(feature, boxes, image_size)

        flat = aligned.reshape(
            batch * region_count,
            self.channels,
            self.roi_size,
            self.roi_size,
        )
        active = valid_mask.flatten().nonzero(as_tuple=False).squeeze(1)
        active_rois = flat.index_select(0, active)
        tokens = active_rois.flatten(2).transpose(1, 2)
        update = self.mixer(
            self.pre_norm(tokens), height=self.roi_size, width=self.roi_size
        )
        refined_active = active_rois + self.residual_scale.to(update.dtype) * (
            update.transpose(1, 2).reshape_as(active_rois)
        )
        refined_flat = torch.zeros_like(flat).index_copy(0, active, refined_active)
        refined = refined_flat.view_as(aligned)
        center = self.roi_size // 2
        center_tokens = refined[:, :, :, center, center]

        if coordinates is None:
            coordinates = self._normalized_centers(
                boxes, image_size=image_size, dtype=center_tokens.dtype
            )
        elif coordinates.shape != (batch, region_count, 2):
            raise ValueError("coordinates must have shape [B, K, 2]")

        return RefinedRegionTokens(
            tokens=center_tokens,
            coordinates=coordinates.to(dtype=center_tokens.dtype),
            valid_mask=valid_mask,
            roi_features=refined,
        )

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, roi_size={self.roi_size}, "
            f"candidate_stride={self.candidate_stride}, output=center_token"
        )


__all__ = ["RefinedRegionTokens", "SelectedRegionRefiner"]
