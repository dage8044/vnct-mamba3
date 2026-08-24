"""Stage-wise quality-aware importance and budget-relative ROI routing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ImportanceSelection:
    """Dense importance and a padded maximum-budget routing decision."""

    score_map: torch.Tensor
    boxes: torch.Tensor
    stage_boxes: torch.Tensor
    centers: torch.Tensor
    indices: torch.Tensor
    marginal_gains: torch.Tensor
    gain_shares: torch.Tensor
    valid_mask: torch.Tensor
    num_selected: torch.Tensor


class _LayerNorm2d(nn.Module):
    """Apply channel LayerNorm independently at every spatial location."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LearnedImportanceSelector(nn.Module):
    """Decode a spatial distribution and route a variable number of ROIs.

    Candidate scores are recomputed after every choice using only importance
    mass not covered by earlier windows. Four proposals are generated before
    the smallest prefix covering ``coverage_threshold`` of those four gains is
    activated. Hard routing is intentionally detached; the differentiable map
    reaches the IQA objective through local soft pooling in the interaction.
    """

    def __init__(
        self,
        channels: int,
        *,
        region_size: int = 5,
        max_regions: int = 4,
        coverage_threshold: float = 0.8,
        candidate_stride: int = 1,
        decoder_kernel_size: int = 5,
        output_init_std: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if channels <= 0 or region_size <= 0 or max_regions <= 0:
            raise ValueError("channels, region_size, and max_regions must be positive")
        if region_size % 2 == 0:
            raise ValueError("region_size must be odd so every ROI has one center")
        if not 0.0 < coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must lie in (0, 1]")
        if candidate_stride <= 0:
            raise ValueError("candidate_stride must be positive")
        if decoder_kernel_size <= 0 or decoder_kernel_size % 2 == 0:
            raise ValueError("decoder_kernel_size must be a positive odd integer")
        if output_init_std <= 0.0:
            raise ValueError("output_init_std must be positive")

        self.channels = int(channels)
        self.region_size = int(region_size)
        self.max_regions = int(max_regions)
        self.coverage_threshold = float(coverage_threshold)
        self.candidate_stride = int(candidate_stride)
        self.eps = float(eps)
        padding = decoder_kernel_size // 2
        self.decoder = nn.Sequential(
            _LayerNorm2d(channels),
            nn.Conv2d(
                channels,
                channels,
                decoder_kernel_size,
                padding=padding,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                decoder_kernel_size,
                padding=padding,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        nn.init.normal_(self.decoder[-1].weight, std=output_init_std)
        nn.init.zeros_(self.decoder[-1].bias)
        self.last_stats: dict[str, torch.Tensor] = {}

    def _candidate_boxes(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        rows = (height - self.region_size) // self.candidate_stride + 1
        columns = (width - self.region_size) // self.candidate_stride + 1
        if rows <= 0 or columns <= 0:
            raise ValueError(
                f"feature size {(height, width)} is smaller than "
                f"region_size={self.region_size}"
            )
        y = torch.arange(rows, device=device) * self.candidate_stride
        x = torch.arange(columns, device=device) * self.candidate_stride
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack(
            (xx, yy, xx + self.region_size, yy + self.region_size), dim=-1
        ).reshape(-1, 4)

    def _marginal_topk(
        self,
        importance: torch.Tensor,
        candidate_boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedily maximize uncovered spatial probability mass."""
        batch, _, height, width = importance.shape
        covered = torch.zeros(
            (batch, 1, height, width), device=importance.device, dtype=torch.bool
        )
        batch_index = torch.arange(batch, device=importance.device)
        yy = torch.arange(height, device=importance.device).view(1, height, 1)
        xx = torch.arange(width, device=importance.device).view(1, 1, width)
        selected_indices: list[torch.Tensor] = []
        selected_gains: list[torch.Tensor] = []

        for _ in range(self.max_regions):
            uncovered = importance.detach().masked_fill(covered, 0.0)
            gains = F.avg_pool2d(
                uncovered,
                kernel_size=self.region_size,
                stride=self.candidate_stride,
            ).flatten(1) * float(self.region_size**2)
            index = gains.argmax(dim=1)
            gain = gains[batch_index, index]
            selected_indices.append(index)
            selected_gains.append(gain)

            box = candidate_boxes[index]
            selected_area = (
                (xx >= box[:, 0, None, None])
                & (xx < box[:, 2, None, None])
                & (yy >= box[:, 1, None, None])
                & (yy < box[:, 3, None, None])
            ).unsqueeze(1)
            covered = covered | selected_area

        return torch.stack(selected_indices, dim=1), torch.stack(selected_gains, dim=1)

    def forward(
        self,
        feature: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> ImportanceSelection:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(f"feature must have shape [B, {self.channels}, H, W]")
        image_height, image_width = image_size
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image dimensions must be positive")

        batch, _, height, width = feature.shape
        logits = self.decoder(feature).float()
        importance = logits.flatten(1).softmax(dim=-1).view(batch, 1, height, width)
        candidates = self._candidate_boxes(height, width, device=feature.device)
        indices, gains = self._marginal_topk(importance, candidates)
        gain_shares = gains / gains.sum(dim=1, keepdim=True).clamp_min(self.eps)
        cumulative = gain_shares.cumsum(dim=1)
        num_selected = (cumulative < self.coverage_threshold).sum(dim=1) + 1
        num_selected = num_selected.clamp(max=self.max_regions)
        slots = torch.arange(self.max_regions, device=feature.device)
        valid_mask = slots.unsqueeze(0) < num_selected.unsqueeze(1)

        stage_boxes = candidates[indices]
        image_scale = importance.new_tensor(
            (image_width / width, image_height / height) * 2
        )
        boxes = stage_boxes.to(dtype=importance.dtype) * image_scale
        centers = 0.5 * (
            stage_boxes[..., :2].to(dtype=importance.dtype)
            + stage_boxes[..., 2:].to(dtype=importance.dtype)
        )
        centers = centers / centers.new_tensor((width, height))

        with torch.no_grad():
            probabilities = importance.flatten(1)
            entropy = -(probabilities * probabilities.clamp_min(self.eps).log()).sum(1)
            entropy = entropy / math.log(probabilities.shape[1])
            self.last_stats = {
                "entropy": entropy.mean(),
                "peak_to_uniform": probabilities.amax(dim=1).mean()
                * probabilities.shape[1],
                "selected_k_mean": num_selected.float().mean(),
            }
            for count in range(1, self.max_regions + 1):
                self.last_stats[f"selected_k_{count}_ratio"] = (
                    num_selected == count
                ).float().mean()
            for slot in range(self.max_regions):
                self.last_stats[f"gain_share_{slot + 1}"] = gain_shares[:, slot].mean()

        return ImportanceSelection(
            score_map=importance.to(dtype=feature.dtype),
            boxes=boxes.to(dtype=feature.dtype),
            stage_boxes=stage_boxes,
            centers=centers.to(dtype=feature.dtype),
            indices=indices,
            marginal_gains=gains.to(dtype=feature.dtype),
            gain_shares=gain_shares.to(dtype=feature.dtype),
            valid_mask=valid_mask,
            num_selected=num_selected,
        )

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, region_size={self.region_size}, "
            f"max_regions={self.max_regions}, "
            f"coverage_threshold={self.coverage_threshold:.3g}, "
            f"candidate_stride={self.candidate_stride}"
        )


__all__ = ["ImportanceSelection", "LearnedImportanceSelector"]
