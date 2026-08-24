"""Fixed natural-scene-statistics selector for the first BIQA baseline.

This module implements the part of DPSF that can be reproduced from public
material: MSCN coefficients are computed for the input image, a zero-mean GGD
is fitted to every candidate patch, and patches whose fitted distribution is
most different from the image distribution are selected.  DPSF reports its
best shared setting at a 56-pixel patch size and eight selected patches.

The DPSF article and source code do not publicly expose the exact candidate
stride or distribution-distance equation.  They are therefore explicit here:
the baseline uses non-overlapping candidates and Euclidean distance between
log GGD shape/scale parameters.  Keeping these choices configurable prevents
the baseline from being mistaken for an exact reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NSSSelection:
    """Outputs of :class:`MSCNGGDSelector`.

    Coordinates use half-open ``(x1, y1, x2, y2)`` image-space boxes.  The
    score map has the same spatial size as the selector input and is normalized
    independently to ``[0, 1]`` for every image.
    """

    score_map: torch.Tensor
    patch_scores: torch.Tensor
    boxes: torch.Tensor
    centers: torch.Tensor
    patch_alpha: torch.Tensor
    patch_sigma: torch.Tensor


class MSCNGGDSelector(nn.Module):
    """Select image patches using fixed MSCN/GGD distribution statistics.

    Args:
        patch_size: Square candidate size in input pixels.
        patch_stride: Candidate stride. ``None`` means non-overlapping patches.
        num_patches: Number of highest-scoring candidates to return.
        local_kernel_size: Gaussian window size used for MSCN normalization.
        local_sigma: Standard deviation of the MSCN Gaussian window.
        alpha_min: Smallest GGD shape parameter in the moment lookup table.
        alpha_max: Largest GGD shape parameter in the moment lookup table.
        alpha_steps: Number of entries in the GGD shape lookup table.
        eps: Numerical stability constant.

    Input tensors must be unnormalized luminance or RGB images in ``[0, 1]``.
    ImageNet-normalized backbone tensors are not valid MSCN inputs.
    """

    def __init__(
        self,
        *,
        patch_size: int = 56,
        patch_stride: int | None = None,
        num_patches: int = 8,
        local_kernel_size: int = 7,
        local_sigma: float = 7.0 / 6.0,
        alpha_min: float = 0.2,
        alpha_max: float = 10.0,
        alpha_steps: int = 9801,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if patch_stride is not None and patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if num_patches <= 0:
            raise ValueError("num_patches must be positive")
        if local_kernel_size <= 0 or local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be a positive odd integer")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError("alpha bounds must satisfy 0 < alpha_min < alpha_max")
        if alpha_steps < 2:
            raise ValueError("alpha_steps must be at least two")

        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride or patch_size)
        self.num_patches = int(num_patches)
        self.local_kernel_size = int(local_kernel_size)
        self.local_sigma = float(local_sigma)
        self.eps = float(eps)

        gaussian = self._gaussian_kernel(local_kernel_size, local_sigma)
        self.register_buffer("gaussian_kernel", gaussian, persistent=False)

        alpha = torch.linspace(alpha_min, alpha_max, alpha_steps)
        inverse_alpha = alpha.reciprocal()
        log_ratio = (
            torch.lgamma(inverse_alpha)
            + torch.lgamma(3.0 * inverse_alpha)
            - 2.0 * torch.lgamma(2.0 * inverse_alpha)
        )
        self.register_buffer("alpha_table", alpha, persistent=False)
        self.register_buffer("rho_table", log_ratio.exp(), persistent=False)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coordinates = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        return torch.outer(kernel_1d, kernel_1d).view(1, 1, size, size)

    @staticmethod
    def _to_luminance(image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("selector input must have shape [B, C, H, W]")
        if image.shape[1] == 1:
            return image
        if image.shape[1] != 3:
            raise ValueError("selector input must have one or three channels")
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        return (image * weights).sum(dim=1, keepdim=True)

    def _mscn(self, luminance: torch.Tensor) -> torch.Tensor:
        padding = self.local_kernel_size // 2
        padded = F.pad(luminance, (padding,) * 4, mode="reflect")
        kernel = self.gaussian_kernel.to(device=luminance.device, dtype=luminance.dtype)
        local_mean = F.conv2d(padded, kernel)
        local_second_moment = F.conv2d(padded.square(), kernel)
        local_variance = (local_second_moment - local_mean.square()).clamp_min(0.0)
        local_sigma = torch.sqrt(local_variance)
        return (luminance - local_mean) / (local_sigma + 1.0 / 255.0)

    def _fit_ggd(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate zero-mean GGD shape and scale by moment matching."""
        samples = samples.float()
        sigma = samples.square().mean(dim=-1).clamp_min(self.eps).sqrt()
        mean_abs = samples.abs().mean(dim=-1).clamp_min(self.eps)
        rho = sigma.square() / mean_abs.square()

        rho_table = self.rho_table.to(device=samples.device)
        flat_rho = rho.reshape(-1, 1)
        indices = (flat_rho - rho_table.view(1, -1)).abs().argmin(dim=-1)
        alpha = self.alpha_table.to(device=samples.device)[indices].view_as(rho)
        return alpha, sigma

    def _candidate_boxes(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        rows = (height - self.patch_size) // self.patch_stride + 1
        columns = (width - self.patch_size) // self.patch_stride + 1
        y = torch.arange(rows, device=device) * self.patch_stride
        x = torch.arange(columns, device=device) * self.patch_stride
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack(
            (xx, yy, xx + self.patch_size, yy + self.patch_size), dim=-1
        ).reshape(-1, 4)

    def _dense_score_map(
        self,
        scores: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        patch_area = self.patch_size**2
        expanded = scores.unsqueeze(1).expand(-1, patch_area, -1)
        score_sum = F.fold(
            expanded,
            output_size=output_size,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        coverage = F.fold(
            torch.ones_like(expanded),
            output_size=output_size,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        dense = score_sum / coverage.clamp_min(1.0)
        minimum = dense.amin(dim=(-2, -1), keepdim=True)
        maximum = dense.amax(dim=(-2, -1), keepdim=True)
        return (dense - minimum) / (maximum - minimum).clamp_min(self.eps)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> NSSSelection:
        image = image.detach()
        luminance = self._to_luminance(image).float()
        height, width = luminance.shape[-2:]
        if min(height, width) < self.patch_size:
            raise ValueError(
                f"input spatial size {(height, width)} is smaller than "
                f"patch_size={self.patch_size}"
            )

        # DPSF's public pipeline crops candidates before MSCN estimation.  Keep
        # patch normalization independent so a distorted patch cannot leak
        # through the local Gaussian window into its neighboring candidate.
        image_mscn = self._mscn(luminance)
        raw_candidates = F.unfold(
            luminance,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        ).transpose(1, 2)
        candidate_count = raw_candidates.shape[1]
        if candidate_count < self.num_patches:
            raise ValueError(
                f"only {candidate_count} candidates are available, fewer than "
                f"num_patches={self.num_patches}"
            )

        candidate_mscn = self._mscn(
            raw_candidates.reshape(-1, 1, self.patch_size, self.patch_size)
        ).flatten(1)
        candidate_mscn = candidate_mscn.view(image.shape[0], candidate_count, -1)
        patch_alpha, patch_sigma = self._fit_ggd(candidate_mscn)
        image_alpha, image_sigma = self._fit_ggd(image_mscn.flatten(1))
        delta_alpha = torch.log(patch_alpha + self.eps) - torch.log(
            image_alpha.unsqueeze(1) + self.eps
        )
        delta_sigma = torch.log(patch_sigma + self.eps) - torch.log(
            image_sigma.unsqueeze(1) + self.eps
        )
        scores = torch.sqrt(delta_alpha.square() + delta_sigma.square())

        top_scores, top_indices = scores.topk(self.num_patches, dim=1, sorted=True)
        candidate_boxes = self._candidate_boxes(height, width, device=image.device)
        boxes = candidate_boxes[top_indices]
        centers = 0.5 * (boxes[..., :2] + boxes[..., 2:])
        normalization = centers.new_tensor((width, height))
        centers = centers / normalization

        batch_indices = torch.arange(image.shape[0], device=image.device).unsqueeze(1)
        selected_alpha = patch_alpha[batch_indices, top_indices]
        selected_sigma = patch_sigma[batch_indices, top_indices]
        score_map = self._dense_score_map(scores, (height, width))
        return NSSSelection(
            score_map=score_map.to(dtype=image.dtype),
            patch_scores=top_scores.to(dtype=image.dtype),
            boxes=boxes,
            centers=centers.to(dtype=image.dtype),
            patch_alpha=selected_alpha.to(dtype=image.dtype),
            patch_sigma=selected_sigma.to(dtype=image.dtype),
        )

    def extra_repr(self) -> str:
        return (
            f"patch_size={self.patch_size}, patch_stride={self.patch_stride}, "
            f"num_patches={self.num_patches}, "
            f"local_kernel_size={self.local_kernel_size}, "
            f"local_sigma={self.local_sigma:.4g}"
        )
