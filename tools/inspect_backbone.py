#!/usr/bin/env python3
"""Inspect VSSD-Small checkpoint coverage and multi-scale feature shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vnct.models.backbones.vssd_small_ncm3 import vssd_small_ncm3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ncm3-scale", type=float, default=0.0)
    parser.add_argument("--mimo-rank", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = vssd_small_ncm3(
        pretrained=args.checkpoint,
        ncm3_scale_init=args.ncm3_scale,
        mimo_rank=args.mimo_rank,
    ).to(args.device).eval()
    if hasattr(model, "checkpoint_report"):
        print(f"checkpoint: {model.checkpoint_report}")
    image = torch.randn(1, 3, args.image_size, args.image_size, device=args.device)
    with torch.inference_mode():
        features = model(image)
    print("features:", [tuple(feature.shape) for feature in features])
    print("finite:", all(torch.isfinite(feature).all().item() for feature in features))


if __name__ == "__main__":
    main()
