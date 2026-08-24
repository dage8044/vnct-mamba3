#!/usr/bin/env python3
"""Visualize fixed or learned sparse selections beside BIQA head weights."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_biqa import REPOSITORY, build_model, load_yaml
from vnct.data import IQAPatchDataset, load_records, split_records


RANK_COLORS = (
    (255, 45, 45),
    (255, 166, 35),
    (40, 210, 255),
    (80, 230, 100),
    (210, 90, 255),
    (255, 235, 55),
    (40, 120, 255),
    (255, 110, 180),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of test images. LIVE uses one image per distortion when possible.",
    )
    parser.add_argument(
        "--crop-index",
        type=int,
        default=0,
        help="Deterministic test crop index used for every selected image.",
    )
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY / path


def _font(size: int = 13) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _heatmap(values: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    values = values.detach().float().cpu().clamp(0.0, 1.0)
    values = TF.resize(values.unsqueeze(0), list(reversed(size)), antialias=True)[0]
    array = values.numpy()
    red = np.clip(2.0 * array, 0.0, 1.0)
    green = np.clip(2.0 - 2.0 * np.abs(array - 0.5), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - array), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.uint8(np.round(rgb * 255.0)))


def _blend_heatmap(image: Image.Image, values: torch.Tensor) -> Image.Image:
    return Image.blend(image, _heatmap(values, image.size), alpha=0.45)


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().cpu()
    minimum, maximum = values.amin(), values.amax()
    return (values - minimum) / (maximum - minimum).clamp_min(1e-8)


def _caption(panel: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    header = 38
    canvas = Image.new("RGB", (panel.width, panel.height + header), "white")
    canvas.paste(panel, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 3), title, fill="black", font=_font(12))
    if subtitle:
        draw.text((5, 20), subtitle, fill=(55, 55, 55), font=_font(9))
    return canvas


def _draw_candidates_and_selection(
    image: Image.Image,
    *,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    patch_size: int,
    patch_stride: int,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    for y in range(0, height - patch_size + 1, patch_stride):
        for x in range(0, width - patch_size + 1, patch_stride):
            draw.rectangle(
                (x, y, x + patch_size - 1, y + patch_size - 1),
                outline=(205, 205, 205),
                width=1,
            )

    font = _font(10)
    for rank, (box, score) in enumerate(zip(boxes.tolist(), scores.tolist(), strict=True)):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=4)
        label = f"#{rank + 1} {score:.3f}"
        text_box = draw.textbbox((x1 + 3, y1 + 3), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x1 + 3, y1 + 3), label, fill=color, font=font)
    return output


def _draw_selection(
    image: Image.Image,
    *,
    boxes: torch.Tensor,
    scores: torch.Tensor,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = _font(10)
    for rank, (box, score) in enumerate(zip(boxes.tolist(), scores.tolist(), strict=True)):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=4)
        label = f"#{rank + 1} w={score:.2f}"
        text_box = draw.textbbox((x1 + 3, y1 + 3), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x1 + 3, y1 + 3), label, fill=color, font=font)
    return output


def _joint_head_map(details: Any) -> torch.Tensor:
    stage_weights = details.stage_weights[0].detach().float().cpu()
    spatial_weights = details.spatial_weights[0].detach().float().cpu()
    grid = int(round(spatial_weights.shape[-1] ** 0.5))
    stage_maps = spatial_weights.reshape(4, grid, grid)
    joint = (stage_weights[:, None, None] * stage_maps).sum(dim=0)
    minimum, maximum = joint.amin(), joint.amax()
    return (joint - minimum) / (maximum - minimum).clamp_min(1e-8)


def _select_record_indices(records: list[Any], count: int) -> list[int]:
    by_distortion: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        distortion = record.path.parent.name
        by_distortion.setdefault(distortion, []).append(index)
    if len(by_distortion) >= count:
        chosen = []
        for distortion in sorted(by_distortion)[:count]:
            candidates = by_distortion[distortion]
            candidates.sort(key=lambda value: records[value].score)
            chosen.append(candidates[len(candidates) // 2])
        return chosen

    ordered = sorted(range(len(records)), key=lambda index: records[index].score)
    if count == 1:
        return [ordered[len(ordered) // 2]]
    return [ordered[round(i * (len(ordered) - 1) / (count - 1))] for i in range(count)]


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.scale <= 0:
        raise ValueError("samples and scale must be positive")

    checkpoint_path = _resolve(args.checkpoint)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device(args.device)

    model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    dataset_config = load_yaml(Path(config["data"]["config"]))["dataset"]
    records = load_records(dataset_config)
    _, test_records = split_records(records, dataset_config["split"])
    crop_size = int(dataset_config["crop_size"])
    dataset = IQAPatchDataset(
        test_records,
        patch_num=args.crop_index + 1,
        crop_size=crop_size,
        resize_shorter_edge=dataset_config.get("resize_shorter_edge"),
        interpolation=str(dataset_config.get("interpolation", "bilinear")),
        training=False,
        seed=int(config["experiment"]["random_seed"]),
    )
    indices = _select_record_indices(test_records, min(args.samples, len(test_records)))

    rows: list[Image.Image] = []
    csv_rows: list[dict[str, object]] = []
    for sample_number, record_index in enumerate(indices, start=1):
        dataset_index = record_index * dataset.patch_num + args.crop_index
        sample = dataset[dataset_index]
        selector_image = sample["selector_image"].unsqueeze(0).to(device)
        backbone_image = sample["backbone_image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            details = model(backbone_image, selector_image, return_details=True)

        image = TF.to_pil_image(sample["selector_image"])
        head_map = _joint_head_map(details)
        path = Path(str(sample["path"]))
        prediction = float(details.score[0].detach().cpu())
        target = float(sample["score"])

        original = _caption(
            image,
            "Model input crop",
            f"{path.parent.name}/{path.name}  DMOS={target:.3f}",
        )
        learned = _caption(
            _blend_heatmap(image, head_map),
            "Learned quality-head weight",
            f"not selector; prediction={prediction:.3f}",
        )
        panels = [original]
        if details.selection is not None:
            selection = details.selection
            boxes = selection.boxes[0].cpu()
            scores = selection.patch_scores[0].float().cpu()
            selector_map = selection.score_map[0, 0].float().cpu()
            assert model.selector is not None
            panels.extend(
                (
                    _caption(
                        _blend_heatmap(image, selector_map),
                        "MSCN/GGD suspicion",
                        "blue=low, red=high (fixed selector)",
                    ),
                    _caption(
                        _draw_candidates_and_selection(
                            image,
                            boxes=boxes,
                            scores=scores,
                            patch_size=model.selector.patch_size,
                            patch_stride=model.selector.patch_stride,
                        ),
                        f"Top-{model.selector.num_patches} selected",
                        "gray=all candidates; color=selection rank",
                    ),
                )
            )
            selections_for_csv = [(0, selection)]
        else:
            selections_for_csv = list(enumerate(details.stage_selections, start=1))
            for stage, selection in selections_for_csv:
                valid = selection.valid_mask[0].cpu()
                boxes = selection.boxes[0, valid].float().cpu()
                weights = selection.gain_shares[0, valid].float().cpu()
                importance = _normalize_map(selection.score_map[0, 0])
                overlay = _blend_heatmap(image, importance)
                panels.append(
                    _caption(
                        _draw_selection(overlay, boxes=boxes, scores=weights),
                        f"Stage {stage} learned importance",
                        "map=min-max; boxes=active marginal coverage; label=gain share",
                    )
                )
        panels.append(learned)
        row = Image.new(
            "RGB",
            (original.width * len(panels), original.height),
            "white",
        )
        for column, panel in enumerate(panels):
            row.paste(panel, (column * original.width, 0))
        if args.scale != 1:
            row = row.resize(
                (row.width * args.scale, row.height * args.scale),
                Image.Resampling.NEAREST,
            )
        row_path = output_dir / f"sample_{sample_number:02d}_{path.parent.name}_{path.stem}.png"
        row.save(row_path)
        rows.append(row)

        for stage, selection in selections_for_csv:
            if hasattr(selection, "marginal_gains"):
                valid = selection.valid_mask[0].cpu()
                boxes = selection.boxes[0, valid].float().cpu()
                scores = selection.marginal_gains[0, valid].float().cpu()
                weights = selection.gain_shares[0, valid].float().cpu()
            else:
                boxes = selection.boxes[0].float().cpu()
                scores = selection.patch_scores[0].float().cpu()
                weights = scores
            for rank, (box, score, weight) in enumerate(
                zip(boxes.tolist(), scores.tolist(), weights.tolist(), strict=True),
                start=1,
            ):
                csv_rows.append(
                    {
                        "sample": sample_number,
                        "path": str(path),
                        "crop_index": args.crop_index,
                        "dmos": target,
                        "prediction": prediction,
                        "stage": stage,
                        "rank": rank,
                        "score": score,
                        "weight": weight,
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                    }
                )

    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output_dir / "selector_contact_sheet.png")

    with (output_dir / "selections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"checkpoint={checkpoint_path}")
    print(f"epoch={checkpoint['epoch']} best_srcc={checkpoint.get('best_srcc')}")
    print(f"samples={len(rows)} output={output_dir / 'selector_contact_sheet.png'}")


if __name__ == "__main__":
    main()
