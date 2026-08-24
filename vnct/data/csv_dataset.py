"""Dataset-agnostic CSV reader used by common BIQA benchmarks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class IQACsvDataset(Dataset[dict[str, object]]):
    """Read image paths and MOS/DMOS values from a metadata CSV.

    The CSV may include a ``split`` column. Splits must be generated at the
    reference-content level before constructing this dataset to prevent BIQA
    train/test leakage.
    """

    def __init__(
        self,
        metadata: str | Path,
        image_root: str | Path,
        *,
        split: str | None = None,
        image_column: str = "image",
        score_column: str = "mos",
        split_column: str = "split",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        with Path(metadata).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"metadata CSV is empty: {metadata}")
        required = {image_column, score_column}
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(f"metadata CSV is missing columns: {sorted(missing)}")
        if split is not None:
            if split_column not in rows[0]:
                raise ValueError(f"split={split!r} requested but column {split_column!r} is absent")
            rows = [row for row in rows if row[split_column] == split]
        self.samples = [
            (self.image_root / row[image_column], float(row[score_column]))
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        path, score = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            sample = self.transform(image) if self.transform is not None else image.copy()
        return {
            "image": sample,
            "score": torch.tensor(score, dtype=torch.float32),
            "path": str(path),
        }
