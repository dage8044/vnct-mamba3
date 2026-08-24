"""Fixed-split, multi-patch datasets for BIQA benchmarks."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class IQARecord:
    path: Path
    score: float
    reference: str
    index: int


def _live_reference_map(root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for info_path in sorted(root.glob("*/info.txt")):
        distortion = info_path.parent.name
        for raw_line in info_path.read_text(errors="replace").splitlines():
            fields = raw_line.strip().split()
            if len(fields) >= 2:
                mapping[f"{distortion}/{fields[1]}"] = fields[0]
    return mapping


def load_records(dataset: Mapping[str, Any]) -> list[IQARecord]:
    """Load configured metadata and derive reference identities when required."""

    metadata = Path(dataset["metadata"])
    image_root = Path(dataset["source_image_root"])
    metadata_format = dataset.get("metadata_format", "csv")
    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        if metadata_format == "csv":
            rows: list[Mapping[Any, str]] = list(csv.DictReader(handle))
        elif metadata_format == "whitespace_score_filename":
            rows = [
                {column: value for column, value in enumerate(line.split())}
                for line in handle
                if line.strip()
            ]
        else:
            raise ValueError(f"unsupported metadata format: {metadata_format}")
    image_column = dataset["image_column"]
    score_column = dataset["score_column"]
    mapping_mode = dataset.get("reference_mapping")
    reference_column = dataset.get("reference_column")
    reference_from_filename = dataset.get("reference_from_filename")
    case_insensitive_filenames = bool(
        dataset.get("case_insensitive_filenames", False)
    )
    casefolded_paths = (
        {
            candidate.name.casefold(): candidate
            for candidate in image_root.iterdir()
            if candidate.is_file()
        }
        if case_insensitive_filenames
        else {}
    )
    live_mapping = (
        _live_reference_map(Path(dataset["root"]))
        if mapping_mode == "distortion_subfolder_info_txt"
        else {}
    )

    records = []
    for index, row in enumerate(rows):
        image_name = row[image_column]
        if mapping_mode == "filename_prefix_before_first_dot":
            reference = image_name.split(".", 1)[0]
        elif mapping_mode == "distortion_subfolder_info_txt":
            try:
                reference = live_mapping[image_name]
            except KeyError as error:
                raise KeyError(
                    f"LIVE reference mapping is missing {image_name!r}"
                ) from error
        elif reference_column is not None:
            reference = row[reference_column].casefold()
        elif reference_from_filename == "first_3_characters":
            reference = image_name[:3].casefold()
        else:
            reference = str(index)
        path = image_root / image_name
        if not path.is_file() and case_insensitive_filenames:
            path = casefolded_paths.get(image_name.casefold(), path)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            IQARecord(
                path=path,
                score=float(row[score_column]),
                reference=reference,
                index=index,
            )
        )
    expected = int(dataset["data_num"])
    if len(records) != expected:
        raise ValueError(f"{dataset['name']} has {len(records)} records; expected {expected}")
    return records


def split_records(
    records: list[IQARecord],
    split: Mapping[str, Any],
) -> tuple[list[IQARecord], list[IQARecord]]:
    """Produce the one fixed seed-3407 split shared by both policies."""

    seed = int(split["seed"])
    ratio = float(split["ratio"][0])
    generator = random.Random(seed)
    method = split["method"]
    if method == "random_image_80_20":
        indices = list(range(len(records)))
        generator.shuffle(indices)
        # LoDa's authentic-dataset preprocessing uses Python ``round`` before
        # slicing (e.g. LIVEC 1162 -> 930 train images).
        boundary = round(ratio * len(indices))
        train_indices = set(indices[:boundary])
        train = [record for index, record in enumerate(records) if index in train_indices]
        test = [record for index, record in enumerate(records) if index not in train_indices]
    elif method == "reference_disjoint_80_20":
        references = sorted({record.reference for record in records})
        generator.shuffle(references)
        # LoDa uses round(0.8 * N); this matters for KADID's 81 references,
        # where 65 references (8,125 images) belong to the training split.
        boundary = round(ratio * len(references))
        train_references = set(references[:boundary])
        train = [record for record in records if record.reference in train_references]
        test = [record for record in records if record.reference not in train_references]
        if {record.reference for record in train} & {record.reference for record in test}:
            raise RuntimeError("reference content leaked across the train/test split")
    else:
        raise ValueError(f"unsupported split method: {method}")
    return train, test


class IQAPatchDataset(Dataset[dict[str, object]]):
    """Expose one random crop per item and retain an image id for aggregation."""

    def __init__(
        self,
        records: list[IQARecord],
        *,
        patch_num: int,
        crop_size: int,
        resize_shorter_edge: int | None,
        interpolation: str = "bilinear",
        training: bool,
        horizontal_flip: float = 0.0,
        vertical_flip: float = 0.0,
        seed: int = 3407,
    ) -> None:
        if patch_num <= 0:
            raise ValueError("patch_num must be positive")
        self.records = records
        self.patch_num = int(patch_num)
        self.crop_size = int(crop_size)
        self.resize_shorter_edge = resize_shorter_edge
        self.interpolation = InterpolationMode(interpolation)
        self.training = bool(training)
        self.horizontal_flip = float(horizontal_flip)
        self.vertical_flip = float(vertical_flip)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records) * self.patch_num

    def _rng(self, record: IQARecord, patch_index: int) -> random.Random:
        if self.training:
            return random
        return random.Random(self.seed + record.index * 1009 + patch_index)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index // self.patch_num]
        patch_index = index % self.patch_num
        rng = self._rng(record, patch_index)
        with Image.open(record.path) as handle:
            image = handle.convert("RGB")
            if self.resize_shorter_edge is not None:
                image = TF.resize(
                    image,
                    self.resize_shorter_edge,
                    interpolation=self.interpolation,
                    antialias=True,
                )
            width, height = image.size
            if height < self.crop_size or width < self.crop_size:
                image = TF.resize(
                    image,
                    [max(height, self.crop_size), max(width, self.crop_size)],
                    interpolation=self.interpolation,
                    antialias=True,
                )
                width, height = image.size
            top = rng.randint(0, height - self.crop_size)
            left = rng.randint(0, width - self.crop_size)
            image = TF.crop(image, top, left, self.crop_size, self.crop_size)
            if self.training and rng.random() < self.horizontal_flip:
                image = TF.hflip(image)
            if self.training and rng.random() < self.vertical_flip:
                image = TF.vflip(image)
            selector_image = TF.to_tensor(image)
        backbone_image = TF.normalize(selector_image, IMAGENET_MEAN, IMAGENET_STD)
        return {
            "backbone_image": backbone_image,
            "selector_image": selector_image,
            "score": torch.tensor(record.score, dtype=torch.float32),
            "image_id": torch.tensor(record.index, dtype=torch.int64),
            "path": str(record.path),
        }


__all__ = [
    "IQARecord",
    "IQAPatchDataset",
    "load_records",
    "split_records",
]
