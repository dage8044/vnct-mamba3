import csv

import numpy as np
import pytest
import torch
from PIL import Image
from pathlib import Path

from vnct.data import IQACsvDataset, IQARecord, load_records, split_records
from vnct.losses import LoDaPLCCLoss
from vnct.metrics import compute_iqa_metrics, compute_loda_metrics


def test_iqa_metrics_perfect_order() -> None:
    metrics = compute_iqa_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert metrics["srcc"] == pytest.approx(1.0)
    assert metrics["plcc"] == pytest.approx(1.0)
    assert metrics["krcc"] == pytest.approx(1.0)
    assert metrics["rmse"] == 0.0


def test_csv_dataset_filters_split(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "mos", "split"])
        writer.writeheader()
        writer.writerow({"image": image_path.name, "mos": "3.5", "split": "train"})
        writer.writerow({"image": image_path.name, "mos": "2.0", "split": "test"})

    dataset = IQACsvDataset(metadata, tmp_path, split="train")
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["score"].item() == 3.5
    assert sample["image"].size == (8, 8)


def test_loda_plcc_loss_matches_released_formula() -> None:
    prediction = torch.tensor([[0.2], [0.8], [0.5], [0.1]], requires_grad=True)
    target = torch.tensor([[1.0], [4.0], [3.0], [2.0]])
    prediction_std, prediction_mean = torch.std_mean(prediction, unbiased=False)
    normalized_prediction = (prediction - prediction_mean) / (prediction_std + 1e-8)
    target_std, target_mean = torch.std_mean(target, unbiased=False)
    normalized_target = (target - target_mean) / (target_std + 1e-8)
    loss0 = torch.nn.functional.mse_loss(normalized_prediction, normalized_target) / 4
    rho = torch.mean(normalized_prediction * normalized_target)
    loss1 = torch.nn.functional.mse_loss(
        rho * normalized_prediction, normalized_target
    ) / 4
    expected = (loss0 + loss1) / 2

    actual = LoDaPLCCLoss()(prediction, target)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()


def test_reference_split_is_disjoint_and_reproducible() -> None:
    records = [
        IQARecord(Path(f"{reference}_{index}.png"), float(index), reference, index)
        for index, reference in enumerate(("a", "a", "b", "b", "c", "c", "d", "d", "e", "e"))
    ]
    specification = {
        "method": "reference_disjoint_80_20",
        "ratio": [0.8, 0.2],
        "seed": 3407,
    }

    train_a, test_a = split_records(records, specification)
    train_b, test_b = split_records(records, specification)

    assert [record.index for record in train_a] == [record.index for record in train_b]
    assert [record.index for record in test_a] == [record.index for record in test_b]
    assert {record.reference for record in train_a}.isdisjoint(
        {record.reference for record in test_a}
    )


def test_load_records_supports_whitespace_metadata_and_casefolded_reference(
    tmp_path,
) -> None:
    for name in ("I01_01_1.bmp", "i01_01_2.bmp"):
        (tmp_path / name).touch()
    metadata = tmp_path / "mos_with_names.txt"
    metadata.write_text("5.5 I01_01_1.bmp\n4.5 i01_01_2.bmp\n")
    records = load_records(
        {
            "name": "tid2013",
            "metadata": str(metadata),
            "metadata_format": "whitespace_score_filename",
            "source_image_root": str(tmp_path),
            "image_column": 1,
            "score_column": 0,
            "reference_from_filename": "first_3_characters",
            "data_num": 2,
        }
    )

    assert [record.score for record in records] == [5.5, 4.5]
    assert [record.reference for record in records] == ["i01", "i01"]


def test_load_records_uses_explicit_csv_reference_column(tmp_path) -> None:
    for name in ("I01_01.png", "I01_02.png"):
        (tmp_path / name).touch()
    metadata = tmp_path / "scores.csv"
    metadata.write_text(
        "image,dmos,reference\nI01_01.png,4.5,I01.png\nI01_02.png,3.5,I01.png\n"
    )
    records = load_records(
        {
            "name": "kadid10k",
            "metadata": str(metadata),
            "metadata_format": "csv",
            "source_image_root": str(tmp_path),
            "image_column": "image",
            "score_column": "dmos",
            "reference_column": "reference",
            "data_num": 2,
        }
    )

    assert [record.reference for record in records] == ["i01.png", "i01.png"]


def test_loda_metrics_apply_logistic_mapping_only_to_linear_metrics() -> None:
    prediction = np.linspace(-2.0, 2.0, 40)
    target = 1.0 + 4.0 / (1.0 + np.exp(-prediction))

    metrics = compute_loda_metrics(prediction, target)

    assert metrics["srcc"] == pytest.approx(1.0)
    assert metrics["plcc"] > 0.999
    assert metrics["rmse"] < 1e-3
