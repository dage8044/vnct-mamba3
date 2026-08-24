#!/usr/bin/env python3
"""Validate all local VSSD/NC-M3 configs against released LoDa constants."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY / "configs/experiments/vssd_small_ncm3_loda_all.yaml"

EXPECTED = {
    "koniq10k": (10073, 8058, 2015, 3, 1888),
    "kadid10k": (10125, 8100, 2025, 3, 1898),
    "spaq": (11125, 8900, 2225, 3, 2085),
    "livec": (1162, 930, 232, 5, 363),
    # LIVE uses a sorted reference list for a reproducible seed-3407 split.
    "live": (779, 615, 164, 5, 240),
    "tid2013": (3000, 2400, 600, 10, 1875),
    "flive": (39810, 31848, 7962, 1, 2488),
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def _metadata_rows(dataset: dict) -> int:
    path = Path(dataset["metadata"])
    if dataset["metadata_format"] == "csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return sum(1 for _ in csv.reader(handle)) - 1
    if dataset["metadata_format"] == "whitespace_score_filename":
        return sum(bool(line.strip()) for line in path.read_text().splitlines())
    raise ValueError(f"unsupported metadata format: {dataset['metadata_format']}")


def main() -> None:
    manifest = _load(MANIFEST)
    experiments = manifest["experiments"]
    if len(experiments) != len(EXPECTED):
        raise AssertionError(f"expected {len(EXPECTED)} experiments, got {len(experiments)}")

    for relative_experiment in experiments:
        experiment_path = REPOSITORY / relative_experiment
        experiment = _load(experiment_path)
        data = experiment["data"]
        dataset_path = REPOSITORY / data["config"]
        dataset = _load(dataset_path)["dataset"]
        name = data["name"]
        if name != dataset["name"]:
            raise AssertionError(f"dataset name mismatch in {experiment_path}")

        total, train, test, train_patches, t_max = EXPECTED[name]
        checks = {
            "data_num": dataset["data_num"] == total,
            "train_data_num": data["train_data_num"] == train,
            "test_data_num": data["test_data_num"] == test,
            "train_patch_num": experiment["train"]["patch_num"] == train_patches,
            "test_patch_num": experiment["test"]["patch_num"] == 15,
            "epochs": experiment["train"]["epochs"] == 10,
            "train_batch": experiment["train"]["batch_size"] == 128,
            "test_batch": experiment["test"]["batch_size"] == 512,
            "t_max": experiment["scheduler"]["t_max"] == t_max,
            "seed": experiment["experiment"]["random_seed"] == 3407,
            "splits": experiment["experiment"]["num_splits"] in (1, 10),
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            raise AssertionError(f"{name}: invalid LoDa values: {failed}")

        root_ready = Path(dataset["source_image_root"]).is_dir()
        metadata_ready = Path(dataset["metadata"]).is_file()
        ready = root_ready and metadata_ready
        if dataset["available"] != ready:
            raise AssertionError(
                f"{name}: available={dataset['available']} but local paths report {ready}"
            )
        if ready:
            rows = _metadata_rows(dataset)
            if rows != total:
                raise AssertionError(f"{name}: metadata has {rows} rows, expected {total}")
        print(
            f"{name:9s} {'READY' if ready else 'MISSING':7s} "
            f"images={dataset['source_image_root']} metadata_rows={total if ready else '-'}"
        )


if __name__ == "__main__":
    main()
