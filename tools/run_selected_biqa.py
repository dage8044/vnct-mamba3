#!/usr/bin/env python3
"""Run a manifest-defined sequence of VNCT-BIQA experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY / "configs/experiments/vssd_small_ncm3_selected_single.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    selector_counts = [int(value) for value in manifest["selector_num_patches"]]
    if not selector_counts or any(value <= 0 for value in selector_counts):
        raise ValueError("selector_num_patches must contain positive integers")

    for experiment in manifest["experiments"]:
        experiment_path = REPOSITORY / experiment
        with experiment_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        for selected in selector_counts:
            for policy in manifest["policies"]:
                command = [
                    sys.executable,
                    "-u",
                    str(REPOSITORY / "tools/train_biqa.py"),
                    "--config",
                    str(experiment_path),
                    "--policy",
                    policy,
                    "--device",
                    args.device,
                    "--num-patches",
                    str(selected),
                ]
                architecture_config = manifest.get("architecture_config")
                if architecture_config:
                    command.extend(
                        (
                            "--architecture-config",
                            str(REPOSITORY / architecture_config),
                        )
                    )
                if args.max_train_steps:
                    command.extend(("--max-train-steps", str(args.max_train_steps)))
                if args.max_eval_steps:
                    command.extend(("--max-eval-steps", str(args.max_eval_steps)))
                timestamp = datetime.now().astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                )
                print(f"[{timestamp}] {' '.join(command)}", flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=REPOSITORY, check=True)


if __name__ == "__main__":
    main()
