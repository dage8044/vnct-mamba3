#!/usr/bin/env python3
"""Train one fixed-split VNCT-BIQA experiment under one parameter policy."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from vnct.data import IQAPatchDataset, load_records, split_records
from vnct.engine import (
    apply_training_mode,
    build_optimizer,
    configure_training_policy,
)
from vnct.losses import LoDaPLCCLoss
from vnct.metrics import compute_loda_metrics
from vnct.models import vssd_small_ncm3_biqa, vssd_small_original_biqa
from vnct.models.selectors import MSCNGGDSelector


REPOSITORY = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--architecture-config",
        type=Path,
        help="Override only the model architecture while retaining the dataset protocol.",
    )
    parser.add_argument(
        "--policy", choices=("new_modules_only", "full"), required=True
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--num-patches",
        type=int,
        help="Override fixed Top-K or the learned selector maximum ROI budget.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Positive values enable a bounded smoke run per epoch.",
    )
    parser.add_argument(
        "--max-eval-steps",
        type=int,
        default=0,
        help="Positive values bound evaluation for a smoke run.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    path = path if path.is_absolute() else REPOSITORY / path
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_initialization(config: dict[str, Any]) -> dict[str, Any]:
    architecture = load_yaml(Path(config["experiment"]["architecture_config"]))
    model_config = architecture["model"]
    path = config["model"].get(
        "initialization_config", model_config["initialization_config"]
    )
    return load_yaml(Path(path))["initialization"]


def resolve_backbone_name(config: dict[str, Any]) -> str:
    architecture = load_yaml(Path(config["experiment"]["architecture_config"]))
    return str(architecture["model"]["backbone"])


def resolve_mimo_rank(config: dict[str, Any]) -> int | None:
    architecture = load_yaml(Path(config["experiment"]["architecture_config"]))
    if architecture["model"]["backbone"] == "vssd_small_original":
        return None
    return int(config["model"].get("mimo_rank", architecture["model"]["mimo_rank"]))


def resolve_selector(config: dict[str, Any]) -> dict[str, Any]:
    architecture = load_yaml(Path(config["experiment"]["architecture_config"]))
    return {**architecture["selector"], **config.get("selector", {})}


def resolve_quality_head_name(config: dict[str, Any]) -> str:
    architecture = load_yaml(Path(config["experiment"]["architecture_config"]))
    return str(architecture["quality_head"]["name"])


def default_run_name(
    config: dict[str, Any], initialization: dict[str, Any], seed: int
) -> str:
    """Keep incompatible architecture revisions in separate result folders."""
    selector = resolve_selector(config)
    selector_name = str(selector.get("name", "mscn_ggd"))
    if selector_name == "none":
        selector_tag = "local_only"
    else:
        selected = int(selector.get("max_regions", selector.get("num_patches", 0)))
        revision = str(selector.get("revision", "")).strip()
        if selector_name == "mscn_ggd":
            selector_tag = f"k{selected}"
        elif revision:
            selector_tag = f"{selector_name}_{revision}_k{selected}"
        else:
            selector_tag = f"{selector_name}_k{selected}"
    backbone_name = resolve_backbone_name(config)
    backbone_tag = (
        "vssd_original"
        if backbone_name == "vssd_small_original"
        else f"rank{resolve_mimo_rank(config)}"
    )
    head_tag = resolve_quality_head_name(config).replace("_patch_weighted", "_pw")
    return (
        f"{backbone_tag}_{initialization['profile']}_{selector_tag}_"
        f"{head_tag}_seed{seed}"
    )


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    architecture_path = config["experiment"]["architecture_config"]
    architecture = load_yaml(Path(architecture_path))
    model_config = architecture["model"]
    selector_config = resolve_selector(config)
    local = architecture["local_path"]
    refinement = architecture["refinement"]
    interaction = architecture["interaction"]
    head = architecture["quality_head"]
    initialization = resolve_initialization(config)
    selector_name = str(selector_config.get("name", "mscn_ggd"))
    selector: MSCNGGDSelector | None
    refinement_enabled = bool(refinement["enabled"])
    if selector_name == "mscn_ggd":
        selector = MSCNGGDSelector(
            patch_size=int(selector_config["patch_size"]),
            patch_stride=int(selector_config["patch_stride"]),
            num_patches=int(selector_config["num_patches"]),
            local_kernel_size=int(selector_config["local_kernel_size"]),
            local_sigma=float(selector_config["local_sigma"]),
        )
    elif selector_name in ("none", "learned_importance"):
        selector = None
    else:
        raise ValueError(f"unsupported selector name: {selector_name!r}")
    pretrained = Path(config["model"].get("pretrained", model_config["pretrained"]))
    if not pretrained.is_absolute():
        pretrained = REPOSITORY / pretrained
    common_arguments = dict(
        pretrained=pretrained,
        selector=selector,
        selector_mode=selector_name,
        enable_refinement=refinement_enabled,
        importance_max_regions=int(
            selector_config.get("max_regions", selector_config.get("num_patches", 4))
        ),
        importance_region_sizes=tuple(
            selector_config.get("region_sizes", refinement["roi_sizes"])
        ),
        importance_coverage_threshold=float(
            selector_config.get("coverage_threshold", 0.8)
        ),
        importance_candidate_stride=int(selector_config.get("candidate_stride", 1)),
        importance_decoder_kernel_size=int(
            selector_config.get("decoder_kernel_size", 5)
        ),
        importance_output_init_std=float(
            selector_config.get("output_init_std", 1e-3)
        ),
        refinement_state_dims=tuple(refinement["state_dims"]),
        refinement_heads=tuple(refinement["num_heads"]),
        roi_sizes=tuple(refinement["roi_sizes"]),
        interaction_dims=tuple(interaction["inner_dims"]),
        interaction_heads=int(interaction["num_heads"]),
        local_grid_size=int(interaction["local_grid_size"]),
        local_dilation=int(local["dilation"]),
        local_ffn_ratio=float(local["ffn_ratio"]),
        local_residual_scale_init=float(initialization["local_residual_scale"]),
        refinement_residual_scale_init=float(
            initialization["refinement_residual_scale"]
        ),
        head_name=str(head["name"]),
        head_embed_dim=int(head["embed_dim"]),
        head_grid_size=int(head["grid_size"]),
        head_num_heads=int(head.get("num_heads", 4)),
        head_depth=int(head.get("depth", 1)),
        head_mlp_ratio=float(head.get("mlp_ratio", 3.0)),
        head_dropout=float(head.get("dropout", 0.0)),
        dropout=float(interaction.get("dropout", 0.0)),
    )
    backbone_name = str(model_config["backbone"])
    if backbone_name == "vssd_small_ncm3":
        mimo_rank = resolve_mimo_rank(config)
        assert mimo_rank is not None
        return vssd_small_ncm3_biqa(
            ncm3_scale_init=float(initialization["ncm3_scale"]),
            mimo_rank=mimo_rank,
            adapter_init_std=float(initialization["adapter_init_std"]),
            **common_arguments,
        )
    if backbone_name == "vssd_small_original":
        return vssd_small_original_biqa(**common_arguments)
    raise ValueError(f"unsupported backbone: {backbone_name!r}")


def build_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    dataset_config = load_yaml(Path(config["data"]["config"]))["dataset"]
    records = load_records(dataset_config)
    train_records, test_records = split_records(records, dataset_config["split"])
    expected = (int(config["data"]["train_data_num"]), int(config["data"]["test_data_num"]))
    actual = (len(train_records), len(test_records))
    if actual != expected:
        raise ValueError(f"configured train/test sizes {expected} do not match fixed split {actual}")
    crop_size = int(dataset_config["crop_size"])
    resize = dataset_config.get("resize_shorter_edge")
    interpolation = str(dataset_config.get("interpolation", "bilinear"))
    train = config["train"]
    test = config["test"]
    seed = int(config["experiment"]["random_seed"])
    train_dataset = IQAPatchDataset(
        train_records,
        patch_num=int(train["patch_num"]),
        crop_size=crop_size,
        resize_shorter_edge=resize,
        interpolation=interpolation,
        training=True,
        horizontal_flip=float(train.get("random_horizontal_flip", 0.0)),
        vertical_flip=float(train.get("random_vertical_flip", 0.0)),
        seed=seed,
    )
    test_dataset = IQAPatchDataset(
        test_records,
        patch_num=int(test["patch_num"]),
        crop_size=crop_size,
        resize_shorter_edge=resize,
        interpolation=interpolation,
        training=False,
        seed=seed,
    )
    common = {"persistent_workers": False}
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train["batch_size"]),
        shuffle=bool(train["shuffle"]),
        num_workers=int(train["num_workers"]),
        pin_memory=bool(train["pin_memory"]),
        drop_last=bool(train["drop_last"]),
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(test["batch_size"]),
        shuffle=False,
        num_workers=int(test["num_workers"]),
        pin_memory=bool(test["pin_memory"]),
        drop_last=False,
        **common,
    )
    return train_loader, test_loader, {"train_images": actual[0], "test_images": actual[1]}


def move_batch(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["backbone_image"].to(device, non_blocking=True),
        batch["selector_image"].to(device, non_blocking=True),
        batch["score"].to(device, non_blocking=True),
    )


def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _restore_rng_state(
    state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device
) -> None:
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def chunked_batch_loss_backward(
    model: torch.nn.Module,
    backbone_image: torch.Tensor,
    selector_image: torch.Tensor,
    target: torch.Tensor,
    loss_function: torch.nn.Module,
    chunk_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Differentiate an exact logical-batch PLCC loss through micro-forwards.

    LoDa's loss depends on batch-wide moments, so ordinary gradient accumulation
    over independent microbatch losses would change the objective.  We first
    obtain dL/dprediction for the complete logical batch, then replay each
    stochastic forward with its saved RNG state and apply that output gradient.
    """

    ranges = [
        (start, min(start + chunk_size, target.numel()))
        for start in range(0, target.numel(), chunk_size)
    ]
    states = []
    detached_predictions = []
    with torch.no_grad():
        for start, end in ranges:
            states.append(_rng_state(device))
            detached_predictions.append(
                model(backbone_image[start:end], selector_image[start:end]).float()
            )
    probe = torch.cat(detached_predictions).detach().requires_grad_(True)
    loss = loss_function(probe, target)
    output_gradient = torch.autograd.grad(loss, probe)[0]
    for (start, end), state in zip(ranges, states, strict=True):
        _restore_rng_state(state, device)
        prediction = model(backbone_image[start:end], selector_image[start:end])
        replay_objective = (
            prediction * output_gradient[start:end].to(dtype=prediction.dtype)
        ).sum()
        replay_objective.backward()
    return {"total": loss.detach(), "main": loss.detach()}


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_function: torch.nn.Module,
    device: torch.device,
    policy: str,
    max_steps: int,
    forward_chunk_size: int,
) -> dict[str, float]:
    apply_training_mode(model, policy)
    accumulated = {"total": 0.0, "main": 0.0}
    count = 0
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        backbone_image, selector_image, target = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if forward_chunk_size and target.numel() > forward_chunk_size:
            losses = chunked_batch_loss_backward(
                model,
                backbone_image,
                selector_image,
                target,
                loss_function,
                forward_chunk_size,
                device,
            )
        else:
            prediction = model(backbone_image, selector_image)
            main_loss = loss_function(prediction, target)
            losses = {"total": main_loss, "main": main_loss}
        if not all(torch.isfinite(value) for value in losses.values()):
            values = {name: float(value.detach()) for name, value in losses.items()}
            raise FloatingPointError(f"non-finite loss at step {step}: {values}")
        if losses["total"].requires_grad:
            losses["total"].backward()
        optimizer.step()
        scheduler.step()
        batch_size = target.numel()
        for name, value in losses.items():
            accumulated[name] += float(value.detach()) * batch_size
        count += batch_size
    if not count:
        raise RuntimeError("training loader produced no optimizer steps")
    return {name: value / count for name, value in accumulated.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    forward_chunk_size: int,
    max_steps: int = 0,
) -> dict[str, float]:
    model.eval()
    predictions: dict[int, list[float]] = defaultdict(list)
    targets: dict[int, float] = {}
    update_stats: dict[str, list[float]] = defaultdict(list)
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        backbone_image, selector_image, target = move_batch(batch, device)
        chunk_size = forward_chunk_size or target.numel()
        prediction = torch.cat(
            [
                model(backbone_image[start : start + chunk_size], selector_image[start : start + chunk_size])
                for start in range(0, target.numel(), chunk_size)
            ]
        ).float().cpu()
        for stage, interaction in enumerate(model.interactions, start=1):
            for name, value in interaction.last_stats.items():
                update_stats[f"stage{stage}_{name}"].append(float(value.cpu()))
        image_ids = batch["image_id"].tolist()
        for image_id, value, score in zip(image_ids, prediction.tolist(), target.cpu().tolist(), strict=True):
            predictions[int(image_id)].append(float(value))
            targets[int(image_id)] = float(score)
    ordered = sorted(predictions)
    image_prediction = [float(np.mean(predictions[index])) for index in ordered]
    image_target = [targets[index] for index in ordered]
    metrics = compute_loda_metrics(image_prediction, image_target)
    metrics["val_loss"] = float(
        LoDaPLCCLoss()(
            torch.tensor(image_prediction, dtype=torch.float32),
            torch.tensor(image_target, dtype=torch.float32),
        )
    )
    metrics.update(
        {name: float(np.mean(values)) for name, values in update_stats.items()}
    )
    return metrics


def diagnostic_statistics(model: torch.nn.Module) -> dict[str, float]:
    stats: dict[str, float] = {}
    for stage, interaction in enumerate(model.interactions, start=1):
        for name, value in interaction.last_stats.items():
            stats[f"stage{stage}_{name}"] = float(value.detach().cpu())
    for stage, mixer in enumerate(model.local_mixers, start=1):
        stats[f"delta_{stage}"] = float(mixer.residual_scale.detach().cpu())
    for stage, refiner in enumerate(model.refiners, start=1):
        stats[f"gamma_{stage}"] = float(refiner.residual_scale.detach().cpu())
    for stage, selector in enumerate(model.importance_selectors, start=1):
        for name, value in selector.last_stats.items():
            stats[f"stage{stage}_selector_{name}"] = float(value.detach().cpu())
        map_gradient = selector.decoder[-1].weight.grad
        if map_gradient is not None:
            stats[f"stage{stage}_selector_map_grad_abs_mean"] = float(
                map_gradient.detach().float().abs().mean().cpu()
            )
    scales = [
        float(module.ncm3_scale.detach().cpu())
        for module in model.backbone.modules()
        if hasattr(module, "ncm3_scale")
    ]
    if scales:
        stats["ncm3_scale_mean"] = float(np.mean(scales))
        stats["ncm3_scale_abs_mean"] = float(np.mean(np.abs(scales)))
    return stats


def _learning_rate(optimizer: torch.optim.Optimizer, origin: str) -> float:
    for group in optimizer.param_groups:
        if str(group.get("name", "")).startswith(origin):
            return float(group["lr"])
    return 0.0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:d}m {seconds:02d}s"


def _console_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.architecture_config is not None:
        config["experiment"]["architecture_config"] = str(
            args.architecture_config
        )
    if args.num_patches is not None:
        if args.num_patches <= 0:
            raise ValueError("--num-patches must be positive")
        config.setdefault("selector", {})["num_patches"] = args.num_patches
        config["selector"]["max_regions"] = args.num_patches
    seed = int(config["experiment"]["random_seed"])
    seed_everything(seed)
    requested_device = args.device or config["runtime"]["device"]
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    initialization = resolve_initialization(config)
    selector_config = resolve_selector(config)
    if float(selector_config.get("auxiliary_loss_weight", 0.0)) != 0.0:
        raise ValueError("the LoDa configuration uses final PLCC loss only")
    model = build_model(config)
    policy_report = configure_training_policy(model, args.policy)
    model.to(device)
    optimizer_config = config["optimizer"]
    new_lr = float(optimizer_config["learning_rate"])
    inherited_lr = float(optimizer_config.get("inherited_learning_rate", new_lr * 0.1))
    optimizer = build_optimizer(
        model,
        policy=args.policy,
        new_learning_rate=new_lr,
        inherited_learning_rate=inherited_lr,
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler_config = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_config["t_max"]),
        eta_min=float(scheduler_config["eta_min"]),
    )
    if args.max_train_steps:
        config["train"]["batch_size"] = min(int(config["train"]["batch_size"]), 4)
        config["train"]["num_workers"] = 0
        config["train"]["drop_last"] = False
    if args.max_eval_steps:
        # Two complete 15-crop images are enough to exercise aggregation and
        # metric plumbing without launching the full validation pass.
        config["test"]["batch_size"] = min(int(config["test"]["batch_size"]), 30)
        config["test"]["num_workers"] = 0
    train_loader, test_loader, split_stats = build_loaders(config)
    forward_chunk_size = int(config["runtime"].get("forward_chunk_size", 8))
    calibration: dict[str, float] = {}
    run_name = default_run_name(config, initialization, seed)
    output_dir = args.output_dir or (
        REPOSITORY / "outputs" / config["data"]["name"] / args.policy / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "config": str(args.config),
        "policy": args.policy,
        "seed": seed,
        "run_name": run_name,
        "mimo_rank": resolve_mimo_rank(config),
        "selector": resolve_selector(config),
        "quality_head": resolve_quality_head_name(config),
        "checkpoint_report": str(model.backbone.checkpoint_report),
        "parameter_report": policy_report.__dict__,
        "initialization": initialization,
        "interaction_calibration": calibration,
        **split_stats,
    }
    (output_dir / "run.json").write_text(json.dumps(run_metadata, indent=2))

    metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    best_srcc = float("-inf")
    best_epoch = 0
    started = time.perf_counter()
    epochs = 1 if args.max_train_steps else int(config["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        print(
            f"[{_console_timestamp()}] "
            f"[{config['data']['name']} | {args.policy}] "
            f"Epoch {epoch:02d}/{epochs:02d} | training started",
            flush=True,
        )
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            LoDaPLCCLoss(),
            device,
            args.policy,
            args.max_train_steps,
            forward_chunk_size,
        )
        train_loss = train_losses["total"]
        train_time = time.perf_counter() - epoch_started
        print(
            f"[{_console_timestamp()}] "
            f"[{config['data']['name']} | {args.policy}] "
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_time {_format_duration(train_time)} | "
            f"train_loss {train_loss:.6f} | evaluating",
            flush=True,
        )
        metrics = evaluate(
            model,
            test_loader,
            device,
            forward_chunk_size=forward_chunk_size,
            max_steps=args.max_eval_steps,
        )
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        epoch_time = time.perf_counter() - epoch_started
        total_time = time.perf_counter() - started
        score_metrics = {
            name: float(metrics[name]) for name in ("srcc", "plcc", "krcc", "rmse")
        }
        val_loss = float(metrics["val_loss"])
        is_best = score_metrics["srcc"] > best_srcc
        if is_best:
            best_srcc = score_metrics["srcc"]
            best_epoch = epoch
        diagnostic_metrics = {
            name: value
            for name, value in metrics.items()
            if name not in score_metrics and name != "val_loss"
        }
        metric_row: dict[str, object] = {
            "timestamp": timestamp,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **score_metrics,
            "is_best_srcc": is_best,
            "best_srcc_so_far": best_srcc,
            "best_epoch_so_far": best_epoch,
            "lr_new": _learning_rate(optimizer, "new"),
            "lr_inherited": _learning_rate(optimizer, "inherited"),
            "epoch_time_sec": epoch_time,
            "total_time_sec": total_time,
        }
        diagnostic_row: dict[str, object] = {
            "timestamp": timestamp,
            "epoch": epoch,
            "train_main_loss": train_losses["main"],
            **diagnostic_statistics(model),
            **diagnostic_metrics,
        }
        metric_rows.append(metric_row)
        diagnostic_rows.append(diagnostic_row)
        _write_csv(output_dir / "log.csv", metric_rows)
        _write_csv(output_dir / "diagnostics.csv", diagnostic_rows)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "policy": args.policy,
            "metrics": score_metrics,
            "diagnostics": diagnostic_row,
            "best_srcc": best_srcc,
            "best_epoch": best_epoch,
        }
        torch.save(checkpoint, output_dir / "last.pth")
        if is_best:
            torch.save(checkpoint, output_dir / "best.pth")
        print(
            f"[{_console_timestamp()}] "
            f"[{config['data']['name']} | {args.policy}] "
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"time {_format_duration(epoch_time)} | "
            f"total {_format_duration(total_time)} | "
            f"train_loss {train_loss:.6f} | "
            f"val_loss {val_loss:.6f} | "
            f"SRCC {score_metrics['srcc']:.4f} | "
            f"PLCC {score_metrics['plcc']:.4f} | "
            f"KRCC {score_metrics['krcc']:.4f} | "
            f"RMSE {score_metrics['rmse']:.4f}"
            f"{' | BEST SRCC' if is_best else ''}",
            flush=True,
        )
    print(
        f"[{_console_timestamp()}] completed in "
        f"{(time.perf_counter() - started) / 60.0:.2f} minutes: {output_dir}"
    )


if __name__ == "__main__":
    main()
