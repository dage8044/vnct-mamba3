"""Parameter policies for checkpoint-preserving and full VNCT training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from vnct.models.backbones.vssd_small_ncm3 import VSSDCheckpointNCM3


POLICIES = ("new_modules_only", "full")


@dataclass(frozen=True)
class TrainingPolicyReport:
    """Parameter counts after applying one of the two supported policies."""

    policy: str
    total: int
    trainable: int
    inherited_trainable: int
    new_trainable: int


def _validate_policy(policy: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"training policy must be one of {POLICIES}, got {policy!r}")


def _is_new_backbone_parameter(name: str) -> bool:
    return name.endswith(
        (
            "trap_weight",
            "trap_bias",
            "decay_weight",
            "decay_bias",
            "mimo_source_weight",
            "mimo_query_weight",
            "mimo_in",
            "mimo_out",
            "ncm3_scale",
        )
    )


def _is_new_parameter(name: str) -> bool:
    return not name.startswith("backbone.") or _is_new_backbone_parameter(name)


def configure_training_policy(model: nn.Module, policy: str) -> TrainingPolicyReport:
    """Apply ``new_modules_only`` or ``full`` without stage-wise exceptions.

    The checkpoint-compatible NC-M3 bridge lives inside the inherited backbone,
    so classifying parameters solely from the ``backbone.`` prefix would freeze
    the very transition that the checkpoint-preserving policy is meant to train.
    """

    _validate_policy(policy)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(policy == "full" or _is_new_parameter(name))

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    inherited_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not _is_new_parameter(name)
    )
    new_trainable = trainable - inherited_trainable
    if new_trainable <= 0:
        raise RuntimeError("the selected policy left no new VNCT parameters trainable")
    if policy == "new_modules_only" and inherited_trainable:
        raise RuntimeError("new_modules_only unexpectedly enabled inherited parameters")
    return TrainingPolicyReport(
        policy=policy,
        total=total,
        trainable=trainable,
        inherited_trainable=inherited_trainable,
        new_trainable=new_trainable,
    )


def apply_training_mode(model: nn.Module, policy: str) -> None:
    """Select module modes while preserving gradients through NC-M3 adapters."""

    _validate_policy(policy)
    model.train()
    if policy == "new_modules_only":
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise AttributeError("VNCT training policies require model.backbone")
        # VSSD-Small has drop_path_rate=0.4.  Eval mode makes the frozen feature
        # extractor deterministic; eval mode does not disable autograd for the
        # explicitly trainable NC-M3 parameters embedded in its mixer modules.
        backbone.eval()


def build_optimizer(
    model: nn.Module,
    *,
    policy: str,
    new_learning_rate: float,
    inherited_learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build AdamW groups with a lower LR for inherited VSSD parameters."""

    _validate_policy(policy)
    if new_learning_rate <= 0.0:
        raise ValueError("new_learning_rate must be positive")
    if policy == "full" and inherited_learning_rate <= 0.0:
        raise ValueError("full training requires a positive inherited_learning_rate")

    buckets: dict[tuple[str, str], list[nn.Parameter]] = {
        (origin, decay): []
        for origin in ("new", "inherited")
        for decay in ("decay", "no_decay")
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        origin = "new" if _is_new_parameter(name) else "inherited"
        decay = "no_decay" if parameter.ndim <= 1 or name.endswith(".bias") else "decay"
        buckets[(origin, decay)].append(parameter)

    groups = []
    for origin, learning_rate in (
        ("new", new_learning_rate),
        ("inherited", inherited_learning_rate),
    ):
        for decay in ("decay", "no_decay"):
            parameters = buckets[(origin, decay)]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": weight_decay if decay == "decay" else 0.0,
                        "name": f"{origin}_{decay}",
                    }
                )
    if not groups:
        raise RuntimeError("no trainable parameters were available for AdamW")
    return torch.optim.AdamW(groups)


def new_backbone_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return bridge parameters for diagnostics and gradient assertions."""

    parameters = []
    for module in model.modules():
        if isinstance(module, VSSDCheckpointNCM3):
            parameters.extend(
                parameter
                for parameter in (
                    module.trap_weight,
                    module.trap_bias,
                    module.decay_weight,
                    module.decay_bias,
                    module.mimo_source_weight,
                    module.mimo_query_weight,
                    module.mimo_in,
                    module.mimo_out,
                    module.ncm3_scale,
                )
                if parameter is not None
            )
    return tuple(parameters)


__all__ = [
    "POLICIES",
    "TrainingPolicyReport",
    "apply_training_mode",
    "build_optimizer",
    "configure_training_policy",
    "new_backbone_parameters",
]
