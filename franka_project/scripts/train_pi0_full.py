#!/usr/bin/env python
"""Full-parameter PI0 finetuning/continuation for single-rate Franka datasets.

Launch one GPU with ``python .../train_pi0_full.py --config CONFIG`` and two
GPUs with ``torchrun --standalone --nproc_per_node=2 ... --config CONFIG``.
``--max-steps`` is an invocation stop point (the scheduler always keeps the
YAML's planned horizon), which makes the S0 checkpoint/resume test meaningful.
For continuation stages it remains an absolute global step; the new scheduler
and epoch cadence use a separate zero-based continuation phase step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

# Keep the project entry runnable without requiring an editable install of the
# small project-local package.  Core LeRobot itself still comes from the active
# environment/repository as usual.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from franka_eef_pipeline.dual_rate_dataset import (
    ACTION_LABEL_MODE_ABSOLUTE_EEF,
    ACTION_LABEL_MODE_DELTA_EEF,
    CartesianAnchorDataset,
    resolve_action_label_spec,
)
from franka_eef_pipeline.pi0_training import (
    STRUCTURAL_ACTION_UNREACHABLE_NUMEL,
    STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS,
    STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT,
    assert_full_parameter_training,
    gradient_coverage_summary,
    load_pi0_full_checkpoint_weights,
    load_pi0_full_policy_and_processors,
    save_pi0_full_checkpoint,
)
from lerobot.optim import (
    load_optimizer_state,
    load_scheduler_state,
    save_optimizer_state,
    save_scheduler_state,
)
from lerobot.utils.random_utils import load_rng_state, save_rng_state, set_seed


REQUIRED_GRADIENT_GROUPS = (
    "vision_encoder",
    "vlm",
    "action_expert",
    "state_action_projections",
)
CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
)
DDP_FIND_UNUSED_PARAMETERS = True
STAGE_DATASET_CONTRACTS = {
    "S0-15": ("action15", 15, 15, 50),
    "S1-15": ("action15", 15, 15, 50),
    "F0-15": ("action15", 15, 15, 50),
    "F0-CONT-15": ("action15", 15, 15, 50),
    "S0-30": ("native30", 30, 30, 50),
    "S1-30": ("native30", 30, 30, 50),
    "F0-30": ("native30", 30, 30, 50),
    "F0-CONT-30": ("native30", 30, 30, 50),
}
CONTINUATION_SOURCE_STAGES = {
    "F0-CONT-15": "F0-15",
    "F0-CONT-30": "F0-30",
}
EXPECTED_STRUCTURAL_MISSING_GRADIENTS = {
    f"model.{core_name}": math.prod(shape)
    for core_name, shape in STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS
}


class TrainingContractError(RuntimeError):
    """Raised when a run could silently violate the frozen experiment contract."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise TrainingContractError(f"expected a JSON object: {path}")
    return value


def _read_json_value(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_keys(mapping: Mapping[str, Any], required: set[str], allowed: set[str], where: str) -> None:
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - allowed)
    if missing or unknown:
        raise TrainingContractError(f"{where} keys invalid: missing={missing}, unknown={unknown}")


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingContractError(f"{where} must be a positive integer, got {value!r}")
    return value


def load_and_validate_config(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    """Load a stage YAML and reject rate, PEFT, precision, or upload drift."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise TrainingContractError("training YAML must contain one mapping")

    regular_top = {
        "schema_version", "stage", "job_name", "dataset", "model", "training", "output", "wandb"
    }
    allowed_stages = set(STAGE_DATASET_CONTRACTS)
    if cfg.get("schema_version") != 1 or cfg.get("stage") not in allowed_stages:
        raise TrainingContractError("schema_version/stage is not a supported Franka stage")
    continuation_mode = cfg["stage"] in CONTINUATION_SOURCE_STAGES
    required_top = regular_top | ({"continuation"} if continuation_mode else set())
    allowed_top = regular_top | {"continuation"}
    _require_keys(cfg, required_top, allowed_top if continuation_mode else regular_top, "root")
    if not isinstance(cfg["job_name"], str) or not cfg["job_name"].strip():
        raise TrainingContractError("job_name must be non-empty")

    if continuation_mode:
        continuation = cfg["continuation"]
        continuation_keys = {
            "source_checkpoint",
            "expected_source_step",
            "expected_source_run_config_sha256",
            "expected_source_checkpoint_manifest_sha256",
            "preserve_optimizer_state",
            "preserve_rng_state",
            "preserve_global_step",
            "preserve_data_offset",
            "scheduler",
        }
        if not isinstance(continuation, Mapping):
            raise TrainingContractError("continuation must be a mapping")
        _require_keys(continuation, continuation_keys, continuation_keys, "continuation")
        if not isinstance(continuation["source_checkpoint"], str) or not continuation[
            "source_checkpoint"
        ].strip():
            raise TrainingContractError("continuation.source_checkpoint must be a non-empty path")
        _positive_int(continuation["expected_source_step"], "continuation.expected_source_step")
        for key in (
            "expected_source_run_config_sha256",
            "expected_source_checkpoint_manifest_sha256",
        ):
            digest = continuation[key]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise TrainingContractError(f"continuation.{key} must be a 64-character digest")
        for key in (
            "preserve_optimizer_state",
            "preserve_rng_state",
            "preserve_global_step",
            "preserve_data_offset",
        ):
            if continuation[key] is not True:
                raise TrainingContractError(f"continuation.{key} must be true")
        continuation_schedulers = {
            "cosine_no_warmup",
            "cosine_restart_no_warmup",
        }
        if continuation["scheduler"] not in continuation_schedulers:
            raise TrainingContractError(
                "continuation.scheduler must be one of "
                f"{sorted(continuation_schedulers)}"
            )

    dataset = cfg["dataset"]
    dataset_allowed = {
        "root", "profile", "observation_fps", "action_fps", "chunk_size",
        "episode_indices", "max_anchors_per_episode", "expected_valid_anchors",
        "expected_scope_content_sha256", "task_instruction",
    }
    _require_keys(
        dataset,
        {"root", "profile", "observation_fps", "action_fps", "chunk_size", "episode_indices",
         "max_anchors_per_episode", "expected_scope_content_sha256"},
        dataset_allowed,
        "dataset",
    )
    actual_dataset_contract = (
        dataset["profile"],
        dataset["observation_fps"],
        dataset["action_fps"],
        dataset["chunk_size"],
    )
    expected_dataset_contract = STAGE_DATASET_CONTRACTS[cfg["stage"]]
    if actual_dataset_contract != expected_dataset_contract:
        expected_profile, expected_obs, expected_action, expected_chunk = expected_dataset_contract
        raise TrainingContractError(
            f"{cfg['stage']} expects {expected_profile}/obs{expected_obs}/action{expected_action} "
            f"with K={expected_chunk}, got {actual_dataset_contract}"
        )
    if "task_instruction" in dataset and (
        not isinstance(dataset["task_instruction"], str)
        or not dataset["task_instruction"].strip()
    ):
        raise TrainingContractError("dataset.task_instruction must be a non-empty string")
    scope_hash = dataset["expected_scope_content_sha256"]
    if (
        not isinstance(scope_hash, str)
        or len(scope_hash) != 64
        or any(character not in "0123456789abcdef" for character in scope_hash)
    ):
        raise TrainingContractError("expected_scope_content_sha256 must be a 64-character digest")
    if dataset["episode_indices"] is not None:
        if not isinstance(dataset["episode_indices"], list) or not dataset["episode_indices"]:
            raise TrainingContractError("episode_indices must be null or a non-empty list")
        for index in dataset["episode_indices"]:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise TrainingContractError("episode_indices must contain non-negative integers")
    if dataset["max_anchors_per_episode"] is not None:
        _positive_int(dataset["max_anchors_per_episode"], "dataset.max_anchors_per_episode")
    if "expected_valid_anchors" in dataset:
        _positive_int(dataset["expected_valid_anchors"], "dataset.expected_valid_anchors")

    model = cfg["model"]
    model_keys = {
        "pretrained_path", "strict", "dtype", "gradient_checkpointing", "compile_model",
        "max_state_dim", "max_action_dim", "freeze_vision_encoder", "train_expert_only",
        "use_relative_actions", "peft",
    }
    _require_keys(model, model_keys, model_keys, "model")
    expected_model = {
        "strict": True,
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "compile_model": False,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "use_relative_actions": False,
        "peft": None,
    }
    drift = {key: {"expected": value, "actual": model.get(key)} for key, value in expected_model.items()
             if model.get(key) != value}
    if drift:
        raise TrainingContractError(f"model is not full-parameter PI0: {drift}")

    training = cfg["training"]
    training_allowed = {
        "seed", "steps", "epochs", "per_device_batch_size", "gradient_accumulation_steps",
        "num_workers", "shuffle", "peak_lr", "final_lr", "warmup_steps", "warmup_fraction",
        "warmup_max_steps", "weight_decay", "betas", "eps", "grad_clip_norm",
        "mixed_precision", "log_freq", "monitor_freq", "monitor_freq_epochs",
        "monitor_max_samples", "save_freq", "save_freq_epochs", "save_optimizer_state",
        "require_gradient_coverage", "gpu_count",
    }
    required_training = {
        "seed", "steps", "per_device_batch_size", "gradient_accumulation_steps", "num_workers",
        "shuffle", "peak_lr", "final_lr", "weight_decay", "betas", "eps", "grad_clip_norm",
        "mixed_precision", "log_freq", "monitor_max_samples",
        "save_optimizer_state", "require_gradient_coverage", "gpu_count",
    }
    _require_keys(training, required_training, training_allowed, "training")
    for key in ("seed", "num_workers"):
        if isinstance(training[key], bool) or not isinstance(training[key], int) or training[key] < 0:
            raise TrainingContractError(f"training.{key} must be a non-negative integer")
    for key in ("per_device_batch_size", "gradient_accumulation_steps", "log_freq",
                "monitor_max_samples", "gpu_count"):
        _positive_int(training[key], f"training.{key}")
    if training["steps"] == "auto":
        _positive_int(training.get("epochs"), "training.epochs")
    else:
        _positive_int(training["steps"], "training.steps")
        if "epochs" in training:
            raise TrainingContractError("fixed-step stages must not also specify epochs")
    if continuation_mode:
        if training["steps"] != "auto":
            raise TrainingContractError("continuation requires training.steps=auto and additional epochs")
        forbidden_warmup = sorted(
            {"warmup_steps", "warmup_fraction", "warmup_max_steps"}.intersection(training)
        )
        if forbidden_warmup:
            raise TrainingContractError(
                f"continuation cosine has no warmup; remove fields {forbidden_warmup}"
            )
    else:
        if ("warmup_steps" in training) == ("warmup_fraction" in training):
            raise TrainingContractError("specify exactly one of warmup_steps or warmup_fraction")
        if "warmup_steps" in training:
            _positive_int(training["warmup_steps"], "training.warmup_steps")
        else:
            fraction = training["warmup_fraction"]
            if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
                raise TrainingContractError("warmup_fraction must be in (0,1)")
            _positive_int(training.get("warmup_max_steps"), "training.warmup_max_steps")
    if ("monitor_freq" in training) == ("monitor_freq_epochs" in training):
        raise TrainingContractError("specify exactly one monitor frequency form")
    _positive_int(training.get("monitor_freq", training.get("monitor_freq_epochs")), "monitor frequency")
    save_frequency_keys = {"save_freq", "save_freq_epochs"}.intersection(training)
    if len(save_frequency_keys) != 1:
        raise TrainingContractError(
            "specify exactly one of training.save_freq or training.save_freq_epochs"
        )
    if "save_freq_epochs" in training:
        _positive_int(training["save_freq_epochs"], "training.save_freq_epochs")
    elif training["save_freq"] != "epoch":
        _positive_int(training["save_freq"], "training.save_freq")
    if not isinstance(training["betas"], list) or len(training["betas"]) != 2:
        raise TrainingContractError("training.betas must contain exactly two values")
    if not (0 <= training["final_lr"] <= training["peak_lr"]):
        raise TrainingContractError("require 0 <= final_lr <= peak_lr")
    if training["mixed_precision"] != "bf16":
        raise TrainingContractError("mixed_precision must be bf16")
    for key in ("save_optimizer_state", "require_gradient_coverage"):
        if training[key] is not True:
            raise TrainingContractError(f"training.{key} must be true")

    output = cfg["output"]
    output_keys = {"root", "immutable_run_directory", "save_to_hub"}
    _require_keys(output, output_keys, output_keys, "output")
    if output["immutable_run_directory"] is not True or output["save_to_hub"] is not False:
        raise TrainingContractError("output must be local, immutable, and never pushed to Hub")

    wandb_cfg = cfg["wandb"]
    wandb_keys = {"enable", "mode", "entity", "project", "disable_artifact", "save_code", "tags"}
    _require_keys(wandb_cfg, wandb_keys, wandb_keys, "wandb")
    if not (
        wandb_cfg["enable"] is True
        and wandb_cfg["mode"] == "online"
        and wandb_cfg["disable_artifact"] is True
        and wandb_cfg["save_code"] is False
        and isinstance(wandb_cfg["entity"], str)
        and bool(wandb_cfg["entity"].strip())
        and isinstance(wandb_cfg["project"], str)
        and bool(wandb_cfg["project"].strip())
    ):
        raise TrainingContractError("W&B must be online metrics-only with artifacts/code disabled")
    if not isinstance(wandb_cfg["tags"], list):
        raise TrainingContractError("wandb.tags must be a list")
    return cfg, config_path, _sha256_file(config_path)


def resolve_checkpoint_interval_steps(
    training: Mapping[str, Any], *, steps_per_epoch: int
) -> int:
    """Resolve the configured checkpoint cadence to optimizer steps.

    ``save_freq`` retains its legacy meaning (``"epoch"`` or optimizer
    steps). ``save_freq_epochs`` is explicit epoch-based cadence and therefore
    remains correct when batch size, world size, or gradient accumulation
    changes the number of optimizer steps in an epoch.
    """

    steps_per_epoch = _positive_int(steps_per_epoch, "steps_per_epoch")
    if "save_freq_epochs" in training:
        epochs = _positive_int(training["save_freq_epochs"], "training.save_freq_epochs")
        return epochs * steps_per_epoch
    save_freq = training.get("save_freq")
    if save_freq == "epoch":
        return steps_per_epoch
    return _positive_int(save_freq, "training.save_freq")


def should_save_checkpoint(
    *, phase_step: int, save_interval: int, global_step: int, stop_step: int
) -> bool:
    """Return whether this successful optimizer step must publish a checkpoint."""

    return phase_step % save_interval == 0 or global_step == stop_step


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    final_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm from ``peak/(warmup+1)`` then cosine-decay exactly to final LR."""

    if not 0 < warmup_steps < total_steps:
        raise TrainingContractError(f"warmup_steps must be in [1,total_steps), got {warmup_steps}/{total_steps}")
    floor = final_lr / peak_lr

    def scale(current_step: int) -> float:
        if current_step < warmup_steps:
            return (current_step + 1) / (warmup_steps + 1)
        progress = min(1.0, (current_step - warmup_steps) / (total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def compute_continuation_step_plan(*, source_step: int, additional_steps: int) -> dict[str, int]:
    """Return the frozen global/phase step axes for a continuation run."""

    source_step = _positive_int(source_step, "source_step")
    additional_steps = _positive_int(additional_steps, "additional_steps")
    return {
        "phase_start_global_step": source_step,
        "planned_phase_steps": additional_steps,
        "planned_final_global_step": source_step + additional_steps,
    }


def build_continuation_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    phase_steps: int,
    expected_current_lr: float,
    start_lr: float,
    final_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Start a no-warmup cosine phase while preserving loaded AdamW moments.

    AdamW checkpoints created under ``LambdaLR`` retain the old schedule's
    ``initial_lr`` in their param groups.  PyTorch will otherwise reuse that
    stale value as ``base_lrs``.  The continuation contract therefore verifies
    the optimizer LR that should be present before scheduler construction, then
    deliberately resets only ``lr`` and ``initial_lr`` to the explicit new
    phase start.  This supports both a continuous decay (start equals the source
    LR) and an audited LR restart without discarding AdamW moments.
    """

    phase_steps = _positive_int(phase_steps, "phase_steps")
    expected_current_lr = float(expected_current_lr)
    start_lr = float(start_lr)
    final_lr = float(final_lr)
    if not math.isfinite(expected_current_lr) or expected_current_lr <= 0:
        raise TrainingContractError(
            "expected_current_lr must be finite and positive, "
            f"got {expected_current_lr!r}"
        )
    if not math.isfinite(start_lr) or start_lr <= 0:
        raise TrainingContractError(f"start_lr must be finite and positive, got {start_lr!r}")
    if not math.isfinite(final_lr) or not 0 <= final_lr <= start_lr:
        raise TrainingContractError(
            f"continuation final_lr must be in [0,start_lr], got {final_lr!r}/{start_lr!r}"
        )
    if not optimizer.param_groups:
        raise TrainingContractError("continuation optimizer has no parameter groups")
    for index, group in enumerate(optimizer.param_groups):
        current_lr = float(group.get("lr", float("nan")))
        if not math.isclose(current_lr, expected_current_lr, rel_tol=1e-12, abs_tol=0.0):
            raise TrainingContractError(
                f"optimizer group {index} current lr {current_lr!r} "
                f"!= expected lr {expected_current_lr!r}"
            )
        group["lr"] = start_lr
        group["initial_lr"] = start_lr

    floor = final_lr / start_lr

    def scale(local_step: int) -> float:
        progress = min(1.0, max(0.0, local_step / phase_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    if any(
        not math.isclose(float(lr), start_lr, rel_tol=1e-12, abs_tol=0.0)
        for lr in scheduler.get_last_lr()
    ):
        raise TrainingContractError("continuation scheduler changed LR at local step zero")
    return scheduler


def convert_uint8_images_to_float01(batch: dict[str, Any], camera_keys: Sequence[str]) -> dict[str, Any]:
    """Apply the only accepted visual input conversion and enforce its range."""

    for key in camera_keys:
        if key not in batch:
            raise TrainingContractError(f"required real camera is missing from batch: {key}")
        image = batch[key]
        if not isinstance(image, torch.Tensor):
            raise TrainingContractError(f"camera {key} must be a tensor")
        if image.dtype == torch.uint8:
            image = image.to(torch.float32).div_(255.0)
        elif image.is_floating_point():
            image = image.to(torch.float32)
        else:
            raise TrainingContractError(f"camera {key} has unsupported dtype {image.dtype}")
        if not bool(torch.isfinite(image).all()) or image.numel() == 0:
            raise TrainingContractError(f"camera {key} is empty or contains NaN/Inf")
        minimum, maximum = float(image.min()), float(image.max())
        if minimum < 0.0 or maximum > 1.0:
            raise TrainingContractError(
                f"camera {key} must be in [0,1] before the PI0 processor, got [{minimum},{maximum}]"
            )
        batch[key] = image
    return batch


def select_monitor_subset(dataset: Any, max_samples: int, seed: int) -> dict[str, Any]:
    """Freeze an in-train monitor subset, preferring D3's frozen logical indices."""

    population = len(dataset)
    sample_size = min(_positive_int(max_samples, "monitor_max_samples"), population)
    if population <= 0:
        raise TrainingContractError("training dataset is empty")
    frozen = [int(index) for index in getattr(dataset, "train_monitor_indices", ())]
    if len(frozen) != len(set(frozen)) or any(index < 0 or index >= population for index in frozen):
        raise TrainingContractError("dataset train_monitor_indices are invalid")
    generator = torch.Generator().manual_seed(seed)
    if len(frozen) > sample_size:
        order = torch.randperm(len(frozen), generator=generator)[:sample_size].tolist()
        selected = [frozen[index] for index in order]
    elif frozen:
        # The converter's frozen in-train diagnostic is the canonical monitor.
        # A stage-filtered view may contain fewer than monitor_max_samples; do
        # not silently change that diagnostic by filling it with new anchors.
        selected = list(frozen)
    else:
        # Tiny S0 selections can have no intersection with the converter's
        # global frozen monitor.  Only in that case freeze a seeded fallback
        # from the actual stage-training population.
        order = torch.randperm(population, generator=generator)[:sample_size].tolist()
        selected = order
    selected = sorted(selected)
    logical = getattr(dataset, "logical_anchors", None)
    anchors = [logical[index] for index in selected] if logical is not None else selected
    payload = {
        "schema_version": 1,
        "source": "training_data",
        "held_out": False,
        "seed": seed,
        "population_size": population,
        "sample_size": len(selected),
        "indices": selected,
        "anchors": anchors,
        "includes_converter_frozen_indices": sorted(set(frozen).intersection(selected)),
    }
    payload["subset_sha256"] = _canonical_sha256(payload)
    return payload


def require_gradient_coverage(report: Mapping[str, Any]) -> None:
    """Fail closed around PI0's exact six-Parameter structural exception.

    Every retained Parameter must have a finite gradient except the six final
    PaliGemma prefix-output Parameters proven permanently outside the action
    loss.  The names, shapes and aggregate numel come from the canonical model
    graph contract; any new, missing, renamed, or resized exception fails.
    """

    groups = report.get("groups")
    if not isinstance(groups, Mapping):
        raise TrainingContractError("gradient report has no groups")
    failures: dict[str, Any] = {}
    overall = report.get("overall")
    if not isinstance(overall, Mapping):
        failures["overall"] = "missing overall gradient summary"
    else:
        overall_reasons = []
        missing_tensor_count = int(overall.get("trainable_tensors", -1)) - int(
            overall.get("gradient_tensors", -1)
        )
        missing_numel = int(overall.get("trainable_numel", -1)) - int(
            overall.get("gradient_numel", -1)
        )
        if missing_tensor_count != STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT:
            overall_reasons.append(
                "missing_tensor_count="
                f"{missing_tensor_count} (expected {STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT})"
            )
        if missing_numel != STRUCTURAL_ACTION_UNREACHABLE_NUMEL:
            overall_reasons.append(
                f"missing_numel={missing_numel} (expected {STRUCTURAL_ACTION_UNREACHABLE_NUMEL})"
            )
        if overall.get("all_gradients_finite") is not True:
            overall_reasons.append("non-finite gradient")
        if overall_reasons:
            failures["overall"] = overall_reasons
    missing_names: list[str] = []
    for name in REQUIRED_GRADIENT_GROUPS:
        group = groups.get(name)
        if not isinstance(group, Mapping):
            failures[name] = "missing group"
            continue
        reasons = []
        if int(group.get("trainable_numel", 0)) <= 0:
            reasons.append("no trainable parameters")
        if int(group.get("gradient_tensors", 0)) <= 0 or int(group.get("gradient_numel", 0)) <= 0:
            reasons.append("no gradients reached this subsystem")
        if group.get("all_gradients_finite") is not True:
            reasons.append("non-finite gradient")
        if int(group.get("nonzero_gradient_tensors", 0)) <= 0:
            reasons.append("all gradient tensors are zero")
        if reasons:
            failures[name] = reasons
    for group in groups.values():
        if not isinstance(group, Mapping):
            continue
        names = group.get("missing_gradient_names")
        if not isinstance(names, list) or any(not isinstance(value, str) for value in names):
            failures["missing_gradient_names"] = "every group must expose a string list"
            continue
        missing_names.extend(names)
    if len(missing_names) != len(set(missing_names)):
        failures["missing_gradient_names"] = "duplicate missing Parameter names"
    actual_missing = set(missing_names)
    expected_missing = set(EXPECTED_STRUCTURAL_MISSING_GRADIENTS)
    if actual_missing != expected_missing:
        failures["structural_exception"] = {
            "unexpected_missing": sorted(actual_missing - expected_missing),
            "expected_but_not_missing": sorted(expected_missing - actual_missing),
        }
    if failures:
        raise TrainingContractError(f"first-backward gradient coverage gate failed: {failures}")


def persist_and_require_gradient_coverage(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    is_main_process: bool,
) -> None:
    """Persist the first-backward evidence before applying the fail-closed gate.

    A failed gate is precisely when the per-parameter report is most useful.
    Writing it first has no effect on gradients, optimizer state, or the gate's
    decision; it only makes a zero-update failure diagnosable after the process
    exits.
    """

    if is_main_process:
        _write_json(report_path, report)
    require_gradient_coverage(report)


def _all_true(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(int(value), device=device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def _fork_rng_cuda_devices(device: torch.device) -> list[int]:
    """Resolve an index-less single-GPU ``torch.device('cuda')`` for fork_rng."""

    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _broadcast_string(value: str | None, accelerator: Any) -> str:
    values: list[Any] = [value]
    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(values, src=0)
    if not isinstance(values[0], str):
        raise TrainingContractError("rank zero failed to broadcast a path")
    return values[0]


class CyclingLoader:
    """Deterministic rank-sharded infinite loader with cheap update-boundary resume."""

    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        seed: int,
        rank: int,
        world_size: int,
        start_microbatch: int,
    ) -> None:
        self.sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=False
        )
        generator = torch.Generator().manual_seed(seed + rank)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            generator=generator,
        )
        if len(self.loader) == 0:
            raise TrainingContractError("rank-local DataLoader has no batches")
        self.batches_per_epoch = len(self.loader)
        self.epoch = start_microbatch // self.batches_per_epoch
        self.offset = start_microbatch % self.batches_per_epoch
        self._iterator: Any = None

    def _new_iterator(self) -> None:
        self.sampler.set_epoch(self.epoch)
        self._iterator = iter(self.loader)
        for _ in range(self.offset):
            next(self._iterator)
        self.offset = 0

    def __next__(self) -> dict[str, Any]:
        if self._iterator is None:
            self._new_iterator()
        try:
            return next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self._new_iterator()
            return next(self._iterator)


class MetricsOnlyWandb:
    """Narrow W&B facade: online scalar logging only; no artifact API is exposed."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        run_dir: Path,
        job_name: str,
        resolved_config: Mapping[str, Any],
        run_id: str | None,
        resume: bool,
    ) -> None:
        import wandb

        wandb_dir = run_dir / "wandb"
        wandb_dir.mkdir(exist_ok=True)
        self._run = wandb.init(
            id=run_id,
            resume="must" if resume else None,
            project=cfg["project"],
            entity=cfg["entity"],
            name=job_name,
            tags=list(cfg["tags"]),
            dir=str(wandb_dir),
            config=_jsonable(resolved_config),
            save_code=False,
            mode="online",
            job_type="train",
        )
        if self._run is None or bool(getattr(self._run, "offline", False)):
            raise TrainingContractError("W&B did not create an online run; refusing silent offline fallback")
        # Use a custom optimizer-step axis.  Train and monitor metrics can then
        # be emitted in separate calls at the same optimizer step without W&B
        # dropping the later call as a non-monotonic internal step.
        self._run.define_metric("trainer/step")
        self._run.define_metric("train/*", step_metric="trainer/step")
        self._run.define_metric("monitor/*", step_metric="trainer/step")
        self.run_id = str(self._run.id)
        self.url = str(self._run.get_url())

    def log(self, metrics: Mapping[str, float | int], step: int) -> None:
        self._run.log({"trainer/step": step, **dict(metrics)})

    def finish(self, exit_code: int = 0) -> None:
        self._run.finish(exit_code=exit_code)


def _geometry_manifest(profile: Mapping[str, Any]) -> dict[str, Any]:
    task_instruction = profile.get("task_instruction")
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise TrainingContractError("dataset profile has no non-empty task_instruction")
    action_label = resolve_action_label_spec(
        profile, source="train_pi0_full._geometry_manifest dataset profile"
    )
    if action_label["mode"] == ACTION_LABEL_MODE_DELTA_EEF:
        action_semantics = {
            "translation": "base-frame target_xyz - current_xyz",
            "rotation": "body rotvec Log(R_current.T @ R_target)",
            "gripper": "future measured target gripper_0_1",
        }
    elif action_label["mode"] == ACTION_LABEL_MODE_ABSOLUTE_EEF:
        action_semantics = {
            "translation": "absolute base-frame target_xyz",
            "rotation": "principal base-frame rotvec Log(R_target)",
            "gripper": "future measured absolute target gripper closed_0_1",
        }
    else:  # pragma: no cover - resolve_action_label_spec rejects this.
        raise AssertionError(f"unreachable action label mode: {action_label['mode']}")
    return {
        "schema_version": 1,
        "task_instruction": task_instruction,
        "state10": "current measured EEF xyz + rotation6d(first two columns) + gripper_0_1",
        "action7": {
            **action_semantics,
            "action_label_mode": action_label["mode"],
            "type": action_label["type"],
            "names": action_label["names"],
            "contract": action_label,
            "frequency_hz": int(profile["action_fps"]),
            "chunk_size": int(profile["chunk_size"]),
        },
        "dataset_profile": dict(profile),
    }


def atomic_save_checkpoint(
    *,
    run_dir: Path,
    step: int,
    accelerator: Any,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    preprocessor: Any,
    postprocessor: Any,
    geometry_manifest: Mapping[str, Any],
    stats_path: Path,
    load_report: Mapping[str, Any],
    config_sha256: str,
    monitor_sha256: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    wandb_run_id: str,
    phase_start_global_step: int = 0,
    source_identity_sha256: str | None = None,
) -> Path:
    """Save all ranks' RNG plus rank-zero model/state, then atomically publish once."""

    target = run_dir / "checkpoints" / f"step-{step:06d}"
    staging = target.with_name(f".{target.name}.staging")
    local_error: str | None = None
    if accelerator.is_main_process:
        if target.exists() or staging.exists():
            local_error = f"refusing to overwrite checkpoint/staging: {target} / {staging}"
        else:
            staging.mkdir(parents=True)
    accelerator.wait_for_everyone()

    try:
        if local_error is None:
            rank_state = staging / "training_state" / f"rank-{accelerator.process_index:02d}"
            rank_state.mkdir(parents=True, exist_ok=False)
            save_rng_state(rank_state)
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(policy)
                assert_full_parameter_training(unwrapped)
                save_pi0_full_checkpoint(
                    unwrapped,
                    preprocessor,
                    postprocessor,
                    staging / "pretrained_model",
                    geometry_manifest=geometry_manifest,
                    stats_manifest=stats_path,
                    load_report=load_report,
                )
                state_dir = staging / "training_state"
                save_optimizer_state(optimizer, state_dir)
                save_scheduler_state(scheduler, state_dir)
                continuation_state = (
                    {
                        "run_mode": "continuation",
                        "global_step": step,
                        "phase_step": step - phase_start_global_step,
                        "phase_start_global_step": phase_start_global_step,
                        "source_identity_sha256": source_identity_sha256,
                        "scheduler_step_axis": "phase_step",
                    }
                    if source_identity_sha256 is not None
                    else {}
                )
                _write_json(
                    state_dir / "training_step.json",
                    {
                        "step": step,
                        "world_size": accelerator.num_processes,
                        "per_device_batch_size": batch_size,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        **continuation_state,
                    },
                )
                _write_json(
                    staging / "checkpoint_manifest.json",
                    {
                        "schema_version": 1,
                        "step": step,
                        "config_sha256": config_sha256,
                        "monitor_subset_sha256": monitor_sha256,
                        "wandb_run_id": wandb_run_id,
                        "world_size": accelerator.num_processes,
                        "hub_upload": False,
                        "wandb_artifact_upload": False,
                        "atomic_publish": True,
                        **continuation_state,
                    },
                )
    except Exception as error:  # synchronize a readable failure instead of hanging peer ranks
        local_error = f"{type(error).__name__}: {error}"

    succeeded = _all_true(local_error is None, accelerator.device)
    if accelerator.is_main_process and succeeded:
        staging.rename(target)
        pointer_tmp = run_dir / "checkpoints" / ".last_checkpoint.json.tmp"
        pointer = {"step": step, "path": str(target.relative_to(run_dir))}
        if source_identity_sha256 is not None:
            pointer.update(
                {
                    "global_step": step,
                    "phase_step": step - phase_start_global_step,
                    "source_identity_sha256": source_identity_sha256,
                }
            )
        _write_json(pointer_tmp, pointer)
        os.replace(pointer_tmp, run_dir / "checkpoints" / "last_checkpoint.json")
    accelerator.wait_for_everyone()
    if not succeeded:
        raise TrainingContractError(f"atomic checkpoint failed on at least one rank; local={local_error}")
    return target


def _resolve_resume_checkpoint(run_dir: Path, expected: Mapping[str, Any]) -> tuple[Path, int]:
    run_manifest = _read_json(run_dir / "run_manifest.json")
    for key, value in expected.items():
        if run_manifest.get(key) != value:
            raise TrainingContractError(
                f"resume run manifest mismatch for {key}: expected={value!r}, actual={run_manifest.get(key)!r}"
            )
    pointer = _read_json(run_dir / "checkpoints" / "last_checkpoint.json")
    checkpoint = (run_dir / str(pointer.get("path", ""))).resolve()
    if not checkpoint.is_relative_to(run_dir.resolve()) or not checkpoint.is_dir():
        raise TrainingContractError("last checkpoint pointer is invalid")
    checkpoint_manifest = _read_json(checkpoint / "checkpoint_manifest.json")
    if (
        checkpoint_manifest.get("atomic_publish") is not True
        or checkpoint_manifest.get("hub_upload") is not False
        or checkpoint_manifest.get("wandb_artifact_upload") is not False
    ):
        raise TrainingContractError("resume checkpoint publication contract is invalid")
    matching_keys = ["config_sha256", "monitor_subset_sha256", "wandb_run_id", "world_size"]
    if "source_identity_sha256" in run_manifest:
        matching_keys.append("source_identity_sha256")
    for key in matching_keys:
        if checkpoint_manifest.get(key) != run_manifest.get(key):
            raise TrainingContractError(f"checkpoint/run manifest mismatch for {key}")
    step = int(checkpoint_manifest.get("step", -1))
    if step != int(pointer.get("step", -2)) or step <= 0:
        raise TrainingContractError("last checkpoint step metadata is inconsistent")
    if run_manifest.get("run_mode") == "continuation":
        phase_start = int(run_manifest.get("phase_start_global_step", -1))
        expected_phase_step = step - phase_start
        continuation_expected = {
            "global_step": step,
            "phase_step": expected_phase_step,
            "phase_start_global_step": phase_start,
            "source_identity_sha256": run_manifest.get("source_identity_sha256"),
            "scheduler_step_axis": "phase_step",
        }
        for key, value in continuation_expected.items():
            if checkpoint_manifest.get(key) != value:
                raise TrainingContractError(
                    f"continuation checkpoint manifest mismatch for {key}"
                )
        for key in ("global_step", "phase_step", "source_identity_sha256"):
            if pointer.get(key) != continuation_expected[key]:
                raise TrainingContractError(f"continuation checkpoint pointer mismatch for {key}")
    return checkpoint, step


def _load_resume_training_state(
    checkpoint: Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    world_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    expected_step: int | None = None,
    phase_start_global_step: int | None = None,
    source_identity_sha256: str | None = None,
) -> dict[str, Any]:
    state_dir = checkpoint / "training_state"
    state = _read_json(state_dir / "training_step.json")
    expected = {
        "world_size": world_size,
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise TrainingContractError(f"resume training-state mismatch for {key}")
    if expected_step is not None and state.get("step") != expected_step:
        raise TrainingContractError(
            f"resume training-state step mismatch: {state.get('step')!r} != {expected_step}"
        )
    if phase_start_global_step is not None:
        global_step = int(state.get("step", -1))
        continuation_expected = {
            "run_mode": "continuation",
            "global_step": global_step,
            "phase_step": global_step - phase_start_global_step,
            "phase_start_global_step": phase_start_global_step,
            "source_identity_sha256": source_identity_sha256,
            "scheduler_step_axis": "phase_step",
        }
        for key, value in continuation_expected.items():
            if state.get(key) != value:
                raise TrainingContractError(
                    f"continuation resume training-state mismatch for {key}"
                )
    weights_report = load_pi0_full_checkpoint_weights(policy, checkpoint / "pretrained_model")
    load_optimizer_state(optimizer, state_dir)
    load_scheduler_state(scheduler, state_dir)
    return weights_report


def _monitor_loss(
    policy: torch.nn.Module,
    preprocessor: Any,
    dataset: Any,
    indices: Sequence[int],
    *,
    batch_size: int,
    seed: int,
    accelerator: Any,
) -> float:
    subset = torch.utils.data.Subset(dataset, list(indices))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    unwrapped = accelerator.unwrap_model(policy)
    was_training = unwrapped.training
    unwrapped.eval()
    weighted_loss = 0.0
    count = 0
    cuda_devices = _fork_rng_cuda_devices(accelerator.device)
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        torch.manual_seed(seed)
        if accelerator.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        for batch in loader:
            convert_uint8_images_to_float01(batch, CAMERA_KEYS)
            processed = preprocessor(batch)
            with accelerator.autocast():
                loss, _ = unwrapped(processed)
            current_batch = int(processed["action"].shape[0])
            weighted_loss += float(loss.detach()) * current_batch
            count += current_batch
    unwrapped.train(was_training)
    if count == 0:
        raise TrainingContractError("monitor subset produced no samples")
    return weighted_loss / count


def _runtime_dataset_contract(cfg: Mapping[str, Any], dataset: CartesianAnchorDataset) -> dict[str, Any]:
    root = Path(cfg["dataset"]["root"]).expanduser().resolve()
    profile_path = root / "meta" / "franka_eef_profile.json"
    profile = _read_json(profile_path)
    expected = cfg["dataset"]
    checks = {
        "profile": expected["profile"],
        "observation_fps": expected["observation_fps"],
        "action_fps": expected["action_fps"],
        "chunk_size": expected["chunk_size"],
        "scope_content_sha256": expected["expected_scope_content_sha256"],
    }
    if "task_instruction" in expected:
        checks["task_instruction"] = expected["task_instruction"]
    mismatches = {key: {"expected": value, "actual": profile.get(key)} for key, value in checks.items()
                  if profile.get(key) != value}
    if "action_label_mode" in expected:
        try:
            actual_action_label = resolve_action_label_spec(
                profile, source=profile_path
            )
        except ValueError as error:
            raise TrainingContractError(str(error)) from error
        expected_action_label_mode = str(expected["action_label_mode"])
        if actual_action_label["mode"] != expected_action_label_mode:
            mismatches["action_label_mode"] = {
                "expected": expected_action_label_mode,
                "actual": actual_action_label["mode"],
            }
        if (
            expected_action_label_mode == ACTION_LABEL_MODE_ABSOLUTE_EEF
            and profile.get("action_label_mode_source")
            != "config.contract.action_label_mode"
        ):
            mismatches["action_label_mode_source"] = {
                "expected": "config.contract.action_label_mode",
                "actual": profile.get("action_label_mode_source"),
            }
        adapter_action_label_mode = getattr(dataset, "action_label_mode", None)
        if adapter_action_label_mode != expected_action_label_mode:
            mismatches["adapter_action_label_mode"] = {
                "expected": expected_action_label_mode,
                "actual": adapter_action_label_mode,
            }
    if mismatches:
        raise TrainingContractError(f"dataset profile mismatch: {mismatches}")
    task_instruction = profile.get("task_instruction")
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise TrainingContractError("dataset profile task_instruction must be a non-empty string")
    if dataset.task_instruction != task_instruction:
        raise TrainingContractError(
            "Cartesian adapter task instruction differs from dataset profile metadata"
        )
    if "expected_valid_anchors" in expected and len(dataset) != expected["expected_valid_anchors"]:
        raise TrainingContractError(
            f"valid anchor count mismatch: expected {expected['expected_valid_anchors']}, got {len(dataset)}"
        )
    if dataset.effective_stats is None or set(dataset.effective_stats) != {"observation.state", "action"}:
        raise TrainingContractError("dataset must expose only effective state10/action7 PI0 stats")
    return profile


def _require_nonempty_file(path: Path, where: str) -> int:
    if not path.is_file():
        raise TrainingContractError(f"{where} is missing: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise TrainingContractError(f"{where} is empty: {path}")
    return size


def _require_same_float(actual: Any, expected: Any, where: str) -> float:
    try:
        actual_float = float(actual)
        expected_float = float(expected)
    except (TypeError, ValueError) as error:
        raise TrainingContractError(f"{where} is not numeric: {actual!r}/{expected!r}") from error
    if not (
        math.isfinite(actual_float)
        and math.isfinite(expected_float)
        and math.isclose(actual_float, expected_float, rel_tol=1e-12, abs_tol=0.0)
    ):
        raise TrainingContractError(f"{where} mismatch: {actual_float!r} != {expected_float!r}")
    return actual_float


def resolve_continuation_source(
    cfg: Mapping[str, Any],
    *,
    dataset_size: int,
    dataset_profile_sha256: str,
    monitor_subset_sha256: str,
    steps_per_epoch: int,
    world_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    """Audit a completed source run before a new continuation run is created.

    This intentionally reads only immutable source artifacts.  Large model and
    optimizer tensors are shape/state-audited when they are loaded; the source
    identity records the existing model ledger and the optimizer file size so
    same-continuation-run resume can reject lineage drift cheaply.
    """

    continuation_stage = cfg.get("stage")
    if continuation_stage not in CONTINUATION_SOURCE_STAGES or not isinstance(
        cfg.get("continuation"), Mapping
    ):
        raise TrainingContractError("resolve_continuation_source requires an F0-CONT stage")
    continuation = cfg["continuation"]
    expected_step = _positive_int(
        continuation["expected_source_step"], "continuation.expected_source_step"
    )
    checkpoint = Path(continuation["source_checkpoint"]).expanduser().resolve()
    if not checkpoint.is_dir() or checkpoint.parent.name != "checkpoints":
        raise TrainingContractError(
            "continuation source must be a complete step-* checkpoint directory"
        )
    if checkpoint.name != f"step-{expected_step:06d}":
        raise TrainingContractError(
            f"source checkpoint name {checkpoint.name!r} does not encode expected step {expected_step}"
        )
    source_run = checkpoint.parent.parent.resolve()
    checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
    checkpoint_manifest_sha256 = _sha256_file(checkpoint_manifest_path)
    if checkpoint_manifest_sha256 != continuation["expected_source_checkpoint_manifest_sha256"]:
        raise TrainingContractError("source checkpoint manifest digest differs from frozen config")
    checkpoint_manifest = _read_json(checkpoint_manifest_path)
    run_manifest = _read_json(source_run / "run_manifest.json")
    pointer = _read_json(source_run / "checkpoints/last_checkpoint.json")
    pointed_checkpoint = (source_run / str(pointer.get("path", ""))).resolve()
    if pointed_checkpoint != checkpoint or pointer.get("step") != expected_step:
        raise TrainingContractError("continuation source is not the source run's last checkpoint")

    resolved_source_config = source_run / "resolved_config.yaml"
    source_cfg, _, source_config_sha256 = load_and_validate_config(resolved_source_config)
    if source_config_sha256 != continuation["expected_source_run_config_sha256"]:
        raise TrainingContractError("source resolved config digest differs from frozen config")
    expected_source_stage = CONTINUATION_SOURCE_STAGES[str(continuation_stage)]
    if source_cfg["stage"] != expected_source_stage:
        raise TrainingContractError(
            f"continuation source stage must be {expected_source_stage}, "
            f"got {source_cfg['stage']!r}"
        )

    required_run_values = {
        "config_sha256": source_config_sha256,
        "monitor_subset_sha256": monitor_subset_sha256,
        "world_size": world_size,
        "planned_total_steps": expected_step,
        "dataset_profile_sha256": dataset_profile_sha256,
        "dataset_root": str(Path(cfg["dataset"]["root"]).expanduser().resolve()),
        "dataset_size": dataset_size,
        "steps_per_epoch": steps_per_epoch,
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for key, expected in required_run_values.items():
        if run_manifest.get(key) != expected:
            raise TrainingContractError(
                f"source run manifest mismatch for {key}: {run_manifest.get(key)!r} != {expected!r}"
            )
    for key in ("config_sha256", "monitor_subset_sha256", "wandb_run_id", "world_size"):
        if checkpoint_manifest.get(key) != run_manifest.get(key):
            raise TrainingContractError(f"source checkpoint/run manifest mismatch for {key}")
    if (
        checkpoint_manifest.get("step") != expected_step
        or checkpoint_manifest.get("atomic_publish") is not True
        or checkpoint_manifest.get("hub_upload") is not False
        or checkpoint_manifest.get("wandb_artifact_upload") is not False
    ):
        raise TrainingContractError("source checkpoint publication contract is invalid")

    source_monitor = _read_json(source_run / "monitor_subset.json")
    stored_monitor_sha = source_monitor.pop("subset_sha256", None)
    if (
        stored_monitor_sha != monitor_subset_sha256
        or _canonical_sha256(source_monitor) != monitor_subset_sha256
    ):
        raise TrainingContractError("source monitor subset is not identical to this continuation")

    current_dataset = dict(cfg["dataset"])
    source_dataset = dict(source_cfg["dataset"])
    current_dataset["root"] = str(Path(current_dataset["root"]).expanduser().resolve())
    source_dataset["root"] = str(Path(source_dataset["root"]).expanduser().resolve())
    if source_dataset != current_dataset:
        raise TrainingContractError("continuation dataset config differs from the source F0 dataset")
    if source_cfg["model"] != cfg["model"]:
        raise TrainingContractError("continuation model construction config differs from source F0")
    continuity_training_keys = (
        "seed",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "shuffle",
        "weight_decay",
        "betas",
        "eps",
        "grad_clip_norm",
        "mixed_precision",
        "gpu_count",
    )
    training_drift = {
        key: {"source": source_cfg["training"].get(key), "continuation": cfg["training"].get(key)}
        for key in continuity_training_keys
        if source_cfg["training"].get(key) != cfg["training"].get(key)
    }
    if training_drift:
        raise TrainingContractError(f"continuation training-state contract drift: {training_drift}")

    state_dir = checkpoint / "training_state"
    training_state = _read_json(state_dir / "training_step.json")
    expected_training_state = {
        "step": expected_step,
        "world_size": world_size,
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for key, expected in expected_training_state.items():
        if training_state.get(key) != expected:
            raise TrainingContractError(
                f"source training state mismatch for {key}: {training_state.get(key)!r} != {expected!r}"
            )

    optimizer_state_path = state_dir / "optimizer_state.safetensors"
    optimizer_size = _require_nonempty_file(
        optimizer_state_path, "source AdamW tensor state"
    )
    optimizer_groups_path = state_dir / "optimizer_param_groups.json"
    _require_nonempty_file(optimizer_groups_path, "source AdamW parameter groups")
    optimizer_groups = _read_json_value(optimizer_groups_path)
    if not isinstance(optimizer_groups, list) or len(optimizer_groups) != 1:
        raise TrainingContractError("source AdamW must have exactly one parameter group")
    source_group = optimizer_groups[0]
    if not isinstance(source_group, Mapping):
        raise TrainingContractError("source AdamW parameter group is malformed")
    parameter_ids = source_group.get("params")
    if not isinstance(parameter_ids, list) or parameter_ids != list(range(len(parameter_ids))):
        raise TrainingContractError("source AdamW parameter IDs must be contiguous and ordered")
    if len(parameter_ids) != 776:
        raise TrainingContractError(f"source AdamW expected 776 parameters, got {len(parameter_ids)}")
    for key in ("betas", "eps", "weight_decay"):
        if key == "betas":
            actual = tuple(float(value) for value in source_group.get(key, ()))
            expected = tuple(float(value) for value in cfg["training"][key])
            if actual != expected:
                raise TrainingContractError(f"source AdamW {key} mismatch: {actual!r} != {expected!r}")
        else:
            _require_same_float(source_group.get(key), cfg["training"][key], f"source AdamW {key}")
    expected_adamw_flags = {
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }
    adamw_flag_drift = {
        key: {"expected": expected, "actual": source_group.get(key)}
        for key, expected in expected_adamw_flags.items()
        if source_group.get(key) != expected
    }
    if adamw_flag_drift:
        raise TrainingContractError(f"source AdamW behavior flag drift: {adamw_flag_drift}")

    scheduler_path = state_dir / "scheduler_state.json"
    scheduler_state = _read_json(scheduler_path)
    if scheduler_state.get("last_epoch") != expected_step:
        raise TrainingContractError("source scheduler last_epoch does not equal source global step")
    last_lrs = scheduler_state.get("_last_lr")
    if not isinstance(last_lrs, list) or len(last_lrs) != 1:
        raise TrainingContractError("source scheduler must expose exactly one current LR")
    source_lr = _require_same_float(last_lrs[0], source_group.get("lr"), "source scheduler/AdamW LR")
    source_initial_lr = _require_same_float(
        source_group.get("initial_lr"),
        source_cfg["training"]["peak_lr"],
        "source AdamW initial_lr/source peak LR",
    )
    if continuation["scheduler"] == "cosine_no_warmup":
        _require_same_float(
            source_lr,
            cfg["training"]["peak_lr"],
            "continuation start/source LR",
        )

    rng_hashes: dict[str, str] = {}
    expected_rank_dirs = {f"rank-{rank:02d}" for rank in range(world_size)}
    actual_rank_dirs = {path.name for path in state_dir.glob("rank-*") if path.is_dir()}
    if actual_rank_dirs != expected_rank_dirs:
        raise TrainingContractError(
            f"source rank RNG directories mismatch: {actual_rank_dirs!r} != {expected_rank_dirs!r}"
        )
    for rank_dir in sorted(expected_rank_dirs):
        rng_path = state_dir / rank_dir / "rng_state.safetensors"
        _require_nonempty_file(rng_path, f"source {rank_dir} RNG state")
        rng_hashes[rank_dir] = _sha256_file(rng_path)
    if len(set(rng_hashes.values())) != len(rng_hashes):
        raise TrainingContractError("source rank RNG states are unexpectedly identical")

    model_dir = checkpoint / "pretrained_model"
    model_state_path = model_dir / "model.safetensors"
    model_size = _require_nonempty_file(model_state_path, "source PI0 model weights")
    model_manifest_path = model_dir / "franka_pi0_checkpoint_manifest.json"
    model_manifest = _read_json(model_manifest_path)
    parameter_training = model_manifest.get("parameter_training")
    model_files = model_manifest.get("files")
    if (
        model_manifest.get("checkpoint_type") != "franka_pi0_full_parameter_eef"
        or model_manifest.get("hub_upload") is not False
        or not isinstance(parameter_training, Mapping)
        or parameter_training.get("lora_parameter_count") != 0
        or parameter_training.get("trainable_fraction") != 1.0
        or not isinstance(model_files, Mapping)
        or not isinstance(model_files.get("model.safetensors"), Mapping)
        or model_files["model.safetensors"].get("size_bytes") != model_size
    ):
        raise TrainingContractError("source model is not the audited full-parameter PI0 checkpoint")

    source_identity = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint),
        "run_dir": str(source_run),
        "global_step": expected_step,
        "source_stage": source_cfg["stage"],
        "run_config_sha256": source_config_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "model_manifest_sha256": _sha256_file(model_manifest_path),
        "model_sha256": model_files["model.safetensors"].get("sha256"),
        "model_size_bytes": model_size,
        "optimizer_param_groups_sha256": _sha256_file(optimizer_groups_path),
        "optimizer_size_bytes": optimizer_size,
        "scheduler_state_sha256": _sha256_file(scheduler_path),
        "rank_rng_sha256": rng_hashes,
        "source_wandb_run_id": run_manifest["wandb_run_id"],
        "source_lr": source_lr,
    }
    source_identity_sha256 = _canonical_sha256(source_identity)
    return {
        "checkpoint": checkpoint,
        "state_dir": state_dir,
        "source_step": expected_step,
        "source_lr": source_lr,
        "source_initial_lr": source_initial_lr,
        "source_identity": source_identity,
        "source_identity_sha256": source_identity_sha256,
    }


def load_continuation_source_state(
    source: Mapping[str, Any],
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Load source weights/AdamW and prove the moments remain on the source step."""

    checkpoint = Path(source["checkpoint"])
    source_step = int(source["source_step"])
    weights_report = load_pi0_full_checkpoint_weights(policy, checkpoint / "pretrained_model")
    # LeRobot's optimizer loader deliberately requires the current and saved
    # param-group dictionaries to have identical keys.  A fresh AdamW has no
    # ``initial_lr`` yet, while every source optimizer saved after LambdaLR
    # construction does.  Seed that audited scheduler-owned key before strict
    # deserialization; the checkpoint then overwrites its value as usual.
    if len(optimizer.param_groups) != 1:
        raise TrainingContractError("continuation AdamW must have exactly one parameter group")
    optimizer.param_groups[0]["initial_lr"] = float(source["source_initial_lr"])
    load_optimizer_state(optimizer, Path(source["state_dir"]))
    optimizer_report = validate_continuation_optimizer_state(
        policy,
        optimizer,
        expected_step=source_step,
        expected_lr=float(source["source_lr"]),
        expected_initial_lr=float(source["source_initial_lr"]),
    )
    optimizer_report["moments_preserved"] = True
    return {"weights": weights_report, "optimizer": optimizer_report}


def validate_continuation_optimizer_state(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_step: int,
    expected_lr: float,
    expected_initial_lr: float | None = None,
) -> dict[str, Any]:
    """Validate loaded AdamW moment topology without scanning its 14 GB values."""

    if len(optimizer.param_groups) != 1:
        raise TrainingContractError("loaded continuation AdamW must have exactly one parameter group")
    group = optimizer.param_groups[0]
    _require_same_float(group.get("lr"), expected_lr, "loaded continuation AdamW LR")
    if expected_initial_lr is not None:
        _require_same_float(
            group.get("initial_lr"),
            expected_initial_lr,
            "loaded continuation AdamW initial_lr",
        )
    parameters = list(group["params"])
    states = optimizer.state
    expected_active = len(parameters) - STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT
    if len(parameters) != 776 or len(states) != expected_active:
        raise TrainingContractError(
            "loaded continuation AdamW state count mismatch: "
            f"parameters={len(parameters)}, active={len(states)}, expected_active={expected_active}"
        )
    for parameter, state in states.items():
        if not isinstance(state, Mapping) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise TrainingContractError("loaded continuation AdamW state is missing moments/step")
        state_step = float(torch.as_tensor(state["step"]).detach().cpu())
        _require_same_float(
            state_step,
            expected_step,
            "loaded continuation AdamW state step",
        )
        if tuple(state["exp_avg"].shape) != tuple(parameter.shape) or tuple(
            state["exp_avg_sq"].shape
        ) != tuple(parameter.shape):
            raise TrainingContractError("loaded continuation AdamW moment shape mismatch")
    active_ids = {id(parameter) for parameter in states}
    missing_names = {
        name for name, parameter in policy.named_parameters() if id(parameter) not in active_ids
    }
    if missing_names != set(EXPECTED_STRUCTURAL_MISSING_GRADIENTS):
        raise TrainingContractError(
            "loaded continuation AdamW stateless parameters differ from the exact structural six: "
            f"{sorted(missing_names)}"
        )
    return {
        "parameter_count": len(parameters),
        "active_state_count": len(states),
        "stateless_parameter_names": sorted(missing_names),
        "optimizer_step": expected_step,
        "lr": float(group["lr"]),
        "initial_lr": float(group["initial_lr"]) if "initial_lr" in group else None,
    }


def validate_continuation_scheduler_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    expected_phase_step: int,
    start_lr: float,
) -> float:
    """Require the continuation scheduler to use the local phase-step axis."""

    if scheduler.last_epoch != expected_phase_step:
        raise TrainingContractError(
            "continuation scheduler phase mismatch: "
            f"last_epoch={scheduler.last_epoch}, expected={expected_phase_step}"
        )
    last_lrs = scheduler.get_last_lr()
    if len(last_lrs) != len(optimizer.param_groups) or not last_lrs:
        raise TrainingContractError("continuation scheduler/optimizer group count mismatch")
    for index, (scheduler_lr, group) in enumerate(zip(last_lrs, optimizer.param_groups, strict=True)):
        current_lr = _require_same_float(
            group.get("lr"), scheduler_lr, f"continuation scheduler/optimizer LR group {index}"
        )
        if current_lr > start_lr and not math.isclose(
            current_lr, start_lr, rel_tol=1e-12, abs_tol=0.0
        ):
            raise TrainingContractError("continuation scheduler LR exceeds configured start LR")
    return float(last_lrs[0])


def _setup_run_dir(output_root: Path, job_name: str, accelerator: Any, resume: Path | None) -> Path:
    value: str | None = None
    if accelerator.is_main_process:
        if resume is not None:
            candidate = resume.expanduser().resolve()
            if not candidate.is_dir():
                raise FileNotFoundError(candidate)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            candidate = output_root / f"{timestamp}_{job_name}_{uuid.uuid4().hex[:8]}"
            candidate.mkdir(parents=True, exist_ok=False)
        value = str(candidate)
    return Path(_broadcast_string(value, accelerator))


def _configure_logging(run_dir: Path, accelerator: Any) -> Path:
    invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_r{accelerator.process_index}"
    log_path = run_dir / "logs" / f"{invocation}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s rank={accelerator.process_index} %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_path


def train(config_path: str | Path, *, max_steps: int | None = None, resume_run_dir: str | Path | None = None) -> Path:
    """Execute one validated base or continuation invocation."""

    cfg, resolved_config_path, config_sha = load_and_validate_config(config_path)
    continuation_mode = cfg["stage"] in CONTINUATION_SOURCE_STAGES
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        step_scheduler_with_optimizer=False,
        # Match LeRobot's official Accelerate training loop: six retained final
        # prefix-output Parameters are structurally outside PI0's action loss,
        # and the first-backward gate accepts only their exact audited ledger.
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS)
        ],
    )
    if accelerator.device.type != "cuda":
        raise TrainingContractError("PI0 full finetuning requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise TrainingContractError("selected CUDA device does not support BF16")
    if accelerator.num_processes != cfg["training"]["gpu_count"]:
        raise TrainingContractError(
            f"launch world size {accelerator.num_processes} != config gpu_count {cfg['training']['gpu_count']}"
        )
    set_seed(cfg["training"]["seed"] + accelerator.process_index)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = CartesianAnchorDataset(
        cfg["dataset"]["root"],
        profile=cfg["dataset"]["profile"],
        episode_indices=cfg["dataset"]["episode_indices"],
        max_anchors_per_episode=cfg["dataset"]["max_anchors_per_episode"],
        video_backend="pyav",
    )
    profile = _runtime_dataset_contract(cfg, dataset)
    batch_size = cfg["training"]["per_device_batch_size"]
    accumulation = cfg["training"]["gradient_accumulation_steps"]
    rank_samples = math.ceil(len(dataset) / accelerator.num_processes)
    rank_batches = math.ceil(rank_samples / batch_size)
    steps_per_epoch = math.ceil(rank_batches / accumulation)
    if cfg["training"]["steps"] == "auto":
        phase_steps = steps_per_epoch * cfg["training"]["epochs"]
    else:
        phase_steps = cfg["training"]["steps"]
    dataset_profile_sha256 = _sha256_file(
        Path(cfg["dataset"]["root"]) / "meta/franka_eef_profile.json"
    )
    monitor_interval = cfg["training"].get("monitor_freq", steps_per_epoch * cfg["training"].get("monitor_freq_epochs", 1))
    save_interval = resolve_checkpoint_interval_steps(
        cfg["training"], steps_per_epoch=steps_per_epoch
    )

    monitor = select_monitor_subset(dataset, cfg["training"]["monitor_max_samples"], cfg["training"]["seed"] + 17)
    continuation_source: dict[str, Any] | None = None
    if continuation_mode:
        continuation_source = resolve_continuation_source(
            cfg,
            dataset_size=len(dataset),
            dataset_profile_sha256=dataset_profile_sha256,
            monitor_subset_sha256=monitor["subset_sha256"],
            steps_per_epoch=steps_per_epoch,
            world_size=accelerator.num_processes,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
        )
        step_plan = compute_continuation_step_plan(
            source_step=continuation_source["source_step"], additional_steps=phase_steps
        )
    else:
        step_plan = {
            "phase_start_global_step": 0,
            "planned_phase_steps": phase_steps,
            "planned_final_global_step": phase_steps,
        }
    phase_start_step = step_plan["phase_start_global_step"]
    planned_total_steps = step_plan["planned_final_global_step"]
    stop_step = planned_total_steps if max_steps is None else _positive_int(max_steps, "--max-steps")
    if stop_step > planned_total_steps:
        raise TrainingContractError(
            f"--max-steps {stop_step} exceeds planned global step {planned_total_steps}"
        )
    if stop_step <= phase_start_step:
        raise TrainingContractError(
            f"stop step {stop_step} must exceed phase start global step {phase_start_step}"
        )
    warmup = 0
    if not continuation_mode:
        warmup = cfg["training"].get("warmup_steps")
        if warmup is None:
            warmup = min(
                cfg["training"]["warmup_max_steps"],
                max(1, int(phase_steps * cfg["training"]["warmup_fraction"])),
            )

    resume_path = Path(resume_run_dir) if resume_run_dir is not None else None
    run_dir = _setup_run_dir(Path(cfg["output"]["root"]).expanduser().resolve(), cfg["job_name"], accelerator, resume_path)
    log_path = _configure_logging(run_dir, accelerator)
    if accelerator.is_main_process and resume_path is None:
        shutil.copy2(resolved_config_path, run_dir / "resolved_config.yaml")
        _write_json(run_dir / "monitor_subset.json", monitor)
    accelerator.wait_for_everyone()

    run_expected = {
        "config_sha256": config_sha,
        "monitor_subset_sha256": monitor["subset_sha256"],
        "world_size": accelerator.num_processes,
        "planned_total_steps": planned_total_steps,
        "dataset_profile_sha256": dataset_profile_sha256,
    }
    if continuation_source is not None:
        run_expected.update(
            {
                "run_mode": "continuation",
                **step_plan,
                "source_identity_sha256": continuation_source["source_identity_sha256"],
            }
        )
    checkpoint: Path | None = None
    start_step = phase_start_step
    existing_manifest: dict[str, Any] | None = None
    if resume_path is not None:
        checkpoint, start_step = _resolve_resume_checkpoint(run_dir, run_expected)
        existing_manifest = _read_json(run_dir / "run_manifest.json")
        if stop_step <= start_step:
            raise TrainingContractError(f"stop step {stop_step} must be greater than resumed step {start_step}")

    policy, preprocessor, postprocessor, base_load_report = load_pi0_full_policy_and_processors(
        cfg["model"]["pretrained_path"], dataset.effective_stats, accelerator.device
    )
    assert_full_parameter_training(policy)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(cfg["training"]["peak_lr"]),
        betas=tuple(float(value) for value in cfg["training"]["betas"]),
        eps=float(cfg["training"]["eps"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    continuation_load_report: dict[str, Any] | None = None
    if continuation_source is not None and checkpoint is None:
        continuation_load_report = load_continuation_source_state(
            continuation_source, policy=policy, optimizer=optimizer
        )
    if continuation_source is not None:
        continuation_start_lr = float(cfg["training"]["peak_lr"])
        expected_optimizer_lr = (
            float(continuation_source["source_lr"])
            if checkpoint is None
            else continuation_start_lr
        )
        scheduler = build_continuation_cosine_scheduler(
            optimizer,
            phase_steps=phase_steps,
            expected_current_lr=expected_optimizer_lr,
            start_lr=continuation_start_lr,
            final_lr=float(cfg["training"]["final_lr"]),
        )
    else:
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=phase_steps,
            warmup_steps=warmup,
            peak_lr=float(cfg["training"]["peak_lr"]),
            final_lr=float(cfg["training"]["final_lr"]),
        )
    resume_weights_report: dict[str, Any] | None = None
    if checkpoint is not None:
        resume_weights_report = _load_resume_training_state(
            checkpoint,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            world_size=accelerator.num_processes,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            expected_step=start_step,
            phase_start_global_step=phase_start_step if continuation_mode else None,
            source_identity_sha256=(
                continuation_source["source_identity_sha256"]
                if continuation_source is not None
                else None
            ),
        )
    continuation_runtime_report: dict[str, Any] | None = None
    if continuation_source is not None:
        continuation_lr = validate_continuation_scheduler_state(
            optimizer,
            scheduler,
            expected_phase_step=start_step - phase_start_step,
            start_lr=float(cfg["training"]["peak_lr"]),
        )
        if checkpoint is not None:
            optimizer_resume_report = validate_continuation_optimizer_state(
                policy,
                optimizer,
                expected_step=start_step,
                expected_lr=continuation_lr,
                expected_initial_lr=float(cfg["training"]["peak_lr"]),
            )
        else:
            assert continuation_load_report is not None
            loaded_source_optimizer = continuation_load_report["optimizer"]
            optimizer_resume_report = {
                **loaded_source_optimizer,
                "loaded_source_lr": loaded_source_optimizer["lr"],
                "lr": continuation_lr,
            }
        continuation_runtime_report = {
            "global_step": start_step,
            "phase_step": start_step - phase_start_step,
            "lr": continuation_lr,
            "source_lr": float(continuation_source["source_lr"]),
            "scheduler_start_lr": float(cfg["training"]["peak_lr"]),
            "optimizer": optimizer_resume_report,
            "source_scheduler_loaded": False,
        }
    policy, optimizer = accelerator.prepare(policy, optimizer)
    assert_full_parameter_training(accelerator.unwrap_model(policy))

    wandb_logger: MetricsOnlyWandb | None = None
    if accelerator.is_main_process:
        wandb_logger = MetricsOnlyWandb(
            cfg["wandb"],
            run_dir=run_dir,
            job_name=cfg["job_name"],
            resolved_config=cfg,
            run_id=existing_manifest.get("wandb_run_id") if existing_manifest else None,
            resume=resume_path is not None,
        )
        if existing_manifest is None:
            run_manifest = {
                "schema_version": 2 if continuation_mode else 1,
                **run_expected,
                "stage": cfg["stage"],
                "job_name": cfg["job_name"],
                "dataset_root": str(Path(cfg["dataset"]["root"]).resolve()),
                "dataset_size": len(dataset),
                "steps_per_epoch": steps_per_epoch,
                "checkpoint_interval_steps": save_interval,
                "checkpoint_interval_epochs": cfg["training"].get("save_freq_epochs"),
                "warmup_steps": warmup,
                "per_device_batch_size": batch_size,
                "gradient_accumulation_steps": accumulation,
                "wandb_run_id": wandb_logger.run_id,
                "wandb_url": wandb_logger.url,
                "wandb_mode": "online",
                "wandb_artifact_upload": False,
                "hub_upload": False,
                "full_parameter_report": base_load_report["parameters"],
            }
            if continuation_source is not None:
                run_manifest.update(
                    {
                        "continuation_source": continuation_source["source_identity"],
                        "scheduler_contract": {
                            "type": cfg["continuation"]["scheduler"],
                            "step_axis": "phase_step",
                            "source_lr": continuation_source["source_lr"],
                            "start_lr": float(cfg["training"]["peak_lr"]),
                            "final_lr": float(cfg["training"]["final_lr"]),
                            "total_phase_steps": phase_steps,
                            "source_scheduler_loaded": False,
                        },
                        "data_offset_contract": {
                            "step_axis": "global_step",
                            "source_microbatch_offset": phase_start_step * accumulation,
                            "gradient_accumulation_steps": accumulation,
                        },
                        "state_preservation": {
                            "optimizer_moments": True,
                            "rank_rng": True,
                            "global_step": True,
                            "dataloader_offset": True,
                        },
                    }
                )
            _write_json(run_dir / "run_manifest.json", run_manifest)
        logging.info(
            "run_dir=%s start_global_step=%d start_phase_step=%d stop_global_step=%d "
            "planned_final_global_step=%d",
            run_dir,
            start_step,
            start_step - phase_start_step,
            stop_step,
            planned_total_steps,
        )
    accelerator.wait_for_everyone()
    wandb_run_id = _read_json(run_dir / "run_manifest.json")["wandb_run_id"]

    loader = CyclingLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg["training"]["num_workers"],
        shuffle=cfg["training"]["shuffle"],
        seed=cfg["training"]["seed"],
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        start_microbatch=start_step * accumulation,
    )
    # Select the rank-local RNG source now, but restore it only after the
    # optional baseline monitor.  That makes the restored state the final
    # stochastic setup action before the first training forward.
    rng_checkpoint = checkpoint
    if rng_checkpoint is None and continuation_source is not None:
        rng_checkpoint = Path(continuation_source["checkpoint"])
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    gradient_gate_done = start_step > phase_start_step
    load_report = {
        "base": base_load_report,
        "continuation_source": continuation_load_report,
        "continuation_runtime": continuation_runtime_report,
        "resume": resume_weights_report,
    }
    geometry = _geometry_manifest(profile)
    stats_path = Path(cfg["dataset"]["root"]) / "meta/pi0_eef_stats.json"
    invocation_started = datetime.now(timezone.utc).isoformat()
    exit_code = 1
    try:
        if resume_path is None and accelerator.is_main_process:
            # Rank-local monitor evaluation must bypass the DDP wrapper.  A
            # DDP forward may synchronize buffers even in eval mode, while all
            # non-main ranks are intentionally waiting at the barrier below.
            initial_monitor = _monitor_loss(
                accelerator.unwrap_model(policy),
                preprocessor,
                dataset,
                monitor["indices"],
                batch_size=batch_size,
                seed=cfg["training"]["seed"] + 29, accelerator=accelerator,
            )
            assert wandb_logger is not None
            wandb_logger.log(
                {
                    "monitor/loss": initial_monitor,
                    "monitor/sample_count": len(monitor["indices"]),
                },
                step,
            )
            logging.info(
                "global_step=%d phase_step=%d monitor_loss=%.8f",
                step,
                step - phase_start_step,
                initial_monitor,
            )
        accelerator.wait_for_everyone()
        if rng_checkpoint is not None:
            load_rng_state(
                rng_checkpoint
                / "training_state"
                / f"rank-{accelerator.process_index:02d}"
            )

        accumulated_loss = torch.zeros((), device=accelerator.device)
        accumulated_microbatches = 0
        update_started = time.perf_counter()
        while step < stop_step:
            batch = next(loader)
            with accelerator.accumulate(policy):
                convert_uint8_images_to_float01(batch, CAMERA_KEYS)
                processed = preprocessor(batch)
                with accelerator.autocast():
                    loss, _ = policy(processed)
                finite = bool(torch.isfinite(loss.detach()))
                if not _all_true(finite, accelerator.device):
                    raise FloatingPointError(f"non-finite loss before backward at update {step + 1}")
                accumulated_loss += loss.detach()
                accumulated_microbatches += 1
                accelerator.backward(loss)
                if not accelerator.sync_gradients:
                    continue
                if not gradient_gate_done:
                    gradient_report = gradient_coverage_summary(accelerator.unwrap_model(policy), inspect_values=True)
                    persist_and_require_gradient_coverage(
                        gradient_report,
                        report_path=run_dir / "first_backward_gradient_coverage.json",
                        is_main_process=accelerator.is_main_process,
                    )
                    gradient_gate_done = True
                grad_norm = accelerator.clip_grad_norm_(
                    policy.parameters(), float(cfg["training"]["grad_clip_norm"])
                )
                if not _all_true(bool(torch.isfinite(torch.as_tensor(grad_norm))), accelerator.device):
                    raise FloatingPointError(f"non-finite gradient norm at update {step + 1}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                phase_step = step - phase_start_step
                mean_local_loss = accumulated_loss / accumulated_microbatches
                mean_loss = accelerator.reduce(mean_local_loss, reduction="mean")
                accumulated_loss.zero_()
                accumulated_microbatches = 0
                duration = time.perf_counter() - update_started

                if phase_step % cfg["training"]["log_freq"] == 0 or step == stop_step:
                    if accelerator.is_main_process:
                        assert wandb_logger is not None
                        metrics = {
                            "train/loss": float(mean_loss),
                            "train/lr": float(scheduler.get_last_lr()[0]),
                            "train/grad_norm": float(torch.as_tensor(grad_norm)),
                            "train/update_seconds": duration,
                            "train/global_samples": step * batch_size * accelerator.num_processes * accumulation,
                            "train/phase_step": phase_step,
                            "train/phase_samples": phase_step
                            * batch_size
                            * accelerator.num_processes
                            * accumulation,
                        }
                        wandb_logger.log(metrics, step)
                        logging.info(
                            "global_step=%d phase_step=%d loss=%.8f lr=%.3e grad_norm=%.5f",
                            step,
                            phase_step,
                            metrics["train/loss"],
                            metrics["train/lr"],
                            metrics["train/grad_norm"],
                        )

                if phase_step % monitor_interval == 0 or step == stop_step:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        monitor_loss = _monitor_loss(
                            accelerator.unwrap_model(policy),
                            preprocessor,
                            dataset,
                            monitor["indices"],
                            batch_size=batch_size,
                            seed=cfg["training"]["seed"] + 29, accelerator=accelerator,
                        )
                        assert wandb_logger is not None
                        wandb_logger.log({"monitor/loss": monitor_loss, "monitor/sample_count": len(monitor["indices"])}, step)
                        logging.info(
                            "global_step=%d phase_step=%d monitor_loss=%.8f",
                            step,
                            phase_step,
                            monitor_loss,
                        )
                    accelerator.wait_for_everyone()

                if should_save_checkpoint(
                    phase_step=phase_step,
                    save_interval=save_interval,
                    global_step=step,
                    stop_step=stop_step,
                ):
                    checkpoint = atomic_save_checkpoint(
                        run_dir=run_dir,
                        step=step,
                        accelerator=accelerator,
                        policy=policy,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        geometry_manifest=geometry,
                        stats_path=stats_path,
                        load_report=load_report,
                        config_sha256=config_sha,
                        monitor_sha256=monitor["subset_sha256"],
                        batch_size=batch_size,
                        gradient_accumulation_steps=accumulation,
                        wandb_run_id=wandb_run_id,
                        phase_start_global_step=phase_start_step,
                        source_identity_sha256=(
                            continuation_source["source_identity_sha256"]
                            if continuation_source is not None
                            else None
                        ),
                    )
                    if accelerator.is_main_process:
                        logging.info("published local checkpoint %s (no Hub/W&B artifact)", checkpoint)
                # Exclude monitor/checkpoint wall time from the next optimizer
                # update's performance metric.
                update_started = time.perf_counter()

        exit_code = 0
        return run_dir
    finally:
        if accelerator.is_main_process:
            _write_json(
                run_dir / "invocations" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.json",
                {
                    "started_at": invocation_started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "start_step": start_step,
                    "start_global_step": start_step,
                    "start_phase_step": start_step - phase_start_step,
                    "requested_stop_step": stop_step,
                    "last_step": step,
                    "last_global_step": step,
                    "last_phase_step": step - phase_start_step,
                    "run_mode": "continuation" if continuation_mode else "base",
                    "exit_code": exit_code,
                    "log_path": str(log_path),
                    "resume": resume_path is not None,
                },
            )
            if wandb_logger is not None:
                wandb_logger.finish(exit_code=exit_code)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Stop after this global optimizer step without changing the YAML scheduler horizon.",
    )
    parser.add_argument(
        "--resume-run-dir", type=Path, default=None,
        help="Explicit immutable run directory whose last local checkpoint and W&B id must be resumed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = train(args.config, max_steps=args.max_steps, resume_run_dir=args.resume_run_dir)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
