#!/usr/bin/env python
"""Stage-free configurable PI0 finetuning for Franka Cartesian datasets.

This entry point is intentionally independent from ``train_pi0_full.py``'s
Codex planning stages.  Dataset geometry is still checked at runtime, but an
arbitrary/absent ``stage`` field has no effect on training.

Trainability is selected through ``model.trainability``.  Built-in presets are
``full``, ``action_expert``, and ``action_expert_paligemma``; ``custom`` accepts
an explicit list drawn from the six audited PI0 components.

One GPU::

    python franka_project/scripts/train_pi0.py --config CONFIG

Two GPUs::

    torchrun --standalone --nproc_per_node=2 \
      franka_project/scripts/train_pi0.py --config CONFIG
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_model as save_safetensors_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

# Reuse the stage-independent, already-audited loop utilities while keeping a
# separate config parser, gradient gate, save path, and training function.
import train_pi0_full as shared  # noqa: E402
from franka_eef_pipeline import pi0_training as pi0_io  # noqa: E402
from franka_eef_pipeline.dual_rate_dataset import (  # noqa: E402
    CartesianAnchorDataset,
    action_label_spec,
)
from franka_eef_pipeline.pi0_trainability import (  # noqa: E402
    PI0TrainabilityError,
    apply_pi0_trainability,
    assert_pi0_trainability,
    enforce_frozen_pi0_eval_mode,
    pi0_gradient_coverage_summary,
    require_pi0_gradient_coverage,
    resolve_pi0_trainability,
    trainable_pi0_parameters,
)
from lerobot.optim import (  # noqa: E402
    load_optimizer_state,
    load_scheduler_state,
    save_optimizer_state,
    save_scheduler_state,
)
from lerobot.utils.random_utils import load_rng_state, save_rng_state, set_seed  # noqa: E402


TrainingContractError = shared.TrainingContractError
CAMERA_KEYS = shared.CAMERA_KEYS


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingContractError(f"{where} must be a mapping")
    return value


def load_and_validate_config(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    """Load a stage-free training YAML and validate only runtime contracts."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise TrainingContractError("training YAML must contain one mapping")

    required_top = {
        "schema_version",
        "job_name",
        "dataset",
        "model",
        "training",
        "output",
        "wandb",
    }
    # ``stage`` is accepted as inert user metadata for easy migration from the
    # planning harness.  It is deliberately not checked or used anywhere.
    shared._require_keys(cfg, required_top, required_top | {"stage"}, "root")
    if cfg.get("schema_version") != 2:
        raise TrainingContractError("stage-free train_pi0 configs require schema_version: 2")
    if not isinstance(cfg["job_name"], str) or not cfg["job_name"].strip():
        raise TrainingContractError("job_name must be non-empty")

    dataset = _require_mapping(cfg["dataset"], "dataset")
    dataset_allowed = {
        "root",
        "profile",
        "observation_fps",
        "action_fps",
        "chunk_size",
        "action_label_mode",
        "episode_indices",
        "max_anchors_per_episode",
        "expected_valid_anchors",
        "expected_scope_content_sha256",
        "task_instruction",
    }
    dataset_required = {
        "root",
        "profile",
        "observation_fps",
        "action_fps",
        "chunk_size",
        "episode_indices",
        "max_anchors_per_episode",
        "expected_scope_content_sha256",
        "task_instruction",
    }
    shared._require_keys(dataset, dataset_required, dataset_allowed, "dataset")
    for key in ("root", "profile", "task_instruction"):
        if not isinstance(dataset[key], str) or not dataset[key].strip():
            raise TrainingContractError(f"dataset.{key} must be a non-empty string")
    for key in ("observation_fps", "action_fps", "chunk_size"):
        shared._positive_int(dataset[key], f"dataset.{key}")
    dataset.setdefault("action_label_mode", "delta_eef")
    try:
        action_label_spec(dataset["action_label_mode"])
    except (TypeError, ValueError) as exc:
        raise TrainingContractError(
            f"dataset.action_label_mode is invalid: {dataset['action_label_mode']!r}"
        ) from exc
    scope_hash = dataset["expected_scope_content_sha256"]
    if (
        not isinstance(scope_hash, str)
        or len(scope_hash) != 64
        or any(character not in "0123456789abcdef" for character in scope_hash)
    ):
        raise TrainingContractError(
            "dataset.expected_scope_content_sha256 must be a lowercase 64-character digest"
        )
    if dataset["episode_indices"] is not None:
        indices = dataset["episode_indices"]
        if not isinstance(indices, list) or not indices:
            raise TrainingContractError("dataset.episode_indices must be null or non-empty")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
            raise TrainingContractError(
                "dataset.episode_indices must contain non-negative integers"
            )
    if dataset["max_anchors_per_episode"] is not None:
        shared._positive_int(
            dataset["max_anchors_per_episode"], "dataset.max_anchors_per_episode"
        )
    if "expected_valid_anchors" in dataset:
        shared._positive_int(
            dataset["expected_valid_anchors"], "dataset.expected_valid_anchors"
        )

    model = _require_mapping(cfg["model"], "model")
    model_keys = {
        "pretrained_path",
        "strict",
        "dtype",
        "gradient_checkpointing",
        "compile_model",
        "max_state_dim",
        "max_action_dim",
        "use_relative_actions",
        "peft",
        "trainability",
    }
    shared._require_keys(model, model_keys, model_keys, "model")
    if not isinstance(model["pretrained_path"], str) or not model["pretrained_path"].strip():
        raise TrainingContractError("model.pretrained_path must be a non-empty local path")
    expected_model = {
        "strict": True,
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "compile_model": False,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "use_relative_actions": False,
        "peft": None,
    }
    drift = {
        key: {"expected": expected, "actual": model.get(key)}
        for key, expected in expected_model.items()
        if model.get(key) != expected
    }
    if drift:
        raise TrainingContractError(f"unsupported PI0 model construction drift: {drift}")
    try:
        resolve_pi0_trainability(model["trainability"])
    except PI0TrainabilityError as exc:
        raise TrainingContractError(str(exc)) from exc

    training = _require_mapping(cfg["training"], "training")
    training_allowed = {
        "seed",
        "steps",
        "epochs",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "shuffle",
        "peak_lr",
        "final_lr",
        "warmup_steps",
        "warmup_fraction",
        "warmup_max_steps",
        "weight_decay",
        "betas",
        "eps",
        "grad_clip_norm",
        "mixed_precision",
        "log_freq",
        "monitor_freq",
        "monitor_freq_epochs",
        "monitor_max_samples",
        "save_freq",
        "save_freq_epochs",
        "save_optimizer_state",
        "require_gradient_coverage",
        "gpu_count",
    }
    training_required = {
        "seed",
        "steps",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "shuffle",
        "peak_lr",
        "final_lr",
        "weight_decay",
        "betas",
        "eps",
        "grad_clip_norm",
        "mixed_precision",
        "log_freq",
        "monitor_max_samples",
        "save_optimizer_state",
        "require_gradient_coverage",
        "gpu_count",
    }
    shared._require_keys(training, training_required, training_allowed, "training")
    for key in ("seed", "num_workers"):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TrainingContractError(f"training.{key} must be a non-negative integer")
    for key in (
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "log_freq",
        "monitor_max_samples",
        "gpu_count",
    ):
        shared._positive_int(training[key], f"training.{key}")
    if training["steps"] == "auto":
        shared._positive_int(training.get("epochs"), "training.epochs")
    else:
        shared._positive_int(training["steps"], "training.steps")
        if "epochs" in training:
            raise TrainingContractError("fixed training.steps must not also specify epochs")
    if ("warmup_steps" in training) == ("warmup_fraction" in training):
        raise TrainingContractError(
            "specify exactly one of training.warmup_steps or training.warmup_fraction"
        )
    if "warmup_steps" in training:
        shared._positive_int(training["warmup_steps"], "training.warmup_steps")
    else:
        fraction = training["warmup_fraction"]
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < fraction < 1:
            raise TrainingContractError("training.warmup_fraction must be in (0,1)")
        shared._positive_int(training.get("warmup_max_steps"), "training.warmup_max_steps")
    if ("monitor_freq" in training) == ("monitor_freq_epochs" in training):
        raise TrainingContractError("specify exactly one monitor frequency form")
    shared._positive_int(
        training.get("monitor_freq", training.get("monitor_freq_epochs")),
        "monitor frequency",
    )
    save_keys = {"save_freq", "save_freq_epochs"}.intersection(training)
    if len(save_keys) != 1:
        raise TrainingContractError(
            "specify exactly one of training.save_freq or training.save_freq_epochs"
        )
    if "save_freq_epochs" in training:
        shared._positive_int(training["save_freq_epochs"], "training.save_freq_epochs")
    elif training["save_freq"] != "epoch":
        shared._positive_int(training["save_freq"], "training.save_freq")
    if not isinstance(training["betas"], list) or len(training["betas"]) != 2:
        raise TrainingContractError("training.betas must contain exactly two values")
    try:
        peak_lr = float(training["peak_lr"])
        final_lr = float(training["final_lr"])
    except (TypeError, ValueError) as exc:
        raise TrainingContractError("training peak/final LR must be numeric") from exc
    if not math.isfinite(peak_lr) or peak_lr <= 0 or not 0 <= final_lr <= peak_lr:
        raise TrainingContractError("require finite peak_lr > 0 and 0 <= final_lr <= peak_lr")
    if training["mixed_precision"] != "bf16":
        raise TrainingContractError("training.mixed_precision must be bf16")
    for key in ("save_optimizer_state", "require_gradient_coverage"):
        if training[key] is not True:
            raise TrainingContractError(f"training.{key} must be true")

    output = _require_mapping(cfg["output"], "output")
    output_keys = {"root", "immutable_run_directory", "save_to_hub"}
    shared._require_keys(output, output_keys, output_keys, "output")
    if not isinstance(output["root"], str) or not output["root"].strip():
        raise TrainingContractError("output.root must be a non-empty local path")
    if output["immutable_run_directory"] is not True or output["save_to_hub"] is not False:
        raise TrainingContractError("output must be local, immutable, and never pushed to Hub")

    wandb_cfg = _require_mapping(cfg["wandb"], "wandb")
    wandb_keys = {
        "enable",
        "mode",
        "entity",
        "project",
        "disable_artifact",
        "save_code",
        "tags",
    }
    shared._require_keys(wandb_cfg, wandb_keys, wandb_keys, "wandb")
    if not (
        wandb_cfg["enable"] is True
        and wandb_cfg["mode"] == "online"
        and wandb_cfg["disable_artifact"] is True
        and wandb_cfg["save_code"] is False
        and isinstance(wandb_cfg["entity"], str)
        and bool(wandb_cfg["entity"].strip())
        and isinstance(wandb_cfg["project"], str)
        and bool(wandb_cfg["project"].strip())
        and isinstance(wandb_cfg["tags"], list)
    ):
        raise TrainingContractError(
            "W&B must be online metrics-only with artifacts/code disabled"
        )

    return cfg, config_path, shared._sha256_file(config_path)


def _write_or_copy_manifest(
    value: Mapping[str, Any] | str | Path, destination: Path, stem: str
) -> Path:
    if isinstance(value, (str, Path)):
        source = Path(value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / f"{stem}{source.suffix or '.json'}"
        shutil.copy2(source, target)
        return target
    if not isinstance(value, Mapping):
        raise TypeError(f"{stem} must be a mapping or file path")
    target = destination / f"{stem}.json"
    shared._write_json(target, value)
    return target


def save_pi0_checkpoint(
    policy: torch.nn.Module,
    preprocessor: Any,
    postprocessor: Any,
    save_directory: str | Path,
    *,
    trainability_spec: Mapping[str, Any],
    expected_trainability_signature: str,
    training_graph: Mapping[str, Any],
    geometry_manifest: Mapping[str, Any] | str | Path,
    stats_manifest: Mapping[str, Any] | str | Path,
    load_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save complete weights plus an exact partial/full parameter ledger."""

    parameter_report = assert_pi0_trainability(
        policy,
        trainability_spec,
        expected_signature=expected_trainability_signature,
    )
    destination = Path(save_directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint directory: {destination}")
    destination.mkdir(parents=True)

    policy.config.save_pretrained(destination, push_to_hub=False)
    # save_model, unlike save_file(state_dict), correctly serializes the tied
    # PaliGemma embedding/LM-head Parameter once.
    save_safetensors_model(policy.model, destination / "model.safetensors")
    preprocessor.save_pretrained(
        destination, push_to_hub=False, config_filename="policy_preprocessor.json"
    )
    postprocessor.save_pretrained(
        destination, push_to_hub=False, config_filename="policy_postprocessor.json"
    )
    geometry_path = _write_or_copy_manifest(
        geometry_manifest, destination, "franka_eef_geometry_manifest"
    )
    stats_path = _write_or_copy_manifest(stats_manifest, destination, "pi0_eef_stats")
    required_paths = [
        destination / "config.json",
        destination / "model.safetensors",
        destination / "policy_preprocessor.json",
        destination / "policy_postprocessor.json",
        geometry_path,
        stats_path,
    ]
    missing = [path.name for path in required_paths if not path.is_file()]
    if missing:
        raise TrainingContractError(f"local PI0 checkpoint save is incomplete: {missing}")

    manifest = {
        "schema_version": 2,
        "checkpoint_type": "franka_pi0_configurable_parameter_eef",
        "weights_namespace": pi0_io.PI0_CORE_WEIGHTS_NAMESPACE,
        "hub_upload": False,
        "wandb_artifact_upload": False,
        "parameter_training": parameter_report,
        # Keep the canonical pre-freeze graph report so the existing strict
        # loader can verify tied/pruned topology before trainability is reapplied.
        "training_graph": dict(training_graph),
        "source_load_report": shared._jsonable(load_report) if load_report is not None else None,
        "files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": shared._sha256_file(path),
            }
            for path in required_paths
        },
    }
    shared._write_json(destination / "franka_pi0_checkpoint_manifest.json", manifest)
    return manifest


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
    trainability_spec: Mapping[str, Any],
    parameter_report: Mapping[str, Any],
    training_graph: Mapping[str, Any],
    geometry_manifest: Mapping[str, Any],
    stats_path: Path,
    load_report: Mapping[str, Any],
    config_sha256: str,
    monitor_sha256: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    wandb_run_id: str,
    dataset_stats_sha256: str,
    global_samples: int,
) -> Path:
    """Atomically publish full weights and selected-parameter optimizer state."""

    target = run_dir / "checkpoints" / f"step-{step:06d}"
    staging = target.with_name(f".{target.name}.staging")
    local_error: str | None = None
    if accelerator.is_main_process:
        try:
            if target.exists() or staging.exists():
                raise FileExistsError(
                    f"refusing to overwrite checkpoint/staging: {target} / {staging}"
                )
            staging.mkdir(parents=True)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
    preflight_succeeded = shared._all_true(local_error is None, accelerator.device)
    if not preflight_succeeded:
        raise TrainingContractError(
            f"checkpoint staging preflight failed on at least one rank; local={local_error}"
        )
    accelerator.wait_for_everyone()

    try:
        if local_error is None:
            rank_state = staging / "training_state" / f"rank-{accelerator.process_index:02d}"
            rank_state.mkdir(parents=True, exist_ok=False)
            save_rng_state(rank_state)
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(policy)
                assert_pi0_trainability(
                    unwrapped,
                    trainability_spec,
                    expected_signature=parameter_report["trainability_signature_sha256"],
                )
                save_pi0_checkpoint(
                    unwrapped,
                    preprocessor,
                    postprocessor,
                    staging / "pretrained_model",
                    trainability_spec=trainability_spec,
                    expected_trainability_signature=parameter_report[
                        "trainability_signature_sha256"
                    ],
                    training_graph=training_graph,
                    geometry_manifest=geometry_manifest,
                    stats_manifest=stats_path,
                    load_report=load_report,
                )
                state_dir = staging / "training_state"
                save_optimizer_state(optimizer, state_dir)
                save_scheduler_state(scheduler, state_dir)
                shared._write_json(
                    state_dir / "training_step.json",
                    {
                        "step": step,
                        "world_size": accelerator.num_processes,
                        "per_device_batch_size": batch_size,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "trainability_signature_sha256": parameter_report[
                            "trainability_signature_sha256"
                        ],
                        "dataset_stats_sha256": dataset_stats_sha256,
                        "global_samples": global_samples,
                    },
                )
                shared._write_json(
                    staging / "checkpoint_manifest.json",
                    {
                        "schema_version": 2,
                        "step": step,
                        "config_sha256": config_sha256,
                        "monitor_subset_sha256": monitor_sha256,
                        "wandb_run_id": wandb_run_id,
                        "world_size": accelerator.num_processes,
                        "trainability_signature_sha256": parameter_report[
                            "trainability_signature_sha256"
                        ],
                        "dataset_stats_sha256": dataset_stats_sha256,
                        "global_samples": global_samples,
                        "hub_upload": False,
                        "wandb_artifact_upload": False,
                        "atomic_publish": True,
                    },
                )
    except Exception as error:  # synchronize a readable failure across ranks
        local_error = f"{type(error).__name__}: {error}"

    writes_succeeded = shared._all_true(local_error is None, accelerator.device)
    if not writes_succeeded:
        raise TrainingContractError(
            f"atomic checkpoint failed on at least one rank; local={local_error}"
        )

    publish_error: str | None = None
    if accelerator.is_main_process:
        try:
            staging.rename(target)
            pointer_tmp = run_dir / "checkpoints" / ".last_checkpoint.json.tmp"
            shared._write_json(
                pointer_tmp, {"step": step, "path": str(target.relative_to(run_dir))}
            )
            os.replace(pointer_tmp, run_dir / "checkpoints" / "last_checkpoint.json")
        except Exception as error:
            publish_error = f"{type(error).__name__}: {error}"
    publish_succeeded = shared._all_true(publish_error is None, accelerator.device)
    accelerator.wait_for_everyone()
    if not publish_succeeded:
        raise TrainingContractError(
            f"checkpoint publish failed on at least one rank; local={publish_error}"
        )
    return target


def _load_resume_optimizer_scheduler(
    checkpoint: Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_step: int,
    world_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    trainability_signature_sha256: str,
    dataset_stats_sha256: str,
) -> int:
    state_dir = checkpoint / "training_state"
    state = shared._read_json(state_dir / "training_step.json")
    expected = {
        "step": expected_step,
        "world_size": world_size,
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "trainability_signature_sha256": trainability_signature_sha256,
        "dataset_stats_sha256": dataset_stats_sha256,
    }
    drift = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if drift:
        raise TrainingContractError(f"resume training-state mismatch: {drift}")
    global_samples = state.get("global_samples")
    if (
        isinstance(global_samples, bool)
        or not isinstance(global_samples, int)
        or global_samples <= 0
    ):
        raise TrainingContractError("resume training-state global_samples must be positive")
    checkpoint_manifest = shared._read_json(checkpoint / "checkpoint_manifest.json")
    if (
        checkpoint_manifest.get("trainability_signature_sha256")
        != trainability_signature_sha256
    ):
        raise TrainingContractError("resume checkpoint trainability signature mismatch")
    if checkpoint_manifest.get("dataset_stats_sha256") != dataset_stats_sha256:
        raise TrainingContractError("resume checkpoint dataset stats digest mismatch")
    if checkpoint_manifest.get("global_samples") != global_samples:
        raise TrainingContractError("resume checkpoint global sample count mismatch")
    model_manifest = shared._read_json(
        checkpoint / "pretrained_model/franka_pi0_checkpoint_manifest.json"
    )
    model_files = model_manifest.get("files")
    if (
        not isinstance(model_files, Mapping)
        or not isinstance(model_files.get("pi0_eef_stats.json"), Mapping)
        or model_files["pi0_eef_stats.json"].get("sha256") != dataset_stats_sha256
    ):
        raise TrainingContractError(
            "resume checkpoint embedded PI0 stats differ from the live dataset stats"
        )
    load_optimizer_state(optimizer, state_dir)
    load_scheduler_state(scheduler, state_dir)
    return global_samples


def _checkpoint_step_from_name(name: str) -> int | None:
    if not name.startswith("step-") or len(name) != len("step-000000"):
        return None
    suffix = name.removeprefix("step-")
    return int(suffix) if suffix.isdigit() else None


def _complete_checkpoint_step(
    checkpoint: Path,
    *,
    step: int,
    expected: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> bool:
    """Return whether a target/staging directory is safe to publish and resume."""

    try:
        checkpoint_manifest = shared._read_json(checkpoint / "checkpoint_manifest.json")
        training_state = shared._read_json(checkpoint / "training_state/training_step.json")
        model_manifest = shared._read_json(
            checkpoint / "pretrained_model/franka_pi0_checkpoint_manifest.json"
        )
    except (FileNotFoundError, json.JSONDecodeError, TrainingContractError):
        return False

    signature = run_manifest.get("trainability_signature_sha256")
    stats_sha256 = expected.get("dataset_stats_sha256")
    outer_expected = {
        "step": step,
        "config_sha256": expected.get("config_sha256"),
        "monitor_subset_sha256": expected.get("monitor_subset_sha256"),
        "wandb_run_id": run_manifest.get("wandb_run_id"),
        "world_size": expected.get("world_size"),
        "trainability_signature_sha256": signature,
        "dataset_stats_sha256": stats_sha256,
        "hub_upload": False,
        "wandb_artifact_upload": False,
        "atomic_publish": True,
    }
    if any(checkpoint_manifest.get(key) != value for key, value in outer_expected.items()):
        return False
    state_expected = {
        "step": step,
        "world_size": expected.get("world_size"),
        "per_device_batch_size": run_manifest.get("per_device_batch_size"),
        "gradient_accumulation_steps": run_manifest.get(
            "gradient_accumulation_steps"
        ),
        "trainability_signature_sha256": signature,
        "dataset_stats_sha256": stats_sha256,
    }
    if any(training_state.get(key) != value for key, value in state_expected.items()):
        return False
    if (
        isinstance(training_state.get("global_samples"), bool)
        or not isinstance(training_state.get("global_samples"), int)
        or training_state["global_samples"] <= 0
        or checkpoint_manifest.get("global_samples") != training_state["global_samples"]
    ):
        return False

    model_files = model_manifest.get("files")
    if (
        model_manifest.get("checkpoint_type")
        != "franka_pi0_configurable_parameter_eef"
        or not isinstance(model_files, Mapping)
        or not isinstance(model_files.get("pi0_eef_stats.json"), Mapping)
        or model_files["pi0_eef_stats.json"].get("sha256") != stats_sha256
    ):
        return False
    required_files = (
        checkpoint / "pretrained_model/config.json",
        checkpoint / "pretrained_model/model.safetensors",
        checkpoint / "pretrained_model/policy_preprocessor.json",
        checkpoint / "pretrained_model/policy_postprocessor.json",
        checkpoint / "pretrained_model/pi0_eef_stats.json",
        checkpoint / "training_state/optimizer_state.safetensors",
        checkpoint / "training_state/optimizer_param_groups.json",
        checkpoint / "training_state/scheduler_state.json",
    )
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required_files):
        return False
    if shared._sha256_file(
        checkpoint / "pretrained_model/pi0_eef_stats.json"
    ) != stats_sha256:
        return False
    world_size = expected.get("world_size")
    if not isinstance(world_size, int) or world_size <= 0:
        return False
    return all(
        (
            checkpoint
            / "training_state"
            / f"rank-{rank:02d}"
            / "rng_state.safetensors"
        ).is_file()
        for rank in range(world_size)
    )


def recover_checkpoint_publication(
    run_dir: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Recover complete staging/targets and quarantine incomplete staging dirs."""

    checkpoints = run_dir / "checkpoints"
    run_manifest = shared._read_json(run_dir / "run_manifest.json")
    identity_drift = {
        key: {"expected": value, "actual": run_manifest.get(key)}
        for key, value in expected.items()
        if run_manifest.get(key) != value
    }
    if identity_drift:
        raise TrainingContractError(
            "resume run identity mismatch before checkpoint recovery: "
            f"{identity_drift}"
        )
    checkpoints.mkdir(parents=True, exist_ok=True)
    recovered_staging: list[str] = []
    quarantined_staging: list[str] = []

    for staging in sorted(checkpoints.glob(".step-*.staging")):
        encoded_name = staging.name.removeprefix(".").removesuffix(".staging")
        step = _checkpoint_step_from_name(encoded_name)
        target = checkpoints / encoded_name
        complete = step is not None and _complete_checkpoint_step(
            staging,
            step=step,
            expected=expected,
            run_manifest=run_manifest,
        )
        if complete and not target.exists():
            staging.rename(target)
            recovered_staging.append(target.name)
            continue
        quarantine = checkpoints / (
            f".abandoned-{encoded_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        staging.rename(quarantine)
        quarantined_staging.append(quarantine.name)

    valid: list[tuple[int, Path]] = []
    for candidate in checkpoints.glob("step-*"):
        if not candidate.is_dir():
            continue
        step = _checkpoint_step_from_name(candidate.name)
        if (
            step is not None
            and step <= int(expected["planned_total_steps"])
            and _complete_checkpoint_step(
                candidate,
                step=step,
                expected=expected,
                run_manifest=run_manifest,
            )
        ):
            valid.append((step, candidate))
    if not valid:
        return {
            "recovered_staging": recovered_staging,
            "quarantined_staging": quarantined_staging,
            "pointer_updated": False,
            "checkpoint": None,
        }

    step, checkpoint = max(valid, key=lambda item: item[0])
    pointer_path = checkpoints / "last_checkpoint.json"
    desired_pointer = {"step": step, "path": str(checkpoint.relative_to(run_dir))}
    current_pointer: Mapping[str, Any] | None = None
    try:
        current_pointer = shared._read_json(pointer_path)
    except (FileNotFoundError, json.JSONDecodeError, TrainingContractError):
        pass
    pointer_updated = current_pointer != desired_pointer
    if pointer_updated:
        pointer_tmp = checkpoints / ".last_checkpoint.recovery.tmp"
        shared._write_json(pointer_tmp, desired_pointer)
        os.replace(pointer_tmp, pointer_path)
    return {
        "recovered_staging": recovered_staging,
        "quarantined_staging": quarantined_staging,
        "pointer_updated": pointer_updated,
        "checkpoint": str(checkpoint),
        "step": step,
    }


def _monitor_loss(
    policy: torch.nn.Module,
    preprocessor: Any,
    dataset: Any,
    indices: Sequence[int],
    *,
    batch_size: int,
    seed: int,
    accelerator: Any,
    trainability_spec: Mapping[str, Any],
) -> float:
    value = shared._monitor_loss(
        policy,
        preprocessor,
        dataset,
        indices,
        batch_size=batch_size,
        seed=seed,
        accelerator=accelerator,
    )
    enforce_frozen_pi0_eval_mode(accelerator.unwrap_model(policy), trainability_spec)
    return value


def _allowed_structural_missing(parameter_report: Mapping[str, Any]) -> list[str]:
    trainable_names = set(parameter_report["trainable_parameter_names"])
    return sorted(
        f"model.{core_name}"
        for core_name, _ in pi0_io.STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS
        if f"model.{core_name}" in trainable_names
    )


def _require_all_ranks_success(
    accelerator: Any, local_error: str | None, where: str
) -> None:
    if not shared._all_true(local_error is None, accelerator.device):
        raise TrainingContractError(
            f"{where} failed on at least one rank; local={local_error}"
        )


def train(
    config_path: str | Path,
    *,
    max_steps: int | None = None,
    resume_run_dir: str | Path | None = None,
) -> Path:
    """Execute one stage-free configurable-parameter training invocation."""

    cfg, resolved_config_path, config_sha = load_and_validate_config(config_path)
    trainability_spec = cfg["model"]["trainability"]

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.device.type != "cuda":
        raise TrainingContractError("PI0 finetuning requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise TrainingContractError("selected CUDA device does not support BF16")
    if accelerator.num_processes != cfg["training"]["gpu_count"]:
        raise TrainingContractError(
            f"launch world size {accelerator.num_processes} != "
            f"config gpu_count {cfg['training']['gpu_count']}"
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
    profile = shared._runtime_dataset_contract(cfg, dataset)
    batch_size = cfg["training"]["per_device_batch_size"]
    accumulation = cfg["training"]["gradient_accumulation_steps"]
    rank_samples = math.ceil(len(dataset) / accelerator.num_processes)
    rank_batches = math.ceil(rank_samples / batch_size)
    steps_per_epoch = math.ceil(rank_batches / accumulation)
    planned_total_steps = (
        steps_per_epoch * cfg["training"]["epochs"]
        if cfg["training"]["steps"] == "auto"
        else cfg["training"]["steps"]
    )
    stop_step = (
        planned_total_steps
        if max_steps is None
        else shared._positive_int(max_steps, "--max-steps")
    )
    if stop_step > planned_total_steps:
        raise TrainingContractError(
            f"--max-steps {stop_step} exceeds planned step {planned_total_steps}"
        )
    warmup = cfg["training"].get("warmup_steps")
    if warmup is None:
        warmup = min(
            cfg["training"]["warmup_max_steps"],
            max(1, int(planned_total_steps * cfg["training"]["warmup_fraction"])),
        )
    if warmup >= planned_total_steps:
        raise TrainingContractError(
            f"warmup steps {warmup} must be less than planned steps {planned_total_steps}"
        )

    dataset_profile_sha256 = shared._sha256_file(
        Path(cfg["dataset"]["root"]) / "meta/franka_eef_profile.json"
    )
    stats_path = Path(cfg["dataset"]["root"]) / "meta/pi0_eef_stats.json"
    dataset_stats_sha256 = shared._sha256_file(stats_path)
    monitor_interval = cfg["training"].get(
        "monitor_freq",
        steps_per_epoch * cfg["training"].get("monitor_freq_epochs", 1),
    )
    save_interval = shared.resolve_checkpoint_interval_steps(
        cfg["training"], steps_per_epoch=steps_per_epoch
    )
    monitor = shared.select_monitor_subset(
        dataset, cfg["training"]["monitor_max_samples"], cfg["training"]["seed"] + 17
    )

    resume_path = Path(resume_run_dir) if resume_run_dir is not None else None
    run_dir = shared._setup_run_dir(
        Path(cfg["output"]["root"]).expanduser().resolve(),
        cfg["job_name"],
        accelerator,
        resume_path,
    )
    log_path = shared._configure_logging(run_dir, accelerator)
    run_artifact_error: str | None = None
    if accelerator.is_main_process and resume_path is None:
        try:
            shutil.copy2(resolved_config_path, run_dir / "resolved_config.yaml")
            shared._write_json(run_dir / "monitor_subset.json", monitor)
        except Exception as error:
            run_artifact_error = f"{type(error).__name__}: {error}"
    _require_all_ranks_success(
        accelerator, run_artifact_error, "initial run artifact publication"
    )
    accelerator.wait_for_everyone()

    # The exact signature is only known after canonicalizing/loading the model.
    checkpoint: Path | None = None
    start_step = 0
    existing_manifest: dict[str, Any] | None = None
    preliminary_expected = {
        "config_sha256": config_sha,
        "monitor_subset_sha256": monitor["subset_sha256"],
        "world_size": accelerator.num_processes,
        "planned_total_steps": planned_total_steps,
        "dataset_profile_sha256": dataset_profile_sha256,
        "dataset_stats_sha256": dataset_stats_sha256,
    }
    if resume_path is not None:
        recovery_error: str | None = None
        recovery_report: dict[str, Any] | None = None
        if accelerator.is_main_process:
            try:
                recovery_report = recover_checkpoint_publication(
                    run_dir, preliminary_expected
                )
                logging.info("checkpoint recovery report: %s", recovery_report)
            except Exception as error:
                recovery_error = f"{type(error).__name__}: {error}"
        _require_all_ranks_success(
            accelerator, recovery_error, "checkpoint publication recovery"
        )
        accelerator.wait_for_everyone()
        resume_resolution_error: str | None = None
        try:
            checkpoint, start_step = shared._resolve_resume_checkpoint(
                run_dir, preliminary_expected
            )
            existing_manifest = shared._read_json(run_dir / "run_manifest.json")
            if stop_step <= start_step:
                raise TrainingContractError(
                    f"stop step {stop_step} must be greater than resumed step {start_step}"
                )
        except Exception as error:
            resume_resolution_error = f"{type(error).__name__}: {error}"
        _require_all_ranks_success(
            accelerator, resume_resolution_error, "resume checkpoint resolution"
        )
        assert checkpoint is not None and existing_manifest is not None

    # For resume, load complete checkpoint weights while the policy is still in
    # the canonical all-trainable state.  Trainability is reapplied immediately
    # afterwards, before constructing the optimizer.
    model_source = (
        checkpoint / "pretrained_model"
        if checkpoint is not None
        else Path(cfg["model"]["pretrained_path"])
    )
    policy, preprocessor, postprocessor, base_load_report = (
        pi0_io.load_pi0_full_policy_and_processors(
            model_source, dataset.effective_stats, accelerator.device
        )
    )
    training_graph = base_load_report["parameters"]["training_graph"]
    parameter_report = apply_pi0_trainability(policy, trainability_spec)
    trainable_parameters = trainable_pi0_parameters(policy, trainability_spec)
    signature = parameter_report["trainability_signature_sha256"]
    if existing_manifest is not None and existing_manifest.get(
        "trainability_signature_sha256"
    ) != signature:
        raise TrainingContractError("resume run uses a different PI0 trainability selection")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg["training"]["peak_lr"]),
        betas=tuple(float(value) for value in cfg["training"]["betas"]),
        eps=float(cfg["training"]["eps"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = shared.build_warmup_cosine_scheduler(
        optimizer,
        total_steps=planned_total_steps,
        warmup_steps=warmup,
        peak_lr=float(cfg["training"]["peak_lr"]),
        final_lr=float(cfg["training"]["final_lr"]),
    )
    global_samples = 0
    if checkpoint is not None:
        resume_state_error: str | None = None
        try:
            global_samples = _load_resume_optimizer_scheduler(
                checkpoint,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_step=start_step,
                world_size=accelerator.num_processes,
                batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                trainability_signature_sha256=signature,
                dataset_stats_sha256=dataset_stats_sha256,
            )
        except Exception as error:
            resume_state_error = f"{type(error).__name__}: {error}"
        _require_all_ranks_success(
            accelerator, resume_state_error, "resume optimizer/scheduler restore"
        )

    policy, optimizer = accelerator.prepare(policy, optimizer)
    assert_pi0_trainability(
        accelerator.unwrap_model(policy),
        trainability_spec,
        expected_signature=signature,
    )

    wandb_logger: shared.MetricsOnlyWandb | None = None
    wandb_setup_error: str | None = None
    if accelerator.is_main_process:
        try:
            wandb_logger = shared.MetricsOnlyWandb(
                cfg["wandb"],
                run_dir=run_dir,
                job_name=cfg["job_name"],
                resolved_config=cfg,
                run_id=existing_manifest.get("wandb_run_id") if existing_manifest else None,
                resume=resume_path is not None,
            )
            if existing_manifest is None:
                run_manifest = {
                    "schema_version": 2,
                    **preliminary_expected,
                    "job_name": cfg["job_name"],
                    "stage_metadata": cfg.get("stage"),
                    "dataset_root": str(Path(cfg["dataset"]["root"]).resolve()),
                    "dataset_size": len(dataset),
                    "steps_per_epoch": steps_per_epoch,
                    "checkpoint_interval_steps": save_interval,
                    "warmup_steps": warmup,
                    "per_device_batch_size": batch_size,
                    "gradient_accumulation_steps": accumulation,
                    "wandb_run_id": wandb_logger.run_id,
                    "wandb_url": wandb_logger.url,
                    "wandb_mode": "online",
                    "wandb_artifact_upload": False,
                    "hub_upload": False,
                    "trainability_signature_sha256": signature,
                    "parameter_training": parameter_report,
                }
                shared._write_json(run_dir / "run_manifest.json", run_manifest)
            logging.info(
                "run_dir=%s preset=%s trainable=%d/%d (%.4f%%) start_step=%d "
                "stop_step=%d planned_total_steps=%d",
                run_dir,
                parameter_report["preset"],
                parameter_report["trainable_numel"],
                parameter_report["parameter_numel"],
                100.0 * parameter_report["trainable_fraction"],
                start_step,
                stop_step,
                planned_total_steps,
            )
        except Exception as error:
            wandb_setup_error = f"{type(error).__name__}: {error}"
    _require_all_ranks_success(accelerator, wandb_setup_error, "W&B/run-manifest setup")
    accelerator.wait_for_everyone()
    wandb_run_id = shared._read_json(run_dir / "run_manifest.json")["wandb_run_id"]

    loader = shared.CyclingLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg["training"]["num_workers"],
        shuffle=cfg["training"]["shuffle"],
        seed=cfg["training"]["seed"],
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        start_microbatch=start_step * accumulation,
    )
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    gradient_gate_done = start_step > 0
    load_report = {
        "pretrained": base_load_report,
        "parameter_training": parameter_report,
        "resume_checkpoint": str(checkpoint) if checkpoint is not None else None,
    }
    geometry = shared._geometry_manifest(profile)
    invocation_started = datetime.now(timezone.utc).isoformat()
    exit_code = 1
    try:
        initial_monitor_error: str | None = None
        if resume_path is None and accelerator.is_main_process:
            try:
                initial_monitor = _monitor_loss(
                    accelerator.unwrap_model(policy),
                    preprocessor,
                    dataset,
                    monitor["indices"],
                    batch_size=batch_size,
                    seed=cfg["training"]["seed"] + 29,
                    accelerator=accelerator,
                    trainability_spec=trainability_spec,
                )
                assert wandb_logger is not None
                wandb_logger.log(
                    {
                        "monitor/loss": initial_monitor,
                        "monitor/sample_count": len(monitor["indices"]),
                    },
                    step,
                )
                logging.info("step=%d monitor_loss=%.8f", step, initial_monitor)
            except Exception as error:
                initial_monitor_error = f"{type(error).__name__}: {error}"
        _require_all_ranks_success(
            accelerator, initial_monitor_error, "initial monitor evaluation"
        )
        accelerator.wait_for_everyone()
        if checkpoint is not None:
            rng_restore_error: str | None = None
            try:
                load_rng_state(
                    checkpoint
                    / "training_state"
                    / f"rank-{accelerator.process_index:02d}"
                )
            except Exception as error:
                rng_restore_error = f"{type(error).__name__}: {error}"
            _require_all_ranks_success(
                accelerator, rng_restore_error, "rank RNG restore"
            )

        accumulated_loss = torch.zeros((), device=accelerator.device)
        accumulated_microbatches = 0
        accumulated_local_samples = 0
        update_started = time.perf_counter()
        while step < stop_step:
            batch = next(loader)
            with accelerator.accumulate(policy):
                shared.convert_uint8_images_to_float01(batch, CAMERA_KEYS)
                processed = preprocessor(batch)
                with accelerator.autocast():
                    loss, _ = policy(processed)
                if not shared._all_true(bool(torch.isfinite(loss.detach())), accelerator.device):
                    raise FloatingPointError(f"non-finite loss before backward at update {step + 1}")
                accumulated_loss += loss.detach()
                accumulated_microbatches += 1
                accumulated_local_samples += int(processed["action"].shape[0])
                accelerator.backward(loss)
                if not accelerator.sync_gradients:
                    continue
                if not gradient_gate_done:
                    gradient_gate_error: str | None = None
                    try:
                        gradient_report = pi0_gradient_coverage_summary(
                            accelerator.unwrap_model(policy),
                            trainability_spec,
                            inspect_values=True,
                        )
                        if accelerator.is_main_process:
                            shared._write_json(
                                run_dir / "first_backward_gradient_coverage.json",
                                gradient_report,
                            )
                        require_pi0_gradient_coverage(
                            gradient_report,
                            allowed_missing_parameter_names=_allowed_structural_missing(
                                parameter_report
                            ),
                        )
                    except Exception as error:
                        gradient_gate_error = f"{type(error).__name__}: {error}"
                    gradient_gate_succeeded = shared._all_true(
                        gradient_gate_error is None, accelerator.device
                    )
                    if not gradient_gate_succeeded:
                        raise TrainingContractError(
                            "first-backward gradient gate failed on at least one rank; "
                            f"local={gradient_gate_error}"
                        )
                    gradient_gate_done = True
                grad_norm = accelerator.clip_grad_norm_(
                    trainable_parameters, float(cfg["training"]["grad_clip_norm"])
                )
                if not shared._all_true(
                    bool(torch.isfinite(torch.as_tensor(grad_norm))), accelerator.device
                ):
                    raise FloatingPointError(f"non-finite gradient norm at update {step + 1}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                mean_local_loss = accumulated_loss / accumulated_microbatches
                mean_loss = accelerator.reduce(mean_local_loss, reduction="mean")
                update_global_samples = accelerator.reduce(
                    torch.tensor(
                        accumulated_local_samples,
                        device=accelerator.device,
                        dtype=torch.int64,
                    ),
                    reduction="sum",
                )
                global_samples += int(update_global_samples)
                accumulated_loss.zero_()
                accumulated_microbatches = 0
                accumulated_local_samples = 0
                duration = time.perf_counter() - update_started

                if step % cfg["training"]["log_freq"] == 0 or step == stop_step:
                    metric_log_error: str | None = None
                    if accelerator.is_main_process:
                        try:
                            assert wandb_logger is not None
                            metrics = {
                                "train/loss": float(mean_loss),
                                "train/lr": float(scheduler.get_last_lr()[0]),
                                "train/grad_norm": float(torch.as_tensor(grad_norm)),
                                "train/update_seconds": duration,
                                "train/global_samples": global_samples,
                            }
                            wandb_logger.log(metrics, step)
                            logging.info(
                                "step=%d loss=%.8f lr=%.3e grad_norm=%.5f",
                                step,
                                metrics["train/loss"],
                                metrics["train/lr"],
                                metrics["train/grad_norm"],
                            )
                        except Exception as error:
                            metric_log_error = f"{type(error).__name__}: {error}"
                    _require_all_ranks_success(
                        accelerator, metric_log_error, "training metric logging"
                    )

                if step % monitor_interval == 0 or step == stop_step:
                    accelerator.wait_for_everyone()
                    monitor_error: str | None = None
                    if accelerator.is_main_process:
                        try:
                            monitor_loss = _monitor_loss(
                                accelerator.unwrap_model(policy),
                                preprocessor,
                                dataset,
                                monitor["indices"],
                                batch_size=batch_size,
                                seed=cfg["training"]["seed"] + 29,
                                accelerator=accelerator,
                                trainability_spec=trainability_spec,
                            )
                            assert wandb_logger is not None
                            wandb_logger.log(
                                {
                                    "monitor/loss": monitor_loss,
                                    "monitor/sample_count": len(monitor["indices"]),
                                },
                                step,
                            )
                            logging.info("step=%d monitor_loss=%.8f", step, monitor_loss)
                        except Exception as error:
                            monitor_error = f"{type(error).__name__}: {error}"
                    _require_all_ranks_success(
                        accelerator, monitor_error, "monitor evaluation/logging"
                    )
                    accelerator.wait_for_everyone()

                if shared.should_save_checkpoint(
                    phase_step=step,
                    save_interval=save_interval,
                    global_step=step,
                    stop_step=stop_step,
                ):
                    published = atomic_save_checkpoint(
                        run_dir=run_dir,
                        step=step,
                        accelerator=accelerator,
                        policy=policy,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        trainability_spec=trainability_spec,
                        parameter_report=parameter_report,
                        training_graph=training_graph,
                        geometry_manifest=geometry,
                        stats_path=stats_path,
                        load_report=load_report,
                        config_sha256=config_sha,
                        monitor_sha256=monitor["subset_sha256"],
                        batch_size=batch_size,
                        gradient_accumulation_steps=accumulation,
                        wandb_run_id=wandb_run_id,
                        dataset_stats_sha256=dataset_stats_sha256,
                        global_samples=global_samples,
                    )
                    if accelerator.is_main_process:
                        logging.info("published local checkpoint %s", published)
                update_started = time.perf_counter()

        exit_code = 0
        return run_dir
    finally:
        if accelerator.is_main_process:
            shared._write_json(
                run_dir
                / "invocations"
                / (
                    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_"
                    f"{uuid.uuid4().hex[:8]}.json"
                ),
                {
                    "started_at": invocation_started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "start_step": start_step,
                    "requested_stop_step": stop_step,
                    "last_step": step,
                    "global_samples": global_samples,
                    "run_mode": "resume" if resume_path is not None else "base",
                    "trainability_signature_sha256": signature,
                    "exit_code": exit_code,
                    "log_path": str(log_path),
                },
            )
            if wandb_logger is not None:
                wandb_logger.finish(exit_code=exit_code)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop at this absolute optimizer step without changing the scheduler horizon.",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Resume the last complete local checkpoint in this immutable run directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = train(
        args.config,
        max_steps=args.max_steps,
        resume_run_dir=args.resume_run_dir,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
