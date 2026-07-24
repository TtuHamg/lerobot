#!/usr/bin/env python
"""Read-only completion verifier for the frozen Franka PI0 F0-15 run.

The default invocation performs only local artifact validation.  It does not
instantiate or load the 3B policy.  ``--deep`` adds a strict final-checkpoint
reload plus one real no-grad dataset forward, while ``--check-wandb`` adds
read-only public-API validation of the metrics-only W&B run.  No verification
level writes to the run directory or to W&B.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from franka_eef_pipeline.dual_rate_dataset import CartesianAnchorDataset  # noqa: E402
from franka_eef_pipeline.pi0_training import (  # noqa: E402
    PALIGEMMA_EMBED_TOKENS_KEY,
    PALIGEMMA_LM_HEAD_KEY,
    PI0_CORE_WEIGHTS_NAMESPACE,
    STRUCTURAL_ACTION_UNREACHABLE_NUMEL,
    STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS,
    STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT,
    UNUSED_EXPERT_LM_HEAD_KEY,
    assert_full_parameter_training,
    load_pi0_full_policy_and_processors,
)
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (  # noqa: E402
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    RNG_STATE,
    SCHEDULER_STATE,
)


CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
)
STAGE = "F0-15"
PRECHECK_STEP = 2
DATASET_SIZE = 7_465
WORLD_SIZE = 2
STEPS_PER_EPOCH = 3_733
EPOCHS = 10
FINAL_F0_STEP = 37_330
EXPECTED_MONITOR_SAMPLES = 747
EXPECTED_WARMUP_STEPS = 1_000
EXPECTED_PARAMETER_COUNT = 3_238_048_528
EXPECTED_PARAMETER_TENSORS = 776
EXPECTED_CONFIG_SHA256 = "5c8dc671d42d112ac20749ce33f1cc183a45c932a2184e92f266b083f41371d5"
EXPECTED_PROFILE_SHA256 = "398ec54d71150d2a76303c6282b07e9564bb8eb9203858dafab9edd6fbda4886"
EXPECTED_SCOPE_SHA256 = "86402a4138002b1ff6e02d95e7f434c5ef3f658c362fb88253ff8b1c44a69461"
EXPECTED_EPOCH_CHECKPOINT_STEPS = tuple(
    STEPS_PER_EPOCH * epoch for epoch in range(1, EPOCHS + 1)
)
EXPECTED_TRAIN_LOG_STEPS = frozenset(
    {PRECHECK_STEP, FINAL_F0_STEP, *range(50, FINAL_F0_STEP + 1, 50)}
)
EXPECTED_MONITOR_STEPS = frozenset(
    {0, PRECHECK_STEP, *EXPECTED_EPOCH_CHECKPOINT_STEPS}
)
EXPECTED_STRUCTURAL_MISSING_GRADIENTS = {
    f"model.{name}" for name, _shape in STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS
}
# AdamW assigns integer state ids in policy.parameters() order.  These are the
# canonical six final-prefix Parameters from the named structural ledger.  The
# adjacent q_proj (id 600) receives an allocated all-zero gradient and therefore
# correctly has optimizer state.
EXPECTED_STATELESS_OPTIMIZER_IDS = frozenset({596, 597, 598, 599, 601, 602})
CHECKPOINT_PATTERN = re.compile(r"step-(\d{6})$")
OPTIMIZER_STATE_KEY_PATTERN = re.compile(r"state/(\d+)/(exp_avg|exp_avg_sq|step)$")
WANDB_HISTORY_ARTIFACT_TYPE = "wandb-history"
WANDB_HISTORY_ARTIFACT_FILE = "0000.parquet"
# The frozen run's compact metrics history is only about 50 KiB.  A 1 MiB
# ceiling leaves ample encoding headroom while making this exception unusable
# for a policy checkpoint or other large payload.
MAX_WANDB_HISTORY_ARTIFACT_BYTES = 1 * 1024 * 1024
FORBIDDEN_WANDB_ARTIFACT_PATTERN = re.compile(
    r"(?:safetensors|checkpoint|checkpoints|ckpt|models?|weights?|"
    r"(?:^|[/_.-])(?:bin|pt|pth)(?:$|[/_.-]))",
    re.IGNORECASE,
)


class VerificationError(RuntimeError):
    """Raised when artifacts do not prove the frozen completed F0 contract."""


@dataclass(frozen=True)
class VerifiedLocalRun:
    run_dir: Path
    config: dict[str, Any]
    run_manifest: dict[str, Any]
    monitor: dict[str, Any]
    checkpoint: Path
    checkpoint_manifest: dict[str, Any]
    model_dir: Path
    report: dict[str, Any]


def _read_json_value(path: Path) -> Any:
    if not path.is_file():
        raise VerificationError(f"missing JSON artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON artifact: {path}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _require_nonempty_file(path: Path) -> None:
    _require(path.is_file() and path.stat().st_size > 0, f"missing/empty required file: {path}")


def _validate_training_graph(graph: Any) -> dict[str, Any]:
    _require(isinstance(graph, dict), "training_graph must be a JSON object")
    expected = {
        "schema_version": 2,
        "paligemma_lm_head_tied": True,
        "expert_lm_head_pruned": True,
        "deduplicated_tied_parameter_count": 526_647_296,
        "pruned_unused_parameter_count": 263_323_648,
        "parameters_removed_from_optimizer": 789_970_944,
    }
    mismatches = {
        key: {"expected": value, "actual": graph.get(key)}
        for key, value in expected.items()
        if graph.get(key) != value
    }
    _require(not mismatches, f"training_graph contract mismatch: {mismatches}")
    structural = graph.get("structural_action_unreachable")
    expected_parameters = [
        {
            "core_parameter_name": name,
            "policy_parameter_name": f"model.{name}",
            "shape": list(shape),
            "numel": math.prod(shape),
        }
        for name, shape in STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS
    ]
    expected_structural = {
        "classification": "permanent_final_prefix_output_outside_action_loss",
        "parameters_retained_and_trainable": True,
        "ddp_requires_find_unused_parameters": True,
        "tensor_count": STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT,
        "numel": STRUCTURAL_ACTION_UNREACHABLE_NUMEL,
        "parameters": expected_parameters,
    }
    _require(
        structural == expected_structural,
        "training_graph structural action-unreachable ledger mismatch",
    )
    return graph


def _validate_full_parameter_report(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(isinstance(value, dict), "run manifest lacks full parameter report")
    report = value
    expected = {
        "use_peft": False,
        "lora_parameter_count": 0,
        "parameter_tensor_count": EXPECTED_PARAMETER_TENSORS,
        "total_parameters": EXPECTED_PARAMETER_COUNT,
        "trainable_parameters": EXPECTED_PARAMETER_COUNT,
        "trainable_fraction": 1.0,
        "gradient_checkpointing_enabled": True,
        "policy_training_mode": True,
        "paligemma_lm_head_tied": True,
        "expert_lm_head_pruned": True,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": report.get(key)}
        for key, expected_value in expected.items()
        if report.get(key) != expected_value
    }
    _require(not mismatches, f"full-parameter report mismatch: {mismatches}")
    return report, _validate_training_graph(report.get("training_graph"))


def _validate_config(config_path: Path) -> dict[str, Any]:
    _require_nonempty_file(config_path)
    _require(
        _sha256_file(config_path) == EXPECTED_CONFIG_SHA256,
        "resolved F0 config differs from the frozen launch config",
    )
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise VerificationError(f"invalid resolved config: {config_path}") from exc
    _require(isinstance(config, dict), "resolved config root must be a mapping")
    dataset = config.get("dataset", {})
    training = config.get("training", {})
    model = config.get("model", {})
    _require(config.get("stage") == STAGE, "verifier accepts only the frozen F0-15 stage")
    _require(
        (
            dataset.get("profile"),
            dataset.get("observation_fps"),
            dataset.get("action_fps"),
            dataset.get("chunk_size"),
        )
        == ("action15", 15, 15, 50),
        "F0 dataset must be obs15/action15 with a 50-step chunk",
    )
    _require(dataset.get("episode_indices") is None, "F0 must use all 22 episodes")
    _require(dataset.get("max_anchors_per_episode") is None, "F0 cannot cap episode anchors")
    _require(dataset.get("expected_valid_anchors") == DATASET_SIZE, "F0 dataset size contract changed")
    _require(
        dataset.get("expected_scope_content_sha256") == EXPECTED_SCOPE_SHA256,
        "F0 22-of-25 scope hash changed",
    )
    expected_training = {
        "epochs": EPOCHS,
        "steps": "auto",
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gpu_count": WORLD_SIZE,
        "log_freq": 50,
        "monitor_freq_epochs": 1,
        "save_freq": "epoch",
        "save_optimizer_state": True,
        "require_gradient_coverage": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": training.get(key)}
        for key, expected in expected_training.items()
        if training.get(key) != expected
    }
    _require(not mismatches, f"F0 training config mismatch: {mismatches}")
    expected_model = {
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "use_relative_actions": False,
        "peft": None,
    }
    model_mismatches = {
        key: {"expected": expected, "actual": model.get(key)}
        for key, expected in expected_model.items()
        if model.get(key) != expected
    }
    _require(not model_mismatches, f"F0 model config is not full-parameter PI0: {model_mismatches}")
    wandb = config.get("wandb", {})
    _require(
        wandb.get("enable") is True
        and wandb.get("mode") == "online"
        and wandb.get("disable_artifact") is True
        and wandb.get("save_code") is False,
        "F0 W&B config must be online metrics-only",
    )
    return config


def _verify_monitor(root: Path, run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    monitor = _read_json(root / "monitor_subset.json")
    unhashed = dict(monitor)
    digest = unhashed.pop("subset_sha256", None)
    _require(digest == _canonical_sha256(unhashed), "monitor subset canonical SHA mismatch")
    _require(digest == run_manifest.get("monitor_subset_sha256"), "monitor/run SHA mismatch")
    _require(
        monitor.get("source") == "training_data" and monitor.get("held_out") is False,
        "monitor is not a fixed in-train diagnostic",
    )
    _require(monitor.get("population_size") == DATASET_SIZE, "monitor population mismatch")
    _require(monitor.get("sample_size") == EXPECTED_MONITOR_SAMPLES, "monitor sample count mismatch")
    indices = monitor.get("indices")
    anchors = monitor.get("anchors")
    _require(
        isinstance(indices, list)
        and len(indices) == EXPECTED_MONITOR_SAMPLES
        and len(set(indices)) == len(indices)
        and all(isinstance(index, int) and 0 <= index < DATASET_SIZE for index in indices),
        "monitor indices are missing, duplicated, or out of range",
    )
    _require(isinstance(anchors, list) and len(anchors) == len(indices), "monitor anchor list mismatch")
    _require(
        monitor.get("includes_converter_frozen_indices") == indices,
        "F0 monitor must use the converter-frozen subset exactly",
    )
    return monitor


def _verify_invocations_and_logs(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "invocations").glob("*.json"))
    records = [_read_json(path) for path in paths]
    records.sort(key=lambda item: str(item.get("started_at", "")))
    _require(
        len(records) == 2,
        f"completed F0 must have exactly precheck+resume invocations, got {len(records)}",
    )
    expected = ((0, PRECHECK_STEP, False), (PRECHECK_STEP, FINAL_F0_STEP, True))
    for index, (record, (start, stop, resumed)) in enumerate(zip(records, expected, strict=True)):
        actual = (record.get("start_step"), record.get("last_step"), record.get("resume"))
        _require(actual == (start, stop, resumed), f"F0 invocation {index} discontinuity: {actual}")
        _require(record.get("requested_stop_step") == stop, f"F0 invocation {index} stop mismatch")
        _require(record.get("exit_code") == 0, f"F0 invocation {index} did not exit successfully")
    logs_root = root / "logs"
    for rank in range(WORLD_SIZE):
        logs = sorted(logs_root.glob(f"*_r{rank}.log"))
        _require(len(logs) == 2, f"F0 rank {rank} must have precheck+resume logs, got {logs}")
        for log in logs:
            _require_nonempty_file(log)
    return records


def _verify_gradient_report(root: Path) -> dict[str, Any]:
    gradient = _read_json(root / "first_backward_gradient_coverage.json")
    _require(gradient.get("inspect_values") is True, "first backward did not inspect gradient values")
    overall = gradient.get("overall")
    _require(isinstance(overall, dict), "first backward report has no overall summary")
    expected_overall = {
        "parameter_tensors": EXPECTED_PARAMETER_TENSORS,
        "parameter_numel": EXPECTED_PARAMETER_COUNT,
        "trainable_tensors": EXPECTED_PARAMETER_TENSORS,
        "trainable_numel": EXPECTED_PARAMETER_COUNT,
        "gradient_tensors": EXPECTED_PARAMETER_TENSORS - STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT,
        "gradient_numel": EXPECTED_PARAMETER_COUNT - STRUCTURAL_ACTION_UNREACHABLE_NUMEL,
        "all_gradients_finite": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": overall.get(key)}
        for key, expected in expected_overall.items()
        if overall.get(key) != expected
    }
    _require(not mismatches, f"first-backward overall contract mismatch: {mismatches}")
    groups = gradient.get("groups")
    _require(isinstance(groups, dict), "first backward report has no subsystem groups")
    missing_names: list[str] = []
    for group_name, group in groups.items():
        _require(isinstance(group, dict), f"invalid first-backward group: {group_name}")
        names = group.get("missing_gradient_names")
        _require(
            isinstance(names, list) and all(isinstance(name, str) for name in names),
            f"first-backward {group_name} lacks a string missing-gradient list",
        )
        missing_names.extend(names)
    _require(len(missing_names) == len(set(missing_names)), "first backward repeats missing names")
    _require(
        set(missing_names) == EXPECTED_STRUCTURAL_MISSING_GRADIENTS,
        "first backward differs from the exact six-Parameter structural ledger",
    )
    for name in ("vision_encoder", "vlm", "action_expert", "state_action_projections"):
        group = groups.get(name)
        _require(isinstance(group, dict), f"first backward missing group: {name}")
        _require(group.get("all_gradients_finite") is True, f"first backward {name} is non-finite")
        _require(int(group.get("nonzero_gradient_tensors", 0)) > 0, f"first backward {name} is all-zero")
        if name != "vlm":
            _require(group.get("gradient_tensor_coverage") == 1.0, f"{name} tensor coverage changed")
            _require(group.get("gradient_numel_coverage") == 1.0, f"{name} numel coverage changed")
    return gradient


def _verify_checkpoint_file_manifest(
    model_dir: Path,
    manifest: Mapping[str, Any],
    *,
    hashes: bool,
) -> None:
    files = manifest.get("files")
    expected_files = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "franka_eef_geometry_manifest.json",
        "pi0_eef_stats.json",
    }
    _require(isinstance(files, dict) and set(files) == expected_files, "model file manifest mismatch")
    for filename, record in files.items():
        _require(isinstance(record, dict), f"invalid model file record: {filename}")
        path = model_dir / filename
        _require_nonempty_file(path)
        _require(path.stat().st_size == record.get("size_bytes"), f"size mismatch: {path}")
        if hashes:
            _require(_sha256_file(path) == record.get("sha256"), f"SHA256 mismatch: {path}")
    for config_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor_config = _read_json(model_dir / config_name)
        steps = processor_config.get("steps")
        _require(isinstance(steps, list), f"processor config has no steps: {config_name}")
        for step in steps:
            _require(isinstance(step, dict), f"invalid processor step in {config_name}")
            state_file = step.get("state_file")
            if state_file is not None:
                _require(isinstance(state_file, str) and state_file, f"invalid processor state in {config_name}")
                state_path = (model_dir / state_file).resolve()
                _require(
                    state_path.is_relative_to(model_dir.resolve()),
                    f"processor state escapes model directory in {config_name}: {state_file}",
                )
                _require_nonempty_file(state_path)


def _verify_checkpoint_shell(
    checkpoint: Path,
    *,
    step: int,
    run_manifest: Mapping[str, Any],
    full_report: Mapping[str, Any],
    graph: Mapping[str, Any],
    verify_hashes: bool,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest = _read_json(checkpoint / "checkpoint_manifest.json")
    expected_manifest = {
        "step": step,
        "config_sha256": run_manifest["config_sha256"],
        "monitor_subset_sha256": run_manifest["monitor_subset_sha256"],
        "wandb_run_id": run_manifest["wandb_run_id"],
        "world_size": WORLD_SIZE,
        "hub_upload": False,
        "wandb_artifact_upload": False,
        "atomic_publish": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    _require(not mismatches, f"checkpoint manifest mismatch at step {step}: {mismatches}")
    state_dir = checkpoint / "training_state"
    training_step = _read_json(state_dir / "training_step.json")
    expected_state = {
        "step": step,
        "world_size": WORLD_SIZE,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    _require(training_step == expected_state, f"training-state contract mismatch at step {step}")
    for filename in (OPTIMIZER_STATE, OPTIMIZER_PARAM_GROUPS, SCHEDULER_STATE):
        _require_nonempty_file(state_dir / filename)
    rank_dirs = sorted(path.name for path in state_dir.glob("rank-*") if path.is_dir())
    _require(rank_dirs == ["rank-00", "rank-01"], f"rank RNG directory mismatch at step {step}")
    for rank in range(WORLD_SIZE):
        _require_nonempty_file(state_dir / f"rank-{rank:02d}" / RNG_STATE)
    model_dir = checkpoint / "pretrained_model"
    model_manifest = _read_json(model_dir / "franka_pi0_checkpoint_manifest.json")
    _require(model_manifest.get("weights_namespace") == PI0_CORE_WEIGHTS_NAMESPACE, "model namespace mismatch")
    _require(model_manifest.get("training_graph") == graph, f"model graph drift at step {step}")
    _require(model_manifest.get("parameter_training") == full_report, f"model parameter drift at step {step}")
    _require(model_manifest.get("hub_upload") is False, f"model permits Hub upload at step {step}")
    _require(
        model_manifest.get("wandb_artifact_upload") is False,
        f"model permits W&B artifact upload at step {step}",
    )
    _verify_checkpoint_file_manifest(model_dir, model_manifest, hashes=verify_hashes)
    return manifest, model_dir, model_manifest


def _verify_final_model_header(model_dir: Path) -> None:
    weights_path = model_dir / "model.safetensors"
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as tensors:
            keys = set(tensors.keys())
            metadata = dict(tensors.metadata() or {})
    except Exception as exc:
        raise VerificationError(f"cannot inspect final model safetensors header: {weights_path}") from exc
    _require(len(keys) == EXPECTED_PARAMETER_TENSORS, "final model physical tensor count mismatch")
    _require(UNUSED_EXPERT_LM_HEAD_KEY not in keys, "final model retained unused expert LM head")
    tied_keys = {PALIGEMMA_LM_HEAD_KEY, PALIGEMMA_EMBED_TOKENS_KEY}
    physical = keys.intersection(tied_keys)
    _require(len(physical) == 1, f"final model tied embedding physical keys mismatch: {physical}")
    physical_key = next(iter(physical))
    logical_key = next(iter(tied_keys - physical))
    _require(metadata.get(logical_key) == physical_key, "final model tied alias metadata mismatch")


def _verify_final_model_config(model_dir: Path) -> None:
    config = _read_json(model_dir / "config.json")
    expected = {
        "type": "pi0",
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "use_relative_actions": False,
        "use_peft": False,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "chunk_size": 50,
        "n_action_steps": 50,
        "push_to_hub": False,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": config.get(key)}
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    _require(not mismatches, f"final saved PI0 config mismatch: {mismatches}")
    _require(config.get("input_features", {}).get("observation.state", {}).get("shape") == [10], "saved state dim != 10")
    _require(config.get("output_features", {}).get("action", {}).get("shape") == [7], "saved action dim != 7")


def _verify_final_optimizer_scheduler(state_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    training = config["training"]
    scheduler = _read_json(state_dir / SCHEDULER_STATE)
    expected_scheduler = {
        "base_lrs": [float(training["peak_lr"])],
        "last_epoch": FINAL_F0_STEP,
        "_step_count": FINAL_F0_STEP + 1,
        "_last_lr": [float(training["final_lr"])],
        "lr_lambdas": [None],
    }
    mismatches = {
        key: {"expected": expected, "actual": scheduler.get(key)}
        for key, expected in expected_scheduler.items()
        if scheduler.get(key) != expected
    }
    _require(not mismatches, f"final scheduler state mismatch: {mismatches}")

    groups = _read_json_value(state_dir / OPTIMIZER_PARAM_GROUPS)
    _require(isinstance(groups, list) and len(groups) == 1 and isinstance(groups[0], dict), "optimizer must have one param group")
    group = groups[0]
    expected_group = {
        "lr": float(training["final_lr"]),
        "initial_lr": float(training["peak_lr"]),
        "betas": [float(value) for value in training["betas"]],
        "eps": float(training["eps"]),
        "weight_decay": float(training["weight_decay"]),
    }
    group_mismatches = {
        key: {"expected": expected, "actual": group.get(key)}
        for key, expected in expected_group.items()
        if group.get(key) != expected
    }
    _require(not group_mismatches, f"final optimizer param-group mismatch: {group_mismatches}")
    _require(group.get("params") == list(range(EXPECTED_PARAMETER_TENSORS)), "optimizer parameter ids changed")

    optimizer_path = state_dir / OPTIMIZER_STATE
    expected_active_ids = set(range(EXPECTED_PARAMETER_TENSORS)) - set(EXPECTED_STATELESS_OPTIMIZER_IDS)
    fields_by_id: dict[int, set[str]] = {}
    step_values: list[float] = []
    try:
        with safe_open(optimizer_path, framework="pt", device="cpu") as tensors:
            keys = list(tensors.keys())
            for key in keys:
                match = OPTIMIZER_STATE_KEY_PATTERN.fullmatch(key)
                _require(match is not None, f"unexpected optimizer tensor key: {key}")
                parameter_id = int(match.group(1))
                field = match.group(2)
                fields_by_id.setdefault(parameter_id, set()).add(field)
            _require(set(fields_by_id) == expected_active_ids, "optimizer active/stateless parameter ids mismatch")
            for parameter_id in sorted(expected_active_ids):
                _require(
                    fields_by_id[parameter_id] == {"exp_avg", "exp_avg_sq", "step"},
                    f"optimizer fields incomplete for parameter {parameter_id}",
                )
                exp_shape = tensors.get_slice(f"state/{parameter_id}/exp_avg").get_shape()
                sq_shape = tensors.get_slice(f"state/{parameter_id}/exp_avg_sq").get_shape()
                step_shape = tensors.get_slice(f"state/{parameter_id}/step").get_shape()
                _require(exp_shape == sq_shape, f"Adam moment shapes differ for parameter {parameter_id}")
                _require(step_shape == [], f"Adam step is not scalar for parameter {parameter_id}")
                step_values.append(float(tensors.get_tensor(f"state/{parameter_id}/step")))
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"cannot inspect final optimizer state: {optimizer_path}") from exc
    _require(
        len(step_values) == EXPECTED_PARAMETER_TENSORS - STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT
        and set(step_values) == {float(FINAL_F0_STEP)},
        "Adam state steps are not all equal to the final F0 step",
    )
    return {
        "scheduler_last_epoch": scheduler["last_epoch"],
        "scheduler_last_lr": scheduler["_last_lr"],
        "optimizer_parameter_ids": EXPECTED_PARAMETER_TENSORS,
        "optimizer_active_state_ids": len(expected_active_ids),
        "optimizer_stateless_ids": sorted(EXPECTED_STATELESS_OPTIMIZER_IDS),
        "optimizer_step": FINAL_F0_STEP,
    }


def _verify_deep_optimizer_parameter_mapping(
    policy: torch.nn.Module,
    state_dir: Path,
    *,
    metadata_names: Iterable[str],
    gradient_names: Iterable[str],
) -> dict[str, Any]:
    """Map saved optimizer ids to the freshly loaded policy's parameter names.

    PyTorch assigns optimizer state ids by walking each param group's Parameters
    in order.  The F0 optimizer was constructed directly from
    ``policy.parameters()``, whose order is the same as
    ``policy.named_parameters()``.  Rebuilding that mapping in deep mode gives
    an independent semantic check that the six ids without Adam state really
    are the six final-prefix Parameters declared by both run metadata and the
    first-backward report; a numerically hard-coded id set alone cannot prove
    that relationship if model registration order drifts.
    """

    groups = _read_json_value(state_dir / OPTIMIZER_PARAM_GROUPS)
    _require(isinstance(groups, list) and groups, "deep optimizer has no param groups")
    ordered_parameter_ids: list[int] = []
    for group_index, group in enumerate(groups):
        _require(isinstance(group, dict), f"deep optimizer param group {group_index} is invalid")
        parameter_ids = group.get("params")
        _require(
            isinstance(parameter_ids, list)
            and all(
                isinstance(parameter_id, int) and not isinstance(parameter_id, bool)
                for parameter_id in parameter_ids
            ),
            f"deep optimizer param group {group_index} has invalid parameter ids",
        )
        ordered_parameter_ids.extend(parameter_ids)
    _require(
        len(ordered_parameter_ids) == len(set(ordered_parameter_ids)),
        "deep optimizer parameter ids are duplicated",
    )

    named_parameters = list(policy.named_parameters())
    parameter_names = [name for name, _parameter in named_parameters]
    _require(
        parameter_names and len(parameter_names) == len(set(parameter_names)),
        "deep policy parameter names are empty/duplicated",
    )
    _require(
        len({id(parameter) for _name, parameter in named_parameters}) == len(named_parameters),
        "deep policy named_parameters contains duplicate Parameter objects",
    )
    _require(
        len(ordered_parameter_ids) == len(named_parameters),
        "deep optimizer/policy parameter tensor counts differ: "
        f"optimizer={len(ordered_parameter_ids)}, policy={len(named_parameters)}",
    )
    id_to_name = dict(zip(ordered_parameter_ids, parameter_names, strict=True))

    optimizer_path = state_dir / OPTIMIZER_STATE
    fields_by_id: dict[int, set[str]] = {}
    try:
        with safe_open(optimizer_path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                match = OPTIMIZER_STATE_KEY_PATTERN.fullmatch(key)
                _require(match is not None, f"unexpected optimizer tensor key during deep mapping: {key}")
                parameter_id = int(match.group(1))
                fields_by_id.setdefault(parameter_id, set()).add(match.group(2))
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(
            f"cannot map final optimizer state to policy Parameters: {optimizer_path}"
        ) from exc
    _require(
        set(fields_by_id).issubset(id_to_name),
        "deep optimizer state contains ids outside its param groups",
    )
    for parameter_id, fields in fields_by_id.items():
        _require(
            fields == {"exp_avg", "exp_avg_sq", "step"},
            f"deep optimizer fields incomplete for parameter {parameter_id}",
        )

    stateless_ids = [
        parameter_id
        for parameter_id in ordered_parameter_ids
        if parameter_id not in fields_by_id
    ]
    stateless_names = [id_to_name[parameter_id] for parameter_id in stateless_ids]
    metadata_name_list = list(metadata_names)
    gradient_name_list = list(gradient_names)
    for source, names in (("run metadata", metadata_name_list), ("gradient report", gradient_name_list)):
        _require(
            all(isinstance(name, str) for name in names) and len(names) == len(set(names)),
            f"deep {source} structural parameter names are invalid/duplicated",
        )
    metadata_name_set = set(metadata_name_list)
    gradient_name_set = set(gradient_name_list)
    _require(
        metadata_name_set == gradient_name_set,
        "deep run metadata and gradient report disagree on structural parameter names",
    )
    _require(
        set(stateless_names) == metadata_name_set,
        "deep stateless optimizer Parameters differ from the declared exact-six structural ledger: "
        f"mapped={sorted(stateless_names)}, declared={sorted(metadata_name_set)}",
    )
    return {
        "optimizer_parameter_ids_mapped": len(ordered_parameter_ids),
        "optimizer_stateless_ids": stateless_ids,
        "optimizer_stateless_parameter_names": stateless_names,
        "metadata_gradient_exact_match": True,
    }


def _verify_resume_provenance(
    root: Path,
    final_model_manifest: Mapping[str, Any],
    full_report: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> None:
    source = final_model_manifest.get("source_load_report")
    _require(isinstance(source, dict), "final model lacks source load report")
    base = source.get("base")
    resume = source.get("resume")
    _require(isinstance(base, dict) and base.get("pretrained", {}).get("strict") is True, "base PI0 load was not strict")
    _require(isinstance(resume, dict) and resume.get("strict") is True, "F0 did not strictly resume step 2")
    _require(resume.get("missing_keys") == [] and resume.get("unexpected_keys") == [], "resume weights were not exact")
    _require(resume.get("project_manifest_present") is True, "resume did not use a project checkpoint")
    expected_model_dir = (root / f"checkpoints/step-{PRECHECK_STEP:06d}/pretrained_model").resolve()
    _require(Path(str(resume.get("directory", ""))).resolve() == expected_model_dir, "resume source is not same-run step 2")
    _require(Path(str(resume.get("weights_path", ""))).resolve() == expected_model_dir / "model.safetensors", "resume weights path mismatch")
    _require(resume.get("parameter_training") == full_report, "resume parameter report drift")
    _require(resume.get("training_graph") == graph, "resume training graph drift")


def verify_local_run(
    run_dir: str | Path,
    *,
    verify_file_hashes: bool = False,
) -> VerifiedLocalRun:
    """Verify a completed F0 run without constructing or loading the policy."""

    root = Path(run_dir).expanduser().resolve()
    _require(root.is_dir(), f"run directory does not exist: {root}")
    _require("_ABORTED_" not in root.name and not (root / "ABORTED.json").exists(), "refusing aborted run")
    checkpoints_root = root / "checkpoints"
    staging = sorted(checkpoints_root.glob(".*.staging"))
    _require(not staging, f"refusing active/incomplete checkpoint staging: {staging}")

    config_path = root / "resolved_config.yaml"
    config = _validate_config(config_path)
    run_manifest = _read_json(root / "run_manifest.json")
    expected_manifest = {
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "stage": STAGE,
        "world_size": WORLD_SIZE,
        "planned_total_steps": FINAL_F0_STEP,
        "dataset_size": DATASET_SIZE,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "warmup_steps": EXPECTED_WARMUP_STEPS,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "dataset_profile_sha256": EXPECTED_PROFILE_SHA256,
        "wandb_mode": "online",
        "wandb_artifact_upload": False,
        "hub_upload": False,
    }
    manifest_mismatches = {
        key: {"expected": expected, "actual": run_manifest.get(key)}
        for key, expected in expected_manifest.items()
        if run_manifest.get(key) != expected
    }
    _require(not manifest_mismatches, f"F0 run manifest mismatch: {manifest_mismatches}")
    _require(run_manifest.get("config_sha256") == _sha256_file(config_path), "run/config SHA mismatch")
    wandb_id = run_manifest.get("wandb_run_id")
    _require(isinstance(wandb_id, str) and wandb_id, "run manifest lacks W&B run id")
    dataset_root = Path(str(config["dataset"]["root"])).expanduser().resolve()
    _require(str(dataset_root) == run_manifest.get("dataset_root"), "dataset root mismatch")
    profile_path = dataset_root / "meta/franka_eef_profile.json"
    _require_nonempty_file(profile_path)
    _require(_sha256_file(profile_path) == EXPECTED_PROFILE_SHA256, "dataset profile is not the frozen F0 profile")
    profile = _read_json(profile_path)
    _require(
        (
            profile.get("profile"),
            profile.get("observation_fps"),
            profile.get("action_fps"),
            profile.get("chunk_size"),
            profile.get("episode_count"),
            profile.get("task_instruction"),
            profile.get("scope_content_sha256"),
        )
        == ("action15", 15, 15, 50, 22, "stack the cups", EXPECTED_SCOPE_SHA256),
        "dataset profile content does not match F0",
    )

    full_report, graph = _validate_full_parameter_report(run_manifest.get("full_parameter_report"))
    monitor = _verify_monitor(root, run_manifest)
    pointer = _read_json(checkpoints_root / "last_checkpoint.json")
    _require(
        pointer.get("step") == FINAL_F0_STEP,
        f"F0 run is not complete: latest pointer is step {pointer.get('step')}, expected {FINAL_F0_STEP}",
    )
    checkpoint = (root / str(pointer.get("path", ""))).resolve()
    _require(checkpoint.is_relative_to(root) and checkpoint.is_dir(), "latest checkpoint pointer escapes/is missing")
    _require(checkpoint.name == f"step-{FINAL_F0_STEP:06d}", "latest checkpoint directory name mismatch")

    published: list[int] = []
    checkpoint_records: dict[int, tuple[dict[str, Any], Path, dict[str, Any]]] = {}
    for path in sorted(checkpoints_root.glob("step-*")):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        _require(match is not None and path.is_dir(), f"invalid published checkpoint path: {path}")
        step = int(match.group(1))
        published.append(step)
    published_set = set(published)
    _require(
        set(EXPECTED_EPOCH_CHECKPOINT_STEPS).issubset(published_set),
        f"F0 is missing epoch checkpoints: {sorted(set(EXPECTED_EPOCH_CHECKPOINT_STEPS) - published_set)}",
    )
    _require(
        published_set - set(EXPECTED_EPOCH_CHECKPOINT_STEPS) <= {PRECHECK_STEP},
        f"F0 has unexpected checkpoint steps: {published}",
    )
    _require(len(published) == len(published_set), "F0 checkpoint steps are duplicated")
    for step in published:
        path = checkpoints_root / f"step-{step:06d}"
        checkpoint_records[step] = _verify_checkpoint_shell(
            path,
            step=step,
            run_manifest=run_manifest,
            full_report=full_report,
            graph=graph,
            verify_hashes=verify_file_hashes and step == FINAL_F0_STEP,
        )

    checkpoint_manifest, model_dir, model_manifest = checkpoint_records[FINAL_F0_STEP]
    _verify_final_model_header(model_dir)
    _verify_final_model_config(model_dir)
    optimizer_scheduler = _verify_final_optimizer_scheduler(checkpoint / "training_state", config)
    _verify_resume_provenance(root, model_manifest, full_report, graph)
    gradient = _verify_gradient_report(root)
    invocations = _verify_invocations_and_logs(root)

    report = {
        "local_status": "PASS",
        "run_dir": str(root),
        "stage": STAGE,
        "dataset_size": DATASET_SIZE,
        "world_size": WORLD_SIZE,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "final_step": FINAL_F0_STEP,
        "published_checkpoint_steps": published,
        "wandb_run_id": wandb_id,
        "checkpoint": str(checkpoint),
        "training_graph": graph,
        "unique_trainable_parameters": EXPECTED_PARAMETER_COUNT,
        "monitor_sample_count": len(monitor["indices"]),
        "invocations": invocations,
        "gradient_tensor_coverage": gradient["overall"]["gradient_tensor_coverage"],
        "optimizer_scheduler": optimizer_scheduler,
        "final_model_hashes_verified": verify_file_hashes,
        "deep_model_loaded": False,
    }
    return VerifiedLocalRun(
        root,
        config,
        run_manifest,
        monitor,
        checkpoint,
        checkpoint_manifest,
        model_dir,
        report,
    )


@contextmanager
def _offline_model_loading() -> Iterable[None]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def load_saved_processors(model_dir: Path, device: torch.device) -> tuple[Any, Any]:
    """Reload the exact saved processors without network access."""

    with _offline_model_loading():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224",
            local_files_only=True,
        )
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            model_dir,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
            overrides={
                "tokenizer_processor": {"tokenizer": tokenizer, "tokenizer_name": None},
                "device_processor": {"device": str(device)},
                "normalizer_processor": {"device": str(device)},
            },
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            model_dir,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            overrides={
                "unnormalizer_processor": {"device": str(device)},
                "device_processor": {"device": "cpu"},
            },
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
    return preprocessor, postprocessor


def _convert_images(batch: dict[str, Any]) -> None:
    for key in CAMERA_KEYS:
        image = batch.get(key)
        _require(isinstance(image, torch.Tensor) and image.dtype is torch.uint8, f"{key} is not raw uint8")
        converted = image.to(torch.float32).div_(255.0)
        _require(bool(torch.isfinite(converted).all()), f"{key} contains NaN/Inf")
        _require(float(converted.min()) >= 0.0 and float(converted.max()) <= 1.0, f"{key} outside [0,1]")
        batch[key] = converted


def verify_deep(local: VerifiedLocalRun, *, device: str) -> dict[str, Any]:
    """Strictly reload the final 3B checkpoint and run one real no-grad loss."""

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable for deep verification")
        _require(
            torch_device.index is not None and torch_device.index < torch.cuda.device_count(),
            "CUDA device is invalid",
        )
        torch.cuda.set_device(torch_device)
        _require(torch.cuda.is_bf16_supported(), "selected CUDA device lacks BF16")
    dataset_cfg = local.config["dataset"]
    dataset = CartesianAnchorDataset(
        dataset_cfg["root"],
        profile=dataset_cfg["profile"],
        episode_indices=dataset_cfg["episode_indices"],
        max_anchors_per_episode=dataset_cfg["max_anchors_per_episode"],
        video_backend="pyav",
    )
    _require(len(dataset) == DATASET_SIZE == local.run_manifest.get("dataset_size"), "deep dataset size mismatch")
    _require(dataset.effective_stats is not None, "deep dataset lacks effective stats")
    with _offline_model_loading():
        policy, factory_preprocessor, factory_postprocessor, load_report = load_pi0_full_policy_and_processors(
            local.model_dir,
            dataset.effective_stats,
            torch_device,
        )
    parameter_report = assert_full_parameter_training(policy)
    _require(parameter_report == local.run_manifest["full_parameter_report"], "deep parameter report mismatch")
    training_graph = local.run_manifest["full_parameter_report"]["training_graph"]
    structural = training_graph["structural_action_unreachable"]
    metadata_names = [entry["policy_parameter_name"] for entry in structural["parameters"]]
    gradient = _read_json(local.run_dir / "first_backward_gradient_coverage.json")
    gradient_names = [
        name
        for group in gradient["groups"].values()
        for name in group["missing_gradient_names"]
    ]
    optimizer_parameter_mapping = _verify_deep_optimizer_parameter_mapping(
        policy,
        local.checkpoint / "training_state",
        metadata_names=metadata_names,
        gradient_names=gradient_names,
    )
    preprocessor, postprocessor = load_saved_processors(local.model_dir, torch_device)

    sample_index = int(local.monitor["indices"][0])
    raw_batch = torch.utils.data.default_collate([dataset[sample_index]])
    raw_action = raw_batch["action"].clone()
    factory_batch = copy.deepcopy(raw_batch)
    saved_batch = copy.deepcopy(raw_batch)
    _convert_images(factory_batch)
    _convert_images(saved_batch)
    factory_processed = factory_preprocessor(factory_batch)
    processed = preprocessor(saved_batch)
    state = processed.get("observation.state")
    action = processed.get("action")
    _require(isinstance(state, torch.Tensor) and tuple(state.shape) == (1, 10), "processed state shape != [1,10]")
    _require(isinstance(action, torch.Tensor) and tuple(action.shape) == (1, 50, 7), "processed action shape != [1,50,7]")
    _require(bool(torch.isfinite(state).all()) and bool(torch.isfinite(action).all()), "processed state/action non-finite")
    for key in CAMERA_KEYS:
        image = processed.get(key)
        _require(isinstance(image, torch.Tensor) and image.dtype is torch.float32, f"processed {key} dtype")
        _require(bool(torch.isfinite(image).all()), f"processed {key} non-finite")
        _require(float(image.min()) >= 0.0 and float(image.max()) <= 1.0, f"processed {key} outside [0,1]")
    factory_state = factory_processed.get("observation.state")
    factory_action = factory_processed.get("action")
    _require(
        isinstance(factory_state, torch.Tensor)
        and isinstance(factory_action, torch.Tensor)
        and torch.allclose(state, factory_state, atol=0.0, rtol=0.0)
        and torch.allclose(action, factory_action, atol=0.0, rtol=0.0),
        "saved processor state differs from the factory built from frozen effective stats",
    )
    restored_action = postprocessor(action.detach())
    factory_restored_action = factory_postprocessor(factory_action.detach())
    _require(isinstance(restored_action, torch.Tensor), "postprocessor did not return a tensor")
    _require(tuple(restored_action.shape) == tuple(raw_action.shape), "postprocessor action shape mismatch")
    _require(bool(torch.isfinite(restored_action).all()), "postprocessor action non-finite")
    _require(torch.allclose(restored_action, raw_action, atol=2e-5, rtol=2e-5), "processor action roundtrip mismatch")
    _require(
        isinstance(factory_restored_action, torch.Tensor)
        and torch.allclose(restored_action, factory_restored_action, atol=0.0, rtol=0.0),
        "saved postprocessor state differs from the factory built from frozen effective stats",
    )

    was_training = policy.training
    policy.eval()
    cuda_devices = [torch_device.index] if torch_device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        torch.manual_seed(int(local.config["training"]["seed"]) + 41)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed(int(local.config["training"]["seed"]) + 41)
        with torch.autocast(
            device_type=torch_device.type,
            dtype=torch.bfloat16,
            enabled=torch_device.type == "cuda",
        ):
            loss, _ = policy(processed)
    policy.train(was_training)
    _require(isinstance(loss, torch.Tensor) and loss.ndim == 0 and bool(torch.isfinite(loss)), "deep forward loss non-finite")
    return {
        "deep_status": "PASS",
        "device": str(torch_device),
        "sample_index": sample_index,
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "processor_roundtrip": True,
        "saved_processors_match_effective_stats": True,
        "forward_loss": float(loss),
        "strict_load": load_report["pretrained"],
        "parameter_training": parameter_report,
        "optimizer_parameter_mapping": optimizer_parameter_mapping,
    }


def _scalar_history(run: Any, metric: str) -> list[dict[str, float | int]]:
    rows = list(run.scan_history(keys=["trainer/step", metric], page_size=1000))
    values: list[dict[str, float | int]] = []
    for row in rows:
        # A W&B parquet scan can return unioned sparse rows from another metric
        # at the same run step.  Those rows contain this requested metric with
        # a literal None and are not observations of the metric.
        if metric not in row or row[metric] is None:
            continue
        value = row[metric]
        _require(
            "trainer/step" in row and row["trainer/step"] is not None,
            f"W&B {metric} has a value without an optimizer step",
        )
        step = row["trainer/step"]
        _require(
            isinstance(step, (int, float)) and not isinstance(step, bool),
            f"W&B {metric} step non-scalar",
        )
        _require(math.isfinite(float(step)), f"W&B {metric} step non-finite")
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"W&B {metric} non-scalar",
        )
        _require(math.isfinite(float(value)), f"W&B {metric} non-finite")
        values.append({"step": int(step), "value": float(value)})
    _require(values, f"W&B history is empty for {metric}")
    steps = [int(row["step"]) for row in values]
    _require(len(steps) == len(set(steps)), f"W&B {metric} contains duplicate optimizer steps")
    return values


def _validate_wandb_artifacts(run: Any, *, run_id: str) -> dict[str, Any]:
    """Allow only W&B's small, automatic metrics-history artifact.

    ``disable_artifact`` prevents our training code from uploading checkpoints,
    but W&B can still publish one ``wandb-history`` artifact asynchronously when
    a run finishes.  This validator therefore uses a narrow content whitelist
    instead of assuming that every finished metrics-only run has zero artifacts.
    """

    artifacts = list(run.logged_artifacts())
    _require(
        len(artifacts) <= 1,
        f"W&B run logged {len(artifacts)} artifacts; at most one automatic history artifact is allowed",
    )
    expected_name = f"run-{run_id}-history:v0"
    expected_description = f"Weights & Biases Run History Data for {run_id}"
    ledger: list[dict[str, Any]] = []
    for artifact in artifacts:
        name = getattr(artifact, "name", None)
        artifact_type = getattr(artifact, "type", None)
        description = getattr(artifact, "description", None)
        size = getattr(artifact, "size", None)
        _require(isinstance(name, str) and name, "W&B artifact lacks a string name")
        _require(isinstance(artifact_type, str) and artifact_type, "W&B artifact lacks a string type")
        _require(isinstance(description, str), f"W&B artifact {name!r} lacks a string description")

        # The exact automatic-history description starts with the W&B brand
        # name, "Weights & Biases".  Do not feed that already-whitelisted
        # description through a generic weight-marker scan.  Names and types
        # remain independently guarded, and every non-history type is denied.
        identifier_text = "\n".join((name, artifact_type))
        if FORBIDDEN_WANDB_ARTIFACT_PATTERN.search(identifier_text):
            raise VerificationError(
                f"W&B artifact {name!r} contains a forbidden model/checkpoint/weight marker"
            )
        _require(
            artifact_type == WANDB_HISTORY_ARTIFACT_TYPE,
            f"unknown W&B artifact type {artifact_type!r} for {name!r}",
        )
        _require(name == expected_name, f"unexpected W&B history artifact name: {name!r}")
        _require(
            description == expected_description,
            f"unexpected W&B history artifact description for {name!r}",
        )
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"W&B history artifact {name!r} has an invalid byte size: {size!r}",
        )
        _require(
            size <= MAX_WANDB_HISTORY_ARTIFACT_BYTES,
            f"W&B history artifact {name!r} is too large: {size} bytes",
        )

        files = list(artifact.files())
        _require(
            len(files) == 1,
            f"W&B history artifact {name!r} must contain exactly one file, found {len(files)}",
        )
        artifact_file = files[0]
        file_name = getattr(artifact_file, "name", None)
        file_size = getattr(artifact_file, "size", None)
        _require(
            isinstance(file_name, str) and file_name,
            f"W&B history artifact {name!r} contains a file without a string name",
        )
        if FORBIDDEN_WANDB_ARTIFACT_PATTERN.search(file_name):
            raise VerificationError(
                f"W&B history artifact {name!r} contains suspicious weight file {file_name!r}"
            )
        _require(
            file_name == WANDB_HISTORY_ARTIFACT_FILE,
            f"unexpected file in W&B history artifact {name!r}: {file_name!r}",
        )
        _require(
            isinstance(file_size, int) and not isinstance(file_size, bool) and file_size > 0,
            f"W&B history artifact file {file_name!r} has an invalid byte size: {file_size!r}",
        )
        _require(
            file_size == size,
            f"W&B history artifact/file size mismatch: artifact={size}, file={file_size}",
        )
        digest = getattr(artifact_file, "digest", None)
        ledger.append(
            {
                "name": name,
                "type": artifact_type,
                "description": description,
                "size_bytes": size,
                "files": [
                    {
                        "name": file_name,
                        "size_bytes": file_size,
                        "digest": str(digest) if digest is not None else None,
                    }
                ],
            }
        )

    return {
        "total_artifacts": len(artifacts),
        "allowed_history_artifacts": len(ledger),
        "model_artifacts": 0,
        "allowed_history_artifact_ledger": ledger,
        "history_artifact_max_size_bytes": MAX_WANDB_HISTORY_ARTIFACT_BYTES,
    }


def verify_wandb(local: VerifiedLocalRun, *, api: Any | None = None) -> dict[str, Any]:
    """Validate the finished metrics-only run via the read-only W&B public API."""

    if api is None:
        import wandb

        api = wandb.Api()
    cfg = local.config["wandb"]
    path = f"{cfg['entity']}/{cfg['project']}/{local.run_manifest['wandb_run_id']}"
    run = api.run(path)
    _require(str(run.id) == local.run_manifest["wandb_run_id"], "online W&B run id mismatch")
    _require(str(run.state).lower() == "finished", f"W&B run is not finished: {run.state}")
    artifact_report = _validate_wandb_artifacts(
        run,
        run_id=local.run_manifest["wandb_run_id"],
    )
    histories = {
        metric: _scalar_history(run, metric)
        for metric in ("train/loss", "train/lr", "train/grad_norm", "monitor/loss")
    }
    for metric in ("train/loss", "train/lr", "train/grad_norm"):
        steps = {int(row["step"]) for row in histories[metric]}
        _require(steps == EXPECTED_TRAIN_LOG_STEPS, f"W&B {metric} cadence mismatch")
    monitor_steps = {int(row["step"]) for row in histories["monitor/loss"]}
    _require(monitor_steps == EXPECTED_MONITOR_STEPS, f"W&B monitor cadence mismatch: {sorted(monitor_steps)}")
    lr_by_step = {int(row["step"]): float(row["value"]) for row in histories["train/lr"]}
    _require(
        math.isclose(lr_by_step[FINAL_F0_STEP], float(local.config["training"]["final_lr"]), rel_tol=0, abs_tol=1e-15),
        "W&B final LR differs from the final scheduler/optimizer LR",
    )
    return {
        "wandb_status": "PASS",
        "path": path,
        "state": str(run.state),
        # Retained for compatibility with earlier machine-readable reports.
        "logged_artifacts": artifact_report["total_artifacts"],
        **artifact_report,
        "expected_train_metric_points": len(EXPECTED_TRAIN_LOG_STEPS),
        "expected_monitor_points": len(EXPECTED_MONITOR_STEPS),
        "history": histories,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--deep", action="store_true", help="Strictly load the 3B final model and run one sample.")
    parser.add_argument("--device", default="cuda:0", help="Device used only with --deep.")
    parser.add_argument("--check-wandb", action="store_true", help="Add read-only W&B API verification.")
    parser.add_argument(
        "--verify-file-hashes",
        action="store_true",
        help="Hash final model files; local default validates recorded sizes and safetensors headers only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    local = verify_local_run(args.run_dir, verify_file_hashes=args.verify_file_hashes)
    report: dict[str, Any] = {"schema_version": 1, **local.report}
    if args.deep:
        report["deep"] = verify_deep(local, device=args.device)
    if args.check_wandb:
        report["wandb"] = verify_wandb(local)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
