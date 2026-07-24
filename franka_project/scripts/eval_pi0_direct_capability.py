#!/usr/bin/env python
"""Evaluate a compatible Cartesian PI0 checkpoint with the M0 metric family.

Unlike the F0/M0 audit tooling, this command does not inspect a training run,
monitor loss, W&B, checkpoint cadence, or optimizer state.  It directly asks a
checkpoint to predict a raw ``[50, 7]`` Cartesian action chunk from the two
camera images, current 10-D EEF proprioception, and language instruction.  The
demonstration action is removed before preprocessing and is used only after
inference to score the prediction.

The dataset profile, task, sampling rates, and population size are discovered
from the dataset metadata and checked against checkpoint-local manifests before
the model is loaded.  The metric scope remains intentionally limited to the F0
section 6.2 translation, rotation-geodesic, and gripper ADE/MAE, FDE, and
fixed-horizon metrics.  The aggregation is episode-macro followed by fixed-seed
mean and population variance.  No comparison baseline or alternative metric
family is computed.

The evaluation is offline, teacher-forced, and in-sample.  It is not a
closed-loop rollout or a real-robot success measurement.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from franka_eef_pipeline.direct_capability_eval import (  # noqa: E402
    fixed_noise_batch,
    split_supervision,
    summarize_m0_section_6_2,
)
from franka_eef_pipeline.dual_rate_dataset import (  # noqa: E402
    CartesianAnchorDataset,
    load_cartesian_profile,
)
from franka_eef_pipeline.pi0_training import (  # noqa: E402
    load_pi0_full_policy_and_processors,
    prepare_effective_pi0_stats,
)


SCHEMA_VERSION = 3
ACTION_CHUNK_SIZE = 50
ACTION_DIM = 7
STATE_DIM = 10
MAX_ACTION_DIM = 32
EXPECTED_NUM_INFERENCE_STEPS = 10
CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
)
DEFAULT_PREDICTION_SEEDS = (1_001_000, 2_001_000, 3_001_000)
EXPECTED_STATE_CONVENTION = (
    "current measured EEF xyz + rotation6d(first two columns) + gripper_0_1"
)
EXPECTED_ACTION_CONVENTION = {
    "translation": "base-frame target_xyz - current_xyz",
    "rotation": "body rotvec Log(R_current.T @ R_target)",
    "gripper": "future measured target gripper_0_1",
}
REQUIRED_DATASET_IDENTITY_FIELDS = (
    "schema_version",
    "profile",
    "dataset_name",
    "repo_id",
    "observation_fps",
    "action_fps",
    "chunk_size",
    "episode_count",
    "task_instruction",
    "main_action_key",
    "requires_project_cartesian_adapter",
    "requires_project_dual_rate_adapter",
    "partial_conversion",
    "source_dataset_hash",
    "logical_anchor_index_sha256",
    "scope_content_sha256",
    "derived_manifest_sha256",
    "conversion_config_sha256",
    "conversion_script_sha256",
)
OPTIONAL_DATASET_IDENTITY_FIELDS = (
    "common_contract_sha256",
    "target_grid",
    "max_camera_interval_ms",
    "scope_name",
)
STATS_IDENTITY_FIELDS = (
    "schema_version",
    "profile",
    "observation_fps",
    "action_fps",
    "chunk_size",
    "episode_count",
    "partial_conversion",
    "source_dataset_hash",
    "logical_anchor_index_sha256",
    "scope_content_sha256",
    "derived_manifest_sha256",
    "conversion_config_sha256",
    "conversion_script_sha256",
)


class DirectEvaluationError(RuntimeError):
    """Raised when direct evaluation violates its no-leakage contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectEvaluationError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot JSON-serialize {type(value).__name__}")


def _atomic_write_text(path: Path, payload: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _atomic_write_json(path: Path, report: Mapping[str, Any]) -> None:
    payload = json.dumps(
        _jsonable(report),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _atomic_write_text(path, payload)


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz"
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectEvaluationError(f"cannot read {label}: {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _scalar_count(value: Any, *, label: str) -> int:
    array = np.asarray(value)
    _require(array.size == 1, f"{label} must contain one scalar")
    scalar = array.reshape(-1)[0]
    try:
        result = int(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DirectEvaluationError(f"{label} must be an integer") from exc
    _require(result == scalar and result > 0, f"{label} must be a positive integer")
    return result


def _effective_stats_report(value: Mapping[str, Any] | Path, *, label: str) -> dict[str, Any]:
    try:
        _, report = prepare_effective_pi0_stats(value)
    except Exception as exc:
        raise DirectEvaluationError(f"invalid {label}: {exc}") from exc
    _require(isinstance(report, dict), f"{label} validation did not return a report")
    return report


def _collect_embedded_stats_reports(value: Any) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("effective_stats_sha256"), str):
            reports.append(dict(value))
        for child in value.values():
            reports.extend(_collect_embedded_stats_reports(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            reports.extend(_collect_embedded_stats_reports(child))
    return reports


def _verify_checkpoint_ledger_file(
    checkpoint_manifest: Mapping[str, Any],
    checkpoint_dir: Path,
    filename: str,
    *,
    verify_sha256: bool,
) -> dict[str, Any]:
    files = checkpoint_manifest.get("files")
    _require(isinstance(files, Mapping), "checkpoint manifest lacks a files ledger")
    entry = files.get(filename)
    _require(isinstance(entry, Mapping), f"checkpoint manifest lacks ledger entry {filename!r}")
    path = checkpoint_dir / filename
    _require(path.is_file(), f"checkpoint file is missing: {path}")
    actual_size = path.stat().st_size
    _require(
        entry.get("size_bytes") == actual_size,
        f"checkpoint ledger size mismatch for {filename}",
    )
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": actual_size,
        "size_verified": True,
        "sha256_verified": False,
    }
    if verify_sha256:
        expected_sha256 = entry.get("sha256")
        _require(
            isinstance(expected_sha256, str) and len(expected_sha256) == 64,
            f"checkpoint ledger has invalid sha256 for {filename}",
        )
        actual_sha256 = _sha256_file(path)
        _require(
            actual_sha256 == expected_sha256,
            f"checkpoint ledger sha256 mismatch for {filename}",
        )
        result.update({"sha256": actual_sha256, "sha256_verified": True})
    return result


def _require_same_field(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> None:
    _require(field in expected, f"dataset profile lacks required identity field {field!r}")
    _require(field in actual, f"{label} lacks required identity field {field!r}")
    _require(
        actual[field] == expected[field],
        f"{label} mismatch for {field}: expected {expected[field]!r}, got {actual[field]!r}",
    )


def _validate_checkpoint_dataset_contract(
    *,
    checkpoint_dir: Path,
    dataset_root: Path,
    dataset: CartesianAnchorDataset,
    dataset_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the checkpoint was trained on this exact dataset contract."""

    task_instruction = dataset_profile.get("task_instruction")
    _require(
        isinstance(task_instruction, str) and bool(task_instruction.strip()),
        "dataset profile requires a non-empty task_instruction",
    )
    _require(dataset.task_instruction == task_instruction, "dataset task_instruction mismatch")
    for field, actual in (
        ("profile", dataset.profile),
        ("observation_fps", dataset.observation_fps),
        ("action_fps", dataset.action_fps),
        ("chunk_size", dataset.chunk_size),
    ):
        _require_same_field(dataset_profile, {field: actual}, field, label="loaded dataset")
    _require(dataset.chunk_size == ACTION_CHUNK_SIZE, "M0 evaluator requires 50-waypoint chunks")
    _require(dataset.action_fps > 0, "dataset action_fps must be positive")
    _require(dataset_profile.get("partial_conversion") is False, "partial datasets are not allowed")
    _require(dataset.effective_stats is not None, "dataset lacks effective PI0 normalization stats")
    _require(
        len(dataset.logical_anchors) == len(dataset),
        "dataset logical-anchor count does not match dataset length",
    )
    episode_indices = {int(anchor["episode_index"]) for anchor in dataset.logical_anchors}
    expected_episodes = int(dataset_profile.get("episode_count", -1))
    _require(expected_episodes > 0, "dataset profile episode_count must be positive")
    _require(
        len(episode_indices) == expected_episodes == dataset.num_episodes,
        "dataset episode count does not match profile metadata",
    )

    dataset_stats_path = dataset_root / "meta/pi0_eef_stats.json"
    dataset_stats_payload = _read_json_object(dataset_stats_path, label="dataset PI0 stats")
    for field in STATS_IDENTITY_FIELDS:
        _require_same_field(
            dataset_profile,
            dataset_stats_payload,
            field,
            label="dataset PI0 stats",
        )
    loaded_stats_report = _effective_stats_report(
        dataset.effective_stats,
        label="loaded dataset effective stats",
    )
    dataset_stats_report = _effective_stats_report(
        dataset_stats_payload,
        label="dataset PI0 stats",
    )
    effective_stats_sha256 = str(dataset_stats_report["effective_stats_sha256"])
    _require(
        loaded_stats_report.get("effective_stats_sha256") == effective_stats_sha256,
        "loaded dataset effective stats do not match dataset PI0 stats",
    )
    stats_counts = dataset_stats_report.get("counts")
    _require(isinstance(stats_counts, Mapping), "dataset PI0 stats report lacks counts")
    _require(
        stats_counts.get("observation.state") == len(dataset),
        "observation.state stats count does not match anchor count",
    )
    _require(
        stats_counts.get("action") == len(dataset) * ACTION_CHUNK_SIZE,
        "action stats count does not match anchor_count * chunk_size",
    )
    for feature, expected_count in (
        ("observation.state", len(dataset)),
        ("action", len(dataset) * ACTION_CHUNK_SIZE),
    ):
        feature_stats = dataset_stats_payload.get(feature)
        _require(isinstance(feature_stats, Mapping), f"dataset PI0 stats lack {feature}")
        _require(
            _scalar_count(feature_stats.get("count"), label=f"{feature}.count")
            == expected_count,
            f"{feature}.count does not match the evaluated population",
        )

    checkpoint_manifest_path = checkpoint_dir / "franka_pi0_checkpoint_manifest.json"
    checkpoint_manifest = _read_json_object(
        checkpoint_manifest_path,
        label="checkpoint manifest",
    )
    allowed_types = {
        "franka_pi0_full_parameter_eef",
        "franka_pi0_configurable_parameter_eef"
    }
    _require(
        checkpoint_manifest.get("checkpoint_type") in allowed_types,
        "checkpoint is not a Franka full-parameter EEF checkpoint",
    )
    ledger = {
        filename: _verify_checkpoint_ledger_file(
            checkpoint_manifest,
            checkpoint_dir,
            filename,
            verify_sha256=filename != "model.safetensors",
        )
        for filename in (
            "config.json",
            "franka_eef_geometry_manifest.json",
            "pi0_eef_stats.json",
            "model.safetensors",
        )
    }

    geometry_path = checkpoint_dir / "franka_eef_geometry_manifest.json"
    geometry = _read_json_object(geometry_path, label="checkpoint geometry manifest")
    geometry_profile = geometry.get("dataset_profile")
    _require(
        isinstance(geometry_profile, Mapping),
        "checkpoint geometry manifest lacks dataset_profile",
    )
    for field in REQUIRED_DATASET_IDENTITY_FIELDS:
        _require_same_field(
            dataset_profile,
            geometry_profile,
            field,
            label="checkpoint dataset profile",
        )
    for field in OPTIONAL_DATASET_IDENTITY_FIELDS:
        if field in dataset_profile or field in geometry_profile:
            _require_same_field(
                dataset_profile,
                geometry_profile,
                field,
                label="checkpoint dataset profile",
            )
    _require(
        geometry.get("task_instruction") == task_instruction,
        "checkpoint geometry task_instruction mismatch",
    )
    _require(
        geometry.get("state10") == EXPECTED_STATE_CONVENTION,
        "checkpoint state10 geometry convention mismatch",
    )
    action7 = geometry.get("action7")
    _require(isinstance(action7, Mapping), "checkpoint geometry manifest lacks action7")
    _require(
        action7.get("frequency_hz") == dataset.action_fps,
        "checkpoint action7 frequency_hz mismatch",
    )
    _require(
        action7.get("chunk_size") == ACTION_CHUNK_SIZE,
        "checkpoint action7 chunk_size mismatch",
    )
    for field, expected in EXPECTED_ACTION_CONVENTION.items():
        _require(
            action7.get(field) == expected,
            f"checkpoint action7 {field} convention mismatch",
        )

    config_path = checkpoint_dir / "config.json"
    config = _read_json_object(config_path, label="checkpoint PI0 config")
    _require(config.get("type") == "pi0", "checkpoint config type must be 'pi0'")
    _require(
        config.get("chunk_size") == ACTION_CHUNK_SIZE
        and config.get("n_action_steps") == ACTION_CHUNK_SIZE,
        "checkpoint config chunk_size/n_action_steps mismatch",
    )
    _require(
        config.get("max_action_dim") == MAX_ACTION_DIM,
        f"checkpoint max_action_dim must be {MAX_ACTION_DIM}",
    )
    _require(
        config.get("num_inference_steps") == EXPECTED_NUM_INFERENCE_STEPS,
        "checkpoint num_inference_steps does not match the M0 protocol",
    )
    input_features = config.get("input_features")
    output_features = config.get("output_features")
    _require(isinstance(input_features, Mapping), "checkpoint config lacks input_features")
    _require(isinstance(output_features, Mapping), "checkpoint config lacks output_features")
    state_feature = input_features.get("observation.state")
    action_feature = output_features.get("action")
    _require(
        isinstance(state_feature, Mapping) and state_feature.get("shape") == [STATE_DIM],
        f"checkpoint state feature must have shape [{STATE_DIM}]",
    )
    _require(
        isinstance(action_feature, Mapping) and action_feature.get("shape") == [ACTION_DIM],
        f"checkpoint action feature must have shape [{ACTION_DIM}]",
    )
    for camera_key in CAMERA_KEYS:
        _require(camera_key in input_features, f"checkpoint config lacks camera {camera_key!r}")

    checkpoint_stats_path = checkpoint_dir / "pi0_eef_stats.json"
    checkpoint_stats_payload = _read_json_object(
        checkpoint_stats_path,
        label="checkpoint PI0 stats",
    )
    for field in STATS_IDENTITY_FIELDS:
        _require_same_field(
            dataset_profile,
            checkpoint_stats_payload,
            field,
            label="checkpoint PI0 stats",
        )
    checkpoint_stats_report = _effective_stats_report(
        checkpoint_stats_payload,
        label="checkpoint PI0 stats",
    )
    _require(
        checkpoint_stats_report.get("effective_stats_sha256") == effective_stats_sha256,
        "checkpoint and dataset effective normalization stats differ",
    )
    _require(
        checkpoint_stats_report.get("counts") == stats_counts,
        "checkpoint and dataset effective-stats counts differ",
    )

    embedded_reports = _collect_embedded_stats_reports(
        checkpoint_manifest.get("source_load_report")
    )
    matching_embedded = [
        report
        for report in embedded_reports
        if report.get("effective_stats_sha256") == effective_stats_sha256
    ]
    if embedded_reports:
        _require(
            bool(matching_embedded),
            "checkpoint source-load report effective stats do not match the dataset",
        )
        for report in matching_embedded:
            counts = report.get("counts")
            if counts is not None:
                _require(
                    counts == stats_counts,
                    "checkpoint source-load report stats counts mismatch",
                )

    return {
        "verified": True,
        "verification_timing": "before_model_load",
        "dataset_profile": str(dataset.profile),
        "task_instruction": task_instruction,
        "observation_hz": int(dataset.observation_fps),
        "action_hz": int(dataset.action_fps),
        "action_chunk_size": int(dataset.chunk_size),
        "anchor_count": len(dataset),
        "episode_count": dataset.num_episodes,
        "identity_fields_verified": list(REQUIRED_DATASET_IDENTITY_FIELDS),
        "optional_identity_fields_verified": [
            field
            for field in OPTIONAL_DATASET_IDENTITY_FIELDS
            if field in dataset_profile or field in geometry_profile
        ],
        "normalization": {
            "effective_stats_sha256": effective_stats_sha256,
            "counts": dict(stats_counts),
            "checkpoint_matches_dataset": True,
            "checkpoint_stats_file_sha256": _sha256_file(checkpoint_stats_path),
            "dataset_stats_file_sha256": _sha256_file(dataset_stats_path),
            "stats_files_byte_identical": (
                checkpoint_stats_path.read_bytes() == dataset_stats_path.read_bytes()
            ),
            "source_load_report_match": bool(matching_embedded),
        },
        "checkpoint_model_contract": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "chunk_size": ACTION_CHUNK_SIZE,
            "max_action_dim": MAX_ACTION_DIM,
            "num_inference_steps": EXPECTED_NUM_INFERENCE_STEPS,
        },
        "checkpoint_ledger": ledger,
    }


def _resolve_checkpoint_dir(value: str | Path) -> Path:
    """Accept either a model directory or a trainer ``step-*`` directory."""

    candidate = Path(value).expanduser().resolve()
    direct = candidate / "config.json"
    direct_weights = candidate / "model.safetensors"
    nested = candidate / "pretrained_model"
    if direct.is_file() and direct_weights.is_file():
        return candidate
    if (nested / "config.json").is_file() and (nested / "model.safetensors").is_file():
        return nested.resolve()
    raise FileNotFoundError(
        "checkpoint must contain config.json/model.safetensors directly or in "
        f"pretrained_model/: {candidate}"
    )


def _validate_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid --device: {value!r}") from exc
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        _require(0 <= index < torch.cuda.device_count(), f"CUDA device index is unavailable: {index}")
        return torch.device("cuda", index)
    _require(device.type == "cpu", "--device must be cpu or cuda[:index]")
    return device


@contextmanager
def _offline_loading() -> Any:
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _convert_images_to_float01(inputs: dict[str, Any]) -> None:
    for key in CAMERA_KEYS:
        image = inputs.get(key)
        _require(isinstance(image, torch.Tensor), f"model input lacks tensor camera {key!r}")
        _require(image.dtype is torch.uint8, f"{key} must be uint8 before explicit conversion")
        converted = image.to(torch.float32).div_(255.0)
        _require(bool(torch.isfinite(converted).all()), f"{key} contains NaN/Inf")
        _require(
            float(converted.min()) >= 0.0 and float(converted.max()) <= 1.0,
            f"{key} is outside [0,1] after conversion",
        )
        inputs[key] = converted


def _update_tensor_digest(digest: Any, key: str, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    digest.update(key.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))


def _update_input_digest(
    digest: Any,
    inputs: Mapping[str, Any],
    logical_indices: Sequence[int],
) -> None:
    digest.update(np.asarray(logical_indices, dtype=np.int64).tobytes())
    for key in ("observation.state", *CAMERA_KEYS):
        value = inputs.get(key)
        _require(isinstance(value, torch.Tensor), f"input digest lacks {key}")
        _update_tensor_digest(digest, key, value)
    tasks = inputs.get("task")
    _require(isinstance(tasks, (list, tuple)), "model input must contain collated language tasks")
    for task in tasks:
        _require(isinstance(task, str), "language task must be a string")
        encoded = task.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)


def _validate_language(
    inputs: Mapping[str, Any],
    batch_size: int,
    expected_task: str,
) -> None:
    tasks = inputs.get("task")
    _require(isinstance(tasks, (list, tuple)) and len(tasks) == batch_size, "invalid task batch")
    _require(
        all(task == expected_task for task in tasks),
        f"dataset language instruction drifted from {expected_task!r}: {tasks}",
    )


def _strip_empty_supervision_placeholders(processed: dict[str, Any]) -> None:
    """Remove canonical ``None`` placeholders while rejecting real supervision."""

    action = processed.pop("action", None)
    action_is_pad = processed.pop("action_is_pad", None)
    _require(action is None, "preprocessor reintroduced target action")
    _require(action_is_pad is None, "preprocessor reintroduced target pad mask")


def _strict_load_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    pretrained = report.get("pretrained")
    effective = report.get("effective_stats")
    parameters = report.get("parameters")
    _require(isinstance(pretrained, Mapping), "loader omitted pretrained report")
    _require(isinstance(effective, Mapping), "loader omitted effective-stats report")
    _require(isinstance(parameters, Mapping), "loader omitted parameter report")
    _require(pretrained.get("strict") is True, "checkpoint was not loaded strictly")
    _require(pretrained.get("missing_keys") == [], "strict load reported missing keys")
    _require(pretrained.get("unexpected_keys") == [], "strict load reported unexpected keys")
    weights_path = Path(str(pretrained.get("weights_path"))).resolve()
    _require(weights_path.is_file(), f"loaded weights path is missing: {weights_path}")
    return {
        "loader": "franka_eef_pipeline.pi0_training.load_pi0_full_policy_and_processors",
        "strict": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "weights_path": str(weights_path),
        "weights_size_bytes": weights_path.stat().st_size,
        "project_manifest_present": bool(pretrained.get("project_manifest_present")),
        "verified_tensor": pretrained.get("verified_tensor"),
        "effective_stats_sha256": effective.get("effective_stats_sha256"),
        "full_parameter_contract": {
            "use_peft": report.get("config", {}).get("use_peft")
            if isinstance(report.get("config"), Mapping)
            else None,
            "trainable_parameters": parameters.get("trainable_parameters"),
            "parameter_tensor_count": parameters.get("parameter_tensor_count"),
        },
    }


def _dataset_identity(
    dataset: CartesianAnchorDataset,
    root: Path,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    logical = dataset.logical_anchors
    episode_ids = sorted({int(anchor["episode_index"]) for anchor in logical})
    raw_episode_ids = sorted({str(anchor["raw_episode_id"]) for anchor in logical})
    logical_indices = [int(anchor["logical_index"]) for anchor in logical]
    return {
        "root": str(root),
        "profile": dataset.profile,
        "task_instruction": dataset.task_instruction,
        "observation_hz": dataset.observation_fps,
        "action_hz": dataset.action_fps,
        "action_chunk_size": dataset.chunk_size,
        "selected_anchor_count": len(dataset),
        "selected_episode_count": len(episode_ids),
        "episode_indices": episode_ids,
        "raw_episode_ids": raw_episode_ids,
        "logical_index_sha256": hashlib.sha256(
            np.asarray(logical_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "source_dataset_hash": profile.get("source_dataset_hash"),
        "logical_anchor_index_sha256": profile.get("logical_anchor_index_sha256"),
        "scope_content_sha256": profile.get("scope_content_sha256"),
    }


def _infer_checkpoint(
    *,
    label: str,
    checkpoint_dir: Path,
    dataset: CartesianAnchorDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    prediction_seeds: Sequence[int],
    num_inference_steps: int,
    expected_task: str,
    expected_effective_stats_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load exactly one policy, predict, then release it before another load."""

    started = time.monotonic()
    policy = preprocessor = postprocessor = None
    try:
        with _offline_loading():
            policy, preprocessor, postprocessor, load_report = (
                load_pi0_full_policy_and_processors(
                    checkpoint_dir,
                    dataset.effective_stats,
                    device,
                )
            )
        strict_load = _strict_load_summary(load_report)
        _require(
            strict_load.get("effective_stats_sha256") == expected_effective_stats_sha256,
            "loader effective normalization stats differ from the verified dataset contract",
        )
        policy.eval()
        _require(policy.training is False, f"{label} policy.eval() did not take effect")

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        states: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        pads: list[np.ndarray] = []
        predictions: list[list[np.ndarray]] = [[] for _ in prediction_seeds]
        input_digest = hashlib.sha256()
        supervision_digest = hashlib.sha256()
        logical_indices = np.asarray(
            [int(anchor["logical_index"]) for anchor in dataset.logical_anchors],
            dtype=np.int64,
        )
        episode_indices = np.asarray(
            [int(anchor["episode_index"]) for anchor in dataset.logical_anchors],
            dtype=np.int64,
        )
        consumed = 0

        for raw_batch in loader:
            raw_target = raw_batch.get("action")
            raw_state = raw_batch.get("observation.state")
            raw_pad = raw_batch.get("action_is_pad")
            _require(isinstance(raw_target, torch.Tensor), "batch lacks target action")
            _require(isinstance(raw_state, torch.Tensor), "batch lacks current state")
            _require(isinstance(raw_pad, torch.Tensor), "batch lacks action_is_pad")

            # Freeze supervision before split_supervision removes it from the
            # model-facing dictionary.  These tensors are never passed to the
            # preprocessor or policy.
            saved_target = raw_target.detach().cpu().to(torch.float32).clone()
            saved_state = raw_state.detach().cpu().to(torch.float32).clone()
            saved_pad = raw_pad.detach().cpu().to(torch.bool).clone()
            current_count = int(saved_target.shape[0])
            current_logical = logical_indices[consumed : consumed + current_count]
            consumed += current_count
            _require(
                tuple(saved_target.shape) == (current_count, ACTION_CHUNK_SIZE, ACTION_DIM),
                f"target action shape drift: {tuple(saved_target.shape)}",
            )
            _require(
                tuple(saved_state.shape) == (current_count, STATE_DIM),
                f"state shape drift: {tuple(saved_state.shape)}",
            )
            _require(
                tuple(saved_pad.shape) == (current_count, ACTION_CHUNK_SIZE),
                f"pad shape drift: {tuple(saved_pad.shape)}",
            )
            _require(
                bool(torch.isfinite(saved_target).all())
                and bool(torch.isfinite(saved_state).all()),
                "supervision/state contains NaN/Inf",
            )

            model_inputs, helper_target, helper_pad = split_supervision(raw_batch)
            _require(isinstance(model_inputs, dict), "split_supervision must return a dict")
            _require("action" not in model_inputs, "target action leaked into model inputs")
            _require("action_is_pad" not in model_inputs, "target pad mask leaked into model inputs")
            _require(torch.equal(helper_target.detach().cpu(), saved_target), "split target changed values")
            _require(torch.equal(helper_pad.detach().cpu(), saved_pad), "split pad mask changed values")
            _validate_language(model_inputs, current_count, expected_task)
            _update_input_digest(input_digest, model_inputs, current_logical.tolist())
            _update_tensor_digest(supervision_digest, "state", saved_state)
            _update_tensor_digest(supervision_digest, "target", saved_target)
            _update_tensor_digest(supervision_digest, "pad", saved_pad)

            _convert_images_to_float01(model_inputs)
            processed = preprocessor(copy.copy(model_inputs))
            _require(isinstance(processed, dict), "PI0 preprocessor did not return a dict")
            # PolicyProcessorPipeline's canonical transition-to-batch converter
            # always emits ``action``.  Target-free inference therefore returns
            # ``action=None`` rather than omitting the key.  Require the value to
            # remain empty, then remove both supervision keys before policy use.
            _strip_empty_supervision_placeholders(processed)

            for seed_ordinal, seed in enumerate(prediction_seeds):
                noise = fixed_noise_batch(
                    int(seed),
                    current_logical.tolist(),
                    shape=(ACTION_CHUNK_SIZE, MAX_ACTION_DIM),
                ).to(device)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with torch.inference_mode(), autocast:
                    normalized = policy.predict_action_chunk(
                        processed,
                        noise=noise,
                        num_steps=num_inference_steps,
                    )
                _require(
                    isinstance(normalized, torch.Tensor)
                    and tuple(normalized.shape)
                    == (current_count, ACTION_CHUNK_SIZE, ACTION_DIM),
                    f"normalized prediction shape drift: {getattr(normalized, 'shape', None)}",
                )
                raw_prediction = postprocessor(normalized.detach().clone())
                _require(
                    isinstance(raw_prediction, torch.Tensor)
                    and tuple(raw_prediction.shape)
                    == (current_count, ACTION_CHUNK_SIZE, ACTION_DIM),
                    f"raw prediction shape drift: {getattr(raw_prediction, 'shape', None)}",
                )
                raw_prediction = raw_prediction.detach().cpu().to(torch.float32)
                _require(bool(torch.isfinite(raw_prediction).all()), "prediction contains NaN/Inf")
                predictions[seed_ordinal].append(raw_prediction.numpy())

            states.append(saved_state.numpy())
            targets.append(saved_target.numpy())
            pads.append(saved_pad.numpy())

        _require(consumed == len(dataset), "DataLoader did not consume every selected anchor")
        arrays = {
            "logical_indices": logical_indices,
            "episode_indices": episode_indices,
            "states": np.concatenate(states).astype(np.float32, copy=False),
            "targets": np.concatenate(targets).astype(np.float32, copy=False),
            "pads": np.concatenate(pads).astype(np.bool_, copy=False),
            "predictions": np.stack(
                [np.concatenate(parts).astype(np.float32, copy=False) for parts in predictions]
            ),
            "prediction_seeds": np.asarray(prediction_seeds, dtype=np.int64),
        }
        _require(
            arrays["predictions"].shape
            == (len(prediction_seeds), len(dataset), ACTION_CHUNK_SIZE, ACTION_DIM),
            "concatenated prediction shape mismatch",
        )
        result = {
            "label": label,
            "checkpoint_dir": str(checkpoint_dir),
            "strict_load": strict_load,
            "policy_eval_called": True,
            "target_action_provided_to_model": False,
            "input_sha256": input_digest.hexdigest(),
            "supervision_sha256": supervision_digest.hexdigest(),
            "prediction_shape": list(arrays["predictions"].shape),
            "inference_seconds": max(time.monotonic() - started, 1e-9),
        }
        return result, arrays
    finally:
        policy = preprocessor = postprocessor = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _aggregate_model_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    pads: np.ndarray,
    episode_indices: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, Any]:
    return summarize_m0_section_6_2(
        predictions,
        targets,
        pads,
        episode_indices,
        seeds,
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    mean = report["metrics"]["mean"]
    variance = report["metrics"]["population_variance"]
    dataset = report["dataset"]
    task_instruction = str(report["model_input_contract"]["language_instruction"])
    timebase = report["metric_timebase"]
    horizon_offsets = timebase["horizon_offsets_seconds"]
    lines = [
        "# PI0 M0 teacher-forced in-sample reconstruction",
        "",
        f"- Status: `{report['status']}`",
        f"- Dataset profile: `{dataset['profile']}` "
        f"({dataset['observation_hz']} Hz observations / {dataset['action_hz']} Hz actions).",
        f"- Selected anchors / episodes: `{dataset['selected_anchor_count']}` / "
        f"`{dataset['selected_episode_count']}`",
        "- Model inputs: two RGB cameras, current 10-D EEF state, and task "
        f"`{task_instruction}`.",
        "- `target_action_provided_to_model=false`; target chunks are used only after inference.",
        "- ADE/MAE includes all 50 waypoints; FDE is waypoint 50 (array index 49).",
        "- Horizon offsets in seconds (waypoints 1/5/10/25/50): "
        + "/".join(f"{float(horizon_offsets[str(horizon)]):.6g}" for horizon in (1, 5, 10, 25, 50))
        + ".",
        "- Aggregation: per episode, then episode macro, then fixed-seed mean/variance.",
        "",
        "## M0 metrics (F0 section 6.2 definitions)",
        "",
        "| component | ADE/MAE mean | FDE mean | ADE/MAE population variance | FDE population variance |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = (
        (
            "translation (mm)",
            "translation_ade_mm",
            "translation_fde_mm",
        ),
        (
            "rotation geodesic (deg)",
            "rotation_geodesic_ade_deg",
            "rotation_geodesic_fde_deg",
        ),
        ("gripper", "gripper_mae", "gripper_fde_mae"),
    )
    for label, ade_key, fde_key in rows:
        lines.append(
            f"| {label} | {float(mean[ade_key]):.8g} | {float(mean[fde_key]):.8g} | "
            f"{float(variance[ade_key]):.8g} | {float(variance[fde_key]):.8g} |"
        )

    lines.extend(
        [
            "",
            "## Fixed-horizon mean",
            "",
            "| component | 1 | 5 | 10 | 25 | 50 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    horizon_rows = (
        ("translation (mm)", "translation_horizon_mm"),
        ("rotation geodesic (deg)", "rotation_geodesic_horizon_deg"),
        ("gripper MAE", "gripper_horizon_mae"),
    )
    for label, key in horizon_rows:
        values = mean[key]
        formatted_values = " | ".join(
            f"{float(values[str(horizon)]):.8g}" for horizon in (1, 5, 10, 25, 50)
        )
        lines.append(f"| {label} | {formatted_values} |")

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This is target-hidden model inference, but it remains an offline action-reconstruction "
            "evaluation. With the current dataset it is in-sample, not a held-out generalization "
            "test. It does not execute actions, close the control loop, test recovery from errors, "
            f"or measure real-robot task success for {task_instruction!r}.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the M0 teacher-forced in-sample metric family for a compatible "
            "Cartesian PI0 checkpoint, with target actions removed from model inputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--prediction-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_PREDICTION_SEEDS),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save raw states, targets, and trained-checkpoint predictions to a sibling NPZ.",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.num_inference_steps != EXPECTED_NUM_INFERENCE_STEPS:
        parser.error(
            f"M0 evaluation requires --num-inference-steps {EXPECTED_NUM_INFERENCE_STEPS}"
        )
    if tuple(args.prediction_seeds) != DEFAULT_PREDICTION_SEEDS:
        parser.error(
            "M0 evaluation requires --prediction-seeds "
            + " ".join(str(seed) for seed in DEFAULT_PREDICTION_SEEDS)
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    device = _validate_device(args.device)
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    dataset_root = args.dataset_root.expanduser().resolve()
    _require(dataset_root.is_dir(), f"dataset root is missing: {dataset_root}")
    output_json = args.output_json.expanduser().resolve()
    _require(output_json.suffix.lower() == ".json", "--output-json must end in .json")
    output_markdown = output_json.with_suffix(".md")
    output_predictions = output_json.with_suffix(".predictions.npz")

    dataset_profile = load_cartesian_profile(dataset_root)
    profile_name = dataset_profile.get("profile")
    _require(isinstance(profile_name, str) and bool(profile_name), "dataset profile is invalid")
    dataset = CartesianAnchorDataset(
        dataset_root,
        profile=profile_name,
        video_backend="pyav",
    )
    contract_validation = _validate_checkpoint_dataset_contract(
        checkpoint_dir=checkpoint_dir,
        dataset_root=dataset_root,
        dataset=dataset,
        dataset_profile=dataset_profile,
    )
    task_instruction = str(contract_validation["task_instruction"])
    effective_stats_sha256 = str(
        contract_validation["normalization"]["effective_stats_sha256"]
    )
    logical_indices = np.asarray(
        [int(anchor["logical_index"]) for anchor in dataset.logical_anchors], dtype=np.int64
    )
    episode_indices = np.asarray(
        [int(anchor["episode_index"]) for anchor in dataset.logical_anchors], dtype=np.int64
    )

    inference_report, arrays = _infer_checkpoint(
        label="trained",
        checkpoint_dir=checkpoint_dir,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        prediction_seeds=args.prediction_seeds,
        num_inference_steps=args.num_inference_steps,
        expected_task=task_instruction,
        expected_effective_stats_sha256=effective_stats_sha256,
    )
    metrics = _aggregate_model_metrics(
        arrays["predictions"],
        arrays["targets"],
        arrays["pads"],
        arrays["episode_indices"],
        args.prediction_seeds,
    )

    prediction_artifact: dict[str, Any] | None = None
    if args.save_predictions:
        arrays_to_save: dict[str, np.ndarray] = {
            "logical_indices": logical_indices,
            "episode_indices": episode_indices,
            "prediction_seeds": np.asarray(args.prediction_seeds, dtype=np.int64),
            "states": arrays["states"],
            "targets": arrays["targets"],
            "pads": arrays["pads"],
            "predictions": arrays["predictions"],
        }
        _atomic_write_npz(output_predictions, arrays_to_save)
        prediction_artifact = {
            "path": str(output_predictions),
            "sha256": _sha256_file(output_predictions),
            "size_bytes": output_predictions.stat().st_size,
        }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "M0_SECTION_6_2_EVALUATION_COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "m0_teacher_forced_in_sample_reconstruction",
        "target_action_provided_to_model": False,
        "policy_eval_called": True,
        "in_sample": True,
        "held_out_validation_or_test": False,
        "closed_loop": False,
        "real_robot_rollout": False,
        "task_success_metric": False,
        "checkpoint_dataset_contract": contract_validation,
        "model_input_contract": {
            "camera_keys": list(CAMERA_KEYS),
            "camera_conversion": "uint8_to_float32_[0,1]_before_processor",
            "state": "current_eef_xyz_rot6d_gripper_10d",
            "language_instruction": task_instruction,
            "removed_before_preprocessor": ["action", "action_is_pad"],
            "asserted_absent_after_preprocessor": ["action", "action_is_pad"],
        },
        "inference": {
            "device": str(device),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "num_inference_steps": args.num_inference_steps,
            "prediction_seeds": list(args.prediction_seeds),
            "noise_seed_rule": "prediction_seed + logical_index",
            "raw_prediction_shape": [ACTION_CHUNK_SIZE, ACTION_DIM],
        },
        "metric_timebase": {
            "action_hz": dataset.action_fps,
            "waypoint_indexing": "horizon h uses array index h-1",
            "horizon_offsets_seconds": {
                str(horizon): (horizon - 1) / dataset.action_fps
                for horizon in (1, 5, 10, 25, 50)
            },
            "chunk_offset_span_seconds": (ACTION_CHUNK_SIZE - 1) / dataset.action_fps,
        },
        "dataset": _dataset_identity(dataset, dataset_root, dataset_profile),
        "checkpoint": inference_report,
        "metrics": metrics,
        "prediction_artifact": prediction_artifact,
        "limitations": [
            f"The supplied {dataset.num_episodes}-episode dataset is verified against the "
            "checkpoint training contract, so model metrics are in-sample.",
            "Teacher-forced current observations do not measure closed-loop drift or recovery.",
            f"No action is executed and no real-robot task-success rate for {task_instruction!r} "
            "is measured.",
            "A small offline action error does not establish safe robot behavior.",
        ],
    }
    _atomic_write_json(output_json, report)
    _atomic_write_text(output_markdown, _markdown_report(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "json": str(output_json),
                "markdown": str(output_markdown),
                "predictions": str(output_predictions) if args.save_predictions else None,
                "anchors": len(dataset),
                "episodes": dataset.num_episodes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
