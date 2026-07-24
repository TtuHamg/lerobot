#!/usr/bin/env python
"""Read-only fixed-noise Cartesian evaluator for a completed PI0 S1-15 run.

The evaluator compares the local PI0 base checkpoint with the run's final
step-300 checkpoint on the exact recorded in-train monitor anchors.  It does
not choose a quality threshold, write to W&B, upload artifacts, or modify a
checkpoint.  ``--output-json`` is the only optional filesystem write.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for import_root in (PROJECT_SRC, SCRIPTS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from franka_eef_pipeline.dual_rate_dataset import CartesianAnchorDataset  # noqa: E402
from franka_eef_pipeline.geometry import rotation_geodesic_angle, so3_exp  # noqa: E402
from franka_eef_pipeline.pi0_training import (  # noqa: E402
    PI0_CORE_WEIGHTS_NAMESPACE,
    load_pi0_full_policy_and_processors,
)
from verify_pi0_s0_run import load_saved_processors  # noqa: E402


FINAL_S1_STEP = 300
NUM_INFERENCE_STEPS = 10
ACTION_CHUNK_SIZE = 50
ACTION_DIM = 7
STATE_DIM = 10
MAX_ACTION_DIM = 32
EXPECTED_PARAMETER_COUNT = 3_238_048_528
EXPECTED_PARAMETER_TENSORS = 776
FIXED_NOISE_SEED_OFFSET = 100_000
RELOAD_RTOL = 1e-5
RELOAD_ATOL = 1e-6
CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
)


class S1EvaluationError(RuntimeError):
    """Raised when S1 artifacts or predictions violate the frozen contract."""


@dataclass(frozen=True)
class VerifiedS1Run:
    run_dir: Path
    config: dict[str, Any]
    run_manifest: dict[str, Any]
    monitor: dict[str, Any]
    dataset_root: Path
    base_model_dir: Path
    checkpoint: Path
    model_dir: Path
    checkpoint_manifest: dict[str, Any]
    model_manifest: dict[str, Any]
    stats_file_sha256: str


@dataclass(frozen=True)
class PredictionSet:
    monitor_indices: tuple[int, ...]
    normalized_actions: torch.Tensor
    raw_actions: torch.Tensor
    target_actions: torch.Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S1EvaluationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise S1EvaluationError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise S1EvaluationError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise S1EvaluationError(f"expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise S1EvaluationError(f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _require_nonempty_file(path: Path) -> None:
    _require(path.is_file() and path.stat().st_size > 0, f"missing/empty required file: {path}")


def _verify_processor_files(model_dir: Path) -> None:
    for config_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        config = _read_json(model_dir / config_name)
        steps = config.get("steps")
        _require(isinstance(steps, list) and steps, f"processor has no steps: {config_name}")
        for step in steps:
            _require(isinstance(step, dict), f"invalid processor step: {config_name}")
            state_file = step.get("state_file")
            if state_file is not None:
                _require(isinstance(state_file, str) and state_file, "invalid processor state file")
                _require_nonempty_file(model_dir / state_file)


def verify_s1_run(run_dir: str | Path) -> VerifiedS1Run:
    """Fail closed unless ``run_dir`` is a complete local S1-15 step-300 run."""

    root = Path(run_dir).expanduser().resolve()
    _require(root.is_dir(), f"run directory does not exist: {root}")
    _require("_ABORTED_" not in root.name, "refusing an aborted run directory")
    _require(not (root / "ABORTED.json").exists(), "refusing a run with ABORTED.json")
    staging = sorted((root / "checkpoints").glob(".*.staging"))
    _require(not staging, f"refusing run with checkpoint staging directories: {staging}")

    config_path = root / "resolved_config.yaml"
    _require_nonempty_file(config_path)
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise S1EvaluationError(f"invalid resolved config: {config_path}") from exc
    _require(isinstance(config, dict), "resolved config root must be a mapping")
    _require(config.get("stage") == "S1-15", "evaluator accepts only S1-15 runs")

    dataset_cfg = _required_mapping(config.get("dataset"), "config.dataset")
    expected_dataset = (
        dataset_cfg.get("profile"),
        dataset_cfg.get("observation_fps"),
        dataset_cfg.get("action_fps"),
        dataset_cfg.get("chunk_size"),
    )
    _require(
        expected_dataset == ("action15", 15, 15, ACTION_CHUNK_SIZE),
        f"S1 must use action15/obs15/action15/chunk50, got {expected_dataset}",
    )
    training_cfg = _required_mapping(config.get("training"), "config.training")
    _require(training_cfg.get("steps") == FINAL_S1_STEP, "S1 training.steps must equal 300")
    seed = training_cfg.get("seed")
    _require(type(seed) is int and seed >= 0, "S1 training.seed must be a non-negative integer")
    model_cfg = _required_mapping(config.get("model"), "config.model")
    _require(model_cfg.get("strict") is True, "S1 model loading must be strict")
    _require(model_cfg.get("max_action_dim") == MAX_ACTION_DIM, "S1 max_action_dim must equal 32")
    _require(model_cfg.get("peft") is None, "S1 evaluator refuses PEFT/LoRA runs")

    run_manifest = _read_json(root / "run_manifest.json")
    _require(run_manifest.get("config_sha256") == _sha256_file(config_path), "run/config SHA mismatch")
    _require(run_manifest.get("stage") == "S1-15", "run manifest stage mismatch")
    _require(run_manifest.get("planned_total_steps") == FINAL_S1_STEP, "planned step mismatch")
    dataset_root = Path(str(dataset_cfg.get("root", ""))).expanduser().resolve()
    _require(str(dataset_root) == run_manifest.get("dataset_root"), "dataset root drift")
    _require(dataset_root.is_dir(), f"dataset root is missing: {dataset_root}")
    profile_path = dataset_root / "meta/franka_eef_profile.json"
    _require(
        _sha256_file(profile_path) == run_manifest.get("dataset_profile_sha256"),
        "dataset profile SHA mismatch",
    )
    dataset_size = run_manifest.get("dataset_size")
    _require(type(dataset_size) is int and dataset_size > 0, "invalid run dataset_size")

    full_report = _required_mapping(run_manifest.get("full_parameter_report"), "full_parameter_report")
    expected_parameter_fields = {
        "total_parameters": EXPECTED_PARAMETER_COUNT,
        "trainable_parameters": EXPECTED_PARAMETER_COUNT,
        "parameter_tensor_count": EXPECTED_PARAMETER_TENSORS,
        "trainable_fraction": 1.0,
        "lora_parameter_count": 0,
        "use_peft": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": full_report.get(key)}
        for key, expected in expected_parameter_fields.items()
        if full_report.get(key) != expected
    }
    _require(not mismatches, f"run is not the frozen full-parameter PI0 graph: {mismatches}")

    monitor = _read_json(root / "monitor_subset.json")
    monitor_payload = dict(monitor)
    monitor_hash = monitor_payload.pop("subset_sha256", None)
    _require(monitor_hash == _canonical_sha256(monitor_payload), "monitor canonical SHA mismatch")
    _require(monitor_hash == run_manifest.get("monitor_subset_sha256"), "monitor/run SHA mismatch")
    _require(
        monitor.get("source") == "training_data" and monitor.get("held_out") is False,
        "S1 monitor must be the recorded in-train subset",
    )
    indices = monitor.get("indices")
    anchors = monitor.get("anchors")
    _require(
        isinstance(indices, list)
        and indices
        and all(type(index) is int for index in indices)
        and len(indices) == len(set(indices)),
        "monitor indices must be non-empty unique integers",
    )
    _require(
        all(0 <= index < dataset_size for index in indices),
        "monitor index points outside the S1 dataset",
    )
    _require(
        isinstance(anchors, list)
        and len(anchors) == len(indices)
        and all(isinstance(anchor, dict) for anchor in anchors),
        "monitor anchors do not align with monitor indices",
    )
    _require(monitor.get("sample_size") == len(indices), "monitor sample_size mismatch")
    _require(monitor.get("population_size") == dataset_size, "monitor population_size mismatch")

    pointer = _read_json(root / "checkpoints/last_checkpoint.json")
    _require(pointer.get("step") == FINAL_S1_STEP, "latest S1 checkpoint is not step 300")
    checkpoint = (root / str(pointer.get("path", ""))).resolve()
    _require(checkpoint.is_relative_to(root), "checkpoint pointer escapes the run directory")
    _require(checkpoint.name == "step-000300" and checkpoint.is_dir(), "step-300 checkpoint missing")
    checkpoint_manifest = _read_json(checkpoint / "checkpoint_manifest.json")
    _require(checkpoint_manifest.get("step") == FINAL_S1_STEP, "checkpoint manifest step mismatch")
    _require(checkpoint_manifest.get("atomic_publish") is True, "checkpoint was not atomically published")
    for key in ("config_sha256", "monitor_subset_sha256", "wandb_run_id", "world_size"):
        _require(checkpoint_manifest.get(key) == run_manifest.get(key), f"checkpoint/run {key} mismatch")

    training_step = _read_json(checkpoint / "training_state/training_step.json")
    _require(training_step.get("step") == FINAL_S1_STEP, "training state is not at step 300")
    invocations = [_read_json(path) for path in sorted((root / "invocations").glob("*.json"))]
    _require(
        any(
            record.get("last_step") == FINAL_S1_STEP
            and record.get("requested_stop_step") == FINAL_S1_STEP
            and record.get("exit_code") == 0
            for record in invocations
        ),
        "no successful invocation completed S1 step 300",
    )

    model_dir = checkpoint / "pretrained_model"
    model_manifest = _read_json(model_dir / "franka_pi0_checkpoint_manifest.json")
    _require(
        model_manifest.get("checkpoint_type") == "franka_pi0_full_parameter_eef",
        "unexpected project checkpoint type",
    )
    _require(
        model_manifest.get("weights_namespace") == PI0_CORE_WEIGHTS_NAMESPACE,
        "unexpected project checkpoint namespace",
    )
    _require(model_manifest.get("parameter_training") == full_report, "model/run parameter report drift")
    _require(
        model_manifest.get("training_graph") == full_report.get("training_graph"),
        "model/run training graph drift",
    )
    files = _required_mapping(model_manifest.get("files"), "model manifest files")
    for filename in (
        "config.json",
        "model.safetensors",
        "pi0_eef_stats.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        record = _required_mapping(files.get(filename), f"model file record {filename}")
        path = model_dir / filename
        _require_nonempty_file(path)
        _require(path.stat().st_size == record.get("size_bytes"), f"model file size mismatch: {path}")
        _require(_sha256_file(path) == record.get("sha256"), f"model file SHA256 mismatch: {path}")
    _verify_processor_files(model_dir)

    live_stats_path = dataset_root / "meta/pi0_eef_stats.json"
    saved_stats_path = model_dir / "pi0_eef_stats.json"
    live_stats_sha = _sha256_file(live_stats_path)
    _require(live_stats_sha == _sha256_file(saved_stats_path), "saved/live action15 stats differ")

    base_model_dir = Path(str(model_cfg.get("pretrained_path", ""))).expanduser().resolve()
    _require(base_model_dir.is_dir(), f"local PI0 base checkpoint is missing: {base_model_dir}")
    return VerifiedS1Run(
        run_dir=root,
        config=config,
        run_manifest=run_manifest,
        monitor=monitor,
        dataset_root=dataset_root,
        base_model_dir=base_model_dir,
        checkpoint=checkpoint,
        model_dir=model_dir,
        checkpoint_manifest=checkpoint_manifest,
        model_manifest=model_manifest,
        stats_file_sha256=live_stats_sha,
    )


def _validate_device(device: str | torch.device) -> torch.device:
    value = torch.device(device)
    _require(value.type in {"cpu", "cuda"}, f"unsupported evaluation device: {value}")
    if value.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        _require(value.index is not None, "use an explicit CUDA device such as cuda:0")
        _require(value.index < torch.cuda.device_count(), f"CUDA device does not exist: {value}")
        torch.cuda.set_device(value)
        _require(torch.cuda.is_bf16_supported(), f"CUDA device lacks BF16 support: {value}")
    return value


@contextmanager
def _offline_model_loading() -> Any:
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


def _convert_images_to_float01(batch: dict[str, Any]) -> None:
    for key in CAMERA_KEYS:
        image = batch.get(key)
        _require(isinstance(image, torch.Tensor), f"camera {key} is not a tensor")
        _require(image.dtype is torch.uint8, f"camera {key} must be raw uint8")
        converted = image.to(torch.float32).div_(255.0)
        _require(converted.numel() > 0 and bool(torch.isfinite(converted).all()), f"camera {key} non-finite")
        _require(
            float(converted.min()) >= 0.0 and float(converted.max()) <= 1.0,
            f"camera {key} lies outside [0,1]",
        )
        batch[key] = converted


def _require_action_tensor(value: Any, name: str, *, batch_size: int) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} is not a tensor")
    expected = (batch_size, ACTION_CHUNK_SIZE, ACTION_DIM)
    _require(tuple(value.shape) == expected, f"{name} shape must be {expected}, got {tuple(value.shape)}")
    _require(value.is_floating_point(), f"{name} must be floating point")
    _require(bool(torch.isfinite(value).all()), f"{name} contains NaN/Inf")
    return value


def fixed_noise_cpu(training_seed: int, monitor_index: int) -> tuple[int, torch.Tensor, str]:
    """Create traversal-order-independent PI0 noise for one monitor anchor."""

    _require(type(training_seed) is int and training_seed >= 0, "training seed must be non-negative")
    _require(type(monitor_index) is int and monitor_index >= 0, "monitor index must be non-negative")
    noise_seed = training_seed + FIXED_NOISE_SEED_OFFSET + monitor_index
    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    noise = torch.randn(
        (1, ACTION_CHUNK_SIZE, MAX_ACTION_DIM),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    _require(bool(torch.isfinite(noise).all()), "fixed noise contains NaN/Inf")
    digest = hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest()
    return noise_seed, noise, digest


def _load_dataset_and_samples(
    verified: VerifiedS1Run,
) -> tuple[CartesianAnchorDataset, dict[int, dict[str, Any]]]:
    dataset_cfg = verified.config["dataset"]
    dataset = CartesianAnchorDataset(
        verified.dataset_root,
        profile=dataset_cfg["profile"],
        episode_indices=dataset_cfg.get("episode_indices"),
        max_anchors_per_episode=dataset_cfg.get("max_anchors_per_episode"),
        video_backend="pyav",
    )
    _require(len(dataset) == verified.run_manifest["dataset_size"], "reloaded dataset size mismatch")
    _require(
        (dataset.profile, dataset.observation_fps, dataset.action_fps, dataset.chunk_size)
        == ("action15", 15, 15, ACTION_CHUNK_SIZE),
        "reloaded dataset profile mismatch",
    )
    _require(
        isinstance(dataset.effective_stats, Mapping)
        and set(dataset.effective_stats) == {"observation.state", "action"},
        "dataset lacks the frozen state10/action7 effective stats",
    )

    indices = tuple(int(index) for index in verified.monitor["indices"])
    expected_anchors = verified.monitor["anchors"]
    samples: dict[int, dict[str, Any]] = {}
    for ordinal, index in enumerate(indices):
        actual_anchor = dataset.logical_anchors[index]
        _require(actual_anchor == expected_anchors[ordinal], f"monitor anchor identity mismatch at {index}")
        sample = dataset[index]
        _require(isinstance(sample, dict), f"dataset sample {index} is not a mapping")
        state = sample.get("observation.state")
        action = sample.get("action")
        _require(
            isinstance(state, torch.Tensor)
            and tuple(state.shape) == (STATE_DIM,)
            and bool(torch.isfinite(state).all()),
            f"raw state contract failed at monitor index {index}",
        )
        _require(
            isinstance(action, torch.Tensor)
            and tuple(action.shape) == (ACTION_CHUNK_SIZE, ACTION_DIM)
            and bool(torch.isfinite(action).all()),
            f"raw action contract failed at monitor index {index}",
        )
        for key in CAMERA_KEYS:
            image = sample.get(key)
            _require(
                isinstance(image, torch.Tensor) and image.dtype is torch.uint8 and image.ndim == 3,
                f"raw camera contract failed for {key} at monitor index {index}",
            )
        samples[index] = copy.deepcopy(sample)
    return dataset, samples


def _strict_load_summary(load_report: Mapping[str, Any], *, expect_project_manifest: bool) -> dict[str, Any]:
    pretrained = _required_mapping(load_report.get("pretrained"), "strict load report")
    _require(pretrained.get("strict") is True, "checkpoint load was not strict")
    _require(pretrained.get("missing_keys") == [], "strict load has missing keys")
    _require(pretrained.get("unexpected_keys") == [], "strict load has unexpected keys")
    _require(
        pretrained.get("project_manifest_present") is expect_project_manifest,
        "project checkpoint manifest presence mismatch",
    )
    _require(
        pretrained.get("weights_namespace") == PI0_CORE_WEIGHTS_NAMESPACE,
        "strict load used the wrong weights namespace",
    )
    effective = _required_mapping(load_report.get("effective_stats"), "effective stats load report")
    stats_sha = effective.get("effective_stats_sha256")
    _require(isinstance(stats_sha, str) and len(stats_sha) == 64, "effective stats hash missing")
    verified_tensor = _required_mapping(pretrained.get("verified_tensor"), "strict verified tensor")
    _require(verified_tensor.get("exact_after_model_dtype_cast") is True, "strict tensor verification failed")
    return {
        "strict": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "project_manifest_present": expect_project_manifest,
        "weights_namespace": pretrained["weights_namespace"],
        "weights_path": str(pretrained.get("weights_path")),
        "weights_size_bytes": int(pretrained.get("weights_size_bytes", 0)),
        "verified_tensor_key": str(verified_tensor.get("checkpoint_key")),
        "verified_tensor_sha256": str(verified_tensor.get("checkpoint_tensor_sha256")),
        "effective_stats_sha256": stats_sha,
    }


def _predict_monitor(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    samples: Mapping[int, Mapping[str, Any]],
    monitor_indices: Sequence[int],
    noises: Mapping[int, torch.Tensor],
    device: torch.device,
) -> PredictionSet:
    normalized_predictions: list[torch.Tensor] = []
    raw_predictions: list[torch.Tensor] = []
    raw_targets: list[torch.Tensor] = []
    policy.eval()

    for index in monitor_indices:
        _require(index in samples and index in noises, f"missing cached sample/noise for index {index}")
        batch = torch.utils.data.default_collate([copy.deepcopy(samples[index])])
        target = (
            _require_action_tensor(batch.get("action"), "raw target action", batch_size=1)
            .detach()
            .cpu()
            .clone()
        )
        _convert_images_to_float01(batch)
        processed = preprocessor(batch)
        _require(isinstance(processed, dict), "preprocessor did not return a batch mapping")
        state = processed.get("observation.state")
        _require(
            isinstance(state, torch.Tensor)
            and tuple(state.shape) == (1, STATE_DIM)
            and bool(torch.isfinite(state).all()),
            "processed state must be finite [1,10]",
        )
        _require_action_tensor(processed.get("action"), "processed target action", batch_size=1)

        noise = noises[index].to(device=device, dtype=torch.float32, non_blocking=False)
        _require(tuple(noise.shape) == (1, ACTION_CHUNK_SIZE, MAX_ACTION_DIM), "fixed noise shape drift")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            prediction = policy.predict_action_chunk(
                processed,
                **{"noise": noise, "num_steps": NUM_INFERENCE_STEPS},
            )
        prediction = _require_action_tensor(prediction, "normalized prediction", batch_size=1)
        normalized_prediction = prediction.detach().to(device="cpu", dtype=torch.float32).clone()
        raw_prediction = postprocessor(prediction.detach().clone())
        raw_prediction = _require_action_tensor(raw_prediction, "raw prediction", batch_size=1)
        normalized_predictions.append(normalized_prediction)
        raw_predictions.append(raw_prediction.detach().to(device="cpu", dtype=torch.float32))
        raw_targets.append(target.to(dtype=torch.float32))

    return PredictionSet(
        monitor_indices=tuple(int(index) for index in monitor_indices),
        normalized_actions=torch.cat(normalized_predictions, dim=0),
        raw_actions=torch.cat(raw_predictions, dim=0),
        target_actions=torch.cat(raw_targets, dim=0),
    )


def _release_model_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _evaluate_checkpoint_once(
    checkpoint_dir: Path,
    dataset_stats: Mapping[str, Any],
    samples: Mapping[int, Mapping[str, Any]],
    monitor_indices: Sequence[int],
    noises: Mapping[int, torch.Tensor],
    device: torch.device,
    *,
    expect_project_manifest: bool,
    saved_processors_dir: Path | None = None,
) -> tuple[PredictionSet, dict[str, Any]]:
    policy = None
    factory_preprocessor = None
    factory_postprocessor = None
    preprocessor = None
    postprocessor = None
    try:
        with _offline_model_loading():
            policy, factory_preprocessor, factory_postprocessor, load_report = (
                load_pi0_full_policy_and_processors(checkpoint_dir, dataset_stats, device)
            )
        summary = _strict_load_summary(
            load_report,
            expect_project_manifest=expect_project_manifest,
        )
        if saved_processors_dir is None:
            preprocessor, postprocessor = factory_preprocessor, factory_postprocessor
            summary["processor_source"] = "factory_from_shared_effective_stats"
        else:
            preprocessor, postprocessor = load_saved_processors(saved_processors_dir, device)
            summary["processor_source"] = "saved_checkpoint_files"
        predictions = _predict_monitor(
            policy,
            preprocessor,
            postprocessor,
            samples,
            monitor_indices,
            noises,
            device,
        )
        return predictions, summary
    finally:
        policy = None
        factory_preprocessor = None
        factory_postprocessor = None
        preprocessor = None
        postprocessor = None
        _release_model_memory(device)


def _metric_values(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    translation_mm = np.linalg.norm(prediction[..., :3] - target[..., :3], axis=-1) * 1000.0
    prediction_rotation = so3_exp(prediction[..., 3:6])
    target_rotation = so3_exp(target[..., 3:6])
    rotation_deg = np.rad2deg(rotation_geodesic_angle(prediction_rotation, target_rotation))
    gripper_error = np.abs(prediction[..., 6] - target[..., 6])
    for name, values in (
        ("translation error", translation_mm),
        ("rotation error", rotation_deg),
        ("gripper error", gripper_error),
    ):
        _require(np.all(np.isfinite(values)), f"{name} contains NaN/Inf")
        _require(np.all(values >= 0.0), f"{name} contains a negative value")
    return translation_mm, rotation_deg, gripper_error


def _mean_metrics(
    translation_mm: np.ndarray,
    rotation_deg: np.ndarray,
    gripper_error: np.ndarray,
) -> dict[str, float]:
    values = {
        "translation_ade_mm": float(np.mean(translation_mm)),
        "rotation_ade_deg": float(np.mean(rotation_deg)),
        "gripper_mae": float(np.mean(gripper_error)),
    }
    _require(all(math.isfinite(value) and value >= 0.0 for value in values.values()), "non-finite metric")
    return values


def compute_action_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    monitor_indices: Sequence[int],
    anchors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute anchor means, episode macro means, and the final episode-macro metric."""

    sample_count = len(monitor_indices)
    prediction = _require_action_tensor(prediction, "metric prediction", batch_size=sample_count)
    target = _require_action_tensor(target, "metric target", batch_size=sample_count)
    _require(len(anchors) == sample_count, "metric anchors do not align with predictions")
    prediction_np = prediction.detach().cpu().to(torch.float64).numpy()
    target_np = target.detach().cpu().to(torch.float64).numpy()
    translation_mm, rotation_deg, gripper_error = _metric_values(prediction_np, target_np)

    per_anchor: list[dict[str, Any]] = []
    episode_values: dict[int, list[dict[str, float]]] = defaultdict(list)
    for ordinal, (index, anchor) in enumerate(zip(monitor_indices, anchors, strict=True)):
        episode_index = anchor.get("episode_index")
        _require(type(episode_index) is int, f"anchor {index} lacks integer episode_index")
        metrics = _mean_metrics(
            translation_mm[ordinal],
            rotation_deg[ordinal],
            gripper_error[ordinal],
        )
        episode_values[episode_index].append(metrics)
        per_anchor.append(
            {
                "monitor_index": int(index),
                "episode_index": episode_index,
                "raw_episode_id": str(anchor.get("raw_episode_id")),
                **metrics,
            }
        )

    per_episode: list[dict[str, Any]] = []
    for episode_index in sorted(episode_values):
        anchor_metrics = episode_values[episode_index]
        episode_metrics = {
            key: float(np.mean([item[key] for item in anchor_metrics]))
            for key in ("translation_ade_mm", "rotation_ade_deg", "gripper_mae")
        }
        per_episode.append(
            {
                "episode_index": episode_index,
                "anchor_count": len(anchor_metrics),
                **episode_metrics,
            }
        )
    aggregate = {
        key: float(np.mean([item[key] for item in per_episode]))
        for key in ("translation_ade_mm", "rotation_ade_deg", "gripper_mae")
    }
    _require(all(math.isfinite(value) for value in aggregate.values()), "aggregate metric non-finite")
    return {
        "aggregation": "episode_macro_of_anchor_horizon_means",
        "aggregate": aggregate,
        "per_episode": per_episode,
        "per_anchor": per_anchor,
    }


def trained_over_base_ratios(
    base_metrics: Mapping[str, float],
    trained_metrics: Mapping[str, float],
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for key in ("translation_ade_mm", "rotation_ade_deg", "gripper_mae"):
        base = base_metrics.get(key)
        trained = trained_metrics.get(key)
        _require(isinstance(base, (int, float)) and math.isfinite(float(base)), f"invalid base {key}")
        _require(
            isinstance(trained, (int, float)) and math.isfinite(float(trained)),
            f"invalid trained {key}",
        )
        _require(float(base) > 0.0, f"cannot report trained/base ratio because base {key} is zero")
        ratio = float(trained) / float(base)
        _require(math.isfinite(ratio) and ratio >= 0.0, f"invalid trained/base ratio for {key}")
        ratios[key] = ratio
    return ratios


def _max_abs_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    _require(tuple(first.shape) == tuple(second.shape), "reload comparison shape mismatch")
    _require(bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all()), "reload comparison non-finite")
    return float(torch.max(torch.abs(first.to(torch.float64) - second.to(torch.float64))))


def _verify_reload_consistency(trained: PredictionSet, reloaded: PredictionSet) -> dict[str, Any]:
    _require(trained.monitor_indices == reloaded.monitor_indices, "reload monitor order changed")
    _require(torch.equal(trained.target_actions, reloaded.target_actions), "reload raw targets changed")
    normalized_max = _max_abs_difference(trained.normalized_actions, reloaded.normalized_actions)
    raw_max = _max_abs_difference(trained.raw_actions, reloaded.raw_actions)
    normalized_allclose = torch.allclose(
        trained.normalized_actions,
        reloaded.normalized_actions,
        rtol=RELOAD_RTOL,
        atol=RELOAD_ATOL,
    )
    raw_allclose = torch.allclose(
        trained.raw_actions,
        reloaded.raw_actions,
        rtol=RELOAD_RTOL,
        atol=RELOAD_ATOL,
    )
    _require(normalized_allclose, f"strict model reload predictions differ; max_abs={normalized_max}")
    _require(raw_allclose, f"saved processor reload predictions differ; max_abs={raw_max}")
    return {
        "allclose": True,
        "rtol": RELOAD_RTOL,
        "atol": RELOAD_ATOL,
        "normalized_action_max_abs_diff": normalized_max,
        "raw_action_max_abs_diff": raw_max,
    }


def evaluate_s1_run(run_dir: str | Path, *, device: str | torch.device = "cuda:0") -> dict[str, Any]:
    """Run base/trained/reloaded fixed-noise evaluation and return JSON-safe data."""

    verified = verify_s1_run(run_dir)
    torch_device = _validate_device(device)
    dataset, samples = _load_dataset_and_samples(verified)
    _require(dataset.effective_stats is not None, "dataset effective stats unexpectedly absent")
    monitor_indices = tuple(int(index) for index in verified.monitor["indices"])
    training_seed = int(verified.config["training"]["seed"])

    noises: dict[int, torch.Tensor] = {}
    noise_records: list[dict[str, Any]] = []
    for index in monitor_indices:
        noise_seed, noise, digest = fixed_noise_cpu(training_seed, index)
        noises[index] = noise
        noise_records.append(
            {
                "monitor_index": index,
                "seed": noise_seed,
                "sha256": digest,
            }
        )

    base_predictions, base_load = _evaluate_checkpoint_once(
        verified.base_model_dir,
        dataset.effective_stats,
        samples,
        monitor_indices,
        noises,
        torch_device,
        expect_project_manifest=False,
    )
    trained_predictions, trained_load = _evaluate_checkpoint_once(
        verified.model_dir,
        dataset.effective_stats,
        samples,
        monitor_indices,
        noises,
        torch_device,
        expect_project_manifest=True,
    )
    reloaded_predictions, reloaded_load = _evaluate_checkpoint_once(
        verified.model_dir,
        dataset.effective_stats,
        samples,
        monitor_indices,
        noises,
        torch_device,
        expect_project_manifest=True,
        saved_processors_dir=verified.model_dir,
    )

    _require(
        torch.equal(base_predictions.target_actions, trained_predictions.target_actions),
        "base/trained raw targets differ",
    )
    stats_hashes = {
        base_load["effective_stats_sha256"],
        trained_load["effective_stats_sha256"],
        reloaded_load["effective_stats_sha256"],
    }
    _require(len(stats_hashes) == 1, "base/trained/reload effective stats differ")
    reload_report = _verify_reload_consistency(trained_predictions, reloaded_predictions)

    anchors = verified.monitor["anchors"]
    base_metrics = compute_action_metrics(
        base_predictions.raw_actions,
        base_predictions.target_actions,
        monitor_indices,
        anchors,
    )
    trained_metrics = compute_action_metrics(
        trained_predictions.raw_actions,
        trained_predictions.target_actions,
        monitor_indices,
        anchors,
    )
    ratios = trained_over_base_ratios(base_metrics["aggregate"], trained_metrics["aggregate"])

    return {
        "schema_version": 1,
        "status": "EVALUATION_COMPLETE",
        "quality_threshold_applied": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "run_dir": str(verified.run_dir),
            "stage": "S1-15",
            "final_step": FINAL_S1_STEP,
            "checkpoint": str(verified.checkpoint),
            "base_model": str(verified.base_model_dir),
        },
        "dataset": {
            "root": str(verified.dataset_root),
            "profile": "action15",
            "observation_fps": 15,
            "action_fps": 15,
            "chunk_size": ACTION_CHUNK_SIZE,
            "dataset_size": len(dataset),
            "stats_file_sha256": verified.stats_file_sha256,
            "effective_stats_sha256": base_load["effective_stats_sha256"],
        },
        "monitor": {
            "held_out": False,
            "subset_sha256": verified.monitor["subset_sha256"],
            "indices": list(monitor_indices),
            "anchors": anchors,
            "sample_count": len(monitor_indices),
        },
        "inference": {
            "device": str(torch_device),
            "num_steps": NUM_INFERENCE_STEPS,
            "noise_shape_per_sample": [1, ACTION_CHUNK_SIZE, MAX_ACTION_DIM],
            "noise_dtype": "float32",
            "noise_generation": "CPU torch.Generator; training_seed + 100000 + monitor_index",
            "per_sample_noise": noise_records,
        },
        "strict_loads": {
            "base": base_load,
            "trained_step300": trained_load,
            "trained_step300_reload": reloaded_load,
        },
        "metrics": {
            "definitions": {
                "translation_ade_mm": "mean L2(pred_delta_xyz-target_delta_xyz), millimetres",
                "rotation_ade_deg": "mean SO(3) geodesic(Exp(pred_rotvec),Exp(target_rotvec)), degrees",
                "gripper_mae": "mean absolute raw gripper_0_1 error; predictions are not clipped",
                "aggregate": "episode macro-average of anchor horizon means",
            },
            "base": base_metrics,
            "trained": trained_metrics,
            "trained_over_base_ratio": ratios,
        },
        "reload_consistency": {
            **reload_report,
            "model": "second independent strict step-300 load",
            "processors": "policy_preprocessor.json/policy_postprocessor.json plus saved state files",
            "fixed_noise_reused": True,
        },
    }


def _write_output_json(path: Path, report: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_s1_run(args.run_dir, device=args.device)
    if args.output_json is not None:
        _write_output_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
