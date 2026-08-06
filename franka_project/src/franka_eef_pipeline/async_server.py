"""Franka-specific adapters for LeRobot's asynchronous policy server.

The transport, request handling, policy-processor orchestration, and action
timing remain owned by :class:`lerobot.async_inference.PolicyServer`.
This module only supplies the project-specific boundaries that the stock
server cannot infer:

* reproduce the training-time, aspect-ratio-preserving PI0 camera resize;
* restore the project's unprefixed, full-parameter PI0 checkpoint strictly;
* decode checkpoint-declared delta or absolute 7D Cartesian actions into absolute 8D
  ``[xyz, quaternion_xyzw, gripper]`` targets before they leave the server;
* reconstruct and run a native-EEF FastWAM checkpoint with its frozen image,
  proprio, normalization, text-context, and action-label contracts.

One server process owns exactly one policy backend.  The launcher selects the
PI0 or FastWAM servicer at startup from ``PolicyServerConfig.policy_type``;
models are never hot-swapped while a robot session is active.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import pickle
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import (
    ASYNC_INFERENCE_PROTOCOL_VERSION,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    make_lerobot_observation,
)
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2
from lerobot.utils.constants import OBS_STATE

from .dual_rate_dataset import (
    ACTION_LABEL_MODE_ABSOLUTE_EEF,
    ACTION_LABEL_MODE_DELTA_EEF,
    resolve_action_label_spec,
)
from .geometry import (
    decode_absolute_action,
    decode_relative_action,
    enforce_quaternion_continuity,
    matrix_to_quaternion_xyzw,
    rotation_6d_to_matrix,
    so3_exp,
    so3_log,
)

FRANKA_CHECKPOINT_TYPE = "franka_pi0_full_parameter_eef"
PI0_CORE_WEIGHTS_NAMESPACE = "pi0_core_unprefixed"
STATE_DIM = 10
ACTION_DIM = 7
ABSOLUTE_ACTION_DIM = 8
FRANKA_STATE_NAMES = (
    "eef.x",
    "eef.y",
    "eef.z",
    "eef.rot6d.col0.x",
    "eef.rot6d.col0.y",
    "eef.rot6d.col0.z",
    "eef.rot6d.col1.x",
    "eef.rot6d.col1.y",
    "eef.rot6d.col1.z",
    "gripper.closed_0_1",
)
FRANKA_CAMERA_SHAPE = (480, 640, 3)
FRANKA_CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
)
FRANKA_PI0_RENAME_MAP = {
    "observation.images.camera1": "observation.images.base_0_rgb",
    "observation.images.camera2": "observation.images.left_wrist_0_rgb",
}
# Compatibility alias retained for existing client commands and imports.
FRANKA_RENAME_MAP = FRANKA_PI0_RENAME_MAP
FRANKA_FASTWAM_RENAME_MAP: dict[str, str] = {}
FRANKA_POLICY_TYPES = ("pi0", "fastwam")

FASTWAM_STATE_DIM = 8
FASTWAM_ACTION_DIM = 7
FASTWAM_RUNTIME_MODEL_TARGET = "fastwam.runtime.create_fastwam"
FASTWAM_JOINT_RUNTIME_MODEL_TARGET = "fastwam.runtime.create_fastwam_joint"
FASTWAM_RUNTIME_MODEL_TARGETS = (
    FASTWAM_RUNTIME_MODEL_TARGET,
    FASTWAM_JOINT_RUNTIME_MODEL_TARGET,
)
FASTWAM_RUNTIME_CONFIG_NAME = "fastwam_runtime.resolved.yaml"
FASTWAM_DATASET_CONTRACT_NAME = "dataset_contract.json"
FASTWAM_RUN_STATS_NAME = "dataset_stats.json"
FASTWAM_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
FASTWAM_CONTEXT_WIDTH = 4096
FASTWAM_FINGER_SCALE = 0.04
FASTWAM_FINGER_SIGNS = (1.0, -1.0)
FASTWAM_CHECKPOINT_PATTERN = re.compile(r"^step_(\d+)\.pt$")
FASTWAM_ACTION_TYPE_BY_LABEL_MODE = {
    ACTION_LABEL_MODE_DELTA_EEF: "adjacent_delta_eef",
    ACTION_LABEL_MODE_ABSOLUTE_EEF: "next_absolute_eef",
}
FASTWAM_ACTION_NAMES_BY_LABEL_MODE = {
    ACTION_LABEL_MODE_DELTA_EEF: [
        "delta_eef.x_m_base",
        "delta_eef.y_m_base",
        "delta_eef.z_m_base",
        "delta_eef.rotvec_x_rad_body",
        "delta_eef.rotvec_y_rad_body",
        "delta_eef.rotvec_z_rad_body",
        "gripper.open_target_0_1",
    ],
    ACTION_LABEL_MODE_ABSOLUTE_EEF: [
        "eef_target.x_m_base",
        "eef_target.y_m_base",
        "eef_target.z_m_base",
        "eef_target.axis_angle_x_rad_base",
        "eef_target.axis_angle_y_rad_base",
        "eef_target.axis_angle_z_rad_base",
        "gripper.open_target_0_1",
    ],
}
FASTWAM_WAN_VAE_RELATIVE_PATH = Path(
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
)

_MANIFEST_NAME = "franka_pi0_checkpoint_manifest.json"
_MANIFEST_TRACKED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "franka_eef_geometry_manifest.json",
    "pi0_eef_stats.json",
)

_STATE_DESCRIPTION = "current measured EEF xyz + rotation6d(first two columns) + gripper_0_1"
_ACTION_SEMANTICS_BY_LABEL_MODE = {
    ACTION_LABEL_MODE_DELTA_EEF: {
        "gripper": "future measured target gripper_0_1",
        "rotation": "body rotvec Log(R_current.T @ R_target)",
        "translation": "base-frame target_xyz - current_xyz",
    },
    ACTION_LABEL_MODE_ABSOLUTE_EEF: {
        "gripper": "future measured absolute target gripper closed_0_1",
        "rotation": "principal base-frame rotvec Log(R_target)",
        "translation": "absolute base-frame target_xyz",
    },
}
_MISSING = object()


class FrankaAsyncPolicyContractError(RuntimeError):
    """Raised when remote inference would violate the frozen Franka contract."""


def _resolve_checkpoint_action_label(
    payload: dict[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """Resolve legacy/new action-label metadata as a serving contract error."""

    try:
        return resolve_action_label_spec(payload, source=name)
    except (TypeError, ValueError) as exc:
        raise FrankaAsyncPolicyContractError(
            f"Invalid Franka action-label contract in {name}: {exc}"
        ) from exc


@dataclass(frozen=True)
class FrankaCheckpointContract:
    """Checkpoint-owned serving values after cross-file contract validation."""

    profile: str
    task_instruction: str
    observation_fps: int
    action_fps: int
    chunk_size: int
    action_label_mode: str
    real_robot_rollout_authorized: bool
    rollout_authorization_declared: bool


@dataclass(frozen=True)
class FrankaFastWAMCheckpointContract:
    """Frozen native-EEF FastWAM serving inputs derived from one training run."""

    checkpoint_path: Path
    run_dir: Path
    runtime_config_path: Path
    stats_path: Path
    text_context_path: Path
    vae_path: Path
    task_instruction: str
    observation_fps: int
    action_fps: int
    chunk_size: int
    action_label_mode: str
    action_video_freq_ratio: int
    num_video_frames: int
    video_fps: float
    image_height: int
    image_width: int
    context_len: int
    model_id: str
    model_target: str
    checkpoint_step: int
    model_dtype: str
    num_inference_steps: int
    seed: int
    finger_scale: float
    finger_signs: tuple[float, float]
    text_context_paths: Mapping[str, Path] | None = None
    task_instructions: tuple[str, ...] | None = None


FrankaServingContract = FrankaCheckpointContract | FrankaFastWAMCheckpointContract


@dataclass(frozen=True)
class _FastWAMMinMaxNormalizer:
    """Exact frozen min/max normalization used by this FastWAM run."""

    state_min: torch.Tensor
    state_max: torch.Tensor
    action_min: torch.Tensor
    action_max: torch.Tensor

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(state, dtype=torch.float32, device="cpu")
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if tuple(value.shape) != (1, FASTWAM_STATE_DIM):
            raise FrankaAsyncPolicyContractError(
                f"FastWAM state must have shape [1,{FASTWAM_STATE_DIM}], got {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise FrankaAsyncPolicyContractError("FastWAM state contains NaN or Inf")
        normalized = 2.0 * (value - self.state_min) / (self.state_max - self.state_min) - 1.0
        return torch.clamp(normalized, -5.0, 5.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(action, dtype=torch.float32, device="cpu")
        if value.ndim != 2 or value.shape[1] != FASTWAM_ACTION_DIM or value.shape[0] <= 0:
            raise FrankaAsyncPolicyContractError(
                f"Normalized FastWAM action must have shape [K,{FASTWAM_ACTION_DIM}], "
                f"got {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise FrankaAsyncPolicyContractError("Normalized FastWAM action contains NaN or Inf")
        return self.action_min + 0.5 * (value + 1.0) * (self.action_max - self.action_min)


@dataclass(frozen=True)
class _FastWAMRuntime:
    model: Any
    normalizer: _FastWAMMinMaxNormalizer
    contexts_by_task: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    # Compatibility fields for single-task fixtures and downstream users.
    context: torch.Tensor | None = None
    context_mask: torch.Tensor | None = None


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise FrankaAsyncPolicyContractError(f"Could not read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise FrankaAsyncPolicyContractError(f"Expected a JSON object in {path}")
    return value


def _read_yaml_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise FrankaAsyncPolicyContractError(f"Could not read valid YAML from {path}") from exc
    if not isinstance(value, dict):
        raise FrankaAsyncPolicyContractError(f"Expected a YAML object in {path}")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrankaAsyncPolicyContractError(f"{name} must be an object")
    return value


def _configured_path(value: Any, *, config_path: Path, name: str) -> Path:
    path_value = _nonempty_string(value, name=name)
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _fastwam_stats_vector(
    stats: Mapping[str, Any],
    *,
    group: str,
    statistic: str,
    dimension: int,
    path: Path,
) -> torch.Tensor:
    group_stats = _mapping(stats.get(group), name=f"{path}:{group}")
    default_stats = _mapping(group_stats.get("default"), name=f"{path}:{group}.default")
    try:
        value = torch.as_tensor(default_stats[f"global_{statistic}"], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise FrankaAsyncPolicyContractError(
            f"{path}:{group}.default.global_{statistic} must be a numeric vector"
        ) from exc
    if tuple(value.shape) != (dimension,) or not bool(torch.isfinite(value).all()):
        raise FrankaAsyncPolicyContractError(
            f"{path}:{group}.default.global_{statistic} must be finite shape ({dimension},)"
        )
    return value


def _load_fastwam_minmax_normalizer(path: Path) -> _FastWAMMinMaxNormalizer:
    stats = _read_json_object(path)
    state_min = _fastwam_stats_vector(
        stats,
        group="state",
        statistic="min",
        dimension=FASTWAM_STATE_DIM,
        path=path,
    )
    state_max = _fastwam_stats_vector(
        stats,
        group="state",
        statistic="max",
        dimension=FASTWAM_STATE_DIM,
        path=path,
    )
    action_min = _fastwam_stats_vector(
        stats,
        group="action",
        statistic="min",
        dimension=FASTWAM_ACTION_DIM,
        path=path,
    )
    action_max = _fastwam_stats_vector(
        stats,
        group="action",
        statistic="max",
        dimension=FASTWAM_ACTION_DIM,
        path=path,
    )
    # The frozen processor's SingleFieldLinearNormalizer has special behavior
    # for ranges below 1e-4.  This run has no such dimensions; reject drift
    # instead of silently using a different formula.
    for name, minimum, maximum in (
        ("state", state_min, state_max),
        ("action", action_min, action_max),
    ):
        value_range = maximum - minimum
        if bool(torch.any(value_range < 1e-4)):
            raise FrankaAsyncPolicyContractError(
                f"FastWAM {name} stats contain a range below the frozen 1e-4 tolerance"
            )
    return _FastWAMMinMaxNormalizer(
        state_min=state_min,
        state_max=state_max,
        action_min=action_min,
        action_max=action_max,
    )


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_processor_state_files(checkpoint_dir: Path, config_name: str) -> None:
    config_path = checkpoint_dir / config_name
    config = _read_json_object(config_path)
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise FrankaAsyncPolicyContractError(f"Processor config has no steps: {config_path}")

    for step in steps:
        if not isinstance(step, dict):
            raise FrankaAsyncPolicyContractError(f"Invalid processor step in {config_path}")
        state_file = step.get("state_file")
        if state_file is None:
            continue
        if not isinstance(state_file, str) or not state_file:
            raise FrankaAsyncPolicyContractError(f"Invalid processor state_file in {config_path}")
        state_path = (checkpoint_dir / state_file).resolve()
        if not state_path.is_relative_to(checkpoint_dir) or not state_path.is_file():
            raise FrankaAsyncPolicyContractError(
                f"Processor state file is missing or escapes the checkpoint: {state_file}"
            )
        if state_path.stat().st_size <= 0:
            raise FrankaAsyncPolicyContractError(f"Processor state file is empty: {state_path}")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FrankaAsyncPolicyContractError(f"{name} must be a positive integer, got {value!r}")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrankaAsyncPolicyContractError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _sha256_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FrankaAsyncPolicyContractError(f"{name} must be a 64-character SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise FrankaAsyncPolicyContractError(f"{name} must be a 64-character SHA-256 hex string") from exc
    return value


def _rollout_authorization(mapping: dict[str, Any], *, name: str) -> tuple[bool, bool]:
    """Parse checkpoint provenance without allowing it to authorize actuation.

    Legacy native30 checkpoints omit this field. Missing means unapproved, just
    like an explicit ``false``; malformed values and self-authorization are
    rejected.
    """

    value = mapping.get("real_robot_rollout_authorized", _MISSING)
    if value is _MISSING:
        return False, False
    if value is not False:
        raise FrankaAsyncPolicyContractError(
            f"{name}.real_robot_rollout_authorized must be false when present, got {value!r}"
        )
    return False, True


def _validate_model_config(model_config: dict[str, Any], *, chunk_size: int) -> None:
    if model_config.get("type") != "pi0":
        raise FrankaAsyncPolicyContractError(
            f"Franka model config type must be 'pi0', got {model_config.get('type')!r}"
        )

    features = (
        ("input_features", OBS_STATE, "STATE", STATE_DIM),
        ("output_features", "action", "ACTION", ACTION_DIM),
    )
    for container_name, feature_name, expected_type, expected_dimension in features:
        container = model_config.get(container_name)
        feature = container.get(feature_name) if isinstance(container, dict) else None
        shape_value = feature.get("shape") if isinstance(feature, dict) else None
        shape = tuple(shape_value) if isinstance(shape_value, (list, tuple)) else None
        actual = {
            "type": feature.get("type") if isinstance(feature, dict) else None,
            "shape": shape,
        }
        expected = {"type": expected_type, "shape": (expected_dimension,)}
        if actual != expected:
            raise FrankaAsyncPolicyContractError(
                f"Franka model config {feature_name!r} mismatch: expected={expected}, actual={actual}"
            )

    for field_name in ("chunk_size", "n_action_steps"):
        actual_chunk_size = _positive_int(model_config.get(field_name), name=f"config.{field_name}")
        if actual_chunk_size != chunk_size:
            raise FrankaAsyncPolicyContractError(
                f"Franka model config {field_name} mismatch: expected={chunk_size}, "
                f"actual={actual_chunk_size}"
            )

    if model_config.get("use_relative_actions") is not False:
        raise FrankaAsyncPolicyContractError(
            "Franka model config use_relative_actions must be false because the project adapter "
            "owns model-action7-to-canonical-absolute8 decoding"
        )

    normalization = model_config.get("normalization_mapping")
    expected_normalization = {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    actual_normalization = (
        {name: normalization.get(name) for name in expected_normalization}
        if isinstance(normalization, dict)
        else None
    )
    if actual_normalization != expected_normalization:
        raise FrankaAsyncPolicyContractError(
            "Franka model normalization mismatch: "
            f"expected={expected_normalization}, actual={actual_normalization}"
        )


def _validate_geometry_and_stats_manifests(
    checkpoint_dir: Path,
    *,
    expected_fps: int | None = None,
    expected_actions_per_chunk: int | None = None,
) -> FrankaCheckpointContract:
    geometry_path = checkpoint_dir / "franka_eef_geometry_manifest.json"
    geometry = _read_json_object(geometry_path)
    if geometry.get("schema_version") != 1:
        raise FrankaAsyncPolicyContractError(
            f"Franka geometry schema_version must be 1, got {geometry.get('schema_version')!r}"
        )
    if geometry.get("state10") != _STATE_DESCRIPTION:
        raise FrankaAsyncPolicyContractError(
            "Franka geometry state10 contract mismatch: "
            f"expected={_STATE_DESCRIPTION!r}, actual={geometry.get('state10')!r}"
        )
    task_instruction = _nonempty_string(geometry.get("task_instruction"), name="geometry.task_instruction")

    action7 = geometry.get("action7")
    profile = geometry.get("dataset_profile")
    if not isinstance(action7, dict):
        raise FrankaAsyncPolicyContractError("Franka geometry manifest has no action7 object")
    if not isinstance(profile, dict):
        raise FrankaAsyncPolicyContractError("Franka geometry manifest has no dataset_profile object")

    # Missing fields are the legacy delta checkpoint contract. Once any new
    # metadata is present, the complete shared spec is mandatory. Compare the
    # full formula/frame/name contract rather than trusting a matching mode.
    action7_label_payload: dict[str, Any] = {}
    if "action_label_mode" in action7:
        action7_label_payload["action_label_mode"] = action7["action_label_mode"]
    if "contract" in action7:
        action7_label_payload["action_label"] = action7["contract"]
    action7_label = _resolve_checkpoint_action_label(
        action7_label_payload,
        name="geometry.action7",
    )
    profile_action_label = _resolve_checkpoint_action_label(
        profile,
        name="geometry.dataset_profile",
    )
    if action7_label != profile_action_label:
        raise FrankaAsyncPolicyContractError(
            "Franka action-label contract mismatch between action7 and dataset_profile"
        )
    action_label_mode = str(action7_label["mode"])
    expected_action_semantics = _ACTION_SEMANTICS_BY_LABEL_MODE[action_label_mode]
    action_semantics = {name: action7.get(name) for name in expected_action_semantics}
    if action_semantics != expected_action_semantics:
        raise FrankaAsyncPolicyContractError(
            "Franka geometry action7 semantics mismatch: "
            f"expected={expected_action_semantics}, actual={action_semantics}"
        )

    if profile.get("schema_version") != 1:
        raise FrankaAsyncPolicyContractError(
            f"Franka dataset profile schema_version must be 1, got {profile.get('schema_version')!r}"
        )
    profile_name = _nonempty_string(profile.get("profile"), name="geometry.dataset_profile.profile")
    observation_fps = _positive_int(
        profile.get("observation_fps"), name="geometry.dataset_profile.observation_fps"
    )
    action_fps = _positive_int(profile.get("action_fps"), name="geometry.dataset_profile.action_fps")
    chunk_size = _positive_int(profile.get("chunk_size"), name="geometry.dataset_profile.chunk_size")
    action_frequency = _positive_int(action7.get("frequency_hz"), name="geometry.action7.frequency_hz")
    action_chunk_size = _positive_int(action7.get("chunk_size"), name="geometry.action7.chunk_size")
    if action_frequency != action_fps or action_chunk_size != chunk_size:
        raise FrankaAsyncPolicyContractError(
            "Franka geometry action7/profile timing mismatch: "
            f"action7=(frequency_hz={action_frequency}, chunk_size={action_chunk_size}), "
            f"profile=(action_fps={action_fps}, chunk_size={chunk_size})"
        )
    if observation_fps != action_fps:
        raise FrankaAsyncPolicyContractError(
            "Franka async server requires equal observation/action rates, got "
            f"observation_fps={observation_fps}, action_fps={action_fps}"
        )
    if profile.get("requires_project_cartesian_adapter") is not True:
        raise FrankaAsyncPolicyContractError(
            "Franka dataset profile requires_project_cartesian_adapter must be true"
        )
    if profile.get("requires_project_dual_rate_adapter") is not False:
        raise FrankaAsyncPolicyContractError(
            "Franka dataset profile requires_project_dual_rate_adapter must be false"
        )
    if profile.get("partial_conversion", _MISSING) is not False:
        raise FrankaAsyncPolicyContractError("Franka dataset profile partial_conversion must be false")
    profile_task = profile.get("task_instruction")
    if profile_task != task_instruction:
        raise FrankaAsyncPolicyContractError(
            f"Franka geometry task mismatch: top_level={task_instruction!r}, dataset_profile={profile_task!r}"
        )
    _, geometry_rollout_declared = _rollout_authorization(profile, name="geometry.dataset_profile")

    stats_path = checkpoint_dir / "pi0_eef_stats.json"
    stats = _read_json_object(stats_path)
    expected_stats = {
        "schema_version": 1,
        "profile": profile_name,
        "observation_fps": observation_fps,
        "action_fps": action_fps,
        "chunk_size": chunk_size,
    }
    actual_stats = {key: stats.get(key) for key in expected_stats}
    if actual_stats != expected_stats:
        raise FrankaAsyncPolicyContractError(
            f"Franka stats manifest contract mismatch: expected={expected_stats}, actual={actual_stats}"
        )
    stats_action_label = _resolve_checkpoint_action_label(stats, name="stats")
    if stats_action_label != profile_action_label:
        raise FrankaAsyncPolicyContractError(
            "Franka action-label contract mismatch between stats and geometry/profile"
        )
    _, stats_rollout_declared = _rollout_authorization(stats, name="stats")
    if geometry_rollout_declared != stats_rollout_declared:
        raise FrankaAsyncPolicyContractError(
            "Franka rollout authorization declaration must match between geometry and stats"
        )

    geometry_dataset_hash = _sha256_string(
        profile.get("source_dataset_hash"), name="geometry.dataset_profile.source_dataset_hash"
    )
    stats_dataset_hash = _sha256_string(stats.get("source_dataset_hash"), name="stats.source_dataset_hash")
    if geometry_dataset_hash != stats_dataset_hash:
        raise FrankaAsyncPolicyContractError(
            "Franka source_dataset_hash mismatch between geometry and stats: "
            f"geometry={geometry_dataset_hash!r}, stats={stats_dataset_hash!r}"
        )

    for feature_name, dimension in ((OBS_STATE, STATE_DIM), ("action", ACTION_DIM)):
        feature_stats = stats.get(feature_name)
        if not isinstance(feature_stats, dict):
            raise FrankaAsyncPolicyContractError(f"Franka stats manifest has no {feature_name!r} statistics")
        for statistic in ("min", "max", "mean", "std", "q01", "q99"):
            try:
                values = np.asarray(feature_stats.get(statistic), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise FrankaAsyncPolicyContractError(
                    f"Franka stats {feature_name}.{statistic} must be finite shape ({dimension},)"
                ) from exc
            if values.shape != (dimension,) or not np.all(np.isfinite(values)):
                raise FrankaAsyncPolicyContractError(
                    f"Franka stats {feature_name}.{statistic} must be finite shape ({dimension},)"
                )

        min_values = np.asarray(feature_stats["min"], dtype=np.float64)
        max_values = np.asarray(feature_stats["max"], dtype=np.float64)
        std_values = np.asarray(feature_stats["std"], dtype=np.float64)
        if np.any(min_values > max_values):
            raise FrankaAsyncPolicyContractError(f"Franka stats {feature_name} has min greater than max")
        if np.any(std_values < 0):
            raise FrankaAsyncPolicyContractError(f"Franka stats {feature_name}.std must be non-negative")
        if min_values[-1] < 0.0 or max_values[-1] > 1.0:
            raise FrankaAsyncPolicyContractError(
                f"Franka stats {feature_name} gripper range must stay within [0, 1]"
            )

    model_config = _read_json_object(checkpoint_dir / "config.json")
    _validate_model_config(model_config, chunk_size=chunk_size)

    if expected_fps is not None and expected_fps != observation_fps:
        raise FrankaAsyncPolicyContractError(
            f"Franka server/checkpoint fps mismatch: server={expected_fps}, checkpoint={observation_fps}"
        )
    if expected_actions_per_chunk is not None and expected_actions_per_chunk != chunk_size:
        raise FrankaAsyncPolicyContractError(
            "Franka server/checkpoint actions_per_chunk mismatch: "
            f"server={expected_actions_per_chunk}, checkpoint={chunk_size}"
        )

    return FrankaCheckpointContract(
        profile=profile_name,
        task_instruction=task_instruction,
        observation_fps=observation_fps,
        action_fps=action_fps,
        chunk_size=chunk_size,
        action_label_mode=action_label_mode,
        real_robot_rollout_authorized=False,
        rollout_authorization_declared=geometry_rollout_declared,
    )


def inspect_franka_checkpoint_contract(
    pretrained_path: str | Path,
    *,
    expected_fps: int | None = None,
    expected_actions_per_chunk: int | None = None,
) -> FrankaCheckpointContract:
    """Read and cross-check cheap checkpoint metadata without verifying file hashes."""

    checkpoint_dir = Path(pretrained_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FrankaAsyncPolicyContractError(
            f"Franka pretrained path must be a local directory: {checkpoint_dir}"
        )
    return _validate_geometry_and_stats_manifests(
        checkpoint_dir,
        expected_fps=expected_fps,
        expected_actions_per_chunk=expected_actions_per_chunk,
    )


def _shape_meta_dimension(shape_meta: Mapping[str, Any], group: str) -> int:
    items = shape_meta.get(group)
    if not isinstance(items, list) or not items:
        raise FrankaAsyncPolicyContractError(f"FastWAM shape_meta.{group} must be a non-empty list")
    total = 0
    for item in items:
        entry = _mapping(item, name=f"FastWAM shape_meta.{group} item")
        shape = entry.get("shape")
        if isinstance(shape, bool) or not isinstance(shape, int) or shape <= 0:
            raise FrankaAsyncPolicyContractError(
                f"FastWAM shape_meta.{group} item has invalid shape {shape!r}"
            )
        total += shape
    return total


def _derive_fastwam_video_timing(
    temporal: Mapping[str, Any],
    *,
    observation_fps: int,
    action_horizon: int,
) -> tuple[int, int, float]:
    """Resolve sparse model-video timing from the dense observation/action window."""

    action_video_freq_ratio = _positive_int(
        temporal.get("action_video_freq_ratio"),
        name="FastWAM temporal.action_video_freq_ratio",
    )
    num_video_frames = _positive_int(
        temporal.get("video_frames"),
        name="FastWAM temporal.video_frames",
    )
    if action_horizon % action_video_freq_ratio != 0:
        raise FrankaAsyncPolicyContractError(
            "FastWAM action_horizon must be divisible by action_video_freq_ratio: "
            f"action_horizon={action_horizon}, "
            f"action_video_freq_ratio={action_video_freq_ratio}"
        )
    expected_video_frames = action_horizon // action_video_freq_ratio + 1
    if num_video_frames != expected_video_frames:
        raise FrankaAsyncPolicyContractError(
            "FastWAM video_frames mismatch: "
            f"expected={expected_video_frames}, actual={num_video_frames}"
        )
    return (
        action_video_freq_ratio,
        num_video_frames,
        observation_fps / action_video_freq_ratio,
    )


def _resolve_fastwam_action_label_mode(action_contract: Mapping[str, Any]) -> str:
    """Validate the exact model-visible FastWAM action contract and return its mode."""

    action_type = action_contract.get("type")
    mode_by_type = {
        action_type_value: mode for mode, action_type_value in FASTWAM_ACTION_TYPE_BY_LABEL_MODE.items()
    }
    action_label_mode = mode_by_type.get(action_type)
    if action_label_mode is None:
        raise FrankaAsyncPolicyContractError(
            "FastWAM representation.action.type must be 'adjacent_delta_eef' or "
            f"'next_absolute_eef', got {action_type!r}"
        )

    declared_mode = action_contract.get("label_mode", _MISSING)
    if declared_mode is _MISSING:
        if action_label_mode != ACTION_LABEL_MODE_DELTA_EEF:
            raise FrankaAsyncPolicyContractError(
                "FastWAM absolute action representation must declare label_mode='absolute_eef'"
            )
        expected_action_contract = {
            "dim": FASTWAM_ACTION_DIM,
            "names": FASTWAM_ACTION_NAMES_BY_LABEL_MODE[ACTION_LABEL_MODE_DELTA_EEF],
            "type": FASTWAM_ACTION_TYPE_BY_LABEL_MODE[ACTION_LABEL_MODE_DELTA_EEF],
            "translation": "p[t+1]-p[t] in base frame, meter",
            "rotation": "Log(R[t].T @ R[t+1]) body rotvec, radian",
            "gripper": "absolute next target; 0=closed, 1=open",
            "delta_dimension_mask": [True, True, True, True, True, True, False],
        }
    else:
        if declared_mode != action_label_mode:
            raise FrankaAsyncPolicyContractError(
                f"FastWAM action label_mode/type mismatch: label_mode={declared_mode!r}, type={action_type!r}"
            )
        if action_label_mode == ACTION_LABEL_MODE_DELTA_EEF:
            expected_action_contract = {
                "label_mode": ACTION_LABEL_MODE_DELTA_EEF,
                "dim": FASTWAM_ACTION_DIM,
                "names": FASTWAM_ACTION_NAMES_BY_LABEL_MODE[ACTION_LABEL_MODE_DELTA_EEF],
                "type": FASTWAM_ACTION_TYPE_BY_LABEL_MODE[ACTION_LABEL_MODE_DELTA_EEF],
                "time_alignment": "action[t] targets the transition from observation t to t+1",
                "translation": "p[t+1]-p[t] in base frame, meter",
                "rotation": "Log(R[t].T @ R[t+1]) body rotvec, radian",
                "gripper": "absolute target at t+1; 0=closed, 1=open",
                "delta_dimension_mask": [True, True, True, True, True, True, False],
                "terminal_carrier": ("pose no-op plus final absolute gripper hold; excluded from training"),
            }
        else:
            expected_action_contract = {
                "label_mode": ACTION_LABEL_MODE_ABSOLUTE_EEF,
                "dim": FASTWAM_ACTION_DIM,
                "names": FASTWAM_ACTION_NAMES_BY_LABEL_MODE[ACTION_LABEL_MODE_ABSOLUTE_EEF],
                "type": FASTWAM_ACTION_TYPE_BY_LABEL_MODE[ACTION_LABEL_MODE_ABSOLUTE_EEF],
                "time_alignment": ("action[t] is the absolute EEF and gripper target at observation t+1"),
                "translation": "p[t+1] in base frame, meter",
                "rotation": "principal Log(R[t+1]) axis-angle in base frame, radian",
                "gripper": "absolute target at t+1; 0=closed, 1=open",
                "delta_dimension_mask": [False, False, False, False, False, False, False],
                "terminal_carrier": ("repeat final absolute EEF and gripper target; excluded from training"),
            }

    actual_action_contract = {key: action_contract.get(key) for key in expected_action_contract}
    if actual_action_contract != expected_action_contract:
        raise FrankaAsyncPolicyContractError(
            "FastWAM action representation mismatch for "
            f"{action_label_mode}: expected={expected_action_contract}, "
            f"actual={actual_action_contract}"
        )
    return action_label_mode


def _validate_fastwam_stats_action_label(
    path: Path,
    *,
    action_label_mode: str,
) -> None:
    """Cross-check normalization-stat semantics against the dataset contract."""

    stats = _read_json_object(path)
    statistics_contract = _mapping(
        stats.get("statistics_contract"),
        name=f"{path}:statistics_contract",
    )
    declared_mode = statistics_contract.get("action_label_mode", _MISSING)
    if declared_mode is _MISSING:
        if action_label_mode != ACTION_LABEL_MODE_DELTA_EEF:
            raise FrankaAsyncPolicyContractError(
                f"{path}:statistics_contract has no absolute action-label metadata"
            )
        return
    expected = {
        "action_label_mode": action_label_mode,
        "action_type": FASTWAM_ACTION_TYPE_BY_LABEL_MODE[action_label_mode],
        "action_names": FASTWAM_ACTION_NAMES_BY_LABEL_MODE[action_label_mode],
    }
    actual = {key: statistics_contract.get(key) for key in expected}
    if actual != expected:
        raise FrankaAsyncPolicyContractError(
            f"{path}:statistics_contract action-label mismatch: expected={expected}, actual={actual}"
        )


def _infer_fastwam_source_checkout(run_dir: Path) -> Path:
    """Find the FastWAM checkout for both flat and staged training runs."""

    resolved_run_dir = run_dir.expanduser().resolve()
    for candidate in (resolved_run_dir, *resolved_run_dir.parents):
        if (candidate / "src" / "fastwam").is_dir() and (
            candidate / "franka_project" / "src"
        ).is_dir():
            return candidate
    raise FrankaAsyncPolicyContractError(
        "Could not infer the FastWAM source checkout from the checkpoint run "
        f"{resolved_run_dir}; expected an ancestor containing src/fastwam and franka_project/src"
    )


def _resolve_fastwam_task_instructions(
    dataset_contract: Mapping[str, Any],
    *,
    contract_path: Path,
) -> tuple[str, ...]:
    """Resolve the checkpoint's task allowlist from its frozen source manifest."""

    task_contract = _mapping(dataset_contract.get("task"), name="FastWAM task contract")
    source = _mapping(dataset_contract.get("source"), name="FastWAM source contract")
    manifest_path = _configured_path(
        source.get("manifest"),
        config_path=contract_path,
        name="FastWAM source.manifest",
    )
    selected_ids_value = source.get("episode_ids")
    selected_ids = (
        {str(value) for value in selected_ids_value}
        if isinstance(selected_ids_value, list)
        else None
    )
    tasks: list[str] = []
    if manifest_path.is_file():
        manifest = _read_yaml_object(manifest_path)
        episodes = manifest.get("episodes")
        if not isinstance(episodes, list):
            raise FrankaAsyncPolicyContractError(
                f"FastWAM source manifest has no episode list: {manifest_path}"
            )
        matched_selected_ids: set[str] = set()
        for episode in episodes:
            if not isinstance(episode, Mapping):
                continue
            episode_id = str(episode.get("episode_id"))
            if selected_ids is not None and episode_id not in selected_ids:
                continue
            matched_selected_ids.add(episode_id)
            task = episode.get("task_instruction")
            if not isinstance(task, str) or not task.strip():
                raise FrankaAsyncPolicyContractError(
                    f"FastWAM selected episode {episode_id!r} has no task_instruction"
                )
            if task not in tasks:
                tasks.append(task)
        if selected_ids is not None and matched_selected_ids != selected_ids:
            missing_ids = sorted(selected_ids - matched_selected_ids)
            raise FrankaAsyncPolicyContractError(
                "FastWAM source manifest is missing selected checkpoint episodes: "
                f"count={len(missing_ids)}, first={missing_ids[:3]}"
            )

    default_task = task_contract.get("default_instruction")
    if not tasks and isinstance(default_task, str) and default_task.strip():
        tasks.append(default_task)
    invalid = [task for task in tasks if task.startswith("__MISSING_")]
    if not tasks or invalid:
        raise FrankaAsyncPolicyContractError(
            "FastWAM checkpoint has no deployable task instructions; "
            f"manifest={manifest_path}, invalid={invalid}"
        )
    return tuple(tasks)


def inspect_fastwam_checkpoint_contract(
    pretrained_path: str | Path,
    *,
    expected_fps: int | None = None,
    expected_actions_per_chunk: int | None = None,
) -> FrankaFastWAMCheckpointContract:
    """Validate the frozen FastWAM run without importing FastWAM or loading 12 GB of weights."""

    checkpoint = Path(pretrained_path).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".pt" or checkpoint.stat().st_size <= 0:
        raise FrankaAsyncPolicyContractError(
            f"FastWAM pretrained path must be a non-empty local .pt file: {checkpoint}"
        )
    match = FASTWAM_CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None:
        raise FrankaAsyncPolicyContractError(
            f"FastWAM checkpoint name must match step_<number>.pt, got {checkpoint.name!r}"
        )
    if checkpoint.parent.name != "weights" or checkpoint.parent.parent.name != "checkpoints":
        raise FrankaAsyncPolicyContractError(
            "FastWAM checkpoint must live at RUN/checkpoints/weights/step_<number>.pt"
        )
    checkpoint_step = int(match.group(1))
    run_dir = checkpoint.parent.parent.parent
    fastwam_repo = _infer_fastwam_source_checkout(run_dir)
    model_base = Path(
        os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", str(fastwam_repo / "checkpoints"))
    ).expanduser().resolve()
    vae_path = (model_base / FASTWAM_WAN_VAE_RELATIVE_PATH).resolve()
    runtime_path = run_dir / FASTWAM_RUNTIME_CONFIG_NAME
    contract_path = run_dir / FASTWAM_DATASET_CONTRACT_NAME
    run_stats_path = run_dir / FASTWAM_RUN_STATS_NAME
    for label, path in (
        ("runtime config", runtime_path),
        ("dataset contract", contract_path),
        ("run normalization stats", run_stats_path),
    ):
        if not path.is_file():
            raise FrankaAsyncPolicyContractError(f"FastWAM {label} is missing: {path}")

    runtime = _read_yaml_object(runtime_path)
    dataset_contract = _read_json_object(contract_path)
    representation = _mapping(dataset_contract.get("representation"), name="FastWAM representation")
    state_contract = _mapping(representation.get("state"), name="FastWAM state representation")
    action_contract = _mapping(representation.get("action"), name="FastWAM action representation")
    gripper_calibration = _mapping(
        representation.get("gripper_calibration"), name="FastWAM gripper calibration"
    )
    temporal = _mapping(dataset_contract.get("temporal"), name="FastWAM temporal contract")
    image_contract = _mapping(dataset_contract.get("images"), name="FastWAM image contract")
    if state_contract.get("dim") != FASTWAM_STATE_DIM:
        raise FrankaAsyncPolicyContractError("FastWAM state contract must be 8D")
    if state_contract.get("type") != "eef_absolute_axis_angle_with_pseudo_fingers":
        raise FrankaAsyncPolicyContractError("FastWAM state representation type mismatch")
    if state_contract.get("rotation") != "principal_axis_angle_rad" or state_contract.get("frame") != "base":
        raise FrankaAsyncPolicyContractError("FastWAM state rotation/frame contract mismatch")
    finger_scale = float(state_contract.get("finger_scale", float("nan")))
    finger_sign_values = state_contract.get("finger_signs")
    if (
        not np.isfinite(finger_scale)
        or finger_scale != FASTWAM_FINGER_SCALE
        or not isinstance(finger_sign_values, list)
        or tuple(finger_sign_values) != FASTWAM_FINGER_SIGNS
    ):
        raise FrankaAsyncPolicyContractError(
            "FastWAM pseudo-finger contract must be scale=0.04 and signs=(1,-1)"
        )
    finger_signs = (float(finger_sign_values[0]), float(finger_sign_values[1]))

    action_label_mode = _resolve_fastwam_action_label_mode(action_contract)
    if gripper_calibration.get("raw_open") != 0.0 or gripper_calibration.get("raw_closed") != 0.8:
        raise FrankaAsyncPolicyContractError("FastWAM gripper calibration must be raw_open=0.0/raw_closed=0.8")
    if gripper_calibration.get("clip") is not True:
        raise FrankaAsyncPolicyContractError("FastWAM gripper calibration must enable clipping")

    observation_fps = _positive_int(temporal.get("fps"), name="FastWAM temporal.fps")
    num_frames = _positive_int(temporal.get("num_frames"), name="FastWAM temporal.num_frames")
    chunk_size = _positive_int(
        temporal.get("action_horizon"), name="FastWAM temporal.action_horizon"
    )
    if num_frames != chunk_size + 1:
        raise FrankaAsyncPolicyContractError("FastWAM num_frames must equal action_horizon + 1")
    action_video_freq_ratio, num_video_frames, video_fps = _derive_fastwam_video_timing(
        temporal,
        observation_fps=observation_fps,
        action_horizon=chunk_size,
    )
    action_fps = observation_fps
    if expected_fps is not None and expected_fps != observation_fps:
        raise FrankaAsyncPolicyContractError(
            f"FastWAM server/checkpoint fps mismatch: server={expected_fps}, checkpoint={observation_fps}"
        )
    if expected_actions_per_chunk is not None and expected_actions_per_chunk != chunk_size:
        raise FrankaAsyncPolicyContractError(
            "FastWAM server/checkpoint actions_per_chunk mismatch: "
            f"server={expected_actions_per_chunk}, checkpoint={chunk_size}"
        )

    image_height = _positive_int(image_contract.get("height"), name="FastWAM images.height")
    image_width = _positive_int(image_contract.get("width"), name="FastWAM images.width")
    expected_images = {
        "preprocess": "center_crop",
        "concat_multi_camera": "horizontal",
        "ordered_camera_names": ["camera1", "camera2"],
    }
    actual_images = {key: image_contract.get(key) for key in expected_images}
    if actual_images != expected_images:
        raise FrankaAsyncPolicyContractError(
            f"FastWAM image contract mismatch: expected={expected_images}, actual={actual_images}"
        )
    cameras = dataset_contract.get("cameras")
    expected_cameras = [
        {"name": "camera1", "key": FRANKA_CAMERA_KEYS[0], "order": 0},
        {"name": "camera2", "key": FRANKA_CAMERA_KEYS[1], "order": 1},
    ]
    if not isinstance(cameras, list):
        raise FrankaAsyncPolicyContractError("FastWAM dataset contract has no camera list")
    actual_cameras = [
        {key: camera.get(key) for key in ("name", "key", "order")}
        for camera in cameras
        if isinstance(camera, Mapping)
    ]
    if actual_cameras != expected_cameras:
        raise FrankaAsyncPolicyContractError(
            f"FastWAM camera order mismatch: expected={expected_cameras}, actual={actual_cameras}"
        )
    task_instructions = _resolve_fastwam_task_instructions(
        dataset_contract,
        contract_path=contract_path,
    )

    data = _mapping(runtime.get("data"), name="FastWAM runtime.data")
    train = _mapping(data.get("train"), name="FastWAM runtime.data.train")
    model = _mapping(runtime.get("model"), name="FastWAM runtime.model")
    processor = _mapping(train.get("processor"), name="FastWAM runtime processor")
    shape_meta = _mapping(train.get("shape_meta"), name="FastWAM runtime shape_meta")
    if _shape_meta_dimension(shape_meta, "state") != FASTWAM_STATE_DIM:
        raise FrankaAsyncPolicyContractError("FastWAM runtime state dimension mismatch")
    if _shape_meta_dimension(shape_meta, "action") != FASTWAM_ACTION_DIM:
        raise FrankaAsyncPolicyContractError("FastWAM runtime action dimension mismatch")
    runtime_camera_order = [
        item.get("key")
        for item in shape_meta.get("images", [])
        if isinstance(item, Mapping)
    ]
    if runtime_camera_order != ["camera1", "camera2"]:
        raise FrankaAsyncPolicyContractError("FastWAM runtime camera order must be camera1,camera2")
    runtime_video_size = train.get("video_size")
    if runtime_video_size != [image_height, image_width * 2]:
        raise FrankaAsyncPolicyContractError(
            f"FastWAM runtime video_size must be {[image_height, image_width * 2]}"
        )
    if (
        train.get("concat_multi_camera") != "horizontal"
        or train.get("num_frames") != num_frames
        or train.get("action_video_freq_ratio") != action_video_freq_ratio
    ):
        raise FrankaAsyncPolicyContractError("FastWAM runtime temporal/image concatenation mismatch")
    if (
        processor.get("proprio_output_dim") != FASTWAM_STATE_DIM
        or processor.get("action_output_dim") != FASTWAM_ACTION_DIM
        or processor.get("use_stepwise_action_norm") is not False
        or processor.get("norm_default_mode") != "min/max"
        or processor.get("norm_exception_mode") is not None
        or processor.get("action_state_transforms") is not None
    ):
        raise FrankaAsyncPolicyContractError("FastWAM runtime processor normalization contract mismatch")
    model_target = _nonempty_string(model.get("_target_"), name="FastWAM runtime model target")
    if model_target not in FASTWAM_RUNTIME_MODEL_TARGETS:
        raise FrankaAsyncPolicyContractError(
            "FastWAM runtime model target mismatch: "
            f"expected one of {FASTWAM_RUNTIME_MODEL_TARGETS}, actual={model_target!r}"
        )
    video_dit = _mapping(model.get("video_dit_config"), name="FastWAM video_dit_config")
    action_dit = _mapping(model.get("action_dit_config"), name="FastWAM action_dit_config")
    if (
        model_target == FASTWAM_JOINT_RUNTIME_MODEL_TARGET
        and video_dit.get("action_conditioned") is not False
    ):
        raise FrankaAsyncPolicyContractError(
            "FastWAMJoint runtime requires video_dit_config.action_conditioned=false"
        )
    if (
        model.get("proprio_dim") != FASTWAM_STATE_DIM
        or model.get("load_text_encoder") is not False
        or video_dit.get("action_dim") != FASTWAM_ACTION_DIM
        or action_dit.get("action_dim") != FASTWAM_ACTION_DIM
        or video_dit.get("video_attention_mask_mode") != "first_frame_causal"
    ):
        raise FrankaAsyncPolicyContractError("FastWAM runtime model dimension/attention contract mismatch")

    model_dtype = _nonempty_string(runtime.get("mixed_precision"), name="FastWAM mixed_precision")
    if model_dtype != "bf16":
        raise FrankaAsyncPolicyContractError(f"FastWAM runtime must use bf16, got {model_dtype!r}")
    num_inference_steps = _positive_int(
        runtime.get("eval_num_inference_steps"), name="FastWAM eval_num_inference_steps"
    )
    seed = runtime.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FrankaAsyncPolicyContractError(f"FastWAM seed must be a non-negative integer, got {seed!r}")
    context_len = _positive_int(train.get("context_len"), name="FastWAM context_len")
    model_id = _nonempty_string(model.get("model_id"), name="FastWAM model_id")

    configured_stats_path = _configured_path(
        train.get("pretrained_norm_stats"),
        config_path=runtime_path,
        name="FastWAM pretrained_norm_stats",
    )
    if not configured_stats_path.is_file():
        raise FrankaAsyncPolicyContractError(
            f"FastWAM configured normalization stats are missing: {configured_stats_path}"
        )
    # The runtime config points at the exact training-time statistics.  The
    # run-local copy is an audit ledger and must agree, but is not the serving
    # source of truth (it may have been reserialized at a different precision).
    configured_normalizer = _load_fastwam_minmax_normalizer(configured_stats_path)
    run_normalizer = _load_fastwam_minmax_normalizer(run_stats_path)
    _validate_fastwam_stats_action_label(
        configured_stats_path,
        action_label_mode=action_label_mode,
    )
    _validate_fastwam_stats_action_label(
        run_stats_path,
        action_label_mode=action_label_mode,
    )
    for field_name in ("state_min", "state_max", "action_min", "action_max"):
        if not torch.equal(
            getattr(run_normalizer, field_name), getattr(configured_normalizer, field_name)
        ):
            raise FrankaAsyncPolicyContractError(
                f"FastWAM run/configured normalization stats differ for {field_name}"
            )

    text_cache_dir = _configured_path(
        train.get("text_embedding_cache_dir"),
        config_path=runtime_path,
        name="FastWAM text_embedding_cache_dir",
    )
    encoder_id = re.sub(r"[^a-z0-9]+", "", model_id.split("/")[-1].lower()) or "textenc"
    text_context_paths: dict[str, Path] = {}
    for task_instruction in task_instructions:
        prompt = FASTWAM_PROMPT_TEMPLATE.format(task=task_instruction)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        text_context_path = text_cache_dir / f"{prompt_hash}.t5_len{context_len}.{encoder_id}.pt"
        if not text_context_path.is_file():
            raise FrankaAsyncPolicyContractError(
                f"FastWAM cached text context is missing for task {task_instruction!r}: "
                f"{text_context_path}"
            )
        text_context_paths[task_instruction] = text_context_path

    return FrankaFastWAMCheckpointContract(
        checkpoint_path=checkpoint,
        run_dir=run_dir,
        runtime_config_path=runtime_path,
        stats_path=configured_stats_path,
        text_context_path=text_context_paths[task_instructions[0]],
        vae_path=vae_path,
        task_instruction=task_instructions[0],
        observation_fps=observation_fps,
        action_fps=action_fps,
        chunk_size=chunk_size,
        action_label_mode=action_label_mode,
        action_video_freq_ratio=action_video_freq_ratio,
        num_video_frames=num_video_frames,
        video_fps=video_fps,
        image_height=image_height,
        image_width=image_width,
        context_len=context_len,
        model_id=model_id,
        model_target=model_target,
        checkpoint_step=checkpoint_step,
        model_dtype=model_dtype,
        num_inference_steps=num_inference_steps,
        seed=seed,
        finger_scale=finger_scale,
        finger_signs=finger_signs,
        text_context_paths=text_context_paths,
        task_instructions=task_instructions,
    )


def validate_franka_server_config(config: PolicyServerConfig) -> FrankaServingContract:
    """Validate CLI inputs against the selected checkpoint's own serving contract."""

    if config.host not in {"localhost", "127.0.0.1"}:
        raise ValueError(
            "Franka async server must bind to a loopback host because its transport is a trusted boundary"
        )
    if config.policy_type not in FRANKA_POLICY_TYPES:
        raise ValueError(
            f"Franka async server requires --policy_type in {FRANKA_POLICY_TYPES}, "
            f"got {config.policy_type!r}"
        )
    if config.fastwam_joint_video_inference and config.policy_type != "fastwam":
        raise ValueError("--fastwam_joint_video_inference requires --policy_type=fastwam")
    if config.fastwam_joint_video_output_dir is not None:
        if config.policy_type != "fastwam":
            raise ValueError("--fastwam_joint_video_output_dir requires --policy_type=fastwam")
        if not config.fastwam_joint_video_inference:
            raise ValueError(
                "--fastwam_joint_video_output_dir requires --fastwam_joint_video_inference=true"
            )
    if config.pretrained_name_or_path is None:
        raise ValueError("Franka async server requires --pretrained_name_or_path")
    if config.actions_per_chunk is None:
        raise ValueError("Franka async server requires --actions_per_chunk")
    if config.policy_device is None:
        raise ValueError("Franka async server requires --policy_device")

    inspector = (
        inspect_franka_checkpoint_contract
        if config.policy_type == "pi0"
        else inspect_fastwam_checkpoint_contract
    )
    return inspector(
        config.pretrained_name_or_path,
        expected_fps=config.fps,
        expected_actions_per_chunk=config.actions_per_chunk,
    )


def _validate_franka_checkpoint(
    pretrained_path: str | Path,
    *,
    expected_fps: int | None = None,
    expected_actions_per_chunk: int | None = None,
) -> tuple[Path, FrankaCheckpointContract]:
    """Validate the project checkpoint envelope before allocating the PI0 model.

    The strict tensor namespace, graph, missing/unexpected keys, and a numerical
    tensor identity check are enforced later by
    :func:`load_pi0_full_checkpoint_weights`. This preflight rejects non-Franka
    or incomplete checkpoint directories before the multi-billion parameter
    model is constructed. Small metadata/configuration files retain their hash
    checks, while the multi-gigabyte model file is checked during strict load.
    """

    checkpoint_dir = Path(pretrained_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FrankaAsyncPolicyContractError(
            f"Franka pretrained path must be a local directory: {checkpoint_dir}"
        )

    manifest_path = checkpoint_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FrankaAsyncPolicyContractError(f"Missing Franka checkpoint manifest: {manifest_path}")
    manifest = _read_json_object(manifest_path)

    file_manifest = manifest.get("files")
    if not isinstance(file_manifest, dict):
        raise FrankaAsyncPolicyContractError("Franka checkpoint manifest has no file ledger")

    tracked_paths: dict[str, tuple[Path, str]] = {}
    for name in _MANIFEST_TRACKED_FILES:
        path = checkpoint_dir / name
        if not path.is_file():
            raise FrankaAsyncPolicyContractError(f"Franka checkpoint is missing required file: {path}")
        record = file_manifest.get(name)
        if not isinstance(record, dict):
            raise FrankaAsyncPolicyContractError(f"Franka checkpoint manifest does not track {name}")
        declared_size = record.get("size_bytes")
        if not isinstance(declared_size, int) or declared_size <= 0:
            raise FrankaAsyncPolicyContractError(f"Invalid declared size for checkpoint file {name}")
        if path.stat().st_size != declared_size:
            raise FrankaAsyncPolicyContractError(
                f"Checkpoint file size mismatch for {name}: "
                f"declared={declared_size}, actual={path.stat().st_size}"
            )
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise FrankaAsyncPolicyContractError(f"Invalid SHA-256 ledger entry for {name}")
        tracked_paths[name] = (path, digest)

    metadata_names = tuple(name for name in _MANIFEST_TRACKED_FILES if name != "model.safetensors")
    for name in metadata_names:
        path, digest = tracked_paths[name]
        actual_digest = _sha256_file(path)
        if actual_digest != digest:
            raise FrankaAsyncPolicyContractError(
                f"Checkpoint file SHA-256 mismatch for {name}: declared={digest}, actual={actual_digest}"
            )

    _validate_processor_state_files(checkpoint_dir, "policy_preprocessor.json")
    _validate_processor_state_files(checkpoint_dir, "policy_postprocessor.json")
    contract = _validate_geometry_and_stats_manifests(
        checkpoint_dir,
        expected_fps=expected_fps,
        expected_actions_per_chunk=expected_actions_per_chunk,
    )

    return checkpoint_dir, contract


def validate_franka_checkpoint(
    pretrained_path: str | Path,
    *,
    expected_fps: int | None = None,
    expected_actions_per_chunk: int | None = None,
) -> Path:
    """Validate a Franka checkpoint and return its resolved local directory."""

    checkpoint_dir, _ = _validate_franka_checkpoint(
        pretrained_path,
        expected_fps=expected_fps,
        expected_actions_per_chunk=expected_actions_per_chunk,
    )
    return checkpoint_dir


def _quaternion_chunk_with_anchor_continuity(
    anchor_rotation: np.ndarray,
    target_rotations: np.ndarray,
) -> np.ndarray:
    anchor_quaternion = matrix_to_quaternion_xyzw(anchor_rotation).reshape(1, 4)
    target_quaternions = matrix_to_quaternion_xyzw(target_rotations).reshape(-1, 4)
    trajectory = np.concatenate((anchor_quaternion, target_quaternions), axis=0)
    return enforce_quaternion_continuity(trajectory)[1:]


def _decode_pi0_delta_action_chunk(anchor_state: np.ndarray, relative_chunk: torch.Tensor) -> torch.Tensor:
    if anchor_state.shape != (STATE_DIM,):
        raise FrankaAsyncPolicyContractError(
            f"Franka anchor state must have shape ({STATE_DIM},), got {anchor_state.shape}"
        )
    if not np.all(np.isfinite(anchor_state)):
        raise FrankaAsyncPolicyContractError("Franka anchor state contains NaN or Inf")
    if relative_chunk.ndim != 2 or tuple(relative_chunk.shape[1:]) != (ACTION_DIM,):
        raise FrankaAsyncPolicyContractError(
            f"Relative action chunk must have shape [K,{ACTION_DIM}], got {tuple(relative_chunk.shape)}"
        )
    if relative_chunk.shape[0] <= 0:
        raise FrankaAsyncPolicyContractError("Relative action chunk must not be empty")
    if not relative_chunk.is_floating_point() or not bool(torch.isfinite(relative_chunk).all()):
        raise FrankaAsyncPolicyContractError("Relative action chunk must be finite floating point")

    try:
        anchor_rotation = rotation_6d_to_matrix(anchor_state[3:9])
        relative_numpy = relative_chunk.detach().to(device="cpu", dtype=torch.float64).numpy()
        position, rotation, gripper = decode_relative_action(
            anchor_state[:3],
            anchor_rotation,
            relative_numpy,
        )
        gripper = np.clip(gripper, 0.0, 1.0)
        quaternion = _quaternion_chunk_with_anchor_continuity(anchor_rotation, rotation)
        absolute = np.concatenate((position, quaternion, gripper), axis=-1)
    except (TypeError, ValueError) as exc:
        raise FrankaAsyncPolicyContractError("Failed to decode the relative Franka action chunk") from exc

    expected_shape = (relative_chunk.shape[0], ABSOLUTE_ACTION_DIM)
    if absolute.shape != expected_shape:
        raise FrankaAsyncPolicyContractError(
            f"Absolute action chunk must have shape {expected_shape}, got {absolute.shape}"
        )
    if not np.all(np.isfinite(absolute)):
        raise FrankaAsyncPolicyContractError("Absolute action chunk contains NaN or Inf")
    return torch.as_tensor(absolute, dtype=torch.float32, device="cpu")


def _decode_absolute_eef_action_chunk(
    anchor_state: np.ndarray,
    absolute_chunk: torch.Tensor,
    *,
    gripper_encoding: str,
) -> torch.Tensor:
    """Decode absolute7 pose targets into canonical absolute8 client actions."""

    state = np.asarray(anchor_state, dtype=np.float64)
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise FrankaAsyncPolicyContractError(
            f"Franka anchor state must be finite shape ({STATE_DIM},), got {state.shape}"
        )
    action = torch.as_tensor(absolute_chunk)
    if (
        action.ndim != 2
        or action.shape[0] <= 0
        or action.shape[1] != ACTION_DIM
        or not action.is_floating_point()
        or not bool(torch.isfinite(action).all())
    ):
        raise FrankaAsyncPolicyContractError(
            f"Absolute EEF action must be finite floating [K,{ACTION_DIM}], got {tuple(action.shape)}"
        )
    values = action.detach().to(device="cpu", dtype=torch.float64).numpy()
    try:
        anchor_rotation = rotation_6d_to_matrix(state[3:9])
        positions, rotations, gripper = decode_absolute_action(values)
        quaternion = _quaternion_chunk_with_anchor_continuity(
            anchor_rotation,
            rotations,
        )
    except (TypeError, ValueError) as exc:
        raise FrankaAsyncPolicyContractError("Failed to decode the absolute EEF action chunk") from exc

    gripper = np.clip(gripper, 0.0, 1.0)
    if gripper_encoding == "open_0_1":
        gripper_closed = 1.0 - gripper
    elif gripper_encoding == "closed_0_1":
        gripper_closed = gripper
    else:
        raise FrankaAsyncPolicyContractError(
            f"Unsupported absolute EEF gripper encoding {gripper_encoding!r}"
        )
    absolute = np.concatenate((positions, quaternion, gripper_closed), axis=-1)
    expected_shape = (len(values), ABSOLUTE_ACTION_DIM)
    if absolute.shape != expected_shape or not np.all(np.isfinite(absolute)):
        raise FrankaAsyncPolicyContractError("Decoded absolute EEF action chunk is invalid")
    return torch.as_tensor(absolute, dtype=torch.float32, device="cpu")


def _center_crop_resize_fastwam_image(
    image: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != FRANKA_CAMERA_SHAPE:
        raise FrankaAsyncPolicyContractError(
            f"FastWAM camera must have shape {FRANKA_CAMERA_SHAPE}, got {value.shape}"
        )
    if value.dtype != np.uint8:
        raise FrankaAsyncPolicyContractError(f"FastWAM camera must have dtype uint8, got {value.dtype}")
    source_height, source_width = value.shape[:2]
    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    pil = Image.fromarray(value)
    if source_aspect > target_aspect:
        crop_width = int(round(source_height * target_aspect))
        left = (source_width - crop_width) // 2
        pil = pil.crop((left, 0, left + crop_width, source_height))
    else:
        crop_height = int(round(source_width / target_aspect))
        top = (source_height - crop_height) // 2
        pil = pil.crop((0, top, source_width, top + crop_height))
    pil = pil.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))


def _prepare_fastwam_image(
    camera1: np.ndarray,
    camera2: np.ndarray,
    *,
    target_height: int = 224,
    target_width: int = 224,
) -> torch.Tensor:
    """Reproduce conversion-time center-crop and FastWAM horizontal camera concatenation."""

    processed = [
        _center_crop_resize_fastwam_image(
            image,
            target_height=target_height,
            target_width=target_width,
        )
        for image in (camera1, camera2)
    ]
    combined = np.concatenate(processed, axis=1)
    tensor = torch.from_numpy(combined).permute(2, 0, 1).to(dtype=torch.float32)
    return tensor.mul_(2.0 / 255.0).sub_(1.0).unsqueeze(0).contiguous()


def _save_fastwam_joint_video(
    video: list[Image.Image],
    *,
    output_dir: str,
    fps: float,
    timestep: int,
) -> Path:
    """Write one decoded FastWAM joint-inference video as an MP4."""

    if not video or not all(isinstance(frame, Image.Image) for frame in video):
        raise FrankaAsyncPolicyContractError(
            "FastWAM infer_joint video must be a non-empty list of PIL images"
        )
    destination = (
        Path(output_dir).expanduser().resolve()
        / f"fastwam_joint_t{timestep:09d}_{time.time_ns()}.mp4"
    )
    try:
        from fastwam.utils.video_io import save_mp4

        save_mp4(video, str(destination), fps=fps)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise FrankaAsyncPolicyContractError(
            f"Could not save FastWAM joint video to {destination}"
        ) from exc
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise FrankaAsyncPolicyContractError(f"FastWAM joint video was not written: {destination}")
    return destination


def _build_fastwam_state(
    state10: np.ndarray,
    *,
    finger_scale: float = FASTWAM_FINGER_SCALE,
    finger_signs: tuple[float, float] = FASTWAM_FINGER_SIGNS,
) -> np.ndarray:
    """Convert canonical Franka state10 to FastWAM's frozen 8D proprio representation."""

    state = np.asarray(state10, dtype=np.float64)
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise FrankaAsyncPolicyContractError(
            f"Franka state must be finite shape ({STATE_DIM},), got {state.shape}"
        )
    closed = float(state[-1])
    if closed < 0.0 or closed > 1.0:
        raise FrankaAsyncPolicyContractError(
            f"Franka gripper.closed_0_1 must be in [0,1], got {closed}"
        )
    if not np.isfinite(finger_scale) or finger_scale <= 0.0:
        raise FrankaAsyncPolicyContractError("FastWAM finger_scale must be positive and finite")
    signs = np.asarray(finger_signs, dtype=np.float64)
    if signs.shape != (2,) or not np.all(np.isfinite(signs)):
        raise FrankaAsyncPolicyContractError("FastWAM finger_signs must contain two finite values")
    try:
        rotation = rotation_6d_to_matrix(state[3:9])
        rotation_vector = so3_log(rotation)
    except ValueError as exc:
        raise FrankaAsyncPolicyContractError("Could not decode FastWAM anchor rotation") from exc
    open_value = 1.0 - closed
    fingers = finger_scale * open_value * signs
    result = np.concatenate((state[:3], rotation_vector, fingers))
    if result.shape != (FASTWAM_STATE_DIM,) or not np.all(np.isfinite(result)):
        raise FrankaAsyncPolicyContractError("Constructed FastWAM state is invalid")
    return result


def _decode_fastwam_absolute_action_chunk(
    anchor_state: np.ndarray,
    action_chunk: torch.Tensor,
    *,
    action_label_mode: str = ACTION_LABEL_MODE_DELTA_EEF,
) -> torch.Tensor:
    """Decode a mode-aware FastWAM action chunk into canonical absolute8 actions."""

    if action_label_mode == ACTION_LABEL_MODE_ABSOLUTE_EEF:
        return _decode_absolute_eef_action_chunk(
            anchor_state,
            action_chunk,
            gripper_encoding="open_0_1",
        )
    if action_label_mode != ACTION_LABEL_MODE_DELTA_EEF:
        raise FrankaAsyncPolicyContractError(f"Unsupported FastWAM action_label_mode {action_label_mode!r}")

    state = np.asarray(anchor_state, dtype=np.float64)
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise FrankaAsyncPolicyContractError(
            f"FastWAM anchor state must be finite shape ({STATE_DIM},), got {state.shape}"
        )
    action = torch.as_tensor(action_chunk, device="cpu")
    if (
        action.ndim != 2
        or action.shape[0] <= 0
        or action.shape[1] != FASTWAM_ACTION_DIM
        or not action.is_floating_point()
        or not bool(torch.isfinite(action).all())
    ):
        raise FrankaAsyncPolicyContractError(
            f"FastWAM adjacent action must be finite floating [K,{FASTWAM_ACTION_DIM}], "
            f"got {tuple(action.shape)}"
        )
    values = action.to(dtype=torch.float64).numpy()
    try:
        position = state[:3].copy()
        rotation = rotation_6d_to_matrix(state[3:9])
        positions = np.empty((len(values), 3), dtype=np.float64)
        rotations = np.empty((len(values), 3, 3), dtype=np.float64)
        for index, row in enumerate(values):
            position = position + row[:3]
            rotation = rotation @ so3_exp(row[3:6])
            positions[index] = position
            rotations[index] = rotation
        quaternion = _quaternion_chunk_with_anchor_continuity(
            rotation_6d_to_matrix(state[3:9]),
            rotations,
        )
    except ValueError as exc:
        raise FrankaAsyncPolicyContractError("Failed to integrate FastWAM adjacent actions") from exc

    # The seventh FastWAM dimension is an absolute open target, while the
    # canonical ROS/client boundary is absolute closed_0_1.
    gripper_closed = 1.0 - np.clip(values[:, 6:7], 0.0, 1.0)
    absolute = np.concatenate((positions, quaternion, gripper_closed), axis=-1)
    if absolute.shape != (len(values), ABSOLUTE_ACTION_DIM) or not np.all(np.isfinite(absolute)):
        raise FrankaAsyncPolicyContractError("Decoded FastWAM absolute action chunk is invalid")
    return torch.as_tensor(absolute, dtype=torch.float32, device="cpu")


def _load_fastwam_text_context(
    path: Path,
    *,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise FrankaAsyncPolicyContractError(
            f"Could not load FastWAM cached text context: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FrankaAsyncPolicyContractError("FastWAM text context payload must be a mapping")
    context_value = payload.get("context")
    mask_value = payload.get("mask")
    if not isinstance(context_value, torch.Tensor) or not isinstance(mask_value, torch.Tensor):
        raise FrankaAsyncPolicyContractError(
            "FastWAM text context payload must contain context/mask tensors"
        )
    if context_value.dtype != torch.bfloat16 or mask_value.dtype != torch.bool:
        raise FrankaAsyncPolicyContractError(
            "FastWAM text context must use bfloat16 context and bool source mask"
        )
    context = context_value.detach().clone().to(device="cpu")
    source_mask = mask_value.detach().clone().to(device="cpu")
    expected_context_shape = (context_len, FASTWAM_CONTEXT_WIDTH)
    if tuple(context.shape) != expected_context_shape or not bool(torch.isfinite(context.float()).all()):
        raise FrankaAsyncPolicyContractError(
            f"FastWAM context must be finite shape {expected_context_shape}, got {tuple(context.shape)}"
        )
    if tuple(source_mask.shape) != (context_len,):
        raise FrankaAsyncPolicyContractError(
            f"FastWAM context mask must have shape ({context_len},), "
            f"got {tuple(source_mask.shape)}"
        )
    # RobotVideoDataset zeroed padding tokens but then exposed an all-true mask
    # during training to match Wan2.2.  Deployment must preserve both details.
    context[~source_mask] = 0.0
    return context, torch.ones_like(source_mask)


def _load_fastwam_runtime(
    contract: FrankaFastWAMCheckpointContract,
    *,
    device: str,
) -> _FastWAMRuntime:
    """Lazily reconstruct FastWAM so PI0-only environments never import its stack."""

    fastwam_repo = _infer_fastwam_source_checkout(contract.run_dir)
    source_roots = (fastwam_repo / "src", fastwam_repo / "franka_project" / "src")
    for source_root in source_roots:
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    try:
        fastwam_package = importlib.import_module("fastwam")
        loaded_source_root = Path(fastwam_package.__file__).resolve().parent
    except (AttributeError, ImportError, TypeError) as exc:
        raise FrankaAsyncPolicyContractError("Could not import the FastWAM source tree") from exc
    if loaded_source_root != fastwam_repo / "src" / "fastwam":
        raise FrankaAsyncPolicyContractError(
            "Imported FastWAM package does not match the source tree inferred from the checkpoint: "
            f"expected={fastwam_repo / 'src' / 'fastwam'}, actual={loaded_source_root}"
        )

    model_base = Path(
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(fastwam_repo / "checkpoints"))
    ).expanduser().resolve()
    if not model_base.is_dir():
        raise FrankaAsyncPolicyContractError(
            f"DIFFSYNTH_MODEL_BASE_PATH does not exist: {model_base}"
        )
    configured_vae = (model_base / FASTWAM_WAN_VAE_RELATIVE_PATH).resolve()
    if configured_vae != contract.vae_path:
        raise FrankaAsyncPolicyContractError(
            "DIFFSYNTH_MODEL_BASE_PATH changed after FastWAM deployment validation"
        )
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise FrankaAsyncPolicyContractError(
            "FastWAM dependencies are unavailable. Run this backend in a dedicated serving "
            "environment containing FastWAM, Hydra, OmegaConf, and a compatible CUDA PyTorch."
        ) from exc

    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise FrankaAsyncPolicyContractError(f"Invalid FastWAM policy device: {device!r}") from exc
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise FrankaAsyncPolicyContractError("FastWAM serving requires an available CUDA device")

    normalizer = _load_fastwam_minmax_normalizer(contract.stats_path)
    contexts_by_task = {
        task: _load_fastwam_text_context(path, context_len=contract.context_len)
        for task, path in (
            contract.text_context_paths
            or {contract.task_instruction: contract.text_context_path}
        ).items()
    }
    try:
        runtime_config = OmegaConf.load(contract.runtime_config_path)
        model_config = OmegaConf.create(OmegaConf.to_container(runtime_config.model, resolve=True))
        model_config.skip_dit_load_from_pretrain = True
        model_config.action_dit_pretrained_path = None
        model = instantiate(
            model_config,
            model_dtype=torch.bfloat16,
            device=str(torch_device),
        )
        payload = torch.load(
            contract.checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        raise FrankaAsyncPolicyContractError(
            f"Could not reconstruct FastWAM from {contract.runtime_config_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FrankaAsyncPolicyContractError("FastWAM checkpoint root must be a mapping")
    if payload.get("step") != contract.checkpoint_step:
        raise FrankaAsyncPolicyContractError(
            "FastWAM checkpoint filename and embedded step disagree: "
            f"filename={contract.checkpoint_step}, embedded={payload.get('step')!r}"
        )
    expected_saved_dtype = {"bf16": "torch.bfloat16"}[contract.model_dtype]
    if payload.get("torch_dtype") != expected_saved_dtype:
        raise FrankaAsyncPolicyContractError(
            "FastWAM checkpoint/runtime dtype mismatch: "
            f"runtime={contract.model_dtype!r}, checkpoint={payload.get('torch_dtype')!r}"
        )
    mot = payload.get("mot")
    proprio = payload.get("proprio_encoder")
    if not isinstance(mot, Mapping) or not isinstance(proprio, Mapping):
        raise FrankaAsyncPolicyContractError("FastWAM checkpoint lacks mot/proprio_encoder mappings")
    try:
        model.mot.load_state_dict(mot, strict=True)
        model.proprio_encoder.load_state_dict(proprio, strict=True)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise FrankaAsyncPolicyContractError("FastWAM checkpoint strict tensor load failed") from exc
    del payload
    gc.collect()
    model.eval()
    model.requires_grad_(False)
    if bool(getattr(model, "training", True)):
        raise FrankaAsyncPolicyContractError("FastWAM model remained in training mode after eval()")
    if int(getattr(model, "proprio_dim", -1)) != FASTWAM_STATE_DIM:
        raise FrankaAsyncPolicyContractError("Loaded FastWAM model proprio dimension mismatch")
    if int(getattr(getattr(model, "action_expert", None), "action_dim", -1)) != FASTWAM_ACTION_DIM:
        raise FrankaAsyncPolicyContractError("Loaded FastWAM model action dimension mismatch")
    return _FastWAMRuntime(
        model=model,
        normalizer=normalizer,
        contexts_by_task=contexts_by_task,
    )


def _reject_remote_policy_conflicts(
    server_config: PolicyServerConfig,
    client_specs: RemotePolicyConfig,
    *,
    expected_policy_type: str,
) -> None:
    # Inspect the instance state, not dataclass class defaults: when an older
    # pickle is decoded by newer code, a newly added default can otherwise make
    # the obsolete payload look current.
    client_state = vars(client_specs)
    protocol_version = client_state.get("protocol_version")
    if protocol_version != ASYNC_INFERENCE_PROTOCOL_VERSION:
        raise FrankaAsyncPolicyContractError(
            "Franka client/server async protocol mismatch: "
            f"server={ASYNC_INFERENCE_PROTOCOL_VERSION}, client={protocol_version!r}; "
            "synchronize both checkouts before deployment"
        )
    client_fps = client_state.get("fps")
    if client_fps != server_config.fps:
        raise FrankaAsyncPolicyContractError(
            "Franka client/server fps mismatch: "
            f"server={server_config.fps}, client={client_fps!r}"
        )
    if client_specs.policy_type is not None and client_specs.policy_type != expected_policy_type:
        raise FrankaAsyncPolicyContractError(
            "Franka client/server policy_type mismatch: "
            f"server={expected_policy_type!r}, client={client_specs.policy_type!r}"
        )
    if (
        server_config.actions_per_chunk is not None
        and client_specs.actions_per_chunk is not None
        and client_specs.actions_per_chunk != server_config.actions_per_chunk
    ):
        raise FrankaAsyncPolicyContractError(
            "Franka client/server actions_per_chunk mismatch: "
            f"server={server_config.actions_per_chunk}, client={client_specs.actions_per_chunk}"
        )


def _validate_franka_remote_policy_specs(
    policy_specs: RemotePolicyConfig,
    *,
    expected_policy_type: str,
    expected_rename_map: Mapping[str, str],
) -> None:
    if policy_specs.policy_type != expected_policy_type:
        raise FrankaAsyncPolicyContractError(
            f"Franka {expected_policy_type} server resolved unexpected policy_type "
            f"{policy_specs.policy_type!r}"
        )
    features = policy_specs.lerobot_features
    expected_keys = {OBS_STATE, *FRANKA_CAMERA_KEYS}
    if set(features) != expected_keys:
        raise FrankaAsyncPolicyContractError(
            "Franka client observation feature keys do not match the frozen contract: "
            f"expected={sorted(expected_keys)}, actual={sorted(features)}"
        )

    state_feature = features[OBS_STATE]
    expected_state = {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": list(FRANKA_STATE_NAMES),
    }
    actual_state = {
        "dtype": state_feature.get("dtype"),
        "shape": tuple(state_feature.get("shape", ())),
        "names": state_feature.get("names"),
    }
    if actual_state != expected_state:
        raise FrankaAsyncPolicyContractError(
            "Franka client state feature does not match the frozen name/order contract: "
            f"expected={expected_state}, actual={actual_state}"
        )

    for camera_key in FRANKA_CAMERA_KEYS:
        camera_feature = features[camera_key]
        expected_camera = {
            "dtype": "image",
            "shape": FRANKA_CAMERA_SHAPE,
            "names": ["height", "width", "channels"],
        }
        actual_camera = {
            "dtype": camera_feature.get("dtype"),
            "shape": tuple(camera_feature.get("shape", ())),
            "names": camera_feature.get("names"),
        }
        if actual_camera != expected_camera:
            raise FrankaAsyncPolicyContractError(
                f"Franka client camera feature {camera_key!r} does not match the frozen contract: "
                f"expected={expected_camera}, actual={actual_camera}"
            )

    expected_map = dict(expected_rename_map)
    if policy_specs.rename_map != expected_map:
        raise FrankaAsyncPolicyContractError(
            f"Franka client rename_map mismatch for {expected_policy_type}: "
            f"expected={expected_map}, actual={policy_specs.rename_map}"
        )


class FrankaPI0PolicyServer(PolicyServer):
    """LeRobot async server with strict Franka PI0 preprocessing, load, and output adapters."""

    prefix = "franka_pi0_policy_server"

    def __init__(self, config: PolicyServerConfig):
        super().__init__(config)
        self._checkpoint_contract: FrankaCheckpointContract | None = None

    def _resolve_policy_specs(self, client_specs: RemotePolicyConfig) -> RemotePolicyConfig:
        _reject_remote_policy_conflicts(
            self.config,
            client_specs,
            expected_policy_type="pi0",
        )
        policy_specs = super()._resolve_policy_specs(client_specs)
        _validate_franka_remote_policy_specs(
            policy_specs,
            expected_policy_type="pi0",
            expected_rename_map=FRANKA_PI0_RENAME_MAP,
        )
        return policy_specs

    def _load_policy(self, policy_type: str, pretrained_name_or_path: str) -> Any:
        if policy_type != "pi0":
            raise FrankaAsyncPolicyContractError(
                f"Franka async server only accepts policy_type='pi0', got {policy_type!r}"
            )
        if self.device is None:
            raise FrankaAsyncPolicyContractError("Policy device is unset before Franka checkpoint load")
        if self.actions_per_chunk is None or self.actions_per_chunk <= 0:
            raise FrankaAsyncPolicyContractError("actions_per_chunk is unset before Franka checkpoint load")

        checkpoint_dir, contract = _validate_franka_checkpoint(
            pretrained_name_or_path,
            expected_fps=self.config.fps,
            expected_actions_per_chunk=self.actions_per_chunk,
        )
        # Keep PI0 imports out of FastWAM-only processes.  The two policy
        # stacks intentionally use different Transformer/Hugging Face
        # dependency generations.
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        from .pi0_training import (
            build_pi0_full_finetune_config,
            canonicalize_pi0_full_training_graph,
            load_pi0_full_checkpoint_weights,
        )

        config = build_pi0_full_finetune_config(checkpoint_dir, device=self.device)
        policy = PI0Policy(config)
        canonicalize_pi0_full_training_graph(policy)
        load_report = load_pi0_full_checkpoint_weights(policy, checkpoint_dir)
        if load_report.get("project_manifest_present") is not True:
            raise FrankaAsyncPolicyContractError(
                "Strict loader did not observe the Franka checkpoint manifest"
            )

        policy.eval()
        if policy.training:
            raise FrankaAsyncPolicyContractError("Franka PI0 policy remained in training mode after eval()")
        self._checkpoint_contract = contract
        self.logger.info(
            "Strictly loaded Franka PI0 checkpoint from %s | "
            "profile=%s task=%r fps=%s chunk_size=%s action_label_mode=%s",
            checkpoint_dir,
            contract.profile,
            contract.task_instruction,
            contract.observation_fps,
            contract.chunk_size,
            contract.action_label_mode,
        )
        return policy

    def _extract_anchor_state(self, observation_t: TimedObservation) -> np.ndarray:
        if self.lerobot_features is None:
            raise FrankaAsyncPolicyContractError("LeRobot observation features are unset")
        try:
            observation = make_lerobot_observation(
                observation_t.get_observation(),
                self.lerobot_features,
            )
            state = np.asarray(observation[OBS_STATE], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise FrankaAsyncPolicyContractError(
                "Could not reconstruct the Franka anchor state from the raw observation"
            ) from exc

        if state.shape != (STATE_DIM,):
            raise FrankaAsyncPolicyContractError(
                f"Raw Franka observation must reconstruct a ({STATE_DIM},) state, got {state.shape}"
            )
        if not np.all(np.isfinite(state)):
            raise FrankaAsyncPolicyContractError("Raw Franka observation state contains NaN or Inf")
        return state

    def _prepare_observation(self, observation_t: TimedObservation) -> Observation:
        """Reproduce the PI0 training image path without distorting camera geometry.

        Training converts the original 480x640 images to float32 in [0, 1] and
        then lets PI0 resize them with aspect-ratio-preserving black padding.
        The stock async helper instead stretches every camera directly to the
        policy resolution, so this adapter performs PI0's exact resize here.
        """
        if self.lerobot_features is None:
            raise FrankaAsyncPolicyContractError("LeRobot observation features are unset")

        # See _load_policy(): FastWAM serving must not import PI0's model stack.
        from lerobot.policies.pi0.modeling_pi0 import resize_with_pad_torch

        raw_observation = observation_t.get_observation()
        contract = getattr(self, "_checkpoint_contract", None)
        if contract is not None:
            task = raw_observation.get("task")
            if not isinstance(task, str) or task != contract.task_instruction:
                raise FrankaAsyncPolicyContractError(
                    "Franka observation task does not match the selected checkpoint: "
                    f"expected={contract.task_instruction!r}, actual={task!r}"
                )
        try:
            lerobot_observation = make_lerobot_observation(
                raw_observation,
                self.lerobot_features,
            )
            state = np.asarray(lerobot_observation[OBS_STATE], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise FrankaAsyncPolicyContractError(
                "Could not reconstruct the Franka policy observation"
            ) from exc

        if state.shape != (STATE_DIM,):
            raise FrankaAsyncPolicyContractError(
                f"Raw Franka observation must reconstruct a ({STATE_DIM},) state, got {state.shape}"
            )
        if not np.all(np.isfinite(state)):
            raise FrankaAsyncPolicyContractError("Raw Franka observation state contains NaN or Inf")

        prepared: Observation = {
            OBS_STATE: torch.from_numpy(np.array(state, copy=True)).unsqueeze(0),
        }
        for camera_key, policy_key in FRANKA_RENAME_MAP.items():
            try:
                image = np.asarray(lerobot_observation[camera_key])
                target_shape = tuple(self.policy_image_features[policy_key].shape)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise FrankaAsyncPolicyContractError(
                    f"Could not prepare Franka camera {camera_key!r} for PI0 feature {policy_key!r}"
                ) from exc

            if image.shape != FRANKA_CAMERA_SHAPE:
                raise FrankaAsyncPolicyContractError(
                    f"Raw Franka camera {camera_key!r} must have shape {FRANKA_CAMERA_SHAPE}, "
                    f"got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise FrankaAsyncPolicyContractError(
                    f"Raw Franka camera {camera_key!r} must have dtype uint8, got {image.dtype}"
                )
            if len(target_shape) != 3 or target_shape[0] != 3:
                raise FrankaAsyncPolicyContractError(
                    f"PI0 image feature {policy_key!r} must be CHW with three channels, got {target_shape}"
                )

            image_tensor = (
                torch.from_numpy(np.array(image, copy=True))
                .permute(2, 0, 1)
                .to(dtype=torch.float32)
                .div_(255.0)
                .unsqueeze(0)
            )
            prepared[policy_key] = resize_with_pad_torch(
                image_tensor,
                target_shape[1],
                target_shape[2],
            ).contiguous()

        if contract is not None:
            prepared["task"] = contract.task_instruction
        elif "task" in raw_observation:
            prepared["task"] = raw_observation["task"]
        return prepared

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        timed_actions = super()._predict_action_chunk(observation_t)
        if self.actions_per_chunk is None or self.actions_per_chunk <= 0:
            raise FrankaAsyncPolicyContractError("actions_per_chunk is unset or invalid")
        if len(timed_actions) != self.actions_per_chunk:
            raise FrankaAsyncPolicyContractError(
                f"PI0 returned {len(timed_actions)} actions; expected {self.actions_per_chunk}"
            )

        actions: list[torch.Tensor] = []
        for index, timed_action in enumerate(timed_actions):
            action = timed_action.get_action()
            if not isinstance(action, torch.Tensor) or tuple(action.shape) != (ACTION_DIM,):
                shape = tuple(action.shape) if isinstance(action, torch.Tensor) else type(action).__name__
                raise FrankaAsyncPolicyContractError(
                    f"Model action {index} must be a ({ACTION_DIM},) tensor, got {shape}"
                )
            actions.append(action)

        model_action_chunk = torch.stack(actions)
        anchor_state = self._extract_anchor_state(observation_t)
        contract = self._checkpoint_contract
        if contract is None:
            raise FrankaAsyncPolicyContractError("PI0 checkpoint contract is unset")
        if contract.action_label_mode == ACTION_LABEL_MODE_DELTA_EEF:
            absolute_chunk = _decode_pi0_delta_action_chunk(
                anchor_state,
                model_action_chunk,
            )
        elif contract.action_label_mode == ACTION_LABEL_MODE_ABSOLUTE_EEF:
            absolute_chunk = _decode_absolute_eef_action_chunk(
                anchor_state,
                model_action_chunk,
                gripper_encoding="closed_0_1",
            )
        else:
            raise FrankaAsyncPolicyContractError(
                f"Unsupported PI0 action_label_mode {contract.action_label_mode!r}"
            )

        # Mutate only after the whole chunk has passed shape, finite, and geometry
        # validation so callers never observe a partially decoded queue.
        for timed_action, absolute_action in zip(timed_actions, absolute_chunk, strict=True):
            timed_action.action = absolute_action
        return timed_actions


class FrankaFastWAMPolicyServer(PolicyServer):
    """LeRobot async transport with the frozen native-EEF FastWAM runtime contract."""

    prefix = "franka_fastwam_policy_server"

    def __init__(self, config: PolicyServerConfig):
        super().__init__(config)
        self._fastwam_runtime: _FastWAMRuntime | None = None
        self._checkpoint_contract: FrankaFastWAMCheckpointContract | None = None

    def _resolve_policy_specs(self, client_specs: RemotePolicyConfig) -> RemotePolicyConfig:
        _reject_remote_policy_conflicts(
            self.config,
            client_specs,
            expected_policy_type="fastwam",
        )
        policy_specs = super()._resolve_policy_specs(client_specs)
        _validate_franka_remote_policy_specs(
            policy_specs,
            expected_policy_type="fastwam",
            expected_rename_map=FRANKA_FASTWAM_RENAME_MAP,
        )
        return policy_specs

    def _make_policy_setup_ack(self, policy_specs: RemotePolicyConfig):
        ack = super()._make_policy_setup_ack(policy_specs)
        if self._checkpoint_contract is not None:
            ack.allowed_tasks.extend(
                self._checkpoint_contract.task_instructions
                or (self._checkpoint_contract.task_instruction,)
            )
        return ack

    def _load_policy(self, policy_type: str, pretrained_name_or_path: str) -> Any:
        if policy_type != "fastwam":
            raise FrankaAsyncPolicyContractError(
                f"Franka FastWAM server only accepts policy_type='fastwam', got {policy_type!r}"
            )
        if self.device is None:
            raise FrankaAsyncPolicyContractError("Policy device is unset before FastWAM checkpoint load")
        if self.actions_per_chunk is None or self.actions_per_chunk <= 0:
            raise FrankaAsyncPolicyContractError("actions_per_chunk is unset before FastWAM checkpoint load")
        contract = inspect_fastwam_checkpoint_contract(
            pretrained_name_or_path,
            expected_fps=self.config.fps,
            expected_actions_per_chunk=self.actions_per_chunk,
        )
        runtime = _load_fastwam_runtime(contract, device=self.device)
        self._checkpoint_contract = contract
        self._fastwam_runtime = runtime
        self.logger.info(
            "Loaded Franka FastWAM checkpoint from %s | "
            "tasks=%r fps=%s chunk_size=%s action_label_mode=%s "
            "video_frames=%s video_fps=%s step=%s",
            contract.checkpoint_path,
            contract.task_instructions or (contract.task_instruction,),
            contract.observation_fps,
            contract.chunk_size,
            contract.action_label_mode,
            contract.num_video_frames,
            contract.video_fps,
            contract.checkpoint_step,
        )
        return runtime.model

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Set up FastWAM without invoking LeRobot's policy registry/processors."""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.PolicySetupAck()
        client_specs = pickle.loads(request.data)  # nosec B301 - trusted internal transport boundary
        if not isinstance(client_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(client_specs)}")
        policy_specs = self._resolve_policy_specs(client_specs)
        client_id = context.peer()
        self.logger.info(
            "Receiving FastWAM policy instructions from %s | checkpoint=%s | actions=%s | device=%s",
            client_id,
            policy_specs.pretrained_name_or_path,
            policy_specs.actions_per_chunk,
            policy_specs.device,
        )

        with self._policy_setup_lock:
            self.device = policy_specs.device
            self.policy_type = policy_specs.policy_type
            self.lerobot_features = policy_specs.lerobot_features
            self.actions_per_chunk = policy_specs.actions_per_chunk
            self.rename_map = policy_specs.rename_map
            policy_setup_key = self._make_policy_setup_key(policy_specs)
            if (
                self._loaded_policy_setup_key == policy_setup_key
                and self.policy is not None
                and self._fastwam_runtime is not None
            ):
                self.logger.info("FastWAM setup unchanged; reusing the loaded runtime.")
                return self._make_policy_setup_ack(policy_specs)
            start = time.perf_counter()
            self.policy = self._load_policy(
                self.policy_type,
                policy_specs.pretrained_name_or_path,
            )
            self._loaded_policy_setup_key = policy_setup_key
            elapsed = time.perf_counter() - start
        self.logger.info("Time taken to load FastWAM on %s: %.4f seconds", self.device, elapsed)
        return self._make_policy_setup_ack(policy_specs)

    def _reconstruct_fastwam_observation(
        self,
        observation_t: TimedObservation,
    ) -> tuple[str, np.ndarray, np.ndarray, torch.Tensor]:
        if self.lerobot_features is None:
            raise FrankaAsyncPolicyContractError("LeRobot observation features are unset")
        contract = self._checkpoint_contract
        if contract is None:
            raise FrankaAsyncPolicyContractError("FastWAM checkpoint contract is unset")
        raw_observation = observation_t.get_observation()
        task = raw_observation.get("task")
        task_instructions = contract.task_instructions or (contract.task_instruction,)
        if not isinstance(task, str) or task not in task_instructions:
            raise FrankaAsyncPolicyContractError(
                "Franka observation task is not in the FastWAM checkpoint allowlist: "
                f"allowed={task_instructions!r}, actual={task!r}"
            )
        try:
            observation = make_lerobot_observation(raw_observation, self.lerobot_features)
            state = np.asarray(observation[OBS_STATE], dtype=np.float64)
            camera1 = np.asarray(observation[FRANKA_CAMERA_KEYS[0]])
            camera2 = np.asarray(observation[FRANKA_CAMERA_KEYS[1]])
        except (KeyError, TypeError, ValueError) as exc:
            raise FrankaAsyncPolicyContractError(
                "Could not reconstruct the FastWAM observation from the client payload"
            ) from exc
        fastwam_state = _build_fastwam_state(
            state,
            finger_scale=contract.finger_scale,
            finger_signs=contract.finger_signs,
        )
        input_image = _prepare_fastwam_image(
            camera1,
            camera2,
            target_height=contract.image_height,
            target_width=contract.image_width,
        )
        return task, state, fastwam_state, input_image

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        runtime = self._fastwam_runtime
        contract = self._checkpoint_contract
        if runtime is None or contract is None or self.policy is None:
            raise FrankaAsyncPolicyContractError("FastWAM runtime is not initialized")
        if self.actions_per_chunk != contract.chunk_size:
            raise FrankaAsyncPolicyContractError(
                "FastWAM runtime chunk size drift: "
                f"server={self.actions_per_chunk}, checkpoint={contract.chunk_size}"
            )
        task, anchor_state, fastwam_state, input_image = self._reconstruct_fastwam_observation(
            observation_t
        )
        if runtime.contexts_by_task is None:
            context, context_mask = runtime.context, runtime.context_mask
        else:
            try:
                context, context_mask = runtime.contexts_by_task[task]
            except KeyError as exc:
                raise FrankaAsyncPolicyContractError(
                    f"FastWAM text context was not preloaded for task {task!r}"
                ) from exc
        if context is None or context_mask is None:
            raise FrankaAsyncPolicyContractError(
                f"FastWAM text context was not preloaded for task {task!r}"
            )
        normalized_state = runtime.normalizer.normalize_state(
            torch.from_numpy(fastwam_state).to(dtype=torch.float32)
        )
        with torch.inference_mode():
            if self.config.fastwam_joint_video_inference:
                infer_joint = getattr(self.policy, "infer_joint", None)
                if not callable(infer_joint):
                    raise FrankaAsyncPolicyContractError(
                        "FastWAM joint video inference was requested, but the loaded policy has no infer_joint()"
                    )
                output = infer_joint(
                    prompt=None,
                    input_image=input_image,
                    num_video_frames=contract.num_video_frames,
                    action_horizon=contract.chunk_size,
                    action=None,
                    proprio=normalized_state,
                    context=context,
                    context_mask=context_mask,
                    text_cfg_scale=1.0,
                    num_inference_steps=contract.num_inference_steps,
                    sigma_shift=None,
                    seed=contract.seed,
                    rand_device="cpu",
                    tiled=False,
                    test_action_with_infer_action=False,
                )
                if not isinstance(output, Mapping) or not isinstance(output.get("video"), list):
                    raise FrankaAsyncPolicyContractError(
                        "FastWAM infer_joint returned no decoded video"
                    )
                joint_video = output["video"]
                if not joint_video or not all(isinstance(frame, Image.Image) for frame in joint_video):
                    raise FrankaAsyncPolicyContractError(
                        "FastWAM infer_joint video must be a non-empty list of PIL images"
                    )
                output_dir = self.config.fastwam_joint_video_output_dir
                if output_dir is not None:
                    saved_path = _save_fastwam_joint_video(
                        joint_video,
                        output_dir=output_dir,
                        fps=contract.video_fps,
                        timestep=observation_t.get_timestep(),
                    )
                    self.logger.info(
                        "Saved FastWAM joint video to %s | frames=%s fps=%s",
                        saved_path,
                        len(joint_video),
                        contract.video_fps,
                    )
                else:
                    self.logger.debug(
                        "FastWAM joint video inference produced %s frames", len(joint_video)
                    )
            else:
                infer_action_kwargs: dict[str, Any] = {
                    "prompt": None,
                    "input_image": input_image,
                    "action_horizon": contract.chunk_size,
                    "proprio": normalized_state,
                    "context": context,
                    "context_mask": context_mask,
                    "text_cfg_scale": 1.0,
                    "num_inference_steps": contract.num_inference_steps,
                    "sigma_shift": None,
                    "seed": contract.seed,
                    "rand_device": "cpu",
                    "tiled": False,
                }
                if contract.model_target == FASTWAM_JOINT_RUNTIME_MODEL_TARGET:
                    infer_action_kwargs["num_video_frames"] = contract.num_video_frames
                output = self.policy.infer_action(**infer_action_kwargs)
        if not isinstance(output, Mapping) or not isinstance(output.get("action"), torch.Tensor):
            raise FrankaAsyncPolicyContractError("FastWAM infer_action returned no tensor action")
        normalized_action = output["action"].detach().to(device="cpu", dtype=torch.float32)
        expected_shape = (contract.chunk_size, FASTWAM_ACTION_DIM)
        if tuple(normalized_action.shape) != expected_shape:
            raise FrankaAsyncPolicyContractError(
                f"FastWAM normalized action must have shape {expected_shape}, "
                f"got {tuple(normalized_action.shape)}"
            )
        physical_action = runtime.normalizer.denormalize_action(normalized_action)
        absolute_action = _decode_fastwam_absolute_action_chunk(
            anchor_state,
            physical_action,
            action_label_mode=contract.action_label_mode,
        )
        return self._time_action_chunk(
            observation_t.get_timestamp(),
            list(absolute_action),
            observation_t.get_timestep(),
        )


FRANKA_POLICY_SERVER_TYPES: dict[str, type[PolicyServer]] = {
    "pi0": FrankaPI0PolicyServer,
    "fastwam": FrankaFastWAMPolicyServer,
}


def create_franka_policy_server(config: PolicyServerConfig) -> PolicyServer:
    """Create the configured Franka backend before registering the gRPC servicer."""

    try:
        server_type = FRANKA_POLICY_SERVER_TYPES[config.policy_type]
    except KeyError as exc:
        raise ValueError(
            f"Franka policy_type must be one of {FRANKA_POLICY_TYPES}, got {config.policy_type!r}"
        ) from exc
    return server_type(config)


__all__ = [
    "ABSOLUTE_ACTION_DIM",
    "FRANKA_CAMERA_KEYS",
    "FRANKA_CAMERA_SHAPE",
    "FRANKA_CHECKPOINT_TYPE",
    "FRANKA_FASTWAM_RENAME_MAP",
    "FRANKA_PI0_RENAME_MAP",
    "FRANKA_POLICY_TYPES",
    "FRANKA_RENAME_MAP",
    "FRANKA_STATE_NAMES",
    "FrankaAsyncPolicyContractError",
    "FrankaCheckpointContract",
    "FrankaFastWAMCheckpointContract",
    "FrankaFastWAMPolicyServer",
    "FrankaPI0PolicyServer",
    "FrankaServingContract",
    "create_franka_policy_server",
    "inspect_fastwam_checkpoint_contract",
    "inspect_franka_checkpoint_contract",
    "validate_franka_checkpoint",
    "validate_franka_server_config",
]
