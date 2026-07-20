"""Franka-specific adapter for LeRobot's asynchronous policy server.

The transport, request handling, policy-processor orchestration, and action
timing remain owned by :class:`lerobot.async_inference.PolicyServer`.
This module only supplies the project-specific boundaries that the stock
server cannot infer:

* reproduce the training-time, aspect-ratio-preserving PI0 camera resize;
* restore the project's unprefixed, full-parameter PI0 checkpoint strictly;
* decode the model's 7D anchor-relative Cartesian actions into absolute 8D
  ``[xyz, quaternion_xyzw, gripper]`` targets before they leave the server.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import (
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    make_lerobot_observation,
)
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.policies.pi0.modeling_pi0 import PI0Policy, resize_with_pad_torch
from lerobot.utils.constants import OBS_STATE

from .geometry import (
    decode_relative_action,
    enforce_quaternion_continuity,
    matrix_to_quaternion_xyzw,
    rotation_6d_to_matrix,
)
from .pi0_training import (
    ACTION_DIM,
    PI0_CORE_WEIGHTS_NAMESPACE,
    STATE_DIM,
    build_pi0_full_finetune_config,
    canonicalize_pi0_full_training_graph,
    load_pi0_full_checkpoint_weights,
)

FRANKA_CHECKPOINT_TYPE = "franka_pi0_full_parameter_eef"
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
FRANKA_RENAME_MAP = {
    "observation.images.camera1": "observation.images.base_0_rgb",
    "observation.images.camera2": "observation.images.left_wrist_0_rgb",
}

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
_ACTION_SEMANTICS = {
    "gripper": "future measured target gripper_0_1",
    "rotation": "body rotvec Log(R_current.T @ R_target)",
    "translation": "base-frame target_xyz - current_xyz",
}
_MISSING = object()


class FrankaAsyncPolicyContractError(RuntimeError):
    """Raised when remote inference would violate the frozen Franka contract."""


@dataclass(frozen=True)
class FrankaCheckpointContract:
    """Checkpoint-owned serving values after cross-file contract validation."""

    profile: str
    task_instruction: str
    observation_fps: int
    action_fps: int
    chunk_size: int
    real_robot_rollout_authorized: bool
    rollout_authorization_declared: bool


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise FrankaAsyncPolicyContractError(f"Could not read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise FrankaAsyncPolicyContractError(f"Expected a JSON object in {path}")
    return value


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
            "owns relative7-to-absolute8 decoding"
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

    action_semantics = {name: action7.get(name) for name in _ACTION_SEMANTICS}
    if action_semantics != _ACTION_SEMANTICS:
        raise FrankaAsyncPolicyContractError(
            "Franka geometry action7 semantics mismatch: "
            f"expected={_ACTION_SEMANTICS}, actual={action_semantics}"
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


def validate_franka_server_config(config: PolicyServerConfig) -> FrankaCheckpointContract:
    """Validate CLI inputs against the selected checkpoint's own serving contract."""

    if config.host not in {"localhost", "127.0.0.1"}:
        raise ValueError(
            "Franka async server must bind to a loopback host because its transport is a trusted boundary"
        )
    if config.policy_type != "pi0":
        raise ValueError(f"Franka async server requires --policy_type=pi0, got {config.policy_type!r}")
    if config.pretrained_name_or_path is None:
        raise ValueError("Franka async server requires --pretrained_name_or_path")
    if config.actions_per_chunk is None:
        raise ValueError("Franka async server requires --actions_per_chunk")
    if config.policy_device is None:
        raise ValueError("Franka async server requires --policy_device")

    return inspect_franka_checkpoint_contract(
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
    model is constructed. Metadata/configuration mismatches are checked before
    hashing the multi-gigabyte model file.
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

    expected_fields = {
        "schema_version": 1,
        "checkpoint_type": FRANKA_CHECKPOINT_TYPE,
        "weights_namespace": PI0_CORE_WEIGHTS_NAMESPACE,
        "hub_upload": False,
        "wandb_artifact_upload": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise FrankaAsyncPolicyContractError(f"Franka checkpoint manifest contract mismatch: {mismatches}")

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

    model_path, model_digest = tracked_paths["model.safetensors"]
    actual_model_digest = _sha256_file(model_path)
    if actual_model_digest != model_digest:
        raise FrankaAsyncPolicyContractError(
            "Checkpoint file SHA-256 mismatch for model.safetensors: "
            f"declared={model_digest}, actual={actual_model_digest}"
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


def _decode_absolute_action_chunk(anchor_state: np.ndarray, relative_chunk: torch.Tensor) -> torch.Tensor:
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
        quaternion = enforce_quaternion_continuity(matrix_to_quaternion_xyzw(rotation))
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


class FrankaPI0PolicyServer(PolicyServer):
    """LeRobot async server with strict Franka PI0 preprocessing, load, and output adapters."""

    prefix = "franka_pi0_policy_server"

    def _resolve_policy_specs(self, client_specs: RemotePolicyConfig) -> RemotePolicyConfig:
        policy_specs = super()._resolve_policy_specs(client_specs)
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

        if policy_specs.rename_map != FRANKA_RENAME_MAP:
            raise FrankaAsyncPolicyContractError(
                "Franka client rename_map does not match the frozen two-camera PI0 mapping: "
                f"expected={FRANKA_RENAME_MAP}, actual={policy_specs.rename_map}"
            )
        return policy_specs

    def _load_policy(self, policy_type: str, pretrained_name_or_path: str) -> PI0Policy:
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
            "Strictly loaded Franka PI0 checkpoint from %s | profile=%s task=%r fps=%s chunk_size=%s",
            checkpoint_dir,
            contract.profile,
            contract.task_instruction,
            contract.observation_fps,
            contract.chunk_size,
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
                    f"Relative action {index} must be a ({ACTION_DIM},) tensor, got {shape}"
                )
            actions.append(action)

        relative_chunk = torch.stack(actions)
        anchor_state = self._extract_anchor_state(observation_t)
        absolute_chunk = _decode_absolute_action_chunk(anchor_state, relative_chunk)

        # Mutate only after the whole chunk has passed shape, finite, and geometry
        # validation so callers never observe a partially decoded queue.
        for timed_action, absolute_action in zip(timed_actions, absolute_chunk, strict=True):
            timed_action.action = absolute_action
        return timed_actions


__all__ = [
    "ABSOLUTE_ACTION_DIM",
    "FRANKA_CAMERA_KEYS",
    "FRANKA_CAMERA_SHAPE",
    "FRANKA_CHECKPOINT_TYPE",
    "FRANKA_RENAME_MAP",
    "FRANKA_STATE_NAMES",
    "FrankaAsyncPolicyContractError",
    "FrankaCheckpointContract",
    "FrankaPI0PolicyServer",
    "inspect_franka_checkpoint_contract",
    "validate_franka_checkpoint",
    "validate_franka_server_config",
]
