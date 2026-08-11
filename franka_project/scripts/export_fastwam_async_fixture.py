#!/usr/bin/env python
"""Export one training-aligned FastWAM observation for the async dry-run client.

The frozen NPZ follows the Franka wire contract: one 10-D canonical state and
two raw 480x640 RGB images.  The source FastWAM dataset stores 224x224 H.264
frames and an 8-D policy state, so this exporter follows the conversion audit
ledger back to the exact raw MCAP messages, rebuilds the wire state, and then
fails closed unless deployment preprocessing reconstructs the stored training
state and conversion-time image preprocessing.

The fixture remains observation-only.  Recorded actions are used only to write
provenance showing that the selected 32-step training window contains a clear
close-and-lift event; they are never placed in the client payload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from franka_eef_pipeline.geometry import (  # noqa: E402
    matrix_to_quaternion_xyzw,
    rotation_6d_to_matrix,
    so3_exp,
    so3_log,
)

DEFAULT_FASTWAM_ROOT = PROJECT_ROOT.parent.parent / "FastWAM"
DEFAULT_OUTPUT = PROJECT_ROOT / "fixtures/async/fastwam_grab_cups_observation_v1.npz"
DEFAULT_EPISODE_INDEX = 4
DEFAULT_FRAME_INDEX = 106

WIRE_STATE_NAMES = (
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
TRAINING_STATE_NAMES = (
    "eef.x_m",
    "eef.y_m",
    "eef.z_m",
    "eef.axis_angle_x_rad",
    "eef.axis_angle_y_rad",
    "eef.axis_angle_z_rad",
    "gripper.finger_left_m",
    "gripper.finger_right_m",
)
TRAINING_ACTION_NAMES = (
    "delta_eef.x_m_base",
    "delta_eef.y_m_base",
    "delta_eef.z_m_base",
    "delta_eef.rotvec_x_rad_body",
    "delta_eef.rotvec_y_rad_body",
    "delta_eef.rotvec_z_rad_body",
    "gripper.open_target_0_1",
)
CAMERA_NAMES = ("camera1", "camera2")
WIRE_CAMERA_SHAPE = (480, 640, 3)
TRAINING_CAMERA_SHAPE = (224, 224, 3)
ACTION_HORIZON = 32
NUM_FRAMES = ACTION_HORIZON + 1
STATE_TOLERANCE = 2e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError(f"expected a JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _require_subset(actual: Any, expected: Any, *, where: str) -> None:
    """Require exact values for a nested contract subset while allowing new metadata."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{where} must be an object, got {type(actual).__name__}")
        for key, expected_value in expected.items():
            if key not in actual:
                raise ValueError(f"{where} is missing required field {key!r}")
            _require_subset(actual[key], expected_value, where=f"{where}.{key}")
        return
    if actual != expected:
        raise ValueError(f"{where} drifted: expected={expected!r}, actual={actual!r}")


def _validate_source_contract(
    profile: dict[str, Any],
    dataset_contract: dict[str, Any],
    info: dict[str, Any],
) -> None:
    profile_name = profile.get("name")
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError("resolved FastWAM dataset profile has no dataset name")
    streams = profile.get("streams")
    if not isinstance(streams, dict):
        raise ValueError("resolved FastWAM dataset profile has no streams object")
    camera_configs = streams.get("cameras")
    if not isinstance(camera_configs, list) or len(camera_configs) != len(CAMERA_NAMES):
        raise ValueError("resolved FastWAM dataset profile must contain exactly two cameras")
    if [camera.get("name") for camera in camera_configs if isinstance(camera, dict)] != list(CAMERA_NAMES):
        raise ValueError("resolved FastWAM camera names/order must be camera1,camera2")
    canonical_topics = {
        "/camera1/camera1/color/image_raw",
        "/camera2/camera2/color/image_raw",
    }
    actual_topics: list[str] = []
    for index, camera in enumerate(camera_configs):
        _require_subset(
            camera,
            {
                "name": CAMERA_NAMES[index],
                "message_type": "sensor_msgs/msg/Image",
                "expected_encoding": "rgb8",
                "expected_height": WIRE_CAMERA_SHAPE[0],
                "expected_width": WIRE_CAMERA_SHAPE[1],
            },
            where=f"resolved FastWAM camera {index}",
        )
        topic = camera.get("topic")
        if not isinstance(topic, str):
            raise ValueError(f"resolved FastWAM camera {index} has no topic")
        actual_topics.append(topic)
    if set(actual_topics) != canonical_topics or len(set(actual_topics)) != len(CAMERA_NAMES):
        raise ValueError(
            "resolved FastWAM physical camera topics must be a one-to-one mapping of "
            f"{sorted(canonical_topics)}, got {actual_topics}"
        )
    profile_task = profile.get("task")
    if not isinstance(profile_task, dict):
        raise ValueError("resolved FastWAM dataset profile has no task contract")
    default_task = profile_task.get("default_instruction")
    manifest_field = profile_task.get("manifest_field")
    default_is_placeholder = isinstance(default_task, str) and default_task.startswith("__MISSING_")
    if default_is_placeholder and not isinstance(manifest_field, str):
        raise ValueError("multi-task FastWAM profile must declare task.manifest_field")
    if not default_is_placeholder and (not isinstance(default_task, str) or not default_task):
        raise ValueError("single-task FastWAM profile must declare task.default_instruction")

    _require_subset(
        profile,
        {
            "schema_version": 1,
            "adapter": "native_franka_mcap_v1",
            "streams": {
                "timestamp_source": "log_time",
                "reference_camera": "camera1",
                "cameras": camera_configs,
                "eef": {
                    "topic": "/franka_robot_state_broadcaster/current_pose",
                    "message_type": "geometry_msgs/msg/PoseStamped",
                    "position_path": "pose.position",
                    "orientation_path": "pose.orientation",
                    "quaternion_order": "xyzw",
                    "frame_id": "base",
                },
                "gripper": {
                    "topic": "/gripper/joint_states",
                    "message_type": "sensor_msgs/msg/JointState",
                    "position_path": "position",
                    "names_path": "name",
                    "joint_name": "robotiq_85_left_knuckle_joint",
                },
            },
            "alignment": {
                "strategy": "latest_not_after",
                "crop_common_interval": True,
            },
            "representation": {
                "state": {
                    "rotation": "axis_angle",
                    "finger_scale": 0.04,
                    "finger_signs": [1.0, -1.0],
                },
                "action": {
                    "type": "adjacent_delta_eef",
                    "translation_frame": "base",
                    "rotation": "body_rotvec",
                    "gripper": "next_absolute_target",
                    "raw_open": 0.0,
                    "raw_closed": 0.8,
                    "clip_gripper": True,
                    "gripper_output": "closed_0_open_1",
                },
            },
            "sampling": {
                "fps": 30,
                "num_frames": NUM_FRAMES,
                "global_sample_stride": 1,
                "action_video_freq_ratio": 4,
            },
            "images": {
                "preprocess": "center_crop",
                "height": TRAINING_CAMERA_SHAPE[0],
                "width": TRAINING_CAMERA_SHAPE[1],
                "fill_rgb": [0, 0, 0],
            },
            "task": profile_task,
            "output": {
                "robot_type": "franka_fr3_cartesian_eef",
            },
        },
        where="resolved FastWAM dataset profile",
    )

    contract_task = dataset_contract.get("task")
    if contract_task != profile_task:
        raise ValueError(
            f"FastWAM profile/dataset task contracts differ: profile={profile_task}, dataset={contract_task}"
        )
    _require_subset(
        dataset_contract,
        {
            "schema_version": 1,
            "dataset_name": profile_name,
            "format": {"name": "lerobot", "codebase_version": "v2.1"},
            "representation": {
                "state": {
                    "dim": 8,
                    "names": list(TRAINING_STATE_NAMES),
                    "type": "eef_absolute_axis_angle_with_pseudo_fingers",
                    "rotation": "principal_axis_angle_rad",
                    "frame": "base",
                    "finger_scale": 0.04,
                    "finger_signs": [1.0, -1.0],
                },
                "action": {
                    "dim": 7,
                    "names": list(TRAINING_ACTION_NAMES),
                    "type": "adjacent_delta_eef",
                },
                "gripper_calibration": {
                    "raw_open": 0.0,
                    "raw_closed": 0.8,
                    "clip": True,
                },
            },
            "temporal": {
                "fps": 30,
                "num_frames": NUM_FRAMES,
                "action_horizon": ACTION_HORIZON,
                "global_sample_stride": 1,
                "action_video_freq_ratio": 4,
                "video_frames": 9,
            },
            "images": {
                "preprocess": "center_crop",
                "height": 224,
                "width": 224,
                "concat_multi_camera": "horizontal",
                "ordered_camera_names": list(CAMERA_NAMES),
            },
            "alignment": {"strategy": "latest_not_after", "crop_common_interval": True},
            "task": profile_task,
        },
        where="FastWAM dataset contract",
    )
    contract_cameras = dataset_contract.get("cameras")
    if contract_cameras is not None:
        expected_cameras = [
            {
                "name": name,
                "key": f"observation.images.{name}",
                "topic": camera_configs[index]["topic"],
                "order": index,
            }
            for index, name in enumerate(CAMERA_NAMES)
        ]
        actual_cameras = [
            {key: camera.get(key) for key in ("name", "key", "topic", "order")}
            for camera in contract_cameras
            if isinstance(camera, dict)
        ]
        if actual_cameras != expected_cameras:
            raise ValueError(
                f"FastWAM profile/dataset camera mappings differ: expected={expected_cameras}, "
                f"actual={actual_cameras}"
            )

    _require_subset(
        info,
        {
            "codebase_version": "v2.1",
            "robot_type": "franka_fr3_cartesian_eef",
            "fps": 30,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [8],
                    "names": list(TRAINING_STATE_NAMES),
                },
                "action": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": list(TRAINING_ACTION_NAMES),
                },
                **{
                    f"observation.images.{name}": {
                        "dtype": "video",
                        "shape": list(TRAINING_CAMERA_SHAPE),
                        "names": ["height", "width", "channels"],
                    }
                    for name in CAMERA_NAMES
                },
            },
        },
        where="FastWAM LeRobot info",
    )


def _fixed_list_numpy(table: Any, key: str, *, width: int) -> np.ndarray:
    column = table[key].combine_chunks()
    values = np.asarray(column.values.to_numpy(zero_copy_only=False))
    result = values.reshape(len(column), width)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"dataset column {key} contains NaN or Inf")
    return result.astype(np.float32, copy=False)


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        path.chmod(0o644)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        path.chmod(0o644)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _center_crop_resize(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != WIRE_CAMERA_SHAPE or value.dtype != np.uint8:
        raise ValueError(
            f"wire image must be uint8 shape {WIRE_CAMERA_SHAPE}, got {value.dtype} {value.shape}"
        )
    source_height, source_width = value.shape[:2]
    source_aspect = source_width / source_height
    target_aspect = width / height
    pil = Image.fromarray(value, mode="RGB")
    if source_aspect > target_aspect:
        crop_width = int(round(source_height * target_aspect))
        left = (source_width - crop_width) // 2
        pil = pil.crop((left, 0, left + crop_width, source_height))
    else:
        crop_height = int(round(source_width / target_aspect))
        top = (source_height - crop_height) // 2
        pil = pil.crop((0, top, source_width, top + crop_height))
    pil = pil.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))


def _wire_to_fastwam_state(
    state10: np.ndarray,
    *,
    finger_scale: float,
    finger_signs: tuple[float, float],
) -> np.ndarray:
    state = np.asarray(state10, dtype=np.float64)
    if state.shape != (10,) or not np.all(np.isfinite(state)):
        raise ValueError(f"wire state must be finite shape (10,), got {state.shape}")
    closed = float(state[-1])
    if not 0.0 <= closed <= 1.0:
        raise ValueError(f"wire gripper.closed_0_1 must be in [0,1], got {closed}")
    rotation = rotation_6d_to_matrix(state[3:9])
    fingers = finger_scale * (1.0 - closed) * np.asarray(finger_signs, dtype=np.float64)
    return np.concatenate((state[:3], so3_log(rotation), fingers))


def _verify_adjacent_action_window(
    state_window: np.ndarray,
    action_window: np.ndarray,
    *,
    finger_scale: float,
    finger_signs: tuple[float, float],
) -> tuple[float, np.ndarray]:
    """Verify adjacent-delta labels and return their absolute8 target trajectory."""

    states = np.asarray(state_window, dtype=np.float64)
    actions = np.asarray(action_window, dtype=np.float64)
    if states.shape != (NUM_FRAMES, len(TRAINING_STATE_NAMES)):
        raise ValueError(f"state window must have shape {(NUM_FRAMES, 8)}, got {states.shape}")
    if actions.shape != (ACTION_HORIZON, len(TRAINING_ACTION_NAMES)):
        raise ValueError(f"action window must have shape {(ACTION_HORIZON, 7)}, got {actions.shape}")
    rotations = np.stack([so3_exp(row[3:6]) for row in states])
    delta_position = np.diff(states[:, :3], axis=0)
    delta_rotation = np.stack(
        [
            so3_log(previous.T @ following)
            for previous, following in zip(rotations[:-1], rotations[1:], strict=True)
        ]
    )
    signs = np.asarray(finger_signs, dtype=np.float64)
    if signs.shape != (2,) or np.any(np.abs(signs) < 1e-12):
        raise ValueError("finger_signs must contain two non-zero values")
    open_from_fingers = states[1:, 6:8] / (float(finger_scale) * signs[None, :])
    finger_disagreement = float(np.max(np.abs(open_from_fingers[:, 0] - open_from_fingers[:, 1])))
    if finger_disagreement > STATE_TOLERANCE:
        raise ValueError(
            f"FastWAM pseudo-finger state disagrees across left/right channels: max_abs={finger_disagreement}"
        )
    open_target = np.mean(open_from_fingers, axis=1)
    expected_actions = np.concatenate(
        (delta_position, delta_rotation, open_target[:, None]),
        axis=1,
    )
    error = float(np.max(np.abs(expected_actions - actions)))
    if error > STATE_TOLERANCE:
        raise ValueError(
            f"stored FastWAM action is not the declared adjacent transition: max_abs_error={error}"
        )
    quaternions = np.stack([matrix_to_quaternion_xyzw(rotation) for rotation in rotations[1:]])
    ground_truth_absolute = np.concatenate(
        (states[1:, :3], quaternions, (1.0 - np.clip(open_target, 0.0, 1.0))[:, None]),
        axis=1,
    )
    return error, np.ascontiguousarray(ground_truth_absolute, dtype=np.float32)


def _load_fastwam_adapter(fastwam_root: Path) -> tuple[Any, Any, Any]:
    source_root = (fastwam_root / "franka_project/src").resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"FastWAM Franka source tree does not exist: {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    readers = importlib.import_module("native_eef_fastwam.readers.native_franka_mcap")
    contracts = importlib.import_module("native_eef_fastwam.contracts")
    geometry = importlib.import_module("native_eef_fastwam.geometry")
    for module in (readers, contracts, geometry):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(source_root):
            raise RuntimeError(
                "native_eef_fastwam resolved from the wrong checkout: "
                f"expected under {source_root}, got {module_path}"
            )
    return readers, contracts, geometry


def _decode_raw_images(
    mcap_path: Path,
    profile: dict[str, Any],
    raw_indices: dict[str, int],
    *,
    decode_sensor_image_rgb: Any,
) -> dict[str, NDArray[np.uint8]]:
    try:
        from mcap_ros2.reader import read_ros2_messages
    except ImportError as error:
        raise ImportError("mcap_ros2 is required to reproduce the FastWAM fixture") from error

    cameras_by_topic = {camera["topic"]: camera for camera in profile["streams"]["cameras"]}
    if set(raw_indices) != set(CAMERA_NAMES):
        raise ValueError(f"raw camera indices must cover {CAMERA_NAMES}, got {sorted(raw_indices)}")
    counters = dict.fromkeys(CAMERA_NAMES, 0)
    images: dict[str, NDArray[np.uint8]] = {}
    for record in read_ros2_messages(mcap_path, topics=list(cameras_by_topic)):
        camera = cameras_by_topic[record.channel.topic]
        name = str(camera["name"])
        raw_index = counters[name]
        counters[name] += 1
        if raw_index == raw_indices[name]:
            image = decode_sensor_image_rgb(record.ros_msg)
            if image.shape != WIRE_CAMERA_SHAPE or image.dtype != np.uint8:
                raise ValueError(
                    f"decoded {name} must be uint8 shape {WIRE_CAMERA_SHAPE}, got {image.dtype} {image.shape}"
                )
            images[name] = np.ascontiguousarray(image)
            if len(images) == len(CAMERA_NAMES):
                break
    missing = sorted(set(CAMERA_NAMES) - set(images))
    if missing:
        raise ValueError(f"selected raw camera messages were not found: {missing}")
    return images


def _decode_video_frame(path: Path, *, frame_index: int) -> NDArray[np.uint8]:
    try:
        import av
    except ImportError as error:
        raise ImportError("PyAV is required to compare the fixture with training H.264 frames") from error

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index == frame_index:
                value = frame.to_ndarray(format="rgb24")
                return np.ascontiguousarray(value, dtype=np.uint8)
    raise IndexError(f"video {path} has no decoded frame {frame_index}")


def _image_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | None]:
    if reference.shape != candidate.shape:
        raise ValueError(f"image comparison shape mismatch: {reference.shape} vs {candidate.shape}")
    delta = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(delta * delta))
    return {
        "mean_absolute_error_uint8": float(np.mean(np.abs(delta))),
        "max_absolute_error_uint8": int(np.max(np.abs(delta))),
        "mse_uint8": mse,
        "psnr_db": None if mse == 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse))),
    }


def _find_unique(records: list[dict[str, Any]], *, field: str, value: Any, where: str) -> dict[str, Any]:
    matches = [record for record in records if record.get(field) == value]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {where} with {field}={value!r}, got {len(matches)}")
    return matches[0]


def export_fixture(
    fastwam_root: Path,
    dataset_root: Path,
    output: Path,
    *,
    episode_index: int,
    frame_index: int,
    policy_pose_frame: str = "eef",
) -> dict[str, Any]:
    fastwam_root = fastwam_root.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if episode_index < 0 or frame_index < 0:
        raise ValueError("episode_index and frame_index must be non-negative")
    if policy_pose_frame not in {"eef", "link8"}:
        raise ValueError("policy_pose_frame must be 'eef' or 'link8'")

    audit_root = dataset_root / "audit"
    profile_path = audit_root / "resolved_dataset_profile.json"
    contract_path = audit_root / "dataset_contract.json"
    info_path = dataset_root / "meta/info.json"
    segments_path = audit_root / "source_to_segments.jsonl"
    alignment_report_path = audit_root / "alignment_report.json"
    tasks_path = dataset_root / "meta/tasks.jsonl"
    episodes_path = dataset_root / "meta/episodes.jsonl"
    profile = _read_mapping(profile_path)
    dataset_contract = _read_mapping(contract_path)
    info = _read_mapping(info_path)
    _validate_source_contract(profile, dataset_contract, info)

    segment = _find_unique(
        _read_jsonl(segments_path),
        field="output_episode_index",
        value=episode_index,
        where="retained source segment",
    )
    if segment.get("retained") is not True:
        raise ValueError(f"output episode {episode_index} does not map to a retained source segment")
    observations = int(segment["observations"])
    if frame_index + ACTION_HORIZON >= observations:
        raise IndexError(
            f"episode {episode_index} frame {frame_index} has no complete {ACTION_HORIZON}-step "
            f"training window (episode length {observations})"
        )

    chunk_size = int(info["chunks_size"])
    format_values = {
        "episode_chunk": episode_index // chunk_size,
        "episode_index": episode_index,
    }
    episode_data_path = dataset_root / str(info["data_path"]).format(**format_values)

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("pyarrow is required to reproduce the FastWAM fixture") from error

    episode_table = pq.read_table(episode_data_path)
    if len(episode_table) != observations:
        raise ValueError(
            f"source segment says {observations} observations but {episode_data_path} has "
            f"{len(episode_table)} rows"
        )
    states = _fixed_list_numpy(episode_table, "observation.state", width=8)
    actions = _fixed_list_numpy(episode_table, "action", width=7)
    frame_indices = np.asarray(episode_table["frame_index"].to_numpy())
    episode_indices = np.asarray(episode_table["episode_index"].to_numpy())
    global_indices = np.asarray(episode_table["index"].to_numpy())
    task_indices = np.asarray(episode_table["task_index"].to_numpy())
    expected_frames = np.arange(observations, dtype=frame_indices.dtype)
    if not np.array_equal(frame_indices, expected_frames):
        raise ValueError(f"frame_index is not contiguous in {episode_data_path}")
    if not np.all(episode_indices == episode_index):
        raise ValueError(f"episode_index column drifted in {episode_data_path}")
    expected_globals = int(segment["output_global_start"]) + expected_frames
    if not np.array_equal(global_indices, expected_globals):
        raise ValueError(f"global index column disagrees with {segments_path}")

    task_record = _find_unique(
        _read_jsonl(tasks_path),
        field="task_index",
        value=int(task_indices[frame_index]),
        where="task record",
    )
    task = str(task_record["task"])
    task_contract = profile["task"]
    default_task = task_contract.get("default_instruction")
    if isinstance(default_task, str) and not default_task.startswith("__MISSING_") and task != default_task:
        raise ValueError(f"fixture task drifted: expected profile task, got {task!r}")
    episode_record = _find_unique(
        _read_jsonl(episodes_path),
        field="episode_index",
        value=episode_index,
        where="episode metadata record",
    )
    if int(episode_record["length"]) != observations or episode_record.get("tasks") != [task]:
        raise ValueError(f"episode metadata disagrees with retained segment: {episode_record}")

    source_episode_id = str(segment["source_episode_id"])
    alignment_report = _read_mapping(alignment_report_path)
    alignment_episode = _find_unique(
        list(alignment_report["episodes"]),
        field="episode_id",
        value=source_episode_id,
        where="alignment report episode",
    )
    mcap_path = Path(str(alignment_episode["mcap_path"])).expanduser().resolve()
    if not mcap_path.is_file():
        raise FileNotFoundError(mcap_path)
    expected_raw_name = str(episode_record["raw_file_name"])
    if mcap_path.name != expected_raw_name:
        raise ValueError(f"episode metadata MCAP name {expected_raw_name!r} != {mcap_path.name!r}")

    alignment_path = audit_root / "alignment" / f"{source_episode_id}.parquet"
    alignment_table = pq.read_table(alignment_path)
    source_row = int(segment["start_observation"]) + frame_index
    alignment_window = alignment_table.slice(source_row, NUM_FRAMES)
    if len(alignment_window) != NUM_FRAMES:
        raise ValueError("alignment sidecar does not contain the complete training window")
    if not all(alignment_window["observation_valid"].to_pylist()):
        raise ValueError("fixture training window contains an invalid aligned observation")
    if not all(alignment_window["transition_to_next_valid"].slice(0, ACTION_HORIZON).to_pylist()):
        raise ValueError("fixture training window contains an invalid action transition")
    retained_segments = np.asarray(alignment_window["retained_local_segment"].to_numpy())
    if not np.all(retained_segments == int(segment["local_segment_index"])):
        raise ValueError("fixture training window crosses a retained segment boundary")
    alignment_row = alignment_window.slice(0, 1).to_pylist()[0]

    readers, contracts, fastwam_geometry = _load_fastwam_adapter(fastwam_root)
    signals = readers.load_episode_signals(mcap_path, profile, episode_id=source_episode_id)
    reference_camera = str(profile["streams"]["reference_camera"])
    reference_raw_index = int(alignment_row["reference_raw_index"])
    reference_time_ns = int(signals.cameras[reference_camera].time_ns[reference_raw_index])
    if reference_time_ns != int(alignment_row["reference_time_ns"]):
        raise ValueError("alignment reference timestamp disagrees with the raw MCAP stream")

    raw_camera_indices = {name: int(alignment_row[f"camera.{name}.raw_index"]) for name in CAMERA_NAMES}
    if raw_camera_indices[reference_camera] != reference_raw_index:
        raise ValueError("reference camera raw index disagrees with alignment sidecar")
    for name, raw_index in raw_camera_indices.items():
        age_ns = reference_time_ns - int(signals.cameras[name].time_ns[raw_index])
        if age_ns != int(alignment_row[f"camera.{name}.age_ns"]):
            raise ValueError(f"{name} alignment age disagrees with the raw MCAP stream")
    eef_raw_index = int(alignment_row["eef_raw_index"])
    gripper_raw_index = int(alignment_row["gripper_raw_index"])
    if reference_time_ns - int(signals.eef.time_ns[eef_raw_index]) != int(alignment_row["eef_age_ns"]):
        raise ValueError("EEF alignment age disagrees with the raw MCAP stream")
    if reference_time_ns - int(signals.gripper.time_ns[gripper_raw_index]) != int(
        alignment_row["gripper_age_ns"]
    ):
        raise ValueError("gripper alignment age disagrees with the raw MCAP stream")

    action_config = profile["representation"]["action"]
    state_config = profile["representation"]["state"]
    raw_open = float(action_config["raw_open"])
    raw_closed = float(action_config["raw_closed"])
    finger_scale = float(state_config["finger_scale"])
    finger_signs = tuple(float(value) for value in state_config["finger_signs"])
    position = signals.eef.values[eef_raw_index, :3]
    quaternion = signals.eef.values[eef_raw_index, 3:7]
    raw_gripper = float(signals.gripper.values[gripper_raw_index, 0])
    training_state_from_raw = contracts.build_eef_state(
        position[None, :],
        quaternion[None, :],
        np.asarray([raw_gripper]),
        raw_open=raw_open,
        raw_closed=raw_closed,
        finger_scale=finger_scale,
        finger_signs=finger_signs,
        clip_gripper=bool(action_config["clip_gripper"]),
    )[0]
    open_value = float(
        contracts.gripper_to_open(
            np.asarray([raw_gripper]),
            raw_open=raw_open,
            raw_closed=raw_closed,
            clip=bool(action_config["clip_gripper"]),
        )[0]
    )
    rotation = fastwam_geometry.quaternion_xyzw_to_matrix(quaternion)
    rotation6d = np.concatenate((rotation[:, 0], rotation[:, 1]))
    wire_state = np.ascontiguousarray(
        np.concatenate((position, rotation6d, np.asarray([1.0 - open_value]))).astype(np.float32)
    )
    stored_training_state = states[frame_index]
    raw_state_error = float(np.max(np.abs(training_state_from_raw - stored_training_state)))
    reconstructed_training_state = _wire_to_fastwam_state(
        wire_state,
        finger_scale=finger_scale,
        finger_signs=finger_signs,
    )
    wire_state_error = float(np.max(np.abs(reconstructed_training_state - stored_training_state)))
    if raw_state_error > STATE_TOLERANCE or wire_state_error > STATE_TOLERANCE:
        raise ValueError(
            "raw/wire state does not reconstruct the stored FastWAM state: "
            f"raw_error={raw_state_error}, wire_error={wire_state_error}"
        )

    raw_images = _decode_raw_images(
        mcap_path,
        profile,
        raw_camera_indices,
        decode_sensor_image_rgb=readers.decode_sensor_image_rgb,
    )
    preprocessed_images: dict[str, np.ndarray] = {}
    decoded_training_images: dict[str, np.ndarray] = {}
    image_evidence: dict[str, Any] = {}
    for name in CAMERA_NAMES:
        deployment_preprocessed = _center_crop_resize(
            raw_images[name],
            height=int(profile["images"]["height"]),
            width=int(profile["images"]["width"]),
        )
        conversion_preprocessed = readers._preprocess_image(raw_images[name], profile["images"])
        if not np.array_equal(deployment_preprocessed, conversion_preprocessed):
            raise ValueError(f"deployment and conversion-time image preprocessing differ for {name}")
        video_key = f"observation.images.{name}"
        video_path = dataset_root / str(info["video_path"]).format(
            **format_values,
            video_key=video_key,
        )
        decoded_training = _decode_video_frame(video_path, frame_index=frame_index)
        if decoded_training.shape != TRAINING_CAMERA_SHAPE or decoded_training.dtype != np.uint8:
            raise ValueError(
                f"decoded training {name} must be uint8 shape {TRAINING_CAMERA_SHAPE}, "
                f"got {decoded_training.dtype} {decoded_training.shape}"
            )
        preprocessed_images[name] = deployment_preprocessed
        decoded_training_images[name] = decoded_training
        image_evidence[name] = {
            "raw_mcap_index": raw_camera_indices[name],
            "raw_wire_sha256": _array_sha256(raw_images[name]),
            "preencode_center_crop_sha256": _array_sha256(deployment_preprocessed),
            "training_h264_frame_sha256": _array_sha256(decoded_training),
            "training_h264_path": str(video_path),
            "training_h264_sha256": _sha256(video_path),
            "preencode_vs_training_h264_decode": _image_difference(
                deployment_preprocessed,
                decoded_training,
            ),
        }

    arrays = {"state": wire_state, **raw_images}
    _atomic_savez(output, arrays)

    state_window = np.ascontiguousarray(states[frame_index : frame_index + NUM_FRAMES])
    action_window = np.ascontiguousarray(actions[frame_index : frame_index + ACTION_HORIZON])
    adjacent_action_error, ground_truth_absolute = _verify_adjacent_action_window(
        state_window,
        action_window,
        finger_scale=finger_scale,
        finger_signs=finger_signs,
    )
    video_offsets = list(range(0, NUM_FRAMES, int(profile["sampling"]["action_video_freq_ratio"])))
    below_close_threshold = np.flatnonzero(action_window[:, -1] <= 0.2)
    combined_preprocessed = np.concatenate([preprocessed_images[name] for name in CAMERA_NAMES], axis=1)
    combined_training = np.concatenate([decoded_training_images[name] for name in CAMERA_NAMES], axis=1)
    metadata = {
        "schema_version": 1,
        "fixture_kind": "fastwam_training_aligned_raw_observation",
        "policy_type": "fastwam",
        "fixture": str(output),
        "fixture_sha256": _sha256(output),
        "source": {
            "fastwam_root": str(fastwam_root),
            "dataset_root": str(dataset_root),
            "dataset_name": dataset_contract["dataset_name"],
            "repo_id": profile["output"]["repo_id"],
            "task": task,
            "output_episode_index": episode_index,
            "output_frame_index": frame_index,
            "output_global_index": int(global_indices[frame_index]),
            "source_episode_id": source_episode_id,
            "source_episode_status": segment.get("source_status"),
            "source_local_segment_index": int(segment["local_segment_index"]),
            "source_alignment_row": source_row,
            "mcap_path": str(mcap_path),
            "mcap_size_bytes": mcap_path.stat().st_size,
            "timestamp_source": profile["streams"]["timestamp_source"],
            "alignment_strategy": profile["alignment"]["strategy"],
            "reference_time_ns": reference_time_ns,
            "raw_indices": {
                **{name: raw_camera_indices[name] for name in CAMERA_NAMES},
                "eef": eef_raw_index,
                "gripper": gripper_raw_index,
            },
            "alignment_age_ns": {
                **{name: int(alignment_row[f"camera.{name}.age_ns"]) for name in CAMERA_NAMES},
                "eef": int(alignment_row["eef_age_ns"]),
                "gripper": int(alignment_row["gripper_age_ns"]),
            },
            "artifact_sha256": {
                "resolved_dataset_profile": _sha256(profile_path),
                "dataset_contract": _sha256(contract_path),
                "info": _sha256(info_path),
                "source_to_segments": _sha256(segments_path),
                "alignment_report": _sha256(alignment_report_path),
                "alignment_sidecar": _sha256(alignment_path),
                "episode_parquet": _sha256(episode_data_path),
                "tasks": _sha256(tasks_path),
                "episodes": _sha256(episodes_path),
            },
        },
        "wire_contract": {
            "state_names": list(WIRE_STATE_NAMES),
            "camera_order": list(CAMERA_NAMES),
            "camera_topics": {camera["name"]: camera["topic"] for camera in profile["streams"]["cameras"]},
            "camera_shape": list(WIRE_CAMERA_SHAPE),
            "camera_dtype": "uint8",
            "policy_pose_frame": policy_pose_frame,
            "policy_pose_frame_basis": "explicit exporter declaration; verify against recorder/FK evidence",
            "rename_map": {},
            "fps": 30,
            "actions_per_chunk": ACTION_HORIZON,
        },
        "preprocessing_parity": {
            "raw_state_to_stored_training_state8_max_abs_error": raw_state_error,
            "wire_state10_to_stored_training_state8_max_abs_error": wire_state_error,
            "state_tolerance": STATE_TOLERANCE,
            "image_preprocess": "raw 480x640 center-crop square, bilinear resize to 224x224",
            "deployment_equals_conversion_preencode": True,
            "images": image_evidence,
            "combined_preencode_shape": list(combined_preprocessed.shape),
            "combined_preencode_sha256": _array_sha256(combined_preprocessed),
            "combined_training_h264_shape": list(combined_training.shape),
            "combined_training_h264_sha256": _array_sha256(combined_training),
            "combined_preencode_vs_training_h264_decode": _image_difference(
                combined_preprocessed,
                combined_training,
            ),
            "note": (
                "NPZ images are exact raw MCAP RGB messages because the async wire contract requires "
                "480x640. Training consumes their 224x224 center-cropped H.264 encodes; the reported "
                "pixel residual is the expected video-codec difference after identical geometry."
            ),
        },
        "training_window_evidence": {
            "num_observation_frames": NUM_FRAMES,
            "num_action_steps": ACTION_HORIZON,
            "video_frame_offsets": video_offsets,
            "video_output_frame_indices": [frame_index + value for value in video_offsets],
            "state_window_sha256": _array_sha256(state_window),
            "action_window_sha256": _array_sha256(action_window),
            "adjacent_action_reconstruction_max_abs_error": adjacent_action_error,
            "ground_truth_absolute8": ground_truth_absolute.astype(float).tolist(),
            "anchor_state8": stored_training_state.astype(float).tolist(),
            "terminal_state8": state_window[-1].astype(float).tolist(),
            "anchor_gripper_open_0_1": float(stored_training_state[-2] / finger_scale),
            "minimum_future_gripper_open_target_0_1": float(np.min(action_window[:, -1])),
            "first_action_step_open_target_at_or_below_0_2_1_based": (
                None if len(below_close_threshold) == 0 else int(below_close_threshold[0] + 1)
            ),
            "anchor_eef_z_m": float(stored_training_state[2]),
            "terminal_eef_z_m": float(state_window[-1, 2]),
            "eef_z_delta_m": float(state_window[-1, 2] - stored_training_state[2]),
            "actions_exported_to_fixture": False,
        },
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()
        },
    }
    _atomic_write_json(output.with_suffix(".json"), metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastwam-root", type=Path, default=DEFAULT_FASTWAM_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Defaults to <fastwam-root>/franka_project/data/lerobot/franka_eef_grab_cups_v2.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode-index", type=int, default=DEFAULT_EPISODE_INDEX)
    parser.add_argument("--frame-index", type=int, default=DEFAULT_FRAME_INDEX)
    parser.add_argument(
        "--policy-pose-frame",
        choices=("eef", "link8"),
        default="eef",
        help="Semantic endpoint represented by the recorded current_pose values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = args.fastwam_root / "franka_project/data/lerobot/franka_eef_grab_cups_v2"
    metadata = export_fixture(
        args.fastwam_root,
        dataset_root,
        args.output,
        episode_index=args.episode_index,
        frame_index=args.frame_index,
        policy_pose_frame=args.policy_pose_frame,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
