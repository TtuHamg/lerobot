#!/usr/bin/env python
"""Export one real Franka LeRobot frame for the async dry-run client.

The fixture is deliberately observation-only.  It contains the exact 10-D
state and two decoded RGB frames used by the project dataset; no action label
is exported, so the client/server smoke cannot accidentally treat a recorded
command as a live command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "data/lerobot/franka_current_eef_obs15_act15_v1_22of25_76b839b2"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "fixtures/async/franka_observation_v1.npz"
STATE_KEY = "observation.state"
CAMERA_KEYS = {
    "camera1": "observation.images.camera1",
    "camera2": "observation.images.camera2",
}
STATE_NAMES = (
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
CAMERA_SHAPE = (480, 640, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _validate_source_contract(profile: dict[str, Any], info: dict[str, Any]) -> None:
    expected_profile = {
        "schema_version": 1,
        "profile": "action15",
        "observation_fps": 15,
        "action_fps": 15,
        "chunk_size": 50,
        "requires_project_cartesian_adapter": True,
        "requires_project_dual_rate_adapter": False,
        "real_robot_rollout_authorized": False,
    }
    profile_drift = {
        key: {"expected": expected, "actual": profile.get(key)}
        for key, expected in expected_profile.items()
        if profile.get(key) != expected
    }
    if profile_drift:
        raise ValueError(f"source Franka profile does not match async fixture contract: {profile_drift}")

    if info.get("fps") != 15 or info.get("robot_type") != "franka_fr3_cartesian_eef":
        raise ValueError(
            "source dataset identity does not match async fixture contract: "
            f"fps={info.get('fps')!r}, robot_type={info.get('robot_type')!r}"
        )
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("source dataset info has no feature mapping")

    state_feature = features.get(STATE_KEY)
    expected_state = {"dtype": "float32", "shape": [10], "names": list(STATE_NAMES)}
    if state_feature != expected_state:
        raise ValueError(
            "source observation.state feature does not match async fixture contract: "
            f"expected={expected_state}, actual={state_feature}"
        )

    for dataset_key in CAMERA_KEYS.values():
        camera_feature = features.get(dataset_key)
        if not isinstance(camera_feature, dict):
            raise ValueError(f"source dataset is missing camera feature {dataset_key}")
        actual_camera = {
            "dtype": camera_feature.get("dtype"),
            "shape": camera_feature.get("shape"),
            "names": camera_feature.get("names"),
        }
        expected_camera = {
            "dtype": "video",
            "shape": list(CAMERA_SHAPE),
            "names": ["height", "width", "channels"],
        }
        if actual_camera != expected_camera:
            raise ValueError(
                f"source camera feature {dataset_key} does not match async fixture contract: "
                f"expected={expected_camera}, actual={actual_camera}"
            )


def _state_array(value: torch.Tensor) -> np.ndarray:
    state = value.detach().cpu().numpy().astype(np.float32, copy=False)
    if state.shape != (10,):
        raise ValueError(f"expected {STATE_KEY} shape (10,), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError(f"{STATE_KEY} contains NaN or Inf")
    return np.ascontiguousarray(state)


def _rgb_hwc_uint8(value: torch.Tensor, *, key: str) -> np.ndarray:
    image = value.detach().cpu().numpy()
    if image.shape != (3, 480, 640):
        raise ValueError(f"expected {key} shape (3,480,640), got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"{key} contains NaN or Inf")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError(f"{key} must be normalized to [0,1] by LeRobotDataset")
    image_uint8 = np.rint(image * 255.0).astype(np.uint8)
    return np.ascontiguousarray(np.moveaxis(image_uint8, 0, -1))


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
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
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


def export_fixture(dataset_root: Path, output: Path, *, index: int) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    output = output.expanduser().resolve()
    profile_path = dataset_root / "meta/franka_eef_profile.json"
    info_path = dataset_root / "meta/info.json"
    profile = _read_mapping(profile_path)
    info = _read_mapping(info_path)
    _validate_source_contract(profile, info)
    repo_id = profile.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"missing repo_id in {profile_path}")

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"fixture index {index} is outside dataset length {len(dataset)}")
    item = dataset[index]

    state = _state_array(item[STATE_KEY])
    images = {
        name: _rgb_hwc_uint8(item[key], key=key) for name, key in CAMERA_KEYS.items()
    }
    _atomic_savez(output, {"state": state, **images})

    metadata = {
        "schema_version": 1,
        "fixture": str(output),
        "fixture_sha256": _sha256(output),
        "source": {
            "dataset_root": str(dataset_root),
            "repo_id": repo_id,
            "index": index,
            "episode_index": int(item["episode_index"].item()),
            "frame_index": int(item["frame_index"].item()),
            "timestamp": float(item["timestamp"].item()),
            "task": str(item["task"]),
            "source_dataset_hash": profile.get("source_dataset_hash"),
            "profile_sha256": _sha256(profile_path),
            "info_sha256": _sha256(info_path),
        },
        "arrays": {
            "state": {"shape": list(state.shape), "dtype": str(state.dtype)},
            **{
                name: {"shape": list(image.shape), "dtype": str(image.dtype)}
                for name, image in images.items()
            },
        },
    }
    _atomic_write_json(output.with_suffix(".json"), metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = export_fixture(args.dataset_root, args.output, index=args.index)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
