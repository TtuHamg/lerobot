from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from franka_eef_pipeline.dual_rate_dataset import (
    ACTION_KEY,
    ACTION_PAD_KEY,
    STATE_KEY,
    AnchorRecord,
    CartesianAnchorDataset,
    UnsupportedActionRateError,
    absolute_carrier_to_relative_action,
    action_delta_timestamps,
    action15_delta_timestamps,
    load_effective_pi0_stats,
    load_valid_action15_anchors,
)
from franka_eef_pipeline.geometry import matrix_to_quaternion_xyzw, matrix_to_rotation_6d
from franka_eef_pipeline.stats import REQUIRED_STATS, canonical_sha256


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _state_and_carrier() -> tuple[torch.Tensor, torch.Tensor]:
    anchor_position = np.array([0.4, -0.2, 0.3], dtype=np.float64)
    anchor_rotation = _rotation_z(np.pi / 2.0)
    state = np.concatenate(
        [anchor_position, matrix_to_rotation_6d(anchor_rotation), np.array([0.25])]
    )

    step = np.arange(1, 51, dtype=np.float64)
    target_position = anchor_position + np.stack(
        [0.001 * step, -0.002 * step, 0.0005 * step], axis=-1
    )
    relative_angle = 0.002 * step
    target_rotation = np.stack(
        [anchor_rotation @ _rotation_z(angle) for angle in relative_angle], axis=0
    )
    target_quaternion = matrix_to_quaternion_xyzw(target_rotation)
    gripper = np.linspace(0.0, 1.0, 50, dtype=np.float64)[:, None]
    carrier = np.concatenate([target_position, target_quaternion, gripper], axis=-1)
    return torch.tensor(state, dtype=torch.float32), torch.tensor(carrier, dtype=torch.float32)


def _features() -> dict[str, dict[str, Any]]:
    return {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (8, 12, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "video",
            "shape": (8, 12, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_KEY: {"dtype": "float32", "shape": (10,), "names": [f"s{i}" for i in range(10)]},
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"carrier{i}" for i in range(8)],
        },
    }


class _FakeBaseDataset:
    def __init__(
        self, item: dict[str, Any], *, length: int = 60, action_fps: int = 15
    ) -> None:
        self.item = item
        self.length = length
        self.delta_timestamps = action_delta_timestamps(
            action_fps=action_fps, chunk_size=50
        )
        self.episodes = None
        self.features = _features()
        self.root = Path("/tmp/fake_franka")
        self.repo_id = "local/fake_franka"
        self.meta = SimpleNamespace(robot_type="franka_fr3_cartesian_eef")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        assert index == int(torch.as_tensor(self.item["index"]).item())
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in self.item.items()
        }


def _anchor(*, logical_index: int = 0, local_row: int = 3) -> AnchorRecord:
    return AnchorRecord(
        logical_index=logical_index,
        episode_index=0,
        raw_episode_id="episode_000",
        anchor_local_index=local_row,
        main_local_row=local_row,
        main_global_row=local_row,
        camera_log_time_ns=1_000_000 + local_row,
    )


def _base_item(
    *,
    pad_last: bool = False,
    episode_index: int = 0,
    task: str = "stack the cups",
) -> dict[str, Any]:
    state, carrier = _state_and_carrier()
    pad = torch.zeros(50, dtype=torch.bool)
    pad[-1] = pad_last
    return {
        "observation.images.camera1": torch.full((3, 8, 12), 11, dtype=torch.uint8),
        "observation.images.camera2": torch.full((3, 8, 12), 22, dtype=torch.uint8),
        STATE_KEY: state,
        ACTION_KEY: carrier,
        ACTION_PAD_KEY: pad,
        "episode_index": torch.tensor(episode_index),
        "frame_index": torch.tensor(3),
        "index": torch.tensor(3),
        "task": task,
    }


def _feature_stats(dimension: int, *, count: int) -> dict[str, list[float] | list[int]]:
    result: dict[str, list[float] | list[int]] = {}
    for name in REQUIRED_STATS:
        if name == "count":
            result[name] = [count]
        elif name == "std":
            result[name] = [1.0] * dimension
        else:
            result[name] = [0.0] * dimension
    return result


def _stats_payload(
    *,
    logical_hash: str = "logical-hash",
    profile: str = "action15",
    fps: int = 15,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": profile,
        "observation_fps": fps,
        "action_fps": fps,
        "chunk_size": 50,
        "source_dataset_hash": "source-hash",
        "logical_anchor_index_sha256": logical_hash,
        STATE_KEY: _feature_stats(10, count=1),
        ACTION_KEY: _feature_stats(7, count=50),
        "observation.images.camera1": _feature_stats(3, count=1),
        "absolute_carrier": _feature_stats(8, count=50),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_synthetic_dataset_root(
    root: Path,
    *,
    anchor_local_row: int = 3,
    profile_name: str = "action15",
    fps: int = 15,
    task_instruction: str | None = None,
) -> AnchorRecord:
    anchor = _anchor(local_row=anchor_local_row)
    logical_payload = anchor.logical_payload()
    logical_hash = canonical_sha256([logical_payload])
    profile = {
        "schema_version": 1,
        "dataset_name": f"synthetic_{profile_name}",
        "repo_id": f"local/synthetic_{profile_name}",
        "profile": profile_name,
        "observation_fps": fps,
        "action_fps": fps,
        "chunk_size": 50,
        "main_action_key": "action",
        "requires_project_cartesian_adapter": True,
        "requires_project_dual_rate_adapter": False,
        "source_dataset_hash": "source-hash",
        "logical_anchor_index_sha256": logical_hash,
        "episode_count": 1,
    }
    if task_instruction is not None:
        profile["task_instruction"] = task_instruction
    info = {
        "fps": fps,
        "features": {
            key: {**value, "shape": list(value["shape"])} for key, value in _features().items()
        },
    }
    episode_index = [
        {
            "episode_index": 0,
            "raw_episode_id": "episode_000",
            "common_camera_anchors": 61,
            "main_rows": 60,
            "valid_anchors": 1,
            "global_main_row_start": 0,
            "global_main_row_end_exclusive": 60,
            "sidecars": {"anchor_map": "sidecars/anchor_map/episode-000000.parquet"},
        }
    ]
    valid_row = {
        "episode_index": 0,
        "raw_episode_id": "episode_000",
        "camera_anchor_local_index": anchor_local_row,
        "main_local_row": anchor_local_row,
        "main_global_row": anchor_local_row,
        "camera_log_time_ns": anchor.camera_log_time_ns,
        "observation_valid": True,
        "valid_anchor_action15": True,
        "action15_local_start": anchor_local_row,
        "action15_count": 50,
    }
    invalid_row = {
        **valid_row,
        "camera_anchor_local_index": 59,
        "main_local_row": 59,
        "main_global_row": 59,
        "camera_log_time_ns": 1_000_059,
        "valid_anchor_action15": False,
        "action15_local_start": -1,
        "action15_count": 0,
    }
    monitor = {
        "schema_version": 1,
        "name": "train_monitor_subset",
        "source": "training_data",
        "held_out": False,
        "population_size": 1,
        "sample_size": 1,
        "anchors": [logical_payload],
        "anchor_list_sha256": canonical_sha256([logical_payload]),
    }

    _write_json(root / "meta/franka_eef_profile.json", profile)
    _write_json(root / "meta/info.json", info)
    _write_json(
        root / "meta/pi0_eef_stats.json",
        _stats_payload(logical_hash=logical_hash, profile=profile_name, fps=fps),
    )
    _write_json(root / "meta/train_monitor_subset.json", monitor)
    _write_json(root / "sidecars/episode_index.json", episode_index)
    anchor_map_path = root / "sidecars/anchor_map/episode-000000.parquet"
    anchor_map_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([valid_row, invalid_row]), anchor_map_path)
    return anchor


def test_cartesian_adapter_maps_cameras_and_converts_8d_carrier_to_50x7() -> None:
    base = _FakeBaseDataset(_base_item())
    dataset = CartesianAnchorDataset.from_base_dataset(
        base,
        [_anchor()],
        train_monitor_indices=[0],
        effective_stats={STATE_KEY: {}, ACTION_KEY: {}},
    )

    item = dataset[0]

    assert item[ACTION_KEY].shape == (50, 7)
    assert item[ACTION_KEY].dtype == torch.float32
    expected_step = torch.arange(1, 51, dtype=torch.float32)
    torch.testing.assert_close(item[ACTION_KEY][:, 0], 0.001 * expected_step, atol=2e-7, rtol=0)
    torch.testing.assert_close(item[ACTION_KEY][:, 1], -0.002 * expected_step, atol=2e-7, rtol=0)
    torch.testing.assert_close(item[ACTION_KEY][:, 5], 0.002 * expected_step, atol=2e-7, rtol=0)
    assert not item[ACTION_PAD_KEY].any()
    assert "observation.images.camera1" not in item
    assert "observation.images.camera2" not in item
    assert item["observation.images.base_0_rgb"][0, 0, 0].item() == 11
    assert item["observation.images.left_wrist_0_rgb"][0, 0, 0].item() == 22
    assert "observation.images.right_wrist_0_rgb" not in item
    assert item["task"] == "stack the cups"

    assert dataset.features[ACTION_KEY]["shape"] == (7,)
    assert dataset.meta.features[ACTION_KEY]["shape"] == (7,)
    assert set(dataset.meta.stats) == {STATE_KEY, ACTION_KEY}
    assert dataset.logical_anchors[0]["main_global_row"] == 3
    assert dataset.valid_anchor_indices == (3,)
    assert dataset.train_monitor_indices == (0,)


def test_absolute_carrier_helper_rejects_wrong_horizon() -> None:
    state, carrier = _state_and_carrier()
    with pytest.raises(ValueError, match="shape"):
        absolute_carrier_to_relative_action(state, carrier[:-1])


def test_cartesian_adapter_rejects_any_padded_action() -> None:
    dataset = CartesianAnchorDataset.from_base_dataset(
        _FakeBaseDataset(_base_item(pad_last=True)),
        [_anchor()],
    )
    with pytest.raises(RuntimeError, match="padded action rows"):
        dataset[0]


def test_cartesian_adapter_rejects_sidecar_episode_mismatch() -> None:
    dataset = CartesianAnchorDataset.from_base_dataset(
        _FakeBaseDataset(_base_item(episode_index=1)),
        [_anchor()],
    )
    with pytest.raises(RuntimeError, match="different episode"):
        dataset[0]


def test_direct_root_constructor_loads_sidecars_monitor_stats_and_exact_offsets(
    tmp_path: Path,
) -> None:
    _write_synthetic_dataset_root(tmp_path)
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> _FakeBaseDataset:
        captured.update(kwargs)
        dataset = _FakeBaseDataset(_base_item())
        dataset.root = Path(kwargs["root"])
        dataset.repo_id = kwargs["repo_id"]
        dataset.delta_timestamps = kwargs["delta_timestamps"]
        return dataset

    dataset = CartesianAnchorDataset(
        tmp_path,
        profile="action15",
        episode_indices=None,
        max_anchors_per_episode=None,
        video_backend="pyav",
        dataset_factory=factory,
    )

    assert len(dataset) == 1
    assert captured["delta_timestamps"] == action15_delta_timestamps()
    assert captured["video_backend"] == "pyav"
    assert captured["return_uint8"] is True
    assert captured["repo_id"] == "local/synthetic_action15"
    assert dataset.train_monitor_indices == (0,)
    assert set(dataset.effective_stats) == {STATE_KEY, ACTION_KEY}
    assert dataset[0][ACTION_KEY].shape == (50, 7)


def test_native30_root_constructor_uses_30hz_offsets_and_profile_task(
    tmp_path: Path,
) -> None:
    task = "pick up the potato chip"
    _write_synthetic_dataset_root(
        tmp_path,
        profile_name="native30",
        fps=30,
        task_instruction=task,
    )
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> _FakeBaseDataset:
        captured.update(kwargs)
        dataset = _FakeBaseDataset(_base_item(task=task), action_fps=30)
        dataset.root = Path(kwargs["root"])
        dataset.repo_id = kwargs["repo_id"]
        dataset.delta_timestamps = kwargs["delta_timestamps"]
        return dataset

    dataset = CartesianAnchorDataset(
        tmp_path,
        profile="native30",
        dataset_factory=factory,
    )

    assert dataset.observation_fps == 30
    assert dataset.action_fps == 30
    assert dataset.chunk_size == 50
    assert dataset.task_instruction == task
    assert captured["delta_timestamps"] == action_delta_timestamps(
        action_fps=30, chunk_size=50
    )
    assert captured["delta_timestamps"][ACTION_KEY][-1] == pytest.approx(49 / 30)
    assert dataset[0]["task"] == task
    assert dataset[0][ACTION_KEY].shape == (50, 7)


def test_valid_anchor_loader_rejects_episode_crossing_horizon(tmp_path: Path) -> None:
    _write_synthetic_dataset_root(tmp_path, anchor_local_row=11)
    with pytest.raises(ValueError, match="crosses the episode boundary"):
        load_valid_action15_anchors(tmp_path)


def test_max_anchors_per_episode_uniformly_covers_start_and_end() -> None:
    anchors = tuple(
        AnchorRecord(
            logical_index=index,
            episode_index=0 if index < 10 else 1,
            raw_episode_id="ep0" if index < 10 else "ep1",
            anchor_local_index=index if index < 10 else index - 10,
            main_local_row=index if index < 10 else index - 10,
            main_global_row=index,
            camera_log_time_ns=index,
        )
        for index in range(15)
    )

    selected = CartesianAnchorDataset._select_anchors(
        anchors,
        episode_indices=None,
        max_anchors_per_episode=3,
    )

    assert [anchor.logical_index for anchor in selected[:3]] == [0, 4, 9]
    assert [anchor.logical_index for anchor in selected[3:]] == [10, 12, 14]


def test_effective_stats_loader_returns_only_state_and_action(tmp_path: Path) -> None:
    path = tmp_path / "pi0_eef_stats.json"
    _write_json(path, _stats_payload())

    stats = load_effective_pi0_stats(path)

    assert set(stats) == {STATE_KEY, ACTION_KEY}
    assert stats[STATE_KEY]["mean"].shape == (10,)
    assert stats[ACTION_KEY]["mean"].shape == (7,)
    assert stats[ACTION_KEY]["count"].dtype == torch.int64


def test_effective_stats_loader_rejects_8d_action_stats(tmp_path: Path) -> None:
    path = tmp_path / "pi0_eef_stats.json"
    payload = _stats_payload()
    payload[ACTION_KEY] = _feature_stats(8, count=50)
    _write_json(path, payload)
    with pytest.raises(ValueError, match=r"action\..* must have shape \(7,\)"):
        load_effective_pi0_stats(path)


def test_action30_profile_fails_before_dataset_construction(tmp_path: Path) -> None:
    factory_called = False

    def factory(**_: Any) -> _FakeBaseDataset:
        nonlocal factory_called
        factory_called = True
        raise AssertionError("factory must not run")

    with pytest.raises(UnsupportedActionRateError, match="30 Hz"):
        CartesianAnchorDataset(tmp_path, profile="action30", dataset_factory=factory)
    assert not factory_called


def test_action30_effective_stats_fail_fast(tmp_path: Path) -> None:
    path = tmp_path / "pi0_eef_stats.json"
    payload = _stats_payload()
    payload.update({"profile": "action30", "action_fps": 30, "chunk_size": 100})
    _write_json(path, payload)
    with pytest.raises(UnsupportedActionRateError, match="30 Hz"):
        load_effective_pi0_stats(path)
