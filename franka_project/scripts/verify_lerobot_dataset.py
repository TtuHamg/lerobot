#!/usr/bin/env python3
"""Verify the paired Franka Cartesian LeRobot datasets produced by D3.

The verifier intentionally treats the 15 Hz and 30 Hz outputs as one atomic
artifact.  It checks LeRobot metadata, the audit sidecars, exact dual-rate
endpoint identities, effective PI0 statistics, the frozen training-monitor
subset, video decoding, and a small lossy-codec pixel comparison against the
source MCAP files.

Examples
--------
Verify the final 22-episode outputs (the default)::

    python franka_project/scripts/verify_lerobot_dataset.py

Verify the isolated one-episode conversion smoke::

    python franka_project/scripts/verify_lerobot_dataset.py \
      --output-base franka_project/.cache/d3_smoke --suffix _partial1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from franka_eef_pipeline.action_chunk import (  # noqa: E402
    AlignmentThresholds,
    align_episode_to_camera,
)
from franka_eef_pipeline.mcap_images import (  # noqa: E402
    iter_aligned_policy_image_pairs,
)
from franka_eef_pipeline.mcap_reader import (  # noqa: E402
    load_episode_signals,
    load_manifest,
    resolve_mcap_path,
)
from franka_eef_pipeline.stats import (  # noqa: E402
    REQUIRED_STATS,
    canonical_sha256,
    json_ready,
    normalize,
    unnormalize,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/data/franka_current_eef_conversion_22of25_v1.yaml"
CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
DEFAULT_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
EXPECTED_STATE_NAMES = [
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
]
EXPECTED_CARRIER_NAMES = [
    "target.x",
    "target.y",
    "target.z",
    "target.qx",
    "target.qy",
    "target.qz",
    "target.qw",
    "target.gripper.closed_0_1",
]


class VerificationFailure(AssertionError):
    """A dataset invariant did not hold."""


@dataclass
class CheckRecorder:
    checks: list[dict[str, Any]]

    def require(self, name: str, condition: bool, details: Any = None) -> None:
        entry = {"name": name, "status": "PASS" if condition else "FAIL"}
        if details is not None:
            entry["details"] = json_ready(details)
        self.checks.append(entry)
        if not condition:
            raise VerificationFailure(f"{name}: {details}")

    def pass_check(self, name: str, details: Any = None) -> None:
        self.require(name, True, details)


@dataclass
class DatasetView:
    key: str
    root: Path
    action_key: str
    info: dict[str, Any]
    profile: dict[str, Any]
    episodes_index: list[dict[str, Any]]
    main: pa.Table
    episodes_meta: pa.Table
    standard_stats: dict[str, Any]
    effective_stats: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet_tree(root: Path, pattern: str) -> pa.Table:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet files match {root / pattern}")
    tables = [pq.read_table(path) for path in files]
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def _column_numpy(table: pa.Table, name: str, dtype: Any | None = None) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"missing column {name!r}; have {table.column_names}")
    array = table[name].combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        result = array.values.to_numpy(zero_copy_only=False).reshape(len(array), array.type.list_size)
    else:
        result = array.to_numpy(zero_copy_only=False)
    return np.asarray(result, dtype=dtype) if dtype is not None else np.asarray(result)


def _all_finite(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.size > 0 and np.all(np.isfinite(array)))


def _thresholds(common: dict[str, Any]) -> AlignmentThresholds:
    ages = common["alignment"]["max_age_ms"]
    return AlignmentThresholds(
        cam2_age_ns=int(round(float(ages["cam2"]) * 1e6)),
        eef_age_ns=int(round(float(ages["eef"]) * 1e6)),
        gripper_age_ns=int(round(float(ages["gripper"]) * 1e6)),
        qpos_age_ns=int(round(float(ages["qpos_audit"]) * 1e6)),
    )


def _root_paths(
    config: dict[str, Any], output_base: Path | None, suffix: str
) -> dict[str, Path]:
    base = (output_base or Path(config["datasets"]["output_base"])).expanduser().resolve()
    return {
        key: base / f"{config['datasets'][key]['name']}{suffix}"
        for key in ("action15", "action30")
    }


def _load_dataset_view(
    key: str,
    root: Path,
    config: dict[str, Any],
    recorder: CheckRecorder,
) -> DatasetView:
    required = [
        "meta/info.json",
        "meta/tasks.parquet",
        "meta/stats.json",
        "meta/franka_eef_profile.json",
        "meta/pi0_eef_stats.json",
        "meta/train_monitor_subset.json",
        "sidecars/episode_index.json",
    ]
    if key == "action30":
        required.append("meta/action_30hz_carrier_stats.json")
    missing = [relative for relative in required if not (root / relative).is_file()]
    recorder.require(
        f"{key}.required_files",
        root.is_dir() and not missing,
        {"root": str(root), "missing": missing},
    )

    info = _load_json(root / "meta/info.json")
    profile = _load_json(root / "meta/franka_eef_profile.json")
    episodes_index = _load_json(root / "sidecars/episode_index.json")
    recorder.require(f"{key}.episode_index_type", isinstance(episodes_index, list) and bool(episodes_index))
    view = DatasetView(
        key=key,
        root=root,
        action_key=str(config["datasets"][key]["main_action_key"]),
        info=info,
        profile=profile,
        episodes_index=episodes_index,
        main=_read_parquet_tree(root, "data/chunk-*/file-*.parquet"),
        episodes_meta=_read_parquet_tree(root, "meta/episodes/chunk-*/file-*.parquet"),
        standard_stats=_load_json(root / "meta/stats.json"),
        effective_stats=_load_json(root / "meta/pi0_eef_stats.json"),
    )
    return view


def _verify_metadata(
    view: DatasetView,
    config: dict[str, Any],
    suffix: str,
    recorder: CheckRecorder,
) -> dict[str, int]:
    dataset_cfg = config["datasets"][view.key]
    task = str(config["contract"]["task_instruction"])
    info = view.info
    profile = view.profile
    features = info.get("features", {})
    expected_name = f"{dataset_cfg['name']}{suffix}"
    expected_repo = f"{dataset_cfg['repo_id']}{suffix}"
    episode_count = len(view.episodes_index)
    main_rows = sum(int(record["main_rows"]) for record in view.episodes_index)

    recorder.require(
        f"{view.key}.info_counts",
        info.get("codebase_version") == "v3.0"
        and int(info.get("fps", -1)) == int(config["contract"]["observation_fps"])
        and int(info.get("total_episodes", -1)) == episode_count
        and int(info.get("total_frames", -1)) == main_rows
        and int(info.get("total_tasks", -1)) == 1
        and info.get("robot_type") == "franka_fr3_cartesian_eef"
        and info.get("splits") == {"train": f"0:{episode_count}"},
        {"episodes": episode_count, "main_rows": main_rows},
    )
    expected_feature_keys = {
        *CAMERA_KEYS,
        "observation.state",
        view.action_key,
        *DEFAULT_KEYS,
    }
    recorder.require(
        f"{view.key}.feature_keys",
        set(features) == expected_feature_keys,
        {"actual": sorted(features), "expected": sorted(expected_feature_keys)},
    )
    recorder.require(
        f"{view.key}.feature_schema",
        features["observation.state"].get("dtype") == "float32"
        and features["observation.state"].get("shape") == [10]
        and features["observation.state"].get("names") == EXPECTED_STATE_NAMES
        and features[view.action_key].get("dtype") == "float32"
        and features[view.action_key].get("shape") == [8]
        and features[view.action_key].get("names") == EXPECTED_CARRIER_NAMES
        and all(
            features[camera].get("dtype") == "video"
            and features[camera].get("shape") == [
                int(config["image"]["height"]),
                int(config["image"]["width"]),
                int(config["image"]["channels"]),
            ]
            for camera in CAMERA_KEYS
        ),
    )
    recorder.require(
        f"{view.key}.profile_contract",
        profile.get("dataset_name") == expected_name
        and profile.get("repo_id") == expected_repo
        and profile.get("profile") == view.key
        and int(profile.get("observation_fps", -1)) == int(dataset_cfg["observation_fps"])
        and int(profile.get("action_fps", -1)) == int(dataset_cfg["action_fps"])
        and int(profile.get("chunk_size", -1)) == int(dataset_cfg["chunk_size"])
        and profile.get("main_action_key") == view.action_key
        and bool(profile.get("requires_project_cartesian_adapter"))
        == bool(dataset_cfg["requires_project_cartesian_adapter"])
        and bool(profile.get("requires_project_dual_rate_adapter"))
        == bool(dataset_cfg.get("requires_project_dual_rate_adapter", False))
        and bool(profile.get("train_in_first_round"))
        == bool(dataset_cfg["train_in_first_round"])
        and profile.get("task_instruction") == task
        and int(profile.get("episode_count", -1)) == episode_count
        and bool(profile.get("partial_conversion")) == bool(suffix),
    )

    tasks = pq.read_table(view.root / "meta/tasks.parquet").to_pylist()
    recorder.require(
        f"{view.key}.task_instruction",
        tasks == [{"task_index": 0, "task": task}],
        tasks,
    )

    meta_rows = sorted(view.episodes_meta.to_pylist(), key=lambda item: int(item["episode_index"]))
    expected_from = 0
    metadata_ok = len(meta_rows) == episode_count
    for episode_index, (record, meta) in enumerate(zip(view.episodes_index, meta_rows, strict=False)):
        start = int(record["global_main_row_start"])
        end = int(record["global_main_row_end_exclusive"])
        metadata_ok &= (
            int(record["episode_index"]) == episode_index
            and start == expected_from
            and end - start == int(record["main_rows"])
            and int(meta["episode_index"]) == episode_index
            and meta["tasks"] == [task]
            and int(meta["length"]) == int(record["main_rows"])
            and int(meta["dataset_from_index"]) == start
            and int(meta["dataset_to_index"]) == end
        )
        expected_from = end
    metadata_ok &= expected_from == main_rows and len(view.main) == main_rows
    recorder.require(f"{view.key}.episode_boundaries", bool(metadata_ok))

    return {"episodes": episode_count, "main_rows": main_rows}


def _verify_main_tables(
    action15: DatasetView,
    action30: DatasetView,
    recorder: CheckRecorder,
) -> None:
    left = action15.main
    right = action30.main
    recorder.require("cross_profile.main_row_count", len(left) == len(right))
    state15 = _column_numpy(left, "observation.state", np.float32)
    state30 = _column_numpy(right, "observation.state", np.float32)
    carrier15 = _column_numpy(left, action15.action_key, np.float32)
    carrier30 = _column_numpy(right, action30.action_key, np.float32)
    recorder.require(
        "cross_profile.main_float32_bit_identity",
        state15.shape[1:] == (10,)
        and carrier15.shape[1:] == (8,)
        and np.array_equal(state15, state30)
        and np.array_equal(carrier15, carrier30)
        and np.all(np.isfinite(state15))
        and np.all(np.isfinite(carrier15)),
        {"rows": len(left)},
    )
    defaults_equal = all(
        np.array_equal(_column_numpy(left, key), _column_numpy(right, key)) for key in DEFAULT_KEYS
    )
    recorder.require("cross_profile.default_columns_identity", defaults_equal)

    expected_index = np.arange(len(left), dtype=np.int64)
    frame = _column_numpy(left, "frame_index", np.int64)
    episode = _column_numpy(left, "episode_index", np.int64)
    task = _column_numpy(left, "task_index", np.int64)
    timestamp = _column_numpy(left, "timestamp", np.float32)
    default_ok = (
        np.array_equal(_column_numpy(left, "index", np.int64), expected_index)
        and np.all(task == 0)
        and np.all(np.isfinite(timestamp))
    )
    for record in action15.episodes_index:
        start = int(record["global_main_row_start"])
        end = int(record["global_main_row_end_exclusive"])
        local = np.arange(end - start, dtype=np.int64)
        default_ok &= np.array_equal(frame[start:end], local)
        default_ok &= np.all(episode[start:end] == int(record["episode_index"]))
        default_ok &= np.allclose(
            timestamp[start:end], local.astype(np.float32) / 15.0, rtol=0.0, atol=4e-6
        )
    recorder.require("main.default_column_semantics", bool(default_ok))


def _verify_sidecars(
    action15: DatasetView,
    action30: DatasetView,
    config: dict[str, Any],
    recorder: CheckRecorder,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    recorder.require(
        "cross_profile.episode_index_json_identity",
        action15.episodes_index == action30.episodes_index,
    )
    horizon = int(config["contract"]["horizon_camera_intervals"])
    main15_all = _column_numpy(action15.main, action15.action_key, np.float32)
    main30_all = _column_numpy(action30.main, action30.action_key, np.float32)
    total_anchors = 0
    total_main = 0
    total_valid = 0
    total_action30 = 0
    logical_anchors: list[dict[str, Any]] = []

    for record in action15.episodes_index:
        episode_index = int(record["episode_index"])
        episode_id = str(record["raw_episode_id"])
        anchor_rel = Path(record["sidecars"]["anchor_map"])
        qpos_rel = Path(record["sidecars"]["qpos"])
        action30_rel = Path(record["sidecars"]["action30"])
        anchor15 = pq.read_table(action15.root / anchor_rel)
        anchor30 = pq.read_table(action30.root / anchor_rel)
        qpos15 = pq.read_table(action15.root / qpos_rel)
        qpos30 = pq.read_table(action30.root / qpos_rel)
        stream30 = pq.read_table(action30.root / action30_rel)

        recorder.require(
            f"episode_{episode_index:02d}.shared_sidecar_identity",
            anchor15.equals(anchor30) and qpos15.equals(qpos30),
            episode_id,
        )
        anchor = anchor15
        count = len(anchor)
        main_rows = count - 1
        start = int(record["global_main_row_start"])
        end = int(record["global_main_row_end_exclusive"])
        local = np.arange(count, dtype=np.int64)
        camera_time = _column_numpy(anchor, "camera_log_time_ns", np.int64)
        observation_valid = _column_numpy(anchor, "observation_valid", bool)
        valid15 = _column_numpy(anchor, "valid_anchor_action15", bool)
        valid30 = _column_numpy(anchor, "valid_anchor_action30", bool)
        qpos_index = _column_numpy(anchor, "qpos_raw_index", np.int64)
        qpos_source = _column_numpy(anchor, "qpos_source_log_time_ns", np.int64)
        qpos_age = _column_numpy(anchor, "qpos_age_ns", np.int64)

        structural_ok = (
            count == int(record["common_camera_anchors"])
            and main_rows == int(record["main_rows"])
            and end - start == main_rows
            and np.all(_column_numpy(anchor, "episode_index", np.int64) == episode_index)
            and np.all(_column_numpy(anchor, "raw_episode_id").astype(str) == episode_id)
            and np.array_equal(_column_numpy(anchor, "camera_anchor_local_index", np.int64), local)
            and np.array_equal(_column_numpy(anchor, "main_local_row", np.int64), np.r_[local[:-1], -1])
            and np.array_equal(
                _column_numpy(anchor, "main_global_row", np.int64),
                np.r_[np.arange(start, end, dtype=np.int64), -1],
            )
            and np.all(np.diff(camera_time) > 0)
        )

        for prefix in ("cam2", "eef", "gripper"):
            source = _column_numpy(anchor, f"{prefix}_source_log_time_ns", np.int64)
            age = _column_numpy(anchor, f"{prefix}_age_ns", np.int64)
            structural_ok &= np.all(source <= camera_time)
            structural_ok &= np.array_equal(age, camera_time - source)
        qpos_available = qpos_index >= 0
        structural_ok &= np.all(qpos_source[qpos_available] <= camera_time[qpos_available])
        structural_ok &= np.array_equal(
            qpos_age[qpos_available], camera_time[qpos_available] - qpos_source[qpos_available]
        )

        complete = local + horizon < count
        structural_ok &= np.array_equal(
            _column_numpy(anchor, "action15_local_start", np.int64), np.where(complete, local, -1)
        )
        structural_ok &= np.array_equal(
            _column_numpy(anchor, "action15_count", np.int32),
            np.where(complete, horizon, 0).astype(np.int32),
        )
        structural_ok &= np.array_equal(
            _column_numpy(anchor, "action30_local_start", np.int64),
            np.where(complete, 2 * local, -1),
        )
        structural_ok &= np.array_equal(
            _column_numpy(anchor, "action30_count", np.int32),
            np.where(complete, 2 * horizon, 0).astype(np.int32),
        )
        recorder.require(f"episode_{episode_index:02d}.anchor_map", bool(structural_ok), episode_id)

        qpos = _column_numpy(qpos15, "qpos", np.float64)
        qpos_log = _column_numpy(qpos15, "log_time_ns", np.int64)
        qpos_ok = (
            qpos.shape[1:] == (7,)
            and np.all(np.isfinite(qpos))
            and np.all(np.diff(qpos_log) > 0)
            and np.array_equal(_column_numpy(qpos15, "raw_index", np.int64), np.arange(len(qpos15)))
            and np.all(_column_numpy(qpos15, "episode_index", np.int64) == episode_index)
            and np.array_equal(qpos_source[qpos_available], qpos_log[qpos_index[qpos_available]])
        )
        recorder.require(f"episode_{episode_index:02d}.qpos_audit", bool(qpos_ok), {"rows": len(qpos15)})

        action_count = 2 * main_rows
        target30 = _column_numpy(stream30, "target_log_time_ns", np.int64)
        eef_source30 = _column_numpy(stream30, "eef_source_log_time_ns", np.int64)
        gripper_source30 = _column_numpy(stream30, "gripper_source_log_time_ns", np.int64)
        side_carrier = _column_numpy(stream30, "carrier", np.float32)
        side_valid = _column_numpy(stream30, "valid", bool)
        midpoint_target = camera_time[:-1] + (camera_time[1:] - camera_time[:-1]) // 2
        stream_ok = (
            len(stream30) == action_count == int(record["action30_carrier_rows"])
            and side_carrier.shape == (action_count, 8)
            and np.all(np.isfinite(side_carrier))
            and np.array_equal(
                _column_numpy(stream30, "action_local_index", np.int64),
                np.arange(action_count, dtype=np.int64),
            )
            and np.array_equal(
                _column_numpy(stream30, "interval_index", np.int64),
                np.repeat(np.arange(main_rows, dtype=np.int64), 2),
            )
            and np.array_equal(
                _column_numpy(stream30, "slot", np.int8),
                np.tile(np.asarray([0, 1], dtype=np.int8), main_rows),
            )
            and np.array_equal(target30[0::2], midpoint_target)
            and np.array_equal(target30[1::2], camera_time[1:])
            and np.all(eef_source30 <= target30)
            and np.all(gripper_source30 <= target30)
            and np.array_equal(
                _column_numpy(stream30, "eef_raw_index", np.int64)[1::2],
                _column_numpy(anchor, "eef_raw_index", np.int64)[1:],
            )
            and np.array_equal(
                _column_numpy(stream30, "gripper_raw_index", np.int64)[1::2],
                _column_numpy(anchor, "gripper_raw_index", np.int64)[1:],
            )
            and np.array_equal(
                eef_source30[1::2],
                _column_numpy(anchor, "eef_source_log_time_ns", np.int64)[1:],
            )
            and np.array_equal(
                gripper_source30[1::2],
                _column_numpy(anchor, "gripper_source_log_time_ns", np.int64)[1:],
            )
            and np.array_equal(side_valid[1::2], observation_valid[1:])
        )
        recorder.require(
            f"episode_{episode_index:02d}.action30_timestamps",
            bool(stream_ok),
            {"rows": action_count},
        )

        main15 = main15_all[start:end]
        main30 = main30_all[start:end]
        endpoint_ok = (
            np.array_equal(side_carrier[1::2], main15)
            and np.array_equal(side_carrier[1::2], main30)
            and np.array_equal(main15, main30)
        )
        recorder.require(
            f"episode_{episode_index:02d}.endpoint_float32_bit_identity",
            endpoint_ok,
            {"endpoints": main_rows},
        )

        expected15 = np.zeros(count, dtype=bool)
        expected30 = np.zeros(count, dtype=bool)
        for anchor_index in range(max(0, count - horizon)):
            expected15[anchor_index] = bool(
                observation_valid[anchor_index]
                and np.all(observation_valid[anchor_index + 1 : anchor_index + horizon + 1])
            )
            expected30[anchor_index] = bool(
                observation_valid[anchor_index]
                and np.all(side_valid[2 * anchor_index : 2 * anchor_index + 2 * horizon])
            )
        valid_indices = np.flatnonzero(valid15)
        horizon_ok = np.array_equal(valid15, expected15) and np.array_equal(valid30, expected30)
        horizon_ok &= np.array_equal(valid15, valid30)
        for anchor_index in valid_indices:
            first30 = 2 * int(anchor_index)
            stop30 = first30 + 2 * horizon
            horizon_ok &= (
                target30[first30] == midpoint_target[anchor_index]
                and target30[first30 + 1] == camera_time[anchor_index + 1]
                and target30[stop30 - 1] == camera_time[anchor_index + horizon]
                and np.array_equal(
                    target30[first30 + 1 : stop30 : 2],
                    camera_time[anchor_index + 1 : anchor_index + horizon + 1],
                )
                and np.all(side_valid[first30:stop30])
            )
        recorder.require(
            f"episode_{episode_index:02d}.valid_anchors_and_t_plus_1_t_plus_50",
            bool(horizon_ok),
            {"valid": int(len(valid_indices)), "horizon": horizon},
        )
        recorder.require(
            f"episode_{episode_index:02d}.record_counts",
            int(record["valid_anchors"]) == len(valid_indices),
        )

        for anchor_index in valid_indices.tolist():
            logical_anchors.append(
                {
                    "logical_index": len(logical_anchors),
                    "episode_index": episode_index,
                    "raw_episode_id": episode_id,
                    "anchor_local_index": int(anchor_index),
                    "main_global_row": int(start + anchor_index),
                    "camera_log_time_ns": int(camera_time[anchor_index]),
                }
            )
        total_anchors += count
        total_main += main_rows
        total_valid += len(valid_indices)
        total_action30 += action_count

    counts = {
        "episodes": len(action15.episodes_index),
        "common_camera_anchors": total_anchors,
        "main_rows": total_main,
        "valid_anchors": total_valid,
        "action15_effective_stats_count": total_valid * horizon,
        "action30_carrier_rows": total_action30,
        "action30_effective_stats_count": total_valid * 2 * horizon,
    }
    recorder.pass_check("sidecars.aggregate_counts", counts)
    return counts, logical_anchors


def _stats_feature_ok(feature: dict[str, Any], shape: tuple[int, ...]) -> bool:
    if not all(field in feature for field in REQUIRED_STATS):
        return False
    for field in REQUIRED_STATS:
        expected_shape = (1,) if field == "count" else shape
        array = np.asarray(feature[field])
        if array.shape != expected_shape or not _all_finite(array):
            return False
    count = np.asarray(feature["count"], dtype=np.int64)
    return bool(count[0] > 0)


def _roundtrip_error(feature: dict[str, Any], epsilon: float) -> float:
    probes = np.stack(
        [
            np.asarray(feature["min"], dtype=np.float64),
            np.asarray(feature["mean"], dtype=np.float64),
            np.asarray(feature["max"], dtype=np.float64),
        ]
    )
    restored = unnormalize(normalize(probes, feature, epsilon=epsilon), feature, epsilon=epsilon)
    return float(np.max(np.abs(restored - probes)))


def _verify_stats(
    views: dict[str, DatasetView],
    config: dict[str, Any],
    counts: dict[str, int],
    recorder: CheckRecorder,
) -> dict[str, Any]:
    epsilon = float(config["statistics"]["epsilon"])
    summary: dict[str, Any] = {}
    for key, view in views.items():
        stats = view.standard_stats
        action_key = view.action_key
        required = {*CAMERA_KEYS, "observation.state", action_key, *DEFAULT_KEYS}
        shapes = {
            "observation.state": (10,),
            action_key: (8,),
            CAMERA_KEYS[0]: (3, 1, 1),
            CAMERA_KEYS[1]: (3, 1, 1),
            **{name: (1,) for name in DEFAULT_KEYS},
        }
        standard_ok = set(stats) == required and all(
            _stats_feature_ok(stats[name], shapes[name]) for name in required
        )
        standard_ok &= int(stats["observation.state"]["count"][0]) == counts["main_rows"]
        standard_ok &= int(stats[action_key]["count"][0]) == counts["main_rows"]
        recorder.require(f"{key}.standard_stats", bool(standard_ok))

        effective = view.effective_stats
        chunk = int(config["datasets"][key]["chunk_size"])
        action_fps = int(config["datasets"][key]["action_fps"])
        expected_action_count = counts[f"{key}_effective_stats_count"]
        effective_ok = (
            int(effective.get("schema_version", -1)) == 1
            and int(effective.get("observation_fps", -1)) == 15
            and int(effective.get("action_fps", -1)) == action_fps
            and int(effective.get("chunk_size", -1)) == chunk
            and effective.get("profile") == key
            and int(effective.get("episode_count", -1)) == counts["episodes"]
            and _stats_feature_ok(effective["observation.state"], (10,))
            and _stats_feature_ok(effective["action"], (7,))
            and int(effective["observation.state"]["count"][0]) == counts["valid_anchors"]
            and int(effective["action"]["count"][0]) == expected_action_count
            and effective.get("source_dataset_hash") == view.profile.get("source_dataset_hash")
            and effective.get("logical_anchor_index_sha256")
            == view.profile.get("logical_anchor_index_sha256")
        )
        recorder.require(f"{key}.effective_pi0_stats", bool(effective_ok))
        errors = {
            "standard_state": _roundtrip_error(stats["observation.state"], epsilon),
            "standard_carrier": _roundtrip_error(stats[action_key], epsilon),
            "effective_state": _roundtrip_error(effective["observation.state"], epsilon),
            "effective_action": _roundtrip_error(effective["action"], epsilon),
        }
        recorder.require(
            f"{key}.normalization_roundtrip",
            max(errors.values()) <= 1e-12,
            errors,
        )
        summary[key] = {
            "standard_state_count": int(stats["observation.state"]["count"][0]),
            "standard_carrier_count": int(stats[action_key]["count"][0]),
            "effective_state_count": int(effective["observation.state"]["count"][0]),
            "effective_action_count": int(effective["action"]["count"][0]),
            "max_normalization_roundtrip_error": max(errors.values()),
        }

    recorder.require(
        "cross_profile.effective_state_stats_identity",
        views["action15"].effective_stats["observation.state"]
        == views["action30"].effective_stats["observation.state"],
    )
    carrier_stats = _load_json(views["action30"].root / "meta/action_30hz_carrier_stats.json")
    carrier_ok = (
        int(carrier_stats.get("observation_fps", -1)) == 15
        and int(carrier_stats.get("action_fps", -1)) == 30
        and carrier_stats.get("carrier_shape") == [8]
        and _stats_feature_ok(carrier_stats["carrier"], (8,))
        and int(carrier_stats["carrier"]["count"][0]) == counts["action30_carrier_rows"]
        and carrier_stats.get("source_dataset_hash")
        == views["action30"].profile.get("source_dataset_hash")
    )
    recorder.require("action30.carrier_sidecar_stats", bool(carrier_ok))
    summary["action30_carrier_count"] = int(carrier_stats["carrier"]["count"][0])
    return summary


def _verify_monitor(
    views: dict[str, DatasetView],
    config: dict[str, Any],
    logical_anchors: list[dict[str, Any]],
    recorder: CheckRecorder,
) -> dict[str, Any]:
    monitor15 = _load_json(views["action15"].root / "meta/train_monitor_subset.json")
    monitor30 = _load_json(views["action30"].root / "meta/train_monitor_subset.json")
    recorder.require("cross_profile.monitor_file_identity", monitor15 == monitor30)
    monitor = monitor15
    cfg = config["monitor"]
    expected_size = min(
        int(cfg["maximum_size"]),
        int(math.ceil(float(cfg["fraction"]) * len(logical_anchors))),
    )
    logical_hash = canonical_sha256(logical_anchors)
    anchors = monitor.get("anchors", [])
    index_by_logical = {int(item["logical_index"]): item for item in logical_anchors}
    selected = [int(item["logical_index"]) for item in anchors]
    subset_ok = (
        monitor.get("name") == cfg["name"]
        and monitor.get("source") == "training_data"
        and monitor.get("held_out") is False
        and int(monitor.get("seed", -1)) == int(cfg["seed"])
        and int(monitor.get("population_size", -1)) == len(logical_anchors)
        and int(monitor.get("sample_size", -1)) == expected_size == len(anchors)
        and selected == sorted(selected)
        and len(selected) == len(set(selected))
        and all(
            index_by_logical.get(index) == anchor
            for index, anchor in zip(selected, anchors, strict=True)
        )
        and monitor.get("logical_anchor_index_sha256") == logical_hash
        and monitor.get("anchor_list_sha256") == canonical_sha256(anchors)
    )
    recorder.require(
        "monitor.valid_training_anchor_subset",
        bool(subset_ok),
        {"population": len(logical_anchors), "sample": len(anchors)},
    )
    recorder.require(
        "monitor.provenance_hashes",
        all(
            views[key].profile.get("logical_anchor_index_sha256") == logical_hash
            and views[key].effective_stats.get("logical_anchor_index_sha256") == logical_hash
            for key in views
        ),
    )
    return {
        "population_size": len(logical_anchors),
        "sample_size": len(anchors),
        "logical_anchor_index_sha256": logical_hash,
        "anchor_list_sha256": monitor["anchor_list_sha256"],
    }


def _verify_30hz_fail_fast_marker(view: DatasetView, recorder: CheckRecorder) -> dict[str, Any]:
    marker = {
        "main_action_key": view.profile.get("main_action_key"),
        "action_feature_present": "action" in view.info.get("features", {}),
        "requires_project_cartesian_adapter": view.profile.get("requires_project_cartesian_adapter"),
        "requires_project_dual_rate_adapter": view.profile.get("requires_project_dual_rate_adapter"),
        "train_in_first_round": view.profile.get("train_in_first_round"),
    }
    recorder.require(
        "action30.fail_fast_profile_marker",
        marker["main_action_key"] == "carrier.endpoint_15hz"
        and marker["action_feature_present"] is False
        and marker["requires_project_cartesian_adapter"] is True
        and marker["requires_project_dual_rate_adapter"] is True
        and marker["train_in_first_round"] is False,
        marker,
    )
    return marker


def _episode_sample_indices(records: list[dict[str, Any]], random_count: int, seed: int) -> list[int]:
    selected: set[int] = set()
    episode_positions = sorted({0, len(records) // 2, len(records) - 1})
    for position in episode_positions:
        record = records[position]
        start = int(record["global_main_row_start"])
        end = int(record["global_main_row_end_exclusive"])
        selected.add(start)
        selected.add(end - 1)
    total = int(records[-1]["global_main_row_end_exclusive"])
    if random_count > 0:
        rng = np.random.default_rng(seed)
        size = min(random_count, total)
        selected.update(int(item) for item in rng.choice(total, size=size, replace=False))
    return sorted(selected)


def _to_hwc_uint8(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3:
        raise VerificationFailure(f"decoded video frame must be 3D, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if float(np.max(array)) <= 1.0 + 1e-6:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0)
        else:
            array = np.rint(np.clip(array, 0.0, 255.0))
    return np.ascontiguousarray(array, dtype=np.uint8)


def _pixel_metrics(reference: np.ndarray, decoded: np.ndarray) -> dict[str, float | None]:
    if reference.shape != decoded.shape:
        raise VerificationFailure(f"pixel shape mismatch: raw={reference.shape}, decoded={decoded.shape}")
    difference = decoded.astype(np.float64) - reference.astype(np.float64)
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(np.square(difference)))
    # JSON has no representation for infinity.  ``None`` means mathematically
    # infinite PSNR (the two uint8 images are bit-identical).
    psnr = None if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    return {"mae": mae, "psnr_db": psnr}


def _decode_videos(
    views: dict[str, DatasetView],
    config: dict[str, Any],
    sample_indices: list[int],
    recorder: CheckRecorder,
) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], dict[str, Any]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    task = str(config["contract"]["task_instruction"])
    expected_shape = (
        int(config["image"]["height"]),
        int(config["image"]["width"]),
        int(config["image"]["channels"]),
    )
    decoded: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    report: dict[str, Any] = {"sample_indices": sample_indices, "profiles": {}}
    for key, view in views.items():
        dataset = LeRobotDataset(
            repo_id=str(view.profile["repo_id"]),
            root=view.root,
            return_uint8=True,
            video_backend=str(config["video"]["video_backend"]),
        )
        recorder.require(
            f"{key}.lerobot_dataset_counts",
            len(dataset) == len(view.main)
            and dataset.meta.total_episodes == len(view.episodes_index),
            {"frames": len(dataset), "episodes": dataset.meta.total_episodes},
        )
        decoded[key] = {}
        sample_ok = True
        for index in sample_indices:
            item = dataset[index]
            sample_ok &= item.get("task") == task
            state = np.asarray(item["observation.state"])
            carrier = np.asarray(item[view.action_key])
            sample_ok &= state.shape == (10,) and carrier.shape == (8,)
            sample_ok &= np.all(np.isfinite(state)) and np.all(np.isfinite(carrier))
            decoded[key][index] = {}
            for camera in CAMERA_KEYS:
                image = _to_hwc_uint8(item[camera])
                sample_ok &= image.shape == expected_shape
                decoded[key][index][camera] = image
        recorder.require(
            f"{key}.random_and_boundary_video_decode",
            bool(sample_ok),
            {"samples": len(sample_indices), "shape": expected_shape},
        )
        report["profiles"][key] = {"decoded_samples": len(sample_indices)}

    cross_metrics: list[dict[str, Any]] = []
    cross_ok = True
    for index in sample_indices:
        for camera in CAMERA_KEYS:
            metric = _pixel_metrics(decoded["action15"][index][camera], decoded["action30"][index][camera])
            cross_ok &= metric["mae"] == 0.0
            cross_metrics.append({"index": index, "camera": camera, **metric})
    recorder.require(
        "cross_profile.decoded_video_bit_identity",
        bool(cross_ok),
        {"comparisons": len(cross_metrics)},
    )
    report["cross_profile"] = {
        "comparisons": len(cross_metrics),
        "max_mae": max((item["mae"] for item in cross_metrics), default=0.0),
        "min_psnr_db": min(
            (item["psnr_db"] for item in cross_metrics if item["psnr_db"] is not None),
            default=None,
        ),
        "infinite_psnr_means_bit_exact": True,
    }
    return decoded, report


def _raw_pixel_qa(
    views: dict[str, DatasetView],
    config: dict[str, Any],
    max_episodes: int,
    max_mae: float,
    min_psnr_db: float,
    recorder: CheckRecorder,
) -> dict[str, Any]:
    manifest = load_manifest(Path(config["scope"]["manifest"]))
    manifest_by_id = {str(item["episode_id"]): item for item in manifest["episodes"]}
    common = _load_yaml(Path(config["contract"]["common"]))
    thresholds = _thresholds(common)
    records = views["action15"].episodes_index
    positions = sorted({0, len(records) // 2, len(records) - 1})[: max(0, max_episodes)]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=str(views["action15"].profile["repo_id"]),
        root=views["action15"].root,
        return_uint8=True,
        video_backend=str(config["video"]["video_backend"]),
    )
    comparisons: list[dict[str, Any]] = []
    for position in positions:
        record = records[position]
        episode_id = str(record["raw_episode_id"])
        manifest_episode = manifest_by_id[episode_id]
        mcap_path = resolve_mcap_path(manifest_episode, data_root=manifest["data_root"])
        signals = load_episode_signals(mcap_path, episode_id=episode_id)
        aligned = align_episode_to_camera(signals, thresholds)
        main_rows = int(record["main_rows"])
        wanted = sorted({0, main_rows // 2, main_rows - 1})
        wanted_set = set(wanted)
        global_start = int(record["global_main_row_start"])
        decoded_cache: dict[int, dict[str, np.ndarray]] = {}
        for local_index in wanted:
            item = dataset[global_start + local_index]
            decoded_cache[local_index] = {
                camera: _to_hwc_uint8(item[camera]) for camera in CAMERA_KEYS
            }
        found: set[int] = set()
        for pair in iter_aligned_policy_image_pairs(mcap_path, aligned):
            if pair.row_index not in wanted_set:
                continue
            found.add(pair.row_index)
            raw_by_camera = {
                CAMERA_KEYS[0]: pair.cam1,
                CAMERA_KEYS[1]: pair.cam2,
            }
            for camera, raw_image in raw_by_camera.items():
                metric = _pixel_metrics(raw_image, decoded_cache[pair.row_index][camera])
                comparisons.append(
                    {
                        "episode_index": int(record["episode_index"]),
                        "raw_episode_id": episode_id,
                        "local_row": int(pair.row_index),
                        "global_row": global_start + int(pair.row_index),
                        "camera": camera,
                        **metric,
                    }
                )
        if found != wanted_set:
            raise VerificationFailure(
                f"{episode_id}: raw image QA found rows {sorted(found)}, expected {wanted}"
            )

    raw_ok = bool(comparisons) and all(
        item["mae"] <= max_mae
        and (item["psnr_db"] is None or item["psnr_db"] >= min_psnr_db)
        for item in comparisons
    )
    summary = {
        "codec_is_lossy": True,
        "comparison_is_bit_exact": False,
        "thresholds": {"max_mae": max_mae, "min_psnr_db": min_psnr_db},
        "episodes_sampled": len(positions),
        "comparisons": len(comparisons),
        "mean_mae": float(np.mean([item["mae"] for item in comparisons])) if comparisons else None,
        "max_mae": max((item["mae"] for item in comparisons), default=None),
        "min_psnr_db": min(
            (item["psnr_db"] for item in comparisons if item["psnr_db"] is not None),
            default=None,
        ),
        "samples": comparisons,
    }
    recorder.require("video.raw_mcap_lossy_pixel_qa", raw_ok, summary)
    return summary


def _verify_full_expected_counts(
    counts: dict[str, int], config: dict[str, Any], suffix: str, recorder: CheckRecorder
) -> None:
    partial_match = re.fullmatch(r"_partial([1-9][0-9]*)", suffix)
    if suffix:
        recorder.require(
            "partial_suffix_contract",
            partial_match is not None
            and counts["episodes"] == int(partial_match.group(1)),
            {"suffix": suffix, "episodes": counts["episodes"]},
        )
        return
    expected = config["expected"]
    mapping = {
        "common_camera_anchors": "common_camera_anchors",
        "main_rows": "main_rows_per_dataset",
        "valid_anchors": "valid_anchor_count",
        "action15_effective_stats_count": "action15_effective_stats_count",
        "action30_carrier_rows": "action30_carrier_rows",
        "action30_effective_stats_count": "action30_effective_stats_count",
    }
    expected_counts = {actual: int(expected[source]) for actual, source in mapping.items()}
    recorder.require(
        "full_22of25_expected_counts",
        counts["episodes"] == int(config["scope"]["episode_count"])
        and all(counts[name] == value for name, value in expected_counts.items()),
        {"actual": counts, "expected": {"episodes": config["scope"]["episode_count"], **expected_counts}},
    )


def _render_markdown(report: dict[str, Any]) -> str:
    counts = report.get("counts", {})
    video = report.get("video", {})
    raw = video.get("raw_mcap_pixel_qa", {})
    check_rows = "\n".join(
        f"| `{item['name']}` | {item['status']} |"
        for item in report.get("checks", [])
    )
    failure = report.get("failure")
    failure_text = f"\nFailure: `{failure}`\n" if failure else ""
    return f"""# D4 Franka LeRobot dataset verification

- Status: **{report.get('status', 'UNKNOWN')}**
- Generated: `{report.get('generated_at_utc', '')}`
- Config: `{report.get('config', '')}`
- Action-15 root: `{report.get('roots', {}).get('action15', '')}`
- Action-30 root: `{report.get('roots', {}).get('action30', '')}`
{failure_text}
## Verified counts

```json
{json.dumps(counts, indent=2, ensure_ascii=False)}
```

## Video QA

- Random/boundary decoded samples: {video.get('decode', {}).get('sample_indices', [])}
- Raw-MCAP comparisons: {raw.get('comparisons', 'not run')}
- Raw-MCAP maximum MAE: {raw.get('max_mae', 'not run')}
- Raw-MCAP minimum PSNR (dB): {raw.get('min_psnr_db', 'not run')}
- H.264 source comparison is intentionally thresholded, not bit-exact.

## Checks

| Check | Result |
| --- | --- |
{check_rows}
"""


def _default_report_paths(suffix: str) -> tuple[Path, Path]:
    if suffix:
        label = suffix.lstrip("_").lower()
        machine = PROJECT_ROOT / "artifacts/verification" / f"d4_verify_{label}.json"
        markdown = PROJECT_ROOT / "reports" / f"D4_DATASET_VERIFICATION_{label.upper()}.md"
    else:
        machine = PROJECT_ROOT / "artifacts/verification/d4_verify_22of25.json"
        markdown = PROJECT_ROOT / "reports/D4_DATASET_VERIFICATION_22OF25.md"
    return machine, markdown


def verify(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    roots = _root_paths(config, args.output_base, args.suffix)
    recorder = CheckRecorder(checks=[])
    report: dict[str, Any] = {
        "schema_version": 1,
        "phase": "D4",
        "status": "RUNNING",
        "generated_at_utc": _utc_now(),
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "suffix": args.suffix,
        "roots": {key: str(value) for key, value in roots.items()},
        "checks": recorder.checks,
    }

    try:
        views = {
            key: _load_dataset_view(key, root, config, recorder)
            for key, root in roots.items()
        }
        metadata_counts = {
            key: _verify_metadata(view, config, args.suffix, recorder)
            for key, view in views.items()
        }
        recorder.require(
            "cross_profile.metadata_counts_identity",
            metadata_counts["action15"] == metadata_counts["action30"],
            metadata_counts,
        )
        _verify_main_tables(views["action15"], views["action30"], recorder)
        counts, logical_anchors = _verify_sidecars(
            views["action15"], views["action30"], config, recorder
        )
        _verify_full_expected_counts(counts, config, args.suffix, recorder)
        report["counts"] = counts
        report["stats"] = _verify_stats(views, config, counts, recorder)
        report["monitor"] = _verify_monitor(views, config, logical_anchors, recorder)
        report["action30_fail_fast_marker"] = _verify_30hz_fail_fast_marker(
            views["action30"], recorder
        )

        sample_indices = _episode_sample_indices(
            views["action15"].episodes_index,
            random_count=args.random_video_samples,
            seed=args.seed,
        )
        _, decode_report = _decode_videos(
            views, config, sample_indices, recorder
        )
        report["video"] = {"decode": decode_report}
        if args.skip_raw_pixel_qa:
            recorder.pass_check("video.raw_mcap_lossy_pixel_qa", {"status": "SKIPPED_BY_CLI"})
            report["video"]["raw_mcap_pixel_qa"] = {"status": "SKIPPED_BY_CLI"}
        else:
            report["video"]["raw_mcap_pixel_qa"] = _raw_pixel_qa(
                views,
                config,
                max_episodes=args.raw_video_episodes,
                max_mae=args.max_video_mae,
                min_psnr_db=args.min_video_psnr_db,
                recorder=recorder,
            )
        report["status"] = "PASS_PARTIAL_SMOKE" if args.suffix else "PASS_FULL_VERIFICATION"
    except Exception as error:
        report["status"] = "FAIL"
        report["failure"] = f"{type(error).__name__}: {error}"
        if not recorder.checks or recorder.checks[-1].get("status") != "FAIL":
            recorder.checks.append(
                {
                    "name": "unhandled_verification_error",
                    "status": "FAIL",
                    "details": report["failure"],
                }
            )

    machine_default, markdown_default = _default_report_paths(args.suffix)
    machine = (args.report_json or machine_default).expanduser().resolve()
    markdown = (args.report_markdown or markdown_default).expanduser().resolve()
    report["outputs"] = {"machine_report": str(machine), "markdown_report": str(markdown)}
    _write_json(machine, report)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_render_markdown(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-base",
        type=Path,
        help="Override the dataset parent directory (e.g. franka_project/.cache/d3_smoke).",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to both configured dataset names (e.g. _partial1).",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--random-video-samples", type=int, default=3)
    parser.add_argument(
        "--raw-video-episodes",
        type=int,
        default=3,
        help="Sample at most this many first/middle/last episodes for raw-MCAP pixel QA.",
    )
    parser.add_argument("--max-video-mae", type=float, default=12.0)
    parser.add_argument("--min-video-psnr-db", type=float, default=25.0)
    parser.add_argument(
        "--skip-raw-pixel-qa",
        action="store_true",
        help="Skip the source-MCAP pixel reread; all encoded video decode checks still run.",
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    report = verify(parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report.get("counts"),
                "failure": report.get("failure"),
                "outputs": report["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
