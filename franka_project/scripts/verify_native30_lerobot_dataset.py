#!/usr/bin/env python3
"""Verify the native 30 Hz Franka chips LeRobot dataset.

This verifier is deliberately separate from ``verify_lerobot_dataset.py``.
That older verifier is a frozen, paired obs15/action15 + obs15/action30 audit and
must not be generalized to the native obs30/action30 dataset.

Two verification depths are available:

``smoke``
    Check the complete on-disk schema and every episode boundary/anchor map,
    decode a small camera sample, and replay numeric alignment for the
    first/middle/last raw episodes.

``full``
    Perform all smoke checks, replay alignment for every raw episode, compare
    the main-table state/carrier rows to the raw-derived values, and recompute
    the effective PI0 state/action statistics.

Examples
--------
Verify a one-episode conversion smoke::

    python franka_project/scripts/verify_native30_lerobot_dataset.py \
      --limit-episodes 1 --mode smoke

Verify the final 46-episode dataset::

    python franka_project/scripts/verify_native30_lerobot_dataset.py --mode full
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    GripperMapping,
    align_episode_to_camera,
    build_dual_rate_carriers,
    model_absolute_action_chunks,
    model_relative_action_chunks,
    valid_anchor_indices,
)
from franka_eef_pipeline.dual_rate_dataset import (  # noqa: E402
    ABSOLUTE_CARRIER_NAMES,
    ACTION_LABEL_MODE_ABSOLUTE_EEF,
    ACTION_LABEL_MODE_DELTA_EEF,
    STATE_NAMES,
    resolve_data_config_action_label_spec,
    resolve_action_label_spec,
)
from franka_eef_pipeline.mcap_reader import (  # noqa: E402
    load_episode_signals,
    load_manifest,
    resolve_mcap_path,
)
from franka_eef_pipeline.stats import (  # noqa: E402
    REQUIRED_STATS,
    canonical_sha256,
    feature_stats,
    json_ready,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/data/franka_chips_current_eef_native30_v1.yaml"
CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
DEFAULT_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
STATE_KEY = "observation.state"
ACTION_KEY = "action"
PROFILE = "native30"
FPS = 30
CHUNK_SIZE = 50
STATE_DIM = 10
CARRIER_DIM = 8
MODEL_ACTION_DIM = 7

CARRIER_NAMES = ABSOLUTE_CARRIER_NAMES


class VerificationFailure(AssertionError):
    """A native-30 dataset invariant did not hold."""


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

    def passed(self, name: str, details: Any = None) -> None:
        self.require(name, True, details)


@dataclass
class DatasetView:
    root: Path
    info: dict[str, Any]
    profile: dict[str, Any]
    episode_index: list[dict[str, Any]]
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


def _dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("config must define dataset")
    return dataset


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
        result = array.values.to_numpy(zero_copy_only=False).reshape(
            len(array), array.type.list_size
        )
    else:
        result = array.to_numpy(zero_copy_only=False)
    return np.asarray(result, dtype=dtype) if dtype is not None else np.asarray(result)


def _resolve_sidecar(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError(f"sidecar path escapes dataset root: {relative}")
    return path


def _thresholds(config: dict[str, Any]) -> AlignmentThresholds:
    ages = config["alignment"]["max_age_ms"]
    return AlignmentThresholds(
        cam2_age_ns=int(round(float(ages["cam2"]) * 1e6)),
        eef_age_ns=int(round(float(ages["eef"]) * 1e6)),
        gripper_age_ns=int(round(float(ages["gripper"]) * 1e6)),
        qpos_age_ns=int(round(float(ages["qpos_audit"]) * 1e6)),
    )


def _gripper_mapping(config: dict[str, Any]) -> GripperMapping:
    gripper = config["gripper"]
    return GripperMapping(
        raw_open=float(gripper["raw_open"]),
        raw_closed=float(gripper["raw_closed"]),
        clip=bool(gripper["clip"]),
    )


def _scope_policy(
    scope: dict[str, Any],
) -> tuple[tuple[str, ...], bool, dict[str, str]]:
    configured = scope.get("allowed_statuses")
    if configured is None:
        required = scope.get("required_status")
        if not isinstance(required, str) or not required:
            raise ValueError("scope must define required_status or allowed_statuses")
        allowed_statuses = (required,)
    else:
        if "required_status" in scope:
            raise ValueError(
                "scope.required_status and scope.allowed_statuses are mutually exclusive"
            )
        if (
            not isinstance(configured, list)
            or not configured
            or any(not isinstance(status, str) or not status for status in configured)
        ):
            raise ValueError("scope.allowed_statuses must be a non-empty list of strings")
        allowed_statuses = tuple(configured)
        if len(set(allowed_statuses)) != len(allowed_statuses):
            raise ValueError("scope.allowed_statuses must not contain duplicates")

    require_eligible = bool(scope.get("require_ok_for_training", True))
    configured_fields = scope.get("training_eligibility_fields")
    if configured_fields is None:
        configured_field = scope.get("training_eligibility_field", "ok_for_training")
        if not isinstance(configured_field, str) or not configured_field:
            raise ValueError("scope.training_eligibility_field must be a non-empty string")
        eligibility_fields = {status: configured_field for status in allowed_statuses}
    else:
        if not isinstance(configured_fields, dict):
            raise ValueError("scope.training_eligibility_fields must be a mapping")
        eligibility_fields = {}
        for status in allowed_statuses:
            field = configured_fields.get(status)
            if not isinstance(field, str) or not field:
                raise ValueError(
                    f"scope.training_eligibility_fields[{status!r}] must be a non-empty string"
                )
            eligibility_fields[status] = field
        extras = set(configured_fields) - set(allowed_statuses)
        if extras:
            raise ValueError(
                "scope.training_eligibility_fields has statuses outside allowed_statuses: "
                f"{sorted(extras)}"
            )
    return allowed_statuses, require_eligible, eligibility_fields


def _source_quality_metadata(episode: dict[str, Any]) -> dict[str, Any]:
    warn_reasons = episode.get("warn_reasons", [])
    if warn_reasons is None:
        warn_reasons = []
    if not isinstance(warn_reasons, list):
        raise ValueError(f"{episode.get('episode_id')}: warn_reasons must be a list")
    return {
        "source_status": str(episode.get("status", "UNKNOWN")),
        "source_ok_for_training": episode.get("ok_for_training"),
        "source_ok_for_training_relaxed": episode.get("ok_for_training_relaxed"),
        "source_validation_summary": episode.get("validation_summary"),
        "source_warn_reasons": [str(reason) for reason in warn_reasons],
        "source_manual_override": episode.get("manual_override"),
        "source_manual_reason": episode.get("manual_reason"),
    }


def _source_quality_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    quality = []
    for episode in episodes:
        status = str(episode.get("status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
        quality.append(_source_quality_metadata(episode))
    return {
        "status_counts": dict(sorted(counts.items())),
        "strict_eligible_episodes": sum(
            item["source_ok_for_training"] is True for item in quality
        ),
        "relaxed_eligible_episodes": sum(
            item["source_ok_for_training_relaxed"] is True for item in quality
        ),
        "warning_reason_count": sum(len(item["source_warn_reasons"]) for item in quality),
    }


def _load_view(root: Path, recorder: CheckRecorder) -> DatasetView:
    required = (
        "meta/info.json",
        "meta/tasks.parquet",
        "meta/stats.json",
        "meta/franka_eef_profile.json",
        "meta/pi0_eef_stats.json",
        "sidecars/episode_index.json",
    )
    missing = [relative for relative in required if not (root / relative).is_file()]
    recorder.require(
        "dataset.required_files",
        root.is_dir() and not missing,
        {"root": str(root), "missing": missing},
    )
    episode_index = _load_json(root / "sidecars/episode_index.json")
    recorder.require(
        "dataset.episode_index_nonempty",
        isinstance(episode_index, list) and bool(episode_index),
    )
    return DatasetView(
        root=root,
        info=_load_json(root / "meta/info.json"),
        profile=_load_json(root / "meta/franka_eef_profile.json"),
        episode_index=episode_index,
        main=_read_parquet_tree(root, "data/chunk-*/file-*.parquet"),
        episodes_meta=_read_parquet_tree(root, "meta/episodes/chunk-*/file-*.parquet"),
        standard_stats=_load_json(root / "meta/stats.json"),
        effective_stats=_load_json(root / "meta/pi0_eef_stats.json"),
    )


def _scope_episodes(
    config: dict[str, Any], limit_episodes: int | None, recorder: CheckRecorder
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], str]:
    scope = config.get("scope", {})
    manifest_value = scope.get("manifest")
    if not manifest_value:
        raise ValueError("config scope.manifest is required")
    manifest_path = Path(manifest_value).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    episodes = list(manifest["episodes"])
    allowed_statuses, require_eligible, eligibility_fields = _scope_policy(scope)
    recorder.require(
        "scope.manifest_training_only",
        bool(episodes)
        and all(
            str(item.get("status")) in allowed_statuses
            and (
                not require_eligible
                or item.get(eligibility_fields[str(item.get("status"))]) is True
            )
            for item in episodes
        ),
        {
            "manifest": str(manifest_path),
            "episodes": len(episodes),
            "allowed_statuses": list(allowed_statuses),
            "require_training_eligibility": require_eligible,
            "training_eligibility_fields": eligibility_fields,
        },
    )
    recorder.require(
        "scope.expected_episode_count",
        int(scope.get("expected_episode_count", -1)) == len(episodes),
        {"configured": scope.get("expected_episode_count"), "manifest": len(episodes)},
    )
    configured_hash = scope.get("scope_content_sha256")
    actual_hash = _sha256_file(manifest_path)
    recorder.require(
        "scope.manifest_hash",
        configured_hash is None or configured_hash == actual_hash,
        {"configured": configured_hash, "actual": actual_hash},
    )
    if limit_episodes is not None:
        if not 0 < limit_episodes <= len(episodes):
            raise ValueError(
                f"--limit-episodes must be in [1,{len(episodes)}], got {limit_episodes}"
            )
        episodes = episodes[:limit_episodes]
    suffix = f"_partial{len(episodes)}" if len(episodes) < int(manifest["n_episodes"]) else ""
    return manifest_path, manifest, episodes, suffix


def _default_root(config: dict[str, Any], suffix: str) -> Path:
    dataset_cfg = _dataset_config(config)
    base = Path(dataset_cfg["output_base"]).expanduser().resolve()
    return base / f"{dataset_cfg['name']}{suffix}"


def _verify_scope_and_profile(
    view: DatasetView,
    config: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    expected_episodes: list[dict[str, Any]],
    suffix: str,
    recorder: CheckRecorder,
) -> None:
    dataset_cfg = _dataset_config(config)
    contract = config["contract"]
    recorder.require(
        "scope.config_native30_contract",
        contract.get("profile") == PROFILE
        and int(contract.get("observation_fps", -1)) == FPS
        and int(contract.get("action_fps", -1)) == FPS
        and int(contract.get("chunk_size", -1)) == CHUNK_SIZE
        and contract.get("timestamp_source") == "mcap_log_time_ns"
        and contract.get("observation_clock") == "cam1"
        and contract.get("alignment_strategy") == "latest_not_after"
        and dataset_cfg.get("main_action_key") == ACTION_KEY
        and dataset_cfg.get("requires_project_cartesian_adapter") is True
        and dataset_cfg.get("requires_project_dual_rate_adapter") is False,
        contract,
    )
    profile = view.profile
    expected_action_label = resolve_data_config_action_label_spec(
        config, source=config_path
    )
    action_label = resolve_action_label_spec(
        profile, source=view.root / "meta/franka_eef_profile.json"
    )
    recorder.require(
        "scope.config_profile_action_label_contract",
        action_label == expected_action_label,
        {"config": expected_action_label, "dataset": action_label},
    )
    expected_ids = [str(item["episode_id"]) for item in expected_episodes]
    actual_ids = [str(item.get("raw_episode_id")) for item in view.episode_index]
    recorder.require(
        "scope.dataset_episode_order",
        actual_ids == expected_ids,
        {"expected": expected_ids, "actual": actual_ids},
    )
    scope = config["scope"]
    if "allowed_statuses" in scope:
        allowed_statuses, require_eligible, eligibility_fields = _scope_policy(scope)
        expected_policy = {
            "allowed_statuses": list(allowed_statuses),
            "require_training_eligibility": require_eligible,
            "training_eligibility_fields": eligibility_fields,
        }
        expected_quality_summary = _source_quality_summary(expected_episodes)
        recorder.require(
            "scope.mixed_quality_profile_policy",
            view.profile.get("scope_policy") == expected_policy
            and view.profile.get("source_quality_summary") == expected_quality_summary
            and view.profile.get("source_status_counts")
            == expected_quality_summary["status_counts"],
            {
                "expected_policy": expected_policy,
                "actual_policy": view.profile.get("scope_policy"),
                "expected_quality_summary": expected_quality_summary,
                "actual_quality_summary": view.profile.get("source_quality_summary"),
            },
        )
        quality_ok = True
        quality_mismatches: list[str] = []
        for record, episode in zip(view.episode_index, expected_episodes, strict=True):
            expected_quality = _source_quality_metadata(episode)
            mismatched_fields = [
                key for key, value in expected_quality.items() if record.get(key) != value
            ]
            if mismatched_fields:
                quality_ok = False
                quality_mismatches.append(
                    f"{episode.get('episode_id')}: {','.join(mismatched_fields)}"
                )
        recorder.require(
            "scope.mixed_quality_episode_provenance",
            quality_ok,
            {"checked": len(expected_episodes), "mismatches": quality_mismatches},
        )
    expected_name = f"{dataset_cfg['name']}{suffix}"
    expected_repo = f"{dataset_cfg['repo_id']}{suffix}"
    profile_ok = (
        profile.get("dataset_name") == expected_name
        and profile.get("repo_id") == expected_repo
        and profile.get("profile") == PROFILE
        and int(profile.get("observation_fps", -1)) == FPS
        and int(profile.get("action_fps", -1)) == FPS
        and int(profile.get("chunk_size", -1)) == CHUNK_SIZE
        and profile.get("main_action_key") == ACTION_KEY
        and profile.get("target_grid") == "next_native_camera_endpoint"
        and profile.get("task_instruction") == config["contract"]["task_instruction"]
        and int(profile.get("episode_count", -1)) == len(expected_episodes)
        and bool(profile.get("partial_conversion")) == bool(suffix)
        and profile.get("requires_project_cartesian_adapter") is True
        and not bool(profile.get("requires_project_dual_rate_adapter", False))
    )
    recorder.require(
        "scope.native30_profile_contract",
        profile_ok,
        {
            "dataset_name": profile.get("dataset_name"),
            "repo_id": profile.get("repo_id"),
            "profile": profile.get("profile"),
            "observation_fps": profile.get("observation_fps"),
            "action_fps": profile.get("action_fps"),
            "chunk_size": profile.get("chunk_size"),
            "action_label_mode": action_label["mode"],
        },
    )
    if "action_label_mode" in contract:
        recorder.require(
            "scope.profile_action_label_mode_source",
            profile.get("action_label_mode_source")
            == "config.contract.action_label_mode",
            {
                "actual": profile.get("action_label_mode_source"),
                "expected": "config.contract.action_label_mode",
            },
        )

    path_fields = (
        ("derived_manifest", manifest_path),
        ("conversion_config", config_path),
    )
    for field, expected_path in path_fields:
        value = profile.get(field)
        if value is not None:
            recorder.require(
                f"scope.profile_{field}",
                Path(value).expanduser().resolve() == expected_path,
                {"profile": value, "expected": str(expected_path)},
            )
    hash_fields = (
        ("derived_manifest_sha256", manifest_path),
        ("conversion_config_sha256", config_path),
    )
    for field, path in hash_fields:
        value = profile.get(field)
        if value is not None:
            recorder.require(
                f"scope.profile_{field}",
                value == _sha256_file(path),
                {"profile": value, "actual": _sha256_file(path)},
            )

    by_id = {str(item["episode_id"]): item for item in manifest["episodes"]}
    file_ok = True
    mismatches: list[str] = []
    for record in view.episode_index:
        episode_id = str(record["raw_episode_id"])
        source = by_id[episode_id]
        path = resolve_mcap_path(source, data_root=manifest["data_root"])
        recorded_path = record.get("mcap_path")
        recorded_size = record.get("mcap_size_bytes")
        if recorded_path is not None and Path(recorded_path).resolve() != path.resolve():
            file_ok = False
            mismatches.append(f"{episode_id}: path")
        if recorded_size is not None and int(recorded_size) != path.stat().st_size:
            file_ok = False
            mismatches.append(f"{episode_id}: size")
    recorder.require(
        "scope.source_mcap_identity",
        file_ok,
        {"checked": len(view.episode_index), "mismatches": mismatches},
    )


def _verify_metadata(
    view: DatasetView,
    config: dict[str, Any],
    expected_episodes: list[dict[str, Any]],
    recorder: CheckRecorder,
) -> int:
    info = view.info
    features = info.get("features", {})
    frame_count = sum(int(item["main_rows"]) for item in view.episode_index)
    recorder.require(
        "metadata.info_counts_and_fps",
        info.get("codebase_version") == "v3.0"
        and int(info.get("fps", -1)) == FPS
        and int(info.get("total_episodes", -1)) == len(expected_episodes)
        and int(info.get("total_frames", -1)) == frame_count
        and int(info.get("total_tasks", -1)) == 1
        and info.get("robot_type") == "franka_fr3_cartesian_eef"
        and info.get("splits") == {"train": f"0:{len(expected_episodes)}"},
        {"episodes": len(expected_episodes), "frames": frame_count, "fps": info.get("fps")},
    )
    required_features = {*CAMERA_KEYS, STATE_KEY, ACTION_KEY, *DEFAULT_KEYS}
    recorder.require(
        "metadata.required_features",
        required_features.issubset(features),
        {"required": sorted(required_features), "actual": sorted(features)},
    )
    shape_ok = (
        features[STATE_KEY].get("dtype") == "float32"
        and features[STATE_KEY].get("shape") == [STATE_DIM]
        and features[STATE_KEY].get("names") == STATE_NAMES
        and features[ACTION_KEY].get("dtype") == "float32"
        and features[ACTION_KEY].get("shape") == [CARRIER_DIM]
        and features[ACTION_KEY].get("names") == CARRIER_NAMES
        and all(
            features[key].get("dtype") == "video"
            and features[key].get("shape")
            == [
                int(config["image"]["height"]),
                int(config["image"]["width"]),
                int(config["image"]["channels"]),
            ]
            for key in CAMERA_KEYS
        )
    )
    recorder.require("metadata.feature_shapes_and_names", shape_ok)

    task = str(config["contract"]["task_instruction"])
    tasks = pq.read_table(view.root / "meta/tasks.parquet").to_pylist()
    recorder.require(
        "metadata.task_instruction",
        tasks == [{"task_index": 0, "task": task}],
        {"expected": task, "actual": tasks},
    )

    episode_meta = sorted(
        view.episodes_meta.to_pylist(), key=lambda item: int(item["episode_index"])
    )
    expected_start = 0
    boundaries_ok = len(episode_meta) == len(view.episode_index)
    for episode_index, (record, meta) in enumerate(
        zip(view.episode_index, episode_meta, strict=False)
    ):
        start = int(record["global_main_row_start"])
        end = int(record["global_main_row_end_exclusive"])
        boundaries_ok &= (
            int(record["episode_index"]) == episode_index
            and start == expected_start
            and end - start == int(record["main_rows"])
            and int(meta["episode_index"]) == episode_index
            and int(meta["length"]) == int(record["main_rows"])
            and int(meta["dataset_from_index"]) == start
            and int(meta["dataset_to_index"]) == end
            and meta["tasks"] == [task]
        )
        expected_start = end
    boundaries_ok &= expected_start == frame_count == len(view.main)
    recorder.require("metadata.episode_boundaries", bool(boundaries_ok))
    return frame_count


def _verify_main_table(view: DatasetView, recorder: CheckRecorder) -> None:
    main = view.main
    state = _column_numpy(main, STATE_KEY, np.float32)
    carrier = _column_numpy(main, ACTION_KEY, np.float32)
    recorder.require(
        "main.state_and_carrier_shape_finite",
        state.shape == (len(main), STATE_DIM)
        and carrier.shape == (len(main), CARRIER_DIM)
        and np.all(np.isfinite(state))
        and np.all(np.isfinite(carrier)),
        {"rows": len(main), "state_shape": state.shape, "action_shape": carrier.shape},
    )
    rotation_col0 = state[:, 3:6]
    rotation_col1 = state[:, 6:9]
    quaternion = carrier[:, 3:7]
    geometry_ok = (
        np.allclose(np.linalg.norm(rotation_col0, axis=1), 1.0, atol=2e-4, rtol=0.0)
        and np.allclose(np.linalg.norm(rotation_col1, axis=1), 1.0, atol=2e-4, rtol=0.0)
        and np.allclose(np.sum(rotation_col0 * rotation_col1, axis=1), 0.0, atol=2e-4, rtol=0.0)
        and np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=2e-4, rtol=0.0)
        and np.all((state[:, 9] >= -1e-6) & (state[:, 9] <= 1.0 + 1e-6))
        and np.all((carrier[:, 7] >= -1e-6) & (carrier[:, 7] <= 1.0 + 1e-6))
    )
    recorder.require("main.rotation_and_gripper_domains", bool(geometry_ok))

    index = _column_numpy(main, "index", np.int64)
    frame = _column_numpy(main, "frame_index", np.int64)
    episode = _column_numpy(main, "episode_index", np.int64)
    task = _column_numpy(main, "task_index", np.int64)
    timestamp = _column_numpy(main, "timestamp", np.float64)
    semantics_ok = np.array_equal(index, np.arange(len(main), dtype=np.int64)) and np.all(task == 0)
    for record in view.episode_index:
        episode_index = int(record["episode_index"])
        start = int(record["global_main_row_start"])
        end = int(record["global_main_row_end_exclusive"])
        local = np.arange(end - start, dtype=np.int64)
        semantics_ok &= np.all(episode[start:end] == episode_index)
        semantics_ok &= np.array_equal(frame[start:end], local)
        semantics_ok &= np.allclose(
            timestamp[start:end], local.astype(np.float64) / FPS, rtol=0.0, atol=4e-6
        )
    recorder.require("main.native30_timestamp_semantics", bool(semantics_ok))


def _stats_feature_ok(feature: Any, shape: tuple[int, ...], *, expected_count: int) -> bool:
    if not isinstance(feature, dict) or not set(REQUIRED_STATS).issubset(feature):
        return False
    for field in REQUIRED_STATS:
        expected_shape = (1,) if field == "count" else shape
        try:
            array = np.asarray(feature[field], dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if array.shape != expected_shape or not np.all(np.isfinite(array)):
            return False
    return int(np.asarray(feature["count"]).reshape(-1)[0]) == expected_count


def _verify_stats_metadata(
    view: DatasetView,
    frame_count: int,
    valid_count: int,
    recorder: CheckRecorder,
) -> None:
    standard = view.standard_stats
    recorder.require(
        "stats.standard_state_and_carrier",
        _stats_feature_ok(standard.get(STATE_KEY), (STATE_DIM,), expected_count=frame_count)
        and _stats_feature_ok(
            standard.get(ACTION_KEY), (CARRIER_DIM,), expected_count=frame_count
        ),
    )
    effective = view.effective_stats
    profile_action_label = resolve_action_label_spec(
        view.profile, source=view.root / "meta/franka_eef_profile.json"
    )
    stats_action_label = resolve_action_label_spec(
        effective, source=view.root / "meta/pi0_eef_stats.json"
    )
    effective_ok = (
        int(effective.get("schema_version", -1)) == 1
        and effective.get("profile") == PROFILE
        and int(effective.get("observation_fps", -1)) == FPS
        and int(effective.get("action_fps", -1)) == FPS
        and int(effective.get("chunk_size", -1)) == CHUNK_SIZE
        and int(effective.get("episode_count", -1)) == len(view.episode_index)
        and _stats_feature_ok(
            effective.get(STATE_KEY), (STATE_DIM,), expected_count=valid_count
        )
        and _stats_feature_ok(
            effective.get(ACTION_KEY),
            (MODEL_ACTION_DIM,),
            expected_count=valid_count * CHUNK_SIZE,
        )
        and effective.get("source_dataset_hash") == view.profile.get("source_dataset_hash")
        and effective.get("logical_anchor_index_sha256")
        == view.profile.get("logical_anchor_index_sha256")
        and stats_action_label == profile_action_label
    )
    recorder.require(
        "stats.effective_native30_pi0",
        effective_ok,
        {"valid_anchors": valid_count, "effective_action_rows": valid_count * CHUNK_SIZE},
    )


def _verify_anchor_maps(
    view: DatasetView, config: dict[str, Any], recorder: CheckRecorder
) -> tuple[int, list[dict[str, Any]], dict[int, pa.Table]]:
    age_limits = _thresholds(config)
    max_interval_ms = config["alignment"].get("max_camera_interval_ms")
    if max_interval_ms is None or float(max_interval_ms) <= 0.0:
        raise ValueError(
            "native30 config must define a positive alignment.max_camera_interval_ms"
        )
    max_interval_ns = int(round(float(max_interval_ms) * 1e6))
    logical_anchors: list[dict[str, Any]] = []
    anchor_tables: dict[int, pa.Table] = {}
    total_valid = 0
    expected_global_start = 0
    for expected_episode_index, record in enumerate(view.episode_index):
        episode_index = int(record["episode_index"])
        episode_id = str(record["raw_episode_id"])
        start = int(record["global_main_row_start"])
        end = int(record["global_main_row_end_exclusive"])
        main_rows = int(record["main_rows"])
        recorder.require(
            f"episode_{episode_index:03d}.index_bounds",
            episode_index == expected_episode_index
            and start == expected_global_start
            and end - start == main_rows
            and main_rows > 0,
        )
        sidecars = record.get("sidecars")
        recorder.require(
            f"episode_{episode_index:03d}.anchor_sidecar_declared",
            isinstance(sidecars, dict) and "anchor_map" in sidecars,
        )
        path = _resolve_sidecar(view.root, str(sidecars["anchor_map"]))
        recorder.require(f"episode_{episode_index:03d}.anchor_sidecar_exists", path.is_file(), path)
        table = pq.read_table(path)
        anchor_tables[episode_index] = table
        required = {
            "episode_index",
            "raw_episode_id",
            "camera_anchor_local_index",
            "main_local_row",
            "main_global_row",
            "camera_log_time_ns",
            "cam1_raw_index",
            "cam2_raw_index",
            "eef_raw_index",
            "gripper_raw_index",
            "cam2_source_log_time_ns",
            "eef_source_log_time_ns",
            "gripper_source_log_time_ns",
            "cam2_age_ns",
            "eef_age_ns",
            "gripper_age_ns",
            "camera_interval_to_next_ns",
            "camera_interval_to_next_valid",
            "observation_valid",
            "valid_anchor_action15",
            "action15_local_start",
            "action15_count",
        }
        recorder.require(
            f"episode_{episode_index:03d}.anchor_columns",
            required.issubset(table.column_names),
            {"missing": sorted(required - set(table.column_names))},
        )
        count = len(table)
        local = np.arange(count, dtype=np.int64)
        camera_time = _column_numpy(table, "camera_log_time_ns", np.int64)
        observation_valid = _column_numpy(table, "observation_valid", bool)
        valid_anchor = _column_numpy(table, "valid_anchor_action15", bool)
        complete = local + CHUNK_SIZE < count
        structural_ok = (
            count == main_rows + 1 == int(record["common_camera_anchors"])
            and np.all(_column_numpy(table, "episode_index", np.int64) == episode_index)
            and np.all(_column_numpy(table, "raw_episode_id").astype(str) == episode_id)
            and np.array_equal(
                _column_numpy(table, "camera_anchor_local_index", np.int64), local
            )
            and np.array_equal(
                _column_numpy(table, "main_local_row", np.int64), np.r_[local[:-1], -1]
            )
            and np.array_equal(
                _column_numpy(table, "main_global_row", np.int64),
                np.r_[np.arange(start, end, dtype=np.int64), -1],
            )
            and np.all(np.diff(camera_time) > 0)
            and np.array_equal(
                _column_numpy(table, "action15_local_start", np.int64),
                np.where(complete, local, -1),
            )
            and np.array_equal(
                _column_numpy(table, "action15_count", np.int64),
                np.where(complete, CHUNK_SIZE, 0),
            )
        )
        for prefix in ("cam2", "eef", "gripper"):
            source_time = _column_numpy(table, f"{prefix}_source_log_time_ns", np.int64)
            age = _column_numpy(table, f"{prefix}_age_ns", np.int64)
            structural_ok &= np.all(source_time <= camera_time)
            structural_ok &= np.array_equal(age, camera_time - source_time)
        expected_observation_valid = (
            (_column_numpy(table, "cam2_age_ns", np.int64) <= age_limits.cam2_age_ns)
            & (_column_numpy(table, "eef_age_ns", np.int64) <= age_limits.eef_age_ns)
            & (_column_numpy(table, "gripper_age_ns", np.int64) <= age_limits.gripper_age_ns)
        )
        structural_ok &= np.array_equal(observation_valid, expected_observation_valid)
        expected_interval_ns = np.r_[np.diff(camera_time), -1].astype(np.int64)
        expected_interval_valid = np.r_[
            np.diff(camera_time) <= max_interval_ns, False
        ].astype(bool)
        structural_ok &= np.array_equal(
            _column_numpy(table, "camera_interval_to_next_ns", np.int64),
            expected_interval_ns,
        )
        structural_ok &= np.array_equal(
            _column_numpy(table, "camera_interval_to_next_valid", bool),
            expected_interval_valid,
        )
        recorder.require(
            f"episode_{episode_index:03d}.anchor_structure_and_causality",
            bool(structural_ok),
            {"episode": episode_id, "anchors": count},
        )

        expected_valid = np.zeros(count, dtype=bool)
        camera_interval_valid = np.diff(camera_time) <= max_interval_ns
        for anchor in range(max(0, count - CHUNK_SIZE)):
            expected_valid[anchor] = bool(
                observation_valid[anchor]
                and np.all(observation_valid[anchor + 1 : anchor + CHUNK_SIZE + 1])
                and np.all(camera_interval_valid[anchor : anchor + CHUNK_SIZE])
            )
        valid_indices = np.flatnonzero(valid_anchor)
        horizon_ok = np.array_equal(valid_anchor, expected_valid)
        horizon_ok &= all(start + int(anchor) + CHUNK_SIZE <= end for anchor in valid_indices)
        recorder.require(
            f"episode_{episode_index:03d}.valid_t_plus_1_to_t_plus_50_no_cross_episode",
            bool(horizon_ok),
            {
                "valid": len(valid_indices),
                "horizon": CHUNK_SIZE,
                "max_camera_interval_ms": float(max_interval_ms),
                "oversized_intervals": int(np.count_nonzero(~camera_interval_valid)),
            },
        )
        recorder.require(
            f"episode_{episode_index:03d}.valid_count",
            int(record["valid_anchors"]) == len(valid_indices),
            {"record": record.get("valid_anchors"), "sidecar": len(valid_indices)},
        )
        for anchor in valid_indices.tolist():
            logical_anchors.append(
                {
                    "logical_index": len(logical_anchors),
                    "episode_index": episode_index,
                    "raw_episode_id": episode_id,
                    "anchor_local_index": int(anchor),
                    "main_global_row": start + int(anchor),
                    "camera_log_time_ns": int(camera_time[anchor]),
                }
            )
        total_valid += len(valid_indices)
        expected_global_start = end

    logical_hash = canonical_sha256(logical_anchors)
    recorder.require(
        "anchors.logical_index_hash",
        view.profile.get("logical_anchor_index_sha256") == logical_hash,
        {"profile": view.profile.get("logical_anchor_index_sha256"), "actual": logical_hash},
    )
    recorder.passed(
        "anchors.aggregate",
        {"episodes": len(view.episode_index), "valid_anchors": total_valid},
    )
    return total_valid, logical_anchors, anchor_tables


def _raw_replay_positions(episode_count: int, mode: str) -> list[int]:
    if mode == "full":
        return list(range(episode_count))
    return sorted({0, episode_count // 2, episode_count - 1})


def _verify_raw_replay(
    view: DatasetView,
    config: dict[str, Any],
    manifest: dict[str, Any],
    anchor_tables: dict[int, pa.Table],
    mode: str,
    recorder: CheckRecorder,
) -> dict[str, Any]:
    thresholds = _thresholds(config)
    gripper_mapping = _gripper_mapping(config)
    max_interval_ms = config["alignment"].get("max_camera_interval_ms")
    if max_interval_ms is None or float(max_interval_ms) <= 0.0:
        raise ValueError(
            "native30 config must define a positive alignment.max_camera_interval_ms"
        )
    max_interval_ns = int(round(float(max_interval_ms) * 1e6))
    manifest_by_id = {str(item["episode_id"]): item for item in manifest["episodes"]}
    main_state = _column_numpy(view.main, STATE_KEY, np.float32)
    main_action = _column_numpy(view.main, ACTION_KEY, np.float32)
    positions = _raw_replay_positions(len(view.episode_index), mode)
    action_label = resolve_action_label_spec(
        view.profile, source=view.root / "meta/franka_eef_profile.json"
    )
    states_for_stats: list[np.ndarray] = []
    actions_for_stats: list[np.ndarray] = []

    for position in positions:
        record = view.episode_index[position]
        episode_index = int(record["episode_index"])
        episode_id = str(record["raw_episode_id"])
        episode = manifest_by_id[episode_id]
        mcap_path = resolve_mcap_path(episode, data_root=manifest["data_root"])
        signals = load_episode_signals(mcap_path, episode_id=episode_id)
        median_rates = {
            "cam1": 1e9 / float(np.median(np.diff(signals.cam1.log_time_ns))),
            "cam2": 1e9 / float(np.median(np.diff(signals.cam2.log_time_ns))),
            "eef": 1e9 / float(np.median(np.diff(signals.eef_pose_xyzw.log_time_ns))),
            "qpos": 1e9 / float(np.median(np.diff(signals.qpos.log_time_ns))),
        }
        rate_minimum = float(config["alignment"]["nominal_rate_hz"]["minimum"])
        rate_maximum = float(config["alignment"]["nominal_rate_hz"]["maximum"])
        recorder.require(
            f"raw_{episode_index:03d}.nominal_30hz_native_streams",
            all(rate_minimum <= rate <= rate_maximum for rate in median_rates.values())
            and all(
                np.isclose(
                    float(record["native_median_rate_hz"][name]),
                    rate,
                    rtol=0.0,
                    atol=1e-12,
                )
                for name, rate in median_rates.items()
            ),
            median_rates,
        )
        aligned = align_episode_to_camera(signals, thresholds)
        carriers = build_dual_rate_carriers(signals, aligned, thresholds, gripper_mapping)
        source_valid_anchors = valid_anchor_indices(
            carriers, profile="action15", horizon_camera_intervals=CHUNK_SIZE
        )
        interval_valid = np.diff(carriers.camera_log_time_ns) <= max_interval_ns
        anchors = np.asarray(
            [
                int(anchor)
                for anchor in source_valid_anchors
                if np.all(interval_valid[int(anchor) : int(anchor) + CHUNK_SIZE])
            ],
            dtype=np.int64,
        )
        sidecar = anchor_tables[episode_index]
        start = int(record["global_main_row_start"])
        end = int(record["global_main_row_end_exclusive"])

        common_interval_ok = (
            aligned.common_start_ns
            == max(
                int(signals.cam1.log_time_ns[0]),
                int(signals.cam2.log_time_ns[0]),
                int(signals.eef_pose_xyzw.log_time_ns[0]),
                int(signals.gripper.log_time_ns[0]),
            )
            and aligned.common_end_ns
            == min(
                int(signals.cam1.log_time_ns[-1]),
                int(signals.cam2.log_time_ns[-1]),
                int(signals.eef_pose_xyzw.log_time_ns[-1]),
                int(signals.gripper.log_time_ns[-1]),
            )
            and aligned.camera_log_time_ns[0] >= aligned.common_start_ns
            and aligned.camera_log_time_ns[-1] <= aligned.common_end_ns
            and int(record.get("common_start_ns", -1)) == aligned.common_start_ns
            and int(record.get("common_end_ns", -1)) == aligned.common_end_ns
            and np.array_equal(
                aligned.camera_log_time_ns,
                _column_numpy(sidecar, "camera_log_time_ns", np.int64),
            )
        )
        recorder.require(
            f"raw_{episode_index:03d}.common_interval_and_camera_axis",
            bool(common_interval_ok),
            episode_id,
        )
        alignment_ok = (
            np.array_equal(
                aligned.cam1_raw_index, _column_numpy(sidecar, "cam1_raw_index", np.int64)
            )
            and np.array_equal(
                aligned.cam2_raw_index, _column_numpy(sidecar, "cam2_raw_index", np.int64)
            )
            and np.array_equal(
                aligned.eef_raw_index, _column_numpy(sidecar, "eef_raw_index", np.int64)
            )
            and np.array_equal(
                aligned.gripper_raw_index,
                _column_numpy(sidecar, "gripper_raw_index", np.int64),
            )
            and np.array_equal(
                aligned.observation_valid,
                _column_numpy(sidecar, "observation_valid", bool),
            )
            and np.array_equal(
                aligned.cam2_age_ns, _column_numpy(sidecar, "cam2_age_ns", np.int64)
            )
            and np.array_equal(
                aligned.eef_age_ns, _column_numpy(sidecar, "eef_age_ns", np.int64)
            )
            and np.array_equal(
                aligned.gripper_age_ns,
                _column_numpy(sidecar, "gripper_age_ns", np.int64),
            )
        )
        recorder.require(f"raw_{episode_index:03d}.alignment_identity", bool(alignment_ok))
        table_ok = (
            end - start == len(carriers.action_15hz)
            and np.array_equal(
                main_state[start:end], carriers.observation_state[:-1].astype(np.float32)
            )
            and np.array_equal(main_action[start:end], carriers.action_15hz.astype(np.float32))
        )
        recorder.require(
            f"raw_{episode_index:03d}.main_state_and_next_endpoint_carrier_identity",
            bool(table_ok),
        )
        valid_sidecar = np.flatnonzero(
            _column_numpy(sidecar, "valid_anchor_action15", bool)
        )
        recorder.require(
            f"raw_{episode_index:03d}.valid_anchor_identity",
            np.array_equal(anchors, valid_sidecar),
        )
        if mode == "full":
            states_for_stats.append(carriers.observation_state[anchors].astype(np.float32))
            if action_label["mode"] == ACTION_LABEL_MODE_DELTA_EEF:
                action_chunks = model_relative_action_chunks(
                    carriers,
                    anchors,
                    profile="action15",
                    horizon_camera_intervals=CHUNK_SIZE,
                )
            elif action_label["mode"] == ACTION_LABEL_MODE_ABSOLUTE_EEF:
                action_chunks = model_absolute_action_chunks(
                    carriers,
                    anchors,
                    profile="action15",
                    horizon_camera_intervals=CHUNK_SIZE,
                )
            else:  # pragma: no cover - resolve_action_label_spec rejects this.
                raise AssertionError(f"unreachable action label mode: {action_label['mode']}")
            actions_for_stats.append(action_chunks.astype(np.float32))

    report: dict[str, Any] = {
        "mode": mode,
        "episodes_replayed": len(positions),
        "episode_positions": positions,
        "action_label_mode": action_label["mode"],
    }
    if mode == "full":
        state = np.concatenate(states_for_stats, axis=0)
        action = np.concatenate(actions_for_stats, axis=0).reshape(-1, MODEL_ACTION_DIM)
        expected = {
            STATE_KEY: feature_stats(state),
            ACTION_KEY: feature_stats(action),
        }
        differences: dict[str, float] = {}
        stats_ok = True
        for key in (STATE_KEY, ACTION_KEY):
            actual_feature = view.effective_stats[key]
            for field in REQUIRED_STATS:
                actual = np.asarray(actual_feature[field], dtype=np.float64)
                wanted = np.asarray(expected[key][field], dtype=np.float64)
                error = float(np.max(np.abs(actual - wanted)))
                differences[f"{key}.{field}"] = error
                if field == "count":
                    stats_ok &= np.array_equal(actual, wanted)
                else:
                    stats_ok &= np.allclose(actual, wanted, rtol=2e-6, atol=1e-8)
        recorder.require(
            "raw.effective_stats_recomputed",
            bool(stats_ok),
            {"max_abs_error": max(differences.values(), default=0.0)},
        )
        report["effective_stats_max_abs_error"] = max(differences.values(), default=0.0)
    return report


def _video_sample_indices(
    records: list[dict[str, Any]], mode: str, random_count: int, seed: int
) -> list[int]:
    selected: set[int] = set()
    if mode == "full":
        positions = range(len(records))
        for position in positions:
            start = int(records[position]["global_main_row_start"])
            end = int(records[position]["global_main_row_end_exclusive"])
            selected.add(start + (end - start) // 2)
    else:
        positions = sorted({0, len(records) // 2, len(records) - 1})
        for position in positions:
            start = int(records[position]["global_main_row_start"])
            end = int(records[position]["global_main_row_end_exclusive"])
            selected.update((start, end - 1))
    if random_count > 0:
        total = int(records[-1]["global_main_row_end_exclusive"])
        rng = np.random.default_rng(seed)
        selected.update(
            int(item)
            for item in rng.choice(total, size=min(total, random_count), replace=False)
        )
    return sorted(selected)


def _image_hwc_uint8(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3:
        raise VerificationFailure(f"decoded image must be 3D, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if float(np.max(array)) <= 1.0 + 1e-6:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0)
        else:
            array = np.rint(np.clip(array, 0.0, 255.0))
    return np.asarray(array, dtype=np.uint8)


def _verify_cameras(
    view: DatasetView,
    config: dict[str, Any],
    mode: str,
    random_count: int,
    seed: int,
    recorder: CheckRecorder,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    samples = _video_sample_indices(view.episode_index, mode, random_count, seed)
    dataset = LeRobotDataset(
        repo_id=str(view.profile["repo_id"]),
        root=view.root,
        return_uint8=True,
        video_backend=str(config["video"]["video_backend"]),
    )
    expected_shape = (
        int(config["image"]["height"]),
        int(config["image"]["width"]),
        int(config["image"]["channels"]),
    )
    ok = len(dataset) == len(view.main) and dataset.meta.total_episodes == len(view.episode_index)
    task = str(config["contract"]["task_instruction"])
    for index in samples:
        item = dataset[index]
        ok &= item.get("task") == task
        ok &= np.asarray(item[STATE_KEY]).shape == (STATE_DIM,)
        ok &= np.asarray(item[ACTION_KEY]).shape == (CARRIER_DIM,)
        for camera in CAMERA_KEYS:
            image = _image_hwc_uint8(item[camera])
            ok &= image.shape == expected_shape
            ok &= image.dtype == np.uint8
    recorder.require(
        "video.both_cameras_decode",
        bool(ok),
        {"samples": len(samples), "shape": expected_shape, "indices": samples},
    )
    return {"decoded_samples": len(samples), "sample_indices": samples}


def _default_report_path(config: dict[str, Any], mode: str, suffix: str) -> Path:
    dataset_name = str(_dataset_config(config)["name"])
    return (
        PROJECT_ROOT
        / "artifacts/verification"
        / f"{dataset_name}{suffix}_{mode}.json"
    )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    recorder = CheckRecorder(checks=[])
    report: dict[str, Any] = {
        "schema_version": 1,
        "phase": "native30_dataset_verification",
        "status": "RUNNING",
        "generated_at_utc": _utc_now(),
        "mode": args.mode,
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "checks": recorder.checks,
    }
    suffix = ""
    try:
        configured_action_label = resolve_data_config_action_label_spec(
            config, source=config_path
        )
        manifest_path, manifest, expected_episodes, suffix = _scope_episodes(
            config, args.limit_episodes, recorder
        )
        root = (
            args.dataset_root.expanduser().resolve()
            if args.dataset_root is not None
            else _default_root(config, suffix)
        )
        report["dataset_root"] = str(root)
        report["manifest"] = str(manifest_path)
        report["expected_episodes"] = len(expected_episodes)
        view = _load_view(root, recorder)
        resolved_action_label = resolve_action_label_spec(
            view.profile, source=root / "meta/franka_eef_profile.json"
        )
        report["action_label_mode"] = resolved_action_label["mode"]
        report["action_label"] = resolved_action_label
        recorder.require(
            "scope.config_action_label_mode",
            resolved_action_label == configured_action_label,
            {
                "config": configured_action_label,
                "dataset": resolved_action_label,
            },
        )
        _verify_scope_and_profile(
            view,
            config,
            config_path,
            manifest_path,
            manifest,
            expected_episodes,
            suffix,
            recorder,
        )
        frame_count = _verify_metadata(view, config, expected_episodes, recorder)
        _verify_main_table(view, recorder)
        valid_count, logical_anchors, anchor_tables = _verify_anchor_maps(
            view, config, recorder
        )
        _verify_stats_metadata(view, frame_count, valid_count, recorder)
        report["counts"] = {
            "episodes": len(view.episode_index),
            "main_rows": frame_count,
            "valid_anchors": valid_count,
            "effective_action_rows": valid_count * CHUNK_SIZE,
        }
        report["logical_anchor_index_sha256"] = canonical_sha256(logical_anchors)
        report["raw_replay"] = _verify_raw_replay(
            view, config, manifest, anchor_tables, args.mode, recorder
        )
        report["video"] = _verify_cameras(
            view,
            config,
            args.mode,
            args.random_video_samples,
            args.seed,
            recorder,
        )
        report["status"] = "PASS_FULL" if args.mode == "full" else "PASS_SMOKE"
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

    report_path = (
        args.report_json
        or _default_report_path(config, args.mode, suffix)
    ).expanduser().resolve()
    report["report_json"] = str(report_path)
    _write_json(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Override the configured output root (the _partialN suffix is otherwise inferred).",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        help="Verify the converter's first-N _partialN output and scope.",
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--random-video-samples", type=int, default=3)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def main() -> int:
    report = verify(parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report.get("counts"),
                "failure": report.get("failure"),
                "report_json": report["report_json"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
