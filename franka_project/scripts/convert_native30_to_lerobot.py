#!/usr/bin/env python3
"""Convert native 30 Hz Franka MCAP episodes into a PI0-ready LeRobot dataset.

This entry point is intentionally separate from ``convert_to_lerobot.py``.  The
older converter implements a frozen 15 Hz observation / dual-rate experiment;
this converter uses every cam1 frame from a native 30 Hz recording and one EEF
endpoint target per camera interval.  The on-disk action is an auditable 8D
absolute carrier.  ``CartesianAnchorDataset(profile="native30")`` converts each
complete 50-row window into the model-visible anchor-relative 7D action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
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
    model_relative_action_chunks,
)
from franka_eef_pipeline.mcap_reader import (  # noqa: E402
    load_episode_signals,
    load_manifest,
    resolve_mcap_path,
)
from franka_eef_pipeline.stats import (  # noqa: E402
    canonical_sha256,
    effective_eef_stats,
    json_ready,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/data/franka_chips_current_eef_native30_v1.yaml"
NATIVE_PROFILE = "native30"
NATIVE_FPS = 30
CHUNK_SIZE = 50
STATE_NAMES = [
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
CARRIER_NAMES = [
    "target.x",
    "target.y",
    "target.z",
    "target.qx",
    "target.qy",
    "target.qz",
    "target.qw",
    "target.gripper.closed_0_1",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _fixed_list(values: np.ndarray, *, dtype: pa.DataType) -> pa.FixedSizeListArray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"fixed-list input must be 2D, got {array.shape}")
    return pa.FixedSizeListArray.from_arrays(
        pa.array(array.reshape(-1), type=dtype), list_size=array.shape[1]
    )


def _write_parquet(path: Path, columns: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression="zstd", compression_level=3)


def _dataset_features(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    image = config["image"]
    image_shape = (int(image["height"]), int(image["width"]), int(image["channels"]))
    return {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": CARRIER_NAMES,
        },
    }


def _validate_contract(config: dict[str, Any]) -> None:
    contract = config.get("contract", {})
    actual = (
        contract.get("profile"),
        contract.get("observation_fps"),
        contract.get("action_fps"),
        contract.get("chunk_size"),
    )
    expected = (NATIVE_PROFILE, NATIVE_FPS, NATIVE_FPS, CHUNK_SIZE)
    if actual != expected:
        raise ValueError(
            "native30 converter contract is immutable: expected "
            f"profile/observation_fps/action_fps/chunk_size={expected}, got {actual}"
        )
    if contract.get("observation_clock") != "cam1":
        raise ValueError("native30 observation_clock must be cam1")
    if contract.get("alignment_strategy") != "latest_not_after":
        raise ValueError("native30 alignment_strategy must be latest_not_after")
    if contract.get("crop_to_required_common_interval") is not True:
        raise ValueError("native30 conversion requires common-interval cropping")
    if config.get("dataset", {}).get("main_action_key") != "action":
        raise ValueError("native30 main_action_key must be 'action'")
    task = contract.get("task_instruction")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task_instruction must be a non-empty string")


def _thresholds(config: dict[str, Any]) -> AlignmentThresholds:
    values = config["alignment"]["max_age_ms"]
    return AlignmentThresholds(
        cam2_age_ns=int(round(float(values["cam2"]) * 1e6)),
        eef_age_ns=int(round(float(values["eef"]) * 1e6)),
        gripper_age_ns=int(round(float(values["gripper"]) * 1e6)),
        qpos_age_ns=int(round(float(values["qpos_audit"]) * 1e6)),
    )


def _gripper_mapping(config: dict[str, Any]) -> GripperMapping:
    values = config["gripper"]
    return GripperMapping(
        raw_open=float(values["raw_open"]),
        raw_closed=float(values["raw_closed"]),
        clip=bool(values["clip"]),
    )


def _scope_policy(
    scope: dict[str, Any],
) -> tuple[tuple[str, ...], bool, dict[str, str]]:
    """Resolve the source-status and training-eligibility policy.

    ``required_status`` remains the backwards-compatible single-status form.
    New mixed-quality scopes use ``allowed_statuses`` together with explicit
    ``training_eligibility_fields``.  This lets PASS continue to require its
    strict flag while WARN uses its relaxed flag, without relabeling either.
    """

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


def _status_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in episodes:
        status = str(episode.get("status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


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
    quality = [_source_quality_metadata(episode) for episode in episodes]
    return {
        "status_counts": _status_counts(episodes),
        "strict_eligible_episodes": sum(
            item["source_ok_for_training"] is True for item in quality
        ),
        "relaxed_eligible_episodes": sum(
            item["source_ok_for_training_relaxed"] is True for item in quality
        ),
        "warning_reason_count": sum(len(item["source_warn_reasons"]) for item in quality),
    }


def _validate_manifest(
    config: dict[str, Any], *, limit_episodes: int | None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    scope = config["scope"]
    manifest_path = Path(scope["manifest"]).expanduser().resolve()
    actual_scope_hash = _sha256_file(manifest_path)
    expected_manifest_hash = str(scope["manifest_sha256"])
    expected_scope_hash = str(scope["scope_content_sha256"])
    if actual_scope_hash != expected_manifest_hash or actual_scope_hash != expected_scope_hash:
        raise RuntimeError(
            "source scope manifest content changed: "
            f"expected manifest={expected_manifest_hash}, scope={expected_scope_hash}; "
            f"got {actual_scope_hash}"
        )
    manifest = load_manifest(manifest_path)
    episodes = list(manifest["episodes"])
    expected_count = int(scope["expected_episode_count"])
    if len(episodes) != expected_count:
        raise RuntimeError(
            f"manifest episode count changed: expected {expected_count}, got {len(episodes)}"
        )
    allowed_statuses, require_eligible, eligibility_fields = _scope_policy(scope)
    seen_ids: set[str] = set()
    all_paths: list[Path] = []
    for episode in episodes:
        episode_id = str(episode.get("episode_id", ""))
        if not episode_id or episode_id in seen_ids:
            raise RuntimeError(f"missing or duplicate episode_id: {episode_id!r}")
        seen_ids.add(episode_id)
        status = episode.get("status")
        if status not in allowed_statuses:
            raise RuntimeError(
                f"{episode_id}: status={status!r}, "
                f"expected one of {allowed_statuses!r}"
            )
        eligibility_field = eligibility_fields[str(status)]
        if require_eligible and episode.get(eligibility_field) is not True:
            raise RuntimeError(f"{episode_id}: {eligibility_field} is not true")
        all_paths.append(resolve_mcap_path(episode, data_root=manifest["data_root"]).resolve())

    if limit_episodes is not None:
        if limit_episodes <= 0:
            raise ValueError("--limit-episodes must be positive")
        episodes = episodes[:limit_episodes]
        all_paths = all_paths[:limit_episodes]
    return manifest, episodes, all_paths


def _manifest_rate_hz(episode: dict[str, Any], topic: str) -> float:
    duration = float(episode.get("duration_s", 0.0))
    counts = episode.get("topic_counts", {})
    if duration <= 0 or not isinstance(counts, dict) or topic not in counts:
        raise RuntimeError(f"{episode.get('episode_id')}: missing rate metadata for {topic}")
    return float(counts[topic]) / duration


def _preflight_summary(
    config_path: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    episodes: list[dict[str, Any]],
    mcap_paths: list[Path],
) -> dict[str, Any]:
    allowed_statuses, require_eligible, eligibility_fields = _scope_policy(config["scope"])
    topics = {
        "cam1": "/camera1/camera1/color/image_raw",
        "cam2": "/camera2/camera2/color/image_raw",
        "eef": "/franka_robot_state_broadcaster/current_pose",
        "qpos": "/franka/joint_states",
    }
    rates = {
        name: np.asarray([_manifest_rate_hz(episode, topic) for episode in episodes])
        for name, topic in topics.items()
    }
    minimum = float(config["alignment"]["nominal_rate_hz"]["minimum"])
    maximum = float(config["alignment"]["nominal_rate_hz"]["maximum"])
    # Message-count / bag-duration is a coarse preflight diagnostic.  The
    # numeric conversion pass below checks median timestamp spacing directly.
    for name, values in rates.items():
        if np.any((values < minimum) | (values > maximum)):
            bad = np.flatnonzero((values < minimum) | (values > maximum)).tolist()
            raise RuntimeError(
                f"manifest {name} rates fall outside [{minimum},{maximum}] Hz at episodes {bad}"
            )
    return {
        "status": "PASS_NATIVE30_PREFLIGHT",
        "config": str(config_path),
        "manifest": str(Path(config["scope"]["manifest"]).resolve()),
        "manifest_sha256": _sha256_file(Path(config["scope"]["manifest"]).resolve()),
        "scope_content_sha256": str(config["scope"]["scope_content_sha256"]),
        "manifest_split": manifest.get("split"),
        "episodes": len(episodes),
        "source_status_counts": _status_counts(episodes),
        "source_quality_summary": _source_quality_summary(episodes),
        "scope_policy": {
            "allowed_statuses": list(allowed_statuses),
            "require_training_eligibility": require_eligible,
            "training_eligibility_fields": eligibility_fields,
        },
        "task_instruction": config["contract"]["task_instruction"],
        "profile": NATIVE_PROFILE,
        "observation_fps": NATIVE_FPS,
        "action_fps": NATIVE_FPS,
        "chunk_size": CHUNK_SIZE,
        "total_mcap_bytes": int(sum(path.stat().st_size for path in mcap_paths)),
        "manifest_rate_hz": {
            name: {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
            for name, values in rates.items()
        },
    }


def _median_rate_hz(timestamps_ns: np.ndarray, *, episode_id: str, stream: str) -> float:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if len(timestamps) < 2:
        raise RuntimeError(f"{episode_id}: {stream} has fewer than two samples")
    delta = np.diff(timestamps)
    if np.any(delta <= 0):
        raise RuntimeError(f"{episode_id}: {stream} timestamps are not strictly increasing")
    return float(1e9 / np.median(delta))


def _validate_native_rates(signals: Any, config: dict[str, Any]) -> dict[str, float]:
    rates = {
        "cam1": _median_rate_hz(
            signals.cam1.log_time_ns, episode_id=signals.episode_id, stream="cam1"
        ),
        "cam2": _median_rate_hz(
            signals.cam2.log_time_ns, episode_id=signals.episode_id, stream="cam2"
        ),
        "eef": _median_rate_hz(
            signals.eef_pose_xyzw.log_time_ns, episode_id=signals.episode_id, stream="eef"
        ),
        "qpos": _median_rate_hz(
            signals.qpos.log_time_ns, episode_id=signals.episode_id, stream="qpos"
        ),
    }
    minimum = float(config["alignment"]["nominal_rate_hz"]["minimum"])
    maximum = float(config["alignment"]["nominal_rate_hz"]["maximum"])
    for name, rate in rates.items():
        if not minimum <= rate <= maximum:
            raise RuntimeError(
                f"{signals.episode_id}: median {name} rate {rate:.3f} Hz is outside "
                f"[{minimum},{maximum}] Hz"
            )
    return rates


def _valid_native30_anchor_indices(
    carriers: Any, config: dict[str, Any]
) -> np.ndarray:
    """Return complete K=50 anchors that do not cross a cam1 time gap.

    Shared carrier validity already enforces cam2/EEF/gripper source-age limits.
    This additional interval mask is needed because a missing cam1 frame has no
    source-age counterpart: treating a 0.3 s jump as one 30 Hz step would
    compress real time inside the model action horizon.
    """

    camera_time = np.asarray(carriers.camera_log_time_ns, dtype=np.int64)
    interval_valid = np.diff(camera_time) <= int(
        round(float(config["alignment"]["max_camera_interval_ms"]) * 1e6)
    )
    count = len(camera_time)
    max_anchor = count - CHUNK_SIZE
    if max_anchor <= 0:
        return np.empty(0, dtype=np.int64)
    anchors = np.arange(max_anchor, dtype=np.int64)
    valid = np.asarray(
        [
            carriers.observation_valid[index]
            and np.all(carriers.action_15hz_valid[index : index + CHUNK_SIZE])
            and np.all(interval_valid[index : index + CHUNK_SIZE])
            for index in anchors
        ],
        dtype=np.bool_,
    )
    return anchors[valid]


def _anchor_map_columns(
    *,
    episode_index: int,
    episode_id: str,
    global_row_offset: int,
    signals: Any,
    aligned: Any,
    anchors: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    count = aligned.num_camera_anchors
    local = np.arange(count, dtype=np.int64)
    complete = local + CHUNK_SIZE < count
    valid = np.zeros(count, dtype=np.bool_)
    valid[anchors] = True
    interval_to_next_ns = np.full(count, -1, dtype=np.int64)
    interval_to_next_ns[:-1] = np.diff(aligned.camera_log_time_ns)
    interval_to_next_valid = np.zeros(count, dtype=np.bool_)
    interval_to_next_valid[:-1] = interval_to_next_ns[:-1] <= int(
        round(float(config["alignment"]["max_camera_interval_ms"]) * 1e6)
    )
    return {
        "episode_index": np.full(count, episode_index, dtype=np.int64),
        "raw_episode_id": pa.array([episode_id] * count, type=pa.string()),
        "camera_anchor_local_index": local,
        "main_local_row": np.where(local < count - 1, local, -1),
        "main_global_row": np.where(local < count - 1, global_row_offset + local, -1),
        "camera_log_time_ns": aligned.camera_log_time_ns,
        "camera_interval_to_next_ns": interval_to_next_ns,
        "camera_interval_to_next_valid": interval_to_next_valid,
        "cam1_raw_index": aligned.cam1_raw_index,
        "cam2_raw_index": aligned.cam2_raw_index,
        "eef_raw_index": aligned.eef_raw_index,
        "gripper_raw_index": aligned.gripper_raw_index,
        "qpos_raw_index": aligned.qpos_raw_index,
        "cam2_source_log_time_ns": signals.cam2.log_time_ns[aligned.cam2_raw_index],
        "eef_source_log_time_ns": signals.eef_pose_xyzw.log_time_ns[aligned.eef_raw_index],
        "gripper_source_log_time_ns": signals.gripper.log_time_ns[aligned.gripper_raw_index],
        "qpos_source_log_time_ns": np.where(
            aligned.qpos_raw_index >= 0,
            signals.qpos.log_time_ns[np.maximum(aligned.qpos_raw_index, 0)],
            -1,
        ),
        "cam2_age_ns": aligned.cam2_age_ns,
        "eef_age_ns": aligned.eef_age_ns,
        "gripper_age_ns": aligned.gripper_age_ns,
        "qpos_age_ns": aligned.qpos_age_ns,
        "observation_valid": aligned.observation_valid,
        # These legacy column names intentionally preserve compatibility with
        # the Cartesian adapter.  At native30, one row is one 30 Hz action.
        "valid_anchor_action15": valid,
        "action15_local_start": np.where(complete, local, -1),
        "action15_count": np.where(complete, CHUNK_SIZE, 0).astype(np.int32),
    }


def _write_episode_sidecars(
    root: Path,
    *,
    episode_index: int,
    episode_id: str,
    global_row_offset: int,
    signals: Any,
    aligned: Any,
    anchors: np.ndarray,
    config: dict[str, Any],
) -> dict[str, str]:
    filename = f"episode-{episode_index:06d}.parquet"
    anchor_path = root / "sidecars/anchor_map" / filename
    _write_parquet(
        anchor_path,
        _anchor_map_columns(
            episode_index=episode_index,
            episode_id=episode_id,
            global_row_offset=global_row_offset,
            signals=signals,
            aligned=aligned,
            anchors=anchors,
            config=config,
        ),
    )
    qpos_path = root / "sidecars/qpos" / filename
    _write_parquet(
        qpos_path,
        {
            "episode_index": np.full(len(signals.qpos), episode_index, dtype=np.int64),
            "raw_index": np.arange(len(signals.qpos), dtype=np.int64),
            "log_time_ns": signals.qpos.log_time_ns,
            "publish_time_ns": signals.qpos.publish_time_ns,
            "header_time_ns": signals.qpos.header_time_ns,
            "qpos": _fixed_list(signals.qpos.values, dtype=pa.float64()),
        },
    )
    return {
        "anchor_map": str(anchor_path.relative_to(root)),
        "qpos": str(qpos_path.relative_to(root)),
    }


def _source_dataset_hash(
    manifest_path: Path, episodes: list[dict[str, Any]], mcap_paths: list[Path]
) -> str:
    identities = [
        {
            "episode_id": episode["episode_id"],
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        for episode, path in zip(episodes, mcap_paths, strict=True)
    ]
    return canonical_sha256(
        {"manifest_sha256": _sha256_file(manifest_path), "mcap_identities": identities}
    )


def convert(
    config_path: Path,
    *,
    output_base: Path | None,
    limit_episodes: int | None,
    preflight_only: bool,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _load_yaml(config_path)
    _validate_contract(config)
    manifest, episodes, mcap_paths = _validate_manifest(
        config, limit_episodes=limit_episodes
    )
    preflight = _preflight_summary(
        config_path, config, manifest, episodes, mcap_paths
    )
    if preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False), flush=True)
        return preflight

    partial = len(episodes) != int(config["scope"]["expected_episode_count"])
    suffix = f"_partial{len(episodes)}" if partial else ""
    dataset_config = config["dataset"]
    dataset_name = str(dataset_config["name"]) + suffix
    repo_id = str(dataset_config["repo_id"]) + suffix
    base = (output_base or Path(dataset_config["output_base"])).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    final_root = base / dataset_name
    stage_root = base / f".{dataset_name}.staging-{os.getpid()}"
    if final_root.exists():
        raise FileExistsError(f"immutable dataset output already exists: {final_root}")
    if stage_root.exists():
        raise FileExistsError(f"staging directory already exists: {stage_root}")

    from lerobot.configs.video import VideoEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    video = config["video"]
    encoder = VideoEncoderConfig(
        vcodec=video["vcodec"],
        pix_fmt=video["pix_fmt"],
        g=int(video["g"]),
        crf=video["crf"],
        preset=video["preset"],
        fast_decode=int(video["fast_decode"]),
        video_backend=video["video_backend"],
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=NATIVE_FPS,
        features=_dataset_features(config),
        root=stage_root,
        robot_type="franka_fr3_cartesian_eef",
        use_videos=True,
        image_writer_processes=int(config["image"]["image_writer_processes"]),
        image_writer_threads=int(config["image"]["image_writer_threads"]),
        batch_encoding_size=int(video["batch_encoding_size"]),
        camera_encoder=encoder,
        encoder_threads=int(video["encoder_threads"]),
    )

    thresholds = _thresholds(config)
    gripper_mapping = _gripper_mapping(config)
    task = str(config["contract"]["task_instruction"])
    manifest_path = Path(config["scope"]["manifest"]).resolve()
    source_hash = _source_dataset_hash(manifest_path, episodes, mcap_paths)
    effective_states: list[np.ndarray] = []
    effective_actions: list[np.ndarray] = []
    episode_records: list[dict[str, Any]] = []
    logical_anchors: list[dict[str, Any]] = []
    global_row_offset = 0
    common_anchor_count = 0

    from franka_eef_pipeline.mcap_images import iter_aligned_policy_image_pairs

    for episode_index, (episode, mcap_path) in enumerate(
        zip(episodes, mcap_paths, strict=True)
    ):
        episode_id = str(episode["episode_id"])
        print(
            f"[{episode_index + 1}/{len(episodes)}] numeric/alignment: {episode_id}",
            flush=True,
        )
        signals = load_episode_signals(mcap_path, episode_id=episode_id)
        native_rates = _validate_native_rates(signals, config)
        aligned = align_episode_to_camera(signals, thresholds)
        # ``action_15hz`` means one endpoint per camera interval in this shared
        # helper.  Because this converter's camera clock is native 30 Hz, this
        # endpoint stream is also native 30 Hz; no midpoint synthesis is used.
        carriers = build_dual_rate_carriers(
            signals, aligned, thresholds, gripper_mapping
        )
        anchors = _valid_native30_anchor_indices(carriers, config)
        main_rows = aligned.num_camera_anchors - 1
        if main_rows <= 0 or len(carriers.action_15hz) != main_rows:
            raise RuntimeError(f"{episode_id}: invalid native30 main-row count")
        if len(anchors) == 0:
            raise RuntimeError(f"{episode_id}: no valid unpadded {CHUNK_SIZE}-step anchors")

        sidecars = _write_episode_sidecars(
            stage_root,
            episode_index=episode_index,
            episode_id=episode_id,
            global_row_offset=global_row_offset,
            signals=signals,
            aligned=aligned,
            anchors=anchors,
            config=config,
        )
        relative_actions = model_relative_action_chunks(
            carriers,
            anchors,
            profile="action15",
            horizon_camera_intervals=CHUNK_SIZE,
        ).astype(np.float32)
        effective_states.append(carriers.observation_state[anchors].astype(np.float32))
        effective_actions.append(relative_actions)
        for anchor in anchors.tolist():
            logical_anchors.append(
                {
                    "logical_index": len(logical_anchors),
                    "episode_index": episode_index,
                    "raw_episode_id": episode_id,
                    "anchor_local_index": int(anchor),
                    "main_global_row": int(global_row_offset + anchor),
                    "camera_log_time_ns": int(carriers.camera_log_time_ns[anchor]),
                }
            )

        print(
            f"[{episode_index + 1}/{len(episodes)}] image/write: {main_rows} rows",
            flush=True,
        )
        written = 0
        for image_pair in iter_aligned_policy_image_pairs(mcap_path, aligned):
            row = image_pair.row_index
            if row != written:
                raise RuntimeError(f"{episode_id}: non-contiguous image row {row} != {written}")
            dataset.add_frame(
                {
                    "observation.images.camera1": image_pair.cam1,
                    "observation.images.camera2": image_pair.cam2,
                    "observation.state": carriers.observation_state[row].astype(np.float32),
                    "action": carriers.action_15hz[row].astype(np.float32),
                    "task": task,
                }
            )
            written += 1
        if written != main_rows:
            raise RuntimeError(
                f"{episode_id}: image row count {written} != expected {main_rows}"
            )
        dataset.save_episode()

        episode_records.append(
            {
                "episode_index": episode_index,
                "raw_episode_id": episode_id,
                "mcap_path": str(mcap_path),
                "mcap_size_bytes": mcap_path.stat().st_size,
                "native_median_rate_hz": native_rates,
                "common_start_ns": aligned.common_start_ns,
                "common_end_ns": aligned.common_end_ns,
                "common_camera_anchors": aligned.num_camera_anchors,
                "main_rows": main_rows,
                "valid_anchors": len(anchors),
                "global_main_row_start": global_row_offset,
                "global_main_row_end_exclusive": global_row_offset + main_rows,
                **_source_quality_metadata(episode),
                "sidecars": sidecars,
            }
        )
        global_row_offset += main_rows
        common_anchor_count += aligned.num_camera_anchors

    print("Finalizing LeRobot videos/tables/stats", flush=True)
    dataset.finalize()
    states = np.concatenate(effective_states, axis=0)
    actions = np.concatenate(effective_actions, axis=0)
    logical_anchor_hash = canonical_sha256(logical_anchors)
    provenance = {
        "profile": NATIVE_PROFILE,
        "scope_content_sha256": str(config["scope"]["scope_content_sha256"]),
        "derived_manifest": str(manifest_path),
        "derived_manifest_sha256": _sha256_file(manifest_path),
        "source_dataset_hash": source_hash,
        "logical_anchor_index_sha256": logical_anchor_hash,
        "max_camera_interval_ms": float(config["alignment"]["max_camera_interval_ms"]),
        "conversion_config": str(config_path),
        "conversion_config_sha256": _sha256_file(config_path),
        "conversion_script": str(Path(__file__).resolve()),
        "conversion_script_sha256": _sha256_file(Path(__file__).resolve()),
        "episode_count": len(episodes),
        "partial_conversion": partial,
        "source_status_counts": preflight["source_status_counts"],
        "source_quality_summary": preflight["source_quality_summary"],
        "scope_policy": preflight["scope_policy"],
    }
    effective_stats = effective_eef_stats(
        states,
        actions,
        observation_fps=NATIVE_FPS,
        action_fps=NATIVE_FPS,
        chunk_size=CHUNK_SIZE,
        source_dataset_hash=source_hash,
    )
    effective_stats.update(provenance)
    _write_json(stage_root / "meta/pi0_eef_stats.json", effective_stats)

    monitor_cfg = config["monitor"]
    monitor_size = min(
        int(monitor_cfg["maximum_size"]),
        max(1, int(math.ceil(float(monitor_cfg["fraction"]) * len(logical_anchors)))),
    )
    rng = np.random.default_rng(int(monitor_cfg["seed"]))
    monitor_indices = np.sort(
        rng.choice(len(logical_anchors), size=monitor_size, replace=False)
    ).tolist()
    monitor = {
        "schema_version": 1,
        "name": monitor_cfg["name"],
        "source": "training_data",
        "held_out": False,
        "interpretation": "in_distribution_training_fit_diagnostic_only",
        "seed": int(monitor_cfg["seed"]),
        "population_size": len(logical_anchors),
        "sample_size": monitor_size,
        "anchors": [logical_anchors[index] for index in monitor_indices],
        "logical_anchor_index_sha256": logical_anchor_hash,
        **provenance,
    }
    monitor["anchor_list_sha256"] = canonical_sha256(monitor["anchors"])
    _write_json(stage_root / "meta/train_monitor_subset.json", monitor)
    _write_json(stage_root / "sidecars/episode_index.json", episode_records)
    _write_json(
        stage_root / "meta/franka_eef_profile.json",
        {
            "schema_version": 1,
            "dataset_name": dataset_name,
            "repo_id": repo_id,
            "profile": NATIVE_PROFILE,
            "observation_fps": NATIVE_FPS,
            "action_fps": NATIVE_FPS,
            "chunk_size": CHUNK_SIZE,
            "main_action_key": "action",
            "requires_project_cartesian_adapter": True,
            "requires_project_dual_rate_adapter": False,
            "target_grid": "next_native_camera_endpoint",
            "task_instruction": task,
            **provenance,
        },
    )
    report = {
        "schema_version": 1,
        "status": "PASS_PARTIAL_SMOKE" if partial else "PASS_FULL_CONVERSION",
        "generated_at_utc": _utc_now(),
        "dataset_root": str(final_root),
        "counts": {
            "episodes": len(episodes),
            "common_camera_anchors": common_anchor_count,
            "main_rows": global_row_offset,
            "valid_anchors": len(logical_anchors),
            "effective_action_samples": int(actions.shape[0] * actions.shape[1]),
        },
        "preflight": preflight,
        "source_status_counts": preflight["source_status_counts"],
        "source_quality_summary": preflight["source_quality_summary"],
        "episodes": episode_records,
        "profile": {
            "name": NATIVE_PROFILE,
            "observation_fps": NATIVE_FPS,
            "action_fps": NATIVE_FPS,
            "chunk_size": CHUNK_SIZE,
            "task_instruction": task,
        },
        "provenance": provenance,
    }
    _write_json(stage_root / "meta/native30_conversion_report.json", report)
    stage_root.rename(final_root)
    print(
        json.dumps(
            {"status": report["status"], "dataset_root": str(final_root), **report["counts"]},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-base",
        type=Path,
        help="Override the dataset parent directory (useful for a partial smoke).",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        help="Convert only the first N PASS episodes; output receives a _partialN suffix.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config/manifest/files/rate metadata without decoding or writing a dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        args.config,
        output_base=args.output_base,
        limit_episodes=args.limit_episodes,
        preflight_only=args.preflight_only,
    )
