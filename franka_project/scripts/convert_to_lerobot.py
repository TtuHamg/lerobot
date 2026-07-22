#!/usr/bin/env python3
"""Convert the frozen Franka 22-of-25 scope into two LeRobot v3 datasets.

The main tables stay on the 15 Hz camera clock.  The 15 Hz dataset stores an
8D next-camera absolute carrier under ``action``.  The 30 Hz dataset stores the
same endpoint carrier under a deliberately non-policy key and writes its true
midpoint/endpoint stream to a sidecar.  Neither raw carrier is a model action:
the project-local dataset adapter converts it to anchor-relative 7D Cartesian
chunks before PI0 normalization.
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
    assert_dual_rate_crosscheck,
    build_dual_rate_carriers,
    model_relative_action_chunks,
    valid_anchor_indices,
)
from franka_eef_pipeline.mcap_reader import (  # noqa: E402
    latest_not_after_indices,
    load_episode_signals,
    load_manifest,
    resolve_mcap_path,
)
from franka_eef_pipeline.stats import (  # noqa: E402
    canonical_sha256,
    effective_eef_stats,
    feature_stats,
    json_ready,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/data/franka_current_eef_conversion_22of25_v1.yaml"
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
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False) + "\n")


def _fixed_list(values: np.ndarray, *, dtype: pa.DataType) -> pa.FixedSizeListArray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"fixed-list input must be 2D, got {array.shape}")
    return pa.FixedSizeListArray.from_arrays(
        pa.array(array.reshape(-1), type=dtype), list_size=array.shape[1]
    )


def _write_parquet(path: Path, columns: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(columns)
    pq.write_table(table, path, compression="zstd", compression_level=3)


def _thresholds(common: dict[str, Any]) -> AlignmentThresholds:
    value = common["alignment"]["max_age_ms"]
    return AlignmentThresholds(
        cam2_age_ns=int(round(float(value["cam2"]) * 1e6)),
        eef_age_ns=int(round(float(value["eef"]) * 1e6)),
        gripper_age_ns=int(round(float(value["gripper"]) * 1e6)),
        qpos_age_ns=int(round(float(value["qpos_audit"]) * 1e6)),
    )


def _gripper_mapping(common: dict[str, Any]) -> GripperMapping:
    value = common["gripper"]
    return GripperMapping(
        raw_open=float(value["raw_open"]),
        raw_closed=float(value["raw_closed"]),
        clip=bool(value["clip"]),
    )


def _dataset_features(action_key: str) -> dict[str, dict[str, Any]]:
    return {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": STATE_NAMES,
        },
        action_key: {
            "dtype": "float32",
            "shape": (8,),
            "names": CARRIER_NAMES,
        },
    }


def _validate_scope(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scope = config["scope"]
    manifest_path = Path(scope["manifest"]).resolve()
    lock_path = Path(scope["lock"]).resolve()
    if _sha256_file(manifest_path) != scope["manifest_sha256"]:
        raise RuntimeError("derived manifest hash does not match the frozen conversion config")
    manifest = load_manifest(manifest_path)
    lock = json.loads(lock_path.read_text())
    if lock["status"] != "PASS_D3_AUTHORIZED_FOR_VERSIONED_22OF25_SCOPE":
        raise RuntimeError("22of25 scope lock does not authorize D3")
    if not lock["gates"]["d3_allowed"] or lock["gates"]["original_25_episode_manifest_allowed_for_d3"]:
        raise RuntimeError("invalid D3 scope gates")
    if int(manifest["n_episodes"]) != int(scope["episode_count"]):
        raise RuntimeError("scope episode count mismatch")
    if manifest["scope_content_sha256"] != scope["scope_content_sha256"]:
        raise RuntimeError("scope content hash mismatch")
    if lock["derived"]["manifest_sha256"] != scope["manifest_sha256"]:
        raise RuntimeError("scope lock points to a different derived manifest")
    by_id = {item["episode_id"]: item for item in lock["mcap_integrity"]["files"]}
    for episode in manifest["episodes"]:
        if episode["status"] != "PASS" or not episode["ok_for_training"]:
            raise RuntimeError(f"non-PASS episode in derived manifest: {episode['episode_id']}")
        path = resolve_mcap_path(episode, data_root=manifest["data_root"])
        identity = by_id.get(episode["episode_id"])
        if identity is None or path.resolve() != Path(identity["path"]).resolve():
            raise RuntimeError(f"MCAP identity missing for {episode['episode_id']}")
        if path.stat().st_size != int(identity["size_bytes"]):
            raise RuntimeError(f"MCAP size changed after scope freeze: {path}")
    return manifest, lock


def _source_dataset_hash(lock: dict[str, Any]) -> str:
    identities = [
        {
            "episode_id": item["episode_id"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in lock["mcap_integrity"]["files"]
    ]
    return canonical_sha256(identities)


def _anchor_map_columns(
    *,
    episode_index: int,
    episode_id: str,
    global_row_offset: int,
    signals: Any,
    aligned: Any,
    anchors15: np.ndarray,
    anchors30: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    count = aligned.num_camera_anchors
    local = np.arange(count, dtype=np.int64)
    main_local = np.where(local < count - 1, local, -1)
    main_global = np.where(local < count - 1, global_row_offset + local, -1)
    valid15 = np.zeros(count, dtype=np.bool_)
    valid30 = np.zeros(count, dtype=np.bool_)
    valid15[anchors15] = True
    valid30[anchors30] = True
    complete = local + horizon < count
    return {
        "episode_index": np.full(count, episode_index, dtype=np.int64),
        "raw_episode_id": pa.array([episode_id] * count, type=pa.string()),
        "camera_anchor_local_index": local,
        "main_local_row": main_local,
        "main_global_row": main_global,
        "camera_log_time_ns": aligned.camera_log_time_ns,
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
        "valid_anchor_action15": valid15,
        "valid_anchor_action30": valid30,
        "action15_local_start": np.where(complete, local, -1),
        "action15_count": np.where(complete, horizon, 0).astype(np.int32),
        "action30_local_start": np.where(complete, 2 * local, -1),
        "action30_count": np.where(complete, 2 * horizon, 0).astype(np.int32),
    }


def _write_episode_sidecars(
    roots: tuple[Path, Path],
    *,
    episode_index: int,
    episode_id: str,
    global_row_offset: int,
    signals: Any,
    aligned: Any,
    carriers: Any,
    anchors15: np.ndarray,
    anchors30: np.ndarray,
    horizon: int,
) -> dict[str, str]:
    filename = f"episode-{episode_index:06d}.parquet"
    anchor_columns = _anchor_map_columns(
        episode_index=episode_index,
        episode_id=episode_id,
        global_row_offset=global_row_offset,
        signals=signals,
        aligned=aligned,
        anchors15=anchors15,
        anchors30=anchors30,
        horizon=horizon,
    )
    qpos_columns = {
        "episode_index": np.full(len(signals.qpos), episode_index, dtype=np.int64),
        "raw_index": np.arange(len(signals.qpos), dtype=np.int64),
        "log_time_ns": signals.qpos.log_time_ns,
        "publish_time_ns": signals.qpos.publish_time_ns,
        "header_time_ns": signals.qpos.header_time_ns,
        "qpos": _fixed_list(signals.qpos.values, dtype=pa.float64()),
    }
    for root in roots:
        _write_parquet(root / "sidecars/anchor_map" / filename, anchor_columns)
        _write_parquet(root / "sidecars/qpos" / filename, qpos_columns)

    action_count = len(carriers.action_30hz)
    action_index = np.arange(action_count, dtype=np.int64)
    eef_raw = latest_not_after_indices(
        signals.eef_pose_xyzw.log_time_ns, carriers.action_30hz_target_time_ns
    )
    gripper_raw = latest_not_after_indices(
        signals.gripper.log_time_ns, carriers.action_30hz_target_time_ns
    )
    action30_columns = {
        "episode_index": np.full(action_count, episode_index, dtype=np.int64),
        "raw_episode_id": pa.array([episode_id] * action_count, type=pa.string()),
        "action_local_index": action_index,
        "interval_index": action_index // 2,
        "slot": (action_index % 2).astype(np.int8),
        "target_log_time_ns": carriers.action_30hz_target_time_ns,
        "eef_source_log_time_ns": carriers.action_30hz_eef_source_time_ns,
        "gripper_source_log_time_ns": carriers.action_30hz_gripper_source_time_ns,
        "eef_raw_index": eef_raw,
        "gripper_raw_index": gripper_raw,
        "valid": carriers.action_30hz_valid,
        # Match the float32 LeRobot main carrier so endpoint crosschecks can be
        # bit-exact after both tables are loaded.
        "carrier": _fixed_list(carriers.action_30hz.astype(np.float32), dtype=pa.float32()),
    }
    action30_path = roots[1] / "sidecars/action_30hz" / filename
    _write_parquet(action30_path, action30_columns)
    return {
        "anchor_map": str(Path("sidecars/anchor_map") / filename),
        "qpos": str(Path("sidecars/qpos") / filename),
        "action30": str(Path("sidecars/action_30hz") / filename),
    }


def _tree_hash(root: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    return canonical_sha256(files), files


def _render_report(report: dict[str, Any]) -> str:
    rows = report["counts"]
    return "\n".join(
        [
            "# D3 Full Dataset Conversion (22of25)",
            "",
            f"- Status: **{report['status']}**",
            f"- Generated: `{report['generated_at_utc']}`",
            f"- Source episodes: **{rows['episodes']}**",
            f"- Common camera anchors: **{rows['common_camera_anchors']}**",
            f"- Main rows per dataset: **{rows['main_rows']}**",
            f"- Valid full-horizon anchors: **{rows['valid_anchors']}**",
            f"- 30 Hz carrier rows: **{rows['action30_carrier_rows']}**",
            "",
            "Both datasets use the 15 Hz camera clock. The 15 Hz main-table `action` and the "
            "30 Hz sidecar are absolute 8D audit carriers; PI0 must use the project Cartesian "
            "adapter and the model-visible 10D/7D effective statistics.",
            "",
            f"- 15 Hz dataset: `{report['datasets']['action15']['root']}`",
            f"- 30 Hz dataset: `{report['datasets']['action30']['root']}`",
            f"- Machine report: `{report['outputs']['machine_report']}`",
            "",
            "No validation/test split was created. The frozen monitor subset is sampled from the "
            "same training anchors and is only an in-distribution fitting diagnostic.",
            "",
        ]
    )


def convert(config_path: Path, *, output_base: Path | None, limit_episodes: int | None) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    manifest, scope_lock = _validate_scope(config)
    common_path = Path(config["contract"]["common"]).resolve()
    common = _load_yaml(common_path)
    thresholds = _thresholds(common)
    gripper_mapping = _gripper_mapping(common)
    horizon = int(config["contract"]["horizon_camera_intervals"])
    task = str(config["contract"]["task_instruction"])
    episodes = list(manifest["episodes"])
    partial = limit_episodes is not None and limit_episodes < len(episodes)
    if limit_episodes is not None:
        if limit_episodes <= 0:
            raise ValueError("--limit-episodes must be positive")
        episodes = episodes[:limit_episodes]

    base = (output_base or Path(config["datasets"]["output_base"])).resolve()
    base.mkdir(parents=True, exist_ok=True)
    suffix = f"_partial{len(episodes)}" if partial else ""
    names = {
        key: str(config["datasets"][key]["name"]) + suffix for key in ("action15", "action30")
    }
    final_roots = {key: base / name for key, name in names.items()}
    for root in final_roots.values():
        if root.exists():
            raise FileExistsError(f"immutable dataset output already exists: {root}")
    stage_roots = {
        key: base / f".{name}.staging-{os.getpid()}" for key, name in names.items()
    }
    for root in stage_roots.values():
        if root.exists():
            raise FileExistsError(f"staging directory already exists: {root}")

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
    datasets = {
        key: LeRobotDataset.create(
            repo_id=str(config["datasets"][key]["repo_id"]) + suffix,
            fps=int(config["contract"]["observation_fps"]),
            features=_dataset_features(config["datasets"][key]["main_action_key"]),
            root=stage_roots[key],
            robot_type="franka_fr3_cartesian_eef",
            use_videos=True,
            image_writer_processes=int(config["image"]["image_writer_processes"]),
            image_writer_threads=int(config["image"]["image_writer_threads_per_dataset"]),
            batch_encoding_size=int(video["batch_encoding_size"]),
            camera_encoder=encoder,
            encoder_threads=int(video["encoder_threads_per_dataset"]),
        )
        for key in ("action15", "action30")
    }

    source_hash = _source_dataset_hash(scope_lock)
    episode_records: list[dict[str, Any]] = []
    effective_states: list[np.ndarray] = []
    effective_actions15: list[np.ndarray] = []
    effective_actions30: list[np.ndarray] = []
    carriers30_all: list[np.ndarray] = []
    logical_anchors: list[dict[str, Any]] = []
    global_row_offset = 0
    common_anchor_count = 0

    from franka_eef_pipeline.mcap_images import iter_aligned_policy_image_pairs

    for episode_index, episode in enumerate(episodes):
        episode_id = episode["episode_id"]
        mcap_path = resolve_mcap_path(episode, data_root=manifest["data_root"])
        print(f"[{episode_index + 1}/{len(episodes)}] numeric pass: {episode_id}", flush=True)
        signals = load_episode_signals(mcap_path, episode_id=episode_id)
        aligned = align_episode_to_camera(signals, thresholds)
        carriers = build_dual_rate_carriers(signals, aligned, thresholds, gripper_mapping)
        assert_dual_rate_crosscheck(carriers, atol=0.0)
        anchors15 = valid_anchor_indices(
            carriers, profile="action15", horizon_camera_intervals=horizon
        )
        anchors30 = valid_anchor_indices(
            carriers, profile="action30", horizon_camera_intervals=horizon
        )
        if not np.array_equal(anchors15, anchors30):
            raise AssertionError(f"valid anchor sets differ: {episode_id}")
        main_rows = aligned.num_camera_anchors - 1
        if main_rows <= 0 or len(carriers.action_15hz) != main_rows:
            raise AssertionError(f"invalid main-row count: {episode_id}")

        sidecars = _write_episode_sidecars(
            (stage_roots["action15"], stage_roots["action30"]),
            episode_index=episode_index,
            episode_id=episode_id,
            global_row_offset=global_row_offset,
            signals=signals,
            aligned=aligned,
            carriers=carriers,
            anchors15=anchors15,
            anchors30=anchors30,
            horizon=horizon,
        )

        effective_states.append(carriers.observation_state[anchors15].astype(np.float32))
        relative15 = model_relative_action_chunks(
            carriers, anchors15, profile="action15", horizon_camera_intervals=horizon
        ).astype(np.float32)
        relative30 = model_relative_action_chunks(
            carriers, anchors30, profile="action30", horizon_camera_intervals=horizon
        ).astype(np.float32)
        effective_actions15.append(relative15)
        effective_actions30.append(relative30)
        carriers30_all.append(carriers.action_30hz.astype(np.float32))
        for anchor in anchors15.tolist():
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

        print(f"[{episode_index + 1}/{len(episodes)}] image pass/write {main_rows} rows", flush=True)
        written = 0
        for image_pair in iter_aligned_policy_image_pairs(mcap_path, aligned):
            row_index = image_pair.row_index
            camera1 = image_pair.cam1
            camera2 = image_pair.cam2
            if row_index != written:
                raise AssertionError(f"non-contiguous image row {row_index} != {written}")
            base_frame = {
                "observation.images.camera1": camera1,
                "observation.images.camera2": camera2,
                "observation.state": carriers.observation_state[row_index].astype(np.float32),
                "task": task,
            }
            datasets["action15"].add_frame(
                {
                    **base_frame,
                    "action": carriers.action_15hz[row_index].astype(np.float32),
                }
            )
            datasets["action30"].add_frame(
                {
                    **base_frame,
                    "carrier.endpoint_15hz": carriers.action_15hz[row_index].astype(np.float32),
                }
            )
            written += 1
        if written != main_rows:
            raise AssertionError(f"image row count mismatch for {episode_id}: {written} != {main_rows}")
        print(f"[{episode_index + 1}/{len(episodes)}] encode/save", flush=True)
        datasets["action15"].save_episode()
        datasets["action30"].save_episode()

        record = {
            "episode_index": episode_index,
            "raw_episode_id": episode_id,
            "mcap_path": str(mcap_path),
            "mcap_size_bytes": mcap_path.stat().st_size,
            "common_camera_anchors": aligned.num_camera_anchors,
            "main_rows": main_rows,
            "valid_anchors": len(anchors15),
            "action30_carrier_rows": len(carriers.action_30hz),
            "global_main_row_start": global_row_offset,
            "global_main_row_end_exclusive": global_row_offset + main_rows,
            "sidecars": sidecars,
        }
        episode_records.append(record)
        global_row_offset += main_rows
        common_anchor_count += aligned.num_camera_anchors

    print("Finalizing LeRobot writers", flush=True)
    datasets["action15"].finalize()
    datasets["action30"].finalize()

    states = np.concatenate(effective_states, axis=0)
    actions15 = np.concatenate(effective_actions15, axis=0)
    actions30 = np.concatenate(effective_actions30, axis=0)
    carrier30 = np.concatenate(carriers30_all, axis=0)
    logical_anchor_hash = canonical_sha256(logical_anchors)
    stats15 = effective_eef_stats(
        states,
        actions15,
        observation_fps=15,
        action_fps=15,
        chunk_size=50,
        source_dataset_hash=source_hash,
    )
    stats30 = effective_eef_stats(
        states,
        actions30,
        observation_fps=15,
        action_fps=30,
        chunk_size=100,
        source_dataset_hash=source_hash,
    )
    provenance = {
        "scope_name": scope_lock["scope"],
        "derived_manifest": str(Path(config["scope"]["manifest"]).resolve()),
        "derived_manifest_sha256": config["scope"]["manifest_sha256"],
        "scope_content_sha256": config["scope"]["scope_content_sha256"],
        "source_dataset_hash": source_hash,
        "conversion_config": str(config_path),
        "conversion_config_sha256": _sha256_file(config_path),
        "conversion_script": str(Path(__file__).resolve()),
        "conversion_script_sha256": _sha256_file(Path(__file__).resolve()),
        "common_contract": str(common_path),
        "common_contract_sha256": _sha256_file(common_path),
        "logical_anchor_index_sha256": logical_anchor_hash,
        "episode_count": len(episodes),
        "partial_conversion": partial,
        "same_machine_as_collection": True,
        "same_configured_eef_tcp_as_collection": True,
        "real_robot_rollout_authorized": False,
    }
    for stats, profile in ((stats15, "action15"), (stats30, "action30")):
        stats.update({"profile": profile, **provenance})
    _write_json(stage_roots["action15"] / "meta/pi0_eef_stats.json", stats15)
    _write_json(stage_roots["action30"] / "meta/pi0_eef_stats.json", stats30)
    _write_json(
        stage_roots["action30"] / "meta/action_30hz_carrier_stats.json",
        {
            "schema_version": 1,
            "observation_fps": 15,
            "action_fps": 30,
            "carrier_shape": [8],
            "carrier": feature_stats(carrier30),
            **provenance,
        },
    )

    monitor_cfg = config["monitor"]
    monitor_size = min(
        int(monitor_cfg["maximum_size"]),
        int(math.ceil(float(monitor_cfg["fraction"]) * len(logical_anchors))),
    )
    rng = np.random.default_rng(int(monitor_cfg["seed"]))
    selected_indices = np.sort(
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
        "anchors": [logical_anchors[index] for index in selected_indices],
        "logical_anchor_index_sha256": logical_anchor_hash,
        **provenance,
    }
    monitor["anchor_list_sha256"] = canonical_sha256(monitor["anchors"])
    for key, root in stage_roots.items():
        _write_json(root / "meta/train_monitor_subset.json", monitor)
        dataset_config = config["datasets"][key]
        _write_json(
            root / "meta/franka_eef_profile.json",
            {
                "schema_version": 1,
                "dataset_name": names[key],
                "repo_id": str(dataset_config["repo_id"]) + suffix,
                "profile": key,
                "observation_fps": int(dataset_config["observation_fps"]),
                "action_fps": int(dataset_config["action_fps"]),
                "chunk_size": int(dataset_config["chunk_size"]),
                "main_action_key": dataset_config["main_action_key"],
                "requires_project_cartesian_adapter": bool(
                    dataset_config["requires_project_cartesian_adapter"]
                ),
                "requires_project_dual_rate_adapter": bool(
                    dataset_config.get("requires_project_dual_rate_adapter", False)
                ),
                "train_in_first_round": bool(dataset_config["train_in_first_round"]),
                "task_instruction": task,
                **provenance,
            },
        )
        _write_json(root / "sidecars/episode_index.json", episode_records)

    counts = {
        "episodes": len(episodes),
        "common_camera_anchors": common_anchor_count,
        "main_rows": global_row_offset,
        "valid_anchors": len(logical_anchors),
        "action15_effective_stats_count": int(actions15.shape[0] * actions15.shape[1]),
        "action30_carrier_rows": int(carrier30.shape[0]),
        "action30_effective_stats_count": int(actions30.shape[0] * actions30.shape[1]),
    }
    if not partial:
        for key, expected_key in (
            ("common_camera_anchors", "common_camera_anchors"),
            ("main_rows", "main_rows_per_dataset"),
            ("valid_anchors", "valid_anchor_count"),
            ("action15_effective_stats_count", "action15_effective_stats_count"),
            ("action30_carrier_rows", "action30_carrier_rows"),
            ("action30_effective_stats_count", "action30_effective_stats_count"),
        ):
            if counts[key] != int(config["expected"][expected_key]):
                raise AssertionError(f"full conversion count mismatch: {key}={counts[key]}")

    tree_data: dict[str, Any] = {}
    for key, root in stage_roots.items():
        tree_sha, files = _tree_hash(root)
        tree_data[key] = {
            "root": str(final_roots[key]),
            "tree_sha256": tree_sha,
            "file_count": len(files),
            "total_bytes": sum(item["size_bytes"] for item in files.values()),
            "files": files,
        }

    for key in ("action15", "action30"):
        stage_roots[key].rename(final_roots[key])

    output_stem = f"d3_conversion_{'partial' + str(len(episodes)) if partial else '22of25'}"
    machine_report = PROJECT_ROOT / "artifacts/conversion" / f"{output_stem}.json"
    markdown_report = PROJECT_ROOT / "reports" / f"D3_CONVERSION_{'PARTIAL' if partial else '22OF25'}.md"
    monitor_report = PROJECT_ROOT / "artifacts/train_monitor" / f"{output_stem}_monitor_subset.json"
    _write_json(monitor_report, monitor)
    report = {
        "schema_version": 1,
        "phase": "D3",
        "status": "PASS_PARTIAL_SMOKE" if partial else "PASS_FULL_CONVERSION",
        "generated_at_utc": _utc_now(),
        "counts": counts,
        "scope": provenance,
        "episodes": episode_records,
        "monitor_subset": {
            "sample_size": monitor_size,
            "seed": monitor["seed"],
            "anchor_list_sha256": monitor["anchor_list_sha256"],
        },
        "datasets": tree_data,
        "outputs": {
            "machine_report": str(machine_report),
            "markdown_report": str(markdown_report),
            "monitor_subset": str(monitor_report),
        },
    }
    _write_json(machine_report, report)
    markdown_report.parent.mkdir(parents=True, exist_ok=True)
    markdown_report.write_text(_render_report(report))
    print(json.dumps({"status": report["status"], "counts": counts}, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-base",
        type=Path,
        help="Override dataset parent directory (primarily for an isolated partial smoke).",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        help="Explicit partial conversion; output names receive a _partialN suffix.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.config, output_base=args.output_base, limit_episodes=args.limit_episodes)
