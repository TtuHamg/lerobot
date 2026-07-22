"""Raw Frank3 data audit used by phase D0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .action_chunk import (
    AlignmentThresholds,
    GripperMapping,
    align_episode_to_camera,
    assert_dual_rate_crosscheck,
    build_dual_rate_carriers,
    valid_anchor_indices,
)
from .mcap_reader import (
    EpisodeSignals,
    load_episode_signals,
    load_manifest,
    resolve_mcap_path,
    stream_rate_summary,
)


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: np.ndarray, *, scale: float = 1.0) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)] * scale
    if len(array) == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    quantile = np.quantile(array, [0.5, 0.9, 0.95, 0.99])
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(quantile[0]),
        "p90": float(quantile[1]),
        "p95": float(quantile[2]),
        "p99": float(quantile[3]),
        "max": float(np.max(array)),
    }


def _latency_ms(stream) -> dict[str, float | int | None]:
    valid = stream.header_time_ns >= 0
    return _stats(stream.log_time_ns[valid] - stream.header_time_ns[valid], scale=1e-6)


def _unique(values: tuple[str, ...]) -> list[str]:
    return sorted(set(values))


def _stream_audit(stream) -> dict[str, Any]:
    summary = stream_rate_summary(stream)
    summary["header_to_log_ms"] = _latency_ms(stream)
    summary["frame_ids"] = _unique(stream.frame_ids)
    summary["header_timestamp_missing"] = int(np.sum(stream.header_time_ns < 0))
    summary["header_timestamp_non_monotonic"] = int(
        np.sum(np.diff(stream.header_time_ns[stream.header_time_ns >= 0]) <= 0)
    )
    return summary


def _episode_audit(
    signals: EpisodeSignals,
    *,
    manifest_episode: dict[str, Any],
    thresholds: AlignmentThresholds,
    gripper_mapping: GripperMapping,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    aligned = align_episode_to_camera(signals, thresholds)
    carriers = build_dual_rate_carriers(signals, aligned, thresholds, gripper_mapping)
    assert_dual_rate_crosscheck(carriers)
    anchor_15 = valid_anchor_indices(carriers, profile="action15")
    anchor_30 = valid_anchor_indices(carriers, profile="action30")

    quaternion = signals.eef_pose_xyzw.values[:, 3:7]
    quaternion_norm = np.linalg.norm(quaternion, axis=-1)
    adjacent_dot = np.sum(quaternion[:-1] * quaternion[1:], axis=-1)
    expected_counts = manifest_episode.get("topic_counts", {})

    common_cam1_count = len(aligned.camera_log_time_ns)
    episode = {
        "episode_id": signals.episode_id,
        "mcap_path": str(signals.mcap_path),
        "mcap_size_bytes": signals.mcap_path.stat().st_size,
        "manifest_duration_s": float(manifest_episode.get("duration_s", 0.0)),
        "streams": {
            "cam1": _stream_audit(signals.cam1),
            "cam2": _stream_audit(signals.cam2),
            "eef": _stream_audit(signals.eef_pose_xyzw),
            "gripper": _stream_audit(signals.gripper),
            "qpos": _stream_audit(signals.qpos),
        },
        "manifest_count_match": {
            "cam1": int(expected_counts.get("/camera1/camera1/color/image_raw", -1)) == len(signals.cam1),
            "cam2": int(expected_counts.get("/camera2/camera2/color/image_raw", -1)) == len(signals.cam2),
            "eef": int(expected_counts.get("/franka_robot_state_broadcaster/current_pose", -1))
            == len(signals.eef_pose_xyzw),
            "gripper": int(expected_counts.get("/gripper/joint_states", -1)) == len(signals.gripper),
            "qpos": int(expected_counts.get("/franka/joint_states", -1)) == len(signals.qpos),
        },
        "image": {
            "cam1_shapes": sorted({f"{h}x{w}" for h, w in zip(signals.cam1.height, signals.cam1.width, strict=True)}),
            "cam2_shapes": sorted({f"{h}x{w}" for h, w in zip(signals.cam2.height, signals.cam2.width, strict=True)}),
            "cam1_encodings": sorted(set(signals.cam1.encoding)),
            "cam2_encodings": sorted(set(signals.cam2.encoding)),
        },
        "common_interval": {
            "start_ns": aligned.common_start_ns,
            "end_ns": aligned.common_end_ns,
            "duration_s": float((aligned.common_end_ns - aligned.common_start_ns) / 1e9),
            "cam1_count": common_cam1_count,
            "cam1_trimmed_prefix": int(aligned.cam1_raw_index[0]),
            "cam1_trimmed_suffix": int(len(signals.cam1) - 1 - aligned.cam1_raw_index[-1]),
            "observation_valid_count": int(np.sum(aligned.observation_valid)),
            "observation_invalid_count": int(np.sum(~aligned.observation_valid)),
        },
        "alignment_age_ms": {
            "cam2": _stats(aligned.cam2_age_ns, scale=1e-6),
            "eef": _stats(aligned.eef_age_ns, scale=1e-6),
            "gripper": _stats(aligned.gripper_age_ns, scale=1e-6),
            "qpos_audit": _stats(
                np.where(aligned.qpos_raw_index >= 0, aligned.qpos_age_ns, np.nan), scale=1e-6
            ),
            "action30_midpoint_eef": _stats(
                carriers.action_30hz_target_time_ns[0::2]
                - carriers.action_30hz_eef_source_time_ns[0::2],
                scale=1e-6,
            ),
            "action30_midpoint_gripper": _stats(
                carriers.action_30hz_target_time_ns[0::2]
                - carriers.action_30hz_gripper_source_time_ns[0::2],
                scale=1e-6,
            ),
        },
        "eef": {
            "finite": bool(np.all(np.isfinite(signals.eef_pose_xyzw.values))),
            "frame_ids": _unique(signals.eef_pose_xyzw.frame_ids),
            "quaternion_norm": _stats(quaternion_norm),
            "quaternion_max_abs_norm_error": float(np.max(np.abs(quaternion_norm - 1.0))),
            "adjacent_raw_negative_dot_count": int(np.sum(adjacent_dot < 0.0)),
            "position_min": np.min(signals.eef_pose_xyzw.values[:, :3], axis=0).tolist(),
            "position_max": np.max(signals.eef_pose_xyzw.values[:, :3], axis=0).tolist(),
        },
        "gripper": {
            "finite": bool(np.all(np.isfinite(signals.gripper.values))),
            "names": list(signals.gripper.names),
            "raw": _stats(signals.gripper.values),
            "normalized": _stats(gripper_mapping.encode(signals.gripper.values)),
        },
        "qpos": {
            "finite": bool(np.all(np.isfinite(signals.qpos.values))),
            "names": list(signals.qpos.names),
            "min": np.min(signals.qpos.values, axis=0).tolist(),
            "max": np.max(signals.qpos.values, axis=0).tolist(),
        },
        "profiles": {
            "action15": {
                "carrier_rows": len(carriers.action_15hz),
                "valid_anchors": len(anchor_15),
            },
            "action30": {
                "carrier_rows": len(carriers.action_30hz),
                "valid_anchors": len(anchor_30),
            },
            "valid_anchor_sets_equal": bool(np.array_equal(anchor_15, anchor_30)),
            "endpoint_crosscheck": True,
        },
    }
    arrays = {
        "cam2_age_ns": aligned.cam2_age_ns,
        "eef_age_ns": aligned.eef_age_ns,
        "gripper_age_ns": aligned.gripper_age_ns,
        "qpos_age_ns": aligned.qpos_age_ns[aligned.qpos_raw_index >= 0],
        "gripper_raw": signals.gripper.values[:, 0],
        "quaternion_norm": quaternion_norm,
    }
    return episode, arrays


def _build_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# D0 Frank3 原始数据审计",
        "",
        f"- 生成时间（UTC）：`{report['generated_at_utc']}`",
        f"- Manifest：`{report['manifest_path']}`",
        f"- Manifest SHA256：`{report['manifest_sha256']}`",
        f"- D0 状态：`{report['phase_status']}`",
        f"- Manifest episode：`{aggregate['manifest_episode_count']}`；实际审计："
        f"`{aggregate['episode_count']}`；缺失原始 MCAP：`{aggregate['missing_raw_episode_count']}`",
        f"- 原始 cam1 帧：`{aggregate['raw_cam1_frames']}`",
        f"- 公共区间 cam1 anchors：`{aggregate['common_camera_anchors']}`",
        f"- 15 Hz 有效完整 horizon anchors：`{aggregate['valid_action15_anchors']}`",
        f"- 30 Hz 有效完整 horizon anchors：`{aggregate['valid_action30_anchors']}`",
        "",
        "## 关键结论",
        "",
    ]
    for conclusion in report["conclusions"]:
        lines.append(f"- {conclusion}")
    if report["missing_raw_episodes"]:
        lines.extend(["", "缺失条目：", ""])
        for item in report["missing_raw_episodes"]:
            lines.append(f"- `{item['episode_id']}`：`{item['expected_path']}`")
    lines.extend(
        [
            "",
            "## Aggregate alignment age（ms）",
            "",
            "| stream | p50 | p95 | p99 | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, value in aggregate["alignment_age_ms"].items():
        lines.append(
            f"| {key} | {value['p50']:.3f} | {value['p95']:.3f} | "
            f"{value['p99']:.3f} | {value['max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Episode 明细",
            "",
            "| episode | raw c1/c2 | common M | invalid obs | anchors 15/30 | "
            "cam2 max ms | EEF max ms | cam1 max dt ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for episode in report["episodes"]:
        lines.append(
            f"| {episode['episode_id']} | {episode['streams']['cam1']['count']}/"
            f"{episode['streams']['cam2']['count']} | {episode['common_interval']['cam1_count']} | "
            f"{episode['common_interval']['observation_invalid_count']} | "
            f"{episode['profiles']['action15']['valid_anchors']}/"
            f"{episode['profiles']['action30']['valid_anchors']} | "
            f"{episode['alignment_age_ms']['cam2']['max']:.3f} | "
            f"{episode['alignment_age_ms']['eef']['max']:.3f} | "
            f"{episode['streams']['cam1']['dt_max_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- 本报告只审计数据完整性、时间语义和离线标签可构造性，不代表真机闭环成功率。",
            "- 使用 MCAP `log_time_ns` 作为本轮同步时钟；header/log latency 同时保存在 JSON。",
            "- `future measured EEF/gripper` 是 realized waypoint proxy，不是采集控制器的 desired command。",
            "",
        ]
    )
    return "\n".join(lines)


def run_raw_audit(config_path: str | Path, *, max_episodes: int | None = None) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text())
    manifest_path = Path(config["raw_manifest"])
    manifest = load_manifest(manifest_path)
    expected = int(config["expected_episodes"])
    episodes = manifest["episodes"]
    if len(episodes) != expected:
        raise ValueError(f"expected {expected} episodes, got {len(episodes)}")
    if any(item.get("status") != "PASS" or not item.get("ok_for_training") for item in episodes):
        raise ValueError("manifest contains a non-PASS or non-training episode")
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    available_episodes: list[tuple[dict[str, Any], Path]] = []
    missing_raw_episodes: list[dict[str, str]] = []
    for manifest_episode in episodes:
        try:
            path = resolve_mcap_path(manifest_episode, data_root=manifest["data_root"])
        except (FileNotFoundError, ValueError) as error:
            missing_raw_episodes.append(
                {
                    "episode_id": str(manifest_episode.get("episode_id")),
                    "expected_path": str(
                        Path(manifest_episode.get("path") or "")
                        / str((manifest_episode.get("mcap_files") or [""])[0])
                    ),
                    "error": str(error),
                }
            )
        else:
            available_episodes.append((manifest_episode, path))
    if not available_episodes:
        raise FileNotFoundError("none of the manifest MCAP files are available")

    max_age = config["alignment"]["max_age_ms"]
    thresholds = AlignmentThresholds(
        cam2_age_ns=int(float(max_age["cam2"]) * 1e6),
        eef_age_ns=int(float(max_age["eef"]) * 1e6),
        gripper_age_ns=int(float(max_age["gripper"]) * 1e6),
        qpos_age_ns=int(float(max_age["qpos_audit"]) * 1e6),
    )
    gripper_config = config["gripper"]
    gripper_mapping = GripperMapping(
        raw_open=float(gripper_config["raw_open"]),
        raw_closed=float(gripper_config["raw_closed"]),
        clip=bool(gripper_config["clip"]),
    )

    episode_reports: list[dict[str, Any]] = []
    aggregate_arrays: dict[str, list[np.ndarray]] = {
        "cam2_age_ns": [],
        "eef_age_ns": [],
        "gripper_age_ns": [],
        "qpos_age_ns": [],
        "gripper_raw": [],
        "quaternion_norm": [],
    }
    for manifest_episode, path in available_episodes:
        signals = load_episode_signals(path, episode_id=manifest_episode["episode_id"])
        episode_report, arrays = _episode_audit(
            signals,
            manifest_episode=manifest_episode,
            thresholds=thresholds,
            gripper_mapping=gripper_mapping,
        )
        episode_reports.append(episode_report)
        for key, value in arrays.items():
            aggregate_arrays[key].append(value)

    combined = {key: np.concatenate(values) for key, values in aggregate_arrays.items()}
    alignment_stats = {
        "cam2": _stats(combined["cam2_age_ns"], scale=1e-6),
        "eef": _stats(combined["eef_age_ns"], scale=1e-6),
        "gripper": _stats(combined["gripper_age_ns"], scale=1e-6),
        "qpos_audit": _stats(combined["qpos_age_ns"], scale=1e-6),
    }
    aggregate = {
        "manifest_episode_count": len(episodes),
        "episode_count": len(episode_reports),
        "missing_raw_episode_count": len(missing_raw_episodes),
        "raw_cam1_frames": int(sum(item["streams"]["cam1"]["count"] for item in episode_reports)),
        "raw_cam2_frames": int(sum(item["streams"]["cam2"]["count"] for item in episode_reports)),
        "common_camera_anchors": int(sum(item["common_interval"]["cam1_count"] for item in episode_reports)),
        "valid_observations": int(
            sum(item["common_interval"]["observation_valid_count"] for item in episode_reports)
        ),
        "invalid_observations": int(
            sum(item["common_interval"]["observation_invalid_count"] for item in episode_reports)
        ),
        "valid_action15_anchors": int(
            sum(item["profiles"]["action15"]["valid_anchors"] for item in episode_reports)
        ),
        "valid_action30_anchors": int(
            sum(item["profiles"]["action30"]["valid_anchors"] for item in episode_reports)
        ),
        "alignment_age_ms": alignment_stats,
        "gripper_raw": _stats(combined["gripper_raw"]),
        "quaternion_norm": _stats(combined["quaternion_norm"]),
        "eef_frame_ids": sorted(
            {frame for item in episode_reports for frame in item["eef"]["frame_ids"]}
        ),
        "joint_name_orders": sorted(
            {tuple(item["qpos"]["names"]) for item in episode_reports}
        ),
        "gripper_name_orders": sorted(
            {tuple(item["gripper"]["names"]) for item in episode_reports}
        ),
    }

    conclusions = [
        "公共有效区间按 cam1/cam2/EEF/gripper 起止覆盖裁剪，未使用零图 fallback。",
        "15 Hz 与 30 Hz carrier 的 endpoint value/source timestamp 全量交叉对账通过。",
        f"EEF frame_id 集合为 {aggregate['eef_frame_ids']}；joint/gripper name order 在 JSON 中逐条记录。",
        f"配置阈值下共有 {aggregate['invalid_observations']} 个 observation 因 source 过旧而无效。",
    ]
    if missing_raw_episodes:
        conclusions.insert(
            0,
            f"D0 未通过：manifest 中有 {len(missing_raw_episodes)} 条原始 MCAP 不存在；"
            "viz_cache 不作为原始数据替代。",
        )
    if aggregate["valid_action15_anchors"] != aggregate["valid_action30_anchors"]:
        conclusions.append("15/30 Hz 有效 anchor 数不一致；D1 前必须定位 midpoint action 对齐失败。")
    else:
        conclusions.append("15/30 Hz 有效 full-horizon anchor 集合数量一致。")

    from datetime import UTC, datetime

    report = {
        "schema_version": 1,
        "phase_status": "BLOCKED_MISSING_RAW_MCAP" if missing_raw_episodes else "PASS",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "timestamp_source": config["timestamp_source"],
        "alignment_thresholds_ms": max_age,
        "gripper_mapping": gripper_config,
        "missing_raw_episodes": missing_raw_episodes,
        "aggregate": aggregate,
        "conclusions": conclusions,
        "episodes": episode_reports,
    }

    output = config["outputs"]
    json_path = Path(output["audit_json"])
    markdown_path = Path(output["audit_markdown"])
    if max_episodes is not None:
        json_path = json_path.with_name(f"{json_path.stem}_first_{max_episodes}{json_path.suffix}")
        markdown_path = markdown_path.with_name(
            f"{markdown_path.stem}_FIRST_{max_episodes}{markdown_path.suffix}"
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    markdown_path.write_text(_build_markdown(report) + "\n")
    report["output_json"] = str(json_path)
    report["output_markdown"] = str(markdown_path)
    return report
