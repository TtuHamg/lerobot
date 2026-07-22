#!/usr/bin/env python3
"""Run the phase-D2 two-episode Cartesian pipeline spike.

This command deliberately stops at numeric EEF/action construction. It does
not decode or write images, create a LeRobot dataset, or start training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from franka_eef_pipeline.action_chunk import (  # noqa: E402
    AlignmentThresholds,
    GripperMapping,
    absolute_action_chunks,
    align_episode_to_camera,
    assert_dual_rate_crosscheck,
    build_dual_rate_carriers,
    model_relative_action_chunks,
    valid_anchor_indices,
)
from franka_eef_pipeline.geometry import (  # noqa: E402
    decode_relative_action,
    quaternion_xyzw_to_matrix,
    rotation_geodesic_angle,
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
    normalize,
    unnormalize,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/data/franka_current_eef_common_v1.yaml"
PROFILE_CONFIGS = {
    "action15": PROJECT_ROOT / "configs/data/franka_current_eef_obs15_act15_v1.yaml",
    "action30": PROJECT_ROOT / "configs/data/franka_current_eef_obs15_act30_v1.yaml",
}
SPIKE_EPISODE_IDS = (
    "run_20260714_212901_00063",
    "run_20260714_211745_00057",
)
OUTPUT_JSON = PROJECT_ROOT / "artifacts/conversion/d2_cartesian_pipeline_spike.json"
OUTPUT_STATS = {
    "action15": PROJECT_ROOT / "artifacts/stats/pi0_eef_stats_spike_action15.json",
    "action30": PROJECT_ROOT / "artifacts/stats/pi0_eef_stats_spike_action30.json",
}
OUTPUT_MARKDOWN = PROJECT_ROOT / "reports/D2_CARTESIAN_PIPELINE_SPIKE.md"

POSITION_TOLERANCE_M = 1e-6
ROTATION_TOLERANCE_RAD = 1e-5
GRIPPER_TOLERANCE = 1e-12
NORMALIZATION_TOLERANCE = 1e-10
SO3_BRANCH_MARGIN_RAD = 0.1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _hash_files(paths: list[Path]) -> dict[str, str]:
    return {_relative_path(path): _sha256_file(path) for path in sorted(paths)}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _thresholds(config: dict[str, Any]) -> AlignmentThresholds:
    max_age_ms = config["alignment"]["max_age_ms"]
    return AlignmentThresholds(
        cam2_age_ns=int(round(float(max_age_ms["cam2"]) * 1e6)),
        eef_age_ns=int(round(float(max_age_ms["eef"]) * 1e6)),
        gripper_age_ns=int(round(float(max_age_ms["gripper"]) * 1e6)),
        qpos_age_ns=int(round(float(max_age_ms["qpos_audit"]) * 1e6)),
    )


def _gripper_mapping(config: dict[str, Any]) -> GripperMapping:
    gripper = config["gripper"]
    return GripperMapping(
        raw_open=float(gripper["raw_open"]),
        raw_closed=float(gripper["raw_closed"]),
        clip=bool(gripper["clip"]),
    )


def _assert_finite(name: str, value: Any) -> None:
    array = np.asarray(value)
    if array.dtype.kind in "biufc" and not np.all(np.isfinite(array)):
        raise AssertionError(f"{name} contains NaN or Inf")


def _assert_stats_finite(stats: dict[str, Any]) -> None:
    for feature_name in ("observation.state", "action"):
        for statistic_name, value in stats[feature_name].items():
            _assert_finite(f"{feature_name}.{statistic_name}", value)


def _roundtrip_metrics(
    carriers: Any,
    anchors: np.ndarray,
    *,
    profile: str,
    horizon_camera_intervals: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    absolute = absolute_action_chunks(
        carriers,
        anchors,
        profile=profile,
        horizon_camera_intervals=horizon_camera_intervals,
    )
    relative = model_relative_action_chunks(
        carriers,
        anchors,
        profile=profile,
        horizon_camera_intervals=horizon_camera_intervals,
    )
    anchor_pose = carriers.observation_pose_xyzw[anchors]
    anchor_rotation = quaternion_xyzw_to_matrix(anchor_pose[:, 3:7])
    decoded_position, decoded_rotation, decoded_gripper = decode_relative_action(
        anchor_pose[:, :3], anchor_rotation, relative
    )
    target_rotation = quaternion_xyzw_to_matrix(absolute[..., 3:7])

    position_error = np.linalg.norm(decoded_position - absolute[..., :3], axis=-1)
    rotation_error = rotation_geodesic_angle(decoded_rotation, target_rotation)
    gripper_error = np.abs(decoded_gripper - absolute[..., 7:8])
    theta = np.linalg.norm(relative[..., 3:6], axis=-1)
    for name, value in (
        ("absolute action carrier", absolute),
        ("relative action", relative),
        ("decoded position", decoded_position),
        ("decoded rotation", decoded_rotation),
        ("decoded gripper", decoded_gripper),
        ("SO(3) theta", theta),
    ):
        _assert_finite(name, value)

    metrics = {
        "position_max_error_m": float(np.max(position_error)),
        "rotation_max_geodesic_error_rad": float(np.max(rotation_error)),
        "gripper_max_abs_error": float(np.max(gripper_error)),
        "so3_theta_max_rad": float(np.max(theta)),
        "so3_theta_p50_rad": float(np.quantile(theta, 0.50)),
        "so3_theta_p95_rad": float(np.quantile(theta, 0.95)),
        "so3_theta_p99_rad": float(np.quantile(theta, 0.99)),
    }
    if metrics["position_max_error_m"] >= POSITION_TOLERANCE_M:
        raise AssertionError(f"{profile}: position round-trip exceeds tolerance: {metrics}")
    if metrics["rotation_max_geodesic_error_rad"] >= ROTATION_TOLERANCE_RAD:
        raise AssertionError(f"{profile}: rotation round-trip exceeds tolerance: {metrics}")
    if metrics["gripper_max_abs_error"] >= GRIPPER_TOLERANCE:
        raise AssertionError(f"{profile}: gripper round-trip exceeds tolerance: {metrics}")
    if metrics["so3_theta_max_rad"] >= np.pi - SO3_BRANCH_MARGIN_RAD:
        raise AssertionError(f"{profile}: SO(3) delta approaches the pi branch cut: {metrics}")
    return absolute, relative, metrics


def _verify_full_horizon(carriers: Any, anchors: np.ndarray) -> dict[str, bool]:
    """Verify the exact t+1..t+50 and midpoint/endpoint time grids."""

    camera_time = carriers.camera_log_time_ns
    if np.any(anchors < 0) or np.any(anchors + 50 >= len(camera_time)):
        raise AssertionError("full-horizon anchor falls outside its episode")
    first_15 = carriers.action_15hz_target_time_ns[anchors]
    last_15 = carriers.action_15hz_target_time_ns[anchors + 49]
    if not np.array_equal(first_15, camera_time[anchors + 1]):
        raise AssertionError("15 Hz first action is not camera target t+1")
    if not np.array_equal(last_15, camera_time[anchors + 50]):
        raise AssertionError("15 Hz final action is not camera target t+50")

    first_30_index = 2 * anchors
    last_30_index = first_30_index + 99
    first_30 = carriers.action_30hz_target_time_ns[first_30_index]
    last_30 = carriers.action_30hz_target_time_ns[last_30_index]
    expected_midpoint = camera_time[anchors] + (
        camera_time[anchors + 1] - camera_time[anchors]
    ) // 2
    if not np.array_equal(first_30, expected_midpoint):
        raise AssertionError("30 Hz first action is not the first camera-interval midpoint")
    if not np.array_equal(last_30, camera_time[anchors + 50]):
        raise AssertionError("30 Hz final action is not camera target t+50")

    offsets_15 = anchors[:, None] + np.arange(50, dtype=np.int64)[None, :]
    offsets_30 = 2 * anchors[:, None] + np.arange(100, dtype=np.int64)[None, :]
    if np.any(
        carriers.action_15hz_eef_source_time_ns[offsets_15]
        > carriers.action_15hz_target_time_ns[offsets_15]
    ):
        raise AssertionError("15 Hz action uses a non-causal EEF source")
    if np.any(
        carriers.action_15hz_gripper_source_time_ns[offsets_15]
        > carriers.action_15hz_target_time_ns[offsets_15]
    ):
        raise AssertionError("15 Hz action uses a non-causal gripper source")
    if np.any(
        carriers.action_30hz_eef_source_time_ns[offsets_30]
        > carriers.action_30hz_target_time_ns[offsets_30]
    ):
        raise AssertionError("30 Hz action uses a non-causal EEF source")
    if np.any(
        carriers.action_30hz_gripper_source_time_ns[offsets_30]
        > carriers.action_30hz_target_time_ns[offsets_30]
    ):
        raise AssertionError("30 Hz action uses a non-causal gripper source")
    return {
        "action15_first_is_t_plus_1": True,
        "action15_last_is_t_plus_50": True,
        "action30_first_is_interval_midpoint": True,
        "action30_last_is_t_plus_50": True,
        "all_action_sources_are_causal": True,
        "no_cross_episode_or_padded_target": True,
    }


def _normalization_roundtrip(values: np.ndarray, stats: dict[str, Any]) -> float:
    reconstructed = unnormalize(normalize(values, stats), stats)
    error = float(np.max(np.abs(reconstructed - values)))
    if not np.isfinite(error) or error >= NORMALIZATION_TOLERANCE:
        raise AssertionError(f"normalization round-trip failed: max_abs_error={error}")
    return error


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D2 Cartesian Pipeline Spike",
        "",
        f"- Phase status: **{report['phase_status']}**",
        f"- Generated at: `{report['generated_at_utc']}`",
        "- Scope: two raw MCAP episodes; numeric EEF/gripper/timestamp paths only.",
        "- Explicitly not performed: image conversion, full dataset conversion, PI0 training.",
        "",
        "## Episode and full-horizon coverage",
        "",
        "| Episode | Camera anchors | Candidate anchors | Valid K=50 | Valid K=100 | Endpoint max abs diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for episode in report["episodes"]:
        lines.append(
            "| {episode_id} | {camera_anchor_count} | {full_horizon_candidate_count} | "
            "{valid_action15_anchor_count} | {valid_action30_anchor_count} | {endpoint_max_abs_difference:.3e} |".format(
                **episode
            )
        )
    lines.extend(
        [
            "",
            "A valid anchor contains the current observation plus the complete future 50-camera-interval "
            "horizon; no action padding is used. The 30 Hz layout is midpoint, endpoint for every "
            "camera interval, so its endpoint slots (`1::2` in zero-based Python indexing) are copied "
            "exactly from the 15 Hz carrier.",
            "",
            "## Geometry and normalization acceptance",
            "",
            "| Profile | State shape | Action shape | Position error (m) | Rotation error (rad) | Gripper error | Max theta (rad) | State norm RT | Action norm RT |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for profile_name in ("action15", "action30"):
        profile = report["profiles"][profile_name]
        geometry = profile["geometry_roundtrip"]
        normalization = profile["normalization_roundtrip"]
        lines.append(
            "| {name} | {state} | {action} | {pos:.3e} | {rot:.3e} | {grip:.3e} | "
            "{theta:.6f} | {state_rt:.3e} | {action_rt:.3e} |".format(
                name=profile_name,
                state="×".join(str(item) for item in profile["observation_state_shape"]),
                action="×".join(str(item) for item in profile["action_chunk_shape"]),
                pos=geometry["position_max_error_m"],
                rot=geometry["rotation_max_geodesic_error_rad"],
                grip=geometry["gripper_max_abs_error"],
                theta=geometry["so3_theta_max_rad"],
                state_rt=normalization["observation_state_max_abs_error"],
                action_rt=normalization["action_max_abs_error"],
            )
        )
    checks = report["checks"]
    lines.extend(
        [
            "",
            "All model-visible arrays and statistics are finite. State is 10D "
            "(`[xyz, rotation6D, gripper]`) and action is 7D "
            "(`[delta xyz, body-frame rotation vector, gripper]`).",
            "",
            "Acceptance thresholds:",
            "",
            f"- Position encode/decode: `< {checks['thresholds']['position_error_m']:.1e} m`",
            f"- Rotation encode/decode: `< {checks['thresholds']['rotation_error_rad']:.1e} rad`",
            f"- Gripper encode/decode: `< {checks['thresholds']['gripper_error']:.1e}`",
            f"- SO(3) branch margin: `theta < pi - {checks['thresholds']['so3_branch_margin_rad']:.1f}`",
            f"- Normalize/unnormalize: `< {checks['thresholds']['normalization_error']:.1e}`",
            "",
            "## Reproducibility hashes",
            "",
            f"- Manifest SHA-256: `{report['inputs']['manifest_sha256']}`",
            f"- Combined config SHA-256: `{report['inputs']['combined_config_sha256']}`",
            f"- Pipeline source SHA-256: `{report['inputs']['pipeline_source_sha256']}`",
            f"- Two-episode source dataset SHA-256: `{report['inputs']['source_dataset_sha256']}`",
            "",
            "Individual config, source-code, and raw MCAP hashes are recorded in the machine-readable JSON.",
            "",
            "## Outputs and boundary",
            "",
            f"- Machine report: `{report['outputs']['machine_json']}`",
            f"- 15 Hz stats: `{report['outputs']['stats_action15']}`",
            f"- 30 Hz stats: `{report['outputs']['stats_action30']}`",
            "",
            "D2 passes for these two available episodes. This result does not authorize D3: it neither "
            "resolves missing raw episodes in D0 nor creates/trains on a full dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def run_spike(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    manifest_path = Path(config["raw_manifest"]).resolve()
    manifest = load_manifest(manifest_path)
    data_root = Path(manifest["data_root"])
    episodes_by_id = {episode["episode_id"]: episode for episode in manifest["episodes"]}
    missing_manifest_ids = [item for item in SPIKE_EPISODE_IDS if item not in episodes_by_id]
    if missing_manifest_ids:
        raise ValueError(f"spike episodes absent from manifest: {missing_manifest_ids}")

    profile_documents = {name: _load_yaml(path) for name, path in PROFILE_CONFIGS.items()}
    for profile_name, expected_chunk in (("action15", 50), ("action30", 100)):
        document = profile_documents[profile_name]
        if int(document["chunk_size"]) != expected_chunk:
            raise ValueError(f"{profile_name}: expected chunk size {expected_chunk}")
        if int(document["horizon_camera_intervals"]) != 50:
            raise ValueError(f"{profile_name}: expected 50 camera intervals")

    source_paths = sorted((PROJECT_ROOT / "src/franka_eef_pipeline").glob("*.py")) + [Path(__file__).resolve()]
    source_hashes = _hash_files(source_paths)
    config_hashes = _hash_files([config_path, *PROFILE_CONFIGS.values()])
    manifest_hash = _sha256_file(manifest_path)
    raw_paths = [resolve_mcap_path(episodes_by_id[item], data_root=data_root) for item in SPIKE_EPISODE_IDS]
    raw_hashes = _hash_files(raw_paths)
    raw_file_sizes = {_relative_path(path): path.stat().st_size for path in raw_paths}
    pipeline_source_hash = canonical_sha256(source_hashes)
    combined_config_hash = canonical_sha256(config_hashes)
    source_dataset_hash = canonical_sha256(
        {
            "manifest_sha256": manifest_hash,
            "episode_ids": SPIKE_EPISODE_IDS,
            "raw_mcap_sha256": raw_hashes,
            "combined_config_sha256": combined_config_hash,
            "pipeline_source_sha256": pipeline_source_hash,
        }
    )

    thresholds = _thresholds(config)
    gripper_mapping = _gripper_mapping(config)
    episode_reports: list[dict[str, Any]] = []
    states: dict[str, list[np.ndarray]] = {"action15": [], "action30": []}
    actions: dict[str, list[np.ndarray]] = {"action15": [], "action30": []}
    geometry_by_profile: dict[str, list[dict[str, float]]] = {"action15": [], "action30": []}

    for episode_id, raw_path in zip(SPIKE_EPISODE_IDS, raw_paths, strict=True):
        signals = load_episode_signals(raw_path, episode_id=episode_id)
        aligned = align_episode_to_camera(signals, thresholds)
        carriers = build_dual_rate_carriers(signals, aligned, thresholds, gripper_mapping)
        assert_dual_rate_crosscheck(carriers, atol=0.0)

        anchors_15 = valid_anchor_indices(
            carriers, profile="action15", horizon_camera_intervals=50
        )
        anchors_30 = valid_anchor_indices(
            carriers, profile="action30", horizon_camera_intervals=50
        )
        if len(anchors_15) == 0 or len(anchors_30) == 0:
            raise AssertionError(f"{episode_id}: no complete action horizon")
        if not np.array_equal(anchors_15, anchors_30):
            raise AssertionError(f"{episode_id}: dual-rate valid anchor sets differ")
        horizon_checks = _verify_full_horizon(carriers, anchors_15)

        absolute_15, relative_15, metrics_15 = _roundtrip_metrics(
            carriers,
            anchors_15,
            profile="action15",
            horizon_camera_intervals=50,
        )
        absolute_30, relative_30, metrics_30 = _roundtrip_metrics(
            carriers,
            anchors_30,
            profile="action30",
            horizon_camera_intervals=50,
        )
        endpoint_difference = float(np.max(np.abs(absolute_30[:, 1::2] - absolute_15)))
        if endpoint_difference != 0.0:
            raise AssertionError(f"{episode_id}: 30 Hz endpoint values are not bit-exact")
        if not np.array_equal(
            carriers.action_30hz_target_time_ns[1::2], carriers.action_15hz_target_time_ns
        ):
            raise AssertionError(f"{episode_id}: endpoint target times differ")

        for profile_name, anchors, relative, metrics in (
            ("action15", anchors_15, relative_15, metrics_15),
            ("action30", anchors_30, relative_30, metrics_30),
        ):
            selected_state = carriers.observation_state[anchors]
            _assert_finite(f"{episode_id} {profile_name} observation state", selected_state)
            states[profile_name].append(selected_state)
            actions[profile_name].append(relative)
            geometry_by_profile[profile_name].append(metrics)

        candidate_count = len(carriers.camera_log_time_ns) - 50
        episode_reports.append(
            {
                "episode_id": episode_id,
                "mcap_path": str(raw_path),
                "mcap_sha256": raw_hashes[_relative_path(raw_path)],
                "common_start_ns": aligned.common_start_ns,
                "common_end_ns": aligned.common_end_ns,
                "camera_anchor_count": len(carriers.camera_log_time_ns),
                "full_horizon_candidate_count": candidate_count,
                "valid_action15_anchor_count": len(anchors_15),
                "valid_action30_anchor_count": len(anchors_30),
                "first_valid_anchor": int(anchors_15[0]),
                "last_valid_anchor": int(anchors_15[-1]),
                "action15_chunk_shape": list(relative_15.shape),
                "action30_chunk_shape": list(relative_30.shape),
                "endpoint_max_abs_difference": endpoint_difference,
                "full_horizon_checks": horizon_checks,
                "action15_geometry_roundtrip": metrics_15,
                "action30_geometry_roundtrip": metrics_30,
            }
        )

    profile_reports: dict[str, Any] = {}
    stats_documents: dict[str, Any] = {}
    for profile_name, action_fps, chunk_size in (
        ("action15", 15, 50),
        ("action30", 30, 100),
    ):
        state = np.concatenate(states[profile_name], axis=0)
        action = np.concatenate(actions[profile_name], axis=0)
        if state.shape != (len(action), 10):
            raise AssertionError(
                f"{profile_name}: expected one 10D state per action chunk, got {state.shape}, {action.shape}"
            )
        if action.shape[1:] != (chunk_size, 7):
            raise AssertionError(f"{profile_name}: bad action chunk shape {action.shape}")
        stats = effective_eef_stats(
            state,
            action,
            observation_fps=15,
            action_fps=action_fps,
            chunk_size=chunk_size,
            source_dataset_hash=source_dataset_hash,
        )
        _assert_stats_finite(stats)
        state_normalization_error = _normalization_roundtrip(state, stats["observation.state"])
        action_normalization_error = _normalization_roundtrip(action, stats["action"])
        geometry = {
            key: max(item[key] for item in geometry_by_profile[profile_name])
            for key in geometry_by_profile[profile_name][0]
        }
        stats_document = json_ready(stats)
        stats_documents[profile_name] = stats_document
        profile_reports[profile_name] = {
            "observation_fps": 15,
            "action_fps": action_fps,
            "chunk_size": chunk_size,
            "anchor_count": len(action),
            "observation_state_shape": list(state.shape),
            "action_chunk_shape": list(action.shape),
            "effective_state_stat_count": int(stats["observation.state"]["count"][0]),
            "effective_action_stat_count": int(stats["action"]["count"][0]),
            "geometry_roundtrip": geometry,
            "normalization_roundtrip": {
                "observation_state_max_abs_error": state_normalization_error,
                "action_max_abs_error": action_normalization_error,
            },
            "stats_sha256": canonical_sha256(stats_document),
            "stats_output": str(OUTPUT_STATS[profile_name]),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "phase": "D2_CARTESIAN_PIPELINE_SPIKE",
        "phase_status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "episode_ids": list(SPIKE_EPISODE_IDS),
            "episode_count": 2,
            "image_conversion_performed": False,
            "full_dataset_conversion_performed": False,
            "training_performed": False,
        },
        "inputs": {
            "config_path": str(config_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "config_sha256": config_hashes,
            "combined_config_sha256": combined_config_hash,
            "source_code_sha256": source_hashes,
            "pipeline_source_sha256": pipeline_source_hash,
            "raw_mcap_sha256": raw_hashes,
            "raw_mcap_size_bytes": raw_file_sizes,
            "source_dataset_sha256": source_dataset_hash,
        },
        "contract": {
            "task_instruction": config["task_instruction"],
            "timestamp_source": config["timestamp_source"],
            "alignment_strategy": config["alignment"]["strategy"],
            "common_interval_required_streams": config["alignment"]["required_streams"],
            "zero_fill_missing_camera": config["alignment"]["zero_fill_missing_camera"],
            "state_dimension": 10,
            "action_dimension": 7,
            "horizon_camera_intervals": 50,
            "action15_chunk_size": 50,
            "action30_chunk_size": 100,
        },
        "episodes": episode_reports,
        "profiles": profile_reports,
        "checks": {
            "status": "PASS",
            "full_horizon_without_padding": True,
            "action15_first_is_t_plus_1": True,
            "action15_last_is_t_plus_50": True,
            "action30_first_is_interval_midpoint": True,
            "action30_last_is_t_plus_50": True,
            "all_action_sources_are_causal": True,
            "dual_rate_valid_anchor_sets_equal": True,
            "action30_endpoint_value_identity": True,
            "action30_endpoint_timestamp_identity": True,
            "all_model_arrays_finite": True,
            "all_effective_stats_finite": True,
            "geometry_encode_decode": True,
            "so3_branch_margin": True,
            "normalization_roundtrip": True,
            "thresholds": {
                "position_error_m": POSITION_TOLERANCE_M,
                "rotation_error_rad": ROTATION_TOLERANCE_RAD,
                "gripper_error": GRIPPER_TOLERANCE,
                "so3_branch_margin_rad": SO3_BRANCH_MARGIN_RAD,
                "normalization_error": NORMALIZATION_TOLERANCE,
            },
        },
        "outputs": {
            "machine_json": str(OUTPUT_JSON),
            "stats_action15": str(OUTPUT_STATS["action15"]),
            "stats_action30": str(OUTPUT_STATS["action30"]),
            "markdown_report": str(OUTPUT_MARKDOWN),
        },
    }

    for path in (OUTPUT_JSON, *OUTPUT_STATS.values(), OUTPUT_MARKDOWN):
        path.parent.mkdir(parents=True, exist_ok=True)
    for profile_name, stats_document in stats_documents.items():
        OUTPUT_STATS[profile_name].write_text(
            json.dumps(stats_document, indent=2, ensure_ascii=False) + "\n"
        )
    OUTPUT_JSON.write_text(json.dumps(json_ready(report), indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MARKDOWN.write_text(_render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run_spike(args.config)
    print(
        json.dumps(
            {
                "phase_status": report["phase_status"],
                "episode_ids": report["scope"]["episode_ids"],
                "action15_anchors": report["profiles"]["action15"]["anchor_count"],
                "action30_anchors": report["profiles"]["action30"]["anchor_count"],
                "machine_json": report["outputs"]["machine_json"],
                "markdown_report": report["outputs"]["markdown_report"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
