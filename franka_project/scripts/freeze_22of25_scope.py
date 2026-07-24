#!/usr/bin/env python3
"""Freeze and verify the versioned Frank3 22-of-25 training scope.

The source manifest and raw MCAP files are read-only inputs. The script writes
only project-local provenance artifacts declared by the scope config.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/data/franka_scope_22of25_v1.yaml"


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, output)


def resolve_declared_mcap(episode: dict[str, Any], *, data_root: Path) -> Path:
    bag_dir = Path(episode.get("path") or data_root / episode["relative_path"])
    files = episode.get("mcap_files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError(
            f"{episode.get('episode_id')}: expected exactly one declared MCAP, got {files}"
        )
    return bag_dir / str(files[0])


def validate_source_manifest(source: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = source.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("source manifest has no episode list")
    expected = int(config["source"]["expected_episode_count"])
    if len(episodes) != expected or int(source.get("n_episodes", -1)) != expected:
        raise ValueError(f"source manifest must contain exactly {expected} episodes")
    ids = [str(item.get("episode_id")) for item in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("source manifest contains duplicate episode_id values")
    required_status = str(config["source"]["required_status"])
    for item in episodes:
        if item.get("status") != required_status:
            raise ValueError(f"non-{required_status} episode in source: {item.get('episode_id')}")
        if config["source"]["require_ok_for_training"] and not item.get("ok_for_training"):
            raise ValueError(f"episode is not training-approved: {item.get('episode_id')}")
    return episodes


def format_report(
    *,
    config: dict[str, Any],
    source_sha256: str,
    config_sha256: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    integrity: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> str:
    total_bytes = sum(int(item["size_bytes"]) for item in integrity)
    lines = [
        "# D1 Frank3 22of25 范围冻结",
        "",
        "## 结论",
        "",
        "D1 已通过版本化派生 manifest 将正式 D3 范围冻结为当前存在的 22 条 strict-PASS MCAP。",
        "原始 25 条 manifest 保持只读且未修改；3 条缺失项被显式记录，不使用 viz_cache 替代。",
        "在同一机器并复用采集时 configured EE/TCP 的约束下，D3 数据转换已获准。",
        "本范围锁不授权真机 rollout，也没有执行数据转换或训练。",
        "",
        "## 冻结汇总",
        "",
        f"- Scope：`{config['name']}`",
        f"- 状态：`{config['status']}`",
        f"- source / included / excluded：`25 / {manifest['n_episodes']} / {len(excluded)}`",
        f"- duration：`{manifest['total_duration_s']:.3f} s`",
        f"- cam1 / cam2：`{manifest['total_cam_frames']} / {manifest['total_cam2_frames']}`",
        f"- MCAP 总字节：`{total_bytes}`",
        "- MCAP identity：逐文件 byte size + streamed full-file SHA-256",
        "",
        "## 显式排除",
        "",
        "| source index | episode | duration s | cam1 | 原因 |",
        "|---:|---|---:|---:|---|",
    ]
    for item in excluded:
        lines.append(
            f"| {item['source_index']} | `{item['episode_id']}` | "
            f"{item['duration_s']:.3f} | {item['cam1_frames']} | `{item['reason']}` |"
        )
    lines.extend(
        [
            "",
            "## 纳入的 MCAP 锁",
            "",
            "| order | episode | bytes | SHA-256 |",
            "|---:|---|---:|---|",
        ]
    )
    for item in integrity:
        lines.append(
            f"| {item['derived_index']} | `{item['episode_id']}` | {item['size_bytes']} | "
            f"`{item['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Source manifest：`{config['source']['manifest']}`",
            f"- Source manifest SHA-256：`{source_sha256}`",
            f"- Scope config SHA-256：`{config_sha256}`",
            f"- Derived manifest：`{config['outputs']['derived_manifest']}`",
            f"- Derived manifest SHA-256：`{manifest_sha256}`",
            f"- D0 audit：`{config['integrity']['d0_audit']}`",
            "",
            "## Gate",
            "",
            "- D3：仅对本锁中的 22 条允许。",
            "- 原始 25 条 manifest：仍不可被描述为 25/25 可转换。",
            "- 真机：必须保持与采集时相同 configured EE/TCP；本锁本身不授权 rollout。",
            "- 若任一 MCAP size/hash 变化，scope lock 立即失效，需重新冻结并重跑 D0/D1。",
            "",
        ]
    )
    return "\n".join(lines)


def freeze(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    source_path = Path(config["source"]["manifest"])
    source_sha256 = sha256_file(source_path)
    expected_source_sha = str(config["source"]["manifest_sha256"])
    if source_sha256 != expected_source_sha:
        raise ValueError(
            f"source manifest hash changed: expected {expected_source_sha}, got {source_sha256}"
        )
    source = load_yaml(source_path)
    source_episodes = validate_source_manifest(source, config)
    data_root = Path(source.get("data_root") or source_path.parent.parent)
    expected_excluded = list(config["selection"]["excluded_missing_episode_ids"])
    expected_excluded_set = set(expected_excluded)
    if len(expected_excluded) != len(expected_excluded_set):
        raise ValueError("excluded_missing_episode_ids contains duplicates")

    d0_path = Path(config["integrity"]["d0_audit"])
    d0 = load_json(d0_path)
    d0_missing = {item["episode_id"] for item in d0.get("missing_raw_episodes", [])}
    if d0_missing != expected_excluded_set:
        raise ValueError(f"D0 missing set differs from scope config: {sorted(d0_missing)}")
    d0_sizes = {
        item["episode_id"]: int(item["mcap_size_bytes"])
        for item in d0.get("episodes", [])
    }

    included: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    chunk_bytes = int(config["integrity"]["hash_chunk_bytes"])
    for source_index, source_episode in enumerate(source_episodes):
        episode_id = str(source_episode["episode_id"])
        mcap_path = resolve_declared_mcap(source_episode, data_root=data_root)
        if not mcap_path.is_file():
            excluded.append(
                {
                    "source_index": source_index,
                    "episode_id": episode_id,
                    "expected_mcap_path": str(mcap_path),
                    "reason": str(config["selection"]["excluded_reason"]),
                    "duration_s": float(source_episode["duration_s"]),
                    "cam1_frames": int(source_episode["approx_cam_frames"]),
                    "source_episode_record_sha256": canonical_sha256(source_episode),
                }
            )
            continue
        if episode_id in expected_excluded_set:
            raise ValueError(f"configured missing episode is now present; re-scope required: {episode_id}")
        size_bytes = mcap_path.stat().st_size
        if config["integrity"]["require_positive_size"] and size_bytes <= 0:
            raise ValueError(f"empty MCAP: {mcap_path}")
        if config["integrity"]["compare_size_with_d0_audit"]:
            expected_size = d0_sizes.get(episode_id)
            if expected_size != size_bytes:
                raise ValueError(
                    f"D0 size mismatch for {episode_id}: expected {expected_size}, got {size_bytes}"
                )
        episode = copy.deepcopy(source_episode)
        derived_index = len(included)
        file_sha256 = sha256_file(mcap_path, chunk_bytes=chunk_bytes)
        episode["scope_source_index"] = source_index
        episode["mcap_integrity"] = [
            {
                "name": mcap_path.name,
                "size_bytes": size_bytes,
                "sha256": file_sha256,
            }
        ]
        included.append(episode)
        integrity.append(
            {
                "derived_index": derived_index,
                "source_index": source_index,
                "episode_id": episode_id,
                "path": str(mcap_path),
                "size_bytes": size_bytes,
                "sha256": file_sha256,
            }
        )

    excluded_ids = [item["episode_id"] for item in excluded]
    if set(excluded_ids) != expected_excluded_set or excluded_ids != expected_excluded:
        raise ValueError(
            "actual exclusions do not exactly match the configured missing IDs in source order: "
            f"{excluded_ids}"
        )
    if len(included) != int(config["selection"]["expected_included_count"]):
        raise ValueError(f"expected 22 included episodes, got {len(included)}")
    if len(excluded) != int(config["selection"]["expected_excluded_count"]):
        raise ValueError(f"expected 3 excluded episodes, got {len(excluded)}")
    source_indices = [int(item["scope_source_index"]) for item in included]
    if source_indices != sorted(source_indices):
        raise ValueError("derived manifest does not preserve source order")

    total_duration_s = round(sum(float(item["duration_s"]) for item in included), 3)
    total_cam1 = sum(int(item["topic_counts"]["/camera1/camera1/color/image_raw"]) for item in included)
    total_cam2 = sum(int(item["topic_counts"]["/camera2/camera2/color/image_raw"]) for item in included)
    expected_totals = config["expected_totals"]
    if abs(total_duration_s - float(expected_totals["duration_s"])) > 1e-9:
        raise ValueError(f"duration mismatch: {total_duration_s}")
    if total_cam1 != int(expected_totals["cam1_frames"]):
        raise ValueError(f"cam1 count mismatch: {total_cam1}")
    if total_cam2 != int(expected_totals["cam2_frames"]):
        raise ValueError(f"cam2 count mismatch: {total_cam2}")

    generated_at = datetime.now(timezone.utc).isoformat()
    config_sha256 = sha256_file(config_path)
    selection_payload = {
        "scope_name": config["name"],
        "source_manifest_sha256": source_sha256,
        "included": [item["episode_id"] for item in integrity],
        "excluded": excluded_ids,
        "mcap_integrity": integrity,
    }
    scope_content_sha256 = canonical_sha256(selection_payload)
    manifest = {
        "schema_version": 1,
        "name": "frank3_train_passed_22of25_v1",
        "status": "FROZEN_D3_AUTHORIZED",
        "generated_at_utc": generated_at,
        "scope_config": str(config_path.resolve()),
        "scope_config_sha256": config_sha256,
        "scope_content_sha256": scope_content_sha256,
        "source_manifest": str(source_path),
        "source_manifest_sha256": source_sha256,
        "data_root": str(data_root),
        "selection_policy": config["selection"]["policy"],
        "n_source_episodes": len(source_episodes),
        "n_episodes": len(included),
        "n_excluded_missing": len(excluded),
        "total_duration_s": total_duration_s,
        "total_cam_frames": total_cam1,
        "total_cam2_frames": total_cam2,
        "task_instruction": config["contract"]["task_instruction"],
        "timestamp_source": config["contract"]["timestamp_source"],
        "eef_semantic": config["contract"]["eef_semantic"],
        "deployment_requirement": config["contract"]["deployment_requirement"],
        "excluded_missing": excluded,
        "mcap_integrity_strategy": {
            "identity": config["integrity"]["mcap_identity_strategy"],
            "algorithm": config["integrity"]["hash_algorithm"],
            "mode": config["integrity"]["hash_mode"],
            "chunk_bytes": chunk_bytes,
        },
        "episodes": included,
    }
    manifest_path = Path(config["outputs"]["derived_manifest"])
    manifest_text = yaml.safe_dump(
        manifest,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    atomic_write_text(manifest_path, manifest_text)
    manifest_sha256 = sha256_file(manifest_path)

    report_path = Path(config["outputs"]["report"])
    report_text = format_report(
        config=config,
        source_sha256=source_sha256,
        config_sha256=config_sha256,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        integrity=integrity,
        excluded=excluded,
    )
    atomic_write_text(report_path, report_text)
    report_sha256 = sha256_file(report_path)

    lock = {
        "schema_version": 1,
        "phase": "D1_SCOPE_FREEZE",
        "scope": config["name"],
        "status": "PASS_D3_AUTHORIZED_FOR_VERSIONED_22OF25_SCOPE",
        "generated_at_utc": generated_at,
        "authorization": config["authorization"],
        "source": {
            "manifest": str(source_path),
            "manifest_sha256": source_sha256,
            "episode_count": len(source_episodes),
            "conversion_authorized": False,
        },
        "derived": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "scope_content_sha256": scope_content_sha256,
            "episode_count": len(included),
            "duration_s": total_duration_s,
            "cam1_frames": total_cam1,
            "cam2_frames": total_cam2,
            "d3_conversion_authorized": True,
        },
        "excluded_missing": excluded,
        "mcap_integrity": {
            "strategy": config["integrity"]["mcap_identity_strategy"],
            "algorithm": "sha256",
            "chunk_bytes": chunk_bytes,
            "total_bytes": sum(int(item["size_bytes"]) for item in integrity),
            "files": integrity,
        },
        "provenance": {
            "scope_config": str(config_path.resolve()),
            "scope_config_sha256": config_sha256,
            "freeze_script": str(Path(__file__).resolve()),
            "freeze_script_sha256": sha256_file(__file__),
            "d0_audit": str(d0_path),
            "d0_audit_sha256": sha256_file(d0_path),
            "d1_contract": config["contract"]["d1_contract"],
            "d1_contract_sha256": sha256_file(config["contract"]["d1_contract"]),
            "report": str(report_path),
            "report_sha256": report_sha256,
        },
        "verification": {
            "status": "PASS",
            "source_hash_match": True,
            "source_order_preserved": True,
            "source_partition_exact": True,
            "d0_missing_set_match": True,
            "d0_size_match_for_all_included": True,
            "full_mcap_sha256_computed": True,
            "raw_files_modified": False,
        },
        "gates": {
            "d3_allowed": True,
            "d3_scope_only": config["name"],
            "d3_input_manifest_only": str(manifest_path),
            "original_25_episode_manifest_allowed_for_d3": False,
            "real_robot_rollout_allowed": False,
            "real_robot_requirement": config["contract"]["deployment_requirement"],
        },
    }
    lock_path = Path(config["outputs"]["lock"])
    atomic_write_text(lock_path, json.dumps(lock, ensure_ascii=False, indent=2) + "\n")

    # Reload all outputs before declaring success.
    reloaded_manifest = load_yaml(manifest_path)
    reloaded_lock = load_json(lock_path)
    if reloaded_manifest["scope_content_sha256"] != scope_content_sha256:
        raise ValueError("derived manifest scope hash changed after serialization")
    if reloaded_lock["derived"]["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("lock manifest hash mismatch after serialization")
    if reloaded_lock["provenance"]["report_sha256"] != sha256_file(report_path):
        raise ValueError("lock report hash mismatch after serialization")
    return lock


def verify(config_path: Path, *, rehash_mcap: bool) -> dict[str, Any]:
    config = load_yaml(config_path)
    lock_path = Path(config["outputs"]["lock"])
    lock = load_json(lock_path)
    errors: list[str] = []
    if sha256_file(config["source"]["manifest"]) != lock["source"]["manifest_sha256"]:
        errors.append("source manifest SHA-256 mismatch")
    if sha256_file(config["outputs"]["derived_manifest"]) != lock["derived"]["manifest_sha256"]:
        errors.append("derived manifest SHA-256 mismatch")
    if sha256_file(config["outputs"]["report"]) != lock["provenance"]["report_sha256"]:
        errors.append("scope report SHA-256 mismatch")
    if sha256_file(config_path) != lock["provenance"]["scope_config_sha256"]:
        errors.append("scope config SHA-256 mismatch")
    if sha256_file(__file__) != lock["provenance"]["freeze_script_sha256"]:
        errors.append("freeze script SHA-256 mismatch")
    if sha256_file(config["contract"]["d1_contract"]) != lock["provenance"]["d1_contract_sha256"]:
        errors.append("D1 contract SHA-256 mismatch")
    for item in lock["mcap_integrity"]["files"]:
        path = Path(item["path"])
        if not path.is_file():
            errors.append(f"MCAP missing: {path}")
            continue
        if path.stat().st_size != int(item["size_bytes"]):
            errors.append(f"MCAP size mismatch: {path}")
        if rehash_mcap and sha256_file(
            path, chunk_bytes=int(lock["mcap_integrity"]["chunk_bytes"])
        ) != item["sha256"]:
            errors.append(f"MCAP SHA-256 mismatch: {path}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "status": "PASS",
        "scope": lock["scope"],
        "episode_count": lock["derived"]["episode_count"],
        "mcap_count": len(lock["mcap_integrity"]["files"]),
        "mcap_content_rehashed": rehash_mcap,
        "lock_sha256": sha256_file(lock_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--rehash-mcap", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.config, rehash_mcap=args.rehash_mcap)
    else:
        if args.rehash_mcap:
            parser.error("--rehash-mcap is only valid with --verify-only")
        config = load_yaml(args.config)
        frozen_outputs = [Path(config["outputs"][key]) for key in ("derived_manifest", "report", "lock")]
        if not args.force and any(path.exists() for path in frozen_outputs):
            parser.error(
                "scope outputs already exist; use --verify-only to validate the frozen scope "
                "or --force to create a new freeze with new artifact hashes"
            )
        result = freeze(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
