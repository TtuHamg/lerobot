#!/usr/bin/env python
"""Auditable, resumable M0 in-sample evaluation for the frozen PI0 F0-15 run.

This tool is intentionally read-only with respect to the training run.  It:

* verifies the completed F0 artifact contract and compares every recorded
  fixed-monitor milestone, selecting ``best_train_monitor`` by minimum loss;
* reconstructs all 10 epoch milestones on the identical 747-anchor monitor
  subset with one fixed prediction seed for like-for-like Cartesian metrics;
* reconstructs the final checkpoint on all 7,465 training anchors with fixed
  PI0 prediction noise and multiple seeds;
* writes independently hashed temporary NPZ shards that can be resumed; and
* atomically publishes a JSON report only after every expected shard passes
  its contract and content checks.

The result is teacher-forced and entirely in-sample.  It is not validation,
test evaluation, a rollout, or evidence of real-robot task success.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for import_root in (PROJECT_SRC, SCRIPTS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from franka_eef_pipeline.dual_rate_dataset import CartesianAnchorDataset  # noqa: E402
from franka_eef_pipeline.geometry import rotation_geodesic_angle, so3_exp  # noqa: E402
from franka_eef_pipeline.pi0_training import (  # noqa: E402
    PI0_CORE_WEIGHTS_NAMESPACE,
    load_pi0_full_policy_and_processors,
)
from verify_pi0_f0_run import (  # noqa: E402
    DATASET_SIZE,
    EXPECTED_EPOCH_CHECKPOINT_STEPS,
    EXPECTED_MONITOR_STEPS,
    FINAL_F0_STEP,
    PRECHECK_STEP,
    VerifiedLocalRun,
    load_saved_processors,
    verify_local_run,
    verify_wandb,
)


SCHEMA_VERSION = 1
ACTION_CHUNK_SIZE = 50
ACTION_DIM = 7
STATE_DIM = 10
MAX_ACTION_DIM = 32
NUM_INFERENCE_STEPS = 10
HORIZONS = (1, 5, 10, 25, 50)
# F0 training seed (1000) plus disjoint million-wide namespaces.  Adding the
# logical anchor index (0..7464) cannot collide across these fixed seeds.
DEFAULT_PREDICTION_SEEDS = (1_001_000, 2_001_000, 3_001_000)
DEFAULT_NUM_SHARDS = 32
DEFAULT_BATCH_SIZE = 4
MONITOR_CHECKPOINT_STEPS = EXPECTED_EPOCH_CHECKPOINT_STEPS
CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
)
MONITOR_PATTERN = re.compile(
    r"\bstep=(\d+)\s+monitor_loss=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\b"
)


class M0EvaluationError(RuntimeError):
    """Raised when an M0 artifact, shard, or metric violates the contract."""


@dataclass(frozen=True)
class PreparedM0:
    local: VerifiedLocalRun
    contract: dict[str, Any]
    contract_sha256: str
    work_dir: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M0EvaluationError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    _require(path.is_file(), f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise M0EvaluationError(f"invalid JSON artifact: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _jsonable(value), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz"
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        np.savez_compressed(temporary_path, **arrays)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            # Hard-link publication is atomic and O_EXCL-like: two workers
            # accidentally assigned the same shard cannot silently replace
            # each other's completed prediction payload.
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise M0EvaluationError(
                f"concurrent worker already published NPZ destination: {path}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _resolved_inside(path: Path, parent: Path) -> bool:
    return path.expanduser().resolve().is_relative_to(parent.expanduser().resolve())


def _safe_child(root: Path, *parts: str) -> Path:
    resolved_root = root.expanduser().resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    _require(
        candidate.is_relative_to(resolved_root),
        f"artifact path escapes M0 work directory: {candidate}",
    )
    return candidate


def shard_indices(dataset_size: int, num_shards: int, shard_index: int) -> np.ndarray:
    """Return one deterministic contiguous shard, with exact full-set coverage."""

    _require(type(dataset_size) is int and dataset_size > 0, "dataset_size must be positive")
    _require(type(num_shards) is int and 1 <= num_shards <= dataset_size, "invalid num_shards")
    _require(type(shard_index) is int and 0 <= shard_index < num_shards, "invalid shard_index")
    boundaries = np.linspace(0, dataset_size, num_shards + 1, dtype=np.int64)
    return np.arange(boundaries[shard_index], boundaries[shard_index + 1], dtype=np.int64)


def parse_monitor_losses(
    log_paths: Sequence[Path], *, expected_steps: Iterable[int] = EXPECTED_MONITOR_STEPS
) -> dict[int, float]:
    """Parse rank-0 monitor records and reject missing/conflicting duplicates."""

    values: dict[int, list[float]] = {}
    for path in log_paths:
        _require(path.is_file() and path.stat().st_size > 0, f"missing/empty monitor log: {path}")
        text = path.read_text(encoding="utf-8", errors="strict")
        for match in MONITOR_PATTERN.finditer(text):
            step = int(match.group(1))
            loss = float(match.group(2))
            _require(math.isfinite(loss) and loss >= 0.0, f"invalid monitor loss at step {step}")
            values.setdefault(step, []).append(loss)

    expected = set(int(step) for step in expected_steps)
    _require(set(values) == expected, f"monitor step mismatch: expected={sorted(expected)}, got={sorted(values)}")
    result: dict[int, float] = {}
    for step, duplicates in values.items():
        reference = duplicates[0]
        _require(
            all(math.isclose(value, reference, rel_tol=0.0, abs_tol=5e-9) for value in duplicates),
            f"conflicting duplicate monitor values at step {step}: {duplicates}",
        )
        result[step] = reference
    return result


def select_best_monitor_checkpoint(
    monitor_losses: Mapping[int, float],
    *,
    eligible_steps: Sequence[int] = EXPECTED_EPOCH_CHECKPOINT_STEPS,
    last_step: int | None = None,
) -> dict[str, Any]:
    """Select the minimum-loss epoch checkpoint, with earlier-step tie-break."""

    eligible = tuple(int(step) for step in eligible_steps)
    _require(eligible and all(step in monitor_losses for step in eligible), "eligible monitor loss missing")
    resolved_last_step = eligible[-1] if last_step is None else int(last_step)
    _require(resolved_last_step in monitor_losses, "last checkpoint monitor loss missing")
    best_step = min(eligible, key=lambda step: (float(monitor_losses[step]), step))
    records = []
    for step in sorted(monitor_losses):
        checkpoint_exists = step == PRECHECK_STEP or step in eligible
        records.append(
            {
                "step": step,
                "train_monitor_loss": float(monitor_losses[step]),
                "checkpoint_exists": checkpoint_exists,
                "eligible_for_best_train_monitor": step in eligible,
                "is_best_train_monitor": step == best_step,
                "is_last": step == resolved_last_step,
            }
        )
    return {
        "selection_metric": "train_monitor_loss",
        "selection_rule": "minimum over epoch milestone checkpoints; earlier step breaks exact ties",
        "best_train_monitor_step": best_step,
        "best_train_monitor_loss": float(monitor_losses[best_step]),
        "last_step": resolved_last_step,
        "last_train_monitor_loss": float(monitor_losses[resolved_last_step]),
        "records": records,
    }


def _invocation_log_paths(local: VerifiedLocalRun) -> list[Path]:
    records = [_read_json(path) for path in sorted((local.run_dir / "invocations").glob("*.json"))]
    records.sort(key=lambda item: str(item.get("started_at", "")))
    paths: list[Path] = []
    for record in records:
        path = Path(str(record.get("log_path", ""))).expanduser().resolve()
        _require(path.is_relative_to((local.run_dir / "logs").resolve()), "invocation log escapes run/logs")
        _require(path.name.endswith("_r0.log"), f"invocation does not point to rank-0 log: {path}")
        paths.append(path)
    _require(len(paths) == 2 and len(set(paths)) == 2, "F0 must have two distinct rank-0 invocation logs")
    return paths


def _processor_state_files(model_dir: Path) -> list[Path]:
    result: list[Path] = []
    for config_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        config = _read_json(model_dir / config_name)
        steps = config.get("steps")
        _require(isinstance(steps, list) and steps, f"processor has no steps: {config_name}")
        for step in steps:
            _require(isinstance(step, dict), f"invalid processor step: {config_name}")
            state_file = step.get("state_file")
            if state_file is None:
                continue
            _require(isinstance(state_file, str) and state_file, "invalid processor state_file")
            path = (model_dir / state_file).resolve()
            _require(path.is_relative_to(model_dir.resolve()), "processor state_file escapes model directory")
            _require(path.is_file() and path.stat().st_size > 0, f"missing processor state: {path}")
            result.append(path)
    _require(len(result) == len(set(result)), "duplicate processor state file reference")
    return result


def _verify_model_files_and_hashes(model_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = model_dir / "franka_pi0_checkpoint_manifest.json"
    manifest = _read_json(manifest_path)
    _require(manifest.get("weights_namespace") == PI0_CORE_WEIGHTS_NAMESPACE, "checkpoint namespace drift")
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "checkpoint model file manifest missing")
    ledger: dict[str, dict[str, Any]] = {}
    for filename, expected in sorted(files.items()):
        _require(isinstance(filename, str) and isinstance(expected, dict), "invalid model file record")
        path = (model_dir / filename).resolve()
        _require(path.is_relative_to(model_dir.resolve()), "model file escapes checkpoint")
        _require(path.is_file(), f"missing checkpoint model file: {path}")
        digest = _sha256_file(path)
        _require(path.stat().st_size == expected.get("size_bytes"), f"checkpoint size mismatch: {path}")
        _require(digest == expected.get("sha256"), f"checkpoint SHA256 mismatch: {path}")
        ledger[filename] = {"size_bytes": path.stat().st_size, "sha256": digest}
    for path in _processor_state_files(model_dir):
        ledger[path.name] = {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    ledger[manifest_path.name] = {
        "size_bytes": manifest_path.stat().st_size,
        "sha256": _sha256_file(manifest_path),
    }
    return ledger


def _dataset_file_ledger(dataset_root: Path) -> dict[str, Any]:
    """Hash every regular file in the frozen local LeRobot dataset."""

    root = dataset_root.expanduser().resolve()
    _require(root.is_dir(), f"dataset root is missing: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        _require(
            path.resolve().is_relative_to(root),
            f"dataset file symlink escapes frozen root: {path}",
        )
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _require(records, "dataset file ledger is empty")
    return {
        "file_count": len(records),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "ledger_sha256": _canonical_sha256(records),
        "files": records,
    }


def _validate_prediction_seeds(prediction_seeds: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in prediction_seeds)
    _require(seeds and len(seeds) == len(set(seeds)), "prediction seeds must be non-empty and unique")
    _require(all(0 <= seed <= (2**63 - 1) - DATASET_SIZE for seed in seeds), "prediction seed out of range")
    _require(
        all(
            abs(first - second) >= DATASET_SIZE
            for ordinal, first in enumerate(seeds)
            for second in seeds[ordinal + 1 :]
        ),
        "prediction seed namespaces overlap after adding logical indices",
    )
    return seeds


def _build_contract(
    local: VerifiedLocalRun,
    *,
    prediction_seeds: Sequence[int],
    num_shards: int,
    batch_size: int,
) -> dict[str, Any]:
    seeds = _validate_prediction_seeds(prediction_seeds)
    _require(1 <= num_shards <= DATASET_SIZE, "num_shards out of range")
    _require(batch_size > 0, "batch_size must be positive")
    log_paths = _invocation_log_paths(local)
    monitor_losses = parse_monitor_losses(log_paths)
    selection = select_best_monitor_checkpoint(monitor_losses)

    final_ledger = _verify_model_files_and_hashes(local.model_dir)
    best_step = int(selection["best_train_monitor_step"])
    best_model_dir = local.run_dir / f"checkpoints/step-{best_step:06d}/pretrained_model"
    best_ledger = final_ledger if best_step == FINAL_F0_STEP else _verify_model_files_and_hashes(best_model_dir)

    dataset_root = Path(str(local.config["dataset"]["root"])).expanduser().resolve()
    profile_path = dataset_root / "meta/franka_eef_profile.json"
    stats_path = dataset_root / "meta/pi0_eef_stats.json"
    profile = _read_json(profile_path)
    stats = _read_json(stats_path)
    saved_stats_path = local.model_dir / "pi0_eef_stats.json"
    _require(_sha256_file(saved_stats_path) == _sha256_file(stats_path), "final checkpoint stats differ from dataset")
    _require(
        profile.get("logical_anchor_index_sha256")
        == stats.get("logical_anchor_index_sha256"),
        "logical anchor hash drift",
    )
    _require(profile.get("scope_content_sha256") == stats.get("scope_content_sha256"), "scope hash drift")
    batches_per_seed = sum(
        math.ceil(len(shard_indices(DATASET_SIZE, num_shards, shard_index)) / batch_size)
        for shard_index in range(num_shards)
    )
    monitor_batches_per_checkpoint = math.ceil(len(local.monitor["indices"]) / batch_size)
    dataset_ledger = _dataset_file_ledger(dataset_root)

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "PI0_F0_M0_teacher_forced_in_sample",
        "tool_sha256": _sha256_file(Path(__file__).resolve()),
        "run": {
            "run_dir": str(local.run_dir),
            "stage": "F0-15",
            "config_sha256": _sha256_file(local.run_dir / "resolved_config.yaml"),
            "run_manifest_sha256": _sha256_file(local.run_dir / "run_manifest.json"),
            "monitor_subset_sha256": local.run_manifest["monitor_subset_sha256"],
            "final_step": FINAL_F0_STEP,
            "final_checkpoint_manifest_sha256": _sha256_file(local.checkpoint / "checkpoint_manifest.json"),
            "final_model_dir": str(local.model_dir),
            "best_train_monitor_step": best_step,
            "best_model_dir": str(best_model_dir.resolve()),
        },
        "dataset": {
            "root": str(dataset_root),
            "size": DATASET_SIZE,
            "episodes": 22,
            "profile": "action15",
            "observation_fps": 15,
            "action_fps": 15,
            "chunk_size": ACTION_CHUNK_SIZE,
            "profile_sha256": _sha256_file(profile_path),
            "stats_sha256": _sha256_file(stats_path),
            "logical_anchor_index_sha256": profile["logical_anchor_index_sha256"],
            "scope_content_sha256": profile["scope_content_sha256"],
            "task_instruction": profile["task_instruction"],
        },
        "checkpoint_selection": selection,
        "inference": {
            "checkpoint_step": FINAL_F0_STEP,
            "prediction_seeds": list(seeds),
            "per_anchor_noise_seed": "prediction_seed + logical_index",
            "noise_generator": "CPU torch.Generator, float32",
            "noise_shape_per_anchor": [ACTION_CHUNK_SIZE, MAX_ACTION_DIM],
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "batch_size": batch_size,
            "model_precision": "bfloat16",
            "cost_units": {
                "anchor_seed_predictions": DATASET_SIZE * len(seeds),
                "predict_action_chunk_calls": batches_per_seed * len(seeds),
                "flow_denoising_model_steps": batches_per_seed
                * len(seeds)
                * NUM_INFERENCE_STEPS,
                "monitor_checkpoint_anchor_predictions": len(local.monitor["indices"])
                * len(MONITOR_CHECKPOINT_STEPS),
                "monitor_checkpoint_predict_action_chunk_calls": (
                    monitor_batches_per_checkpoint * len(MONITOR_CHECKPOINT_STEPS)
                ),
                "monitor_checkpoint_flow_denoising_model_steps": (
                    monitor_batches_per_checkpoint
                    * len(MONITOR_CHECKPOINT_STEPS)
                    * NUM_INFERENCE_STEPS
                ),
                "note": "wall time is hardware/video-I/O dependent; each shard records measured elapsed time",
            },
        },
        "monitor_cartesian": {
            "checkpoint_steps": list(MONITOR_CHECKPOINT_STEPS),
            "logical_indices": list(local.monitor["indices"]),
            "sample_count": len(local.monitor["indices"]),
            "prediction_seed": seeds[0],
            "selection_uses_cartesian_metrics": False,
            "selection_remains": "minimum recorded train_monitor_loss",
        },
        "sharding": {
            "num_shards": num_shards,
            "partition": "contiguous np.linspace boundaries over logical indices [0,7465)",
        },
        "integrity": {
            "rank0_logs": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
                for path in log_paths
            ],
            "final_checkpoint_files": final_ledger,
            "best_checkpoint_files": best_ledger,
            "final_and_best_full_hashes_verified": True,
            "saved_stats_equal_dataset_stats": True,
            "dataset_files": dataset_ledger,
        },
        "interpretation": {
            "has_held_out_validation": False,
            "has_test_set": False,
            "monitor_subset_source": "training_data",
            "generalization_claim": False,
            "real_robot_rollout": False,
        },
    }


def _ensure_contract_file(work_dir: Path, contract: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload = dict(_jsonable(contract))
    digest = _canonical_sha256(payload)
    record = {"contract_sha256": digest, **payload}
    path = work_dir / "contract.json"
    if path.exists():
        existing = _read_json(path)
        existing_digest = existing.pop("contract_sha256", None)
        _require(existing_digest == _canonical_sha256(existing), "stored contract self-hash mismatch")
        _require(existing_digest == digest and existing == payload, "work directory belongs to a different M0 contract")
    else:
        _atomic_write_json(path, record)
    return record, digest


def prepare_m0(
    run_dir: Path,
    *,
    work_dir: Path,
    prediction_seeds: Sequence[int],
    num_shards: int,
    batch_size: int,
) -> PreparedM0:
    """Strictly verify the completed run and create/reuse its immutable M0 contract."""

    try:
        # ``_build_contract`` hashes the final and selected-best model files,
        # including processor state tensors that the general F0 shell verifier
        # does not list in its model manifest.  Avoid hashing the 7.3 GB final
        # weights twice in the same prepare invocation.
        local = verify_local_run(run_dir, verify_file_hashes=False)
    except Exception as exc:
        raise M0EvaluationError(f"F0 completion verification failed: {exc}") from exc
    resolved_work = work_dir.expanduser().resolve()
    dataset_root = Path(str(local.config["dataset"]["root"])).expanduser().resolve()
    _require(
        resolved_work.is_relative_to(PROJECT_ROOT.resolve()),
        "M0 work_dir must remain under franka_project",
    )
    _require(not _resolved_inside(resolved_work, local.run_dir), "M0 work_dir must not modify the training run")
    _require(
        not _resolved_inside(resolved_work, dataset_root),
        "M0 work_dir must not modify the frozen dataset",
    )
    resolved_work.mkdir(parents=True, exist_ok=True)
    contract = _build_contract(
        local,
        prediction_seeds=prediction_seeds,
        num_shards=num_shards,
        batch_size=batch_size,
    )
    stored, digest = _ensure_contract_file(resolved_work, contract)
    return PreparedM0(local=local, contract=stored, contract_sha256=digest, work_dir=resolved_work)


def _shard_paths(work_dir: Path, shard_index: int) -> tuple[Path, Path]:
    stem = f"part-{shard_index:05d}"
    return (
        _safe_child(work_dir, "shards", f"{stem}.npz"),
        _safe_child(work_dir, "shards", f"{stem}.json"),
    )


def _expected_shard_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_indices: np.ndarray,
    prediction_seeds: Sequence[int],
) -> None:
    required = {"logical_indices", "episode_indices", "states", "targets", "predictions", "prediction_seeds"}
    _require(set(arrays) == required, f"shard array keys mismatch: {sorted(arrays)}")
    count = len(expected_indices)
    _require(np.array_equal(arrays["logical_indices"], expected_indices), "shard logical indices mismatch")
    _require(arrays["episode_indices"].shape == (count,), "shard episode shape mismatch")
    _require(arrays["states"].shape == (count, STATE_DIM), "shard state shape mismatch")
    _require(arrays["targets"].shape == (count, ACTION_CHUNK_SIZE, ACTION_DIM), "shard target shape mismatch")
    _require(
        arrays["predictions"].shape == (len(prediction_seeds), count, ACTION_CHUNK_SIZE, ACTION_DIM),
        "shard prediction shape mismatch",
    )
    _require(
        np.array_equal(
            arrays["prediction_seeds"], np.asarray(prediction_seeds, dtype=np.int64)
        ),
        "shard seeds mismatch",
    )
    _require(np.all(np.isfinite(arrays["states"])), "shard state contains NaN/Inf")
    _require(np.all(np.isfinite(arrays["targets"])), "shard target contains NaN/Inf")
    _require(np.issubdtype(arrays["episode_indices"].dtype, np.integer), "episode indices are not integers")


def _load_verified_shard(
    prepared: PreparedM0,
    shard_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path, manifest_path = _shard_paths(prepared.work_dir, shard_index)
    manifest = _read_json(manifest_path)
    unhashed_manifest = dict(manifest)
    manifest_sha = unhashed_manifest.pop("manifest_sha256", None)
    _require(
        isinstance(manifest_sha, str) and manifest_sha == _canonical_sha256(unhashed_manifest),
        f"shard manifest self-hash mismatch for {shard_index}",
    )
    expected_indices = shard_indices(
        DATASET_SIZE, int(prepared.contract["sharding"]["num_shards"]), shard_index
    )
    expected_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "contract_sha256": prepared.contract_sha256,
        "shard_index": shard_index,
        "num_shards": int(prepared.contract["sharding"]["num_shards"]),
        "logical_index_start": int(expected_indices[0]),
        "logical_index_end_exclusive": int(expected_indices[-1]) + 1,
        "anchor_count": len(expected_indices),
        "prediction_seeds": list(prepared.contract["inference"]["prediction_seeds"]),
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    _require(not mismatches, f"shard manifest mismatch for {shard_index}: {mismatches}")
    _require(npz_path.is_file(), f"missing shard NPZ: {npz_path}")
    _require(npz_path.stat().st_size == manifest.get("npz_size_bytes"), "shard NPZ size mismatch")
    _require(_sha256_file(npz_path) == manifest.get("npz_sha256"), "shard NPZ SHA256 mismatch")
    elapsed = manifest.get("evaluation_seconds")
    throughput = manifest.get("anchor_seed_predictions_per_second")
    _require(
        isinstance(elapsed, (int, float)) and math.isfinite(float(elapsed)) and float(elapsed) > 0.0,
        "invalid shard evaluation duration",
    )
    _require(
        isinstance(throughput, (int, float))
        and math.isfinite(float(throughput))
        and float(throughput) > 0.0,
        "invalid shard evaluation throughput",
    )
    strict_load = manifest.get("strict_load")
    processor_check = manifest.get("processor_check")
    _require(isinstance(strict_load, Mapping) and strict_load.get("strict") is True, "shard lacks strict load proof")
    _require(
        isinstance(processor_check, Mapping)
        and processor_check.get("action_roundtrip") is True,
        "shard lacks processor roundtrip proof",
    )
    try:
        with np.load(npz_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    except Exception as exc:
        raise M0EvaluationError(f"cannot load shard NPZ: {npz_path}") from exc
    _expected_shard_arrays(
        arrays,
        expected_indices=expected_indices,
        prediction_seeds=prepared.contract["inference"]["prediction_seeds"],
    )
    return arrays, manifest


def _publish_shard(
    prepared: PreparedM0,
    shard_index: int,
    arrays: Mapping[str, np.ndarray],
    *,
    strict_load: Mapping[str, Any],
    processor_check: Mapping[str, Any],
    evaluation_seconds: float,
) -> dict[str, Any]:
    expected_indices = shard_indices(
        DATASET_SIZE, int(prepared.contract["sharding"]["num_shards"]), shard_index
    )
    _expected_shard_arrays(
        arrays,
        expected_indices=expected_indices,
        prediction_seeds=prepared.contract["inference"]["prediction_seeds"],
    )
    npz_path, manifest_path = _shard_paths(prepared.work_dir, shard_index)
    _require(not npz_path.exists() and not manifest_path.exists(), "refusing to overwrite an existing shard")
    _atomic_write_npz(npz_path, arrays)
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": prepared.contract_sha256,
        "shard_index": shard_index,
        "num_shards": int(prepared.contract["sharding"]["num_shards"]),
        "logical_index_start": int(expected_indices[0]),
        "logical_index_end_exclusive": int(expected_indices[-1]) + 1,
        "anchor_count": len(expected_indices),
        "prediction_seeds": list(prepared.contract["inference"]["prediction_seeds"]),
        "evaluation_seconds": float(evaluation_seconds),
        "anchor_seed_predictions_per_second": (
            len(expected_indices)
            * len(prepared.contract["inference"]["prediction_seeds"])
            / float(evaluation_seconds)
        ),
        "npz_path": str(npz_path),
        "npz_size_bytes": npz_path.stat().st_size,
        "npz_sha256": _sha256_file(npz_path),
        "strict_load": _jsonable(strict_load),
        "processor_check": _jsonable(processor_check),
    }
    manifest = {"manifest_sha256": _canonical_sha256(manifest_payload), **manifest_payload}
    _atomic_write_json(manifest_path, manifest)
    _load_verified_shard(prepared, shard_index)
    return manifest


def _validate_device(device: str | torch.device) -> torch.device:
    result = torch.device(device)
    _require(result.type in {"cpu", "cuda"}, f"unsupported device: {result}")
    if result.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        _require(result.index is not None and result.index < torch.cuda.device_count(), "invalid CUDA device")
        torch.cuda.set_device(result)
        _require(torch.cuda.is_bf16_supported(), "selected CUDA device lacks BF16")
    return result


@contextmanager
def _offline_loading() -> Iterable[None]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _convert_images(batch: dict[str, Any]) -> None:
    for key in CAMERA_KEYS:
        image = batch.get(key)
        _require(isinstance(image, torch.Tensor) and image.dtype is torch.uint8, f"{key} must be uint8")
        converted = image.to(torch.float32).div_(255.0)
        _require(bool(torch.isfinite(converted).all()), f"{key} contains NaN/Inf")
        _require(float(converted.min()) >= 0.0 and float(converted.max()) <= 1.0, f"{key} outside [0,1]")
        batch[key] = converted


def fixed_noise_batch(prediction_seed: int, logical_indices: Sequence[int]) -> torch.Tensor:
    """Generate batch/traversal-order-independent padded PI0 noise."""

    noises: list[torch.Tensor] = []
    for logical_index in logical_indices:
        noise_seed = int(prediction_seed) + int(logical_index)
        _require(0 <= noise_seed < 2**63, "per-anchor prediction noise seed out of range")
        generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        noises.append(
            torch.randn(
                (ACTION_CHUNK_SIZE, MAX_ACTION_DIM),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
        )
    result = torch.stack(noises, dim=0)
    _require(bool(torch.isfinite(result).all()), "fixed prediction noise contains NaN/Inf")
    return result


def _logical_anchor(dataset: Any, logical_index: int) -> Mapping[str, Any]:
    anchor = dataset.logical_anchors[int(logical_index)]
    _require(isinstance(anchor, Mapping), "dataset logical anchor is not a mapping")
    _require(
        anchor.get("logical_index") == int(logical_index),
        f"dataset logical anchor identity drift at {logical_index}",
    )
    _require(
        type(anchor.get("episode_index")) is int
        and isinstance(anchor.get("raw_episode_id"), str)
        and bool(anchor.get("raw_episode_id")),
        f"dataset logical anchor episode identity invalid at {logical_index}",
    )
    return anchor


def _strict_load_summary(load_report: Mapping[str, Any]) -> dict[str, Any]:
    pretrained = load_report.get("pretrained")
    effective = load_report.get("effective_stats")
    _require(isinstance(pretrained, Mapping) and isinstance(effective, Mapping), "invalid strict load report")
    expected = {
        "strict": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "project_manifest_present": True,
        "weights_namespace": PI0_CORE_WEIGHTS_NAMESPACE,
    }
    mismatches = {
        key: {"expected": value, "actual": pretrained.get(key)}
        for key, value in expected.items()
        if pretrained.get(key) != value
    }
    _require(not mismatches, f"checkpoint strict load mismatch: {mismatches}")
    verified_tensor = pretrained.get("verified_tensor")
    _require(
        isinstance(verified_tensor, Mapping) and verified_tensor.get("exact_after_model_dtype_cast") is True,
        "strict load tensor verification failed",
    )
    stats_sha = effective.get("effective_stats_sha256")
    _require(isinstance(stats_sha, str) and len(stats_sha) == 64, "effective stats hash missing")
    return {
        **expected,
        "weights_path": str(pretrained.get("weights_path")),
        "weights_size_bytes": int(pretrained.get("weights_size_bytes", 0)),
        "verified_tensor_key": str(verified_tensor.get("checkpoint_key")),
        "verified_tensor_sha256": str(verified_tensor.get("checkpoint_tensor_sha256")),
        "effective_stats_sha256": stats_sha,
    }


def _check_processors(
    raw_batch: Mapping[str, Any],
    factory_preprocessor: Any,
    factory_postprocessor: Any,
    saved_preprocessor: Any,
    saved_postprocessor: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    factory_batch = copy.deepcopy(dict(raw_batch))
    saved_batch = copy.deepcopy(dict(raw_batch))
    raw_action = saved_batch["action"].detach().cpu().clone()
    _convert_images(factory_batch)
    _convert_images(saved_batch)
    factory_processed = factory_preprocessor(factory_batch)
    saved_processed = saved_preprocessor(saved_batch)
    for key in ("observation.state", "action"):
        first, second = factory_processed.get(key), saved_processed.get(key)
        _require(isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor), f"processor lacks {key}")
        _require(torch.equal(first, second), f"saved {key} processor differs from frozen effective stats")
        _require(bool(torch.isfinite(second).all()), f"saved processed {key} contains NaN/Inf")
    factory_restored = factory_postprocessor(factory_processed["action"].detach().clone())
    saved_restored = saved_postprocessor(saved_processed["action"].detach().clone())
    _require(
        isinstance(factory_restored, torch.Tensor)
        and isinstance(saved_restored, torch.Tensor),
        "postprocessor output type drift",
    )
    _require(torch.equal(factory_restored, saved_restored), "saved postprocessor differs from frozen effective stats")
    _require(
        torch.allclose(saved_restored.detach().cpu(), raw_action, atol=2e-5, rtol=2e-5),
        "saved processor action roundtrip failed",
    )
    report = {
        "saved_processors_match_factory_effective_stats": True,
        "action_roundtrip": True,
        "roundtrip_atol": 2e-5,
        "roundtrip_rtol": 2e-5,
    }
    return saved_processed, report


def _infer_logical_indices(
    *,
    policy: Any,
    factory_preprocessor: Any,
    factory_postprocessor: Any,
    saved_preprocessor: Any,
    saved_postprocessor: Any,
    dataset: Any,
    logical_indices: np.ndarray,
    prediction_seeds: Sequence[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one checkpoint over an explicit ordered set of teacher-forced anchors."""

    _require(
        logical_indices.ndim == 1
        and len(logical_indices) > 0
        and len(np.unique(logical_indices)) == len(logical_indices),
        "inference logical indices must be non-empty and unique",
    )
    seeds = tuple(int(seed) for seed in prediction_seeds)
    subset = torch.utils.data.Subset(dataset, logical_indices.tolist())
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    states_parts: list[np.ndarray] = []
    targets_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    predictions_parts: list[list[np.ndarray]] = [[] for _ in seeds]
    consumed = 0
    processor_report: dict[str, Any] | None = None
    for raw_batch in loader:
        current = logical_indices[
            consumed : consumed + int(raw_batch["action"].shape[0])
        ]
        consumed += len(current)
        raw_state = raw_batch["observation.state"].detach().cpu().to(torch.float32)
        raw_target = raw_batch["action"].detach().cpu().to(torch.float32)
        _require(
            tuple(raw_state.shape) == (len(current), STATE_DIM),
            "raw state shape drift",
        )
        _require(
            tuple(raw_target.shape) == (len(current), ACTION_CHUNK_SIZE, ACTION_DIM),
            "raw target shape drift",
        )
        _require(
            bool(torch.isfinite(raw_state).all())
            and bool(torch.isfinite(raw_target).all()),
            "raw sample non-finite",
        )

        if processor_report is None:
            processed, processor_report = _check_processors(
                raw_batch,
                factory_preprocessor,
                factory_postprocessor,
                saved_preprocessor,
                saved_postprocessor,
            )
        else:
            _convert_images(raw_batch)
            processed = saved_preprocessor(raw_batch)
        _require(
            isinstance(processed, dict),
            "saved preprocessor did not return a mapping",
        )

        for seed_ordinal, seed in enumerate(seeds):
            noise = fixed_noise_batch(seed, current.tolist()).to(device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                normalized = policy.predict_action_chunk(
                    processed, noise=noise, num_steps=NUM_INFERENCE_STEPS
                )
            _require(
                isinstance(normalized, torch.Tensor)
                and tuple(normalized.shape)
                == (len(current), ACTION_CHUNK_SIZE, ACTION_DIM),
                "normalized prediction shape drift",
            )
            raw_prediction = saved_postprocessor(normalized.detach().clone())
            _require(
                isinstance(raw_prediction, torch.Tensor)
                and tuple(raw_prediction.shape)
                == (len(current), ACTION_CHUNK_SIZE, ACTION_DIM),
                "raw prediction shape drift",
            )
            predictions_parts[seed_ordinal].append(
                raw_prediction.detach().cpu().to(torch.float32).numpy()
            )

        states_parts.append(raw_state.numpy())
        targets_parts.append(raw_target.numpy())
        episode_parts.append(
            np.asarray(
                [
                    int(_logical_anchor(dataset, int(index))["episode_index"])
                    for index in current
                ],
                dtype=np.int64,
            )
        )
    _require(
        consumed == len(logical_indices),
        "DataLoader did not consume the exact logical-index set",
    )
    _require(processor_report is not None, "processor check was not executed")
    arrays = {
        "logical_indices": logical_indices.astype(np.int64, copy=False),
        "episode_indices": np.concatenate(episode_parts).astype(np.int64, copy=False),
        "states": np.concatenate(states_parts).astype(np.float32, copy=False),
        "targets": np.concatenate(targets_parts).astype(np.float32, copy=False),
        "predictions": np.stack(
            [
                np.concatenate(parts).astype(np.float32, copy=False)
                for parts in predictions_parts
            ]
        ),
        "prediction_seeds": np.asarray(seeds, dtype=np.int64),
    }
    return arrays, processor_report


def _evaluate_missing_shards(
    prepared: PreparedM0,
    shard_indices_to_run: Sequence[int],
    *,
    device: str,
    num_workers: int,
) -> list[dict[str, Any]]:
    num_shards = int(prepared.contract["sharding"]["num_shards"])
    seeds = tuple(int(seed) for seed in prepared.contract["inference"]["prediction_seeds"])
    batch_size = int(prepared.contract["inference"]["batch_size"])
    requested = tuple(int(index) for index in shard_indices_to_run)
    _require(requested and len(requested) == len(set(requested)), "shards must be non-empty and unique")
    _require(all(0 <= index < num_shards for index in requested), "requested shard index out of range")

    manifests: list[dict[str, Any]] = []
    missing: list[int] = []
    for index in requested:
        npz_path, manifest_path = _shard_paths(prepared.work_dir, index)
        if npz_path.exists() and not manifest_path.exists():
            # A crash after the atomic NPZ rename but before publishing its
            # completion manifest leaves an unaudited orphan.  It is safe to
            # discard because this tool owns the work directory and a shard is
            # complete only when both files validate.
            npz_path.unlink()
        if npz_path.exists() or manifest_path.exists():
            _arrays, manifest = _load_verified_shard(prepared, index)
            manifests.append({**manifest, "resume_status": "SKIPPED_ALREADY_COMPLETE"})
        else:
            missing.append(index)
    if not missing:
        return manifests

    torch_device = _validate_device(device)
    dataset_cfg = prepared.local.config["dataset"]
    dataset = CartesianAnchorDataset(
        dataset_cfg["root"],
        profile=dataset_cfg["profile"],
        episode_indices=dataset_cfg.get("episode_indices"),
        max_anchors_per_episode=dataset_cfg.get("max_anchors_per_episode"),
        video_backend="pyav",
    )
    _require(len(dataset) == DATASET_SIZE, "reloaded F0 dataset size mismatch")
    _require(dataset.effective_stats is not None, "reloaded dataset lacks effective stats")

    policy = factory_preprocessor = factory_postprocessor = saved_preprocessor = saved_postprocessor = None
    try:
        with _offline_loading():
            policy, factory_preprocessor, factory_postprocessor, load_report = (
                load_pi0_full_policy_and_processors(prepared.local.model_dir, dataset.effective_stats, torch_device)
            )
        strict_load = _strict_load_summary(load_report)
        _require(
            Path(strict_load["weights_path"]).resolve()
            == (prepared.local.model_dir / "model.safetensors").resolve(),
            "strict loader used the wrong final-checkpoint weights",
        )
        saved_preprocessor, saved_postprocessor = load_saved_processors(prepared.local.model_dir, torch_device)
        policy.eval()

        for shard_index in missing:
            shard_started = time.monotonic()
            indices = shard_indices(DATASET_SIZE, num_shards, shard_index)
            arrays, processor_report = _infer_logical_indices(
                policy=policy,
                factory_preprocessor=factory_preprocessor,
                factory_postprocessor=factory_postprocessor,
                saved_preprocessor=saved_preprocessor,
                saved_postprocessor=saved_postprocessor,
                dataset=dataset,
                logical_indices=indices,
                prediction_seeds=seeds,
                batch_size=batch_size,
                num_workers=num_workers,
                device=torch_device,
            )
            manifest = _publish_shard(
                prepared,
                shard_index,
                arrays,
                strict_load=strict_load,
                processor_check=processor_report,
                evaluation_seconds=max(time.monotonic() - shard_started, 1e-9),
            )
            manifests.append({**manifest, "resume_status": "COMPUTED"})
    finally:
        policy = factory_preprocessor = factory_postprocessor = saved_preprocessor = saved_postprocessor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return manifests


def _monitor_checkpoint_paths(work_dir: Path, step: int) -> tuple[Path, Path]:
    stem = f"step-{step:06d}"
    return (
        _safe_child(work_dir, "monitor_checkpoints", f"{stem}.npz"),
        _safe_child(work_dir, "monitor_checkpoints", f"{stem}.json"),
    )


def _load_verified_monitor_checkpoint(
    prepared: PreparedM0,
    step: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path, manifest_path = _monitor_checkpoint_paths(prepared.work_dir, step)
    manifest = _read_json(manifest_path)
    unhashed = dict(manifest)
    manifest_sha = unhashed.pop("manifest_sha256", None)
    _require(
        isinstance(manifest_sha, str) and manifest_sha == _canonical_sha256(unhashed),
        f"monitor checkpoint manifest self-hash mismatch at step {step}",
    )
    indices = np.asarray(
        prepared.contract["monitor_cartesian"]["logical_indices"], dtype=np.int64
    )
    prediction_seed = int(prepared.contract["monitor_cartesian"]["prediction_seed"])
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "contract_sha256": prepared.contract_sha256,
        "checkpoint_step": step,
        "anchor_count": len(indices),
        "logical_indices_sha256": _canonical_sha256(indices.tolist()),
        "prediction_seed": prediction_seed,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    _require(
        not mismatches,
        f"monitor checkpoint manifest mismatch at step {step}: {mismatches}",
    )
    _require(npz_path.is_file(), f"missing monitor checkpoint NPZ: {npz_path}")
    _require(
        npz_path.stat().st_size == manifest.get("npz_size_bytes"),
        f"monitor checkpoint NPZ size mismatch at step {step}",
    )
    _require(
        _sha256_file(npz_path) == manifest.get("npz_sha256"),
        f"monitor checkpoint NPZ hash mismatch at step {step}",
    )
    try:
        with np.load(npz_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    except Exception as exc:
        raise M0EvaluationError(
            f"cannot load monitor checkpoint NPZ at step {step}"
        ) from exc
    _expected_shard_arrays(
        arrays,
        expected_indices=indices,
        prediction_seeds=[prediction_seed],
    )
    _require(
        isinstance(manifest.get("strict_load"), Mapping)
        and manifest["strict_load"].get("strict") is True,
        f"monitor checkpoint lacks strict load proof at step {step}",
    )
    _require(
        isinstance(manifest.get("processor_check"), Mapping)
        and manifest["processor_check"].get("action_roundtrip") is True,
        f"monitor checkpoint lacks processor proof at step {step}",
    )
    ledger_sha = manifest.get("checkpoint_file_ledger_sha256")
    checkpoint_ledger = manifest.get("checkpoint_file_ledger")
    _require(
        isinstance(ledger_sha, str) and len(ledger_sha) == 64,
        f"monitor checkpoint lacks file-ledger hash at step {step}",
    )
    _require(
        isinstance(checkpoint_ledger, Mapping)
        and _canonical_sha256(checkpoint_ledger) == ledger_sha,
        f"monitor checkpoint file ledger/hash mismatch at step {step}",
    )
    return arrays, manifest


def _publish_monitor_checkpoint(
    prepared: PreparedM0,
    step: int,
    arrays: Mapping[str, np.ndarray],
    *,
    strict_load: Mapping[str, Any],
    processor_check: Mapping[str, Any],
    checkpoint_file_ledger: Mapping[str, Any],
    evaluation_seconds: float,
) -> dict[str, Any]:
    indices = np.asarray(
        prepared.contract["monitor_cartesian"]["logical_indices"], dtype=np.int64
    )
    prediction_seed = int(prepared.contract["monitor_cartesian"]["prediction_seed"])
    _expected_shard_arrays(
        arrays,
        expected_indices=indices,
        prediction_seeds=[prediction_seed],
    )
    npz_path, manifest_path = _monitor_checkpoint_paths(prepared.work_dir, step)
    _require(
        not npz_path.exists() and not manifest_path.exists(),
        f"refusing to overwrite monitor checkpoint step {step}",
    )
    _atomic_write_npz(npz_path, arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": prepared.contract_sha256,
        "checkpoint_step": step,
        "anchor_count": len(indices),
        "logical_indices_sha256": _canonical_sha256(indices.tolist()),
        "prediction_seed": prediction_seed,
        "evaluation_seconds": float(evaluation_seconds),
        "npz_path": str(npz_path),
        "npz_size_bytes": npz_path.stat().st_size,
        "npz_sha256": _sha256_file(npz_path),
        "strict_load": _jsonable(strict_load),
        "processor_check": _jsonable(processor_check),
        "checkpoint_file_ledger_sha256": _canonical_sha256(
            checkpoint_file_ledger
        ),
        "checkpoint_file_ledger": _jsonable(checkpoint_file_ledger),
    }
    manifest = {"manifest_sha256": _canonical_sha256(payload), **payload}
    _atomic_write_json(manifest_path, manifest)
    _load_verified_monitor_checkpoint(prepared, step)
    return manifest


def _evaluate_monitor_checkpoints(
    prepared: PreparedM0,
    checkpoint_steps: Sequence[int],
    *,
    device: str,
    num_workers: int,
) -> list[dict[str, Any]]:
    """Evaluate each epoch milestone on the identical 747-anchor monitor set."""

    allowed = tuple(int(step) for step in prepared.contract["monitor_cartesian"]["checkpoint_steps"])
    requested = tuple(int(step) for step in checkpoint_steps)
    _require(
        requested
        and len(requested) == len(set(requested))
        and set(requested).issubset(allowed),
        "monitor checkpoint steps must be a non-empty unique subset of epoch milestones",
    )
    manifests: list[dict[str, Any]] = []
    missing: list[int] = []
    for step in requested:
        npz_path, manifest_path = _monitor_checkpoint_paths(prepared.work_dir, step)
        if npz_path.exists() and not manifest_path.exists():
            npz_path.unlink()
        if npz_path.exists() or manifest_path.exists():
            _arrays, manifest = _load_verified_monitor_checkpoint(prepared, step)
            manifests.append({**manifest, "resume_status": "SKIPPED_ALREADY_COMPLETE"})
        else:
            missing.append(step)
    if not missing:
        return manifests

    torch_device = _validate_device(device)
    dataset_cfg = prepared.local.config["dataset"]
    dataset = CartesianAnchorDataset(
        dataset_cfg["root"],
        profile=dataset_cfg["profile"],
        episode_indices=dataset_cfg.get("episode_indices"),
        max_anchors_per_episode=dataset_cfg.get("max_anchors_per_episode"),
        video_backend="pyav",
    )
    _require(len(dataset) == DATASET_SIZE, "monitor dataset size mismatch")
    _require(dataset.effective_stats is not None, "monitor dataset lacks effective stats")
    indices = np.asarray(
        prepared.contract["monitor_cartesian"]["logical_indices"], dtype=np.int64
    )
    for ordinal, logical_index in enumerate(indices):
        _require(
            dict(_logical_anchor(dataset, int(logical_index)))
            == prepared.local.monitor["anchors"][ordinal],
            f"monitor anchor identity mismatch at logical index {logical_index}",
        )
    seed = int(prepared.contract["monitor_cartesian"]["prediction_seed"])
    batch_size = int(prepared.contract["inference"]["batch_size"])

    for step in missing:
        checkpoint_started = time.monotonic()
        model_dir = prepared.local.run_dir / f"checkpoints/step-{step:06d}/pretrained_model"
        checkpoint_ledger = _verify_model_files_and_hashes(model_dir)
        policy = factory_preprocessor = factory_postprocessor = None
        saved_preprocessor = saved_postprocessor = None
        try:
            with _offline_loading():
                policy, factory_preprocessor, factory_postprocessor, load_report = (
                    load_pi0_full_policy_and_processors(
                        model_dir, dataset.effective_stats, torch_device
                    )
                )
            strict_load = _strict_load_summary(load_report)
            _require(
                Path(strict_load["weights_path"]).resolve()
                == (model_dir / "model.safetensors").resolve(),
                f"strict loader used the wrong milestone weights at step {step}",
            )
            saved_preprocessor, saved_postprocessor = load_saved_processors(
                model_dir, torch_device
            )
            policy.eval()
            arrays, processor_report = _infer_logical_indices(
                policy=policy,
                factory_preprocessor=factory_preprocessor,
                factory_postprocessor=factory_postprocessor,
                saved_preprocessor=saved_preprocessor,
                saved_postprocessor=saved_postprocessor,
                dataset=dataset,
                logical_indices=indices,
                prediction_seeds=[seed],
                batch_size=batch_size,
                num_workers=num_workers,
                device=torch_device,
            )
            manifest = _publish_monitor_checkpoint(
                prepared,
                step,
                arrays,
                strict_load=strict_load,
                processor_check=processor_report,
                checkpoint_file_ledger=checkpoint_ledger,
                evaluation_seconds=max(time.monotonic() - checkpoint_started, 1e-9),
            )
            manifests.append({**manifest, "resume_status": "COMPUTED"})
        finally:
            policy = factory_preprocessor = factory_postprocessor = None
            saved_preprocessor = saved_postprocessor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return manifests


def _nullable_mean(values: np.ndarray) -> float | None:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    return float(np.mean(values[finite], dtype=np.float64))


def _rotation_errors_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    shape = prediction.shape[:-1]
    result = np.full(shape, np.nan, dtype=np.float64)
    finite = np.all(np.isfinite(prediction), axis=-1) & np.all(np.isfinite(target), axis=-1)
    if np.any(finite):
        first = so3_exp(prediction[finite].astype(np.float64, copy=False))
        second = so3_exp(target[finite].astype(np.float64, copy=False))
        result[finite] = np.rad2deg(rotation_geodesic_angle(first, second))
    return result


def _adjacent_rotation_deg(rotation_vectors: np.ndarray) -> np.ndarray:
    shape = rotation_vectors.shape[:-2] + (rotation_vectors.shape[-2] - 1,)
    result = np.full(shape, np.nan, dtype=np.float64)
    first_vec = rotation_vectors[..., :-1, :]
    second_vec = rotation_vectors[..., 1:, :]
    finite = np.all(np.isfinite(first_vec), axis=-1) & np.all(np.isfinite(second_vec), axis=-1)
    if np.any(finite):
        first = so3_exp(first_vec[finite].astype(np.float64, copy=False))
        second = so3_exp(second_vec[finite].astype(np.float64, copy=False))
        result[finite] = np.rad2deg(rotation_geodesic_angle(first, second))
    return result


def _error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    translation_valid = np.all(np.isfinite(prediction[..., :3]), axis=-1)
    translation = np.full(prediction.shape[:-1], np.nan, dtype=np.float64)
    translation[translation_valid] = (
        np.linalg.norm(
            prediction[..., :3][translation_valid].astype(np.float64)
            - target[..., :3][translation_valid].astype(np.float64),
            axis=-1,
        )
        * 1000.0
    )
    rotation = _rotation_errors_deg(prediction[..., 3:6], target[..., 3:6])
    gripper = np.abs(
        prediction[..., 6].astype(np.float64, copy=False) - target[..., 6].astype(np.float64, copy=False)
    )
    gripper[~np.isfinite(prediction[..., 6])] = np.nan
    return {
        "translation_ade_mm": _nullable_mean(translation),
        "translation_fde_mm": _nullable_mean(translation[..., -1]),
        "translation_horizon_mm": {
            str(horizon): _nullable_mean(translation[..., horizon - 1]) for horizon in HORIZONS
        },
        "rotation_geodesic_ade_deg": _nullable_mean(rotation),
        "rotation_geodesic_fde_deg": _nullable_mean(rotation[..., -1]),
        "rotation_geodesic_horizon_deg": {
            str(horizon): _nullable_mean(rotation[..., horizon - 1]) for horizon in HORIZONS
        },
        "gripper_mae": _nullable_mean(gripper),
        "gripper_fde_mae": _nullable_mean(gripper[..., -1]),
        "gripper_horizon_mae": {
            str(horizon): _nullable_mean(gripper[..., horizon - 1]) for horizon in HORIZONS
        },
    }


def _fraction(mask: np.ndarray, denominator_mask: np.ndarray | None = None) -> float | None:
    denominator = np.ones(mask.shape, dtype=bool) if denominator_mask is None else denominator_mask
    count = int(np.count_nonzero(denominator))
    return None if count == 0 else float(np.count_nonzero(mask & denominator) / count)


def _prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    states: np.ndarray,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    _require(prediction.shape == target.shape and prediction.ndim == 3, "metric action shape mismatch")
    _require(prediction.shape[1:] == (ACTION_CHUNK_SIZE, ACTION_DIM), "metric action contract mismatch")
    _require(states.shape == (len(prediction), STATE_DIM), "metric state shape mismatch")
    _require(np.all(np.isfinite(target)) and np.all(np.isfinite(states)), "target/state must be finite")

    action_stats = stats.get("action")
    state_stats = stats.get("observation.state")
    _require(isinstance(action_stats, Mapping) and isinstance(state_stats, Mapping), "stats features missing")
    limits = {
        name: np.asarray(action_stats[name], dtype=np.float64)
        for name in ("min", "max", "q01", "q99")
    }
    _require(all(value.shape == (ACTION_DIM,) for value in limits.values()), "action stats shape drift")
    workspace_min = np.asarray(state_stats["min"], dtype=np.float64)[:3]
    workspace_max = np.asarray(state_stats["max"], dtype=np.float64)[:3]

    finite_element = np.isfinite(prediction)
    finite_waypoint = np.all(finite_element, axis=-1)
    q_violation_elements = finite_element & (
        (prediction < limits["q01"][None, None, :])
        | (prediction > limits["q99"][None, None, :])
    )
    envelope_violation_elements = finite_element & (
        (prediction < limits["min"][None, None, :])
        | (prediction > limits["max"][None, None, :])
    )
    q_waypoint = np.any(q_violation_elements, axis=-1)
    envelope_waypoint = np.any(envelope_violation_elements, axis=-1)
    rotvec_norm = np.linalg.norm(prediction[..., 3:6].astype(np.float64), axis=-1)
    rotvec_finite = np.all(np.isfinite(prediction[..., 3:6]), axis=-1)
    gripper_finite = np.isfinite(prediction[..., 6])
    gripper_oob = gripper_finite & ((prediction[..., 6] < 0.0) | (prediction[..., 6] > 1.0))

    decoded_xyz = states[:, None, :3].astype(np.float64) + prediction[..., :3].astype(np.float64)
    decoded_finite = np.all(np.isfinite(decoded_xyz), axis=-1)
    workspace_oob = decoded_finite & np.any(
        (decoded_xyz < workspace_min[None, None, :])
        | (decoded_xyz > workspace_max[None, None, :]),
        axis=-1,
    )

    delta_xyz = np.diff(prediction[..., :3].astype(np.float64), axis=1)
    delta_xyz_valid = np.all(np.isfinite(delta_xyz), axis=-1)
    translation_step = np.linalg.norm(delta_xyz, axis=-1) * 1000.0
    translation_step[~delta_xyz_valid] = np.nan
    second_xyz = np.diff(prediction[..., :3].astype(np.float64), n=2, axis=1)
    second_valid = np.all(np.isfinite(second_xyz), axis=-1)
    translation_second = np.linalg.norm(second_xyz, axis=-1) * 1000.0
    translation_second[~second_valid] = np.nan
    rotation_step = _adjacent_rotation_deg(prediction[..., 3:6])
    gripper_step = np.abs(np.diff(prediction[..., 6].astype(np.float64), axis=1))
    gripper_step[~np.isfinite(gripper_step)] = np.nan

    return {
        "counts": {
            "anchors": int(len(prediction)),
            "waypoints": int(prediction.shape[0] * prediction.shape[1]),
            "action_elements": int(prediction.size),
        },
        "error": _error_metrics(prediction, target),
        "nonfinite": {
            "action_element_fraction": float(1.0 - np.mean(finite_element)),
            "waypoint_any_fraction": float(1.0 - np.mean(finite_waypoint)),
            "action_element_count": int(np.size(finite_element) - np.count_nonzero(finite_element)),
            "waypoint_any_count": int(np.size(finite_waypoint) - np.count_nonzero(finite_waypoint)),
        },
        "action_envelope": {
            "outside_train_q01_q99_element_fraction_of_finite": _fraction(q_violation_elements, finite_element),
            "outside_train_q01_q99_waypoint_any_fraction": _fraction(q_waypoint | ~finite_waypoint),
            "outside_train_min_max_element_fraction_of_finite": _fraction(envelope_violation_elements, finite_element),
            "outside_train_min_max_waypoint_any_fraction": _fraction(envelope_waypoint | ~finite_waypoint),
            "rotvec_norm_gt_pi_waypoint_fraction": _fraction((rotvec_norm > math.pi) | ~rotvec_finite),
            "gripper_outside_0_1_waypoint_fraction": _fraction(gripper_oob | ~gripper_finite),
        },
        "workspace_training_envelope": {
            "decoded_xyz_outside_train_state_min_max_fraction": _fraction(workspace_oob | ~decoded_finite),
        },
        "smoothness": {
            "adjacent_translation_step_mean_mm": _nullable_mean(translation_step),
            "translation_second_difference_mean_mm": _nullable_mean(translation_second),
            "adjacent_rotation_geodesic_mean_deg": _nullable_mean(rotation_step),
            "adjacent_gripper_step_mean": _nullable_mean(gripper_step),
        },
    }


def _tree_mean_variance(trees: Sequence[Any]) -> tuple[Any, Any]:
    _require(trees, "cannot aggregate an empty tree list")
    first = trees[0]
    if isinstance(first, Mapping):
        _require(
            all(
                isinstance(tree, Mapping) and set(tree) == set(first)
                for tree in trees
            ),
            "tree structure mismatch",
        )
        means: dict[str, Any] = {}
        variances: dict[str, Any] = {}
        for key in first:
            mean, variance = _tree_mean_variance([tree[key] for tree in trees])
            means[str(key)] = mean
            variances[str(key)] = variance
        return means, variances
    if first is None or isinstance(first, (int, float, np.number)):
        values = [float(value) for value in trees if value is not None and math.isfinite(float(value))]
        if not values:
            return None, None
        array = np.asarray(values, dtype=np.float64)
        return float(np.mean(array)), float(np.var(array, ddof=0))
    _require(all(tree == first for tree in trees), "non-numeric tree leaf mismatch")
    return first, None


def compute_reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    states: np.ndarray,
    episode_indices: np.ndarray,
    raw_episode_ids: Mapping[int, str],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute micro, episode-macro, and every-episode reconstruction metrics."""

    _require(episode_indices.shape == (len(prediction),), "episode index shape mismatch")
    unique_episodes = sorted(int(value) for value in np.unique(episode_indices))
    _require(unique_episodes == sorted(raw_episode_ids), "episode identity mapping mismatch")
    per_episode: list[dict[str, Any]] = []
    episode_metric_trees: list[dict[str, Any]] = []
    for episode_index in unique_episodes:
        selected = episode_indices == episode_index
        metrics = _prediction_metrics(prediction[selected], target[selected], states[selected], stats)
        episode_metric_trees.append(metrics)
        per_episode.append(
            {
                "episode_index": episode_index,
                "raw_episode_id": raw_episode_ids[episode_index],
                "metrics": metrics,
            }
        )
    macro_mean, _unused = _tree_mean_variance(episode_metric_trees)
    # Counts are totals, not means; keep them out of the episode-macro metric tree.
    macro_mean.pop("counts", None)
    return {
        "aggregation": {
            "primary": "episode_macro",
            "episode_macro_definition": "unweighted mean of the 22 per-episode metrics",
            "overall_anchor_micro_definition": "all anchor-waypoints pooled; longer episodes receive more weight",
        },
        "episode_macro": macro_mean,
        "overall_anchor_micro": _prediction_metrics(prediction, target, states, stats),
        "per_episode": per_episode,
    }


def summarize_prediction_seeds(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(per_seed, "seed results are empty")
    macro_mean, macro_variance = _tree_mean_variance([item["metrics"]["episode_macro"] for item in per_seed])
    overall_mean, overall_variance = _tree_mean_variance([item["metrics"]["overall_anchor_micro"] for item in per_seed])
    episode_ids = [item["episode_index"] for item in per_seed[0]["metrics"]["per_episode"]]
    per_episode: list[dict[str, Any]] = []
    for ordinal, episode_index in enumerate(episode_ids):
        records = [item["metrics"]["per_episode"][ordinal] for item in per_seed]
        _require(all(record["episode_index"] == episode_index for record in records), "seed episode order drift")
        mean, variance = _tree_mean_variance([record["metrics"] for record in records])
        per_episode.append(
            {
                "episode_index": episode_index,
                "raw_episode_id": records[0]["raw_episode_id"],
                "mean": mean,
                "population_variance": variance,
            }
        )
    return {
        "variance_definition": "population variance (ddof=0) across fixed PI0 prediction-noise seeds",
        "episode_macro": {"mean": macro_mean, "population_variance": macro_variance},
        "overall_anchor_micro": {"mean": overall_mean, "population_variance": overall_variance},
        "per_episode": per_episode,
    }


def _combine_shards(prepared: PreparedM0) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    num_shards = int(prepared.contract["sharding"]["num_shards"])
    arrays_by_key: dict[str, list[np.ndarray]] = {
        "logical_indices": [],
        "episode_indices": [],
        "states": [],
        "targets": [],
        "predictions": [],
    }
    manifests: list[dict[str, Any]] = []
    for shard_index in range(num_shards):
        arrays, manifest = _load_verified_shard(prepared, shard_index)
        manifests.append(manifest)
        for key in arrays_by_key:
            arrays_by_key[key].append(arrays[key])
    combined = {
        "logical_indices": np.concatenate(arrays_by_key["logical_indices"], axis=0),
        "episode_indices": np.concatenate(arrays_by_key["episode_indices"], axis=0),
        "states": np.concatenate(arrays_by_key["states"], axis=0),
        "targets": np.concatenate(arrays_by_key["targets"], axis=0),
        "predictions": np.concatenate(arrays_by_key["predictions"], axis=1),
        "prediction_seeds": np.asarray(prepared.contract["inference"]["prediction_seeds"], dtype=np.int64),
    }
    _expected_shard_arrays(
        combined,
        expected_indices=np.arange(DATASET_SIZE, dtype=np.int64),
        prediction_seeds=prepared.contract["inference"]["prediction_seeds"],
    )
    return combined, manifests


def _build_monitor_checkpoint_comparison(
    prepared: PreparedM0,
    *,
    full_arrays: Mapping[str, np.ndarray],
    stats: Mapping[str, Any],
    raw_episode_ids: Mapping[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all milestone predictions and compute their identical monitor metrics."""

    records_by_step = {
        int(record["step"]): record
        for record in prepared.contract["checkpoint_selection"]["records"]
    }
    results: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    reference: dict[str, np.ndarray] | None = None
    for step in prepared.contract["monitor_cartesian"]["checkpoint_steps"]:
        arrays, manifest = _load_verified_monitor_checkpoint(prepared, int(step))
        manifests.append(manifest)
        if reference is None:
            reference = arrays
            full_logical_indices = full_arrays["logical_indices"]
            position_by_logical = {
                int(logical_index): ordinal
                for ordinal, logical_index in enumerate(full_logical_indices)
            }
            _require(
                all(int(index) in position_by_logical for index in arrays["logical_indices"]),
                "monitor anchor is absent from full reconstruction arrays",
            )
            full_positions = np.asarray(
                [position_by_logical[int(index)] for index in arrays["logical_indices"]],
                dtype=np.int64,
            )
            for key in ("logical_indices", "episode_indices", "states", "targets"):
                _require(
                    np.array_equal(arrays[key], full_arrays[key][full_positions]),
                    f"monitor {key} differs from full teacher-forced reconstruction inputs",
                )
        else:
            for key in ("logical_indices", "episode_indices", "states", "targets"):
                _require(
                    np.array_equal(reference[key], arrays[key]),
                    f"monitor teacher-forced inputs drifted at checkpoint step {step}",
                )
        episode_ids = sorted(int(value) for value in np.unique(arrays["episode_indices"]))
        subset_episode_ids = {index: raw_episode_ids[index] for index in episode_ids}
        metrics = compute_reconstruction_metrics(
            arrays["predictions"][0],
            arrays["targets"],
            arrays["states"],
            arrays["episode_indices"],
            subset_episode_ids,
            stats,
        )
        selection_record = records_by_step.get(int(step))
        _require(selection_record is not None, f"monitor loss missing for checkpoint {step}")
        results.append(
            {
                **selection_record,
                "prediction_seed": int(arrays["prediction_seeds"][0]),
                "teacher_forced_cartesian_metrics": metrics,
                "artifact_npz_sha256": manifest["npz_sha256"],
                "checkpoint_file_ledger_sha256": manifest[
                    "checkpoint_file_ledger_sha256"
                ],
            }
        )
    _require(
        [int(result["step"]) for result in results]
        == list(prepared.contract["monitor_cartesian"]["checkpoint_steps"]),
        "monitor checkpoint comparison order/coverage mismatch",
    )
    return results, manifests


def _verify_wandb_gate(local: VerifiedLocalRun) -> dict[str, Any]:
    """Require the completed metrics-only W&B contract without embedding full history."""

    try:
        report = verify_wandb(local)
    except Exception as exc:
        raise M0EvaluationError(f"W&B completion verification failed: {exc}") from exc
    history = report.get("history")
    _require(isinstance(history, Mapping), "W&B verifier returned no metric history")
    monitor_history = history.get("monitor/loss")
    _require(
        isinstance(monitor_history, list) and monitor_history,
        "W&B verifier returned no monitor/loss history",
    )
    return {
        key: value for key, value in report.items() if key != "history"
    } | {
        "history_sha256": _canonical_sha256(history),
        "metric_point_counts": {
            str(metric): len(rows) for metric, rows in history.items()
        },
        "monitor_history": monitor_history,
    }


def finalize_report(prepared: PreparedM0, *, output_json: Path) -> dict[str, Any]:
    """Verify and merge all shards, then atomically publish the final M0 JSON."""

    output_json = output_json.expanduser().resolve()
    dataset_root = Path(
        str(prepared.local.config["dataset"]["root"])
    ).expanduser().resolve()
    _require(
        output_json.is_relative_to(PROJECT_ROOT.resolve()),
        "M0 report must remain under franka_project",
    )
    _require(not _resolved_inside(output_json, prepared.local.run_dir), "M0 report must not modify training run")
    _require(
        not _resolved_inside(output_json, dataset_root),
        "M0 report must not modify the frozen dataset",
    )
    arrays, shard_manifests = _combine_shards(prepared)

    dataset_cfg = prepared.local.config["dataset"]
    dataset = CartesianAnchorDataset(
        dataset_cfg["root"],
        profile=dataset_cfg["profile"],
        episode_indices=dataset_cfg.get("episode_indices"),
        max_anchors_per_episode=dataset_cfg.get("max_anchors_per_episode"),
        video_backend="pyav",
    )
    _require(len(dataset) == DATASET_SIZE, "finalize dataset size drift")
    expected_episode_indices = np.asarray(
        [
            int(_logical_anchor(dataset, index)["episode_index"])
            for index in range(len(dataset))
        ],
        dtype=np.int64,
    )
    _require(np.array_equal(arrays["episode_indices"], expected_episode_indices), "shard episode identities drifted")
    raw_episode_ids: dict[int, str] = {}
    for index in range(len(dataset)):
        anchor = _logical_anchor(dataset, index)
        episode_index = int(anchor["episode_index"])
        raw_episode_id = str(anchor["raw_episode_id"])
        existing = raw_episode_ids.setdefault(episode_index, raw_episode_id)
        _require(existing == raw_episode_id, "episode has conflicting raw ids")
    _require(len(raw_episode_ids) == 22, "final M0 report must contain all 22 episodes")

    stats_path = Path(dataset_cfg["root"]) / "meta/pi0_eef_stats.json"
    _require(_sha256_file(stats_path) == prepared.contract["dataset"]["stats_sha256"], "stats changed after contract")
    stats = _read_json(stats_path)
    per_seed: list[dict[str, Any]] = []
    for ordinal, seed in enumerate(arrays["prediction_seeds"].tolist()):
        metrics = compute_reconstruction_metrics(
            arrays["predictions"][ordinal],
            arrays["targets"],
            arrays["states"],
            arrays["episode_indices"],
            raw_episode_ids,
            stats,
        )
        per_seed.append({"prediction_seed": int(seed), "metrics": metrics})
    seed_summary = summarize_prediction_seeds(per_seed)
    target_reference = compute_reconstruction_metrics(
        arrays["targets"],
        arrays["targets"],
        arrays["states"],
        arrays["episode_indices"],
        raw_episode_ids,
        stats,
    )
    checkpoint_comparison, monitor_manifests = _build_monitor_checkpoint_comparison(
        prepared,
        full_arrays=arrays,
        stats=stats,
        raw_episode_ids=raw_episode_ids,
    )
    wandb_proof = _verify_wandb_gate(prepared.local)
    wandb_monitor = {
        int(item["step"]): float(item["value"])
        for item in wandb_proof["monitor_history"]
    }
    logged_monitor = {
        int(item["step"]): float(item["train_monitor_loss"])
        for item in prepared.contract["checkpoint_selection"]["records"]
    }
    _require(
        set(wandb_monitor) == set(logged_monitor)
        and all(
            math.isclose(
                wandb_monitor[step], value, rel_tol=0.0, abs_tol=5e-8
            )
            for step, value in logged_monitor.items()
        ),
        "local log and W&B monitor histories disagree",
    )
    total_shard_gpu_seconds = float(
        sum(float(manifest["evaluation_seconds"]) for manifest in shard_manifests)
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "M0_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "teacher_forced_in_sample_offline_trajectory_reconstruction",
        "has_held_out_validation": False,
        "has_test_set": False,
        "monitor_subset_source": "training_data",
        "generalization_claim": False,
        "quality_threshold_applied": False,
        "real_robot_rollout_performed": False,
        "run": prepared.contract["run"],
        "dataset": prepared.contract["dataset"],
        "inference": prepared.contract["inference"],
        "checkpoint_selection": prepared.contract["checkpoint_selection"],
        "monitor_checkpoint_cartesian_comparison": {
            "selection_uses_cartesian_metrics": False,
            "selection_remains": "minimum recorded train_monitor_loss",
            "fixed_prediction_seed": prepared.contract["monitor_cartesian"][
                "prediction_seed"
            ],
            "checkpoints": checkpoint_comparison,
        },
        "reconstruction": {
            "checkpoint_step": FINAL_F0_STEP,
            "checkpoint_is_best_train_monitor": (
                prepared.contract["checkpoint_selection"]["best_train_monitor_step"] == FINAL_F0_STEP
            ),
            "teacher_forcing": "each anchor uses recorded images and recorded current EEF+gripper",
            "prediction_seed_results": per_seed,
            "prediction_seed_summary": seed_summary,
            "target_reference": target_reference,
        },
        "metric_definitions": {
            "translation": "L2 error between predicted and measured future base-frame delta xyz; mm",
            "rotation": "SO(3) geodesic between Exp(predicted body rotvec) and Exp(target body rotvec); degrees",
            "ADE_FDE": "ADE pools the 50 waypoints; FDE uses waypoint 50; horizon k uses waypoint k",
            "gripper": "absolute error in un-clipped gripper_0_1 prediction",
            "nonfinite": (
                "reported explicitly; invalid values are excluded only from the affected "
                "mean and counted unsafe"
            ),
            "action_quantile_envelope": (
                "component-wise q01/q99 from frozen full training action statistics; "
                "diagnostic, not a robot safety limit"
            ),
            "action_min_max_envelope": "component-wise observed training min/max; diagnostic, not a robot safety limit",
            "workspace_training_envelope": (
                "decoded anchor_xyz + predicted delta_xyz outside observed training state "
                "xyz min/max; not Franka physical workspace"
            ),
            "smoothness": (
                "adjacent/second differences within each predicted 50-waypoint chunk; "
                "no cross-anchor stitching"
            ),
            "aggregate": "episode_macro is primary; overall_anchor_micro is also reported for audit",
        },
        "audit": {
            "contract_sha256": prepared.contract_sha256,
            "contract_path": str(prepared.work_dir / "contract.json"),
            "shard_count": len(shard_manifests),
            "summed_shard_evaluation_seconds": total_shard_gpu_seconds,
            "aggregate_anchor_seed_predictions_per_gpu_second": (
                DATASET_SIZE
                * len(prepared.contract["inference"]["prediction_seeds"])
                / total_shard_gpu_seconds
            ),
            "timing_note": "summed shard time is GPU/process time; parallel wall time is lower",
            "shards": [
                {
                    "shard_index": manifest["shard_index"],
                    "anchor_count": manifest["anchor_count"],
                    "npz_size_bytes": manifest["npz_size_bytes"],
                    "npz_sha256": manifest["npz_sha256"],
                    "evaluation_seconds": manifest["evaluation_seconds"],
                    "anchor_seed_predictions_per_second": manifest[
                        "anchor_seed_predictions_per_second"
                    ],
                    "strict_load": manifest["strict_load"],
                    "processor_check": manifest["processor_check"],
                }
                for manifest in shard_manifests
            ],
            "scope_config_checkpoint_processor_hashes": prepared.contract["integrity"],
            "monitor_checkpoint_artifacts": [
                {
                    "checkpoint_step": manifest["checkpoint_step"],
                    "npz_sha256": manifest["npz_sha256"],
                    "checkpoint_file_ledger_sha256": manifest[
                        "checkpoint_file_ledger_sha256"
                    ],
                    "strict_load": manifest["strict_load"],
                    "processor_check": manifest["processor_check"],
                    "evaluation_seconds": manifest["evaluation_seconds"],
                }
                for manifest in monitor_manifests
            ],
            "wandb_metrics_only_run": wandb_proof,
            "final_json_atomic_publish": True,
        },
        "limitations": [
            (
                "All 22 episodes and all 7,465 anchors were used for training; there is "
                "no held-out validation or test set."
            ),
            (
                "Teacher-forced reconstruction measures fit to recorded successful "
                "trajectories, not closed-loop stability or task success."
            ),
            (
                "Action/workspace envelope metrics are empirical training-data "
                "diagnostics, not collision, joint-limit, IK, or hardware safety checks."
            ),
            "Prediction-seed variance measures PI0 sampling noise only; it is not training-seed variance.",
        ],
    }
    _atomic_write_json(output_json, report)
    _require(_read_json(output_json)["status"] == "M0_COMPLETE", "atomic final report readback failed")
    return report


def _default_work_dir(run_dir: Path) -> Path:
    return PROJECT_ROOT / "artifacts" / "m0" / run_dir.expanduser().resolve().name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("prepare", "monitor", "evaluate", "finalize"),
        default="evaluate",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--prediction-seeds", type=int, nargs="+", default=list(DEFAULT_PREDICTION_SEEDS))
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS)
    parser.add_argument("--shard-index", type=int, help="Evaluate one shard; omit to evaluate all shards sequentially.")
    parser.add_argument(
        "--shard-indices",
        type=int,
        nargs="+",
        help="Evaluate an explicit shard set (useful for assigning even/odd shards to two GPUs).",
    )
    parser.add_argument(
        "--monitor-checkpoint-steps",
        type=int,
        nargs="+",
        help="Evaluate an explicit epoch-milestone subset; omit to evaluate all 10.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    work_dir = args.work_dir or _default_work_dir(args.run_dir)
    prepared = prepare_m0(
        args.run_dir,
        work_dir=work_dir,
        prediction_seeds=args.prediction_seeds,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
    )
    if args.mode == "prepare":
        result: Mapping[str, Any] = {
            "status": "M0_PREPARED",
            "work_dir": str(prepared.work_dir),
            "contract_sha256": prepared.contract_sha256,
            "checkpoint_selection": prepared.contract["checkpoint_selection"],
            "inference": prepared.contract["inference"],
        }
    elif args.mode == "monitor":
        steps = args.monitor_checkpoint_steps or list(MONITOR_CHECKPOINT_STEPS)
        manifests = _evaluate_monitor_checkpoints(
            prepared, steps, device=args.device, num_workers=args.num_workers
        )
        result = {
            "status": "M0_MONITOR_CHECKPOINTS_PROCESSED",
            "work_dir": str(prepared.work_dir),
            "contract_sha256": prepared.contract_sha256,
            "checkpoints": manifests,
        }
    elif args.mode == "evaluate":
        _require(
            not (args.shard_index is not None and args.shard_indices is not None),
            "use only one of --shard-index and --shard-indices",
        )
        if args.shard_indices is not None:
            indices = args.shard_indices
        elif args.shard_index is not None:
            indices = [args.shard_index]
        else:
            indices = list(range(int(prepared.contract["sharding"]["num_shards"])))
        manifests = _evaluate_missing_shards(
            prepared, indices, device=args.device, num_workers=args.num_workers
        )
        result = {
            "status": "M0_SHARDS_PROCESSED",
            "work_dir": str(prepared.work_dir),
            "contract_sha256": prepared.contract_sha256,
            "shards": manifests,
        }
    else:
        output_json = args.output_json or (prepared.work_dir / "m0_report.json")
        result = finalize_report(prepared, output_json=output_json)
    print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
