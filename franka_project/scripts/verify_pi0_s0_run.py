#!/usr/bin/env python
"""Read-only verifier for a completed Franka PI0 S0 run.

The default path validates local manifests, state files, the saved processors,
strict checkpoint loading, and one real action15 forward pass.  ``--check-wandb``
adds read-only public-API checks; this script never creates runs or artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from franka_eef_pipeline.dual_rate_dataset import CartesianAnchorDataset  # noqa: E402
from franka_eef_pipeline.pi0_training import (  # noqa: E402
    PI0_CORE_WEIGHTS_NAMESPACE,
    STRUCTURAL_ACTION_UNREACHABLE_NUMEL,
    STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS,
    STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT,
    assert_full_parameter_training,
    load_pi0_full_policy_and_processors,
)
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (  # noqa: E402
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    RNG_STATE,
    SCHEDULER_STATE,
)


CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
)
FINAL_S0_STEP = 3
EXPECTED_PARAMETER_COUNT = 3_238_048_528
EXPECTED_PARAMETER_TENSORS = 776
EXPECTED_STRUCTURAL_MISSING_GRADIENTS = {
    f"model.{name}" for name, _shape in STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS
}
CHECKPOINT_PATTERN = re.compile(r"step-(\d{6})$")


class VerificationError(RuntimeError):
    """Raised when an artifact could not have come from the frozen S0 contract."""


@dataclass(frozen=True)
class VerifiedLocalRun:
    run_dir: Path
    config: dict[str, Any]
    run_manifest: dict[str, Any]
    monitor: dict[str, Any]
    checkpoint: Path
    checkpoint_manifest: dict[str, Any]
    model_dir: Path
    report: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _require_nonempty_file(path: Path) -> None:
    _require(path.is_file() and path.stat().st_size > 0, f"missing/empty required file: {path}")


def _validate_training_graph(graph: Any) -> dict[str, Any]:
    _require(isinstance(graph, dict), "training_graph must be a JSON object")
    expected = {
        "paligemma_lm_head_tied": True,
        "expert_lm_head_pruned": True,
        "deduplicated_tied_parameter_count": 526_647_296,
        "pruned_unused_parameter_count": 263_323_648,
        "parameters_removed_from_optimizer": 789_970_944,
    }
    mismatches = {
        key: {"expected": value, "actual": graph.get(key)}
        for key, value in expected.items()
        if graph.get(key) != value
    }
    _require(not mismatches, f"training_graph contract mismatch: {mismatches}")
    structural = graph.get("structural_action_unreachable")
    _require(isinstance(structural, dict), "training_graph lacks structural action-unreachable ledger")
    expected_parameters = [
        {
            "core_parameter_name": name,
            "policy_parameter_name": f"model.{name}",
            "shape": list(shape),
            "numel": math.prod(shape),
        }
        for name, shape in STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS
    ]
    expected_structural = {
        "classification": "permanent_final_prefix_output_outside_action_loss",
        "parameters_retained_and_trainable": True,
        "ddp_requires_find_unused_parameters": True,
        "tensor_count": STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT,
        "numel": STRUCTURAL_ACTION_UNREACHABLE_NUMEL,
        "parameters": expected_parameters,
    }
    _require(
        structural == expected_structural,
        "training_graph structural action-unreachable ledger mismatch",
    )
    return graph


def _verify_invocations(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((run_dir / "invocations").glob("*.json"))
    records = [_read_json(path) for path in paths]
    records.sort(key=lambda item: str(item.get("started_at", "")))
    _require(len(records) == 2, f"completed S0 must have exactly initial+resume invocations, got {len(records)}")
    expected = ((0, 2, False), (2, 3, True))
    for index, (record, (start, stop, resumed)) in enumerate(zip(records, expected, strict=True)):
        actual = (
            record.get("start_step"),
            record.get("last_step"),
            record.get("resume"),
        )
        _require(actual == (start, stop, resumed), f"S0 invocation {index} discontinuity: {actual}")
        _require(record.get("requested_stop_step") == stop, f"S0 invocation {index} stop mismatch")
        _require(record.get("exit_code") == 0, f"S0 invocation {index} did not exit successfully")
    return records


def _verify_checkpoint_file_manifest(model_dir: Path, manifest: Mapping[str, Any], *, hashes: bool) -> None:
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "model checkpoint manifest has no file records")
    for filename, record in files.items():
        _require(isinstance(filename, str) and isinstance(record, dict), "invalid checkpoint file record")
        path = model_dir / filename
        _require_nonempty_file(path)
        _require(path.stat().st_size == record.get("size_bytes"), f"size mismatch: {path}")
        if hashes:
            _require(_sha256_file(path) == record.get("sha256"), f"SHA256 mismatch: {path}")

    # Processor state safetensors are referenced from their JSON configs but
    # are not included in the current model manifest's files table.
    for config_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor_config = _read_json(model_dir / config_name)
        steps = processor_config.get("steps")
        _require(isinstance(steps, list), f"processor config has no steps: {config_name}")
        for step in steps:
            _require(isinstance(step, dict), f"invalid processor step in {config_name}")
            state_file = step.get("state_file")
            if state_file is not None:
                _require(isinstance(state_file, str), f"invalid processor state file in {config_name}")
                _require_nonempty_file(model_dir / state_file)


def verify_local_run(run_dir: str | Path, *, verify_file_hashes: bool = True) -> VerifiedLocalRun:
    """Validate a completed run and its latest atomically-published checkpoint."""

    root = Path(run_dir).expanduser().resolve()
    _require(root.is_dir(), f"run directory does not exist: {root}")
    _require("_ABORTED_" not in root.name and not (root / "ABORTED.json").exists(), "refusing aborted run")
    checkpoints_root = root / "checkpoints"
    staging = sorted(checkpoints_root.glob(".*.staging"))
    _require(not staging, f"refusing run with active/incomplete checkpoint staging: {staging}")

    config_path = root / "resolved_config.yaml"
    _require_nonempty_file(config_path)
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise VerificationError(f"invalid resolved config: {config_path}") from exc
    _require(isinstance(config, dict), "resolved config root must be a mapping")
    _require(config.get("stage") == "S0-15", "verifier accepts only S0-15 runs")
    _require(config.get("training", {}).get("steps") == FINAL_S0_STEP, "S0 planned steps must equal 3")
    _require(
        (config.get("dataset", {}).get("profile"), config.get("dataset", {}).get("observation_fps"),
         config.get("dataset", {}).get("action_fps")) == ("action15", 15, 15),
        "S0 dataset must be action15/obs15/action15",
    )

    run_manifest = _read_json(root / "run_manifest.json")
    _require(run_manifest.get("config_sha256") == _sha256_file(config_path), "run/config SHA256 mismatch")
    _require(run_manifest.get("stage") == "S0-15", "run manifest stage mismatch")
    _require(run_manifest.get("planned_total_steps") == FINAL_S0_STEP, "run manifest planned step mismatch")
    _require(run_manifest.get("wandb_mode") == "online", "run manifest W&B mode is not online")
    _require(run_manifest.get("wandb_artifact_upload") is False, "run manifest permits W&B artifacts")
    _require(run_manifest.get("hub_upload") is False, "run manifest permits Hub upload")
    wandb_id = run_manifest.get("wandb_run_id")
    _require(isinstance(wandb_id, str) and wandb_id, "run manifest lacks W&B run id")

    full_report = run_manifest.get("full_parameter_report")
    _require(isinstance(full_report, dict), "run manifest lacks full parameter report")
    graph = _validate_training_graph(full_report.get("training_graph"))
    _require(full_report.get("total_parameters") == EXPECTED_PARAMETER_COUNT, "unique parameter count mismatch")
    _require(full_report.get("trainable_parameters") == EXPECTED_PARAMETER_COUNT, "trainable count mismatch")
    _require(full_report.get("parameter_tensor_count") == EXPECTED_PARAMETER_TENSORS, "parameter tensor count mismatch")
    _require(full_report.get("trainable_fraction") == 1.0, "run was not full-parameter training")

    monitor = _read_json(root / "monitor_subset.json")
    monitor_without_hash = dict(monitor)
    monitor_hash = monitor_without_hash.pop("subset_sha256", None)
    _require(monitor_hash == _canonical_sha256(monitor_without_hash), "monitor subset canonical SHA mismatch")
    _require(monitor_hash == run_manifest.get("monitor_subset_sha256"), "monitor/run SHA mismatch")
    _require(monitor.get("source") == "training_data" and monitor.get("held_out") is False, "monitor is not in-train")
    indices = monitor.get("indices")
    _require(isinstance(indices, list) and indices, "monitor subset has no indices")

    dataset_root = Path(str(config["dataset"]["root"])).expanduser().resolve()
    _require(str(dataset_root) == run_manifest.get("dataset_root"), "dataset root mismatch")
    profile_path = dataset_root / "meta/franka_eef_profile.json"
    _require_nonempty_file(profile_path)
    _require(_sha256_file(profile_path) == run_manifest.get("dataset_profile_sha256"), "dataset profile SHA mismatch")

    pointer = _read_json(checkpoints_root / "last_checkpoint.json")
    _require(pointer.get("step") == FINAL_S0_STEP, "latest S0 checkpoint is not final step 3")
    checkpoint = (root / str(pointer.get("path", ""))).resolve()
    _require(checkpoint.is_relative_to(root) and checkpoint.is_dir(), "latest checkpoint pointer escapes/is missing")
    _require(checkpoint.name == f"step-{FINAL_S0_STEP:06d}", "latest checkpoint directory name mismatch")
    checkpoint_manifest = _read_json(checkpoint / "checkpoint_manifest.json")
    for key in ("config_sha256", "monitor_subset_sha256", "wandb_run_id", "world_size"):
        _require(checkpoint_manifest.get(key) == run_manifest.get(key), f"checkpoint/run {key} mismatch")
    _require(checkpoint_manifest.get("step") == FINAL_S0_STEP, "checkpoint manifest step mismatch")
    _require(checkpoint_manifest.get("atomic_publish") is True, "checkpoint was not atomically published")
    _require(checkpoint_manifest.get("hub_upload") is False, "checkpoint permits Hub upload")
    _require(checkpoint_manifest.get("wandb_artifact_upload") is False, "checkpoint permits artifact upload")

    published_steps: list[int] = []
    for path in sorted(checkpoints_root.glob("step-*")):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        _require(match is not None and path.is_dir(), f"invalid published checkpoint path: {path}")
        step = int(match.group(1))
        manifest = _read_json(path / "checkpoint_manifest.json")
        _require(manifest.get("step") == step, f"published checkpoint step mismatch: {path}")
        _require(manifest.get("wandb_run_id") == wandb_id, f"W&B run id drift at {path}")
        published_steps.append(step)
    _require(published_steps == [2, 3], f"S0 published checkpoints must be [2,3], got {published_steps}")

    training_state = checkpoint / "training_state"
    training_step = _read_json(training_state / "training_step.json")
    _require(training_step.get("step") == FINAL_S0_STEP, "training-state step mismatch")
    _require(training_step.get("world_size") == run_manifest.get("world_size"), "training-state world size mismatch")
    _require_nonempty_file(training_state / OPTIMIZER_STATE)
    _require_nonempty_file(training_state / OPTIMIZER_PARAM_GROUPS)
    _require_nonempty_file(training_state / SCHEDULER_STATE)
    for rank in range(int(run_manifest["world_size"])):
        _require_nonempty_file(training_state / f"rank-{rank:02d}" / RNG_STATE)

    gradient = _read_json(root / "first_backward_gradient_coverage.json")
    overall = gradient.get("overall", {})
    missing_tensor_count = int(overall.get("trainable_tensors", -1)) - int(
        overall.get("gradient_tensors", -1)
    )
    missing_numel = int(overall.get("trainable_numel", -1)) - int(
        overall.get("gradient_numel", -1)
    )
    _require(
        missing_tensor_count == STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT,
        f"first backward missing tensor count changed: {missing_tensor_count}",
    )
    _require(
        missing_numel == STRUCTURAL_ACTION_UNREACHABLE_NUMEL,
        f"first backward missing numel changed: {missing_numel}",
    )
    _require(overall.get("all_gradients_finite") is True, "first backward contains non-finite gradients")
    groups = gradient.get("groups")
    _require(isinstance(groups, dict), "first backward report has no subsystem groups")
    missing_names: list[str] = []
    for group in groups.values():
        _require(isinstance(group, dict), "first backward contains an invalid subsystem group")
        names = group.get("missing_gradient_names")
        _require(
            isinstance(names, list) and all(isinstance(name, str) for name in names),
            "first backward group lacks a string missing-gradient list",
        )
        missing_names.extend(names)
    _require(len(missing_names) == len(set(missing_names)), "first backward repeats missing gradient names")
    _require(
        set(missing_names) == EXPECTED_STRUCTURAL_MISSING_GRADIENTS,
        "first backward missing-gradient ledger differs from the six structural parameters",
    )
    for name in ("vision_encoder", "vlm", "action_expert", "state_action_projections"):
        group = groups.get(name)
        _require(isinstance(group, dict), f"first backward missing group: {name}")
        _require(group.get("all_gradients_finite") is True, f"first backward {name} non-finite")
        _require(int(group.get("nonzero_gradient_tensors", 0)) > 0, f"first backward {name} all-zero")
        if name != "vlm":
            _require(group.get("gradient_tensor_coverage") == 1.0, f"first backward {name} tensor coverage")
            _require(group.get("gradient_numel_coverage") == 1.0, f"first backward {name} numel coverage")

    model_dir = checkpoint / "pretrained_model"
    model_manifest = _read_json(model_dir / "franka_pi0_checkpoint_manifest.json")
    _require(model_manifest.get("weights_namespace") == PI0_CORE_WEIGHTS_NAMESPACE, "model namespace mismatch")
    _require(model_manifest.get("training_graph") == graph, "saved model/run training_graph mismatch")
    _require(model_manifest.get("parameter_training") == full_report, "saved/run parameter report mismatch")
    _require(model_manifest.get("hub_upload") is False, "saved model permits Hub upload")
    _require(model_manifest.get("wandb_artifact_upload") is False, "saved model permits artifact upload")
    _verify_checkpoint_file_manifest(model_dir, model_manifest, hashes=verify_file_hashes)
    invocations = _verify_invocations(root)

    report = {
        "local_status": "PASS",
        "run_dir": str(root),
        "stage": "S0-15",
        "final_step": FINAL_S0_STEP,
        "published_checkpoint_steps": published_steps,
        "wandb_run_id": wandb_id,
        "checkpoint": str(checkpoint),
        "training_graph": graph,
        "unique_trainable_parameters": EXPECTED_PARAMETER_COUNT,
        "monitor_indices": indices,
        "invocations": invocations,
        "file_hashes_verified": verify_file_hashes,
    }
    return VerifiedLocalRun(root, config, run_manifest, monitor, checkpoint, checkpoint_manifest, model_dir, report)


@contextmanager
def _offline_model_loading() -> Iterable[None]:
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


def load_saved_processors(model_dir: Path, device: torch.device) -> tuple[Any, Any]:
    """Reload exactly the saved local pipelines with canonical converters."""

    with _offline_model_loading():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224",
            local_files_only=True,
        )
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            model_dir,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
            overrides={
                "tokenizer_processor": {"tokenizer": tokenizer, "tokenizer_name": None},
                "device_processor": {"device": str(device)},
                "normalizer_processor": {"device": str(device)},
            },
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            model_dir,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            overrides={
                "unnormalizer_processor": {"device": str(device)},
                "device_processor": {"device": "cpu"},
            },
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
    return preprocessor, postprocessor


def _convert_images(batch: dict[str, Any]) -> None:
    for key in CAMERA_KEYS:
        image = batch.get(key)
        _require(isinstance(image, torch.Tensor) and image.dtype is torch.uint8, f"{key} is not raw uint8")
        converted = image.to(torch.float32).div_(255.0)
        _require(bool(torch.isfinite(converted).all()), f"{key} contains NaN/Inf")
        _require(float(converted.min()) >= 0.0 and float(converted.max()) <= 1.0, f"{key} outside [0,1]")
        batch[key] = converted


def verify_deep(local: VerifiedLocalRun, *, device: str) -> dict[str, Any]:
    """Strictly reload model/processors and execute one real no-grad loss."""

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable for deep verification")
        _require(torch_device.index is not None and torch_device.index < torch.cuda.device_count(), "CUDA device invalid")
        torch.cuda.set_device(torch_device)
        _require(torch.cuda.is_bf16_supported(), "selected CUDA device lacks BF16")

    dataset_cfg = local.config["dataset"]
    dataset = CartesianAnchorDataset(
        dataset_cfg["root"],
        profile=dataset_cfg["profile"],
        episode_indices=dataset_cfg["episode_indices"],
        max_anchors_per_episode=dataset_cfg["max_anchors_per_episode"],
        video_backend="pyav",
    )
    _require(len(dataset) == local.run_manifest.get("dataset_size"), "reloaded dataset size mismatch")
    _require(dataset.effective_stats is not None, "reloaded dataset lacks effective stats")

    with _offline_model_loading():
        policy, _, _, load_report = load_pi0_full_policy_and_processors(
            local.model_dir,
            dataset.effective_stats,
            torch_device,
        )
    parameter_report = assert_full_parameter_training(policy)
    _require(parameter_report == local.run_manifest["full_parameter_report"], "strict reload parameter report mismatch")
    preprocessor, postprocessor = load_saved_processors(local.model_dir, torch_device)

    sample_index = int(local.monitor["indices"][0])
    raw_batch = torch.utils.data.default_collate([dataset[sample_index]])
    raw_action = raw_batch["action"].clone()
    _convert_images(raw_batch)
    processed = preprocessor(raw_batch)
    state = processed.get("observation.state")
    action = processed.get("action")
    _require(isinstance(state, torch.Tensor) and tuple(state.shape) == (1, 10), "processed state shape != [1,10]")
    _require(isinstance(action, torch.Tensor) and tuple(action.shape) == (1, 50, 7), "processed action shape != [1,50,7]")
    _require(bool(torch.isfinite(state).all()) and bool(torch.isfinite(action).all()), "processed state/action non-finite")
    for key in CAMERA_KEYS:
        image = processed.get(key)
        _require(isinstance(image, torch.Tensor) and image.dtype is torch.float32, f"processed {key} dtype")
        _require(bool(torch.isfinite(image).all()), f"processed {key} non-finite")
        _require(float(image.min()) >= 0.0 and float(image.max()) <= 1.0, f"processed {key} outside [0,1]")

    restored_action = postprocessor(action.detach())
    _require(isinstance(restored_action, torch.Tensor), "postprocessor did not return a tensor")
    _require(tuple(restored_action.shape) == tuple(raw_action.shape), "postprocessor action shape mismatch")
    _require(bool(torch.isfinite(restored_action).all()), "postprocessor action non-finite")
    _require(torch.allclose(restored_action, raw_action, atol=2e-5, rtol=2e-5), "processor action roundtrip mismatch")

    was_training = policy.training
    policy.eval()
    cuda_devices = [torch_device.index] if torch_device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        torch.manual_seed(int(local.config["training"]["seed"]) + 41)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed(int(local.config["training"]["seed"]) + 41)
        with torch.autocast(
            device_type=torch_device.type,
            dtype=torch.bfloat16,
            enabled=torch_device.type == "cuda",
        ):
            loss, _ = policy(processed)
    policy.train(was_training)
    _require(isinstance(loss, torch.Tensor) and loss.ndim == 0 and bool(torch.isfinite(loss)), "forward loss non-finite")
    return {
        "deep_status": "PASS",
        "device": str(torch_device),
        "sample_index": sample_index,
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "processor_roundtrip": True,
        "forward_loss": float(loss),
        "strict_load": load_report["pretrained"],
        "parameter_training": parameter_report,
    }


def _scalar_history(run: Any, metric: str) -> list[dict[str, float | int]]:
    rows = list(run.scan_history(keys=["trainer/step", metric], page_size=1000))
    values: list[dict[str, float | int]] = []
    for row in rows:
        if metric not in row or "trainer/step" not in row:
            continue
        step, value = row["trainer/step"], row[metric]
        _require(isinstance(step, (int, float)) and math.isfinite(float(step)), f"W&B {metric} step non-scalar")
        _require(isinstance(value, (int, float)) and math.isfinite(float(value)), f"W&B {metric} non-scalar")
        values.append({"step": int(step), "value": float(value)})
    _require(values, f"W&B history is empty for {metric}")
    return values


def verify_wandb(local: VerifiedLocalRun, *, api: Any | None = None) -> dict[str, Any]:
    """Use only the W&B public read API; never call init/log/artifact APIs."""

    if api is None:
        import wandb

        api = wandb.Api()
    cfg = local.config["wandb"]
    path = f"{cfg['entity']}/{cfg['project']}/{local.run_manifest['wandb_run_id']}"
    run = api.run(path)
    _require(str(run.id) == local.run_manifest["wandb_run_id"], "online W&B run id mismatch")
    _require(str(run.state).lower() == "finished", f"W&B run status is not finished: {run.state}")
    artifacts = list(run.logged_artifacts())
    _require(len(artifacts) == 0, f"W&B run logged {len(artifacts)} artifacts")
    histories = {
        metric: _scalar_history(run, metric)
        for metric in ("train/loss", "train/lr", "train/grad_norm", "monitor/loss")
    }
    final_step = int(local.checkpoint_manifest["step"])
    for metric in ("train/loss", "train/lr", "train/grad_norm"):
        _require({row["step"] for row in histories[metric]} >= set(range(1, final_step + 1)), f"W&B {metric} steps incomplete")
    monitor_steps = {row["step"] for row in histories["monitor/loss"]}
    _require({0, 2, final_step}.issubset(monitor_steps), f"W&B monitor steps incomplete: {monitor_steps}")
    return {
        "wandb_status": "PASS",
        "path": path,
        "state": str(run.state),
        "logged_artifacts": 0,
        "history": histories,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-deep", action="store_true", help="Only validate local manifests/state files.")
    parser.add_argument("--check-wandb", action="store_true", help="Add read-only W&B API verification.")
    parser.add_argument("--skip-file-hashes", action="store_true", help="Check sizes but do not re-hash files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    local = verify_local_run(args.run_dir, verify_file_hashes=not args.skip_file_hashes)
    report: dict[str, Any] = {"schema_version": 1, **local.report}
    if not args.skip_deep:
        report["deep"] = verify_deep(local, device=args.device)
    if args.check_wandb:
        report["wandb"] = verify_wandb(local)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
