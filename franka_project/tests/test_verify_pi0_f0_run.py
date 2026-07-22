from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/verify_pi0_f0_run.py"
FROZEN_CONFIG = PROJECT_ROOT / "configs/train/pi0_full_eef_f0_15hz.yaml"
SPEC = importlib.util.spec_from_file_location("verify_pi0_f0_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


_MODEL_SAFETENSORS_BYTES: bytes | None = None
_OPTIMIZER_SAFETENSORS_BYTES: bytes | None = None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _training_graph() -> dict[str, Any]:
    structural_parameters = [
        {
            "core_parameter_name": name,
            "policy_parameter_name": f"model.{name}",
            "shape": list(shape),
            "numel": math.prod(shape),
        }
        for name, shape in verifier.STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS
    ]
    return {
        "schema_version": 2,
        "paligemma_lm_head_tied": True,
        "paligemma_tied_weight_shape": [257152, 2048],
        "paligemma_lm_head_key": verifier.PALIGEMMA_LM_HEAD_KEY,
        "paligemma_embed_tokens_key": verifier.PALIGEMMA_EMBED_TOKENS_KEY,
        "expert_lm_head_pruned": True,
        "unused_expert_lm_head_key": verifier.UNUSED_EXPERT_LM_HEAD_KEY,
        "unused_expert_lm_head_shape": [257152, 1024],
        "deduplicated_tied_parameter_count": 526_647_296,
        "pruned_unused_parameter_count": 263_323_648,
        "parameters_removed_from_optimizer": 789_970_944,
        "structural_action_unreachable": {
            "classification": "permanent_final_prefix_output_outside_action_loss",
            "parameters_retained_and_trainable": True,
            "ddp_requires_find_unused_parameters": True,
            "tensor_count": verifier.STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT,
            "numel": verifier.STRUCTURAL_ACTION_UNREACHABLE_NUMEL,
            "parameters": structural_parameters,
        },
    }


def _full_parameter_report() -> dict[str, Any]:
    return {
        "use_peft": False,
        "lora_parameter_count": 0,
        "parameter_tensor_count": verifier.EXPECTED_PARAMETER_TENSORS,
        "total_parameters": verifier.EXPECTED_PARAMETER_COUNT,
        "trainable_parameters": verifier.EXPECTED_PARAMETER_COUNT,
        "trainable_fraction": 1.0,
        "gradient_checkpointing_enabled": True,
        "policy_training_mode": True,
        "paligemma_lm_head_tied": True,
        "expert_lm_head_pruned": True,
        "training_graph": _training_graph(),
    }


def _saved_model_config() -> dict[str, Any]:
    return {
        "type": "pi0",
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "use_relative_actions": False,
        "use_peft": False,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "chunk_size": 50,
        "n_action_steps": 50,
        "push_to_hub": False,
        "input_features": {"observation.state": {"type": "STATE", "shape": [10]}},
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
    }


def _write_model_safetensors(path: Path) -> None:
    global _MODEL_SAFETENSORS_BYTES
    if _MODEL_SAFETENSORS_BYTES is not None:
        path.write_bytes(_MODEL_SAFETENSORS_BYTES)
        return
    tensors = {verifier.PALIGEMMA_LM_HEAD_KEY: torch.zeros(1)}
    tensors.update(
        {
            f"test.parameter_{index:03d}": torch.tensor(float(index))
            for index in range(verifier.EXPECTED_PARAMETER_TENSORS - 1)
        }
    )
    save_file(
        tensors,
        path,
        metadata={verifier.PALIGEMMA_EMBED_TOKENS_KEY: verifier.PALIGEMMA_LM_HEAD_KEY},
    )
    _MODEL_SAFETENSORS_BYTES = path.read_bytes()


def _write_optimizer_safetensors(path: Path) -> None:
    global _OPTIMIZER_SAFETENSORS_BYTES
    if _OPTIMIZER_SAFETENSORS_BYTES is not None:
        path.write_bytes(_OPTIMIZER_SAFETENSORS_BYTES)
        return
    tensors: dict[str, torch.Tensor] = {}
    active = set(range(verifier.EXPECTED_PARAMETER_TENSORS)) - set(
        verifier.EXPECTED_STATELESS_OPTIMIZER_IDS
    )
    for parameter_id in sorted(active):
        tensors[f"state/{parameter_id}/exp_avg"] = torch.zeros(1)
        tensors[f"state/{parameter_id}/exp_avg_sq"] = torch.zeros(1)
        tensors[f"state/{parameter_id}/step"] = torch.tensor(float(verifier.FINAL_F0_STEP))
    save_file(tensors, path)
    _OPTIMIZER_SAFETENSORS_BYTES = path.read_bytes()


def _write_model_checkpoint(
    model: Path,
    *,
    step: int,
    root: Path,
    full_report: dict[str, Any],
) -> None:
    model.mkdir(parents=True)
    _write_json(model / "config.json", _saved_model_config())
    _write_model_safetensors(model / "model.safetensors")
    _write_json(
        model / "policy_preprocessor.json",
        {"steps": [{"registry_name": "normalizer_processor", "state_file": "pre-state.safetensors"}]},
    )
    _write_json(
        model / "policy_postprocessor.json",
        {"steps": [{"registry_name": "unnormalizer_processor", "state_file": "post-state.safetensors"}]},
    )
    (model / "pre-state.safetensors").write_bytes(b"pre")
    (model / "post-state.safetensors").write_bytes(b"post")
    _write_json(
        model / "franka_eef_geometry_manifest.json",
        {"task_instruction": "stack the cups", "action7": {"frequency_hz": 15, "chunk_size": 50}},
    )
    _write_json(model / "pi0_eef_stats.json", {"observation_fps": 15, "action_fps": 15})
    names = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "franka_eef_geometry_manifest.json",
        "pi0_eef_stats.json",
    )
    precheck_model = (root / f"checkpoints/step-{verifier.PRECHECK_STEP:06d}/pretrained_model").resolve()
    resume = None
    if step != verifier.PRECHECK_STEP:
        resume = {
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "project_manifest_present": True,
            "directory": str(precheck_model),
            "weights_path": str(precheck_model / "model.safetensors"),
            "parameter_training": full_report,
            "training_graph": _training_graph(),
        }
    manifest = {
        "schema_version": 1,
        "weights_namespace": verifier.PI0_CORE_WEIGHTS_NAMESPACE,
        "hub_upload": False,
        "wandb_artifact_upload": False,
        "parameter_training": full_report,
        "training_graph": _training_graph(),
        "source_load_report": {
            "base": {"pretrained": {"strict": True}},
            "resume": resume,
        },
        "files": {
            name: {
                "size_bytes": (model / name).stat().st_size,
                "sha256": verifier._sha256_file(model / name),
            }
            for name in names
        },
    }
    _write_json(model / "franka_pi0_checkpoint_manifest.json", manifest)


def _write_training_state(state: Path, *, step: int, training: dict[str, Any]) -> None:
    state.mkdir(parents=True)
    _write_json(
        state / "training_step.json",
        {
            "step": step,
            "world_size": verifier.WORLD_SIZE,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
    )
    for rank in range(verifier.WORLD_SIZE):
        rank_path = state / f"rank-{rank:02d}" / verifier.RNG_STATE
        rank_path.parent.mkdir()
        rank_path.write_bytes(f"rng-{rank}".encode())
    if step == verifier.FINAL_F0_STEP:
        _write_optimizer_safetensors(state / verifier.OPTIMIZER_STATE)
        _write_json(
            state / verifier.OPTIMIZER_PARAM_GROUPS,
            [
                {
                    "lr": float(training["final_lr"]),
                    "initial_lr": float(training["peak_lr"]),
                    "betas": [float(value) for value in training["betas"]],
                    "eps": float(training["eps"]),
                    "weight_decay": float(training["weight_decay"]),
                    "params": list(range(verifier.EXPECTED_PARAMETER_TENSORS)),
                }
            ],
        )
        _write_json(
            state / verifier.SCHEDULER_STATE,
            {
                "base_lrs": [float(training["peak_lr"])],
                "last_epoch": verifier.FINAL_F0_STEP,
                "_step_count": verifier.FINAL_F0_STEP + 1,
                "_last_lr": [float(training["final_lr"])],
                "lr_lambdas": [None],
            },
        )
    else:
        (state / verifier.OPTIMIZER_STATE).write_bytes(b"optimizer")
        _write_json(state / verifier.OPTIMIZER_PARAM_GROUPS, [{}])
        _write_json(state / verifier.SCHEDULER_STATE, {})


def _gradient_report() -> dict[str, Any]:
    reached = {
        "trainable_tensors": 1,
        "trainable_numel": 1,
        "gradient_tensors": 1,
        "gradient_numel": 1,
        "gradient_tensor_coverage": 1.0,
        "gradient_numel_coverage": 1.0,
        "all_gradients_finite": True,
        "nonzero_gradient_tensors": 1,
        "missing_gradient_names": [],
    }
    structural_names = sorted(verifier.EXPECTED_STRUCTURAL_MISSING_GRADIENTS)
    vlm = {
        "trainable_tensors": 7,
        "trainable_numel": verifier.STRUCTURAL_ACTION_UNREACHABLE_NUMEL + 1,
        "gradient_tensors": 1,
        "gradient_numel": 1,
        "gradient_tensor_coverage": 1 / 7,
        "gradient_numel_coverage": 1 / (verifier.STRUCTURAL_ACTION_UNREACHABLE_NUMEL + 1),
        "all_gradients_finite": True,
        "nonzero_gradient_tensors": 1,
        "missing_gradient_names": structural_names,
    }
    return {
        "inspect_values": True,
        "overall": {
            "parameter_tensors": verifier.EXPECTED_PARAMETER_TENSORS,
            "parameter_numel": verifier.EXPECTED_PARAMETER_COUNT,
            "trainable_tensors": verifier.EXPECTED_PARAMETER_TENSORS,
            "trainable_numel": verifier.EXPECTED_PARAMETER_COUNT,
            "gradient_tensors": (
                verifier.EXPECTED_PARAMETER_TENSORS
                - verifier.STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT
            ),
            "gradient_numel": (
                verifier.EXPECTED_PARAMETER_COUNT
                - verifier.STRUCTURAL_ACTION_UNREACHABLE_NUMEL
            ),
            "gradient_tensor_coverage": (
                verifier.EXPECTED_PARAMETER_TENSORS
                - verifier.STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT
            )
            / verifier.EXPECTED_PARAMETER_TENSORS,
            "gradient_numel_coverage": (
                verifier.EXPECTED_PARAMETER_COUNT
                - verifier.STRUCTURAL_ACTION_UNREACHABLE_NUMEL
            )
            / verifier.EXPECTED_PARAMETER_COUNT,
            "all_gradients_finite": True,
            "nonzero_gradient_tensors": 769,
        },
        "groups": {
            "vision_encoder": dict(reached),
            "vlm": vlm,
            "action_expert": dict(reached),
            "state_action_projections": dict(reached),
        },
    }


def _build_complete_run(tmp_path: Path) -> Path:
    root = tmp_path / "f0-run"
    root.mkdir(parents=True)
    config_path = root / "resolved_config.yaml"
    config_path.write_bytes(FROZEN_CONFIG.read_bytes())
    config = yaml.safe_load(config_path.read_text())
    dataset_root = Path(config["dataset"]["root"]).resolve()
    profile_path = dataset_root / "meta/franka_eef_profile.json"

    indices = list(range(verifier.EXPECTED_MONITOR_SAMPLES))
    monitor = {
        "schema_version": 1,
        "source": "training_data",
        "held_out": False,
        "seed": 1017,
        "population_size": verifier.DATASET_SIZE,
        "sample_size": verifier.EXPECTED_MONITOR_SAMPLES,
        "indices": indices,
        "anchors": [{"logical_index": index} for index in indices],
        "includes_converter_frozen_indices": indices,
    }
    monitor["subset_sha256"] = verifier._canonical_sha256(monitor)
    _write_json(root / "monitor_subset.json", monitor)

    full_report = _full_parameter_report()
    run_manifest = {
        "schema_version": 1,
        "config_sha256": verifier._sha256_file(config_path),
        "monitor_subset_sha256": monitor["subset_sha256"],
        "world_size": verifier.WORLD_SIZE,
        "planned_total_steps": verifier.FINAL_F0_STEP,
        "dataset_profile_sha256": verifier._sha256_file(profile_path),
        "stage": verifier.STAGE,
        "job_name": config["job_name"],
        "dataset_root": str(dataset_root),
        "dataset_size": verifier.DATASET_SIZE,
        "steps_per_epoch": verifier.STEPS_PER_EPOCH,
        "warmup_steps": verifier.EXPECTED_WARMUP_STEPS,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "wandb_run_id": "same-id",
        "wandb_url": "https://wandb.invalid/same-id",
        "wandb_mode": "online",
        "wandb_artifact_upload": False,
        "hub_upload": False,
        "full_parameter_report": full_report,
    }
    _write_json(root / "run_manifest.json", run_manifest)

    steps = (verifier.PRECHECK_STEP, *verifier.EXPECTED_EPOCH_CHECKPOINT_STEPS)
    for step in steps:
        checkpoint = root / "checkpoints" / f"step-{step:06d}"
        _write_json(
            checkpoint / "checkpoint_manifest.json",
            {
                "schema_version": 1,
                "step": step,
                "config_sha256": run_manifest["config_sha256"],
                "monitor_subset_sha256": monitor["subset_sha256"],
                "wandb_run_id": "same-id",
                "world_size": verifier.WORLD_SIZE,
                "hub_upload": False,
                "wandb_artifact_upload": False,
                "atomic_publish": True,
            },
        )
        _write_training_state(checkpoint / "training_state", step=step, training=config["training"])
        _write_model_checkpoint(
            checkpoint / "pretrained_model",
            step=step,
            root=root,
            full_report=full_report,
        )
    _write_json(
        root / "checkpoints/last_checkpoint.json",
        {
            "step": verifier.FINAL_F0_STEP,
            "path": f"checkpoints/step-{verifier.FINAL_F0_STEP:06d}",
        },
    )
    _write_json(root / "first_backward_gradient_coverage.json", _gradient_report())

    invocations = (
        {
            "started_at": "1",
            "start_step": 0,
            "requested_stop_step": verifier.PRECHECK_STEP,
            "last_step": verifier.PRECHECK_STEP,
            "exit_code": 0,
            "resume": False,
        },
        {
            "started_at": "2",
            "start_step": verifier.PRECHECK_STEP,
            "requested_stop_step": verifier.FINAL_F0_STEP,
            "last_step": verifier.FINAL_F0_STEP,
            "exit_code": 0,
            "resume": True,
        },
    )
    for index, invocation in enumerate(invocations):
        _write_json(root / f"invocations/{index}.json", invocation)
    for invocation in range(2):
        for rank in range(verifier.WORLD_SIZE):
            path = root / "logs" / f"20260715T00000{invocation}_r{rank}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"rank={rank} invocation={invocation}\n")
    return root


def test_local_verifier_accepts_complete_f0_chain_and_strict_final_state(tmp_path: Path) -> None:
    root = _build_complete_run(tmp_path)

    local = verifier.verify_local_run(root)

    assert local.report["local_status"] == "PASS"
    assert local.report["dataset_size"] == 7_465
    assert local.report["world_size"] == 2
    assert local.report["steps_per_epoch"] == 3_733
    assert local.report["final_step"] == 37_330
    assert local.report["published_checkpoint_steps"] == [
        verifier.PRECHECK_STEP,
        *verifier.EXPECTED_EPOCH_CHECKPOINT_STEPS,
    ]
    assert local.report["optimizer_scheduler"]["optimizer_active_state_ids"] == 770
    assert local.report["optimizer_scheduler"]["optimizer_step"] == 37_330
    assert local.report["deep_model_loaded"] is False


def test_local_verifier_allows_epoch_chain_without_retained_step2_checkpoint(tmp_path: Path) -> None:
    root = _build_complete_run(tmp_path)
    shutil.rmtree(root / "checkpoints/step-000002")

    local = verifier.verify_local_run(root)

    assert local.report["published_checkpoint_steps"] == list(
        verifier.EXPECTED_EPOCH_CHECKPOINT_STEPS
    )


def test_local_verifier_rejects_running_pointer_and_missing_epoch_checkpoint(tmp_path: Path) -> None:
    root = _build_complete_run(tmp_path)
    _write_json(
        root / "checkpoints/last_checkpoint.json",
        {"step": 3_733, "path": "checkpoints/step-003733"},
    )
    with pytest.raises(verifier.VerificationError, match="not complete"):
        verifier.verify_local_run(root)

    root = _build_complete_run(tmp_path / "second")
    shutil.rmtree(root / "checkpoints/step-007466")
    with pytest.raises(verifier.VerificationError, match="missing epoch checkpoints"):
        verifier.verify_local_run(root)


def test_local_verifier_rejects_identity_rank_rng_and_resume_drift(tmp_path: Path) -> None:
    root = _build_complete_run(tmp_path)
    manifest_path = root / "checkpoints/step-003733/checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["wandb_run_id"] = "different"
    _write_json(manifest_path, manifest)
    with pytest.raises(verifier.VerificationError, match="checkpoint manifest mismatch"):
        verifier.verify_local_run(root)

    root = _build_complete_run(tmp_path / "world")
    run_manifest_path = root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest["world_size"] = 1
    _write_json(run_manifest_path, run_manifest)
    with pytest.raises(verifier.VerificationError, match="run manifest mismatch"):
        verifier.verify_local_run(root)

    root = _build_complete_run(tmp_path / "rng")
    (root / "checkpoints/step-037330/training_state/rank-01/rng_state.safetensors").unlink()
    with pytest.raises(verifier.VerificationError, match="rank-01"):
        verifier.verify_local_run(root)

    root = _build_complete_run(tmp_path / "resume")
    invocation_path = root / "invocations/1.json"
    invocation = json.loads(invocation_path.read_text())
    invocation["resume"] = False
    _write_json(invocation_path, invocation)
    with pytest.raises(verifier.VerificationError, match="invocation 1 discontinuity"):
        verifier.verify_local_run(root)

    root = _build_complete_run(tmp_path / "source")
    model_manifest_path = (
        root
        / "checkpoints/step-037330/pretrained_model/franka_pi0_checkpoint_manifest.json"
    )
    model_manifest = json.loads(model_manifest_path.read_text())
    model_manifest["source_load_report"]["resume"]["directory"] = str(
        root / "checkpoints/step-003733/pretrained_model"
    )
    _write_json(model_manifest_path, model_manifest)
    with pytest.raises(verifier.VerificationError, match="same-run step 2"):
        verifier.verify_local_run(root)


def test_local_verifier_rejects_scheduler_optimizer_and_exact6_drift(tmp_path: Path) -> None:
    root = _build_complete_run(tmp_path)
    scheduler_path = root / "checkpoints/step-037330/training_state/scheduler_state.json"
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["last_epoch"] -= 1
    _write_json(scheduler_path, scheduler)
    with pytest.raises(verifier.VerificationError, match="scheduler state mismatch"):
        verifier.verify_local_run(root)

    root = _build_complete_run(tmp_path / "optimizer")
    group_path = root / "checkpoints/step-037330/training_state/optimizer_param_groups.json"
    groups = json.loads(group_path.read_text())
    groups[0]["params"].pop()
    _write_json(group_path, groups)
    with pytest.raises(verifier.VerificationError, match="optimizer parameter ids changed"):
        verifier.verify_local_run(root)

    root = _build_complete_run(tmp_path / "gradient")
    gradient_path = root / "first_backward_gradient_coverage.json"
    gradient = json.loads(gradient_path.read_text())
    gradient["groups"]["vlm"]["missing_gradient_names"].append("model.seventh.weight")
    _write_json(gradient_path, gradient)
    with pytest.raises(verifier.VerificationError, match="exact six"):
        verifier.verify_local_run(root)


def _write_tiny_optimizer_state(path: Path, active_ids: set[int]) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for parameter_id in active_ids:
        tensors[f"state/{parameter_id}/exp_avg"] = torch.zeros(())
        tensors[f"state/{parameter_id}/exp_avg_sq"] = torch.zeros(())
        tensors[f"state/{parameter_id}/step"] = torch.ones(())
    save_file(tensors, path)


def test_deep_optimizer_mapping_uses_param_group_order_to_name_stateless_parameters(
    tmp_path: Path,
) -> None:
    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alpha = torch.nn.Parameter(torch.zeros(()))
            self.structural_first = torch.nn.Parameter(torch.zeros(()))
            self.gamma = torch.nn.Parameter(torch.zeros(()))
            self.structural_second = torch.nn.Parameter(torch.zeros(()))

    state_dir = tmp_path / "training_state"
    state_dir.mkdir()
    # Deliberately non-contiguous ids prove that mapping follows saved group
    # order rather than treating an optimizer id as a named-parameter index.
    _write_json(
        state_dir / verifier.OPTIMIZER_PARAM_GROUPS,
        [{"params": [41, 7, 99, 12]}],
    )
    _write_tiny_optimizer_state(state_dir / verifier.OPTIMIZER_STATE, {41, 99})
    declared = {"structural_first", "structural_second"}

    report = verifier._verify_deep_optimizer_parameter_mapping(
        TinyPolicy(),
        state_dir,
        metadata_names=declared,
        gradient_names=declared,
    )

    assert report["optimizer_parameter_ids_mapped"] == 4
    assert report["optimizer_stateless_ids"] == [7, 12]
    assert report["optimizer_stateless_parameter_names"] == [
        "structural_first",
        "structural_second",
    ]
    assert report["metadata_gradient_exact_match"] is True


def test_deep_optimizer_mapping_rejects_stateless_name_or_declaration_drift(tmp_path: Path) -> None:
    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alpha = torch.nn.Parameter(torch.zeros(()))
            self.structural = torch.nn.Parameter(torch.zeros(()))
            self.gamma = torch.nn.Parameter(torch.zeros(()))

    state_dir = tmp_path / "training_state"
    state_dir.mkdir()
    _write_json(state_dir / verifier.OPTIMIZER_PARAM_GROUPS, [{"params": [8, 3, 21]}])
    _write_tiny_optimizer_state(state_dir / verifier.OPTIMIZER_STATE, {3, 21})

    with pytest.raises(verifier.VerificationError, match="stateless optimizer Parameters differ"):
        verifier._verify_deep_optimizer_parameter_mapping(
            TinyPolicy(),
            state_dir,
            metadata_names={"structural"},
            gradient_names={"structural"},
        )

    with pytest.raises(verifier.VerificationError, match="metadata and gradient report disagree"):
        verifier._verify_deep_optimizer_parameter_mapping(
            TinyPolicy(),
            state_dir,
            metadata_names={"alpha"},
            gradient_names={"structural"},
        )


def test_default_cli_is_local_only_and_never_calls_deep_or_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local = verifier.verify_local_run(_build_complete_run(tmp_path))
    calls: list[str] = []
    monkeypatch.setattr(verifier, "verify_local_run", lambda *args, **kwargs: local)
    monkeypatch.setattr(verifier, "verify_deep", lambda *args, **kwargs: calls.append("deep"))
    monkeypatch.setattr(verifier, "verify_wandb", lambda *args, **kwargs: calls.append("wandb"))

    assert verifier.main(["--run-dir", str(local.run_dir)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert calls == []
    assert payload["local_status"] == "PASS"
    assert "deep" not in payload and "wandb" not in payload


def test_deep_verifier_uses_full_dataset_contract_and_one_finite_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = verifier.verify_local_run(_build_complete_run(tmp_path))

    class FakeDataset:
        effective_stats = {"observation.state": {}, "action": {}}

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return verifier.DATASET_SIZE

        def __getitem__(self, index):
            return {
                "observation.images.base_0_rgb": torch.zeros(3, 2, 2, dtype=torch.uint8),
                "observation.images.left_wrist_0_rgb": torch.full((3, 2, 2), 255, dtype=torch.uint8),
                "observation.state": torch.zeros(10),
                "action": torch.zeros(50, 7),
                "task": "stack the cups",
            }

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, batch):
            return self.weight.square(), {}

    policy = FakePolicy()
    monkeypatch.setattr(verifier, "CartesianAnchorDataset", FakeDataset)
    monkeypatch.setattr(
        verifier,
        "load_pi0_full_policy_and_processors",
        lambda *args, **kwargs: (
            policy,
            lambda batch: batch,
            lambda action: action,
            {"pretrained": {"strict": True}},
        ),
    )
    monkeypatch.setattr(
        verifier,
        "assert_full_parameter_training",
        lambda policy: local.run_manifest["full_parameter_report"],
    )
    monkeypatch.setattr(
        verifier,
        "load_saved_processors",
        lambda *args: (lambda batch: batch, lambda action: action),
    )
    monkeypatch.setattr(
        verifier,
        "_verify_deep_optimizer_parameter_mapping",
        lambda *args, **kwargs: {
            "optimizer_parameter_ids_mapped": verifier.EXPECTED_PARAMETER_TENSORS,
            "optimizer_stateless_ids": sorted(verifier.EXPECTED_STATELESS_OPTIMIZER_IDS),
            "optimizer_stateless_parameter_names": sorted(
                verifier.EXPECTED_STRUCTURAL_MISSING_GRADIENTS
            ),
            "metadata_gradient_exact_match": True,
        },
    )

    report = verifier.verify_deep(local, device="cpu")

    assert report["deep_status"] == "PASS"
    assert report["state_shape"] == [1, 10]
    assert report["action_shape"] == [1, 50, 7]
    assert report["processor_roundtrip"] is True
    assert report["saved_processors_match_effective_stats"] is True
    assert report["optimizer_parameter_mapping"]["metadata_gradient_exact_match"] is True
    assert report["forward_loss"] == 1.0


class _FakeArtifactFile:
    def __init__(self, name: str, size: int, digest: str | None = "test-digest"):
        self.name = name
        self.size = size
        self.digest = digest


class _FakeArtifact:
    def __init__(
        self,
        *,
        name: str = "run-same-id-history:v0",
        artifact_type: str = "wandb-history",
        description: str = "Weights & Biases Run History Data for same-id",
        size: int = 51_454,
        files: list[_FakeArtifactFile] | None = None,
    ):
        self.name = name
        self.type = artifact_type
        self.description = description
        self.size = size
        self._files = files if files is not None else [_FakeArtifactFile("0000.parquet", size)]

    def files(self):
        return list(self._files)


class _FakeRun:
    id = "same-id"
    state = "finished"

    def __init__(
        self,
        *,
        final_lr: float,
        artifacts: list[_FakeArtifact] | None = None,
        drop_monitor: int | None = None,
        drop_train: int | None = None,
    ):
        self.artifacts = artifacts or []
        self.drop_monitor = drop_monitor
        self.drop_train = drop_train
        self.final_lr = final_lr

    def logged_artifacts(self):
        return list(self.artifacts)

    def scan_history(self, *, keys, page_size):
        metric = keys[1]
        if metric == "monitor/loss":
            steps = sorted(verifier.EXPECTED_MONITOR_STEPS - {self.drop_monitor})
            placeholder_steps = sorted(verifier.EXPECTED_TRAIN_LOG_STEPS)
        else:
            steps = sorted(verifier.EXPECTED_TRAIN_LOG_STEPS - {self.drop_train})
            placeholder_steps = sorted(verifier.EXPECTED_MONITOR_STEPS)
        rows = []
        for step in steps:
            value = 0.5
            if metric == "train/lr" and step == verifier.FINAL_F0_STEP:
                value = self.final_lr
            rows.append({"trainer/step": step, metric: value})
        # W&B's history parquet contains sparse union rows.  In particular,
        # train scans include monitor-only rows (and vice versa), represented
        # by the requested metric being explicitly None.  Some share a step
        # with a real metric observation.
        rows.extend({"trainer/step": step, metric: None} for step in placeholder_steps)
        return rows


class _FakeApi:
    def __init__(self, run: _FakeRun):
        self.fake_run = run
        self.paths: list[str] = []

    def run(self, path):
        self.paths.append(path)
        return self.fake_run


def _fake_wandb_api(local, *, artifacts=None, drop_monitor=None, drop_train=None):
    return _FakeApi(
        _FakeRun(
            final_lr=float(local.config["training"]["final_lr"]),
            artifacts=artifacts,
            drop_monitor=drop_monitor,
            drop_train=drop_train,
        )
    )


def test_wandb_verifier_remains_compatible_with_zero_artifacts(tmp_path: Path) -> None:
    local = verifier.verify_local_run(_build_complete_run(tmp_path))
    api = _fake_wandb_api(local)

    report = verifier.verify_wandb(local, api=api)

    assert report["wandb_status"] == "PASS"
    assert report["logged_artifacts"] == 0
    assert report["total_artifacts"] == 0
    assert report["allowed_history_artifacts"] == 0
    assert report["model_artifacts"] == 0
    assert report["allowed_history_artifact_ledger"] == []
    assert report["expected_train_metric_points"] == len(verifier.EXPECTED_TRAIN_LOG_STEPS)
    assert report["expected_monitor_points"] == 12
    assert len(report["history"]["train/loss"]) == 748
    assert len(report["history"]["train/lr"]) == 748
    assert len(report["history"]["train/grad_norm"]) == 748
    assert len(report["history"]["monitor/loss"]) == 12
    assert api.paths == ["ttuhamg/franka_pi0_full_eef/same-id"]


def test_wandb_verifier_allows_and_registers_single_automatic_history_artifact(
    tmp_path: Path,
) -> None:
    local = verifier.verify_local_run(_build_complete_run(tmp_path))
    artifact = _FakeArtifact()

    report = verifier.verify_wandb(
        local,
        api=_fake_wandb_api(local, artifacts=[artifact]),
    )

    assert report["wandb_status"] == "PASS"
    assert report["logged_artifacts"] == 1
    assert report["total_artifacts"] == 1
    assert report["allowed_history_artifacts"] == 1
    assert report["model_artifacts"] == 0
    assert report["allowed_history_artifact_ledger"] == [
        {
            "name": "run-same-id-history:v0",
            "type": "wandb-history",
            "description": "Weights & Biases Run History Data for same-id",
            "size_bytes": 51_454,
            "files": [
                {
                    "name": "0000.parquet",
                    "size_bytes": 51_454,
                    "digest": "test-digest",
                }
            ],
        }
    ]


def test_wandb_history_whitelist_matches_real_f0_run_metadata() -> None:
    artifact = _FakeArtifact(
        name="run-jkrz5th0-history:v0",
        description="Weights & Biases Run History Data for jkrz5th0",
    )
    run = _FakeRun(final_lr=0.0, artifacts=[artifact])

    report = verifier._validate_wandb_artifacts(run, run_id="jkrz5th0")

    assert report["total_artifacts"] == 1
    assert report["allowed_history_artifacts"] == 1
    assert report["model_artifacts"] == 0
    assert report["allowed_history_artifact_ledger"][0]["description"] == (
        "Weights & Biases Run History Data for jkrz5th0"
    )


@pytest.mark.parametrize(
    ("artifacts", "message"),
    [
        (
            [_FakeArtifact(name="policy:v0", artifact_type="model")],
            "forbidden model/checkpoint/weight marker",
        ),
        (
            [_FakeArtifact(artifact_type="dataset")],
            "unknown W&B artifact type",
        ),
        *[
            (
                [
                    _FakeArtifact(
                        files=[_FakeArtifactFile(f"payload.{suffix}", 51_454)],
                    )
                ],
                "suspicious weight file",
            )
            for suffix in ("safetensors", "bin", "pt", "pth")
        ],
        (
            [
                _FakeArtifact(
                    files=[_FakeArtifactFile("checkpoint/latest.parquet", 51_454)],
                )
            ],
            "suspicious weight file",
        ),
        (
            [
                _FakeArtifact(
                    size=verifier.MAX_WANDB_HISTORY_ARTIFACT_BYTES + 1,
                )
            ],
            "is too large",
        ),
        (
            [
                _FakeArtifact(
                    size=51_455,
                    files=[
                        _FakeArtifactFile("0000.parquet", 51_454),
                        _FakeArtifactFile("metadata.json", 1),
                    ],
                )
            ],
            "must contain exactly one file",
        ),
        (
            [_FakeArtifact(), _FakeArtifact()],
            "at most one automatic history artifact",
        ),
    ],
)
def test_wandb_verifier_rejects_non_history_or_suspicious_artifacts(
    tmp_path: Path,
    artifacts: list[_FakeArtifact],
    message: str,
) -> None:
    local = verifier.verify_local_run(_build_complete_run(tmp_path))

    with pytest.raises(verifier.VerificationError, match=message):
        verifier.verify_wandb(
            local,
            api=_fake_wandb_api(local, artifacts=artifacts),
        )


def test_wandb_verifier_still_rejects_incomplete_monitor_cadence(tmp_path: Path) -> None:
    local = verifier.verify_local_run(_build_complete_run(tmp_path))

    with pytest.raises(verifier.VerificationError, match="monitor cadence mismatch"):
        verifier.verify_wandb(
            local,
            api=_fake_wandb_api(local, drop_monitor=verifier.STEPS_PER_EPOCH),
        )


def test_wandb_verifier_rejects_missing_real_train_point_despite_none_placeholder(
    tmp_path: Path,
) -> None:
    local = verifier.verify_local_run(_build_complete_run(tmp_path))

    with pytest.raises(verifier.VerificationError, match="train/loss cadence mismatch"):
        verifier.verify_wandb(
            local,
            # Step 2 remains present as a monitor-originated None placeholder,
            # but its real train value is absent and must not count.
            api=_fake_wandb_api(local, drop_train=verifier.PRECHECK_STEP),
        )


class _FakeScalarHistoryRun:
    def __init__(self, rows):
        self.rows = rows

    def scan_history(self, *, keys, page_size):
        return list(self.rows)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"trainer/step": 1, "metric": float("nan")}, "non-finite"),
        ({"trainer/step": 1, "metric": "0.5"}, "non-scalar"),
        ({"metric": 0.5}, "without an optimizer step"),
        ({"trainer/step": None, "metric": 0.5}, "without an optimizer step"),
    ],
)
def test_scalar_history_rejects_bad_real_values_or_missing_steps(row, message) -> None:
    with pytest.raises(verifier.VerificationError, match=message):
        verifier._scalar_history(_FakeScalarHistoryRun([row]), "metric")
