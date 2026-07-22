from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/verify_pi0_s0_run.py"
SPEC = importlib.util.spec_from_file_location("verify_pi0_s0_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


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
        "paligemma_lm_head_tied": True,
        "paligemma_tied_weight_shape": [257152, 2048],
        "paligemma_lm_head_key": "paligemma_with_expert.paligemma.lm_head.weight",
        "paligemma_embed_tokens_key": (
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        ),
        "expert_lm_head_pruned": True,
        "unused_expert_lm_head_key": "paligemma_with_expert.gemma_expert.lm_head.weight",
        "unused_expert_lm_head_shape": [257152, 1024],
        "deduplicated_tied_parameter_count": 526647296,
        "pruned_unused_parameter_count": 263323648,
        "parameters_removed_from_optimizer": 789970944,
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


def _build_complete_run(tmp_path: Path) -> Path:
    root = tmp_path / "s0-run"
    dataset_root = tmp_path / "dataset"
    profile_path = dataset_root / "meta/franka_eef_profile.json"
    _write_json(profile_path, {"profile": "action15"})
    config = {
        "stage": "S0-15",
        "job_name": "unit-s0",
        "dataset": {
            "root": str(dataset_root),
            "profile": "action15",
            "observation_fps": 15,
            "action_fps": 15,
            "episode_indices": [0, 1],
            "max_anchors_per_episode": 1,
        },
        "training": {"steps": 3, "seed": 1000},
        "wandb": {"entity": "entity", "project": "project"},
    }
    config_path = root / "resolved_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monitor = {
        "schema_version": 1,
        "source": "training_data",
        "held_out": False,
        "seed": 1017,
        "population_size": 2,
        "sample_size": 2,
        "indices": [0, 1],
        "anchors": [0, 1],
        "includes_converter_frozen_indices": [],
    }
    monitor["subset_sha256"] = verifier._canonical_sha256(monitor)
    _write_json(root / "monitor_subset.json", monitor)

    full_report = _full_parameter_report()
    run_manifest = {
        "schema_version": 1,
        "config_sha256": verifier._sha256_file(config_path),
        "monitor_subset_sha256": monitor["subset_sha256"],
        "world_size": 1,
        "planned_total_steps": 3,
        "dataset_profile_sha256": verifier._sha256_file(profile_path),
        "stage": "S0-15",
        "job_name": "unit-s0",
        "dataset_root": str(dataset_root.resolve()),
        "dataset_size": 2,
        "wandb_run_id": "same-id",
        "wandb_mode": "online",
        "wandb_artifact_upload": False,
        "hub_upload": False,
        "full_parameter_report": full_report,
    }
    _write_json(root / "run_manifest.json", run_manifest)

    for step in (2, 3):
        checkpoint = root / "checkpoints" / f"step-{step:06d}"
        _write_json(
            checkpoint / "checkpoint_manifest.json",
            {
                "schema_version": 1,
                "step": step,
                "config_sha256": run_manifest["config_sha256"],
                "monitor_subset_sha256": monitor["subset_sha256"],
                "wandb_run_id": "same-id",
                "world_size": 1,
                "hub_upload": False,
                "wandb_artifact_upload": False,
                "atomic_publish": True,
            },
        )
    _write_json(root / "checkpoints/last_checkpoint.json", {"step": 3, "path": "checkpoints/step-000003"})

    checkpoint = root / "checkpoints/step-000003"
    state = checkpoint / "training_state"
    _write_json(
        state / "training_step.json",
        {"step": 3, "world_size": 1, "per_device_batch_size": 1, "gradient_accumulation_steps": 1},
    )
    for path in (
        state / verifier.OPTIMIZER_STATE,
        state / verifier.OPTIMIZER_PARAM_GROUPS,
        state / verifier.SCHEDULER_STATE,
        state / "rank-00" / verifier.RNG_STATE,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"state")

    model = checkpoint / "pretrained_model"
    files = {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "policy_preprocessor.json": json.dumps(
            {"steps": [{"registry_name": "normalizer_processor", "state_file": "pre-state.safetensors"}]}
        ).encode(),
        "policy_postprocessor.json": json.dumps(
            {"steps": [{"registry_name": "unnormalizer_processor", "state_file": "post-state.safetensors"}]}
        ).encode(),
        "franka_eef_geometry_manifest.json": b"{}",
        "pi0_eef_stats.json": b"{}",
    }
    for name, data in files.items():
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (model / "pre-state.safetensors").write_bytes(b"pre")
    (model / "post-state.safetensors").write_bytes(b"post")
    model_manifest = {
        "schema_version": 1,
        "weights_namespace": verifier.PI0_CORE_WEIGHTS_NAMESPACE,
        "hub_upload": False,
        "wandb_artifact_upload": False,
        "parameter_training": full_report,
        "training_graph": _training_graph(),
        "files": {
            name: {"size_bytes": len(data), "sha256": verifier._sha256_file(model / name)}
            for name, data in files.items()
        },
    }
    _write_json(model / "franka_pi0_checkpoint_manifest.json", model_manifest)

    reached_group = {
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
    vlm_group = {
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
    gradient = {
        "overall": {
            "trainable_tensors": verifier.EXPECTED_PARAMETER_TENSORS,
            "trainable_numel": verifier.EXPECTED_PARAMETER_COUNT,
            "gradient_tensors": (
                verifier.EXPECTED_PARAMETER_TENSORS
                - verifier.STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT
            ),
            "gradient_numel": (
                verifier.EXPECTED_PARAMETER_COUNT - verifier.STRUCTURAL_ACTION_UNREACHABLE_NUMEL
            ),
            "all_gradients_finite": True,
        },
        "groups": {
            "vision_encoder": dict(reached_group),
            "vlm": vlm_group,
            "action_expert": dict(reached_group),
            "state_action_projections": dict(reached_group),
        },
    }
    _write_json(root / "first_backward_gradient_coverage.json", gradient)
    invocations = (
        {"started_at": "1", "start_step": 0, "requested_stop_step": 2, "last_step": 2, "exit_code": 0, "resume": False},
        {"started_at": "2", "start_step": 2, "requested_stop_step": 3, "last_step": 3, "exit_code": 0, "resume": True},
    )
    for index, invocation in enumerate(invocations):
        _write_json(root / f"invocations/{index}.json", invocation)
    return root


def test_local_verifier_checks_complete_same_run_checkpoint_chain(tmp_path: Path) -> None:
    root = _build_complete_run(tmp_path)

    verified = verifier.verify_local_run(root)

    assert verified.report["local_status"] == "PASS"
    assert verified.report["published_checkpoint_steps"] == [2, 3]
    assert verified.report["wandb_run_id"] == "same-id"
    assert verified.report["training_graph"]["expert_lm_head_pruned"] is True


def test_local_verifier_rejects_staging_and_wandb_id_drift(tmp_path: Path) -> None:
    root = _build_complete_run(tmp_path)
    staging = root / "checkpoints/.step-000004.staging"
    staging.mkdir()
    with pytest.raises(verifier.VerificationError, match="staging"):
        verifier.verify_local_run(root)
    staging.rmdir()

    manifest_path = root / "checkpoints/step-000002/checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["wandb_run_id"] = "different-id"
    _write_json(manifest_path, manifest)
    with pytest.raises(verifier.VerificationError, match="W&B run id drift"):
        verifier.verify_local_run(root)


def test_saved_processor_reload_is_local_with_correct_converters_and_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    tokenizer = object()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    def fake_from_pretrained(cls, path, **kwargs):
        calls.append({"path": path, **kwargs})
        return f"processor-{len(calls)}"

    monkeypatch.setattr(
        verifier.PolicyProcessorPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    pre, post = verifier.load_saved_processors(tmp_path, torch.device("cuda:1"))

    assert (pre, post) == ("processor-1", "processor-2")
    assert calls[0]["local_files_only"] is True
    assert calls[0]["to_transition"] is verifier.batch_to_transition
    assert calls[0]["to_output"] is verifier.transition_to_batch
    assert calls[0]["overrides"]["tokenizer_processor"]["tokenizer"] is tokenizer
    assert calls[0]["overrides"]["normalizer_processor"]["device"] == "cuda:1"
    assert calls[1]["to_transition"] is verifier.policy_action_to_transition
    assert calls[1]["to_output"] is verifier.transition_to_policy_action
    assert calls[1]["overrides"]["unnormalizer_processor"]["device"] == "cuda:1"
    assert calls[1]["overrides"]["device_processor"]["device"] == "cpu"


def test_deep_verifier_processes_real_shaped_sample_and_finite_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = verifier.verify_local_run(_build_complete_run(tmp_path))

    class FakeDataset:
        effective_stats = {"observation.state": {}, "action": {}}

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 2

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
        lambda *args, **kwargs: (policy, None, None, {"pretrained": {"strict": True}}),
    )
    monkeypatch.setattr(verifier, "assert_full_parameter_training", lambda policy: _full_parameter_report())
    monkeypatch.setattr(verifier, "load_saved_processors", lambda *args: (lambda batch: batch, lambda action: action))

    report = verifier.verify_deep(local, device="cpu")

    assert report["deep_status"] == "PASS"
    assert report["state_shape"] == [1, 10]
    assert report["action_shape"] == [1, 50, 7]
    assert report["processor_roundtrip"] is True
    assert report["forward_loss"] == 1.0


def test_optional_wandb_check_is_read_only_and_requires_scalar_histories(tmp_path: Path) -> None:
    local = verifier.verify_local_run(_build_complete_run(tmp_path))

    class FakeRun:
        id = "same-id"
        state = "finished"

        def logged_artifacts(self):
            return []

        def scan_history(self, *, keys, page_size):
            metric = keys[1]
            steps = (0, 2, 3) if metric == "monitor/loss" else (1, 2, 3)
            return [{"trainer/step": step, metric: float(step) + 0.5} for step in steps]

    class FakeApi:
        def __init__(self):
            self.paths: list[str] = []

        def run(self, path):
            self.paths.append(path)
            return FakeRun()

    api = FakeApi()
    report = verifier.verify_wandb(local, api=api)

    assert api.paths == ["entity/project/same-id"]
    assert report["wandb_status"] == "PASS"
    assert report["logged_artifacts"] == 0
    assert set(report["history"]) == {"train/loss", "train/lr", "train/grad_norm", "monitor/loss"}
