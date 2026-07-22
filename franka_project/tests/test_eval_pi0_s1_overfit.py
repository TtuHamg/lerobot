from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/eval_pi0_s1_overfit.py"
SPEC = importlib.util.spec_from_file_location("eval_pi0_s1_overfit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_s1_run(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    root = tmp_path / "s1-run"
    dataset_root = tmp_path / "dataset"
    base_model = tmp_path / "pi0-base"
    base_model.mkdir()
    profile_path = dataset_root / "meta/franka_eef_profile.json"
    stats_path = dataset_root / "meta/pi0_eef_stats.json"
    _write_json(
        profile_path,
        {
            "profile": "action15",
            "observation_fps": 15,
            "action_fps": 15,
            "chunk_size": 50,
        },
    )
    _write_json(stats_path, {"observation.state": {}, "action": {}})

    config = {
        "stage": "S1-15",
        "job_name": "unit-s1",
        "dataset": {
            "root": str(dataset_root),
            "profile": "action15",
            "observation_fps": 15,
            "action_fps": 15,
            "chunk_size": 50,
            "episode_indices": [0, 1],
            "max_anchors_per_episode": 1,
        },
        "model": {
            "pretrained_path": str(base_model),
            "strict": True,
            "max_action_dim": 32,
            "peft": None,
        },
        "training": {"steps": 300, "seed": 1000},
    }
    config_path = root / "resolved_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    anchors = [
        {
            "logical_index": 10,
            "episode_index": 0,
            "raw_episode_id": "episode-a",
            "anchor_local_index": 10,
            "main_global_row": 10,
            "camera_log_time_ns": 100,
        },
        {
            "logical_index": 20,
            "episode_index": 1,
            "raw_episode_id": "episode-b",
            "anchor_local_index": 5,
            "main_global_row": 20,
            "camera_log_time_ns": 200,
        },
    ]
    monitor = {
        "schema_version": 1,
        "source": "training_data",
        "held_out": False,
        "seed": 1017,
        "population_size": 2,
        "sample_size": 2,
        "indices": [0, 1],
        "anchors": anchors,
        "includes_converter_frozen_indices": [0, 1],
    }
    monitor["subset_sha256"] = evaluator._canonical_sha256(monitor)
    _write_json(root / "monitor_subset.json", monitor)

    full_report = {
        "total_parameters": evaluator.EXPECTED_PARAMETER_COUNT,
        "trainable_parameters": evaluator.EXPECTED_PARAMETER_COUNT,
        "parameter_tensor_count": evaluator.EXPECTED_PARAMETER_TENSORS,
        "trainable_fraction": 1.0,
        "lora_parameter_count": 0,
        "use_peft": False,
        "training_graph": {"schema_version": 2},
    }
    run_manifest = {
        "config_sha256": evaluator._sha256_file(config_path),
        "dataset_profile_sha256": evaluator._sha256_file(profile_path),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_size": 2,
        "full_parameter_report": full_report,
        "monitor_subset_sha256": monitor["subset_sha256"],
        "planned_total_steps": 300,
        "stage": "S1-15",
        "wandb_run_id": "unit-run",
        "world_size": 1,
    }
    _write_json(root / "run_manifest.json", run_manifest)

    checkpoint = root / "checkpoints/step-000300"
    _write_json(root / "checkpoints/last_checkpoint.json", {"step": 300, "path": "checkpoints/step-000300"})
    _write_json(
        checkpoint / "checkpoint_manifest.json",
        {
            "step": 300,
            "atomic_publish": True,
            "config_sha256": run_manifest["config_sha256"],
            "monitor_subset_sha256": monitor["subset_sha256"],
            "wandb_run_id": "unit-run",
            "world_size": 1,
        },
    )
    _write_json(checkpoint / "training_state/training_step.json", {"step": 300})
    _write_json(
        root / "invocations/complete.json",
        {"start_step": 0, "requested_stop_step": 300, "last_step": 300, "exit_code": 0},
    )

    model_dir = checkpoint / "pretrained_model"
    files = {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "pi0_eef_stats.json": stats_path.read_bytes(),
        "policy_preprocessor.json": json.dumps(
            {"steps": [{"registry_name": "normalizer_processor", "state_file": "pre.safetensors"}]}
        ).encode(),
        "policy_postprocessor.json": json.dumps(
            {"steps": [{"registry_name": "unnormalizer_processor", "state_file": "post.safetensors"}]}
        ).encode(),
    }
    for name, content in files.items():
        path = model_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (model_dir / "pre.safetensors").write_bytes(b"pre")
    (model_dir / "post.safetensors").write_bytes(b"post")
    _write_json(
        model_dir / "franka_pi0_checkpoint_manifest.json",
        {
            "checkpoint_type": "franka_pi0_full_parameter_eef",
            "weights_namespace": evaluator.PI0_CORE_WEIGHTS_NAMESPACE,
            "parameter_training": full_report,
            "training_graph": full_report["training_graph"],
            "files": {
                name: {
                    "size_bytes": len(content),
                    "sha256": evaluator._sha256_file(model_dir / name),
                }
                for name, content in files.items()
            },
        },
    )
    return root, anchors


def test_metrics_use_so3_geodesic_and_episode_macro_aggregation() -> None:
    prediction = torch.zeros(3, 50, 7)
    target = torch.zeros_like(prediction)
    prediction[0, :, 0] = 0.001
    prediction[1, :, 0] = 0.003
    prediction[2, :, 0] = 0.010
    prediction[:, :, 5] = math.radians(-179.0)
    target[:, :, 5] = math.radians(179.0)
    prediction[0, :, 6] = 0.1
    prediction[1, :, 6] = 0.3
    prediction[2, :, 6] = 0.9
    anchors = [
        {"episode_index": 0, "raw_episode_id": "a"},
        {"episode_index": 0, "raw_episode_id": "a"},
        {"episode_index": 1, "raw_episode_id": "b"},
    ]

    report = evaluator.compute_action_metrics(prediction, target, [2, 4, 8], anchors)

    assert report["aggregation"] == "episode_macro_of_anchor_horizon_means"
    assert report["aggregate"]["translation_ade_mm"] == pytest.approx(6.0)
    assert report["aggregate"]["rotation_ade_deg"] == pytest.approx(2.0, abs=1e-5)
    assert report["aggregate"]["gripper_mae"] == pytest.approx(0.55)
    assert [item["anchor_count"] for item in report["per_episode"]] == [2, 1]


def test_metrics_and_ratios_fail_closed_on_shape_nonfinite_and_zero_base() -> None:
    ratios = evaluator.trained_over_base_ratios(
        {"translation_ade_mm": 10.0, "rotation_ade_deg": 5.0, "gripper_mae": 0.2},
        {"translation_ade_mm": 5.0, "rotation_ade_deg": 2.5, "gripper_mae": 0.1},
    )
    assert ratios == {
        "translation_ade_mm": 0.5,
        "rotation_ade_deg": 0.5,
        "gripper_mae": 0.5,
    }
    with pytest.raises(evaluator.S1EvaluationError, match="base gripper_mae is zero"):
        evaluator.trained_over_base_ratios(
            {"translation_ade_mm": 1.0, "rotation_ade_deg": 1.0, "gripper_mae": 0.0},
            {"translation_ade_mm": 1.0, "rotation_ade_deg": 1.0, "gripper_mae": 0.0},
        )

    target = torch.zeros(1, 50, 7)
    prediction = target.clone()
    prediction[0, 0, 0] = torch.nan
    with pytest.raises(evaluator.S1EvaluationError, match="NaN/Inf"):
        evaluator.compute_action_metrics(
            prediction,
            target,
            [0],
            [{"episode_index": 0, "raw_episode_id": "a"}],
        )
    with pytest.raises(evaluator.S1EvaluationError, match="shape"):
        evaluator.compute_action_metrics(
            torch.zeros(1, 49, 7),
            target,
            [0],
            [{"episode_index": 0, "raw_episode_id": "a"}],
        )


def test_fixed_noise_is_per_index_deterministic_float32_and_padded() -> None:
    seed_a, first, digest_a = evaluator.fixed_noise_cpu(1000, 9)
    seed_b, second, digest_b = evaluator.fixed_noise_cpu(1000, 9)
    _, different, digest_different = evaluator.fixed_noise_cpu(1000, 11)

    assert seed_a == seed_b == 101009
    assert first.dtype is torch.float32
    assert tuple(first.shape) == (1, 50, 32)
    assert torch.equal(first, second)
    assert digest_a == digest_b
    assert not torch.equal(first, different)
    assert digest_a != digest_different


def test_local_s1_contract_accepts_complete_run_and_rejects_drift(tmp_path: Path) -> None:
    root, _ = _build_s1_run(tmp_path)

    verified = evaluator.verify_s1_run(root)

    assert verified.checkpoint.name == "step-000300"
    assert verified.monitor["indices"] == [0, 1]
    assert verified.stats_file_sha256 == evaluator._sha256_file(
        Path(verified.dataset_root) / "meta/pi0_eef_stats.json"
    )

    monitor_path = root / "monitor_subset.json"
    monitor = json.loads(monitor_path.read_text())
    monitor["indices"] = [1, 0]
    _write_json(monitor_path, monitor)
    with pytest.raises(evaluator.S1EvaluationError, match="monitor canonical SHA mismatch"):
        evaluator.verify_s1_run(root)


def test_local_s1_contract_rejects_model_file_sha_drift(tmp_path: Path) -> None:
    root, _ = _build_s1_run(tmp_path)
    model_dir = root / "checkpoints/step-000300/pretrained_model"
    config_path = model_dir / "config.json"
    # Preserve the byte count so this specifically exercises the digest gate.
    config_path.write_bytes(b"[]")

    with pytest.raises(evaluator.S1EvaluationError, match="model file SHA256 mismatch"):
        evaluator.verify_s1_run(root)


def test_full_evaluator_uses_fixed_noise_postprocess_and_saved_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, anchors = _build_s1_run(tmp_path)
    load_kinds: list[str] = []
    prediction_calls: dict[str, list[dict[str, Any]]] = {"base": [], "trained": []}
    saved_processor_calls: list[tuple[Path, torch.device]] = []

    class FakeDataset:
        profile = "action15"
        observation_fps = 15
        action_fps = 15
        chunk_size = 50
        effective_stats = {"observation.state": {}, "action": {}}
        logical_anchors = tuple(anchors)

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                "observation.images.base_0_rgb": torch.zeros(3, 2, 2, dtype=torch.uint8),
                "observation.images.left_wrist_0_rgb": torch.full((3, 2, 2), 255, dtype=torch.uint8),
                "observation.state": torch.zeros(10, dtype=torch.float32),
                "action": torch.zeros(50, 7, dtype=torch.float32),
                "task": "stack the cups",
            }

    class FakePolicy:
        def __init__(self, kind: str):
            self.kind = kind
            self.config = SimpleNamespace(chunk_size=50, max_action_dim=32)

        def eval(self):
            return self

        def predict_action_chunk(self, batch, **kwargs):
            prediction_calls[self.kind].append(
                {
                    "noise": kwargs["noise"].detach().cpu().clone(),
                    "num_steps": kwargs["num_steps"],
                }
            )
            scale = 1.0 if self.kind == "base" else 0.5
            prediction = batch["action"].clone()
            prediction[..., 0] += 0.010 * scale
            prediction[..., 5] += 0.200 * scale
            prediction[..., 6] += 0.400 * scale
            return prediction

    def identity(value):
        return value

    def fake_loader(path, stats, device):
        kind = "base" if Path(path).name == "pi0-base" else "trained"
        load_kinds.append(kind)
        project = kind == "trained"
        report = {
            "pretrained": {
                "strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "project_manifest_present": project,
                "weights_namespace": evaluator.PI0_CORE_WEIGHTS_NAMESPACE,
                "weights_path": str(Path(path) / "model.safetensors"),
                "weights_size_bytes": 7,
                "verified_tensor": {
                    "exact_after_model_dtype_cast": True,
                    "checkpoint_key": "action_in_proj.bias",
                    "checkpoint_tensor_sha256": "f" * 64,
                },
            },
            "effective_stats": {"effective_stats_sha256": "e" * 64},
        }
        return FakePolicy(kind), identity, identity, report

    def fake_saved_processors(path, device):
        saved_processor_calls.append((Path(path), device))
        return identity, identity

    monkeypatch.setattr(evaluator, "CartesianAnchorDataset", FakeDataset)
    monkeypatch.setattr(evaluator, "load_pi0_full_policy_and_processors", fake_loader)
    monkeypatch.setattr(evaluator, "load_saved_processors", fake_saved_processors)

    report = evaluator.evaluate_s1_run(root, device="cpu")

    assert report["status"] == "EVALUATION_COMPLETE"
    assert report["quality_threshold_applied"] is False
    assert load_kinds == ["base", "trained", "trained"]
    assert len(saved_processor_calls) == 1
    assert report["metrics"]["trained_over_base_ratio"] == pytest.approx(
        {
            "translation_ade_mm": 0.5,
            "rotation_ade_deg": 0.5,
            "gripper_mae": 0.5,
        }
    )
    assert report["reload_consistency"]["allclose"] is True
    assert report["reload_consistency"]["normalized_action_max_abs_diff"] == 0.0
    assert all(call["num_steps"] == 10 for calls in prediction_calls.values() for call in calls)
    assert all(tuple(call["noise"].shape) == (1, 50, 32) for calls in prediction_calls.values() for call in calls)
    for sample_index in range(2):
        assert torch.equal(
            prediction_calls["base"][sample_index]["noise"],
            prediction_calls["trained"][sample_index]["noise"],
        )
        assert torch.equal(
            prediction_calls["trained"][sample_index]["noise"],
            prediction_calls["trained"][sample_index + 2]["noise"],
        )


def test_reload_consistency_and_json_writer_fail_closed(tmp_path: Path) -> None:
    actions = torch.zeros(1, 50, 7)
    first = evaluator.PredictionSet((0,), actions, actions, actions)
    changed = actions.clone()
    changed[0, 0, 0] = 0.1
    second = evaluator.PredictionSet((0,), changed, changed, actions)
    with pytest.raises(evaluator.S1EvaluationError, match="reload predictions differ"):
        evaluator._verify_reload_consistency(first, second)

    output = tmp_path / "nested/result.json"
    evaluator._write_output_json(output, {"finite": 1.0})
    assert json.loads(output.read_text()) == {"finite": 1.0}
    with pytest.raises(ValueError, match="Out of range float values"):
        evaluator._write_output_json(output, {"not_finite": float("nan")})
