from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from franka_eef_pipeline.direct_capability_eval import (
    compute_action_errors,
    fixed_noise_batch,
    split_supervision,
    summarize_m0_section_6_2,
)


def _actions(batch: int = 3, horizon: int = 50) -> np.ndarray:
    return np.zeros((batch, horizon, 7), dtype=np.float32)


def _load_evaluator(module_name: str) -> object:
    script = Path(__file__).resolve().parents[1] / "scripts/eval_pi0_direct_capability.py"
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)
    return evaluator


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_contract_fixture(
    evaluator: object,
    tmp_path: Path,
    *,
    profile_name: str,
    fps: int,
    task_instruction: str,
) -> tuple[Path, Path, object, dict[str, object]]:
    dataset_root = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoint"
    profile: dict[str, object] = {
        "schema_version": 1,
        "profile": profile_name,
        "dataset_name": f"synthetic_{profile_name}",
        "repo_id": f"local/synthetic_{profile_name}",
        "observation_fps": fps,
        "action_fps": fps,
        "chunk_size": 50,
        "episode_count": 2,
        "task_instruction": task_instruction,
        "main_action_key": "action",
        "requires_project_cartesian_adapter": True,
        "requires_project_dual_rate_adapter": False,
        "partial_conversion": False,
        "source_dataset_hash": _identity_hash("source"),
        "logical_anchor_index_sha256": _identity_hash("anchors"),
        "scope_content_sha256": _identity_hash("scope"),
        "derived_manifest_sha256": _identity_hash("manifest"),
        "conversion_config_sha256": _identity_hash("config"),
        "conversion_script_sha256": _identity_hash("script"),
        "target_grid": "next_native_camera_endpoint",
    }
    stats: dict[str, object] = {
        field: copy.deepcopy(profile[field]) for field in evaluator.STATS_IDENTITY_FIELDS
    }
    stats.update(
        {
            "observation.state": {
                "mean": [0.0] * 10,
                "std": [1.0] * 10,
                "count": [4],
            },
            "action": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "count": [200],
            },
        }
    )
    dataset_stats_path = dataset_root / "meta/pi0_eef_stats.json"
    _write_json(dataset_stats_path, stats)

    class FakeDataset:
        def __len__(self) -> int:
            return len(self.logical_anchors)

    dataset = FakeDataset()
    dataset.profile = profile_name
    dataset.task_instruction = task_instruction
    dataset.observation_fps = fps
    dataset.action_fps = fps
    dataset.chunk_size = 50
    dataset.num_episodes = 2
    dataset.logical_anchors = tuple(
        {
            "logical_index": index,
            "episode_index": index % 2,
            "raw_episode_id": f"episode-{index % 2}",
        }
        for index in range(4)
    )
    dataset.effective_stats = {
        "observation.state": copy.deepcopy(stats["observation.state"]),
        "action": copy.deepcopy(stats["action"]),
    }

    config = {
        "type": "pi0",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "input_features": {
            "observation.images.base_0_rgb": {"shape": [3, 224, 224]},
            "observation.images.left_wrist_0_rgb": {"shape": [3, 224, 224]},
            "observation.state": {"shape": [10]},
        },
        "output_features": {"action": {"shape": [7]}},
    }
    geometry = {
        "schema_version": 1,
        "dataset_profile": copy.deepcopy(profile),
        "task_instruction": task_instruction,
        "state10": evaluator.EXPECTED_STATE_CONVENTION,
        "action7": {
            "frequency_hz": fps,
            "chunk_size": 50,
            **copy.deepcopy(evaluator.EXPECTED_ACTION_CONVENTION),
        },
    }
    checkpoint_dir.mkdir(parents=True)
    _write_json(checkpoint_dir / "config.json", config)
    _write_json(checkpoint_dir / "franka_eef_geometry_manifest.json", geometry)
    _write_json(checkpoint_dir / "pi0_eef_stats.json", stats)
    (checkpoint_dir / "model.safetensors").write_bytes(b"synthetic-weights")
    _, stats_report = evaluator.prepare_effective_pi0_stats(stats)
    checkpoint_manifest = {
        "schema_version": 1,
        "checkpoint_type": "franka_pi0_full_parameter_eef",
        "source_load_report": {"base": {"effective_stats": stats_report}},
        "files": {
            filename: {
                "size_bytes": (checkpoint_dir / filename).stat().st_size,
                "sha256": _file_sha256(checkpoint_dir / filename),
            }
            for filename in (
                "config.json",
                "franka_eef_geometry_manifest.json",
                "pi0_eef_stats.json",
                "model.safetensors",
            )
        },
    }
    _write_json(checkpoint_dir / "franka_pi0_checkpoint_manifest.json", checkpoint_manifest)
    return dataset_root, checkpoint_dir, dataset, profile


def _refresh_checkpoint_ledger(checkpoint_dir: Path, filename: str) -> None:
    manifest_path = checkpoint_dir / "franka_pi0_checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = checkpoint_dir / filename
    manifest["files"][filename] = {
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }
    _write_json(manifest_path, manifest)


def test_split_supervision_removes_target_and_pad_without_mutating_batch() -> None:
    action = torch.ones((2, 50, 7))
    pad = torch.zeros((2, 50), dtype=torch.bool)
    batch = {"observation.state": torch.zeros((2, 10)), "action": action, "action_is_pad": pad}

    inputs, target, target_pad = split_supervision(batch)

    assert set(inputs) == {"observation.state"}
    assert "action" in batch and "action_is_pad" in batch
    assert target.data_ptr() != action.data_ptr()
    assert target_pad.data_ptr() != pad.data_ptr()
    action.zero_()
    assert torch.all(target == 1)


def test_fixed_noise_is_stable_per_logical_id_across_order_and_shards() -> None:
    whole = fixed_noise_batch(123, [8, 3, 19], (3, 4, 5), torch.float32, "cpu")
    reordered = fixed_noise_batch(123, [19, 8], (2, 4, 5), torch.float32, "cpu")
    singleton = fixed_noise_batch(123, [3], (4, 5), torch.float32, "cpu")

    assert torch.equal(whole[2], reordered[0])
    assert torch.equal(whole[0], reordered[1])
    assert torch.equal(whole[1], singleton[0])
    assert not torch.equal(whole, fixed_noise_batch(124, [8, 3, 19], whole.shape))

    # A batch size equal to the first per-sample dimension must not be
    # mistaken for an explicit batch dimension.
    fifty = fixed_noise_batch(123, list(range(50)), (50, 32))
    assert fifty.shape == (50, 50, 32)


def test_action_errors_use_so3_geodesic_not_rotvec_subtraction() -> None:
    prediction = _actions(batch=1)
    target = _actions(batch=1)
    prediction[..., 0] = 0.001
    prediction[..., 5] = math.radians(-179.0)
    target[..., 5] = math.radians(179.0)
    prediction[..., 6] = 0.51
    target[..., 6] = 0.49

    errors = compute_action_errors(prediction, target)

    assert errors["translation_mm"] == pytest.approx(np.ones((1, 50)))
    assert errors["rotation_geodesic_deg"] == pytest.approx(
        np.full((1, 50), 2.0), abs=2e-5
    )
    assert errors["gripper_abs"] == pytest.approx(np.full((1, 50), 0.02))


def test_section_6_2_ade_includes_waypoint_one_and_uses_fixed_horizons() -> None:
    target = _actions(batch=1)
    predictions = np.zeros((1, 1, 50, 7), dtype=np.float32)
    predictions[0, 0, 0, 0] = 0.050

    report = summarize_m0_section_6_2(
        predictions,
        target,
        np.zeros((1, 50), dtype=bool),
        [0],
        [1001000],
    )

    mean = report["mean"]
    assert mean["translation_ade_mm"] == pytest.approx(1.0)
    assert mean["translation_fde_mm"] == pytest.approx(0.0)
    assert mean["translation_horizon_mm"] == pytest.approx(
        {"1": 50.0, "5": 0.0, "10": 0.0, "25": 0.0, "50": 0.0}
    )
    assert set(mean) == {
        "translation_ade_mm",
        "translation_fde_mm",
        "translation_horizon_mm",
        "rotation_geodesic_ade_deg",
        "rotation_geodesic_fde_deg",
        "rotation_geodesic_horizon_deg",
        "gripper_mae",
        "gripper_fde_mae",
        "gripper_horizon_mae",
    }
    serialized = json.dumps(report, allow_nan=False)
    for forbidden in ("micro", "p50", "p90", "p95", "tolerance", "binary_accuracy"):
        assert forbidden not in serialized


def test_section_6_2_is_episode_macro_then_seed_mean_and_population_variance() -> None:
    target = _actions()
    predictions = np.zeros((2, 3, 50, 7), dtype=np.float32)
    predictions[0, :, :, 0] = np.asarray([0.001, 0.003, 0.010])[:, None]
    predictions[1, :, :, 0] = np.asarray([0.003, 0.005, 0.012])[:, None]
    predictions[0, :, :, 5] = math.radians(-179.0)
    predictions[1, :, :, 5] = math.radians(-177.0)
    target[:, :, 5] = math.radians(179.0)

    report = summarize_m0_section_6_2(
        predictions,
        target,
        np.zeros((3, 50), dtype=bool),
        [0, 0, 1],
        [1001000, 2001000],
    )

    # Seed 1 episode-macro translation = ((1 + 3) / 2 + 10) / 2 = 6 mm.
    # Seed 2 is 8 mm, so mean=7 and population variance=1.
    assert report["mean"]["translation_ade_mm"] == pytest.approx(7.0)
    assert report["mean"]["translation_fde_mm"] == pytest.approx(7.0)
    assert report["population_variance"]["translation_ade_mm"] == pytest.approx(1.0)
    assert report["population_variance"]["translation_fde_mm"] == pytest.approx(1.0)
    assert report["mean"]["rotation_geodesic_ade_deg"] == pytest.approx(3.0, abs=2e-5)
    assert report["population_variance"]["rotation_geodesic_ade_deg"] == pytest.approx(
        1.0, abs=2e-5
    )
    json.dumps(report, allow_nan=False)


def test_section_6_2_rejects_padded_chunks() -> None:
    prediction = _actions(batch=1)
    target = _actions(batch=1)
    pad = np.zeros((1, 50), dtype=bool)
    pad[:, -1] = True

    with pytest.raises(ValueError, match="does not admit padded"):
        summarize_m0_section_6_2(prediction[None], target, pad, [0], [1001000])


@pytest.mark.parametrize("which", ["prediction", "target"])
def test_action_error_validation_rejects_nonfinite(which: str) -> None:
    prediction = _actions(batch=1)
    target = _actions(batch=1)
    (prediction if which == "prediction" else target)[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_action_errors(prediction, target)


@pytest.mark.parametrize(
    ("profile_name", "fps", "task_instruction"),
    [
        ("action15", 15, "stack the cups"),
        ("native30", 30, "pick up the potato chip"),
    ],
)
def test_checkpoint_dataset_contract_accepts_supported_profiles(
    tmp_path: Path,
    profile_name: str,
    fps: int,
    task_instruction: str,
) -> None:
    evaluator = _load_evaluator(f"eval_pi0_contract_{profile_name}_test")
    dataset_root, checkpoint_dir, dataset, profile = _write_contract_fixture(
        evaluator,
        tmp_path,
        profile_name=profile_name,
        fps=fps,
        task_instruction=task_instruction,
    )

    report = evaluator._validate_checkpoint_dataset_contract(
        checkpoint_dir=checkpoint_dir,
        dataset_root=dataset_root,
        dataset=dataset,
        dataset_profile=profile,
    )

    assert report["verified"] is True
    assert report["verification_timing"] == "before_model_load"
    assert report["dataset_profile"] == profile_name
    assert report["task_instruction"] == task_instruction
    assert report["observation_hz"] == fps
    assert report["action_hz"] == fps
    assert report["anchor_count"] == 4
    assert report["episode_count"] == 2
    assert report["normalization"]["checkpoint_matches_dataset"] is True
    assert report["normalization"]["stats_files_byte_identical"] is True
    assert report["checkpoint_ledger"]["model.safetensors"]["size_verified"] is True
    assert report["checkpoint_ledger"]["model.safetensors"]["sha256_verified"] is False


def test_checkpoint_dataset_contract_rejects_identity_mismatch(tmp_path: Path) -> None:
    evaluator = _load_evaluator("eval_pi0_contract_identity_mismatch_test")
    dataset_root, checkpoint_dir, dataset, profile = _write_contract_fixture(
        evaluator,
        tmp_path,
        profile_name="native30",
        fps=30,
        task_instruction="pick up the potato chip",
    )
    geometry_path = checkpoint_dir / "franka_eef_geometry_manifest.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    geometry["dataset_profile"]["source_dataset_hash"] = "0" * 64
    _write_json(geometry_path, geometry)
    _refresh_checkpoint_ledger(checkpoint_dir, geometry_path.name)

    with pytest.raises(evaluator.DirectEvaluationError, match="source_dataset_hash"):
        evaluator._validate_checkpoint_dataset_contract(
            checkpoint_dir=checkpoint_dir,
            dataset_root=dataset_root,
            dataset=dataset,
            dataset_profile=profile,
        )


def test_checkpoint_dataset_contract_rejects_normalization_mismatch(tmp_path: Path) -> None:
    evaluator = _load_evaluator("eval_pi0_contract_stats_mismatch_test")
    dataset_root, checkpoint_dir, dataset, profile = _write_contract_fixture(
        evaluator,
        tmp_path,
        profile_name="native30",
        fps=30,
        task_instruction="pick up the potato chip",
    )
    stats_path = checkpoint_dir / "pi0_eef_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["action"]["mean"][0] = 0.25
    _write_json(stats_path, stats)
    _refresh_checkpoint_ledger(checkpoint_dir, stats_path.name)

    with pytest.raises(evaluator.DirectEvaluationError, match="normalization stats differ"):
        evaluator._validate_checkpoint_dataset_contract(
            checkpoint_dir=checkpoint_dir,
            dataset_root=dataset_root,
            dataset=dataset,
            dataset_profile=profile,
        )


def test_checkpoint_dataset_contract_rejects_population_count_mismatch(tmp_path: Path) -> None:
    evaluator = _load_evaluator("eval_pi0_contract_count_mismatch_test")
    dataset_root, checkpoint_dir, dataset, profile = _write_contract_fixture(
        evaluator,
        tmp_path,
        profile_name="native30",
        fps=30,
        task_instruction="pick up the potato chip",
    )
    dataset.logical_anchors = dataset.logical_anchors[:3]

    with pytest.raises(evaluator.DirectEvaluationError, match="stats count.*anchor count"):
        evaluator._validate_checkpoint_dataset_contract(
            checkpoint_dir=checkpoint_dir,
            dataset_root=dataset_root,
            dataset=dataset,
            dataset_profile=profile,
        )


@pytest.mark.parametrize(
    ("profile_name", "fps", "task_instruction", "anchor_count", "episode_count"),
    [
        ("action15", 15, "stack the cups", 7, 2),
        ("native30", 30, "pick up the potato chip", 11, 3),
    ],
)
def test_main_discovers_dataset_contract_and_emits_only_m0_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
    fps: int,
    task_instruction: str,
    anchor_count: int,
    episode_count: int,
) -> None:
    evaluator = _load_evaluator(f"eval_pi0_direct_capability_main_{profile_name}_test")
    required = [
        "--checkpoint-dir",
        str(tmp_path),
        "--dataset-root",
        str(tmp_path),
        "--output-json",
        str(tmp_path / "result.json"),
    ]
    for removed_option in ("--base-checkpoint-dir", "--max-anchors-per-episode"):
        with pytest.raises(SystemExit):
            evaluator._parse_args([*required, removed_option, "1"])

    current_profile = profile_name
    current_task = task_instruction
    current_fps = fps
    current_episode_count = episode_count
    current_anchor_count = anchor_count

    class FakeDataset:
        effective_stats: dict[str, object] = {}
        profile = current_profile
        task_instruction = current_task
        observation_fps = current_fps
        action_fps = current_fps
        chunk_size = 50
        num_episodes = current_episode_count
        logical_anchors = tuple(
            {
                "logical_index": index,
                "episode_index": index % current_episode_count,
                "raw_episode_id": f"episode-{index % current_episode_count}",
            }
            for index in range(current_anchor_count)
        )

        def __init__(self, *_args: object, **kwargs: object) -> None:
            assert kwargs["profile"] == profile_name

        def __len__(self) -> int:
            return len(self.logical_anchors)

    output_json = tmp_path / "result.json"
    args = SimpleNamespace(
        checkpoint_dir=tmp_path,
        dataset_root=tmp_path,
        output_json=output_json,
        prediction_seeds=[1_001_000, 2_001_000, 3_001_000],
        batch_size=4,
        device="cpu",
        workers=0,
        num_inference_steps=10,
        save_predictions=False,
    )
    calls: list[str] = []

    def fake_infer(**kwargs: object) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        calls.append(str(kwargs["label"]))
        assert kwargs["expected_task"] == task_instruction
        assert kwargs["expected_effective_stats_sha256"] == "f" * 64
        return (
            {
                "label": kwargs["label"],
                "target_action_provided_to_model": False,
                "policy_eval_called": True,
            },
            {
                "predictions": np.zeros((3, 1, 50, 7), dtype=np.float32),
                "targets": np.zeros((1, 50, 7), dtype=np.float32),
                "pads": np.zeros((1, 50), dtype=bool),
                "states": np.zeros((1, 10), dtype=np.float32),
                "episode_indices": np.zeros(1, dtype=np.int64),
            },
        )

    def metric_tree() -> dict[str, object]:
        horizons = {str(horizon): 0.0 for horizon in (1, 5, 10, 25, 50)}
        return {
            "translation_ade_mm": 0.0,
            "translation_fde_mm": 0.0,
            "translation_horizon_mm": horizons,
            "rotation_geodesic_ade_deg": 0.0,
            "rotation_geodesic_fde_deg": 0.0,
            "rotation_geodesic_horizon_deg": horizons,
            "gripper_mae": 0.0,
            "gripper_fde_mae": 0.0,
            "gripper_horizon_mae": horizons,
        }

    monkeypatch.setattr(evaluator, "_parse_args", lambda _argv: args)
    monkeypatch.setattr(evaluator, "_validate_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(evaluator, "_resolve_checkpoint_dir", lambda value: Path(value))
    monkeypatch.setattr(
        evaluator,
        "load_cartesian_profile",
        lambda _root: {
            "profile": profile_name,
            "task_instruction": task_instruction,
            "source_dataset_hash": "a" * 64,
            "logical_anchor_index_sha256": "b" * 64,
            "scope_content_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(evaluator, "CartesianAnchorDataset", FakeDataset)
    monkeypatch.setattr(
        evaluator,
        "_validate_checkpoint_dataset_contract",
        lambda **_kwargs: {
            "verified": True,
            "task_instruction": task_instruction,
            "normalization": {"effective_stats_sha256": "f" * 64},
        },
    )
    monkeypatch.setattr(evaluator, "_infer_checkpoint", fake_infer)
    monkeypatch.setattr(
        evaluator,
        "_aggregate_model_metrics",
        lambda *_args: {"mean": metric_tree(), "population_variance": metric_tree()},
    )

    assert evaluator.main([]) == 0
    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert calls == ["trained"]
    assert result["status"] == "M0_SECTION_6_2_EVALUATION_COMPLETE"
    assert set(result).isdisjoint({"models", "baselines", "comparisons"})
    assert set(result["metrics"]) == {"mean", "population_variance"}
    assert result["dataset"]["profile"] == profile_name
    assert result["dataset"]["observation_hz"] == fps
    assert result["dataset"]["action_hz"] == fps
    assert result["dataset"]["selected_anchor_count"] == anchor_count
    assert result["dataset"]["selected_episode_count"] == episode_count
    assert result["model_input_contract"]["language_instruction"] == task_instruction
    assert result["metric_timebase"]["horizon_offsets_seconds"]["50"] == pytest.approx(
        49 / fps
    )
    markdown = output_json.with_suffix(".md").read_text(encoding="utf-8")
    assert task_instruction in markdown
    if profile_name == "native30":
        for stale_text in ("stack the cups", "22-episode", "cup-stacking"):
            assert stale_text not in markdown


@pytest.mark.parametrize("task_instruction", ["stack the cups", "pick up the potato chip"])
def test_direct_cli_inference_hides_target_from_preprocessor_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_instruction: str,
) -> None:
    module_suffix = "chips" if "chip" in task_instruction else "cups"
    evaluator = _load_evaluator(f"eval_pi0_direct_capability_inference_{module_suffix}_test")
    placeholders: dict[str, object] = {
        "observation.state": torch.zeros(1, 10),
        "action": None,
        "action_is_pad": None,
    }
    evaluator._strip_empty_supervision_placeholders(placeholders)
    assert set(placeholders) == {"observation.state"}
    for key in ("action", "action_is_pad"):
        leaked = {key: torch.ones(1)}
        with pytest.raises(evaluator.DirectEvaluationError, match="reintroduced target"):
            evaluator._strip_empty_supervision_placeholders(leaked)

    current_task = task_instruction

    class FakeDataset(torch.utils.data.Dataset):
        effective_stats: dict[str, object] = {}
        logical_anchors = (
            {"logical_index": 101, "episode_index": 0, "raw_episode_id": "episode-a"},
            {"logical_index": 205, "episode_index": 1, "raw_episode_id": "episode-b"},
        )

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict[str, object]:
            action = torch.zeros((50, 7), dtype=torch.float32)
            action[:, 0] = float(index + 1) / 1000.0
            return {
                "observation.state": torch.zeros(10, dtype=torch.float32),
                "observation.images.base_0_rgb": torch.full(
                    (3, 4, 4), index, dtype=torch.uint8
                ),
                "observation.images.left_wrist_0_rgb": torch.full(
                    (3, 4, 4), index + 1, dtype=torch.uint8
                ),
                "task": current_task,
                "action": action,
                "action_is_pad": torch.zeros(50, dtype=torch.bool),
            }

    class FakePolicy:
        training = True
        calls = 0

        def eval(self) -> "FakePolicy":
            self.training = False
            return self

        def predict_action_chunk(
            self, inputs: dict[str, object], *, noise: torch.Tensor, num_steps: int
        ) -> torch.Tensor:
            assert not self.training
            assert num_steps == 3
            assert "action" not in inputs and "action_is_pad" not in inputs
            assert inputs["observation.images.base_0_rgb"].dtype == torch.float32
            self.calls += 1
            return torch.zeros((noise.shape[0], 50, 7), dtype=torch.float32)

    policy = FakePolicy()

    def preprocessor(inputs: dict[str, object]) -> dict[str, object]:
        assert "action" not in inputs and "action_is_pad" not in inputs
        # LeRobot's canonical transition_to_batch converter always returns an
        # action key, using None when target supervision was absent.
        return {**inputs, "action": None}

    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"fake")

    def fake_loader(*_args: object, **_kwargs: object) -> tuple[object, object, object, dict]:
        return (
            policy,
            preprocessor,
            lambda value: value,
            {
                "pretrained": {
                    "strict": True,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "weights_path": str(weights),
                    "project_manifest_present": True,
                    "verified_tensor": {"exact_after_model_dtype_cast": True},
                },
                "effective_stats": {"effective_stats_sha256": "f" * 64},
                "parameters": {
                    "trainable_parameters": 123,
                    "parameter_tensor_count": 4,
                },
                "config": {"use_peft": False},
            },
        )

    monkeypatch.setattr(evaluator, "load_pi0_full_policy_and_processors", fake_loader)
    report, arrays = evaluator._infer_checkpoint(
        label="fake",
        checkpoint_dir=tmp_path,
        dataset=FakeDataset(),
        device=torch.device("cpu"),
        batch_size=2,
        workers=0,
        prediction_seeds=[1000, 2000],
        num_inference_steps=3,
        expected_task=task_instruction,
        expected_effective_stats_sha256="f" * 64,
    )

    assert report["target_action_provided_to_model"] is False
    assert report["policy_eval_called"] is True
    assert arrays["predictions"].shape == (2, 2, 50, 7)
    assert arrays["targets"][0, 0, 0] == pytest.approx(0.001)
    assert arrays["targets"][1, 0, 0] == pytest.approx(0.002)
    assert policy.calls == 2


def test_language_validation_rejects_task_drift() -> None:
    evaluator = _load_evaluator("eval_pi0_direct_capability_task_drift_test")

    evaluator._validate_language({"task": ["pick up the potato chip"]}, 1, "pick up the potato chip")
    with pytest.raises(evaluator.DirectEvaluationError, match="language instruction drifted"):
        evaluator._validate_language({"task": ["stack the cups"]}, 1, "pick up the potato chip")
