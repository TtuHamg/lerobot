from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/eval_pi0_f0_m0.py"
SPEC = importlib.util.spec_from_file_location("eval_pi0_f0_m0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


def _stats() -> dict[str, Any]:
    return {
        "action": {
            "min": [-10.0] * 7,
            "max": [10.0] * 7,
            "q01": [-5.0] * 7,
            "q99": [5.0] * 7,
        },
        "observation.state": {
            "min": [-10.0] * 10,
            "max": [10.0] * 10,
        },
    }


def test_geometry_horizons_fde_and_episode_macro_are_exact() -> None:
    prediction = np.zeros((3, 50, 7), dtype=np.float32)
    target = np.zeros_like(prediction)
    prediction[0, :, 0] = 0.001
    prediction[1, :, 0] = 0.003
    prediction[2, :, 0] = 0.010
    prediction[:, :, 5] = math.radians(-179.0)
    target[:, :, 5] = math.radians(179.0)
    prediction[0, :, 6] = 0.1
    prediction[1, :, 6] = 0.3
    prediction[2, :, 6] = 0.9
    states = np.zeros((3, 10), dtype=np.float32)
    episodes = np.asarray([0, 0, 1], dtype=np.int64)

    report = evaluator.compute_reconstruction_metrics(
        prediction, target, states, episodes, {0: "episode-a", 1: "episode-b"}, _stats()
    )

    assert report["aggregation"]["primary"] == "episode_macro"
    error = report["episode_macro"]["error"]
    assert error["translation_ade_mm"] == pytest.approx(6.0)
    assert error["translation_fde_mm"] == pytest.approx(6.0)
    assert error["translation_horizon_mm"] == pytest.approx(
        {"1": 6.0, "5": 6.0, "10": 6.0, "25": 6.0, "50": 6.0}
    )
    assert error["rotation_geodesic_ade_deg"] == pytest.approx(2.0, abs=2e-5)
    assert error["rotation_geodesic_fde_deg"] == pytest.approx(2.0, abs=2e-5)
    assert error["gripper_mae"] == pytest.approx(0.55)
    assert [item["metrics"]["counts"]["anchors"] for item in report["per_episode"]] == [2, 1]

    # The pooled result intentionally differs, proving the primary metric does
    # not let the longer episode dominate.
    assert report["overall_anchor_micro"]["error"]["translation_ade_mm"] == pytest.approx(14.0 / 3.0)


def test_nonfinite_safety_workspace_and_smoothness_are_reported_without_crashing() -> None:
    prediction = np.zeros((1, 50, 7), dtype=np.float32)
    target = np.zeros_like(prediction)
    states = np.zeros((1, 10), dtype=np.float32)
    prediction[0, 0, 0] = np.nan
    prediction[0, 1, 0] = 20.0
    prediction[0, 2, 3] = math.pi + 0.1
    prediction[0, 3, 6] = 1.2

    metrics = evaluator._prediction_metrics(prediction, target, states, _stats())

    assert metrics["nonfinite"]["action_element_count"] == 1
    assert metrics["nonfinite"]["waypoint_any_count"] == 1
    assert metrics["action_envelope"]["outside_train_min_max_waypoint_any_fraction"] == pytest.approx(2 / 50)
    assert metrics["action_envelope"]["rotvec_norm_gt_pi_waypoint_fraction"] == pytest.approx(1 / 50)
    assert metrics["action_envelope"]["gripper_outside_0_1_waypoint_fraction"] == pytest.approx(1 / 50)
    assert metrics["workspace_training_envelope"][
        "decoded_xyz_outside_train_state_min_max_fraction"
    ] == pytest.approx(2 / 50)
    assert metrics["error"]["translation_ade_mm"] == pytest.approx(20_000.0 / 49.0)
    assert metrics["smoothness"]["adjacent_translation_step_mean_mm"] is not None


def test_prediction_seed_mean_and_population_variance_preserve_episode_details() -> None:
    target = np.zeros((1, 50, 7), dtype=np.float32)
    states = np.zeros((1, 10), dtype=np.float32)
    episodes = np.asarray([0], dtype=np.int64)
    records = []
    for seed, delta_mm in ((11, 1.0), (22, 3.0)):
        prediction = target.copy()
        prediction[..., 0] = delta_mm / 1000.0
        records.append(
            {
                "prediction_seed": seed,
                "metrics": evaluator.compute_reconstruction_metrics(
                    prediction, target, states, episodes, {0: "episode-a"}, _stats()
                ),
            }
        )

    summary = evaluator.summarize_prediction_seeds(records)

    translation = summary["episode_macro"]["mean"]["error"]["translation_ade_mm"]
    variance = summary["episode_macro"]["population_variance"]["error"]["translation_ade_mm"]
    assert translation == pytest.approx(2.0)
    assert variance == pytest.approx(1.0)
    assert summary["per_episode"][0]["raw_episode_id"] == "episode-a"
    assert summary["per_episode"][0]["mean"]["error"]["translation_fde_mm"] == pytest.approx(2.0)


def test_monitor_parser_rejects_conflicts_and_selects_only_eligible_checkpoints(tmp_path: Path) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text("x step=0 monitor_loss=0.9\nx step=2 monitor_loss=0.8\n", encoding="utf-8")
    second.write_text(
        "x step=10 monitor_loss=0.4\nx step=20 monitor_loss=0.2\nx step=30 monitor_loss=0.3\n",
        encoding="utf-8",
    )

    losses = evaluator.parse_monitor_losses([first, second], expected_steps={0, 2, 10, 20, 30})
    selection = evaluator.select_best_monitor_checkpoint(losses, eligible_steps=(10, 20, 30))
    assert selection["best_train_monitor_step"] == 20
    assert selection["best_train_monitor_loss"] == pytest.approx(0.2)
    assert next(item for item in selection["records"] if item["step"] == 2)[
        "eligible_for_best_train_monitor"
    ] is False

    second.write_text(second.read_text() + "x step=20 monitor_loss=0.25\n", encoding="utf-8")
    with pytest.raises(evaluator.M0EvaluationError, match="conflicting duplicate"):
        evaluator.parse_monitor_losses([first, second], expected_steps={0, 2, 10, 20, 30})


def _prepared(tmp_path: Path) -> Any:
    contract = {
        "sharding": {"num_shards": 2},
        "inference": {"prediction_seeds": [101, 202]},
        "monitor_cartesian": {
            "checkpoint_steps": [10],
            "logical_indices": [0],
            "prediction_seed": 101,
        },
    }
    return evaluator.PreparedM0(
        local=None,
        contract=contract,
        contract_sha256=evaluator._canonical_sha256(contract),
        work_dir=tmp_path,
    )


def test_shard_publish_resume_hash_and_contract_fail_closed(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    indices = evaluator.shard_indices(evaluator.DATASET_SIZE, 2, 0)
    count = len(indices)
    arrays = {
        "logical_indices": indices,
        "episode_indices": np.zeros(count, dtype=np.int64),
        "states": np.zeros((count, 10), dtype=np.float32),
        "targets": np.zeros((count, 50, 7), dtype=np.float32),
        "predictions": np.zeros((2, count, 50, 7), dtype=np.float32),
        "prediction_seeds": np.asarray([101, 202], dtype=np.int64),
    }

    manifest = evaluator._publish_shard(
        prepared,
        0,
        arrays,
        strict_load={"strict": True},
        processor_check={"action_roundtrip": True},
        evaluation_seconds=1.0,
    )
    loaded, resumed = evaluator._load_verified_shard(prepared, 0)
    assert manifest["npz_sha256"] == resumed["npz_sha256"]
    assert np.array_equal(loaded["logical_indices"], indices)
    with pytest.raises(evaluator.M0EvaluationError, match="overwrite"):
        evaluator._publish_shard(
            prepared,
            0,
            arrays,
            strict_load={"strict": True},
            processor_check={"action_roundtrip": True},
            evaluation_seconds=1.0,
        )

    npz_path, _manifest_path = evaluator._shard_paths(tmp_path, 0)
    with npz_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(evaluator.M0EvaluationError, match="size mismatch"):
        evaluator._load_verified_shard(prepared, 0)

    monitor_actions = np.zeros((1, 50, 7), dtype=np.float32)
    evaluator._publish_monitor_checkpoint(
        prepared,
        10,
        {
            "logical_indices": np.asarray([0], dtype=np.int64),
            "episode_indices": np.asarray([0], dtype=np.int64),
            "states": np.zeros((1, 10), dtype=np.float32),
            "targets": monitor_actions,
            "predictions": monitor_actions[None, ...],
            "prediction_seeds": np.asarray([101], dtype=np.int64),
        },
        strict_load={"strict": True},
        processor_check={"action_roundtrip": True},
        checkpoint_file_ledger={"model": {"sha256": "f" * 64}},
        evaluation_seconds=1.0,
    )
    _monitor_npz, monitor_manifest_path = evaluator._monitor_checkpoint_paths(
        tmp_path, 10
    )
    monitor_manifest = json.loads(monitor_manifest_path.read_text())
    monitor_payload = dict(monitor_manifest)
    monitor_payload.pop("manifest_sha256")
    monitor_payload["checkpoint_file_ledger"]["model"]["sha256"] = "0" * 64
    monitor_manifest_path.write_text(
        json.dumps(
            {
                "manifest_sha256": evaluator._canonical_sha256(monitor_payload),
                **monitor_payload,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(evaluator.M0EvaluationError, match="ledger/hash mismatch"):
        evaluator._load_verified_monitor_checkpoint(prepared, 10)

    contract_dir = tmp_path / "contract"
    contract_dir.mkdir()
    evaluator._ensure_contract_file(contract_dir, {"a": 1})
    evaluator._ensure_contract_file(contract_dir, {"a": 1})
    with pytest.raises(evaluator.M0EvaluationError, match="different M0 contract"):
        evaluator._ensure_contract_file(contract_dir, {"a": 2})


def test_sharding_and_noise_have_exact_coverage_and_order_independent_seeds() -> None:
    shards = [evaluator.shard_indices(17, 4, index) for index in range(4)]
    assert np.array_equal(np.concatenate(shards), np.arange(17))
    assert max(len(shard) for shard in shards) - min(len(shard) for shard in shards) <= 1

    first = evaluator.fixed_noise_batch(1000, [4, 9])
    reordered = evaluator.fixed_noise_batch(1000, [9, 4])
    assert first.dtype is torch.float32
    assert tuple(first.shape) == (2, 50, 32)
    assert torch.equal(first[0], reordered[1])
    assert torch.equal(first[1], reordered[0])


def test_dataset_ledger_hashes_all_runtime_files_and_artifact_paths_reject_symlinks(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "data").mkdir(parents=True)
    (dataset / "videos").mkdir()
    (dataset / "data/part.parquet").write_bytes(b"parquet")
    (dataset / "videos/camera.mp4").write_bytes(b"video")
    first = evaluator._dataset_file_ledger(dataset)
    assert first["file_count"] == 2
    assert {record["path"] for record in first["files"]} == {
        "data/part.parquet",
        "videos/camera.mp4",
    }
    (dataset / "videos/camera.mp4").write_bytes(b"changed")
    second = evaluator._dataset_file_ledger(dataset)
    assert second["ledger_sha256"] != first["ledger_sha256"]

    work = tmp_path / "work"
    outside = tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    (work / "shards").symlink_to(outside, target_is_directory=True)
    with pytest.raises(evaluator.M0EvaluationError, match="escapes M0 work"):
        evaluator._shard_paths(work, 0)


def test_one_cpu_shard_runs_saved_processors_fixed_noise_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LogicalAnchors:
        def __getitem__(self, index: int) -> Any:
            return {
                "logical_index": index,
                "episode_index": index % 22,
                "raw_episode_id": f"episode-{index % 22:02d}",
            }

    class FakeDataset:
        effective_stats = {"observation.state": {}, "action": {}}
        logical_anchors = LogicalAnchors()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __len__(self) -> int:
            return evaluator.DATASET_SIZE

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "observation.images.base_0_rgb": torch.zeros(3, 2, 2, dtype=torch.uint8),
                "observation.images.left_wrist_0_rgb": torch.full((3, 2, 2), 255, dtype=torch.uint8),
                "observation.state": torch.zeros(10, dtype=torch.float32),
                "action": torch.zeros(50, 7, dtype=torch.float32),
                "task": "stack the cups",
            }

    class FakePolicy:
        def eval(self) -> "FakePolicy":
            return self

        def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
            assert kwargs["num_steps"] == 10
            return batch["action"] + kwargs["noise"][..., :7] * 0.0

    def identity(value: Any) -> Any:
        return value

    load_calls: list[Path] = []

    def fake_loader(path: Path, stats: Any, device: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        load_calls.append(Path(path))
        return (
            FakePolicy(),
            identity,
            identity,
            {
                "pretrained": {
                    "strict": True,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "project_manifest_present": True,
                    "weights_namespace": evaluator.PI0_CORE_WEIGHTS_NAMESPACE,
                    "weights_path": str(Path(path) / "model.safetensors"),
                    "weights_size_bytes": 1,
                    "verified_tensor": {
                        "exact_after_model_dtype_cast": True,
                        "checkpoint_key": "x",
                        "checkpoint_tensor_sha256": "f" * 64,
                    },
                },
                "effective_stats": {"effective_stats_sha256": "e" * 64},
            },
        )

    monkeypatch.setattr(evaluator, "CartesianAnchorDataset", FakeDataset)
    monkeypatch.setattr(evaluator, "load_pi0_full_policy_and_processors", fake_loader)
    monkeypatch.setattr(evaluator, "load_saved_processors", lambda path, device: (identity, identity))

    # One anchor per shard keeps this integration test CPU-only and small.
    contract = {
        "sharding": {"num_shards": evaluator.DATASET_SIZE},
        "inference": {"prediction_seeds": [100, 200], "batch_size": 1},
    }
    prepared = evaluator.PreparedM0(
        local=SimpleNamespace(
            config={
                "dataset": {
                    "root": str(tmp_path / "dataset"),
                    "profile": "action15",
                    "episode_indices": None,
                    "max_anchors_per_episode": None,
                }
            },
            model_dir=tmp_path / "model",
        ),
        contract=contract,
        contract_sha256=evaluator._canonical_sha256(contract),
        work_dir=tmp_path / "work",
    )

    first_result = evaluator._evaluate_missing_shards(
        prepared, [0], device="cpu", num_workers=0
    )
    assert first_result[0]["resume_status"] == "COMPUTED"
    assert len(load_calls) == 1
    arrays, manifest = evaluator._load_verified_shard(prepared, 0)
    assert arrays["predictions"].shape == (2, 1, 50, 7)
    assert manifest["processor_check"]["saved_processors_match_factory_effective_stats"] is True

    second_result = evaluator._evaluate_missing_shards(
        prepared, [0], device="cpu", num_workers=0
    )
    assert second_result[0]["resume_status"] == "SKIPPED_ALREADY_COMPLETE"
    assert len(load_calls) == 1


def test_finalize_merges_every_episode_seed_statistics_and_atomically_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluator, "DATASET_SIZE", 22)
    monkeypatch.setattr(evaluator, "PROJECT_ROOT", tmp_path)
    anchors = [
        {
            "logical_index": index,
            "episode_index": index,
            "raw_episode_id": f"episode-{index:02d}",
        }
        for index in range(22)
    ]

    class FakeDataset:
        logical_anchors = anchors

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __len__(self) -> int:
            return 22

    monkeypatch.setattr(evaluator, "CartesianAnchorDataset", FakeDataset)
    dataset_root = tmp_path / "dataset"
    stats_path = dataset_root / "meta/pi0_eef_stats.json"
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text(json.dumps(_stats()), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contract = {
        "run": {"run_dir": str(run_dir), "final_step": evaluator.FINAL_F0_STEP},
        "dataset": {"stats_sha256": evaluator._sha256_file(stats_path)},
        "checkpoint_selection": {
            "best_train_monitor_step": evaluator.FINAL_F0_STEP,
            "records": [
                {
                    "step": evaluator.FINAL_F0_STEP,
                    "train_monitor_loss": 0.1,
                    "checkpoint_exists": True,
                    "eligible_for_best_train_monitor": True,
                    "is_best_train_monitor": True,
                    "is_last": True,
                }
            ],
        },
        "monitor_cartesian": {
            "checkpoint_steps": [evaluator.FINAL_F0_STEP],
            "logical_indices": [0, 1],
            "prediction_seed": 100_000,
        },
        "integrity": {"unit_test": True},
        "sharding": {"num_shards": 2},
        "inference": {"prediction_seeds": [100_000, 200_000], "batch_size": 1},
    }
    work_dir = tmp_path / "work"
    prepared = evaluator.PreparedM0(
        local=SimpleNamespace(
            config={
                "dataset": {
                    "root": str(dataset_root),
                    "profile": "action15",
                    "episode_indices": None,
                    "max_anchors_per_episode": None,
                }
            },
            run_dir=run_dir,
        ),
        contract=contract,
        contract_sha256=evaluator._canonical_sha256(contract),
        work_dir=work_dir,
    )
    for shard_index in range(2):
        indices = evaluator.shard_indices(22, 2, shard_index)
        count = len(indices)
        targets = np.zeros((count, 50, 7), dtype=np.float32)
        predictions = np.zeros((2, count, 50, 7), dtype=np.float32)
        predictions[0, ..., 0] = 0.001
        predictions[1, ..., 0] = 0.003
        evaluator._publish_shard(
            prepared,
            shard_index,
            {
                "logical_indices": indices,
                "episode_indices": indices.copy(),
                "states": np.zeros((count, 10), dtype=np.float32),
                "targets": targets,
                "predictions": predictions,
                "prediction_seeds": np.asarray([100_000, 200_000], dtype=np.int64),
            },
            strict_load={"strict": True},
            processor_check={"action_roundtrip": True},
            evaluation_seconds=1.0,
        )

    monitor_targets = np.zeros((2, 50, 7), dtype=np.float32)
    evaluator._publish_monitor_checkpoint(
        prepared,
        evaluator.FINAL_F0_STEP,
        {
            "logical_indices": np.asarray([0, 1], dtype=np.int64),
            "episode_indices": np.asarray([0, 1], dtype=np.int64),
            "states": np.zeros((2, 10), dtype=np.float32),
            "targets": monitor_targets,
            "predictions": monitor_targets[None, ...],
            "prediction_seeds": np.asarray([100_000], dtype=np.int64),
        },
        strict_load={"strict": True},
        processor_check={"action_roundtrip": True},
        checkpoint_file_ledger={"model.safetensors": {"sha256": "f" * 64}},
        evaluation_seconds=1.0,
    )
    monkeypatch.setattr(
        evaluator,
        "verify_wandb",
        lambda local: {
            "wandb_status": "PASS",
            "state": "finished",
            "logged_artifacts": 0,
            "history": {
                "monitor/loss": [
                    {"step": evaluator.FINAL_F0_STEP, "value": 0.1}
                ]
            },
        },
    )

    output = tmp_path / "artifact/m0.json"
    report = evaluator.finalize_report(prepared, output_json=output)

    assert json.loads(output.read_text())["status"] == "M0_COMPLETE"
    assert report["has_held_out_validation"] is False
    assert report["has_test_set"] is False
    assert report["audit"]["wandb_metrics_only_run"]["logged_artifacts"] == 0
    assert len(report["monitor_checkpoint_cartesian_comparison"]["checkpoints"]) == 1
    assert len(report["reconstruction"]["prediction_seed_results"][0]["metrics"]["per_episode"]) == 22
    seed_summary = report["reconstruction"]["prediction_seed_summary"]["episode_macro"]
    assert seed_summary["mean"]["error"]["translation_ade_mm"] == pytest.approx(2.0)
    assert seed_summary["population_variance"]["error"]["translation_ade_mm"] == pytest.approx(1.0)
