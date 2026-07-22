from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_pi0 as harness  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/train/pi0_ActTrans_eef_move_cups_30hz.yaml"


def _config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, cfg: dict[str, Any]) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_move_cups_config_is_stage_free_and_valid() -> None:
    cfg, path, digest = harness.load_and_validate_config(CONFIG_PATH)

    assert path == CONFIG_PATH.resolve()
    assert len(digest) == 64
    assert "stage" not in cfg
    assert cfg["schema_version"] == 2
    assert cfg["dataset"]["expected_valid_anchors"] == 11_714
    assert cfg["model"]["trainability"] == {
        "preset": "action_expert_paligemma",
        "components": None,
    }


def test_arbitrary_stage_is_accepted_as_inert_metadata(tmp_path: Path) -> None:
    cfg = _config()
    cfg["stage"] = "anything-the-user-wants"

    validated, _, _ = harness.load_and_validate_config(_write_config(tmp_path, cfg))

    assert validated["stage"] == "anything-the-user-wants"


@pytest.mark.parametrize("preset", ["full", "action_expert", "action_expert_paligemma"])
def test_all_builtin_trainability_presets_validate(tmp_path: Path, preset: str) -> None:
    cfg = _config()
    cfg["model"]["trainability"] = {"preset": preset, "components": None}

    validated, _, _ = harness.load_and_validate_config(_write_config(tmp_path, cfg))

    assert validated["model"]["trainability"]["preset"] == preset


def test_custom_component_selection_validates(tmp_path: Path) -> None:
    cfg = _config()
    cfg["model"]["trainability"] = {
        "preset": "custom",
        "components": ["multimodal_projector", "action_expert", "action_projections"],
    }

    validated, _, _ = harness.load_and_validate_config(_write_config(tmp_path, cfg))

    assert validated["model"]["trainability"]["preset"] == "custom"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda cfg: cfg["model"].__setitem__(
                "trainability", {"preset": "not-a-mode", "components": None}
            ),
            "unknown PI0 trainability preset",
        ),
        (
            lambda cfg: cfg["model"]["trainability"].__setitem__(
                "components", ["action_expert"]
            ),
            "null/omitted",
        ),
        (
            lambda cfg: cfg["model"].__setitem__("freeze_vision_encoder", True),
            "unknown",
        ),
        (lambda cfg: cfg.__setitem__("schema_version", 1), "schema_version"),
    ],
)
def test_config_rejects_ambiguous_or_legacy_trainability(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    cfg = _config()
    mutation(cfg)

    with pytest.raises(harness.TrainingContractError, match=match):
        harness.load_and_validate_config(_write_config(tmp_path, cfg))


class _TinyProcessor:
    def __init__(self, name: str) -> None:
        self.name = name

    def save_pretrained(
        self,
        destination: str | Path,
        *,
        push_to_hub: bool,
        config_filename: str,
    ) -> None:
        assert push_to_hub is False
        Path(destination, config_filename).write_text(
            json.dumps({"name": self.name}), encoding="utf-8"
        )


class _SingleRankAccelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1
    device = torch.device("cpu")

    @staticmethod
    def wait_for_everyone() -> None:
        return None

    @staticmethod
    def unwrap_model(policy: torch.nn.Module) -> torch.nn.Module:
        return policy


def test_checkpoint_saver_records_partial_parameter_ledger(
    tmp_path: Path,
) -> None:
    # Reuse the purpose-built canonical tiny graph without coupling production
    # code to a mock-specific model implementation.
    import test_pi0_trainability as trainability_tests

    policy = trainability_tests._TinyPolicy()
    spec = {"preset": "action_expert", "components": None}
    parameter_report = harness.apply_pi0_trainability(policy, spec)
    destination = tmp_path / "checkpoint"

    manifest = harness.save_pi0_checkpoint(
        policy,
        _TinyProcessor("pre"),
        _TinyProcessor("post"),
        destination,
        trainability_spec=spec,
        expected_trainability_signature=parameter_report[
            "trainability_signature_sha256"
        ],
        training_graph={"schema_version": 99, "test": True},
        geometry_manifest={"task_instruction": "move cup"},
        stats_manifest={"observation.state": {}, "action": {}},
        load_report={"strict": True},
    )

    assert (destination / "model.safetensors").stat().st_size > 0
    assert manifest["checkpoint_type"] == "franka_pi0_configurable_parameter_eef"
    assert manifest["parameter_training"]["preset"] == "action_expert"
    assert manifest["parameter_training"]["trainability_signature_sha256"] == (
        parameter_report["trainability_signature_sha256"]
    )
    stored = json.loads(
        (destination / "franka_pi0_checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert stored["training_graph"] == {"schema_version": 99, "test": True}


def test_atomic_checkpoint_publishes_pointer_and_refuses_overwrite(tmp_path: Path) -> None:
    import test_pi0_trainability as trainability_tests

    policy = trainability_tests._TinyPolicy()
    spec = {"preset": "action_expert", "components": None}
    parameter_report = harness.apply_pi0_trainability(policy, spec)
    parameters = harness.trainable_pi0_parameters(policy, spec)
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    sum(parameter.square().sum() for parameter in parameters).backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"observation.state": {}, "action": {}}))
    stats_sha256 = harness.shared._sha256_file(stats_path)

    published = harness.atomic_save_checkpoint(
        run_dir=run_dir,
        step=1,
        accelerator=_SingleRankAccelerator(),
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        preprocessor=_TinyProcessor("pre"),
        postprocessor=_TinyProcessor("post"),
        trainability_spec=spec,
        parameter_report=parameter_report,
        training_graph={"schema_version": 99, "test": True},
        geometry_manifest={"task_instruction": "move cup"},
        stats_path=stats_path,
        load_report={"strict": True},
        config_sha256="b" * 64,
        monitor_sha256="c" * 64,
        batch_size=2,
        gradient_accumulation_steps=1,
        wandb_run_id="test-run",
        dataset_stats_sha256=stats_sha256,
        global_samples=2,
    )

    assert published == run_dir / "checkpoints/step-000001"
    pointer = json.loads(
        (run_dir / "checkpoints/last_checkpoint.json").read_text(encoding="utf-8")
    )
    assert pointer == {"step": 1, "path": "checkpoints/step-000001"}
    assert not (run_dir / "checkpoints/.step-000001.staging").exists()

    expected = {
        "config_sha256": "b" * 64,
        "monitor_subset_sha256": "c" * 64,
        "world_size": 1,
        "planned_total_steps": 10,
        "dataset_profile_sha256": "e" * 64,
        "dataset_stats_sha256": stats_sha256,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                **expected,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "wandb_run_id": "test-run",
                "trainability_signature_sha256": parameter_report[
                    "trainability_signature_sha256"
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "checkpoints/last_checkpoint.json").unlink()
    complete_staging = run_dir / "checkpoints/.step-000001.staging"
    published.rename(complete_staging)
    incomplete = run_dir / "checkpoints/.step-000002.staging"
    incomplete.mkdir()

    wrong_identity = {**expected, "dataset_stats_sha256": "f" * 64}
    with pytest.raises(
        harness.TrainingContractError,
        match="identity mismatch before checkpoint recovery",
    ):
        harness.recover_checkpoint_publication(run_dir, wrong_identity)
    assert complete_staging.is_dir()
    assert incomplete.is_dir()

    recovery = harness.recover_checkpoint_publication(run_dir, expected)

    assert recovery["pointer_updated"] is True
    assert recovery["step"] == 1
    assert recovery["recovered_staging"] == ["step-000001"]
    assert recovery["quarantined_staging"]
    assert published.is_dir()
    assert not complete_staging.exists()
    assert not incomplete.exists()
    assert json.loads(
        (run_dir / "checkpoints/last_checkpoint.json").read_text(encoding="utf-8")
    ) == {"step": 1, "path": "checkpoints/step-000001"}

    with pytest.raises(harness.TrainingContractError, match="staging preflight"):
        harness.atomic_save_checkpoint(
            run_dir=run_dir,
            step=1,
            accelerator=_SingleRankAccelerator(),
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            preprocessor=_TinyProcessor("pre"),
            postprocessor=_TinyProcessor("post"),
            trainability_spec=spec,
            parameter_report=parameter_report,
            training_graph={"schema_version": 99, "test": True},
            geometry_manifest={"task_instruction": "move cup"},
            stats_path=stats_path,
            load_report={"strict": True},
            config_sha256="b" * 64,
            monitor_sha256="c" * 64,
            batch_size=2,
            gradient_accumulation_steps=1,
            wandb_run_id="test-run",
            dataset_stats_sha256=stats_sha256,
            global_samples=2,
        )


def test_partial_checkpoint_strictly_reloads_then_reapplies_same_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import test_pi0_training as io_tests
    from franka_eef_pipeline import pi0_training as pi0_io

    monkeypatch.setattr(
        pi0_io,
        "STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS",
        io_tests._TINY_STRUCTURAL_ACTION_UNREACHABLE_SPECS,
    )
    monkeypatch.setattr(pi0_io, "STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT", 6)
    monkeypatch.setattr(pi0_io, "STRUCTURAL_ACTION_UNREACHABLE_NUMEL", 20)

    source = tmp_path / "source"
    io_tests._write_local_checkpoint(source)
    config = pi0_io.build_pi0_full_finetune_config(source, device="cpu")
    policy = io_tests._TinyPolicy(config)
    training_graph = pi0_io.canonicalize_pi0_full_training_graph(policy)
    spec = {"preset": "action_expert", "components": None}
    parameter_report = harness.apply_pi0_trainability(policy, spec)
    destination = tmp_path / "partial"

    harness.save_pi0_checkpoint(
        policy,
        io_tests._TinyProcessor("pre"),
        io_tests._TinyProcessor("post"),
        destination,
        trainability_spec=spec,
        expected_trainability_signature=parameter_report[
            "trainability_signature_sha256"
        ],
        training_graph=training_graph,
        geometry_manifest={"task_instruction": "move cup"},
        stats_manifest=io_tests._effective_stats(),
    )

    monkeypatch.setattr(pi0_io, "PI0Policy", io_tests._TinyPolicy)
    monkeypatch.setattr(
        pi0_io,
        "make_pi0_pre_post_processors",
        lambda *, config, dataset_stats: ("pre", "post"),
    )
    reloaded, _, _, load_report = pi0_io.load_pi0_full_policy_and_processors(
        destination,
        destination / "pi0_eef_stats.json",
        "cpu",
    )
    assert load_report["pretrained"]["strict"] is True
    assert all(parameter.requires_grad for parameter in reloaded.parameters())

    reapplied = harness.apply_pi0_trainability(reloaded, spec)
    assert reapplied["trainability_signature_sha256"] == parameter_report[
        "trainability_signature_sha256"
    ]
    assert reapplied["trainable_parameter_names"] == parameter_report[
        "trainable_parameter_names"
    ]


def test_resume_state_rejects_trainability_signature_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-000001"
    state_dir = checkpoint / "training_state"
    state_dir.mkdir(parents=True)
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps({"trainability_signature_sha256": "different"}),
        encoding="utf-8",
    )
    (state_dir / "training_step.json").write_text(
        json.dumps(
            {
                "step": 1,
                "world_size": 1,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "trainability_signature_sha256": "different",
            }
        ),
        encoding="utf-8",
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    with pytest.raises(harness.TrainingContractError, match="resume training-state mismatch"):
        harness._load_resume_optimizer_scheduler(
            checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_step=1,
            world_size=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            trainability_signature_sha256="expected",
            dataset_stats_sha256="d" * 64,
        )


def test_resume_state_restores_optimizer_moments_and_scheduler(tmp_path: Path) -> None:
    from lerobot.optim import save_optimizer_state, save_scheduler_state

    checkpoint = tmp_path / "step-000001"
    state_dir = checkpoint / "training_state"
    state_dir.mkdir(parents=True)
    signature = "a" * 64
    stats_sha256 = "d" * 64

    source_parameter = torch.nn.Parameter(torch.tensor(1.0))
    source_optimizer = torch.optim.AdamW([source_parameter], lr=1.0e-4)
    source_scheduler = torch.optim.lr_scheduler.LambdaLR(
        source_optimizer, lambda step: 1.0 / (step + 1)
    )
    source_parameter.square().backward()
    source_optimizer.step()
    source_scheduler.step()
    save_optimizer_state(source_optimizer, state_dir)
    save_scheduler_state(source_scheduler, state_dir)
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "trainability_signature_sha256": signature,
                "dataset_stats_sha256": stats_sha256,
                "global_samples": 2,
            }
        ),
        encoding="utf-8",
    )
    (state_dir / "training_step.json").write_text(
        json.dumps(
            {
                "step": 1,
                "world_size": 1,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "trainability_signature_sha256": signature,
                "dataset_stats_sha256": stats_sha256,
                "global_samples": 2,
            }
        ),
        encoding="utf-8",
    )
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir()
    (pretrained / "franka_pi0_checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "pi0_eef_stats.json": {"sha256": stats_sha256},
                }
            }
        ),
        encoding="utf-8",
    )

    restored_parameter = torch.nn.Parameter(torch.tensor(5.0))
    restored_optimizer = torch.optim.AdamW([restored_parameter], lr=1.0e-4)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda step: 1.0 / (step + 1)
    )
    restored_global_samples = harness._load_resume_optimizer_scheduler(
        checkpoint,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_step=1,
        world_size=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        trainability_signature_sha256=signature,
        dataset_stats_sha256=stats_sha256,
    )

    source_state = source_optimizer.state[source_parameter]
    restored_state = restored_optimizer.state[restored_parameter]
    torch.testing.assert_close(restored_state["exp_avg"], source_state["exp_avg"])
    torch.testing.assert_close(restored_state["exp_avg_sq"], source_state["exp_avg_sq"])
    assert float(restored_state["step"]) == float(source_state["step"])
    assert restored_scheduler.state_dict() == source_scheduler.state_dict()
    assert restored_global_samples == 2
