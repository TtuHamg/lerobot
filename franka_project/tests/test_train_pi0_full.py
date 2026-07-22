from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_pi0_full as harness  # noqa: E402


CONFIG_DIR = PROJECT_ROOT / "configs/train"


@pytest.mark.parametrize(
    "name,stage",
    [
        ("pi0_full_eef_s0_15hz.yaml", "S0-15"),
        ("pi0_full_eef_s1_15hz.yaml", "S1-15"),
        ("pi0_full_eef_f0_15hz.yaml", "F0-15"),
        ("pi0_full_eef_chips_f0_30hz.yaml", "F0-30"),
    ],
)
def test_frozen_stage_configs_pass_strict_validation(name: str, stage: str) -> None:
    cfg, path, digest = harness.load_and_validate_config(CONFIG_DIR / name)

    assert cfg["stage"] == stage
    assert path.is_absolute()
    assert len(digest) == 64
    assert cfg["model"]["peft"] is None
    assert cfg["model"]["freeze_vision_encoder"] is False
    assert cfg["wandb"]["mode"] == "online"
    assert cfg["wandb"]["disable_artifact"] is True


def test_native30_stage_contract_and_profile_geometry_are_supported(tmp_path: Path) -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "pi0_full_eef_f0_15hz.yaml").read_text())
    cfg["stage"] = "F0-30"
    cfg["dataset"].update(
        {
            "profile": "native30",
            "observation_fps": 30,
            "action_fps": 30,
            "chunk_size": 50,
            "task_instruction": "pick up the potato chip",
        }
    )
    path = tmp_path / "native30.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    validated, _, _ = harness.load_and_validate_config(path)
    assert validated["stage"] == "F0-30"
    geometry = harness._geometry_manifest(
        {
            "profile": "native30",
            "observation_fps": 30,
            "action_fps": 30,
            "chunk_size": 50,
            "task_instruction": "pick up the potato chip",
        }
    )
    assert geometry["task_instruction"] == "pick up the potato chip"
    assert geometry["action7"]["frequency_hz"] == 30
    assert geometry["action7"]["chunk_size"] == 50


def test_epoch_checkpoint_frequency_is_explicit_and_batch_independent() -> None:
    cfg, _, _ = harness.load_and_validate_config(
        CONFIG_DIR / "pi0_full_eef_chips_f0_30hz.yaml"
    )

    assert "save_freq" not in cfg["training"]
    assert cfg["training"]["save_freq_epochs"] == 10
    assert harness.resolve_checkpoint_interval_steps(
        cfg["training"], steps_per_epoch=723
    ) == 7_230


def test_legacy_and_final_checkpoint_cadence_remain_supported() -> None:
    assert harness.resolve_checkpoint_interval_steps(
        {"save_freq": "epoch"}, steps_per_epoch=723
    ) == 723
    assert harness.resolve_checkpoint_interval_steps(
        {"save_freq": 500}, steps_per_epoch=723
    ) == 500
    assert not harness.should_save_checkpoint(
        phase_step=7_229,
        save_interval=7_230,
        global_step=7_229,
        stop_step=10_000,
    )
    assert harness.should_save_checkpoint(
        phase_step=7_230,
        save_interval=7_230,
        global_step=7_230,
        stop_step=10_000,
    )
    # A successful early-stop/final invocation remains resumable even when it
    # ends between configured epoch boundaries.
    assert harness.should_save_checkpoint(
        phase_step=2,
        save_interval=7_230,
        global_step=37_332,
        stop_step=37_332,
    )


@pytest.mark.parametrize(
    "save_fields",
    [
        {},
        {"save_freq": "epoch", "save_freq_epochs": 10},
        {"save_freq_epochs": 0},
    ],
)
def test_checkpoint_frequency_config_fails_closed(
    tmp_path: Path, save_fields: dict[str, Any]
) -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "pi0_full_eef_chips_f0_30hz.yaml").read_text())
    cfg["training"].pop("save_freq", None)
    cfg["training"].pop("save_freq_epochs", None)
    cfg["training"].update(save_fields)
    path = tmp_path / "invalid_save_frequency.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(harness.TrainingContractError, match="save_freq"):
        harness.load_and_validate_config(path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda cfg: cfg["model"].__setitem__("peft", {"type": "lora"}), "full-parameter"),
        (lambda cfg: cfg["model"].__setitem__("freeze_vision_encoder", True), "full-parameter"),
        (lambda cfg: cfg["dataset"].__setitem__("action_fps", 30), "obs15/action15"),
        (lambda cfg: cfg["wandb"].__setitem__("mode", "offline"), "online metrics-only"),
        (lambda cfg: cfg["output"].__setitem__("save_to_hub", True), "never pushed"),
    ],
)
def test_config_validation_fails_closed_on_training_drift(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "pi0_full_eef_s0_15hz.yaml").read_text())
    mutation(cfg)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(harness.TrainingContractError, match=match):
        harness.load_and_validate_config(path)


def test_uint8_images_are_divided_by_255_and_float_range_is_gated() -> None:
    batch = {
        key: torch.tensor([[[[0, 255]]]], dtype=torch.uint8)
        for key in harness.CAMERA_KEYS
    }

    result = harness.convert_uint8_images_to_float01(batch, harness.CAMERA_KEYS)

    for key in harness.CAMERA_KEYS:
        assert result[key].dtype == torch.float32
        torch.testing.assert_close(
            result[key].flatten(), torch.tensor([0.0, 1.0]), rtol=0, atol=0
        )

    invalid = {key: torch.ones(1, 3, 2, 2) for key in harness.CAMERA_KEYS}
    invalid[harness.CAMERA_KEYS[0]][0, 0, 0, 0] = 1.01
    with pytest.raises(harness.TrainingContractError, match=r"\[0,1\]"):
        harness.convert_uint8_images_to_float01(invalid, harness.CAMERA_KEYS)


def test_warmup_cosine_scheduler_reaches_peak_then_final() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-5)
    scheduler = harness.build_warmup_cosine_scheduler(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        peak_lr=1.0e-5,
        final_lr=1.0e-6,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5 / 3)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-5 / 3)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)
    for _ in range(8):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)


class _MonitorDataset:
    train_monitor_indices = (1, 7)

    def __init__(self, length: int = 10) -> None:
        self.length = length
        self.logical_anchors = tuple({"logical_index": index} for index in range(length))

    def __len__(self) -> int:
        return self.length


def test_monitor_subset_is_fixed_in_train_and_deterministic() -> None:
    dataset = _MonitorDataset()

    first = harness.select_monitor_subset(dataset, max_samples=5, seed=1017)
    second = harness.select_monitor_subset(dataset, max_samples=5, seed=1017)

    assert first == second
    assert first["held_out"] is False
    assert first["source"] == "training_data"
    assert first["sample_size"] == 2
    assert first["indices"] == [1, 7]
    assert len(first["subset_sha256"]) == 64

    fallback_dataset = _MonitorDataset(length=4)
    fallback_dataset.train_monitor_indices = ()
    fallback = harness.select_monitor_subset(fallback_dataset, max_samples=3, seed=1017)
    assert fallback["sample_size"] == 3
    assert len(set(fallback["indices"])) == 3


def _gradient_group(
    *,
    reached: bool = True,
    finite: bool = True,
    missing_gradient_names: list[str] | None = None,
    missing_numel: int = 0,
) -> dict[str, Any]:
    missing_gradient_names = missing_gradient_names or []
    return {
        "trainable_tensors": 1 + len(missing_gradient_names),
        "trainable_numel": 100 + missing_numel,
        "gradient_tensors": 1 if reached else 0,
        "gradient_numel": 100 if reached else 0,
        "all_gradients_finite": finite,
        "nonzero_gradient_tensors": 1 if reached else 0,
        "missing_gradient_names": missing_gradient_names,
    }


def _valid_gradient_report() -> dict[str, Any]:
    expected_names = sorted(harness.EXPECTED_STRUCTURAL_MISSING_GRADIENTS)
    groups = {name: _gradient_group() for name in harness.REQUIRED_GRADIENT_GROUPS}
    groups["vlm"] = _gradient_group(
        missing_gradient_names=expected_names,
        missing_numel=harness.STRUCTURAL_ACTION_UNREACHABLE_NUMEL,
    )
    return {
        "overall": {
            "trainable_tensors": len(harness.REQUIRED_GRADIENT_GROUPS)
            + harness.STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT,
            "gradient_tensors": len(harness.REQUIRED_GRADIENT_GROUPS),
            "trainable_numel": 400 + harness.STRUCTURAL_ACTION_UNREACHABLE_NUMEL,
            "gradient_numel": 400,
            "all_gradients_finite": True,
        },
        "groups": groups,
    }


def test_gradient_gate_allows_only_exact_structural_exception_and_all_subsystems() -> None:
    assert harness.DDP_FIND_UNUSED_PARAMETERS is True
    assert len(harness.EXPECTED_STRUCTURAL_MISSING_GRADIENTS) == 6
    assert (
        sum(harness.EXPECTED_STRUCTURAL_MISSING_GRADIENTS.values())
        == harness.STRUCTURAL_ACTION_UNREACHABLE_NUMEL
        == 104_861_696
    )

    report = _valid_gradient_report()
    harness.require_gradient_coverage(report)

    broken = copy.deepcopy(report)
    broken["groups"]["vision_encoder"] = _gradient_group(reached=False)
    with pytest.raises(harness.TrainingContractError, match="vision_encoder"):
        harness.require_gradient_coverage(broken)

    broken = copy.deepcopy(report)
    broken["groups"]["action_expert"] = _gradient_group(finite=False)
    with pytest.raises(harness.TrainingContractError, match="action_expert"):
        harness.require_gradient_coverage(broken)

    broken = copy.deepcopy(report)
    broken["overall"]["gradient_numel"] -= 1
    with pytest.raises(harness.TrainingContractError, match="overall"):
        harness.require_gradient_coverage(broken)

    broken = copy.deepcopy(report)
    broken["groups"]["vlm"]["missing_gradient_names"][0] = "model.unexpected.weight"
    with pytest.raises(harness.TrainingContractError, match="structural_exception"):
        harness.require_gradient_coverage(broken)

    broken = copy.deepcopy(report)
    broken["groups"]["vlm"]["missing_gradient_names"].pop()
    with pytest.raises(harness.TrainingContractError, match="expected_but_not_missing"):
        harness.require_gradient_coverage(broken)

    broken = copy.deepcopy(report)
    broken["overall"]["trainable_tensors"] += 1
    broken["overall"]["trainable_numel"] += 1
    broken["groups"]["vlm"]["missing_gradient_names"].append("model.seventh_missing.weight")
    with pytest.raises(
        harness.TrainingContractError,
        match="missing_tensor_count=7|seventh_missing",
    ):
        harness.require_gradient_coverage(broken)


def test_failed_gradient_gate_persists_full_diagnostic_before_raising(tmp_path: Path) -> None:
    report = {
        "overall": {
            "gradient_tensor_coverage": 0.5,
            "gradient_numel_coverage": 0.25,
            "all_gradients_finite": True,
        },
        "groups": {
            name: {
                "trainable_numel": 10,
                "gradient_tensors": 1,
                "gradient_numel": 5,
                "all_gradients_finite": True,
                "nonzero_gradient_tensors": 1,
                "missing_gradient_names": [f"model.{name}.missing.weight"],
            }
            for name in harness.REQUIRED_GRADIENT_GROUPS
        },
    }
    path = tmp_path / "first_backward_gradient_coverage.json"

    with pytest.raises(harness.TrainingContractError, match="coverage gate failed"):
        harness.persist_and_require_gradient_coverage(
            report,
            report_path=path,
            is_main_process=True,
        )

    assert json.loads(path.read_text()) == report


def test_fork_rng_cuda_devices_resolves_indexless_single_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert harness._fork_rng_cuda_devices(torch.device("cuda")) == [3]
    assert harness._fork_rng_cuda_devices(torch.device("cuda:1")) == [1]
    assert harness._fork_rng_cuda_devices(torch.device("cpu")) == []


def test_resume_training_state_loads_each_state_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    state_dir = checkpoint / "training_state"
    state_dir.mkdir(parents=True)
    (state_dir / "training_step.json").write_text(
        json.dumps(
            {
                "step": 2,
                "world_size": 2,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
            }
        )
    )
    calls: list[tuple[str, Path]] = []
    expected_report = {"strict": True}

    def fake_weights(policy: object, path: Path) -> dict[str, bool]:
        calls.append(("weights", path))
        return expected_report

    monkeypatch.setattr(harness, "load_pi0_full_checkpoint_weights", fake_weights)
    monkeypatch.setattr(
        harness,
        "load_optimizer_state",
        lambda optimizer, path: calls.append(("optimizer", path)),
    )
    monkeypatch.setattr(
        harness,
        "load_scheduler_state",
        lambda scheduler, path: calls.append(("scheduler", path)),
    )

    report = harness._load_resume_training_state(
        checkpoint,
        policy=object(),
        optimizer=object(),
        scheduler=object(),
        world_size=2,
        batch_size=1,
        gradient_accumulation_steps=1,
    )

    assert report is expected_report
    assert calls == [
        ("weights", checkpoint / "pretrained_model"),
        ("optimizer", state_dir),
        ("scheduler", state_dir),
    ]


class _TinySequenceDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"index": torch.tensor(index)}


def test_cycling_loader_resume_starts_at_requested_microbatch() -> None:
    first = harness.CyclingLoader(
        _TinySequenceDataset(),
        batch_size=2,
        num_workers=0,
        shuffle=False,
        seed=4,
        rank=0,
        world_size=1,
        start_microbatch=0,
    )
    assert next(first)["index"].tolist() == [0, 1]
    assert next(first)["index"].tolist() == [2, 3]

    resumed = harness.CyclingLoader(
        _TinySequenceDataset(),
        batch_size=2,
        num_workers=0,
        shuffle=False,
        seed=4,
        rank=0,
        world_size=1,
        start_microbatch=2,
    )
    assert next(resumed)["index"].tolist() == [4, 5]
    assert next(resumed)["index"].tolist() == [0, 1]


class _FakeAccelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1
    device = torch.device("cpu")

    def wait_for_everyone(self) -> None:
        return None

    def unwrap_model(self, policy: Any) -> Any:
        return policy


def test_checkpoint_is_staged_then_published_and_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_rng(path: Path) -> None:
        (path / "rng_state.safetensors").write_bytes(b"rng")

    def fake_policy_save(
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        path: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"weights")
        (path / "franka_pi0_checkpoint_manifest.json").write_text(
            json.dumps({"weights_namespace": "pi0_core_unprefixed"}), encoding="utf-8"
        )
        return {}

    def fake_optimizer(_: Any, path: Path) -> None:
        (path / "optimizer_state.safetensors").write_bytes(b"optimizer")

    def fake_scheduler(_: Any, path: Path) -> None:
        (path / "scheduler_state.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(harness, "save_rng_state", fake_rng)
    monkeypatch.setattr(harness, "save_pi0_full_checkpoint", fake_policy_save)
    monkeypatch.setattr(harness, "save_optimizer_state", fake_optimizer)
    monkeypatch.setattr(harness, "save_scheduler_state", fake_scheduler)
    monkeypatch.setattr(harness, "assert_full_parameter_training", lambda _: {})
    stats = tmp_path / "stats.json"
    stats.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    checkpoint = harness.atomic_save_checkpoint(
        run_dir=run_dir,
        step=2,
        accelerator=_FakeAccelerator(),
        policy=object(),
        optimizer=object(),
        scheduler=object(),
        preprocessor=object(),
        postprocessor=object(),
        geometry_manifest={},
        stats_path=stats,
        load_report={},
        config_sha256="c" * 64,
        monitor_sha256="m" * 64,
        batch_size=1,
        gradient_accumulation_steps=1,
        wandb_run_id="run-id",
    )

    assert checkpoint.name == "step-000002"
    assert checkpoint.is_dir()
    assert not (run_dir / "checkpoints/.step-000002.staging").exists()
    pointer = json.loads((run_dir / "checkpoints/last_checkpoint.json").read_text())
    assert pointer["step"] == 2
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    assert manifest["hub_upload"] is False
    assert manifest["wandb_artifact_upload"] is False

    with pytest.raises(harness.TrainingContractError, match="overwrite"):
        harness.atomic_save_checkpoint(
            run_dir=run_dir,
            step=2,
            accelerator=_FakeAccelerator(),
            policy=object(),
            optimizer=object(),
            scheduler=object(),
            preprocessor=object(),
            postprocessor=object(),
            geometry_manifest={},
            stats_path=stats,
            load_report={},
            config_sha256="c" * 64,
            monitor_sha256="m" * 64,
            batch_size=1,
            gradient_accumulation_steps=1,
            wandb_run_id="run-id",
        )


def test_wandb_facade_is_online_metrics_only_and_uses_custom_step_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRun:
        offline = False
        id = "same-run-id"

        def __init__(self) -> None:
            self.defined: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            self.logged: list[dict[str, Any]] = []
            self.finished: list[int] = []

        def get_url(self) -> str:
            return "https://wandb.invalid/same-run-id"

        def define_metric(self, *args: Any, **kwargs: Any) -> None:
            self.defined.append((args, kwargs))

        def log(self, metrics: dict[str, Any]) -> None:
            self.logged.append(metrics)

        def finish(self, *, exit_code: int) -> None:
            self.finished.append(exit_code)

    run = FakeRun()
    captured_init: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> FakeRun:
        captured_init.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    logger = harness.MetricsOnlyWandb(
        {
            "project": "project",
            "entity": "entity",
            "tags": ["full_parameter"],
        },
        run_dir=tmp_path,
        job_name="s0",
        resolved_config={"model": {"peft": None}},
        run_id="same-run-id",
        resume=True,
    )
    logger.log({"train/loss": 1.25}, step=2)
    logger.log({"monitor/loss": 1.0}, step=2)
    logger.finish(exit_code=0)

    assert captured_init["mode"] == "online"
    assert captured_init["save_code"] is False
    assert captured_init["resume"] == "must"
    assert run.logged == [
        {"trainer/step": 2, "train/loss": 1.25},
        {"trainer/step": 2, "monitor/loss": 1.0},
    ]
    assert run.finished == [0]
