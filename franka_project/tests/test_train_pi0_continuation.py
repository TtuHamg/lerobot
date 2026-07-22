from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_pi0_full as harness  # noqa: E402


CONFIG_DIR = PROJECT_ROOT / "configs/train"


def _continuation_config(source_checkpoint: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((CONFIG_DIR / "pi0_full_eef_f0_15hz.yaml").read_text())
    cfg["stage"] = "F0-CONT-15"
    cfg["job_name"] = "pi0_full_eef_f0_continue20"
    cfg["training"]["epochs"] = 20
    cfg["training"]["peak_lr"] = 1.0e-6
    cfg["training"]["final_lr"] = 1.0e-7
    for key in ("warmup_steps", "warmup_fraction", "warmup_max_steps"):
        cfg["training"].pop(key, None)
    cfg["continuation"] = {
        "source_checkpoint": str(source_checkpoint),
        "expected_source_step": 37_330,
        "expected_source_run_config_sha256": "a" * 64,
        "expected_source_checkpoint_manifest_sha256": "b" * 64,
        "preserve_optimizer_state": True,
        "preserve_rng_state": True,
        "preserve_global_step": True,
        "preserve_data_offset": True,
        "scheduler": "cosine_no_warmup",
    }
    return cfg


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_source_tree(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_root = (tmp_path / "dataset").resolve()
    source_run = (tmp_path / "source-run").resolve()
    checkpoint = source_run / "checkpoints/step-037330"
    state_dir = checkpoint / "training_state"
    model_dir = checkpoint / "pretrained_model"
    state_dir.mkdir(parents=True)
    model_dir.mkdir()

    source_cfg = yaml.safe_load((CONFIG_DIR / "pi0_full_eef_f0_15hz.yaml").read_text())
    source_cfg["dataset"]["root"] = str(dataset_root)
    source_cfg["model"]["pretrained_path"] = str((tmp_path / "base-pi0").resolve())
    source_cfg["output"]["root"] = str((tmp_path / "source-output").resolve())
    source_config_path = source_run / "resolved_config.yaml"
    source_config_path.parent.mkdir(parents=True, exist_ok=True)
    source_config_path.write_text(yaml.safe_dump(source_cfg, sort_keys=False), encoding="utf-8")
    source_config_sha = harness._sha256_file(source_config_path)

    monitor_payload = {
        "held_out": False,
        "source": "training_data",
        "seed": 1017,
        "population": 7465,
        "sample_size": 2,
        "indices": [1, 7],
    }
    monitor_sha = harness._canonical_sha256(monitor_payload)
    _write_json(source_run / "monitor_subset.json", {**monitor_payload, "subset_sha256": monitor_sha})

    dataset_profile_sha = "d" * 64
    run_manifest = {
        "schema_version": 1,
        "config_sha256": source_config_sha,
        "monitor_subset_sha256": monitor_sha,
        "world_size": 2,
        "planned_total_steps": 37_330,
        "dataset_profile_sha256": dataset_profile_sha,
        "dataset_root": str(dataset_root),
        "dataset_size": 7465,
        "steps_per_epoch": 3733,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "wandb_run_id": "source-wandb-id",
    }
    _write_json(source_run / "run_manifest.json", run_manifest)

    checkpoint_manifest = {
        "schema_version": 1,
        "step": 37_330,
        "config_sha256": source_config_sha,
        "monitor_subset_sha256": monitor_sha,
        "wandb_run_id": "source-wandb-id",
        "world_size": 2,
        "hub_upload": False,
        "wandb_artifact_upload": False,
        "atomic_publish": True,
    }
    checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
    _write_json(checkpoint_manifest_path, checkpoint_manifest)
    _write_json(
        source_run / "checkpoints/last_checkpoint.json",
        {"step": 37_330, "path": "checkpoints/step-037330"},
    )
    _write_json(
        state_dir / "training_step.json",
        {
            "step": 37_330,
            "world_size": 2,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
    )
    (state_dir / "optimizer_state.safetensors").write_bytes(b"optimizer-moments")
    _write_json(
        state_dir / "optimizer_param_groups.json",
        [
            {
                "params": list(range(776)),
                "lr": 1.0e-6,
                "initial_lr": 1.0e-5,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": True,
            }
        ],
    )
    _write_json(
        state_dir / "scheduler_state.json",
        {"last_epoch": 37_330, "_last_lr": [1.0e-6]},
    )
    for rank in range(2):
        rank_dir = state_dir / f"rank-{rank:02d}"
        rank_dir.mkdir()
        (rank_dir / "rng_state.safetensors").write_bytes(f"rank-{rank}-rng".encode())

    model_bytes = b"full-parameter-pi0"
    (model_dir / "model.safetensors").write_bytes(model_bytes)
    _write_json(
        model_dir / "franka_pi0_checkpoint_manifest.json",
        {
            "checkpoint_type": "franka_pi0_full_parameter_eef",
            "hub_upload": False,
            "parameter_training": {"lora_parameter_count": 0, "trainable_fraction": 1.0},
            "files": {
                "model.safetensors": {
                    "size_bytes": len(model_bytes),
                    "sha256": "e" * 64,
                }
            },
        },
    )

    continuation_cfg = _continuation_config(checkpoint)
    continuation_cfg["dataset"] = copy.deepcopy(source_cfg["dataset"])
    continuation_cfg["model"] = copy.deepcopy(source_cfg["model"])
    continuation_cfg["training"].update(
        {
            key: copy.deepcopy(source_cfg["training"][key])
            for key in (
                "seed",
                "per_device_batch_size",
                "gradient_accumulation_steps",
                "num_workers",
                "shuffle",
                "weight_decay",
                "betas",
                "eps",
                "grad_clip_norm",
                "mixed_precision",
                "gpu_count",
            )
        }
    )
    continuation_cfg["continuation"]["expected_source_run_config_sha256"] = source_config_sha
    continuation_cfg["continuation"]["expected_source_checkpoint_manifest_sha256"] = (
        harness._sha256_file(checkpoint_manifest_path)
    )
    kwargs = {
        "dataset_size": 7465,
        "dataset_profile_sha256": dataset_profile_sha,
        "monitor_subset_sha256": monitor_sha,
        "steps_per_epoch": 3733,
        "world_size": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    return continuation_cfg, kwargs


def _single_parameter_adamw(*, lr: float = 1.0e-5) -> tuple[torch.nn.Parameter, torch.optim.AdamW]:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    parameter.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return parameter, optimizer


def _clone_optimizer_moments(
    optimizer: torch.optim.Optimizer,
) -> dict[torch.nn.Parameter, dict[str, object]]:
    return {
        parameter: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in state.items()
        }
        for parameter, state in optimizer.state.items()
    }


def _assert_optimizer_moments_equal(
    optimizer: torch.optim.Optimizer,
    expected: dict[torch.nn.Parameter, dict[str, object]],
) -> None:
    assert optimizer.state.keys() == expected.keys()
    for parameter, expected_state in expected.items():
        assert optimizer.state[parameter].keys() == expected_state.keys()
        for key, expected_value in expected_state.items():
            actual_value = optimizer.state[parameter][key]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                assert actual_value == expected_value


def test_continuation_step_plan_preserves_cumulative_global_axis() -> None:
    plan = harness.compute_continuation_step_plan(
        source_step=37_330,
        additional_steps=74_660,
    )

    assert plan == {
        "phase_start_global_step": 37_330,
        "planned_phase_steps": 74_660,
        "planned_final_global_step": 111_990,
    }


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"source_step": 0, "additional_steps": 1}, "source_step"),
        ({"source_step": 1, "additional_steps": 0}, "additional_steps"),
        ({"source_step": True, "additional_steps": 1}, "source_step"),
    ],
)
def test_continuation_step_plan_rejects_non_positive_or_boolean_steps(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(harness.TrainingContractError, match=match):
        harness.compute_continuation_step_plan(**kwargs)


def test_continuation_scheduler_preserves_adamw_moments_and_never_jumps_lr() -> None:
    _, optimizer = _single_parameter_adamw()
    source_lr = 1.0e-6

    # This is the exact hazardous state stored by F0: the current LR is at the
    # old cosine floor, but LambdaLR left the old peak in ``initial_lr``.
    optimizer.param_groups[0]["lr"] = source_lr
    optimizer.param_groups[0]["initial_lr"] = 1.0e-5
    moments_before = _clone_optimizer_moments(optimizer)

    scheduler = harness.build_continuation_cosine_scheduler(
        optimizer,
        phase_steps=8,
        expected_current_lr=source_lr,
        start_lr=source_lr,
        final_lr=1.0e-7,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(source_lr)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(source_lr)
    assert scheduler.base_lrs == pytest.approx([source_lr])
    assert scheduler.get_last_lr() == pytest.approx([source_lr])
    _assert_optimizer_moments_equal(optimizer, moments_before)

    learning_rates = [float(scheduler.get_last_lr()[0])]
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        learning_rates.append(float(scheduler.get_last_lr()[0]))

    assert learning_rates[0] == pytest.approx(source_lr)
    assert learning_rates[-1] == pytest.approx(1.0e-7)
    assert max(learning_rates) <= source_lr
    assert all(left >= right for left, right in zip(learning_rates, learning_rates[1:]))


def test_continuation_scheduler_rejects_lr_or_horizon_drift() -> None:
    _, optimizer = _single_parameter_adamw(lr=5.0e-7)

    with pytest.raises(harness.TrainingContractError, match="current lr"):
        harness.build_continuation_cosine_scheduler(
            optimizer,
            phase_steps=10,
            expected_current_lr=1.0e-6,
            start_lr=1.0e-6,
            final_lr=1.0e-7,
        )

    optimizer.param_groups[0]["lr"] = 1.0e-6
    with pytest.raises(harness.TrainingContractError, match="final_lr"):
        harness.build_continuation_cosine_scheduler(
            optimizer,
            phase_steps=10,
            expected_current_lr=1.0e-6,
            start_lr=1.0e-6,
            final_lr=2.0e-6,
        )

    with pytest.raises(harness.TrainingContractError, match="phase_steps"):
        harness.build_continuation_cosine_scheduler(
            optimizer,
            phase_steps=0,
            expected_current_lr=1.0e-6,
            start_lr=1.0e-6,
            final_lr=1.0e-7,
        )


def test_continuation_scheduler_can_restart_lr_without_resetting_adamw_moments() -> None:
    _, optimizer = _single_parameter_adamw()
    source_lr = 1.0e-6
    restart_lr = 1.0e-5
    optimizer.param_groups[0]["lr"] = source_lr
    optimizer.param_groups[0]["initial_lr"] = restart_lr
    moments_before = _clone_optimizer_moments(optimizer)

    scheduler = harness.build_continuation_cosine_scheduler(
        optimizer,
        phase_steps=8,
        expected_current_lr=source_lr,
        start_lr=restart_lr,
        final_lr=source_lr,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(restart_lr)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(restart_lr)
    assert scheduler.base_lrs == pytest.approx([restart_lr])
    assert scheduler.get_last_lr() == pytest.approx([restart_lr])
    _assert_optimizer_moments_equal(optimizer, moments_before)

    learning_rates = [float(scheduler.get_last_lr()[0])]
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        learning_rates.append(float(scheduler.get_last_lr()[0]))

    assert learning_rates[-1] == pytest.approx(source_lr)
    assert all(left >= right for left, right in zip(learning_rates, learning_rates[1:]))


def test_continuation_config_is_explicit_and_has_no_warmup(tmp_path: Path) -> None:
    cfg = _continuation_config(tmp_path / "source/checkpoints/step-037330")
    path = tmp_path / "continuation.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    resolved, _, digest = harness.load_and_validate_config(path)

    assert resolved["stage"] == "F0-CONT-15"
    assert resolved["training"]["epochs"] == 20
    assert resolved["training"]["peak_lr"] == pytest.approx(1.0e-6)
    assert resolved["training"]["final_lr"] == pytest.approx(1.0e-7)
    assert not {"warmup_steps", "warmup_fraction", "warmup_max_steps"}.intersection(
        resolved["training"]
    )
    assert len(digest) == 64


def test_frozen_continue20_config_passes_strict_validation() -> None:
    cfg, path, digest = harness.load_and_validate_config(
        CONFIG_DIR / "pi0_full_eef_f0_continue20_15hz.yaml"
    )

    assert path.is_absolute()
    assert len(digest) == 64
    assert cfg["stage"] == "F0-CONT-15"
    assert cfg["training"]["epochs"] == 20
    assert cfg["continuation"]["expected_source_step"] == 37_330
    assert cfg["continuation"]["scheduler"] == "cosine_no_warmup"


def test_frozen_chips_continue50_restart_config_passes_strict_validation() -> None:
    cfg, path, digest = harness.load_and_validate_config(
        CONFIG_DIR / "pi0_full_eef_chips_f0_continue50_restart_30hz.yaml"
    )

    assert path.is_absolute()
    assert len(digest) == 64
    assert cfg["stage"] == "F0-CONT-30"
    assert cfg["training"]["epochs"] == 50
    assert cfg["training"]["peak_lr"] == pytest.approx(1.0e-5)
    assert cfg["training"]["final_lr"] == pytest.approx(1.0e-6)
    assert cfg["continuation"]["expected_source_step"] == 36_150
    assert cfg["continuation"]["scheduler"] == "cosine_restart_no_warmup"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda cfg: cfg["continuation"].__setitem__("preserve_optimizer_state", False),
            "preserve_optimizer_state",
        ),
        (
            lambda cfg: cfg["continuation"].__setitem__("preserve_rng_state", False),
            "preserve_rng_state",
        ),
        (
            lambda cfg: cfg["training"].__setitem__("warmup_steps", 1),
            "no warmup",
        ),
        (
            lambda cfg: cfg["continuation"].__setitem__("scheduler", "warmup_cosine"),
            "cosine_no_warmup",
        ),
    ],
)
def test_continuation_config_fails_closed(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    cfg = _continuation_config(tmp_path / "source/checkpoints/step-037330")
    mutation(cfg)
    path = tmp_path / "invalid-continuation.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(harness.TrainingContractError, match=match):
        harness.load_and_validate_config(path)


def test_resolve_continuation_source_builds_a_frozen_lineage(tmp_path: Path) -> None:
    cfg, kwargs = _build_source_tree(tmp_path)

    source = harness.resolve_continuation_source(cfg, **kwargs)

    assert source["checkpoint"] == Path(cfg["continuation"]["source_checkpoint"])
    assert source["source_step"] == 37_330
    assert source["source_lr"] == pytest.approx(1.0e-6)
    assert source["source_initial_lr"] == pytest.approx(1.0e-5)
    assert len(source["source_identity_sha256"]) == 64
    identity = source["source_identity"]
    assert identity["global_step"] == 37_330
    assert identity["source_stage"] == "F0-15"
    assert identity["source_wandb_run_id"] == "source-wandb-id"
    assert identity["optimizer_size_bytes"] > 0
    assert set(identity["rank_rng_sha256"]) == {"rank-00", "rank-01"}
    assert len(set(identity["rank_rng_sha256"].values())) == 2


def test_resolve_continuation_source_allows_explicit_lr_restart(tmp_path: Path) -> None:
    cfg, kwargs = _build_source_tree(tmp_path)
    cfg["continuation"]["scheduler"] = "cosine_restart_no_warmup"
    cfg["training"]["peak_lr"] = 1.0e-5
    cfg["training"]["final_lr"] = 1.0e-6

    source = harness.resolve_continuation_source(cfg, **kwargs)

    assert source["source_lr"] == pytest.approx(1.0e-6)
    assert cfg["training"]["peak_lr"] == pytest.approx(1.0e-5)


@pytest.mark.parametrize(
    "runtime_key,runtime_value,match",
    [
        ("world_size", 1, "world_size"),
        ("batch_size", 2, "per_device_batch_size"),
        ("gradient_accumulation_steps", 2, "gradient_accumulation_steps"),
    ],
)
def test_resolve_continuation_source_rejects_runtime_shape_drift(
    tmp_path: Path,
    runtime_key: str,
    runtime_value: int,
    match: str,
) -> None:
    cfg, kwargs = _build_source_tree(tmp_path)
    kwargs[runtime_key] = runtime_value

    with pytest.raises(harness.TrainingContractError, match=match):
        harness.resolve_continuation_source(cfg, **kwargs)


def test_resolve_continuation_source_rejects_source_and_training_drift(tmp_path: Path) -> None:
    cfg, kwargs = _build_source_tree(tmp_path)
    cfg["training"]["seed"] += 1
    with pytest.raises(harness.TrainingContractError, match="training-state contract drift"):
        harness.resolve_continuation_source(cfg, **kwargs)

    cfg, kwargs = _build_source_tree(tmp_path / "bad-path")
    cfg["continuation"]["source_checkpoint"] = str(tmp_path / "not-a-step-directory")
    with pytest.raises(harness.TrainingContractError, match=r"complete step-\*"):
        harness.resolve_continuation_source(cfg, **kwargs)


def test_resolve_continuation_source_rejects_optimizer_lr_and_rng_drift(tmp_path: Path) -> None:
    cfg, kwargs = _build_source_tree(tmp_path / "bad-optimizer")
    checkpoint = Path(cfg["continuation"]["source_checkpoint"])
    groups_path = checkpoint / "training_state/optimizer_param_groups.json"
    groups = json.loads(groups_path.read_text())
    groups[0]["params"][0], groups[0]["params"][1] = groups[0]["params"][1], groups[0]["params"][0]
    _write_json(groups_path, groups)
    with pytest.raises(harness.TrainingContractError, match="contiguous and ordered"):
        harness.resolve_continuation_source(cfg, **kwargs)

    cfg, kwargs = _build_source_tree(tmp_path / "bad-optimizer-flags")
    checkpoint = Path(cfg["continuation"]["source_checkpoint"])
    groups_path = checkpoint / "training_state/optimizer_param_groups.json"
    groups = json.loads(groups_path.read_text())
    groups[0]["maximize"] = True
    _write_json(groups_path, groups)
    with pytest.raises(harness.TrainingContractError, match="behavior flag drift"):
        harness.resolve_continuation_source(cfg, **kwargs)

    cfg, kwargs = _build_source_tree(tmp_path / "bad-lr")
    checkpoint = Path(cfg["continuation"]["source_checkpoint"])
    scheduler_path = checkpoint / "training_state/scheduler_state.json"
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["_last_lr"] = [2.0e-6]
    _write_json(scheduler_path, scheduler)
    with pytest.raises(harness.TrainingContractError, match="scheduler/AdamW LR"):
        harness.resolve_continuation_source(cfg, **kwargs)

    cfg, kwargs = _build_source_tree(tmp_path / "bad-rng")
    checkpoint = Path(cfg["continuation"]["source_checkpoint"])
    rank0 = checkpoint / "training_state/rank-00/rng_state.safetensors"
    rank1 = checkpoint / "training_state/rank-01/rng_state.safetensors"
    rank1.write_bytes(rank0.read_bytes())
    with pytest.raises(harness.TrainingContractError, match="unexpectedly identical"):
        harness.resolve_continuation_source(cfg, **kwargs)


class _NamedParameterPolicy:
    def __init__(self) -> None:
        self.items: list[tuple[str, torch.nn.Parameter]] = [
            (f"model.active_{index:03d}.weight", torch.nn.Parameter(torch.tensor([float(index)])))
            for index in range(770)
        ]
        self.items.extend(
            (name, torch.nn.Parameter(torch.tensor([0.0])))
            for name in sorted(harness.EXPECTED_STRUCTURAL_MISSING_GRADIENTS)
        )

    def named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return self.items

    def parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for _, parameter in self.items]


def test_load_continuation_source_loads_weights_and_adamw_but_not_old_scheduler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _NamedParameterPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1.0e-5)
    checkpoint = tmp_path / "step-037330"
    state_dir = checkpoint / "training_state"
    calls: list[tuple[str, Path]] = []

    def fake_weights(_: Any, path: Path) -> dict[str, bool]:
        calls.append(("weights", path))
        return {"strict": True}

    def fake_optimizer_load(target: torch.optim.Optimizer, path: Path) -> None:
        calls.append(("optimizer", path))
        target.param_groups[0]["lr"] = 1.0e-6
        for parameter in target.param_groups[0]["params"][:770]:
            target.state[parameter] = {
                "step": torch.tensor(37_330.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.ones_like(parameter),
            }

    monkeypatch.setattr(harness, "load_pi0_full_checkpoint_weights", fake_weights)
    monkeypatch.setattr(harness, "load_optimizer_state", fake_optimizer_load)
    monkeypatch.setattr(
        harness,
        "load_scheduler_state",
        lambda *_: pytest.fail("the completed F0 scheduler must not be loaded"),
    )

    report = harness.load_continuation_source_state(
        {
            "checkpoint": checkpoint,
            "state_dir": state_dir,
            "source_step": 37_330,
            "source_lr": 1.0e-6,
            "source_initial_lr": 1.0e-5,
        },
        policy=policy,
        optimizer=optimizer,
    )

    assert calls == [
        ("weights", checkpoint / "pretrained_model"),
        ("optimizer", state_dir),
    ]
    assert report["weights"] == {"strict": True}
    assert report["optimizer"]["active_state_count"] == 770
    assert report["optimizer"]["optimizer_step"] == 37_330
    assert report["optimizer"]["lr"] == pytest.approx(1.0e-6)
    assert report["optimizer"]["initial_lr"] == pytest.approx(1.0e-5)
    assert report["optimizer"]["moments_preserved"] is True
    assert report["optimizer"]["stateless_parameter_names"] == sorted(
        harness.EXPECTED_STRUCTURAL_MISSING_GRADIENTS
    )


def test_first_continuation_strict_optimizer_load_seeds_scheduler_owned_initial_lr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_policy = _NamedParameterPolicy()
    source_optimizer = torch.optim.AdamW(
        source_policy.parameters(),
        lr=1.0e-5,
        betas=(0.9, 0.95),
    )
    for parameter in source_optimizer.param_groups[0]["params"][:770]:
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    source_optimizer.param_groups[0]["initial_lr"] = 1.0e-5
    source_optimizer.param_groups[0]["lr"] = 1.0e-6

    checkpoint = tmp_path / "step-000001"
    state_dir = checkpoint / "training_state"
    state_dir.mkdir(parents=True)
    harness.save_optimizer_state(source_optimizer, state_dir)

    target_policy = _NamedParameterPolicy()
    target_optimizer = torch.optim.AdamW(
        target_policy.parameters(),
        lr=1.0e-5,
        betas=(0.9, 0.95),
    )
    assert "initial_lr" not in target_optimizer.param_groups[0]
    monkeypatch.setattr(
        harness,
        "load_pi0_full_checkpoint_weights",
        lambda *_: {"strict": True},
    )

    report = harness.load_continuation_source_state(
        {
            "checkpoint": checkpoint,
            "state_dir": state_dir,
            "source_step": 1,
            "source_lr": 1.0e-6,
            "source_initial_lr": 1.0e-5,
        },
        policy=target_policy,
        optimizer=target_optimizer,
    )

    assert report["optimizer"]["optimizer_step"] == 1
    assert report["optimizer"]["lr"] == pytest.approx(1.0e-6)
    assert report["optimizer"]["initial_lr"] == pytest.approx(1.0e-5)
    assert report["optimizer"]["moments_preserved"] is True


class _FakeAccelerator:
    is_main_process = True
    process_index = 0
    num_processes = 2
    device = torch.device("cpu")

    def wait_for_everyone(self) -> None:
        return None

    def unwrap_model(self, policy: Any) -> Any:
        return policy


def _patch_checkpoint_writers(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_continuation_checkpoint_persists_global_and_phase_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_checkpoint_writers(monkeypatch)
    run_dir = tmp_path / "continuation-run"
    run_dir.mkdir()
    stats = tmp_path / "stats.json"
    stats.write_text("{}", encoding="utf-8")
    source_identity_sha = "s" * 64

    checkpoint = harness.atomic_save_checkpoint(
        run_dir=run_dir,
        step=37_332,
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
        wandb_run_id="continuation-wandb-id",
        phase_start_global_step=37_330,
        source_identity_sha256=source_identity_sha,
    )

    checkpoint_manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    training_step = json.loads((checkpoint / "training_state/training_step.json").read_text())
    pointer = json.loads((run_dir / "checkpoints/last_checkpoint.json").read_text())
    for record in (checkpoint_manifest, training_step):
        assert record["run_mode"] == "continuation"
        assert record["global_step"] == 37_332
        assert record["phase_step"] == 2
        assert record["phase_start_global_step"] == 37_330
        assert record["source_identity_sha256"] == source_identity_sha
        assert record["scheduler_step_axis"] == "phase_step"
    assert pointer == {
        "step": 37_332,
        "global_step": 37_332,
        "phase_step": 2,
        "path": "checkpoints/step-037332",
        "source_identity_sha256": source_identity_sha,
    }


def test_new_continuation_run_can_resume_only_the_same_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_checkpoint_writers(monkeypatch)
    run_dir = tmp_path / "continuation-run"
    run_dir.mkdir()
    stats = tmp_path / "stats.json"
    stats.write_text("{}", encoding="utf-8")
    source_identity_sha = "s" * 64
    checkpoint = harness.atomic_save_checkpoint(
        run_dir=run_dir,
        step=37_332,
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
        wandb_run_id="continuation-wandb-id",
        phase_start_global_step=37_330,
        source_identity_sha256=source_identity_sha,
    )
    expected = {
        "config_sha256": "c" * 64,
        "monitor_subset_sha256": "m" * 64,
        "world_size": 2,
        "planned_total_steps": 111_990,
        "dataset_profile_sha256": "d" * 64,
        "run_mode": "continuation",
        "phase_start_global_step": 37_330,
        "planned_phase_steps": 74_660,
        "planned_final_global_step": 111_990,
        "source_identity_sha256": source_identity_sha,
    }
    _write_json(
        run_dir / "run_manifest.json",
        {
            **expected,
            "wandb_run_id": "continuation-wandb-id",
        },
    )

    resolved, step = harness._resolve_resume_checkpoint(run_dir, expected)
    assert resolved == checkpoint
    assert step == 37_332

    checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
    checkpoint_manifest["source_identity_sha256"] = "x" * 64
    _write_json(checkpoint_manifest_path, checkpoint_manifest)
    with pytest.raises(harness.TrainingContractError, match="source_identity_sha256"):
        harness._resolve_resume_checkpoint(run_dir, expected)


def test_new_continuation_run_resume_loads_its_new_scheduler_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoints/step-037332"
    state_dir = checkpoint / "training_state"
    state_dir.mkdir(parents=True)
    _write_json(
        state_dir / "training_step.json",
        {
            "step": 37_332,
            "global_step": 37_332,
            "phase_step": 2,
            "phase_start_global_step": 37_330,
            "source_identity_sha256": "s" * 64,
            "scheduler_step_axis": "phase_step",
            "run_mode": "continuation",
            "world_size": 2,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
    )
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        harness,
        "load_pi0_full_checkpoint_weights",
        lambda _policy, path: calls.append(("weights", path)) or {"strict": True},
    )
    monkeypatch.setattr(
        harness,
        "load_optimizer_state",
        lambda _optimizer, path: calls.append(("optimizer", path)),
    )
    monkeypatch.setattr(
        harness,
        "load_scheduler_state",
        lambda _scheduler, path: calls.append(("scheduler", path)),
    )

    harness._load_resume_training_state(
        checkpoint,
        policy=object(),
        optimizer=object(),
        scheduler=object(),
        world_size=2,
        batch_size=1,
        gradient_accumulation_steps=1,
        expected_step=37_332,
        phase_start_global_step=37_330,
        source_identity_sha256="s" * 64,
    )

    assert calls == [
        ("weights", checkpoint / "pretrained_model"),
        ("optimizer", state_dir),
        ("scheduler", state_dir),
    ]

    training_state = json.loads((state_dir / "training_step.json").read_text())
    training_state["source_identity_sha256"] = "x" * 64
    _write_json(state_dir / "training_step.json", training_state)
    with pytest.raises(harness.TrainingContractError, match="source_identity_sha256"):
        harness._load_resume_training_state(
            checkpoint,
            policy=object(),
            optimizer=object(),
            scheduler=object(),
            world_size=2,
            batch_size=1,
            gradient_accumulation_steps=1,
            expected_step=37_332,
            phase_start_global_step=37_330,
            source_identity_sha256="s" * 64,
        )


def test_continuation_scheduler_resume_state_uses_phase_step_axis() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-6)
    scheduler = harness.build_continuation_cosine_scheduler(
        optimizer,
        phase_steps=10,
        expected_current_lr=1.0e-6,
        start_lr=1.0e-6,
        final_lr=1.0e-7,
    )
    for _ in range(2):
        optimizer.step()
        scheduler.step()

    lr = harness.validate_continuation_scheduler_state(
        optimizer,
        scheduler,
        expected_phase_step=2,
        start_lr=1.0e-6,
    )
    assert lr == pytest.approx(scheduler.get_last_lr()[0])

    with pytest.raises(harness.TrainingContractError, match="phase mismatch"):
        harness.validate_continuation_scheduler_state(
            optimizer,
            scheduler,
            expected_phase_step=37_332,
            start_lr=1.0e-6,
        )


def test_continuation_optimizer_scheduler_real_state_roundtrip(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-6, betas=(0.9, 0.95))
    scheduler = harness.build_continuation_cosine_scheduler(
        optimizer,
        phase_steps=8,
        expected_current_lr=1.0e-6,
        start_lr=1.0e-6,
        final_lr=1.0e-7,
    )
    for _ in range(2):
        parameter.grad = torch.tensor([0.25, -0.5])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    expected_lr = float(scheduler.get_last_lr()[0])
    expected_moments = {
        key: value.detach().clone()
        for key, value in optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
    }
    harness.save_optimizer_state(optimizer, tmp_path)
    harness.save_scheduler_state(scheduler, tmp_path)

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored_optimizer = torch.optim.AdamW(
        [restored_parameter], lr=1.0e-6, betas=(0.9, 0.95)
    )
    restored_scheduler = harness.build_continuation_cosine_scheduler(
        restored_optimizer,
        phase_steps=8,
        expected_current_lr=1.0e-6,
        start_lr=1.0e-6,
        final_lr=1.0e-7,
    )
    harness.load_optimizer_state(restored_optimizer, tmp_path)
    harness.load_scheduler_state(restored_scheduler, tmp_path)

    assert restored_scheduler.last_epoch == 2
    assert restored_scheduler.get_last_lr() == pytest.approx([expected_lr])
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    for key, value in expected_moments.items():
        torch.testing.assert_close(
            restored_optimizer.state[restored_parameter][key], value, rtol=0, atol=0
        )


def test_restart_continuation_optimizer_scheduler_real_state_roundtrip(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-6, betas=(0.9, 0.95))
    parameter.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler = harness.build_continuation_cosine_scheduler(
        optimizer,
        phase_steps=8,
        expected_current_lr=1.0e-6,
        start_lr=1.0e-5,
        final_lr=1.0e-6,
    )
    for _ in range(3):
        parameter.grad = torch.tensor([0.25, -0.5])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    expected_lr = float(scheduler.get_last_lr()[0])
    expected_moments = {
        key: value.detach().clone()
        for key, value in optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
    }
    harness.save_optimizer_state(optimizer, tmp_path)
    harness.save_scheduler_state(scheduler, tmp_path)

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored_optimizer = torch.optim.AdamW(
        [restored_parameter], lr=1.0e-5, betas=(0.9, 0.95)
    )
    restored_scheduler = harness.build_continuation_cosine_scheduler(
        restored_optimizer,
        phase_steps=8,
        expected_current_lr=1.0e-5,
        start_lr=1.0e-5,
        final_lr=1.0e-6,
    )
    harness.load_optimizer_state(restored_optimizer, tmp_path)
    harness.load_scheduler_state(restored_scheduler, tmp_path)

    assert restored_scheduler.last_epoch == 3
    assert restored_scheduler.get_last_lr() == pytest.approx([expected_lr])
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    for key, value in expected_moments.items():
        torch.testing.assert_close(
            restored_optimizer.state[restored_parameter][key], value, rtol=0, atol=0
        )
