from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from franka_eef_pipeline import pi0_training
from franka_eef_pipeline.pi0_training import (
    ACTION_DIM,
    CANONICAL_PI0_IMAGE_KEYS,
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
    PALIGEMMA_EMBED_TOKENS_KEY,
    PALIGEMMA_LM_HEAD_KEY,
    PI0CheckpointLoadError,
    PI0EffectiveStatsError,
    PI0TrainingContractError,
    STATE_DIM,
    UNUSED_EXPERT_LM_HEAD_KEY,
    assert_canonical_pi0_training_graph,
    assert_full_parameter_training,
    build_pi0_full_finetune_config,
    canonicalize_pi0_full_training_graph,
    gradient_coverage_summary,
    load_pi0_full_checkpoint_weights,
    load_pi0_full_policy_and_processors,
    prepare_effective_pi0_stats,
    save_pi0_full_checkpoint,
)


def _base_config_payload() -> dict[str, Any]:
    image_features = {
        key: {"type": "VISUAL", "shape": [3, 224, 224]}
        for key in CANONICAL_PI0_IMAGE_KEYS
    }
    return {
        "type": "pi0",
        "n_obs_steps": 1,
        "input_features": {
            **image_features,
            "observation.state": {"type": "STATE", "shape": [32]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [32]}},
        "device": "cpu",
        "use_amp": False,
        "push_to_hub": True,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "dtype": "float32",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_action_dim": 32,
        "max_state_dim": 32,
        "image_resolution": [224, 224],
        "gradient_checkpointing": False,
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "use_relative_actions": False,
    }


class _TinyNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))


class _TinySelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(2, 2, bias=False)


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(2, 2, bias=False)
        self.up_proj = nn.Linear(2, 2, bias=False)
        self.down_proj = nn.Linear(2, 2, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinySelfAttention()
        self.post_attention_layernorm = _TinyNorm()
        self.mlp = _TinyMLP()


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(5, 3)
        self.layers = nn.ModuleList([_TinyLayer()])
        self.norm = _TinyNorm()


class _TinyPaliModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyLanguageModel()


class _TinyPaliGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyPaliModel()
        self.lm_head = nn.Linear(3, 5, bias=False)


class _TinyExpertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = None
        self.layer = nn.Linear(2, 2)


class _TinyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyExpertModel()
        self.lm_head = nn.Linear(2, 5, bias=False)


class _TinyCombined(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma = _TinyPaliGemma()
        self.gemma_expert = _TinyExpert()


class _TinyCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = _TinyCombined()
        self.action_in_proj = nn.Linear(3, 2)
        self.state_proj = nn.Linear(2, 2)
        self.gradient_checkpointing_enabled = True


class _TinyPolicy(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = _TinyCore()

    def save_pretrained(self, save_directory: str | Path, *, push_to_hub: bool = False) -> None:
        raise AssertionError("Checkpoint helper must save policy.model with canonical unprefixed keys")


_TINY_STRUCTURAL_ACTION_UNREACHABLE_SPECS = (
    (
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.o_proj.weight",
        (2, 2),
    ),
    (
        "paligemma_with_expert.paligemma.model.language_model.layers.0.post_attention_layernorm.weight",
        (2,),
    ),
    (
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj.weight",
        (2, 2),
    ),
    (
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.up_proj.weight",
        (2, 2),
    ),
    (
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight",
        (2, 2),
    ),
    ("paligemma_with_expert.paligemma.model.language_model.norm.weight", (2,)),
)


@pytest.fixture(autouse=True)
def _use_tiny_structural_action_unreachable_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scale only the unit-test graph while exercising the same strict ledger logic."""

    monkeypatch.setattr(
        pi0_training,
        "STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS",
        _TINY_STRUCTURAL_ACTION_UNREACHABLE_SPECS,
    )
    monkeypatch.setattr(pi0_training, "STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT", 6)
    monkeypatch.setattr(pi0_training, "STRUCTURAL_ACTION_UNREACHABLE_NUMEL", 20)


class _TinyProcessor:
    def __init__(self, name: str) -> None:
        self.name = name

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        push_to_hub: bool,
        config_filename: str,
    ) -> None:
        assert push_to_hub is False
        Path(save_directory, config_filename).write_text(
            json.dumps({"name": self.name}),
            encoding="utf-8",
        )


def _write_local_checkpoint(path: Path, *, complete_weights: bool = True) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_base_config_payload()), encoding="utf-8")
    core = _TinyCore()
    with torch.no_grad():
        for index, parameter in enumerate(core.parameters(), start=1):
            parameter.fill_(index / 10)
    state = core.state_dict()
    # The canonical public pi0_base stores the PaliGemma tied tensor under the
    # LM-head name only and still contains the unused expert wrapper head.
    state.pop(PALIGEMMA_EMBED_TOKENS_KEY)
    if not complete_weights:
        state = {"action_in_proj.bias": state["action_in_proj.bias"]}
    save_file(state, path / "model.safetensors")


def _effective_stats() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "observation_fps": 15,
        "action_fps": 15,
        "chunk_size": 50,
        "source_dataset_hash": "test",
        "observation.state": {
            "mean": [0.1] * STATE_DIM,
            "std": [0.2] * STATE_DIM,
            "count": [11],
        },
        "action": {
            "mean": [0.3] * ACTION_DIM,
            "std": [0.4] * ACTION_DIM,
            "count": [550],
        },
    }


def test_build_config_preserves_canonical_images_and_sets_full_training_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pi0_base"
    _write_local_checkpoint(checkpoint)

    config = build_pi0_full_finetune_config(checkpoint, device="cpu")

    assert tuple(config.image_features) == CANONICAL_PI0_IMAGE_KEYS
    assert config.input_features["observation.state"].shape == (STATE_DIM,)
    assert config.output_features["action"].shape == (ACTION_DIM,)
    assert config.max_state_dim == MAX_STATE_DIM
    assert config.max_action_dim == MAX_ACTION_DIM
    assert config.dtype == "bfloat16"
    assert config.gradient_checkpointing is True
    assert config.freeze_vision_encoder is False
    assert config.train_expert_only is False
    assert config.use_relative_actions is False
    assert config.use_peft is False
    assert config.push_to_hub is False


def test_effective_stats_only_inject_model_visible_10d_and_7d() -> None:
    prepared, report = prepare_effective_pi0_stats(_effective_stats())

    assert tuple(prepared) == ("observation.state", "action")
    assert prepared["observation.state"]["mean"].shape == (STATE_DIM,)
    assert prepared["action"]["std"].shape == (ACTION_DIM,)
    assert report["counts"] == {"observation.state": 11, "action": 550}
    assert "schema_version" in report["ignored_top_level_keys"]

    invalid = _effective_stats()
    invalid["action"]["mean"] = [0.0] * 8
    with pytest.raises(PI0EffectiveStatsError, match="shape"):
        prepare_effective_pi0_stats(invalid)


def test_full_loader_is_strict_numerically_verified_and_injects_official_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "pi0_base"
    _write_local_checkpoint(checkpoint)
    captured: dict[str, Any] = {}

    def fake_processors(*, config, dataset_stats):
        captured["config"] = config
        captured["dataset_stats"] = dataset_stats
        return "preprocessor", "postprocessor"

    monkeypatch.setattr(pi0_training, "PI0Policy", _TinyPolicy)
    monkeypatch.setattr(pi0_training, "make_pi0_pre_post_processors", fake_processors)

    policy, preprocessor, postprocessor, report = load_pi0_full_policy_and_processors(
        checkpoint,
        _effective_stats(),
        "cpu",
    )

    assert isinstance(policy, _TinyPolicy)
    assert policy.training is True
    assert (preprocessor, postprocessor) == ("preprocessor", "postprocessor")
    assert tuple(captured["dataset_stats"]) == ("observation.state", "action")
    assert report["pretrained"]["strict"] is True
    assert report["pretrained"]["verified_tensor"]["exact_after_model_dtype_cast"] is True
    assert tuple(report["pretrained"]["dropped_checkpoint_keys"]) == (UNUSED_EXPERT_LM_HEAD_KEY,)
    assert report["parameters"]["trainable_fraction"] == 1.0
    assert report["parameters"]["paligemma_lm_head_tied"] is True
    assert report["parameters"]["expert_lm_head_pruned"] is True
    assert (
        policy.model.paligemma_with_expert.paligemma.lm_head.weight
        is policy.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
    )
    assert not hasattr(policy.model.paligemma_with_expert.gemma_expert, "lm_head")
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as tensors:
        expected_bias = tensors.get_tensor("action_in_proj.bias")
    assert torch.equal(policy.model.action_in_proj.bias, expected_bias)


def test_full_loader_never_falls_back_to_random_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "broken_pi0_base"
    _write_local_checkpoint(checkpoint, complete_weights=False)
    monkeypatch.setattr(pi0_training, "PI0Policy", _TinyPolicy)

    with pytest.raises(PI0CheckpointLoadError, match="random weights will not be returned"):
        load_pi0_full_policy_and_processors(checkpoint, _effective_stats(), "cpu")


def test_strict_loader_accepts_the_symmetric_physical_tied_embedding_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "symmetric_alias"
    _write_local_checkpoint(checkpoint)
    state = load_file(checkpoint / "model.safetensors")
    state[PALIGEMMA_EMBED_TOKENS_KEY] = state.pop(PALIGEMMA_LM_HEAD_KEY)
    save_file(state, checkpoint / "replacement.safetensors")
    (checkpoint / "replacement.safetensors").replace(checkpoint / "model.safetensors")
    monkeypatch.setattr(pi0_training, "PI0Policy", _TinyPolicy)
    monkeypatch.setattr(
        pi0_training,
        "make_pi0_pre_post_processors",
        lambda *, config, dataset_stats: ("pre", "post"),
    )

    _, _, _, report = load_pi0_full_policy_and_processors(
        checkpoint,
        _effective_stats(),
        "cpu",
    )

    assert report["pretrained"]["resolved_checkpoint_aliases"] == {
        PALIGEMMA_LM_HEAD_KEY: PALIGEMMA_EMBED_TOKENS_KEY
    }


def test_strict_loader_rejects_wrong_shape_before_dropping_expert_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "wrong_expert_head"
    _write_local_checkpoint(checkpoint)
    state = load_file(checkpoint / "model.safetensors")
    state[UNUSED_EXPERT_LM_HEAD_KEY] = torch.zeros(6, 2)
    save_file(state, checkpoint / "replacement.safetensors")
    (checkpoint / "replacement.safetensors").replace(checkpoint / "model.safetensors")
    monkeypatch.setattr(pi0_training, "PI0Policy", _TinyPolicy)

    with pytest.raises(PI0CheckpointLoadError, match="wrong shape.*safely dropped"):
        load_pi0_full_policy_and_processors(checkpoint, _effective_stats(), "cpu")


def test_strict_loader_drops_no_unknown_unexpected_checkpoint_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "unknown_tensor"
    _write_local_checkpoint(checkpoint)
    state = load_file(checkpoint / "model.safetensors")
    state["paligemma_with_expert.gemma_expert.not_a_canonical_head.weight"] = torch.ones(1)
    save_file(state, checkpoint / "replacement.safetensors")
    (checkpoint / "replacement.safetensors").replace(checkpoint / "model.safetensors")
    monkeypatch.setattr(pi0_training, "PI0Policy", _TinyPolicy)

    with pytest.raises(PI0CheckpointLoadError, match="unexpected=.*not_a_canonical_head"):
        load_pi0_full_policy_and_processors(checkpoint, _effective_stats(), "cpu")


def test_explicit_resume_loader_rejects_policy_wrapper_model_prefix(tmp_path: Path) -> None:
    checkpoint = tmp_path / "prefixed_checkpoint"
    _write_local_checkpoint(checkpoint)
    config = build_pi0_full_finetune_config(checkpoint, device="cpu")
    policy = _TinyPolicy(config)
    canonicalize_pi0_full_training_graph(policy)
    prefixed = {f"model.{key}": value for key, value in _TinyCore().state_dict().items()}
    (checkpoint / "model.safetensors").unlink()
    save_file(prefixed, checkpoint / "model.safetensors")

    with pytest.raises(PI0CheckpointLoadError, match="model\\.` keys"):
        load_pi0_full_checkpoint_weights(policy, checkpoint)


def test_parameter_assertion_and_gradient_coverage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pi0_base"
    _write_local_checkpoint(checkpoint)
    config = build_pi0_full_finetune_config(checkpoint, device="cpu")
    policy = _TinyPolicy(config)
    graph_report = canonicalize_pi0_full_training_graph(policy)

    parameter_report = assert_full_parameter_training(policy)
    assert parameter_report["total_parameters"] == parameter_report["trainable_parameters"]
    assert parameter_report["training_graph"] == graph_report
    assert graph_report["deduplicated_tied_parameter_count"] == 15
    assert graph_report["pruned_unused_parameter_count"] == 10
    assert graph_report["parameters_removed_from_optimizer"] == 25
    assert graph_report["schema_version"] == 2
    structural = graph_report["structural_action_unreachable"]
    assert structural["parameters_retained_and_trainable"] is True
    assert structural["ddp_requires_find_unused_parameters"] is True
    assert structural["tensor_count"] == 6
    assert structural["numel"] == 20
    assert [entry["shape"] for entry in structural["parameters"]] == [
        [2, 2],
        [2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2],
    ]

    loss = sum(parameter.square().sum() for parameter in policy.parameters())
    loss.backward()
    gradient_report = gradient_coverage_summary(policy)
    assert gradient_report["overall"]["gradient_tensor_coverage"] == 1.0
    assert gradient_report["overall"]["gradient_numel_coverage"] == 1.0
    assert gradient_report["overall"]["all_gradients_finite"] is True
    assert "state_action_projections" in gradient_report["groups"]

    policy.model.state_proj.weight.requires_grad_(False)
    with pytest.raises(PI0TrainingContractError, match="frozen"):
        assert_full_parameter_training(policy)


def test_checkpoint_helper_saves_local_policy_processors_and_manifests_and_strictly_reloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_local_checkpoint(source)
    config = build_pi0_full_finetune_config(source, device="cpu")
    policy = _TinyPolicy(config)
    canonicalize_pi0_full_training_graph(policy)
    destination = tmp_path / "saved"

    manifest = save_pi0_full_checkpoint(
        policy,
        _TinyProcessor("pre"),
        _TinyProcessor("post"),
        destination,
        geometry_manifest={"rotation": "so3_log_body", "translation": "base"},
        stats_manifest=_effective_stats(),
        load_report={"strict": True},
    )

    assert manifest["hub_upload"] is False
    assert manifest["wandb_artifact_upload"] is False
    for filename in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "franka_eef_geometry_manifest.json",
        "pi0_eef_stats.json",
        "franka_pi0_checkpoint_manifest.json",
    ):
        assert (destination / filename).is_file()

    with safe_open(destination / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        saved_keys = tuple(checkpoint.keys())
    assert saved_keys
    assert all(not key.startswith("model.") for key in saved_keys)
    assert UNUSED_EXPERT_LM_HEAD_KEY not in saved_keys
    assert len(set(saved_keys).intersection({PALIGEMMA_LM_HEAD_KEY, PALIGEMMA_EMBED_TOKENS_KEY})) == 1
    assert manifest["training_graph"]["paligemma_lm_head_tied"] is True
    assert manifest["training_graph"]["expert_lm_head_pruned"] is True
    assert manifest["training_graph"]["structural_action_unreachable"]["numel"] == 20

    monkeypatch.setattr(pi0_training, "PI0Policy", _TinyPolicy)
    monkeypatch.setattr(
        pi0_training,
        "make_pi0_pre_post_processors",
        lambda *, config, dataset_stats: ("reloaded_pre", "reloaded_post"),
    )
    reloaded, preprocessor, postprocessor, load_report = load_pi0_full_policy_and_processors(
        destination,
        destination / "pi0_eef_stats.json",
        "cpu",
    )
    assert isinstance(reloaded, _TinyPolicy)
    assert (preprocessor, postprocessor) == ("reloaded_pre", "reloaded_post")
    assert load_report["pretrained"]["strict"] is True
    assert load_report["pretrained"]["missing_keys"] == []
    assert load_report["pretrained"]["unexpected_keys"] == []
    assert load_report["pretrained"]["dropped_checkpoint_keys"] == {}
    assert_canonical_pi0_training_graph(reloaded)

    manifest_path = destination / "franka_pi0_checkpoint_manifest.json"
    tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered_manifest["training_graph"]["expert_lm_head_pruned"] = False
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(PI0CheckpointLoadError, match="manifest training_graph"):
        load_pi0_full_checkpoint_weights(reloaded, destination)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_pi0_full_checkpoint(
            policy,
            _TinyProcessor("pre"),
            _TinyProcessor("post"),
            destination,
            geometry_manifest={},
            stats_manifest=_effective_stats(),
        )
