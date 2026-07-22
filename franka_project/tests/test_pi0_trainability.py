from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from franka_eef_pipeline.pi0_trainability import (
    PI0_TRAINABLE_COMPONENTS,
    PI0TrainabilityError,
    apply_pi0_trainability,
    assert_pi0_trainability,
    enforce_frozen_pi0_eval_mode,
    pi0_gradient_coverage_summary,
    pi0_parameter_groups,
    require_pi0_gradient_coverage,
    resolve_pi0_trainability,
    trainable_pi0_parameters,
)


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(7, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)


class _TinyPaliModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.2))
        self.multi_modal_projector = nn.Linear(4, 4)
        self.language_model = _TinyLanguageModel()


class _TinyPaliGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyPaliModel()
        self.lm_head = nn.Linear(4, 7, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight


class _TinyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))


class _TinyCombined(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma = _TinyPaliGemma()
        self.gemma_expert = _TinyExpert()
        self.freeze_vision_encoder = False
        self.train_expert_only = False


class _TinyCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = _TinyCombined()
        self.state_proj = nn.Linear(4, 4)
        self.action_in_proj = nn.Linear(4, 4)
        self.action_out_proj = nn.Linear(4, 4)
        self.action_time_mlp_in = nn.Linear(4, 4)
        self.action_time_mlp_out = nn.Linear(4, 4)


class _TinyConfig(SimpleNamespace):
    def save_pretrained(self, destination: str | Path, *, push_to_hub: bool) -> None:
        assert push_to_hub is False
        Path(destination, "config.json").write_text(
            json.dumps(
                {
                    "freeze_vision_encoder": self.freeze_vision_encoder,
                    "train_expert_only": self.train_expert_only,
                }
            ),
            encoding="utf-8",
        )


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _TinyConfig(
            freeze_vision_encoder=False,
            train_expert_only=False,
        )
        self.model = _TinyCore()
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(0.25)


def _spec(preset: str, components: list[str] | None = None) -> dict[str, object]:
    return {"preset": preset, "components": components}


def test_six_component_partition_is_disjoint_and_exhaustive() -> None:
    policy = _TinyPolicy()

    groups = pi0_parameter_groups(policy)

    assert tuple(groups) == PI0_TRAINABLE_COMPONENTS
    grouped = [parameter for entries in groups.values() for _, parameter in entries]
    assert len(grouped) == len(list(policy.parameters()))
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    embedding = policy.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
    lm_head = policy.model.paligemma_with_expert.paligemma.lm_head
    assert embedding.weight is lm_head.weight
    assert groups["text_embedding"][0][1] is embedding.weight


@pytest.mark.parametrize(
    ("preset", "expected_components"),
    [
        ("full", set(PI0_TRAINABLE_COMPONENTS)),
        ("action_expert", {"action_expert", "action_projections"}),
        (
            "action_expert_paligemma",
            {
                "multimodal_projector",
                "paligemma_transformer",
                "action_expert",
                "action_projections",
            },
        ),
    ],
)
def test_presets_apply_exact_requires_grad_sets(
    preset: str, expected_components: set[str]
) -> None:
    policy = _TinyPolicy()
    spec = _spec(preset)

    report = apply_pi0_trainability(policy, spec)
    groups = pi0_parameter_groups(policy)

    assert set(report["components"]) == expected_components
    for component, entries in groups.items():
        assert {parameter.requires_grad for _, parameter in entries} == {
            component in expected_components
        }
    optimizer_parameters = trainable_pi0_parameters(policy, spec)
    assert len(optimizer_parameters) == report["trainable_tensors"]
    assert len({id(parameter) for parameter in optimizer_parameters}) == len(
        optimizer_parameters
    )
    assert report["trainable_parameter_names"] == [
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    ]


def test_action_expert_paligemma_freezes_vision_and_tied_text_embedding() -> None:
    policy = _TinyPolicy()
    spec = _spec("action_expert_paligemma")

    report = apply_pi0_trainability(policy, spec)
    combined = policy.model.paligemma_with_expert
    embedding = combined.paligemma.model.language_model.embed_tokens

    assert embedding.weight.requires_grad is False
    assert combined.paligemma.lm_head.weight.requires_grad is False
    assert combined.paligemma.model.vision_tower.training is False
    assert combined.paligemma.model.multi_modal_projector.weight.requires_grad is True
    assert combined.paligemma.model.language_model.layers[0].weight.requires_grad is True
    assert combined.gemma_expert.model[0].weight.requires_grad is True
    assert report["frozen_components"] == ["vision_encoder", "text_embedding"]
    assert policy.config.freeze_vision_encoder is True
    assert policy.config.train_expert_only is False

    policy.eval()
    policy.train()
    enforce_frozen_pi0_eval_mode(policy, spec)
    assert combined.paligemma.model.vision_tower.training is False


def test_action_expert_mode_keeps_entire_paligemma_in_eval() -> None:
    policy = _TinyPolicy()
    spec = _spec("action_expert")

    apply_pi0_trainability(policy, spec)
    combined = policy.model.paligemma_with_expert

    assert policy.config.train_expert_only is True
    assert combined.train_expert_only is True
    assert combined.paligemma.training is False
    assert combined.gemma_expert.training is True


def test_custom_components_are_canonicalized_and_unknown_values_fail() -> None:
    resolved = resolve_pi0_trainability(
        _spec("custom", ["action_projections", "vision_encoder"])
    )
    assert resolved["components"] == ["vision_encoder", "action_projections"]

    with pytest.raises(PI0TrainabilityError, match="unknown"):
        resolve_pi0_trainability(_spec("custom", ["not_a_component"]))
    with pytest.raises(PI0TrainabilityError, match="null/omitted"):
        resolve_pi0_trainability(_spec("full", ["action_expert"]))


def test_unclassified_model_parameter_fails_closed() -> None:
    policy = _TinyPolicy()
    policy.model.unclassified = nn.Linear(1, 1)

    with pytest.raises(PI0TrainabilityError, match="unclassified"):
        pi0_parameter_groups(policy)


def test_component_gradient_gate_accepts_only_explicit_missing_parameters() -> None:
    policy = _TinyPolicy()
    spec = _spec("action_expert")
    apply_pi0_trainability(policy, spec)
    loss = sum(parameter.square().sum() for parameter in trainable_pi0_parameters(policy, spec))
    loss.backward()

    report = pi0_gradient_coverage_summary(policy, spec)
    require_pi0_gradient_coverage(report, allowed_missing_parameter_names=[])

    missing_name, missing_parameter = next(
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
    missing_parameter.grad = None
    report = pi0_gradient_coverage_summary(policy, spec)
    with pytest.raises(PI0TrainabilityError, match="missing_gradients"):
        require_pi0_gradient_coverage(report, allowed_missing_parameter_names=[])
    require_pi0_gradient_coverage(
        report, allowed_missing_parameter_names=[missing_name]
    )


def test_live_signature_detects_trainability_drift() -> None:
    policy = _TinyPolicy()
    spec = _spec("action_expert")
    report = apply_pi0_trainability(policy, spec)
    trainable_pi0_parameters(policy, spec)[0].requires_grad_(False)

    with pytest.raises(PI0TrainabilityError, match="differs"):
        assert_pi0_trainability(
            policy,
            spec,
            expected_signature=report["trainability_signature_sha256"],
        )
