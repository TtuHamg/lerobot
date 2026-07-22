"""Audited component-wise trainability for the project PI0 policy.

The public PI0 flags only distinguish full training, a frozen vision tower,
and an entirely frozen PaliGemma.  Franka experiments need a slightly finer
partition, most notably the ability to train the PaliGemma transformer while
keeping both the vision tower and the tied text embedding frozen.

This module deliberately selects parameters by module ownership and Parameter
identity.  It never accepts arbitrary name prefixes or regular expressions.
The six components must form a disjoint, exhaustive partition of the live
canonical PI0 graph; a model-version drift therefore fails before an optimizer
is constructed instead of silently training an unintended tensor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch


PI0_TRAINABLE_COMPONENTS = (
    "vision_encoder",
    "multimodal_projector",
    "text_embedding",
    "paligemma_transformer",
    "action_expert",
    "action_projections",
)

PI0_TRAINABILITY_PRESETS = {
    "full": PI0_TRAINABLE_COMPONENTS,
    "action_expert": (
        "action_expert",
        "action_projections",
    ),
    "action_expert_paligemma": (
        "multimodal_projector",
        "paligemma_transformer",
        "action_expert",
        "action_projections",
    ),
}

_ACTION_PROJECTION_NAMES = (
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)


class PI0TrainabilityError(RuntimeError):
    """Raised when a requested PI0 trainability contract is ambiguous or unsafe."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_pi0_trainability(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a preset/custom YAML mapping to one canonical component list."""

    if not isinstance(spec, Mapping):
        raise PI0TrainabilityError("model.trainability must be a mapping")
    unknown_keys = sorted(set(spec) - {"preset", "components"})
    if unknown_keys:
        raise PI0TrainabilityError(
            f"model.trainability has unknown keys: {unknown_keys}"
        )

    preset = spec.get("preset")
    if not isinstance(preset, str) or not preset.strip():
        raise PI0TrainabilityError("model.trainability.preset must be a non-empty string")
    preset = preset.strip()
    configured_components = spec.get("components")

    if preset == "custom":
        if (
            not isinstance(configured_components, list)
            or not configured_components
            or any(not isinstance(value, str) for value in configured_components)
        ):
            raise PI0TrainabilityError(
                "custom trainability requires a non-empty string list in components"
            )
        if len(configured_components) != len(set(configured_components)):
            raise PI0TrainabilityError("custom trainability components contain duplicates")
        unknown = sorted(set(configured_components) - set(PI0_TRAINABLE_COMPONENTS))
        if unknown:
            raise PI0TrainabilityError(
                f"unknown PI0 trainability components: {unknown}; "
                f"allowed={list(PI0_TRAINABLE_COMPONENTS)}"
            )
        selected = tuple(
            component
            for component in PI0_TRAINABLE_COMPONENTS
            if component in configured_components
        )
    else:
        if preset not in PI0_TRAINABILITY_PRESETS:
            allowed = sorted((*PI0_TRAINABILITY_PRESETS, "custom"))
            raise PI0TrainabilityError(
                f"unknown PI0 trainability preset {preset!r}; allowed={allowed}"
            )
        if configured_components is not None:
            raise PI0TrainabilityError(
                "model.trainability.components must be null/omitted unless preset=custom"
            )
        selected = PI0_TRAINABILITY_PRESETS[preset]

    return {
        "schema_version": 1,
        "preset": preset,
        "components": list(selected),
    }


def _pi0_modules(policy: torch.nn.Module) -> dict[str, Any]:
    try:
        core = policy.model
        combined = core.paligemma_with_expert
        paligemma = combined.paligemma
        pali_model = paligemma.model
        language_model = pali_model.language_model
        embedding = language_model.embed_tokens
        expert = combined.gemma_expert
        projection_modules = tuple(getattr(core, name) for name in _ACTION_PROJECTION_NAMES)
    except AttributeError as exc:
        raise PI0TrainabilityError(
            "policy does not expose the canonical PI0 PaliGemma/action-expert graph"
        ) from exc

    lm_head = getattr(paligemma, "lm_head", None)
    if not isinstance(embedding, torch.nn.Embedding):
        raise PI0TrainabilityError("PaliGemma text embedding is missing or is not nn.Embedding")
    if not isinstance(lm_head, torch.nn.Linear) or lm_head.weight is not embedding.weight:
        raise PI0TrainabilityError(
            "PaliGemma lm_head.weight must be pointer-tied to the text embedding before selection"
        )
    if hasattr(expert, "lm_head"):
        raise PI0TrainabilityError(
            "the unused action-expert LM head must be pruned before parameter selection"
        )

    return {
        "core": core,
        "combined": combined,
        "paligemma": paligemma,
        "vision_encoder": pali_model.vision_tower,
        "multimodal_projector": pali_model.multi_modal_projector,
        "text_embedding": embedding,
        "paligemma_transformer": language_model,
        "action_expert": expert,
        "action_projections": projection_modules,
    }


def _parameter_ids(value: Any) -> set[int]:
    modules = value if isinstance(value, tuple) else (value,)
    return {id(parameter) for module in modules for parameter in module.parameters()}


def pi0_parameter_groups(
    policy: torch.nn.Module,
) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Return the exhaustive six-way partition of unique policy Parameters."""

    modules = _pi0_modules(policy)
    named_parameters = list(policy.named_parameters())
    id_to_named = {id(parameter): (name, parameter) for name, parameter in named_parameters}
    if len(id_to_named) != len(named_parameters):
        raise PI0TrainabilityError("policy.named_parameters() unexpectedly returned duplicate objects")

    component_ids = {
        "vision_encoder": _parameter_ids(modules["vision_encoder"]),
        "multimodal_projector": _parameter_ids(modules["multimodal_projector"]),
        "text_embedding": _parameter_ids(modules["text_embedding"]),
        "paligemma_transformer": _parameter_ids(modules["paligemma_transformer"])
        - _parameter_ids(modules["text_embedding"]),
        "action_expert": _parameter_ids(modules["action_expert"]),
        "action_projections": _parameter_ids(modules["action_projections"]),
    }

    owners: dict[int, list[str]] = {}
    for component, parameter_ids in component_ids.items():
        if not parameter_ids:
            raise PI0TrainabilityError(f"PI0 component {component!r} contains no Parameters")
        for parameter_id in parameter_ids:
            owners.setdefault(parameter_id, []).append(component)

    overlaps = {
        id_to_named.get(parameter_id, (f"id={parameter_id}", None))[0]: components
        for parameter_id, components in owners.items()
        if len(components) != 1
    }
    if overlaps:
        raise PI0TrainabilityError(f"PI0 component parameter ownership overlaps: {overlaps}")

    live_ids = set(id_to_named)
    classified_ids = set(owners)
    missing_ids = live_ids - classified_ids
    extra_ids = classified_ids - live_ids
    if missing_ids or extra_ids:
        missing_names = sorted(id_to_named[parameter_id][0] for parameter_id in missing_ids)
        raise PI0TrainabilityError(
            "PI0 component partition does not exactly cover the live model: "
            f"unclassified={missing_names}, non_live_parameter_ids={sorted(extra_ids)}"
        )

    return {
        component: [
            (name, parameter)
            for name, parameter in named_parameters
            if owners[id(parameter)][0] == component
        ]
        for component in PI0_TRAINABLE_COMPONENTS
    }


def enforce_frozen_pi0_eval_mode(
    policy: torch.nn.Module, spec: Mapping[str, Any]
) -> None:
    """Keep frozen stateful submodules in eval mode after any policy.train() call."""

    resolved = resolve_pi0_trainability(spec)
    selected = set(resolved["components"])
    modules = _pi0_modules(policy)
    paligemma_components = {
        "vision_encoder",
        "multimodal_projector",
        "text_embedding",
        "paligemma_transformer",
    }

    if not selected.intersection(paligemma_components):
        modules["paligemma"].eval()

    if "vision_encoder" not in selected:
        modules["vision_encoder"].eval()
    if "multimodal_projector" not in selected:
        modules["multimodal_projector"].eval()
    if "text_embedding" not in selected:
        modules["text_embedding"].eval()
    if "paligemma_transformer" not in selected:
        modules["paligemma_transformer"].eval()
    if "action_expert" not in selected:
        modules["action_expert"].eval()
    if "action_projections" not in selected:
        for module in modules["action_projections"]:
            module.eval()


def assert_pi0_trainability(
    policy: torch.nn.Module,
    spec: Mapping[str, Any],
    *,
    expected_signature: str | None = None,
) -> dict[str, Any]:
    """Prove that live requires_grad flags exactly match the requested components."""

    resolved = resolve_pi0_trainability(spec)
    selected = set(resolved["components"])
    groups = pi0_parameter_groups(policy)
    group_reports: dict[str, Any] = {}
    total_numel = 0
    trainable_numel = 0
    total_tensors = 0
    trainable_tensors = 0

    mismatches: list[str] = []
    for component in PI0_TRAINABLE_COMPONENTS:
        entries = groups[component]
        expected_trainable = component in selected
        for name, parameter in entries:
            if parameter.requires_grad is not expected_trainable:
                mismatches.append(
                    f"{name}: requires_grad={parameter.requires_grad}, expected={expected_trainable}"
                )
        component_numel = sum(parameter.numel() for _, parameter in entries)
        component_trainable = sum(
            parameter.numel() for _, parameter in entries if parameter.requires_grad
        )
        component_trainable_tensors = sum(
            int(parameter.requires_grad) for _, parameter in entries
        )
        group_reports[component] = {
            "parameter_tensors": len(entries),
            "parameter_numel": component_numel,
            "trainable_tensors": component_trainable_tensors,
            "trainable_numel": component_trainable,
        }
        total_tensors += len(entries)
        total_numel += component_numel
        trainable_tensors += component_trainable_tensors
        trainable_numel += component_trainable
    if mismatches:
        raise PI0TrainabilityError(
            f"live PI0 trainability differs from requested selection: {mismatches[:20]}"
        )
    if trainable_tensors <= 0 or trainable_numel <= 0:
        raise PI0TrainabilityError("PI0 trainability selection produced no optimizer Parameters")

    # Preserve the exact optimizer order, not merely the set of names.  LeRobot
    # optimizer checkpoints address Parameters by contiguous numeric IDs, so a
    # resume must reject a code-version drift that reorders otherwise identical
    # names.
    trainable_names = [
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    ]
    signature_payload = {
        "preset": resolved["preset"],
        "components": resolved["components"],
        "trainable_parameter_names": trainable_names,
    }
    signature = _canonical_sha256(signature_payload)
    if expected_signature is not None and signature != expected_signature:
        raise PI0TrainabilityError(
            f"PI0 trainability signature mismatch: {signature} != {expected_signature}"
        )

    return {
        "schema_version": 1,
        **resolved,
        "frozen_components": [
            component for component in PI0_TRAINABLE_COMPONENTS if component not in selected
        ],
        "parameter_tensors": total_tensors,
        "parameter_numel": total_numel,
        "trainable_tensors": trainable_tensors,
        "trainable_numel": trainable_numel,
        "trainable_fraction": trainable_numel / total_numel,
        "groups": group_reports,
        "trainable_parameter_names": trainable_names,
        "trainability_signature_sha256": signature,
    }


def apply_pi0_trainability(
    policy: torch.nn.Module, spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze everything, unfreeze selected components, and return an audit report."""

    resolved = resolve_pi0_trainability(spec)
    selected = set(resolved["components"])
    groups = pi0_parameter_groups(policy)

    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for component in selected:
        for _, parameter in groups[component]:
            parameter.requires_grad_(True)

    modules = _pi0_modules(policy)
    paligemma_components = {
        "vision_encoder",
        "multimodal_projector",
        "text_embedding",
        "paligemma_transformer",
    }
    freeze_vision = "vision_encoder" not in selected
    freeze_entire_paligemma = not bool(selected.intersection(paligemma_components))
    for owner in (getattr(policy, "config", None), modules["combined"]):
        if owner is not None:
            owner.freeze_vision_encoder = freeze_vision
            owner.train_expert_only = freeze_entire_paligemma

    policy.train()
    enforce_frozen_pi0_eval_mode(policy, spec)
    return assert_pi0_trainability(policy, spec)


def trainable_pi0_parameters(
    policy: torch.nn.Module, spec: Mapping[str, Any]
) -> list[torch.nn.Parameter]:
    """Return unique trainable Parameters in stable named_parameters order."""

    report = assert_pi0_trainability(policy, spec)
    expected_names = set(report["trainable_parameter_names"])
    parameters = [
        parameter
        for name, parameter in policy.named_parameters()
        if name in expected_names
    ]
    if len(parameters) != report["trainable_tensors"] or len({id(value) for value in parameters}) != len(
        parameters
    ):
        raise PI0TrainabilityError("optimizer Parameter list is incomplete or contains duplicates")
    return parameters


def pi0_gradient_coverage_summary(
    policy: torch.nn.Module,
    spec: Mapping[str, Any],
    *,
    inspect_values: bool = True,
) -> dict[str, Any]:
    """Summarize first-backward gradients using the same six ownership groups."""

    trainability = assert_pi0_trainability(policy, spec)
    groups = pi0_parameter_groups(policy)
    selected = set(trainability["components"])
    reports: dict[str, Any] = {}
    all_missing: list[str] = []

    for component in PI0_TRAINABLE_COMPONENTS:
        report = {
            "selected": component in selected,
            "parameter_tensors": len(groups[component]),
            "parameter_numel": sum(parameter.numel() for _, parameter in groups[component]),
            "trainable_tensors": 0,
            "trainable_numel": 0,
            "gradient_tensors": 0,
            "gradient_numel": 0,
            "nonzero_gradient_tensors": 0,
            "all_gradients_finite": True,
            "missing_gradient_names": [],
            "unexpected_frozen_gradient_names": [],
        }
        for name, parameter in groups[component]:
            if parameter.requires_grad:
                report["trainable_tensors"] += 1
                report["trainable_numel"] += parameter.numel()
                if parameter.grad is None:
                    report["missing_gradient_names"].append(name)
                    all_missing.append(name)
                    continue
                gradient = parameter.grad.detach()
                report["gradient_tensors"] += 1
                report["gradient_numel"] += gradient.numel()
                if inspect_values:
                    report["all_gradients_finite"] &= bool(torch.isfinite(gradient).all())
                    report["nonzero_gradient_tensors"] += int(bool(torch.count_nonzero(gradient)))
            elif parameter.grad is not None:
                report["unexpected_frozen_gradient_names"].append(name)
        reports[component] = report

    return {
        "schema_version": 1,
        "inspect_values": inspect_values,
        "trainability_signature_sha256": trainability["trainability_signature_sha256"],
        "trainable_components": trainability["components"],
        "missing_gradient_names": sorted(all_missing),
        "groups": reports,
    }


def require_pi0_gradient_coverage(
    report: Mapping[str, Any],
    *,
    allowed_missing_parameter_names: Sequence[str],
) -> None:
    """Fail unless every selected/reachable component receives finite gradients."""

    groups = report.get("groups")
    if not isinstance(groups, Mapping):
        raise PI0TrainabilityError("gradient report has no component groups")
    if report.get("inspect_values") is not True:
        raise PI0TrainabilityError(
            "gradient coverage enforcement requires inspect_values=True"
        )
    expected_missing = set(allowed_missing_parameter_names)
    actual_missing = set(report.get("missing_gradient_names", ()))
    failures: dict[str, Any] = {}
    if actual_missing != expected_missing:
        failures["missing_gradients"] = {
            "unexpected": sorted(actual_missing - expected_missing),
            "expected_but_present": sorted(expected_missing - actual_missing),
        }

    for component in PI0_TRAINABLE_COMPONENTS:
        group = groups.get(component)
        if not isinstance(group, Mapping):
            failures[component] = "missing component report"
            continue
        selected = bool(group.get("selected"))
        if group.get("unexpected_frozen_gradient_names"):
            failures[f"{component}.frozen_gradients"] = group[
                "unexpected_frozen_gradient_names"
            ]
        if not selected:
            if int(group.get("trainable_tensors", -1)) != 0:
                failures[component] = "frozen component contains trainable Parameters"
            continue
        reasons: list[str] = []
        if int(group.get("trainable_tensors", 0)) <= 0:
            reasons.append("selected component has no trainable Parameters")
        if int(group.get("gradient_tensors", 0)) <= 0:
            reasons.append("no gradients reached selected component")
        if group.get("all_gradients_finite") is not True:
            reasons.append("non-finite gradient")
        if int(group.get("nonzero_gradient_tensors", 0)) <= 0:
            reasons.append("all gradients are zero")
        component_unexpected_missing = set(group.get("missing_gradient_names", ())) - expected_missing
        if component_unexpected_missing:
            reasons.append(
                f"unexpected missing gradients: {sorted(component_unexpected_missing)}"
            )
        if reasons:
            failures[component] = reasons

    if failures:
        raise PI0TrainabilityError(
            f"first-backward PI0 gradient coverage gate failed: {failures}"
        )


__all__ = [
    "PI0_TRAINABILITY_PRESETS",
    "PI0_TRAINABLE_COMPONENTS",
    "PI0TrainabilityError",
    "apply_pi0_trainability",
    "assert_pi0_trainability",
    "enforce_frozen_pi0_eval_mode",
    "pi0_gradient_coverage_summary",
    "pi0_parameter_groups",
    "require_pi0_gradient_coverage",
    "resolve_pi0_trainability",
    "trainable_pi0_parameters",
]
