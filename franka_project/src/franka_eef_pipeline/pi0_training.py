"""Strict project-local PI0 loading helpers for Franka EEF full finetuning.

This module deliberately does not contain a training loop.  It owns the narrow
boundary between the canonical local ``lerobot/pi0_base`` checkpoint and the
Franka policy contract:

* keep the checkpoint's three canonical image slots;
* expose a 10D state and a 7D action to the official PI0 processors while the
  model remains padded to 32D internally;
* fail closed when pretrained weights or effective statistics do not match;
* prove that the load changed model tensors to the checkpoint values; and
* assert that the policy is a true full-parameter (non-PEFT/LoRA) model.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


CANONICAL_PI0_IMAGE_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
STATE_DIM = 10
ACTION_DIM = 7
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
EFFECTIVE_STATS_KEYS = (OBS_STATE, ACTION)
PI0_CORE_WEIGHTS_NAMESPACE = "pi0_core_unprefixed"
PALIGEMMA_LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"
PALIGEMMA_EMBED_TOKENS_KEY = (
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
)
UNUSED_EXPERT_LM_HEAD_KEY = "paligemma_with_expert.gemma_expert.lm_head.weight"
# PI0's action loss consumes only the joint transformer's suffix output.  In
# the final PaliGemma layer there is no subsequent layer in which the updated
# prefix can become K/V for the suffix, so these six registered Parameters are
# permanently action-loss unreachable.  They remain trainable Parameters (no
# freezing/pruning); DDP and the first-backward gate use this exact ledger as a
# fail-closed structural exception.
STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS = (
    (
        "paligemma_with_expert.paligemma.model.language_model.layers.17.self_attn.o_proj.weight",
        (2048, 2048),
    ),
    (
        "paligemma_with_expert.paligemma.model.language_model.layers.17.post_attention_layernorm.weight",
        (2048,),
    ),
    (
        "paligemma_with_expert.paligemma.model.language_model.layers.17.mlp.gate_proj.weight",
        (16384, 2048),
    ),
    (
        "paligemma_with_expert.paligemma.model.language_model.layers.17.mlp.up_proj.weight",
        (16384, 2048),
    ),
    (
        "paligemma_with_expert.paligemma.model.language_model.layers.17.mlp.down_proj.weight",
        (2048, 16384),
    ),
    (
        "paligemma_with_expert.paligemma.model.language_model.norm.weight",
        (2048,),
    ),
)
STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT = 6
STRUCTURAL_ACTION_UNREACHABLE_NUMEL = 104_861_696
_TRAINING_GRAPH_METADATA_ATTR = "_franka_pi0_full_training_graph"
_REQUIRED_EFFECTIVE_STAT_FIELDS = ("mean", "std", "count")
_PREFERRED_VERIFICATION_KEYS = (
    "action_in_proj.bias",
    "state_proj.bias",
    "action_out_proj.bias",
    "action_in_proj.weight",
)
_TIED_PALIGEMMA_KEYS = (PALIGEMMA_LM_HEAD_KEY, PALIGEMMA_EMBED_TOKENS_KEY)


class PI0TrainingContractError(RuntimeError):
    """Base error for a violated Franka PI0 full-training contract."""


class PI0CheckpointLoadError(PI0TrainingContractError):
    """Raised when the local PI0 checkpoint cannot be loaded and verified strictly."""


class PI0EffectiveStatsError(PI0TrainingContractError):
    """Raised when model-visible 10D/7D effective statistics are invalid."""


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    raw_bytes = contiguous.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_local_checkpoint(pretrained_path: str | Path) -> tuple[Path, Path, Path]:
    checkpoint_dir = Path(pretrained_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"PI0 pretrained_path must be a local directory: {checkpoint_dir}")

    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.safetensors"
    missing = [str(path) for path in (config_path, weights_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Local PI0 checkpoint is incomplete; missing: {missing}")
    return checkpoint_dir, config_path.resolve(), weights_path.resolve()


def _assert_pi0_config_contract(config: PI0Config) -> None:
    image_keys = tuple(config.image_features)
    if image_keys != CANONICAL_PI0_IMAGE_KEYS:
        raise PI0TrainingContractError(
            "The local pi0_base config must retain exactly the three canonical image slots in order; "
            f"expected={CANONICAL_PI0_IMAGE_KEYS}, got={image_keys}"
        )

    state_feature = config.input_features.get(OBS_STATE) if config.input_features else None
    action_feature = config.output_features.get(ACTION) if config.output_features else None
    if state_feature != PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)):
        raise PI0TrainingContractError(f"PI0 state feature must be 10D, got {state_feature}")
    if action_feature != PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)):
        raise PI0TrainingContractError(f"PI0 action feature must be 7D, got {action_feature}")
    if (config.max_state_dim, config.max_action_dim) != (MAX_STATE_DIM, MAX_ACTION_DIM):
        raise PI0TrainingContractError(
            "PI0 internal padded dimensions must remain state/action=32/32, got "
            f"{config.max_state_dim}/{config.max_action_dim}"
        )

    expected_flags = {
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "use_relative_actions": False,
        "use_peft": False,
        "push_to_hub": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": getattr(config, key, None)}
        for key, expected in expected_flags.items()
        if getattr(config, key, None) != expected
    }
    if mismatches:
        raise PI0TrainingContractError(f"PI0 full-training config flags do not match: {mismatches}")

    expected_normalization = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    if config.normalization_mapping != expected_normalization:
        raise PI0TrainingContractError(
            "PI0 normalization mapping must be VISUAL=IDENTITY, STATE/ACTION=MEAN_STD; "
            f"got {config.normalization_mapping}"
        )


def build_pi0_full_finetune_config(
    pretrained_path: str | Path,
    *,
    device: str | torch.device,
) -> PI0Config:
    """Load the local base config and freeze the Franka 10D/7D full-training contract."""

    checkpoint_dir, _, _ = _resolve_local_checkpoint(pretrained_path)
    loaded = PreTrainedConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    if not isinstance(loaded, PI0Config):
        raise PI0TrainingContractError(
            f"Expected a PI0Config in {checkpoint_dir}, got {type(loaded).__name__}"
        )

    checkpoint_images = loaded.image_features
    checkpoint_image_keys = tuple(checkpoint_images)
    if checkpoint_image_keys != CANONICAL_PI0_IMAGE_KEYS:
        raise PI0TrainingContractError(
            "Refusing to rewrite an unknown camera contract from the local checkpoint; "
            f"expected={CANONICAL_PI0_IMAGE_KEYS}, got={checkpoint_image_keys}"
        )
    if any(feature.type is not FeatureType.VISUAL for feature in checkpoint_images.values()):
        raise PI0TrainingContractError("All three canonical checkpoint image slots must be VISUAL features")

    loaded.input_features = {
        **checkpoint_images,
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    loaded.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    loaded.max_state_dim = MAX_STATE_DIM
    loaded.max_action_dim = MAX_ACTION_DIM
    loaded.dtype = "bfloat16"
    loaded.gradient_checkpointing = True
    loaded.freeze_vision_encoder = False
    loaded.train_expert_only = False
    loaded.use_relative_actions = False
    loaded.use_peft = False
    loaded.push_to_hub = False
    loaded.repo_id = None
    loaded.pretrained_path = checkpoint_dir
    loaded.device = str(device)
    loaded.empty_cameras = 0
    loaded.compile_model = False
    loaded.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    loaded.validate_features()
    _assert_pi0_config_contract(loaded)
    return loaded


def _pi0_training_graph_modules(policy: torch.nn.Module) -> tuple[Any, Any, Any, Any]:
    """Return the four modules whose ownership defines the PI0 embedding/head graph."""

    try:
        core = policy.model
        combined = core.paligemma_with_expert
        paligemma = combined.paligemma
        expert = combined.gemma_expert
        embedding = paligemma.model.language_model.embed_tokens
    except AttributeError as exc:
        raise PI0TrainingContractError(
            "PI0 policy lacks the canonical PaliGemma/action-expert module graph"
        ) from exc
    return core, paligemma, expert, embedding


def _structural_action_unreachable_report(core: torch.nn.Module) -> dict[str, Any]:
    """Validate and serialize the exact retained-Parameter structural exception."""

    named_parameters = dict(core.named_parameters())
    entries: list[dict[str, Any]] = []
    total_numel = 0
    for core_name, expected_shape in STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS:
        parameter = named_parameters.get(core_name)
        if not isinstance(parameter, torch.nn.Parameter):
            raise PI0TrainingContractError(
                f"Structural action-unreachable PI0 Parameter is missing: {core_name}"
            )
        actual_shape = tuple(int(value) for value in parameter.shape)
        if actual_shape != expected_shape:
            raise PI0TrainingContractError(
                "Structural action-unreachable PI0 Parameter shape mismatch: "
                f"{core_name}: expected={expected_shape}, actual={actual_shape}"
            )
        if parameter.requires_grad is not True:
            raise PI0TrainingContractError(
                f"Structural action-unreachable Parameter must remain trainable, not frozen: {core_name}"
            )
        numel = parameter.numel()
        total_numel += numel
        entries.append(
            {
                "core_parameter_name": core_name,
                "policy_parameter_name": f"model.{core_name}",
                "shape": list(expected_shape),
                "numel": numel,
            }
        )
    if len(entries) != STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT:
        raise PI0TrainingContractError(
            "Structural action-unreachable tensor count changed: "
            f"expected={STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT}, actual={len(entries)}"
        )
    if total_numel != STRUCTURAL_ACTION_UNREACHABLE_NUMEL:
        raise PI0TrainingContractError(
            "Structural action-unreachable numel changed: "
            f"expected={STRUCTURAL_ACTION_UNREACHABLE_NUMEL}, actual={total_numel}"
        )
    return {
        "classification": "permanent_final_prefix_output_outside_action_loss",
        "parameters_retained_and_trainable": True,
        "ddp_requires_find_unused_parameters": True,
        "tensor_count": len(entries),
        "numel": total_numel,
        "parameters": entries,
    }


def assert_canonical_pi0_training_graph(policy: torch.nn.Module) -> dict[str, Any]:
    """Require one used PaliGemma embedding and no unused action-expert LM head.

    The local LeRobot PI0 construction replaces both Hugging Face backbone models
    after their causal-LM wrappers have already constructed output heads.  Without
    this project-local repair, the PaliGemma head is a duplicate of the language
    input embedding and the expert head is entirely outside PI0's diffusion graph.
    """

    core, paligemma, expert, embedding = _pi0_training_graph_modules(policy)
    lm_head = getattr(paligemma, "lm_head", None)
    if not isinstance(embedding, torch.nn.Embedding):
        raise PI0TrainingContractError("PaliGemma input embedding is missing or has an unknown type")
    if not isinstance(lm_head, torch.nn.Linear) or lm_head.bias is not None:
        raise PI0TrainingContractError("PaliGemma must expose one bias-free canonical LM head")
    if lm_head.weight is not embedding.weight:
        raise PI0TrainingContractError(
            "PaliGemma lm_head.weight is not pointer-tied to language_model.embed_tokens.weight"
        )
    if getattr(expert.model, "embed_tokens", object()) is not None:
        raise PI0TrainingContractError("PI0 action expert must not own a token embedding")
    if hasattr(expert, "lm_head") or any(
        name.startswith("paligemma_with_expert.gemma_expert.lm_head.")
        for name, _ in core.named_parameters()
    ):
        raise PI0TrainingContractError("Unused PI0 action-expert lm_head was not pruned")

    metadata = getattr(core, _TRAINING_GRAPH_METADATA_ATTR, None)
    if not isinstance(metadata, Mapping):
        raise PI0TrainingContractError("PI0 canonical training-graph metadata is missing")
    required_metadata = {
        "schema_version": 2,
        "paligemma_lm_head_key": PALIGEMMA_LM_HEAD_KEY,
        "paligemma_embed_tokens_key": PALIGEMMA_EMBED_TOKENS_KEY,
        "unused_expert_lm_head_key": UNUSED_EXPERT_LM_HEAD_KEY,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in required_metadata.items()
        if metadata.get(key) != expected
    }
    tied_shape = tuple(int(value) for value in metadata.get("paligemma_tied_weight_shape", ()))
    expert_shape = tuple(int(value) for value in metadata.get("unused_expert_lm_head_shape", ()))
    structural_report = _structural_action_unreachable_report(core)
    if tied_shape != tuple(embedding.weight.shape):
        mismatches["paligemma_tied_weight_shape"] = {
            "expected": list(embedding.weight.shape),
            "actual": list(tied_shape),
        }
    if len(expert_shape) != 2 or any(value <= 0 for value in expert_shape):
        mismatches["unused_expert_lm_head_shape"] = {
            "expected": "two positive dimensions",
            "actual": list(expert_shape),
        }
    if mismatches:
        raise PI0TrainingContractError(f"PI0 canonical training-graph metadata mismatch: {mismatches}")

    tied_numel = embedding.weight.numel()
    expert_numel = math.prod(expert_shape)
    if int(metadata.get("deduplicated_tied_parameter_count", -1)) != tied_numel:
        raise PI0TrainingContractError("PaliGemma tied-parameter count metadata is inconsistent")
    if int(metadata.get("pruned_unused_parameter_count", -1)) != expert_numel:
        raise PI0TrainingContractError("Action-expert pruned-parameter count metadata is inconsistent")
    if metadata.get("structural_action_unreachable") != structural_report:
        raise PI0TrainingContractError(
            "PI0 structural action-unreachable metadata does not match the live Parameter graph"
        )
    return {
        "schema_version": 2,
        "paligemma_lm_head_tied": True,
        "paligemma_tied_weight_shape": list(tied_shape),
        "paligemma_lm_head_key": PALIGEMMA_LM_HEAD_KEY,
        "paligemma_embed_tokens_key": PALIGEMMA_EMBED_TOKENS_KEY,
        "expert_lm_head_pruned": True,
        "unused_expert_lm_head_key": UNUSED_EXPERT_LM_HEAD_KEY,
        "unused_expert_lm_head_shape": list(expert_shape),
        "deduplicated_tied_parameter_count": tied_numel,
        "pruned_unused_parameter_count": expert_numel,
        "parameters_removed_from_optimizer": tied_numel + expert_numel,
        "structural_action_unreachable": structural_report,
    }


def canonicalize_pi0_full_training_graph(policy: torch.nn.Module) -> dict[str, Any]:
    """Tie/prune the two causal-LM heads before any checkpoint or optimizer load."""

    core, paligemma, expert, embedding = _pi0_training_graph_modules(policy)
    existing_metadata = getattr(core, _TRAINING_GRAPH_METADATA_ATTR, None)
    if existing_metadata is not None:
        return assert_canonical_pi0_training_graph(policy)

    lm_head = getattr(paligemma, "lm_head", None)
    expert_lm_head = getattr(expert, "lm_head", None)
    if not isinstance(embedding, torch.nn.Embedding):
        raise PI0TrainingContractError("Cannot tie a missing/non-Embedding PaliGemma input embedding")
    if not isinstance(lm_head, torch.nn.Linear) or lm_head.bias is not None:
        raise PI0TrainingContractError("Cannot tie a missing/non-canonical PaliGemma LM head")
    if tuple(lm_head.weight.shape) != tuple(embedding.weight.shape):
        raise PI0TrainingContractError(
            "Cannot tie PaliGemma LM head/input embedding with different shapes: "
            f"{tuple(lm_head.weight.shape)} != {tuple(embedding.weight.shape)}"
        )
    if not isinstance(expert_lm_head, torch.nn.Linear) or expert_lm_head.bias is not None:
        raise PI0TrainingContractError("Cannot prune a missing/non-canonical action-expert LM head")
    if getattr(expert.model, "embed_tokens", object()) is not None:
        raise PI0TrainingContractError("Action expert unexpectedly owns a token embedding")

    metadata = {
        "schema_version": 2,
        "paligemma_lm_head_key": PALIGEMMA_LM_HEAD_KEY,
        "paligemma_embed_tokens_key": PALIGEMMA_EMBED_TOKENS_KEY,
        "paligemma_tied_weight_shape": tuple(int(value) for value in embedding.weight.shape),
        "unused_expert_lm_head_key": UNUSED_EXPERT_LM_HEAD_KEY,
        "unused_expert_lm_head_shape": tuple(int(value) for value in expert_lm_head.weight.shape),
        "deduplicated_tied_parameter_count": embedding.weight.numel(),
        "pruned_unused_parameter_count": expert_lm_head.weight.numel(),
        "structural_action_unreachable": _structural_action_unreachable_report(core),
    }

    # Assignment replaces the wrapper's orphan Parameter with the exact input
    # embedding Parameter.  Deleting the expert head removes its Parameters from
    # named_parameters/state_dict entirely; PI0 calls expert.model directly.
    paligemma.lm_head.weight = embedding.weight
    del expert.lm_head
    setattr(core, _TRAINING_GRAPH_METADATA_ATTR, metadata)
    return assert_canonical_pi0_training_graph(policy)


def _load_stats_source(dataset_stats: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], str]:
    if isinstance(dataset_stats, (str, Path)):
        stats_path = Path(dataset_stats).expanduser().resolve()
        if not stats_path.is_file():
            raise FileNotFoundError(f"Effective PI0 stats file does not exist: {stats_path}")
        with stats_path.open(encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, dict):
            raise PI0EffectiveStatsError(f"Effective stats root must be an object: {stats_path}")
        return loaded, str(stats_path)
    if not isinstance(dataset_stats, Mapping):
        raise TypeError("dataset_stats must be a mapping or a local JSON path")
    return dict(dataset_stats), "in_memory"


def prepare_effective_pi0_stats(
    dataset_stats: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Validate and isolate only the model-visible 10D state and 7D action stats."""

    raw_stats, source = _load_stats_source(dataset_stats)
    dimensions = {OBS_STATE: STATE_DIM, ACTION: ACTION_DIM}
    prepared: dict[str, dict[str, torch.Tensor]] = {}

    for feature_key, dimension in dimensions.items():
        feature_stats = raw_stats.get(feature_key)
        if not isinstance(feature_stats, Mapping):
            raise PI0EffectiveStatsError(f"Missing effective stats object for {feature_key!r}")
        missing_fields = [field for field in _REQUIRED_EFFECTIVE_STAT_FIELDS if field not in feature_stats]
        if missing_fields:
            raise PI0EffectiveStatsError(f"{feature_key!r} is missing stats fields {missing_fields}")

        tensor_stats: dict[str, torch.Tensor] = {}
        for stat_name, value in feature_stats.items():
            try:
                tensor = torch.as_tensor(value).detach().cpu()
            except (TypeError, ValueError) as exc:
                raise PI0EffectiveStatsError(
                    f"{feature_key}.{stat_name} cannot be converted to a tensor"
                ) from exc
            if tensor.numel() == 0:
                raise PI0EffectiveStatsError(f"{feature_key}.{stat_name} is empty")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise PI0EffectiveStatsError(f"{feature_key}.{stat_name} contains NaN/Inf")
            tensor_stats[str(stat_name)] = tensor

        for stat_name in ("mean", "std"):
            if tuple(tensor_stats[stat_name].shape) != (dimension,):
                raise PI0EffectiveStatsError(
                    f"{feature_key}.{stat_name} must have shape ({dimension},), "
                    f"got {tuple(tensor_stats[stat_name].shape)}"
                )
        if bool(torch.any(tensor_stats["std"] < 0)):
            raise PI0EffectiveStatsError(f"{feature_key}.std cannot contain negative values")
        if tensor_stats["count"].numel() != 1 or int(tensor_stats["count"].reshape(-1)[0]) <= 0:
            raise PI0EffectiveStatsError(f"{feature_key}.count must be one positive scalar")
        prepared[feature_key] = tensor_stats

    report = {
        "source": source,
        "effective_stats_sha256": _canonical_json_sha256(prepared),
        "injected_feature_keys": list(prepared),
        "ignored_top_level_keys": sorted(set(raw_stats) - set(EFFECTIVE_STATS_KEYS)),
        "dimensions": {OBS_STATE: STATE_DIM, ACTION: ACTION_DIM},
        "counts": {
            key: int(stats["count"].reshape(-1)[0])
            for key, stats in prepared.items()
        },
    }
    return prepared, report


def _select_verification_key(model: torch.nn.Module, weights_path: Path) -> str:
    model_keys = set(model.state_dict())
    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
        for key in _PREFERRED_VERIFICATION_KEYS:
            if key in checkpoint_keys and key in model_keys:
                return key

        common = sorted(checkpoint_keys & model_keys)
        if not common:
            raise PI0CheckpointLoadError(
                "No common tensor key exists between model and local model.safetensors"
            )
        return min(common, key=lambda key: math.prod(checkpoint.get_slice(key).get_shape()))


def _inspect_checkpoint_training_graph(
    policy: PI0Policy,
    weights_path: Path,
) -> dict[str, Any]:
    """Validate the only tied alias and unused base-checkpoint tensor we permit."""

    graph = assert_canonical_pi0_training_graph(policy)
    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
        present_tied_keys = [key for key in _TIED_PALIGEMMA_KEYS if key in checkpoint_keys]
        if len(present_tied_keys) != 1:
            raise PI0CheckpointLoadError(
                "PI0 checkpoint must store exactly one physical tensor for the pointer-tied PaliGemma "
                f"embedding/lm_head aliases; present={present_tied_keys}; random weights will not be returned"
            )
        physical_tied_key = present_tied_keys[0]
        tied_shape = tuple(checkpoint.get_slice(physical_tied_key).get_shape())
        expected_tied_shape = tuple(graph["paligemma_tied_weight_shape"])
        if tied_shape != expected_tied_shape:
            raise PI0CheckpointLoadError(
                "Canonical PaliGemma tied checkpoint tensor has the wrong shape: "
                f"key={physical_tied_key}, expected={expected_tied_shape}, got={tied_shape}; "
                "random weights will not be returned"
            )

        dropped_checkpoint_keys: dict[str, Any] = {}
        if UNUSED_EXPERT_LM_HEAD_KEY in checkpoint_keys:
            expert_slice = checkpoint.get_slice(UNUSED_EXPERT_LM_HEAD_KEY)
            expert_shape = tuple(expert_slice.get_shape())
            expected_expert_shape = tuple(graph["unused_expert_lm_head_shape"])
            if expert_shape != expected_expert_shape:
                raise PI0CheckpointLoadError(
                    "Canonical unused action-expert checkpoint head has the wrong shape and cannot be "
                    f"safely dropped: expected={expected_expert_shape}, got={expert_shape}; "
                    "random weights will not be returned"
                )
            dropped_checkpoint_keys[UNUSED_EXPERT_LM_HEAD_KEY] = {
                "reason": "unused causal-LM output head; PI0 diffusion calls gemma_expert.model directly",
                "shape": list(expert_shape),
                "dtype": str(expert_slice.get_dtype()),
                "verified_before_drop": True,
            }

    logical_alias_key = next(key for key in _TIED_PALIGEMMA_KEYS if key != physical_tied_key)
    return {
        "physical_paligemma_tied_key": physical_tied_key,
        "logical_paligemma_alias_key": logical_alias_key,
        "paligemma_tied_shape": list(tied_shape),
        "resolved_checkpoint_aliases": {logical_alias_key: physical_tied_key},
        "dropped_checkpoint_keys": dropped_checkpoint_keys,
    }


def _strict_load_and_verify_weights(
    policy: PI0Policy,
    weights_path: Path,
) -> dict[str, Any]:
    """Strictly load the checkpoint into ``policy.model`` and numerically verify one tensor."""

    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = tuple(checkpoint.keys())
    prefixed_keys = [key for key in checkpoint_keys if key.startswith("model.")]
    if prefixed_keys:
        raise PI0CheckpointLoadError(
            "PI0 checkpoint uses policy-wrapper `model.` keys instead of the required unprefixed core-model "
            f"namespace; examples={prefixed_keys[:5]}; random weights will not be returned"
        )

    checkpoint_graph = _inspect_checkpoint_training_graph(policy, weights_path)
    verification_key = _select_verification_key(policy.model, weights_path)
    try:
        missing_keys, unexpected_keys = load_safetensors_model(
            policy.model,
            weights_path,
            strict=False,
            device=str(policy.config.device),
        )
    except Exception as exc:
        raise PI0CheckpointLoadError(
            f"Strict PI0 safetensors load failed for {weights_path}; random weights will not be returned"
        ) from exc

    unresolved_missing = set(missing_keys)
    # safetensors.load_model understands shared storage, but tolerate versions
    # that still report the non-physical logical alias as missing.  No copy is
    # needed (or allowed): both module paths already point to one Parameter.
    unresolved_missing.discard(checkpoint_graph["logical_paligemma_alias_key"])
    unresolved_unexpected = set(unexpected_keys)
    unresolved_unexpected.difference_update(checkpoint_graph["dropped_checkpoint_keys"])

    if unresolved_missing or unresolved_unexpected:
        raise PI0CheckpointLoadError(
            "Strict PI0 load found incompatible keys after applying only the canonical explicit aliases: "
            f"missing={sorted(unresolved_missing)}, unexpected={sorted(unresolved_unexpected)}; "
            "random weights will not be returned"
        )

    # Loading must not break the Parameter alias or recreate the pruned module.
    assert_canonical_pi0_training_graph(policy)

    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_tensor = checkpoint.get_tensor(verification_key)
    loaded_tensor = policy.model.state_dict()[verification_key].detach().cpu()
    expected_tensor = checkpoint_tensor.to(dtype=loaded_tensor.dtype)
    if loaded_tensor.shape != expected_tensor.shape or not torch.equal(loaded_tensor, expected_tensor):
        max_abs_error = None
        if loaded_tensor.shape == expected_tensor.shape and loaded_tensor.is_floating_point():
            max_abs_error = float((loaded_tensor.float() - expected_tensor.float()).abs().max())
        raise PI0CheckpointLoadError(
            "PI0 safetensors numerical verification failed after strict load: "
            f"key={verification_key}, max_abs_error={max_abs_error}"
        )

    return {
        "strict": True,
        "weights_namespace": PI0_CORE_WEIGHTS_NAMESPACE,
        "missing_keys": [],
        "unexpected_keys": [],
        "resolved_checkpoint_aliases": checkpoint_graph["resolved_checkpoint_aliases"],
        "dropped_checkpoint_keys": checkpoint_graph["dropped_checkpoint_keys"],
        "checkpoint_training_graph": {
            key: value
            for key, value in checkpoint_graph.items()
            if key not in {"resolved_checkpoint_aliases", "dropped_checkpoint_keys"}
        },
        "verified_tensor": {
            "checkpoint_key": verification_key,
            "model_key": f"model.{verification_key}",
            "shape": list(loaded_tensor.shape),
            "checkpoint_dtype": str(checkpoint_tensor.dtype),
            "model_dtype": str(loaded_tensor.dtype),
            "checkpoint_tensor_sha256": _tensor_sha256(checkpoint_tensor),
            "exact_after_model_dtype_cast": True,
        },
    }


def load_pi0_full_checkpoint_weights(
    policy: PI0Policy,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Strictly restore project/base weights into an already constructed full PI0 policy.

    This is the stable resume boundary paired with :func:`save_pi0_full_checkpoint`.
    Both canonical base checkpoints and project checkpoints use unprefixed
    ``PI0Pytorch`` keys.  Only the explicitly documented base-checkpoint embedding
    alias is accepted; every other missing/unexpected key is fatal.
    """

    _assert_pi0_config_contract(policy.config)
    training_graph_report = assert_canonical_pi0_training_graph(policy)
    resolved_dir, config_path, weights_path = _resolve_local_checkpoint(checkpoint_dir)

    checkpoint_manifest_path = resolved_dir / "franka_pi0_checkpoint_manifest.json"
    checkpoint_manifest: dict[str, Any] | None = None
    if checkpoint_manifest_path.is_file():
        with checkpoint_manifest_path.open(encoding="utf-8") as stream:
            checkpoint_manifest = json.load(stream)
        manifest_namespace = checkpoint_manifest.get("weights_namespace")
        if manifest_namespace != PI0_CORE_WEIGHTS_NAMESPACE:
            raise PI0CheckpointLoadError(
                "Project checkpoint manifest has an incompatible weights namespace: "
                f"expected={PI0_CORE_WEIGHTS_NAMESPACE!r}, got={manifest_namespace!r}"
            )
        manifest_training_graph = checkpoint_manifest.get("training_graph")
        if manifest_training_graph != training_graph_report:
            raise PI0CheckpointLoadError(
                "Project checkpoint manifest training_graph does not match the live canonical PI0 graph: "
                f"expected={training_graph_report}, got={manifest_training_graph}"
            )

    weights_report = _strict_load_and_verify_weights(policy, weights_path)
    parameter_report = assert_full_parameter_training(policy)
    return {
        "directory": str(resolved_dir),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "weights_path": str(weights_path),
        "weights_size_bytes": weights_path.stat().st_size,
        "project_manifest_path": str(checkpoint_manifest_path) if checkpoint_manifest is not None else None,
        "project_manifest_present": checkpoint_manifest is not None,
        **weights_report,
        "training_graph": training_graph_report,
        "parameter_training": parameter_report,
    }


def _looks_like_peft_or_lora_module(module: torch.nn.Module) -> bool:
    module_path = module.__class__.__module__.lower()
    class_name = module.__class__.__name__.lower()
    return module_path == "peft" or module_path.startswith("peft.") or "loralayer" in class_name


def assert_full_parameter_training(policy: torch.nn.Module) -> dict[str, Any]:
    """Fail unless ``policy`` is an unfrozen, non-PEFT/LoRA PI0 training model."""

    config = getattr(policy, "config", None)
    if not isinstance(config, PI0Config):
        raise PI0TrainingContractError(f"Expected policy.config to be PI0Config, got {type(config).__name__}")
    _assert_pi0_config_contract(config)

    if any(_looks_like_peft_or_lora_module(module) for module in policy.modules()):
        raise PI0TrainingContractError("A PEFT/LoRA module is present in the PI0 policy")
    suspicious_names = [
        name
        for name, _ in policy.named_parameters()
        if any(marker in name.lower() for marker in ("lora_", ".lora.", "peft", "adapter_"))
    ]
    if suspicious_names:
        raise PI0TrainingContractError(f"PEFT/LoRA parameter names are present: {suspicious_names[:10]}")

    parameters = list(policy.named_parameters())
    if not parameters:
        raise PI0TrainingContractError("PI0 policy contains no parameters")
    frozen = [name for name, parameter in parameters if not parameter.requires_grad]
    if frozen:
        raise PI0TrainingContractError(
            f"Full-parameter training requires every tensor trainable; frozen={frozen[:20]}"
        )

    core_model = getattr(policy, "model", None)
    if core_model is None or getattr(core_model, "gradient_checkpointing_enabled", None) is not True:
        raise PI0TrainingContractError("PI0 core gradient checkpointing is not enabled")
    training_graph = assert_canonical_pi0_training_graph(policy)

    total_numel = sum(parameter.numel() for _, parameter in parameters)
    return {
        "use_peft": False,
        "lora_parameter_count": 0,
        "parameter_tensor_count": len(parameters),
        "total_parameters": total_numel,
        "trainable_parameters": total_numel,
        "trainable_fraction": 1.0,
        "gradient_checkpointing_enabled": True,
        "policy_training_mode": bool(policy.training),
        "paligemma_lm_head_tied": True,
        "expert_lm_head_pruned": True,
        "training_graph": training_graph,
    }


def _gradient_group(parameter_name: str) -> str:
    name = parameter_name.lower()
    if "vision_tower" in name:
        return "vision_encoder"
    if "gemma_expert" in name:
        return "action_expert"
    if any(
        marker in name
        for marker in (
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp",
        )
    ):
        return "state_action_projections"
    if "paligemma" in name or "multi_modal_projector" in name:
        return "vlm"
    return "other"


def gradient_coverage_summary(
    policy: torch.nn.Module,
    *,
    inspect_values: bool = True,
) -> dict[str, Any]:
    """Return JSON-ready gradient coverage after backward, grouped by PI0 subsystem."""

    groups: dict[str, dict[str, Any]] = {}
    for name, parameter in policy.named_parameters():
        group_name = _gradient_group(name)
        group = groups.setdefault(
            group_name,
            {
                "parameter_tensors": 0,
                "parameter_numel": 0,
                "trainable_tensors": 0,
                "trainable_numel": 0,
                "gradient_tensors": 0,
                "gradient_numel": 0,
                "all_gradients_finite": True,
                "nonzero_gradient_tensors": 0,
                "missing_gradient_names": [],
            },
        )
        group["parameter_tensors"] += 1
        group["parameter_numel"] += parameter.numel()
        if not parameter.requires_grad:
            continue
        group["trainable_tensors"] += 1
        group["trainable_numel"] += parameter.numel()
        if parameter.grad is None:
            group["missing_gradient_names"].append(name)
            continue

        gradient = parameter.grad.detach()
        group["gradient_tensors"] += 1
        group["gradient_numel"] += gradient.numel()
        if inspect_values:
            group["all_gradients_finite"] &= bool(torch.isfinite(gradient).all())
            group["nonzero_gradient_tensors"] += int(bool(torch.count_nonzero(gradient)))

    totals = {
        field: sum(group[field] for group in groups.values())
        for field in (
            "parameter_tensors",
            "parameter_numel",
            "trainable_tensors",
            "trainable_numel",
            "gradient_tensors",
            "gradient_numel",
            "nonzero_gradient_tensors",
        )
    }
    for group in groups.values():
        trainable_tensors = group["trainable_tensors"]
        trainable_numel = group["trainable_numel"]
        group["gradient_tensor_coverage"] = (
            group["gradient_tensors"] / trainable_tensors if trainable_tensors else 1.0
        )
        group["gradient_numel_coverage"] = (
            group["gradient_numel"] / trainable_numel if trainable_numel else 1.0
        )
        group["missing_gradient_names"] = group["missing_gradient_names"][:20]

    totals["gradient_tensor_coverage"] = (
        totals["gradient_tensors"] / totals["trainable_tensors"]
        if totals["trainable_tensors"]
        else 1.0
    )
    totals["gradient_numel_coverage"] = (
        totals["gradient_numel"] / totals["trainable_numel"]
        if totals["trainable_numel"]
        else 1.0
    )
    totals["all_gradients_finite"] = all(
        group["all_gradients_finite"] for group in groups.values()
    )
    return {"inspect_values": inspect_values, "overall": totals, "groups": groups}


def load_pi0_full_policy_and_processors(
    pretrained_path: str | Path,
    dataset_stats: Mapping[str, Any] | str | Path,
    device: str | torch.device,
) -> tuple[PI0Policy, Any, Any, dict[str, Any]]:
    """Strictly load local PI0 base weights and official 10D/7D processors.

    There is intentionally no fallback to random initialization.  Any config,
    safetensors, numerical-verification, stats, tokenizer, or processor failure
    propagates as an exception.
    """

    checkpoint_dir, _, _ = _resolve_local_checkpoint(pretrained_path)
    config = build_pi0_full_finetune_config(checkpoint_dir, device=device)
    effective_stats, stats_report = prepare_effective_pi0_stats(dataset_stats)

    try:
        policy = PI0Policy(config)
    except Exception as exc:
        raise PI0CheckpointLoadError("PI0 model construction failed before strict weight loading") from exc

    # Repair the wrapper-owned heads before loading any tensor or constructing
    # an optimizer.  The canonical graph metadata separately records the six
    # retained final-prefix Parameters that are structurally outside the action
    # loss and therefore require DDP unused-parameter discovery.
    canonicalize_pi0_full_training_graph(policy)
    weights_report = load_pi0_full_checkpoint_weights(policy, checkpoint_dir)
    policy.train()
    parameter_report = assert_full_parameter_training(policy)

    try:
        preprocessor, postprocessor = make_pi0_pre_post_processors(
            config=config,
            dataset_stats=effective_stats,
        )
    except Exception as exc:
        raise PI0TrainingContractError(
            "Official PI0 pre/post processor construction failed with effective 10D/7D stats"
        ) from exc

    load_report = {
        "schema_version": 1,
        "loader": "franka_eef_pipeline.pi0_training.load_pi0_full_policy_and_processors",
        "pretrained": {
            **weights_report,
        },
        "config": {
            "canonical_image_slots": list(CANONICAL_PI0_IMAGE_KEYS),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "max_state_dim": MAX_STATE_DIM,
            "max_action_dim": MAX_ACTION_DIM,
            "dtype": config.dtype,
            "device": str(config.device),
            "gradient_checkpointing": config.gradient_checkpointing,
            "freeze_vision_encoder": config.freeze_vision_encoder,
            "train_expert_only": config.train_expert_only,
            "use_relative_actions": config.use_relative_actions,
            "use_peft": config.use_peft,
            "push_to_hub": config.push_to_hub,
        },
        "effective_stats": stats_report,
        "parameters": parameter_report,
        "processors": {
            "factory": "lerobot.policies.pi0.processor_pi0.make_pi0_pre_post_processors",
            "normalization": {
                "VISUAL": "IDENTITY",
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD",
            },
            "injected_feature_keys": list(EFFECTIVE_STATS_KEYS),
        },
    }
    return policy, preprocessor, postprocessor, load_report


def _write_or_copy_manifest(
    value: Mapping[str, Any] | str | Path,
    destination_dir: Path,
    stem: str,
) -> Path:
    if isinstance(value, (str, Path)):
        source = Path(value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Manifest source does not exist: {source}")
        suffix = source.suffix or ".json"
        destination = destination_dir / f"{stem}{suffix}"
        shutil.copy2(source, destination)
        return destination

    if not isinstance(value, Mapping):
        raise TypeError(f"{stem} must be a mapping or local file path")
    destination = destination_dir / f"{stem}.json"
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return destination


def save_pi0_full_checkpoint(
    policy: PI0Policy,
    preprocessor: Any,
    postprocessor: Any,
    save_directory: str | Path,
    *,
    geometry_manifest: Mapping[str, Any] | str | Path,
    stats_manifest: Mapping[str, Any] | str | Path,
    load_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a complete local checkpoint without any Hub or W&B upload side effect."""

    parameter_report = assert_full_parameter_training(policy)
    destination = Path(save_directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint directory: {destination}")
    destination.mkdir(parents=True)

    # Save the core model rather than the policy wrapper.  The canonical pi0_base
    # safetensors and this module's strict loader both use unprefixed PI0Pytorch
    # keys (``action_in_proj.*``, not ``model.action_in_proj.*``).  Calling
    # PreTrainedPolicy.save_pretrained(policy) would add a uniform ``model.``
    # prefix and make the project checkpoint incompatible with its own loader.
    policy.config.save_pretrained(destination, push_to_hub=False)
    save_safetensors_model(policy.model, destination / "model.safetensors")
    preprocessor.save_pretrained(
        destination,
        push_to_hub=False,
        config_filename="policy_preprocessor.json",
    )
    postprocessor.save_pretrained(
        destination,
        push_to_hub=False,
        config_filename="policy_postprocessor.json",
    )
    geometry_path = _write_or_copy_manifest(
        geometry_manifest,
        destination,
        "franka_eef_geometry_manifest",
    )
    stats_path = _write_or_copy_manifest(stats_manifest, destination, "pi0_eef_stats")

    required_paths = [
        destination / "config.json",
        destination / "model.safetensors",
        destination / "policy_preprocessor.json",
        destination / "policy_postprocessor.json",
        geometry_path,
        stats_path,
    ]
    missing = [path.name for path in required_paths if not path.is_file()]
    if missing:
        raise PI0TrainingContractError(f"Local checkpoint save is incomplete; missing={missing}")

    manifest = {
        "schema_version": 1,
        "checkpoint_type": "franka_pi0_full_parameter_eef",
        "weights_namespace": PI0_CORE_WEIGHTS_NAMESPACE,
        "hub_upload": False,
        "wandb_artifact_upload": False,
        "parameter_training": parameter_report,
        "training_graph": parameter_report["training_graph"],
        "source_load_report": _jsonable(load_report) if load_report is not None else None,
        "files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in required_paths
        },
    }
    checkpoint_manifest = destination / "franka_pi0_checkpoint_manifest.json"
    with checkpoint_manifest.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return manifest


__all__ = [
    "ACTION_DIM",
    "CANONICAL_PI0_IMAGE_KEYS",
    "MAX_ACTION_DIM",
    "MAX_STATE_DIM",
    "PALIGEMMA_EMBED_TOKENS_KEY",
    "PALIGEMMA_LM_HEAD_KEY",
    "PI0_CORE_WEIGHTS_NAMESPACE",
    "PI0CheckpointLoadError",
    "PI0EffectiveStatsError",
    "PI0TrainingContractError",
    "STATE_DIM",
    "STRUCTURAL_ACTION_UNREACHABLE_NUMEL",
    "STRUCTURAL_ACTION_UNREACHABLE_PARAMETER_SPECS",
    "STRUCTURAL_ACTION_UNREACHABLE_TENSOR_COUNT",
    "UNUSED_EXPERT_LM_HEAD_KEY",
    "assert_canonical_pi0_training_graph",
    "assert_full_parameter_training",
    "build_pi0_full_finetune_config",
    "canonicalize_pi0_full_training_graph",
    "gradient_coverage_summary",
    "load_pi0_full_checkpoint_weights",
    "load_pi0_full_policy_and_processors",
    "prepare_effective_pi0_stats",
    "save_pi0_full_checkpoint",
]
