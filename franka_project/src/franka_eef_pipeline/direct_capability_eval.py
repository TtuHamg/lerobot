"""Pure helpers for the M0 PI0 reconstruction metric family.

The evaluator has one deliberately narrow job: hide action supervision from
the policy, generate order-independent PI0 noise, and compute exactly the M0
teacher-forced in-sample metrics defined in F0 report section 6.2. It does not
construct reference baselines or calculate alternative metric families.  The
metric math is independent of the dataset task and sampling frequency as long
as the Cartesian action contract remains ``[50, 7]``.

Action convention: ``[delta_xyz_m, body_rotvec_rad, gripper_0_1]``. Rotation
error is the SO(3) geodesic distance between exponential-map rotations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .geometry import rotation_geodesic_angle, so3_exp


ACTION_DIM = 7
ACTION_CHUNK_SIZE = 50
HORIZONS = (1, 5, 10, 25, 50)
_ERROR_KEYS = ("translation_mm", "rotation_geodesic_deg", "gripper_abs")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _numpy(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    _require(result.dtype != object, f"{name} must be a numeric array")
    return result


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _validate_action_array(value: Any, *, name: str) -> np.ndarray:
    result = _numpy(value, name=name)
    _require(
        result.ndim == 3 and result.shape[1:] == (ACTION_CHUNK_SIZE, ACTION_DIM),
        f"{name} must have shape [batch, {ACTION_CHUNK_SIZE}, {ACTION_DIM}]",
    )
    _require(result.shape[0] > 0, f"{name} cannot be empty")
    _require(np.issubdtype(result.dtype, np.number), f"{name} must be numeric")
    _require(np.all(np.isfinite(result)), f"{name} contains NaN/Inf")
    return result


def _validate_pad(pad: Any | None, shape: tuple[int, int]) -> np.ndarray:
    if pad is None:
        return np.zeros(shape, dtype=bool)
    result = _numpy(pad, name="action_is_pad")
    _require(result.shape == shape, f"action_is_pad must have shape {shape}")
    if result.dtype != np.bool_:
        _require(np.all((result == 0) | (result == 1)), "action_is_pad must contain only 0/1")
    return result.astype(bool, copy=False)


def split_supervision(batch: Mapping[str, Any]) -> tuple[dict[str, Any], Any, Any]:
    """Copy supervision out of a batch and return target-free model inputs."""

    _require(isinstance(batch, Mapping), "batch must be a mapping")
    _require("action" in batch, "batch is missing action supervision")
    _require("action_is_pad" in batch, "batch is missing action_is_pad supervision")
    target = batch["action"]
    target_array = _validate_action_array(target, name="action")
    pad = batch["action_is_pad"]
    _validate_pad(pad, target_array.shape[:2])
    inputs = {key: value for key, value in batch.items() if key not in {"action", "action_is_pad"}}
    _require("action" not in inputs and "action_is_pad" not in inputs, "supervision leaked into inputs")
    return inputs, _clone(target), _clone(pad)


def fixed_noise_batch(
    seed: int,
    logical_ids: Sequence[Any] | np.ndarray | torch.Tensor,
    shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create Gaussian noise stable per logical anchor and independent of batching."""

    _require(type(seed) is int, "seed must be an integer")
    ids_array = _numpy(logical_ids, name="logical_ids").reshape(-1)
    _require(ids_array.size > 0, "logical_ids cannot be empty")
    dimensions = tuple(int(item) for item in shape)
    _require(len(dimensions) > 0 and all(item > 0 for item in dimensions), "shape must be positive")
    if len(dimensions) == 2:
        dimensions = (len(ids_array), *dimensions)
    else:
        _require(
            dimensions[0] == len(ids_array),
            "explicit noise batch size does not match logical_ids",
        )
    _require(dtype.is_floating_point, "noise dtype must be floating point")

    generation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    samples: list[torch.Tensor] = []
    for logical_id in ids_array:
        scalar = logical_id.item() if isinstance(logical_id, np.generic) else logical_id
        _require(
            isinstance(scalar, (int, np.integer)) and not isinstance(scalar, (bool, np.bool_)),
            "logical_ids must be integers",
        )
        item_seed = seed + int(scalar)
        _require(0 <= item_seed < 2**63 - 1, "seed + logical_id is outside torch seed range")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(item_seed)
        samples.append(
            torch.randn(dimensions[1:], generator=generator, dtype=generation_dtype, device="cpu")
        )
    return torch.stack(samples).to(dtype=dtype, device=torch.device(device))


def compute_action_errors(
    prediction: Any,
    target: Any,
    pad: Any | None = None,
) -> dict[str, np.ndarray]:
    """Compute the three per-waypoint error arrays used by the M0 protocol."""

    prediction_array = _validate_action_array(prediction, name="prediction").astype(np.float64)
    target_array = _validate_action_array(target, name="target").astype(np.float64)
    _require(prediction_array.shape == target_array.shape, "prediction and target shapes differ")
    pad_array = _validate_pad(pad, prediction_array.shape[:2])
    _require(
        not np.any(pad_array),
        "M0 evaluation requires valid, unpadded 50-waypoint action chunks",
    )

    translation_mm = (
        np.linalg.norm(prediction_array[..., :3] - target_array[..., :3], axis=-1) * 1000.0
    )
    prediction_rotation = so3_exp(prediction_array[..., 3:6])
    target_rotation = so3_exp(target_array[..., 3:6])
    rotation_geodesic_deg = np.rad2deg(
        rotation_geodesic_angle(prediction_rotation, target_rotation)
    )
    gripper_abs = np.abs(prediction_array[..., 6] - target_array[..., 6])
    result = {
        "translation_mm": translation_mm,
        "rotation_geodesic_deg": rotation_geodesic_deg,
        "gripper_abs": gripper_abs,
    }
    for name, values in result.items():
        _require(np.all(np.isfinite(values)), f"{name} contains NaN/Inf")
        _require(np.all(values >= 0.0), f"{name} contains negative values")
    return result


def _validated_errors(errors: Mapping[str, Any]) -> dict[str, np.ndarray]:
    _require(isinstance(errors, Mapping), "errors must be a mapping")
    _require(set(errors) == set(_ERROR_KEYS), f"errors must contain exactly {_ERROR_KEYS}")
    result = {
        key: _numpy(errors[key], name=key).astype(np.float64, copy=False) for key in _ERROR_KEYS
    }
    shapes = {value.shape for value in result.values()}
    _require(
        len(shapes) == 1 and next(iter(shapes))[1:] == (ACTION_CHUNK_SIZE,),
        f"error arrays must have shape [anchors, {ACTION_CHUNK_SIZE}]",
    )
    for key, value in result.items():
        _require(value.shape[0] > 0, f"{key} cannot be empty")
        _require(np.all(np.isfinite(value)) and np.all(value >= 0.0), f"invalid {key}")
    return result


def _episode_groups(episode_ids: Any, count: int) -> list[np.ndarray]:
    values = _numpy(episode_ids, name="episode_ids").reshape(-1)
    _require(len(values) == count, "episode_ids length does not match anchor count")
    order: list[Any] = []
    positions: dict[Any, list[int]] = {}
    for index, raw in enumerate(values):
        item = raw.item() if isinstance(raw, np.generic) else raw
        try:
            hash(item)
        except TypeError as exc:
            raise ValueError("episode_ids must contain hashable scalar values") from exc
        if item not in positions:
            order.append(item)
            positions[item] = []
        positions[item].append(index)
    _require(order, "episode_ids cannot be empty")
    return [np.asarray(positions[item], dtype=np.int64) for item in order]


def _episode_macro_error_tree(
    errors: Mapping[str, Any],
    episode_ids: Any,
) -> dict[str, Any]:
    values = _validated_errors(errors)
    anchor_count = next(iter(values.values())).shape[0]
    groups = _episode_groups(episode_ids, anchor_count)

    def macro(key: str, waypoint: int | None = None) -> float:
        array = values[key]
        episode_means = [
            float(np.mean(array[indices] if waypoint is None else array[indices, waypoint]))
            for indices in groups
        ]
        return float(np.mean(episode_means, dtype=np.float64))

    return {
        "translation_ade_mm": macro("translation_mm"),
        "translation_fde_mm": macro("translation_mm", -1),
        "translation_horizon_mm": {
            str(horizon): macro("translation_mm", horizon - 1) for horizon in HORIZONS
        },
        "rotation_geodesic_ade_deg": macro("rotation_geodesic_deg"),
        "rotation_geodesic_fde_deg": macro("rotation_geodesic_deg", -1),
        "rotation_geodesic_horizon_deg": {
            str(horizon): macro("rotation_geodesic_deg", horizon - 1) for horizon in HORIZONS
        },
        "gripper_mae": macro("gripper_abs"),
        "gripper_fde_mae": macro("gripper_abs", -1),
        "gripper_horizon_mae": {
            str(horizon): macro("gripper_abs", horizon - 1) for horizon in HORIZONS
        },
    }


def _tree_mean_variance(trees: Sequence[Any]) -> tuple[Any, Any]:
    _require(len(trees) > 0, "cannot aggregate an empty metric list")
    first = trees[0]
    if isinstance(first, Mapping):
        _require(
            all(isinstance(tree, Mapping) and set(tree) == set(first) for tree in trees),
            "metric tree structure mismatch",
        )
        means: dict[str, Any] = {}
        variances: dict[str, Any] = {}
        for key in first:
            mean, variance = _tree_mean_variance([tree[key] for tree in trees])
            means[str(key)] = mean
            variances[str(key)] = variance
        return means, variances
    array = np.asarray(trees, dtype=np.float64)
    _require(array.ndim == 1 and np.all(np.isfinite(array)), "metric leaves must be finite scalars")
    return float(np.mean(array)), float(np.var(array, ddof=0))


def summarize_m0_section_6_2(
    predictions: Any,
    target: Any,
    pad: Any,
    episode_ids: Any,
    prediction_seeds: Sequence[int],
) -> dict[str, Any]:
    """Return only the M0 episode-macro mean/variance metric family."""

    prediction_array = _numpy(predictions, name="predictions")
    _require(
        prediction_array.ndim == 4
        and prediction_array.shape[2:] == (ACTION_CHUNK_SIZE, ACTION_DIM),
        f"predictions must have shape [seeds, anchors, {ACTION_CHUNK_SIZE}, {ACTION_DIM}]",
    )
    target_array = _validate_action_array(target, name="target")
    _require(prediction_array.shape[1:] == target_array.shape, "prediction and target shapes differ")
    seeds = tuple(int(seed) for seed in prediction_seeds)
    _require(len(seeds) == prediction_array.shape[0], "prediction seed count mismatch")
    _require(len(seeds) > 0 and len(set(seeds)) == len(seeds), "prediction seeds must be unique")
    pad_array = _validate_pad(pad, target_array.shape[:2])
    _require(not np.any(pad_array), "M0 evaluation does not admit padded action chunks")

    per_seed = [
        _episode_macro_error_tree(
            compute_action_errors(prediction, target_array, pad_array),
            episode_ids,
        )
        for prediction in prediction_array
    ]
    mean, variance = _tree_mean_variance(per_seed)
    return {
        "scope": "m0_teacher_forced_in_sample_reconstruction",
        "metric_definition_reference": "F0_PI0_FULL_TRAINING_22OF25.md section 6.2",
        "aggregation": {
            "primary": "episode_macro_then_prediction_seed_mean",
            "ade_mae": "mean over all 50 waypoints within each episode",
            "fde": "waypoint 50 (array index 49)",
            "horizons": list(HORIZONS),
            "episode_macro": "unweighted mean of per-episode metrics",
            "prediction_seed_mean": "unweighted mean across fixed prediction seeds",
            "prediction_seed_variance": "population variance across fixed prediction seeds (ddof=0)",
        },
        "mean": mean,
        "population_variance": variance,
    }


__all__ = [
    "ACTION_CHUNK_SIZE",
    "ACTION_DIM",
    "HORIZONS",
    "compute_action_errors",
    "fixed_noise_batch",
    "split_supervision",
    "summarize_m0_section_6_2",
]
