"""LeRobot-style effective statistics for model-visible EEF features."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


REQUIRED_STATS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def feature_stats(values: ArrayLike) -> dict[str, NDArray[np.floating]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"feature values must be [samples,dim], got {array.shape}")
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("feature values must be non-empty and finite")
    quantiles = np.quantile(array, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
        "count": np.asarray([len(array)], dtype=np.int64),
        "q01": quantiles[0],
        "q10": quantiles[1],
        "q50": quantiles[2],
        "q90": quantiles[3],
        "q99": quantiles[4],
    }


def effective_eef_stats(
    observation_state: ArrayLike,
    action_chunks: ArrayLike,
    *,
    observation_fps: int,
    action_fps: int,
    chunk_size: int,
    source_dataset_hash: str,
) -> dict[str, Any]:
    state = np.asarray(observation_state, dtype=np.float64)
    action = np.asarray(action_chunks, dtype=np.float64)
    if state.ndim != 2 or state.shape[-1] != 10:
        raise ValueError(f"observation_state must be [N,10], got {state.shape}")
    if action.ndim != 3 or action.shape[1:] != (chunk_size, 7):
        raise ValueError(f"action_chunks must be [A,{chunk_size},7], got {action.shape}")
    return {
        "schema_version": 1,
        "observation_fps": int(observation_fps),
        "action_fps": int(action_fps),
        "chunk_size": int(chunk_size),
        "source_dataset_hash": source_dataset_hash,
        "observation.state": feature_stats(state),
        "action": feature_stats(action.reshape((-1, 7))),
    }


def normalize(values: ArrayLike, stats: dict[str, ArrayLike], *, epsilon: float = 1e-6) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    scale = np.maximum(std, epsilon)
    return (array - mean) / scale


def unnormalize(values: ArrayLike, stats: dict[str, ArrayLike], *, epsilon: float = 1e-6) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    scale = np.maximum(std, epsilon)
    return array * scale + mean


def json_ready(stats: Any) -> Any:
    if isinstance(stats, np.ndarray):
        return stats.tolist()
    if isinstance(stats, np.generic):
        return stats.item()
    if isinstance(stats, dict):
        return {key: json_ready(value) for key, value in stats.items()}
    if isinstance(stats, (list, tuple)):
        return [json_ready(value) for value in stats]
    return stats


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
