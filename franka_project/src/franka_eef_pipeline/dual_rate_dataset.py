"""Project-local LeRobot adapter for the Franka Cartesian EEF datasets.

The adapter supports single-rate 15 Hz and native single-rate 30 Hz datasets.
In both profiles the LeRobot table stores an auditable absolute carrier action
with layout ``[xyz, quaternion_xyzw, gripper]``.  This module exposes only
anchors marked valid by the conversion sidecars and converts each 50-row
carrier window to the model-visible 7D Cartesian action online.

The older 15 Hz-observation / 30 Hz-action dataset remains unsupported here. It
requires a true dual-rate sidecar sampler and must never be interpreted as
ordinary LeRobot row offsets.
"""

from __future__ import annotations

import copy
import json
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import encode_relative_action, quaternion_xyzw_to_matrix, rotation_6d_to_matrix
from .stats import REQUIRED_STATS, canonical_sha256


ACTION_KEY = "action"
ACTION_PAD_KEY = "action_is_pad"
STATE_KEY = "observation.state"

ACTION15_PROFILE = "action15"
ACTION15_OBSERVATION_FPS = 15
ACTION15_ACTION_FPS = 15
ACTION15_CHUNK_SIZE = 50
NATIVE30_PROFILE = "native30"
NATIVE30_OBSERVATION_FPS = 30
NATIVE30_ACTION_FPS = 30
NATIVE30_CHUNK_SIZE = 50
STATE_DIM = 10
ABSOLUTE_CARRIER_DIM = 8
MODEL_ACTION_DIM = 7

SUPPORTED_PROFILE_SPECS = {
    ACTION15_PROFILE: (
        ACTION15_OBSERVATION_FPS,
        ACTION15_ACTION_FPS,
        ACTION15_CHUNK_SIZE,
    ),
    NATIVE30_PROFILE: (
        NATIVE30_OBSERVATION_FPS,
        NATIVE30_ACTION_FPS,
        NATIVE30_CHUNK_SIZE,
    ),
}
LEGACY_DUAL_RATE_PROFILES = {"action30", "obs15_action30"}

DEFAULT_CAMERA_KEY_MAP = {
    "observation.images.camera1": "observation.images.base_0_rgb",
    "observation.images.camera2": "observation.images.left_wrist_0_rgb",
}

MODEL_ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "rotvec_x",
    "rotvec_y",
    "rotvec_z",
    "gripper_0_1",
]


class UnsupportedActionRateError(NotImplementedError):
    """Raised when the unsupported observation/action dual-rate path is requested."""


@dataclass(frozen=True, slots=True)
class AnchorRecord:
    """One compact training index and its row in the full LeRobot table."""

    logical_index: int
    episode_index: int
    raw_episode_id: str
    anchor_local_index: int
    main_local_row: int
    main_global_row: int
    camera_log_time_ns: int

    def logical_payload(self) -> dict[str, int | str]:
        """Return the exact payload hashed by the D3 converter."""

        return {
            "logical_index": self.logical_index,
            "episode_index": self.episode_index,
            "raw_episode_id": self.raw_episode_id,
            "anchor_local_index": self.anchor_local_index,
            "main_global_row": self.main_global_row,
            "camera_log_time_ns": self.camera_log_time_ns,
        }


def action15_delta_timestamps() -> dict[str, list[float]]:
    """Return carrier offsets 0..49 at the dataset's 15 Hz row rate."""

    return {
        ACTION_KEY: [
            offset / ACTION15_ACTION_FPS for offset in range(ACTION15_CHUNK_SIZE)
        ]
    }


def action_delta_timestamps(*, action_fps: int, chunk_size: int) -> dict[str, list[float]]:
    """Return same-rate carrier offsets for a supported Cartesian profile."""

    if (action_fps, chunk_size) not in {
        (ACTION15_ACTION_FPS, ACTION15_CHUNK_SIZE),
        (NATIVE30_ACTION_FPS, NATIVE30_CHUNK_SIZE),
    }:
        raise ValueError(
            f"unsupported same-rate action_fps/chunk_size: {(action_fps, chunk_size)}"
        )
    return {ACTION_KEY: [offset / action_fps for offset in range(chunk_size)]}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required dataset metadata is missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _int_field(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    scalar = array.reshape(-1)[0]
    integer = int(scalar)
    if integer != scalar:
        raise ValueError(f"{name} must be an integer, got {scalar!r}")
    return integer


def _require_supported_profile(
    payload: Mapping[str, Any], *, source: Path | str
) -> tuple[int, int, int]:
    profile = str(payload.get("profile", ""))
    observation_fps = payload.get("observation_fps")
    action_fps = payload.get("action_fps")
    chunk_size = payload.get("chunk_size")
    requires_dual_rate = bool(payload.get("requires_project_dual_rate_adapter", False))

    if profile in LEGACY_DUAL_RATE_PROFILES or chunk_size == 100 or requires_dual_rate:
        raise UnsupportedActionRateError(
            f"dual-rate 30 Hz Franka actions are not enabled in CartesianAnchorDataset ({source}); "
            "native 30/30 uses profile='native30', while obs15/action30 requires an "
            "explicit sidecar adapter"
        )

    expected = SUPPORTED_PROFILE_SPECS.get(profile)
    actual = (observation_fps, action_fps, chunk_size)
    if expected is None or actual != expected:
        supported = [
            (name, observation_rate, action_rate, horizon)
            for name, (observation_rate, action_rate, horizon) in SUPPORTED_PROFILE_SPECS.items()
        ]
        raise ValueError(
            f"unsupported Franka profile/rate contract in {source}: supported={supported}, "
            f"got={(profile, *actual)}"
        )
    return expected


def load_cartesian_profile(dataset_root: str | Path) -> dict[str, Any]:
    """Load and validate the supported Cartesian dataset profile metadata."""

    root = Path(dataset_root)
    path = root / "meta/franka_eef_profile.json"
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"profile metadata must be a JSON object: {path}")
    _require_supported_profile(payload, source=path)
    if payload.get("main_action_key") != ACTION_KEY:
        raise ValueError(f"profile main_action_key must be {ACTION_KEY!r}: {path}")
    if payload.get("requires_project_cartesian_adapter") is not True:
        raise ValueError(f"profile must require the project Cartesian adapter: {path}")
    return payload


def load_action15_profile(dataset_root: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for :func:`load_cartesian_profile`."""

    return load_cartesian_profile(dataset_root)


def _validate_on_disk_schema(
    dataset_root: Path, *, expected_observation_fps: int
) -> dict[str, Any]:
    path = dataset_root / "meta/info.json"
    info = _read_json(path)
    if not isinstance(info, dict) or not isinstance(info.get("features"), dict):
        raise ValueError(f"invalid LeRobot info metadata: {path}")
    if info.get("fps") != expected_observation_fps:
        raise ValueError(
            f"LeRobot table fps must be {expected_observation_fps}: {path}"
        )

    features = info["features"]
    expected_shapes = {
        STATE_KEY: [STATE_DIM],
        ACTION_KEY: [ABSOLUTE_CARRIER_DIM],
        **{
            source: [480, 640, 3]
            for source in DEFAULT_CAMERA_KEY_MAP
        },
    }
    for key, expected_shape in expected_shapes.items():
        if key not in features:
            raise ValueError(f"required on-disk feature {key!r} is missing: {path}")
        shape = list(features[key].get("shape", []))
        if key.startswith("observation.images."):
            if len(shape) != 3 or shape[-1] != 3:
                raise ValueError(f"camera feature {key!r} must be HWC RGB, got {shape}: {path}")
        elif shape != expected_shape:
            raise ValueError(
                f"on-disk feature {key!r} must have shape {expected_shape}, got {shape}: {path}"
            )
    return info


def _resolve_sidecar(dataset_root: Path, relative_path: str) -> Path:
    root = dataset_root.resolve()
    path = (dataset_root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"sidecar path escapes dataset root: {relative_path}")
    return path


def load_valid_cartesian_anchors(dataset_root: str | Path) -> tuple[AnchorRecord, ...]:
    """Build the compact training index from Cartesian anchor-map sidecars.

    A row is admitted only when ``valid_anchor_action15`` is true, its sidecar
    declares the profile's full carrier horizon, and the complete row range
    remains inside that episode's main-table boundary.  The ``action15`` column
    names are retained in native30 files for backward-compatible sidecar schema.
    """

    root = Path(dataset_root)
    profile = load_cartesian_profile(root)
    _, _, chunk_size = _require_supported_profile(profile, source=root)
    episode_index_path = root / "sidecars/episode_index.json"
    episode_entries = _read_json(episode_index_path)
    if not isinstance(episode_entries, list) or not episode_entries:
        raise ValueError(f"episode index must be a non-empty JSON list: {episode_index_path}")

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - LeRobot itself depends on pyarrow.
        raise RuntimeError("pyarrow is required to read Franka anchor-map sidecars") from error

    anchors: list[AnchorRecord] = []
    seen_global_rows: set[int] = set()
    required_columns = [
        "episode_index",
        "raw_episode_id",
        "camera_anchor_local_index",
        "main_local_row",
        "main_global_row",
        "camera_log_time_ns",
        "observation_valid",
        "valid_anchor_action15",
        "action15_local_start",
        "action15_count",
    ]

    for expected_episode_index, episode in enumerate(episode_entries):
        if not isinstance(episode, dict):
            raise ValueError(f"episode index entry {expected_episode_index} must be an object")
        episode_index = _int_field(episode.get("episode_index"), name="episode_index")
        if episode_index != expected_episode_index:
            raise ValueError(
                "episode_index.json must be in contiguous dataset order; "
                f"expected {expected_episode_index}, got {episode_index}"
            )
        raw_episode_id = str(episode.get("raw_episode_id", ""))
        sidecars = episode.get("sidecars")
        if not raw_episode_id or not isinstance(sidecars, dict) or "anchor_map" not in sidecars:
            raise ValueError(f"incomplete sidecar entry for episode {episode_index}")

        global_start = _int_field(
            episode.get("global_main_row_start"), name="global_main_row_start"
        )
        global_end = _int_field(
            episode.get("global_main_row_end_exclusive"),
            name="global_main_row_end_exclusive",
        )
        main_rows = _int_field(episode.get("main_rows"), name="main_rows")
        expected_valid_count = _int_field(
            episode.get("valid_anchors"), name="valid_anchors"
        )
        if global_end - global_start != main_rows or main_rows <= 0:
            raise ValueError(f"invalid main-table bounds for episode {episode_index}")

        anchor_map_path = _resolve_sidecar(root, str(sidecars["anchor_map"]))
        if not anchor_map_path.is_file():
            raise FileNotFoundError(f"anchor-map sidecar is missing: {anchor_map_path}")
        table = pq.read_table(anchor_map_path, columns=required_columns)
        episode_anchor_count = 0
        for row in table.to_pylist():
            if not bool(row["valid_anchor_action15"]):
                continue
            episode_anchor_count += 1
            row_episode = _int_field(row["episode_index"], name="anchor episode_index")
            row_raw_episode = str(row["raw_episode_id"])
            anchor_local = _int_field(
                row["camera_anchor_local_index"], name="camera_anchor_local_index"
            )
            main_local = _int_field(row["main_local_row"], name="main_local_row")
            main_global = _int_field(row["main_global_row"], name="main_global_row")
            action_start = _int_field(
                row["action15_local_start"], name="action15_local_start"
            )
            action_count = _int_field(row["action15_count"], name="action15_count")

            if row_episode != episode_index or row_raw_episode != raw_episode_id:
                raise ValueError(f"anchor-map episode identity mismatch: {anchor_map_path}")
            if not bool(row["observation_valid"]):
                raise ValueError(f"valid action anchor has an invalid observation: {anchor_map_path}")
            if anchor_local != main_local or action_start != main_local:
                raise ValueError(
                    f"action15 anchor/start does not match the main row in {anchor_map_path}"
                )
            if main_global != global_start + main_local:
                raise ValueError(f"local/global row mismatch in {anchor_map_path}")
            if action_count != chunk_size:
                raise ValueError(
                    f"valid Cartesian anchor must declare {chunk_size} rows, "
                    f"got {action_count}: {anchor_map_path}"
                )
            if not (global_start <= main_global and main_global + chunk_size <= global_end):
                raise ValueError(
                    f"Cartesian action horizon crosses the episode boundary at global row "
                    f"{main_global}"
                )
            if main_global in seen_global_rows:
                raise ValueError(f"duplicate valid main_global_row: {main_global}")
            seen_global_rows.add(main_global)
            anchors.append(
                AnchorRecord(
                    logical_index=len(anchors),
                    episode_index=episode_index,
                    raw_episode_id=raw_episode_id,
                    anchor_local_index=anchor_local,
                    main_local_row=main_local,
                    main_global_row=main_global,
                    camera_log_time_ns=_int_field(
                        row["camera_log_time_ns"], name="camera_log_time_ns"
                    ),
                )
            )

        if episode_anchor_count != expected_valid_count:
            raise ValueError(
                f"valid-anchor count mismatch for episode {episode_index}: "
                f"index={expected_valid_count}, sidecar={episode_anchor_count}"
            )

    if not anchors:
        raise ValueError("the dataset has no valid full-horizon Cartesian anchors")
    if any(
        first.main_global_row >= second.main_global_row
        for first, second in zip(anchors, anchors[1:], strict=False)
    ):
        raise ValueError("valid anchors must be strictly ordered by main_global_row")

    logical_hash = canonical_sha256([anchor.logical_payload() for anchor in anchors])
    expected_hash = profile.get("logical_anchor_index_sha256")
    if expected_hash is not None and logical_hash != expected_hash:
        raise ValueError(
            "valid-anchor sidecars do not match profile logical_anchor_index_sha256: "
            f"expected {expected_hash}, got {logical_hash}"
        )
    if profile.get("episode_count") is not None and int(profile["episode_count"]) != len(
        episode_entries
    ):
        raise ValueError("profile episode_count does not match sidecars/episode_index.json")
    return tuple(anchors)


def load_valid_action15_anchors(dataset_root: str | Path) -> tuple[AnchorRecord, ...]:
    """Backward-compatible alias for :func:`load_valid_cartesian_anchors`."""

    return load_valid_cartesian_anchors(dataset_root)


def load_train_monitor_indices(
    monitor_path: str | Path,
    valid_anchors: Sequence[AnchorRecord],
) -> tuple[int, ...]:
    """Resolve the frozen monitor subset to compact wrapper indices."""

    path = Path(monitor_path)
    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("anchors"), list):
        raise ValueError(f"invalid train-monitor metadata: {path}")
    if payload.get("source") != "training_data" or payload.get("held_out") is not False:
        raise ValueError(f"monitor subset must be an in-sample training diagnostic: {path}")
    if int(payload.get("population_size", -1)) != len(valid_anchors):
        raise ValueError(f"monitor population_size does not match valid anchors: {path}")
    monitor_rows = payload["anchors"]
    if int(payload.get("sample_size", -1)) != len(monitor_rows):
        raise ValueError(f"monitor sample_size does not match its anchor list: {path}")
    expected_anchor_hash = payload.get("anchor_list_sha256")
    if expected_anchor_hash is not None:
        actual_anchor_hash = canonical_sha256(monitor_rows)
        if actual_anchor_hash != expected_anchor_hash:
            raise ValueError(
                f"monitor anchor_list_sha256 mismatch: expected {expected_anchor_hash}, "
                f"got {actual_anchor_hash}"
            )

    indices: list[int] = []
    seen: set[int] = set()
    for monitor_anchor in monitor_rows:
        if not isinstance(monitor_anchor, dict):
            raise ValueError(f"monitor anchor entries must be objects: {path}")
        logical_index = _int_field(monitor_anchor.get("logical_index"), name="logical_index")
        if not 0 <= logical_index < len(valid_anchors):
            raise ValueError(f"monitor logical_index is outside the valid-anchor list: {logical_index}")
        if logical_index in seen:
            raise ValueError(f"monitor contains duplicate logical_index: {logical_index}")
        anchor = valid_anchors[logical_index]
        for key, expected in anchor.logical_payload().items():
            if monitor_anchor.get(key) != expected:
                raise ValueError(
                    f"monitor anchor {logical_index} disagrees with sidecars on {key}: "
                    f"expected {expected!r}, got {monitor_anchor.get(key)!r}"
                )
        seen.add(logical_index)
        indices.append(logical_index)
    return tuple(indices)


def load_effective_pi0_stats(
    stats_path: str | Path,
    *,
    expected_source_dataset_hash: str | None = None,
    expected_logical_anchor_index_sha256: str | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load only model-visible 10D state and 7D action statistics.

    On-disk image statistics and the absolute 8D carrier statistics are never
    returned. Rate/chunk metadata is checked before any tensor is exposed.
    """

    path = Path(stats_path)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"effective stats must be a JSON object: {path}")
    _require_supported_profile(payload, source=path)
    if (
        expected_source_dataset_hash is not None
        and payload.get("source_dataset_hash") != expected_source_dataset_hash
    ):
        raise ValueError(f"effective stats source_dataset_hash mismatch: {path}")
    if (
        expected_logical_anchor_index_sha256 is not None
        and payload.get("logical_anchor_index_sha256")
        != expected_logical_anchor_index_sha256
    ):
        raise ValueError(f"effective stats logical-anchor hash mismatch: {path}")

    result: dict[str, dict[str, torch.Tensor]] = {}
    for feature_key, dimension in ((STATE_KEY, STATE_DIM), (ACTION_KEY, MODEL_ACTION_DIM)):
        feature_stats = payload.get(feature_key)
        if not isinstance(feature_stats, dict):
            raise ValueError(f"effective stats are missing {feature_key!r}: {path}")
        if not set(REQUIRED_STATS).issubset(feature_stats):
            missing = sorted(set(REQUIRED_STATS) - set(feature_stats))
            raise ValueError(f"effective stats {feature_key!r} are missing {missing}: {path}")

        tensor_stats: dict[str, torch.Tensor] = {}
        for stat_name in REQUIRED_STATS:
            dtype = torch.int64 if stat_name == "count" else torch.float32
            tensor = torch.as_tensor(feature_stats[stat_name], dtype=dtype)
            expected_shape = (1,) if stat_name == "count" else (dimension,)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{feature_key}.{stat_name} must have shape {expected_shape}, "
                    f"got {tuple(tensor.shape)}: {path}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{feature_key}.{stat_name} contains NaN or Inf: {path}")
            tensor_stats[stat_name] = tensor
        if tensor_stats["count"].item() <= 0:
            raise ValueError(f"{feature_key}.count must be positive: {path}")
        if torch.any(tensor_stats["std"] < 0):
            raise ValueError(f"{feature_key}.std must be non-negative: {path}")
        result[feature_key] = tensor_stats
    return result


def absolute_carrier_to_relative_action(
    state: Any,
    absolute_carrier: Any,
) -> torch.Tensor:
    """Convert current 10D rot6d state plus ``[50,8]`` carrier to ``[50,7]``."""

    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    carrier_tensor = torch.as_tensor(absolute_carrier, dtype=torch.float32)
    if tuple(state_tensor.shape) != (STATE_DIM,):
        raise ValueError(f"{STATE_KEY} must have shape ({STATE_DIM},), got {tuple(state_tensor.shape)}")
    if tuple(carrier_tensor.shape) != (ACTION15_CHUNK_SIZE, ABSOLUTE_CARRIER_DIM):
        raise ValueError(
            f"absolute action carrier must have shape "
            f"({ACTION15_CHUNK_SIZE},{ABSOLUTE_CARRIER_DIM}), got {tuple(carrier_tensor.shape)}"
        )
    if not torch.isfinite(state_tensor).all() or not torch.isfinite(carrier_tensor).all():
        raise ValueError("state and absolute action carrier must be finite")

    state_numpy = state_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    carrier_numpy = carrier_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    anchor_rotation = rotation_6d_to_matrix(state_numpy[3:9])
    target_rotation = quaternion_xyzw_to_matrix(carrier_numpy[:, 3:7])
    relative = encode_relative_action(
        state_numpy[:3],
        anchor_rotation,
        carrier_numpy[:, :3],
        target_rotation,
        carrier_numpy[:, 7],
    )
    return torch.as_tensor(relative, dtype=torch.float32, device=carrier_tensor.device)


class CartesianDatasetMetadata:
    """Minimal effective metadata view used by LeRobot policy factories.

    Unknown attributes delegate to the original metadata object, while feature
    names/shapes, statistics, fps and counts describe this adapter's model-visible
    dataset rather than the on-disk 8D carrier table.
    """

    def __init__(
        self,
        base_meta: Any,
        *,
        features: Mapping[str, Mapping[str, Any]],
        stats: Mapping[str, Mapping[str, torch.Tensor]] | None,
        fps: int,
        total_frames: int,
        total_episodes: int,
        camera_keys: Sequence[str],
    ) -> None:
        self._base_meta = base_meta
        self.features = copy.deepcopy(dict(features))
        self.stats = (
            {key: dict(value) for key, value in stats.items()} if stats is not None else None
        )
        self.fps = fps
        self.total_frames = total_frames
        self.total_episodes = total_episodes
        self.camera_keys = list(camera_keys)
        self.video_keys = list(camera_keys)

    def __getattr__(self, name: str) -> Any:
        if self._base_meta is None:
            raise AttributeError(name)
        return getattr(self._base_meta, name)


class CartesianAnchorDataset(Dataset[dict[str, Any]]):
    """Compact single-rate dataset view over valid Cartesian anchors.

    The normal trainer-facing constructor is::

        CartesianAnchorDataset(
            root,
            profile="action15",
            episode_indices=None,
            max_anchors_per_episode=None,
            video_backend="pyav",
        )

    ``from_base_dataset`` exists only for focused tests and local composition.
    """

    def __init__(
        self,
        root: str | Path,
        profile: str = ACTION15_PROFILE,
        episode_indices: Sequence[int] | None = None,
        max_anchors_per_episode: int | None = None,
        video_backend: str = "pyav",
        *,
        repo_id: str | None = None,
        dataset_factory: Callable[..., Dataset] | None = None,
        dataset_kwargs: Mapping[str, Any] | None = None,
        camera_key_map: Mapping[str, str] = DEFAULT_CAMERA_KEY_MAP,
    ) -> None:
        if profile in LEGACY_DUAL_RATE_PROFILES:
            raise UnsupportedActionRateError(
                "dual-rate 30 Hz obs15/action30 requires an explicit sidecar adapter; "
                "use profile='native30' only for native obs30/action30 data"
            )
        if profile not in SUPPORTED_PROFILE_SPECS:
            raise ValueError(f"unsupported Cartesian profile: {profile!r}")
        dataset_root = Path(root).resolve()
        profile_metadata = load_cartesian_profile(dataset_root)
        if profile_metadata["profile"] != profile:
            raise ValueError(
                f"requested profile {profile!r} does not match dataset profile "
                f"{profile_metadata['profile']!r}"
            )
        observation_fps, action_fps, chunk_size = _require_supported_profile(
            profile_metadata, source=dataset_root / "meta/franka_eef_profile.json"
        )
        _validate_on_disk_schema(
            dataset_root, expected_observation_fps=observation_fps
        )
        all_anchors = load_valid_cartesian_anchors(dataset_root)
        all_monitor_logical_indices = load_train_monitor_indices(
            dataset_root / "meta/train_monitor_subset.json", all_anchors
        )
        effective_stats = load_effective_pi0_stats(
            dataset_root / "meta/pi0_eef_stats.json",
            expected_source_dataset_hash=profile_metadata.get("source_dataset_hash"),
            expected_logical_anchor_index_sha256=profile_metadata.get(
                "logical_anchor_index_sha256"
            ),
        )

        selected_episode_indices = self._validate_episode_selection(
            episode_indices,
            episode_count=int(profile_metadata.get("episode_count", 0)),
        )
        selected_anchors = self._select_anchors(
            all_anchors,
            episode_indices=selected_episode_indices,
            max_anchors_per_episode=max_anchors_per_episode,
        )
        compact_index_by_logical = {
            anchor.logical_index: compact_index
            for compact_index, anchor in enumerate(selected_anchors)
        }
        monitor_indices = tuple(
            compact_index_by_logical[logical_index]
            for logical_index in all_monitor_logical_indices
            if logical_index in compact_index_by_logical
        )

        kwargs = dict(dataset_kwargs or {})
        reserved = {"repo_id", "root", "episodes", "episode_filter", "delta_timestamps"}
        overlap = reserved.intersection(kwargs)
        if overlap:
            raise ValueError(
                f"dataset_kwargs cannot override sidecar-critical arguments: {sorted(overlap)}"
            )
        if "video_backend" in kwargs and kwargs["video_backend"] != video_backend:
            raise ValueError("video_backend was specified twice with different values")
        kwargs["video_backend"] = video_backend
        kwargs.setdefault("return_uint8", True)
        if dataset_factory is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset_factory = LeRobotDataset
        resolved_repo_id = repo_id or profile_metadata.get("repo_id")
        if not resolved_repo_id:
            raise ValueError("repo_id is missing from both the call and profile metadata")
        base_dataset = dataset_factory(
            repo_id=str(resolved_repo_id),
            root=dataset_root,
            delta_timestamps=action_delta_timestamps(
                action_fps=action_fps, chunk_size=chunk_size
            ),
            **kwargs,
        )
        self._initialize_components(
            base_dataset,
            selected_anchors,
            train_monitor_indices=monitor_indices,
            effective_stats=effective_stats,
            camera_key_map=camera_key_map,
            profile=str(profile_metadata["profile"]),
            observation_fps=int(profile_metadata["observation_fps"]),
            action_fps=int(profile_metadata["action_fps"]),
            chunk_size=int(profile_metadata["chunk_size"]),
            task_instruction=profile_metadata.get("task_instruction"),
        )

    @staticmethod
    def _validate_episode_selection(
        episode_indices: Sequence[int] | None,
        *,
        episode_count: int,
    ) -> tuple[int, ...] | None:
        if episode_indices is None:
            return None
        selected = tuple(operator.index(index) for index in episode_indices)
        if not selected:
            raise ValueError("episode_indices cannot be empty")
        if len(selected) != len(set(selected)):
            raise ValueError("episode_indices contains duplicates")
        if any(index < 0 or index >= episode_count for index in selected):
            raise ValueError(
                f"episode_indices must be in [0,{episode_count}), got {selected}"
            )
        return selected

    @staticmethod
    def _select_anchors(
        anchors: Sequence[AnchorRecord],
        *,
        episode_indices: Sequence[int] | None,
        max_anchors_per_episode: int | None,
    ) -> tuple[AnchorRecord, ...]:
        if max_anchors_per_episode is not None:
            max_anchors_per_episode = operator.index(max_anchors_per_episode)
            if max_anchors_per_episode <= 0:
                raise ValueError("max_anchors_per_episode must be positive")
        allowed_episodes = set(episode_indices) if episode_indices is not None else None
        anchors_by_episode: dict[int, list[AnchorRecord]] = {}
        for anchor in anchors:
            if allowed_episodes is not None and anchor.episode_index not in allowed_episodes:
                continue
            anchors_by_episode.setdefault(anchor.episode_index, []).append(anchor)

        selected: list[AnchorRecord] = []
        for episode_index in sorted(anchors_by_episode):
            episode_anchors = anchors_by_episode[episode_index]
            if (
                max_anchors_per_episode is None
                or len(episode_anchors) <= max_anchors_per_episode
            ):
                selected.extend(episode_anchors)
                continue
            # Deterministic coverage of the full valid trajectory, including
            # both its first and last anchors.  The linspace step is >= 1 here,
            # so nearest-integer indices remain unique.
            positions = np.rint(
                np.linspace(
                    0,
                    len(episode_anchors) - 1,
                    num=max_anchors_per_episode,
                )
            ).astype(np.int64)
            if len(np.unique(positions)) != max_anchors_per_episode:
                raise AssertionError("uniform anchor selection produced duplicate positions")
            selected.extend(episode_anchors[int(position)] for position in positions)
        if not selected:
            raise ValueError("episode/max-anchor selection contains no valid full-horizon anchors")
        return tuple(selected)

    @classmethod
    def from_base_dataset(
        cls,
        base_dataset: Dataset,
        valid_anchors: Sequence[AnchorRecord],
        *,
        train_monitor_indices: Sequence[int] = (),
        effective_stats: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        camera_key_map: Mapping[str, str] = DEFAULT_CAMERA_KEY_MAP,
        profile: str = ACTION15_PROFILE,
        observation_fps: int = ACTION15_OBSERVATION_FPS,
        action_fps: int = ACTION15_ACTION_FPS,
        chunk_size: int = ACTION15_CHUNK_SIZE,
        task_instruction: str | None = None,
    ) -> "CartesianAnchorDataset":
        """Construct from an already-windowed base dataset (primarily for tests)."""

        instance = cls.__new__(cls)
        instance._initialize_components(
            base_dataset,
            valid_anchors,
            train_monitor_indices=train_monitor_indices,
            effective_stats=effective_stats,
            camera_key_map=camera_key_map,
            profile=profile,
            observation_fps=observation_fps,
            action_fps=action_fps,
            chunk_size=chunk_size,
            task_instruction=task_instruction,
        )
        return instance

    def _initialize_components(
        self,
        base_dataset: Dataset,
        valid_anchors: Sequence[AnchorRecord],
        *,
        train_monitor_indices: Sequence[int],
        effective_stats: Mapping[str, Mapping[str, torch.Tensor]] | None,
        camera_key_map: Mapping[str, str],
        profile: str,
        observation_fps: int,
        action_fps: int,
        chunk_size: int,
        task_instruction: str | None,
    ) -> None:
        _require_supported_profile(
            {
                "profile": profile,
                "observation_fps": observation_fps,
                "action_fps": action_fps,
                "chunk_size": chunk_size,
            },
            source="CartesianAnchorDataset constructor",
        )
        anchors = tuple(valid_anchors)
        if not anchors:
            raise ValueError("CartesianAnchorDataset requires at least one valid anchor")
        for anchor in anchors:
            if not isinstance(anchor, AnchorRecord):
                raise TypeError("valid_anchors entries must be AnchorRecord instances")
        logical_indices = [anchor.logical_index for anchor in anchors]
        if len(logical_indices) != len(set(logical_indices)) or any(
            first >= second
            for first, second in zip(logical_indices, logical_indices[1:], strict=False)
        ):
            raise ValueError("valid_anchors logical indices must be unique and strictly ordered")
        global_rows = [anchor.main_global_row for anchor in anchors]
        if len(global_rows) != len(set(global_rows)):
            raise ValueError("valid_anchors contains duplicate main_global_row values")
        if min(global_rows) < 0 or max(global_rows) >= len(base_dataset):
            raise ValueError("valid anchor points outside the base LeRobot dataset")

        selected_episodes = getattr(base_dataset, "episodes", None)
        if selected_episodes is not None:
            raise ValueError(
                "the base LeRobotDataset must load all episodes because sidecars use global row indices"
            )
        expected_deltas = action_delta_timestamps(
            action_fps=action_fps, chunk_size=chunk_size
        )
        actual_deltas = getattr(base_dataset, "delta_timestamps", None)
        if not isinstance(actual_deltas, dict) or set(actual_deltas) != {ACTION_KEY}:
            raise ValueError(
                "base LeRobotDataset must be constructed with only action delta timestamps 0..49"
            )
        actual_action_deltas = actual_deltas[ACTION_KEY]
        if len(actual_action_deltas) != chunk_size or not np.allclose(
            actual_action_deltas,
            expected_deltas[ACTION_KEY],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "base LeRobotDataset action delta timestamps must match "
                f"[0/{action_fps}, ..., {chunk_size - 1}/{action_fps}]"
            )

        camera_map = dict(camera_key_map)
        if len(camera_map) != 2 or len(set(camera_map.values())) != 2:
            raise ValueError("camera_key_map must map exactly two distinct source cameras")
        monitor_indices = tuple(operator.index(index) for index in train_monitor_indices)
        if len(monitor_indices) != len(set(monitor_indices)):
            raise ValueError("train_monitor_indices contains duplicates")
        if any(index < 0 or index >= len(anchors) for index in monitor_indices):
            raise ValueError("train_monitor_indices points outside valid_anchors")

        self.base_dataset = base_dataset
        self.valid_anchors = anchors
        self.logical_anchors = tuple(anchor.logical_payload() for anchor in anchors)
        self.valid_anchor_indices = tuple(global_rows)
        self.train_monitor_indices = monitor_indices
        self.train_monitor_anchors = tuple(anchors[index] for index in monitor_indices)
        self.effective_stats = (
            {key: dict(value) for key, value in effective_stats.items()}
            if effective_stats is not None
            else None
        )
        self.camera_key_map = camera_map
        self.profile = profile
        self.observation_fps = observation_fps
        self.action_fps = action_fps
        self.chunk_size = chunk_size
        if task_instruction is not None and (
            not isinstance(task_instruction, str) or not task_instruction.strip()
        ):
            raise ValueError("task_instruction must be a non-empty string when provided")
        self.task_instruction = task_instruction
        self._features = self._build_effective_features()
        self.meta = CartesianDatasetMetadata(
            getattr(base_dataset, "meta", None),
            features=self._features,
            stats=self.effective_stats,
            fps=self.observation_fps,
            total_frames=len(anchors),
            total_episodes=len({anchor.episode_index for anchor in anchors}),
            camera_keys=tuple(self.camera_key_map.values()),
        )

    @classmethod
    def from_lerobot_root(
        cls,
        dataset_root: str | Path,
        *,
        repo_id: str | None = None,
        dataset_factory: Callable[..., Dataset] | None = None,
        dataset_kwargs: Mapping[str, Any] | None = None,
        camera_key_map: Mapping[str, str] = DEFAULT_CAMERA_KEY_MAP,
        profile: str = ACTION15_PROFILE,
        episode_indices: Sequence[int] | None = None,
        max_anchors_per_episode: int | None = None,
        video_backend: str = "pyav",
    ) -> "CartesianAnchorDataset":
        """Compatibility alias for the trainer-facing constructor."""

        return cls(
            dataset_root,
            profile=profile,
            episode_indices=episode_indices,
            max_anchors_per_episode=max_anchors_per_episode,
            video_backend=video_backend,
            repo_id=repo_id,
            dataset_factory=dataset_factory,
            dataset_kwargs=dataset_kwargs,
            camera_key_map=camera_key_map,
        )

    def _build_effective_features(self) -> dict[str, dict[str, Any]]:
        raw_features = getattr(self.base_dataset, "features", None)
        if not isinstance(raw_features, Mapping):
            raise ValueError("base dataset must expose a LeRobot-compatible features mapping")
        features = copy.deepcopy(dict(raw_features))
        for source, destination in self.camera_key_map.items():
            if source not in features:
                raise ValueError(f"base feature schema is missing camera {source!r}")
            if destination in features and destination != source:
                raise ValueError(f"effective camera key already exists: {destination!r}")
            features[destination] = features.pop(source)
        if ACTION_KEY not in features or STATE_KEY not in features:
            raise ValueError("base feature schema is missing state or action")
        if tuple(features[ACTION_KEY].get("shape", ())) != (ABSOLUTE_CARRIER_DIM,):
            raise ValueError("base action feature must be the 8D absolute carrier")
        if tuple(features[STATE_KEY].get("shape", ())) != (STATE_DIM,):
            raise ValueError("base state feature must be 10D")
        features[ACTION_KEY]["shape"] = (MODEL_ACTION_DIM,)
        features[ACTION_KEY]["names"] = MODEL_ACTION_NAMES.copy()
        return features

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        """Model-visible feature schema (10D state, 7D action, canonical cameras)."""

        return copy.deepcopy(self._features)

    @property
    def fps(self) -> int:
        return self.observation_fps

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return len({anchor.episode_index for anchor in self.valid_anchors})

    @property
    def root(self) -> Path | None:
        value = getattr(self.base_dataset, "root", None)
        return Path(value) if value is not None else None

    @property
    def repo_id(self) -> str | None:
        return getattr(self.base_dataset, "repo_id", None)

    @property
    def camera_keys(self) -> tuple[str, str]:
        return tuple(self.camera_key_map.values())  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self.valid_anchors)

    def __getitem__(self, index: int) -> dict[str, Any]:
        logical_index = operator.index(index)
        if logical_index < 0:
            logical_index += len(self)
        if logical_index < 0 or logical_index >= len(self):
            raise IndexError(f"CartesianAnchorDataset index out of range: {index}")
        anchor = self.valid_anchors[logical_index]
        item = dict(self.base_dataset[anchor.main_global_row])

        if STATE_KEY not in item or ACTION_KEY not in item or ACTION_PAD_KEY not in item:
            raise KeyError(
                "base LeRobot item must contain observation.state, action, and action_is_pad"
            )
        if "episode_index" not in item or "index" not in item or "frame_index" not in item:
            raise KeyError("base LeRobot item is missing episode/index/frame_index audit fields")
        if _int_field(item["episode_index"], name="item episode_index") != anchor.episode_index:
            raise RuntimeError("sidecar anchor resolved to a different episode in the base dataset")
        if _int_field(item["index"], name="item index") != anchor.main_global_row:
            raise RuntimeError("sidecar main_global_row does not match the base dataset index field")
        if _int_field(item["frame_index"], name="item frame_index") != anchor.main_local_row:
            raise RuntimeError("sidecar main_local_row does not match the base dataset frame_index")

        pad_mask = torch.as_tensor(item[ACTION_PAD_KEY], dtype=torch.bool)
        if tuple(pad_mask.shape) != (self.chunk_size,):
            raise RuntimeError(
                f"action_is_pad must have shape ({self.chunk_size},), "
                f"got {tuple(pad_mask.shape)}"
            )
        if torch.any(pad_mask):
            raise RuntimeError(
                "a D3-valid anchor produced padded action rows; refusing an episode-crossing horizon"
            )

        state = torch.as_tensor(item[STATE_KEY], dtype=torch.float32)
        relative_action = absolute_carrier_to_relative_action(state, item[ACTION_KEY])
        item[STATE_KEY] = state
        item[ACTION_KEY] = relative_action
        item[ACTION_PAD_KEY] = pad_mask
        if self.task_instruction is not None and item.get("task") != self.task_instruction:
            raise RuntimeError(
                "LeRobot row task instruction does not match franka_eef_profile.json: "
                f"{item.get('task')!r} != {self.task_instruction!r}"
            )

        for source, destination in self.camera_key_map.items():
            if source not in item:
                raise KeyError(f"base LeRobot item is missing camera {source!r}")
            if destination in item and destination != source:
                raise KeyError(f"refusing to overwrite existing canonical camera {destination!r}")
            item[destination] = item.pop(source)
        return item


__all__ = [
    "ACTION15_ACTION_FPS",
    "ACTION15_CHUNK_SIZE",
    "ACTION15_OBSERVATION_FPS",
    "ACTION15_PROFILE",
    "NATIVE30_ACTION_FPS",
    "NATIVE30_CHUNK_SIZE",
    "NATIVE30_OBSERVATION_FPS",
    "NATIVE30_PROFILE",
    "ACTION_KEY",
    "ACTION_PAD_KEY",
    "AnchorRecord",
    "CartesianAnchorDataset",
    "CartesianDatasetMetadata",
    "DEFAULT_CAMERA_KEY_MAP",
    "MODEL_ACTION_DIM",
    "STATE_DIM",
    "STATE_KEY",
    "UnsupportedActionRateError",
    "absolute_carrier_to_relative_action",
    "action_delta_timestamps",
    "action15_delta_timestamps",
    "load_cartesian_profile",
    "load_action15_profile",
    "load_effective_pi0_stats",
    "load_train_monitor_indices",
    "load_valid_cartesian_anchors",
    "load_valid_action15_anchors",
]
