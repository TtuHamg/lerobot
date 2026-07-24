"""Pure-Python contracts for the FastWAM joint-space Franka ROS 2 boundary.

This module has no dependency on ``rclpy`` or ROS message packages. It mirrors
:mod:`ros2_contract` but replaces the Cartesian EEF state with the raw
joint-space state (seven Franka joints + one raw gripper position) that the
FastWAM policy consumes and produces.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contract import CAMERA_NAMES, CAMERA_SHAPE
from .joint_contract import JOINT_STATE_NAMES
from .ros2_contract import (
    FloatArray,
    ImageSample,
    JointStateSample,
    RgbImage,
    Ros2ContractError,
    _float_array,
    _integer,
    _nonempty_string,
    _positive_integer,
    _rgb_image,
)

_SOURCE_NAMES = ("camera1", "camera2", "qpos", "gripper")
_SKEW_SOURCE_NAMES = ("camera2", "qpos", "gripper")
_ACTION_DIM = 8
_QPOS_DIM = 7
_UINT64_MAX = 2**64 - 1


class _GripperSample:
    __slots__ = ("stamp_ns", "received_monotonic_ns", "position")

    def __init__(self, *, stamp_ns: int, received_monotonic_ns: int, position: float) -> None:
        self.stamp_ns = stamp_ns
        self.received_monotonic_ns = received_monotonic_ns
        self.position = position


class JointObservationSnapshot:
    """Immutable synchronized joint-space observation."""

    __slots__ = (
        "anchor_stamp_ns",
        "created_monotonic_ns",
        "frame_id",
        "gripper_pos",
        "_camera1",
        "_camera2",
        "_qpos",
        "_source_stamps_ns",
    )

    def __init__(
        self,
        *,
        anchor_stamp_ns: int,
        created_monotonic_ns: int,
        camera1: ArrayLike,
        camera2: ArrayLike,
        qpos: ArrayLike,
        gripper_pos: float,
        frame_id: str,
        source_stamps_ns: Mapping[str, int],
    ) -> None:
        self.anchor_stamp_ns = _positive_integer(anchor_stamp_ns, name="anchor_stamp_ns")
        self.created_monotonic_ns = _positive_integer(
            created_monotonic_ns,
            name="created_monotonic_ns",
        )
        self.frame_id = _nonempty_string(frame_id, name="frame_id")
        self._camera1 = _rgb_image(camera1, name="camera1", expected_shape=CAMERA_SHAPE)
        self._camera2 = _rgb_image(camera2, name="camera2", expected_shape=CAMERA_SHAPE)
        self._qpos = _float_array(qpos, shape=(_QPOS_DIM,), name="qpos")
        if isinstance(gripper_pos, bool) or not isinstance(gripper_pos, Real):
            raise Ros2ContractError("gripper_pos must be a real scalar")
        gripper = float(gripper_pos)
        if not math.isfinite(gripper):
            raise Ros2ContractError(f"gripper_pos must be finite, got {gripper}")
        self.gripper_pos = gripper

        stamps = dict(source_stamps_ns)
        if set(stamps) != set(_SOURCE_NAMES):
            raise Ros2ContractError(
                f"source_stamps_ns must contain exactly {_SOURCE_NAMES}, got {tuple(sorted(stamps))}"
            )
        self._source_stamps_ns = {
            name: _positive_integer(stamps[name], name=f"source_stamps_ns[{name!r}]")
            for name in _SOURCE_NAMES
        }
        if self._source_stamps_ns["camera1"] != self.anchor_stamp_ns:
            raise Ros2ContractError("camera1 source stamp must equal anchor_stamp_ns")

    @property
    def camera1(self) -> RgbImage:
        return self._camera1.copy()

    @property
    def camera2(self) -> RgbImage:
        return self._camera2.copy()

    @property
    def qpos(self) -> FloatArray:
        return self._qpos.copy()

    @property
    def source_stamps_ns(self) -> dict[str, int]:
        return self._source_stamps_ns.copy()

    @property
    def state(self) -> FloatArray:
        return np.concatenate((self._qpos, np.asarray([self.gripper_pos])))

    def as_policy_observation(self) -> dict[str, float | RgbImage]:
        state = self.state
        observation: dict[str, float | RgbImage] = {
            name: float(value) for name, value in zip(JOINT_STATE_NAMES, state, strict=True)
        }
        observation[CAMERA_NAMES[0]] = self._camera1.copy()
        observation[CAMERA_NAMES[1]] = self._camera2.copy()
        return observation

    def as_robot_observation(self) -> dict[str, float | RgbImage]:
        return self.as_policy_observation()

    def as_sideband(self) -> dict[str, Any]:
        return {
            "anchor_stamp_ns": self.anchor_stamp_ns,
            "created_monotonic_ns": self.created_monotonic_ns,
            "frame_id": self.frame_id,
            "source_stamps_ns": self._source_stamps_ns.copy(),
            "qpos": self._qpos.copy(),
            "gripper.pos": self.gripper_pos,
        }


class JointObservationCache:
    """Thread-safe causal cache anchored by the newest camera1 source stamp."""

    def __init__(
        self,
        *,
        joint_names: Sequence[str],
        base_frame: str,
        max_skew_ns: Mapping[str, int],
        max_age_ns: int,
        buffer_size: int = 32,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        names = tuple(joint_names)
        if len(names) != _QPOS_DIM:
            raise Ros2ContractError(f"joint_names must contain {_QPOS_DIM} names, got {len(names)}")
        if any(not isinstance(name, str) or not name for name in names):
            raise Ros2ContractError("joint_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise Ros2ContractError("joint_names contain duplicates")
        self.joint_names = names
        self.base_frame = _nonempty_string(base_frame, name="base_frame")
        if not isinstance(max_skew_ns, Mapping):
            raise Ros2ContractError("max_skew_ns must be a per-source mapping")
        skew_limits = dict(max_skew_ns)
        if set(skew_limits) != set(_SKEW_SOURCE_NAMES):
            raise Ros2ContractError(
                f"max_skew_ns must contain exactly {_SKEW_SOURCE_NAMES}, "
                f"got {tuple(sorted(skew_limits))}"
            )
        self.max_skew_ns = {
            source: _positive_integer(skew_limits[source], name=f"max_skew_ns[{source!r}]")
            for source in _SKEW_SOURCE_NAMES
        }
        self.max_age_ns = _positive_integer(max_age_ns, name="max_age_ns")
        self.buffer_size = _positive_integer(buffer_size, name="buffer_size")
        if not callable(monotonic_ns):
            raise Ros2ContractError("monotonic_ns must be callable")
        self._monotonic_ns = monotonic_ns
        self._lock = threading.RLock()
        self._buffers: dict[str, list[Any]] = {name: [] for name in _SOURCE_NAMES}

    def _insert(self, source: str, sample: Any) -> None:
        with self._lock:
            buffer = self._buffers[source]
            for index, existing in enumerate(buffer):
                if existing.stamp_ns == sample.stamp_ns:
                    if sample.received_monotonic_ns >= existing.received_monotonic_ns:
                        buffer[index] = sample
                    break
            else:
                buffer.append(sample)
            buffer.sort(key=lambda item: (item.stamp_ns, item.received_monotonic_ns))
            if len(buffer) > self.buffer_size:
                del buffer[: len(buffer) - self.buffer_size]

    def _validate_image(self, sample: ImageSample, *, name: str) -> None:
        if not isinstance(sample, ImageSample):
            raise Ros2ContractError(f"{name} must be an ImageSample")
        if sample.image.shape != CAMERA_SHAPE:
            raise Ros2ContractError(f"{name} must have shape {CAMERA_SHAPE}, got {sample.image.shape}")

    def update_camera1(self, sample: ImageSample) -> None:
        self._validate_image(sample, name="camera1")
        self._insert("camera1", sample)

    def update_camera2(self, sample: ImageSample) -> None:
        self._validate_image(sample, name="camera2")
        self._insert("camera2", sample)

    def update_qpos(self, sample: JointStateSample) -> None:
        if not isinstance(sample, JointStateSample):
            raise Ros2ContractError("qpos must be a JointStateSample")
        if set(sample.names) != set(self.joint_names) or len(sample.names) != len(self.joint_names):
            missing = sorted(set(self.joint_names) - set(sample.names))
            extra = sorted(set(sample.names) - set(self.joint_names))
            raise Ros2ContractError(f"qpos joint names do not match; missing={missing}, extra={extra}")
        positions_by_name = dict(zip(sample.names, sample.positions, strict=True))
        canonical = JointStateSample(
            stamp_ns=sample.stamp_ns,
            received_monotonic_ns=sample.received_monotonic_ns,
            names=self.joint_names,
            positions=np.asarray([positions_by_name[name] for name in self.joint_names]),
        )
        self._insert("qpos", canonical)

    def update_gripper(
        self,
        position: float,
        *,
        stamp_ns: int,
        received_monotonic_ns: int,
    ) -> None:
        if isinstance(position, bool) or not isinstance(position, Real):
            raise Ros2ContractError("gripper position must be a real scalar")
        value = float(position)
        if not math.isfinite(value):
            raise Ros2ContractError(f"gripper position must be finite, got {value}")
        sample = _GripperSample(
            stamp_ns=_positive_integer(stamp_ns, name="stamp_ns"),
            received_monotonic_ns=_positive_integer(
                received_monotonic_ns,
                name="received_monotonic_ns",
            ),
            position=value,
        )
        self._insert("gripper", sample)

    def _now_ns(self) -> int:
        return _positive_integer(self._monotonic_ns(), name="monotonic clock")

    def _require_fresh(self, source: str, sample: Any, *, now_ns: int) -> None:
        age_ns = now_ns - sample.received_monotonic_ns
        if age_ns < 0:
            raise Ros2ContractError(f"{source} arrival timestamp is in the future by {-age_ns} ns")
        if age_ns > self.max_age_ns:
            raise Ros2ContractError(
                f"{source} is stale: age_ns={age_ns}, max_age_ns={self.max_age_ns}"
            )

    def _select(self, source: str, *, anchor_stamp_ns: int, now_ns: int) -> Any:
        candidates = [sample for sample in self._buffers[source] if sample.stamp_ns <= anchor_stamp_ns]
        if not candidates:
            raise Ros2ContractError(
                f"no {source} sample exists at or before camera1 anchor {anchor_stamp_ns}"
            )
        sample = max(candidates, key=lambda item: (item.stamp_ns, item.received_monotonic_ns))
        skew_ns = anchor_stamp_ns - sample.stamp_ns
        max_skew_ns = self.max_skew_ns[source]
        if skew_ns > max_skew_ns:
            raise Ros2ContractError(
                f"{source} skew exceeds limit: skew_ns={skew_ns}, max_skew_ns={max_skew_ns}"
            )
        self._require_fresh(source, sample, now_ns=now_ns)
        return sample

    @property
    def latest_anchor_stamp_ns(self) -> int | None:
        with self._lock:
            if not self._buffers["camera1"]:
                return None
            return max(sample.stamp_ns for sample in self._buffers["camera1"])

    def get_joint_snapshot(self) -> JointObservationSnapshot:
        now_ns = self._now_ns()
        with self._lock:
            if not self._buffers["camera1"]:
                raise Ros2ContractError("no camera1 anchor is available")
            anchor = max(
                self._buffers["camera1"],
                key=lambda item: (item.stamp_ns, item.received_monotonic_ns),
            )
            self._require_fresh("camera1", anchor, now_ns=now_ns)
            camera2 = self._select("camera2", anchor_stamp_ns=anchor.stamp_ns, now_ns=now_ns)
            qpos = self._select("qpos", anchor_stamp_ns=anchor.stamp_ns, now_ns=now_ns)
            gripper = self._select("gripper", anchor_stamp_ns=anchor.stamp_ns, now_ns=now_ns)
            source_stamps_ns = {
                "camera1": anchor.stamp_ns,
                "camera2": camera2.stamp_ns,
                "qpos": qpos.stamp_ns,
                "gripper": gripper.stamp_ns,
            }
            return JointObservationSnapshot(
                anchor_stamp_ns=anchor.stamp_ns,
                created_monotonic_ns=now_ns,
                camera1=anchor.image,
                camera2=camera2.image,
                qpos=qpos.positions,
                gripper_pos=gripper.position,
                frame_id=self.base_frame,
                source_stamps_ns=source_stamps_ns,
            )

    def clear(self) -> None:
        with self._lock:
            for buffer in self._buffers.values():
                buffer.clear()


class JointActionChunk:
    """Validated client-side representation of one absolute joint-space plan."""

    __slots__ = (
        "schema_version",
        "source_timestep",
        "source_observation_timestamp_ns",
        "server_send_timestamp_ns",
        "received_monotonic_ns",
        "period_ns",
        "frame_id",
        "session_id",
        "plan_id",
        "valid_for_ns",
        "_actions",
        "_timesteps",
    )

    def __init__(
        self,
        *,
        actions: ArrayLike,
        timesteps: Sequence[int],
        source_timestep: int,
        source_observation_timestamp_ns: int,
        server_send_timestamp_ns: int,
        received_monotonic_ns: int,
        period_ns: int,
        frame_id: str,
        session_id: str,
        plan_id: int,
        valid_for_ns: int,
    ) -> None:
        try:
            raw_actions = np.asarray(actions)
        except (TypeError, ValueError) as error:
            raise Ros2ContractError("actions are not a rectangular numeric array") from error
        if raw_actions.dtype.kind != "f":
            raise Ros2ContractError(f"actions must have floating dtype, got {raw_actions.dtype}")
        if raw_actions.ndim != 2 or raw_actions.shape[1:] != (_ACTION_DIM,) or raw_actions.shape[0] == 0:
            raise Ros2ContractError(
                f"actions must have non-empty shape [K,{_ACTION_DIM}], got {raw_actions.shape}"
            )
        if not np.isfinite(raw_actions).all():
            raise Ros2ContractError("actions contain NaN or Inf")
        canonical_actions = np.asarray(raw_actions, dtype=np.float64)
        if not np.isfinite(canonical_actions).all():
            raise Ros2ContractError("actions overflow float64")

        canonical_timesteps = tuple(
            _integer(timestep, name=f"timesteps[{index}]")
            for index, timestep in enumerate(timesteps)
        )
        if len(canonical_timesteps) != canonical_actions.shape[0]:
            raise Ros2ContractError(
                "timesteps length must equal action chunk length; "
                f"got {len(canonical_timesteps)} and {canonical_actions.shape[0]}"
            )
        for previous, current in zip(canonical_timesteps, canonical_timesteps[1:]):
            if current != previous + 1:
                raise Ros2ContractError("action timesteps must be strictly contiguous")

        canonical_source_timestep = _integer(source_timestep, name="source_timestep")
        if canonical_source_timestep > canonical_timesteps[0]:
            raise Ros2ContractError(
                "source_timestep must not be later than the first action timestep; "
                f"got {canonical_timesteps[0]} and {canonical_source_timestep}"
            )

        self.schema_version = 1
        self.source_timestep = canonical_source_timestep
        self.source_observation_timestamp_ns = _positive_integer(
            source_observation_timestamp_ns,
            name="source_observation_timestamp_ns",
        )
        self.server_send_timestamp_ns = _positive_integer(
            server_send_timestamp_ns,
            name="server_send_timestamp_ns",
        )
        self.received_monotonic_ns = _positive_integer(
            received_monotonic_ns,
            name="received_monotonic_ns",
        )
        self.period_ns = _positive_integer(period_ns, name="period_ns")
        self.frame_id = _nonempty_string(frame_id, name="frame_id")
        self.session_id = _nonempty_string(session_id, name="session_id")
        self.plan_id = _integer(plan_id, name="plan_id")
        if self.plan_id > _UINT64_MAX:
            raise Ros2ContractError(f"plan_id must fit uint64, got {self.plan_id}")
        self.valid_for_ns = _positive_integer(valid_for_ns, name="valid_for_ns")
        canonical_actions = np.ascontiguousarray(canonical_actions).copy()
        canonical_actions.setflags(write=False)
        self._actions = canonical_actions
        self._timesteps = canonical_timesteps

    def __len__(self) -> int:
        return self._actions.shape[0]

    @property
    def actions(self) -> NDArray[np.float64]:
        return self._actions.copy()

    @property
    def timesteps(self) -> tuple[int, ...]:
        return self._timesteps

    @property
    def positions(self) -> NDArray[np.float64]:
        """Copied ``[K,7]`` joint-angle rows matching the ROS message field."""

        return self._actions[:, :7].copy()

    @property
    def gripper(self) -> NDArray[np.float64]:
        """Copied ``[K]`` raw gripper position values matching the ROS message field."""

        return self._actions[:, 7].copy()

    @property
    def client_observation_stamp_ns(self) -> int:
        return self.source_observation_timestamp_ns

    @property
    def source_observation_stamp_ns(self) -> int:
        return self.source_observation_timestamp_ns

    @property
    def server_send_stamp_ns(self) -> int:
        return self.server_send_timestamp_ns

    @property
    def action_stamps_ns(self) -> tuple[int, ...]:
        first = self.source_observation_timestamp_ns
        first_timestep = self.source_timestep
        return tuple(
            first + (timestep - first_timestep) * self.period_ns for timestep in self._timesteps
        )

    def is_valid_at(self, monotonic_ns: int) -> bool:
        now_ns = _positive_integer(monotonic_ns, name="monotonic_ns")
        return self.received_monotonic_ns <= now_ns <= self.received_monotonic_ns + self.valid_for_ns

    def require_valid_at(self, monotonic_ns: int) -> None:
        if not self.is_valid_at(monotonic_ns):
            raise Ros2ContractError(
                f"action chunk is outside its local validity window at monotonic_ns={monotonic_ns}"
            )

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "actions": self._actions.copy(),
            "positions": self._actions[:, :7].copy(),
            "gripper": self._actions[:, 7].copy(),
            "timesteps": self._timesteps,
            "action_stamps_ns": self.action_stamps_ns,
            "source_timestep": self.source_timestep,
            "source_observation_timestamp_ns": self.source_observation_timestamp_ns,
            "client_observation_stamp_ns": self.client_observation_stamp_ns,
            "server_send_timestamp_ns": self.server_send_timestamp_ns,
            "server_send_stamp_ns": self.server_send_stamp_ns,
            "received_monotonic_ns": self.received_monotonic_ns,
            "period_ns": self.period_ns,
            "frame_id": self.frame_id,
            "session_id": self.session_id,
            "plan_id": self.plan_id,
            "valid_for_ns": self.valid_for_ns,
        }


__all__ = [
    "JointActionChunk",
    "JointObservationCache",
    "JointObservationSnapshot",
]
