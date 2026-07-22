"""Pure-Python contracts for the Franka ROS 2 boundary.

This module deliberately has no dependency on ``rclpy`` or ROS message
packages. The runtime adapter translates ROS messages into the samples below
and translates :class:`AbsoluteActionChunk` back to ROS without moving
synchronization or validation rules into callback code.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contract import CAMERA_NAMES, CAMERA_SHAPE, STATE_NAMES


FloatArray = NDArray[np.float64]
RgbImage = NDArray[np.uint8]

_SOURCE_NAMES = ("camera1", "camera2", "eef", "qpos", "gripper")
_SKEW_SOURCE_NAMES = ("camera2", "eef", "qpos", "gripper")
_ACTION_DIM = 8
_QPOS_DIM = 7
_QUATERNION_NORM_TOLERANCE = 1e-3
_UINT64_MAX = 2**64 - 1


class Ros2ContractError(ValueError):
    """Raised when a ROS-bound sample violates the frozen Franka contract."""


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise Ros2ContractError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if result < minimum:
        raise Ros2ContractError(f"{name} must be >= {minimum}, got {result}")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    return _integer(value, name=name, minimum=1)


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Ros2ContractError(f"{name} must be a non-empty string")
    return value.strip()


def _float_array(value: ArrayLike, *, shape: tuple[int, ...], name: str) -> FloatArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise Ros2ContractError(f"{name} is not a numeric array") from error
    if raw.dtype.kind not in "fiu":
        raise Ros2ContractError(f"{name} must contain real numbers, got dtype {raw.dtype}")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape != shape:
        raise Ros2ContractError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise Ros2ContractError(f"{name} contains NaN or Inf")
    result = np.ascontiguousarray(result).copy()
    result.setflags(write=False)
    return result


def _rgb_image(
    value: ArrayLike,
    *,
    name: str,
    expected_shape: tuple[int, int, int] | None = None,
) -> RgbImage:
    try:
        image = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise Ros2ContractError(f"{name} is not an image array") from error
    if image.dtype != np.uint8:
        raise Ros2ContractError(f"{name} must have dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise Ros2ContractError(f"{name} must have shape [H,W,3], got {image.shape}")
    if expected_shape is not None and image.shape != expected_shape:
        raise Ros2ContractError(f"{name} must have shape {expected_shape}, got {image.shape}")
    result = np.ascontiguousarray(image).copy()
    result.setflags(write=False)
    return result


def _unit_quaternion(value: ArrayLike, *, name: str = "quaternion_xyzw") -> FloatArray:
    quaternion = _float_array(value, shape=(4,), name=name)
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_QUATERNION_NORM_TOLERANCE):
        raise Ros2ContractError(
            f"{name} must be unit length; norm={norm:.9g}, "
            f"tolerance={_QUATERNION_NORM_TOLERANCE:.9g}"
        )
    normalized = np.ascontiguousarray(quaternion / norm)
    normalized.setflags(write=False)
    return normalized


def _quaternion_xyzw_to_rotation_6d(quaternion: ArrayLike) -> FloatArray:
    """Convert one unit xyzw quaternion to the matrix-first-two-columns representation."""

    x, y, z, w = _unit_quaternion(quaternion)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rotation_6d = np.asarray(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy + wz),
            2.0 * (xz - wy),
            2.0 * (xy - wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz + wx),
        ],
        dtype=np.float64,
    )
    rotation_6d.setflags(write=False)
    return rotation_6d


def decode_ros_image(
    *,
    height: int,
    width: int,
    encoding: str,
    step: int,
    data: bytes | bytearray | memoryview | NDArray[np.uint8],
) -> RgbImage:
    """Decode an ``rgb8`` or ``bgr8`` ROS image payload into copied HWC RGB.

    Row padding is accepted through ``step`` and removed.  Compressed image
    formats are intentionally outside this wire contract.
    """

    height = _positive_integer(height, name="height")
    width = _positive_integer(width, name="width")
    step = _positive_integer(step, name="step")
    if not isinstance(encoding, str):
        raise Ros2ContractError("encoding must be a string")
    normalized_encoding = encoding.lower()
    if normalized_encoding not in {"rgb8", "bgr8"}:
        raise Ros2ContractError(f"unsupported image encoding {encoding!r}; expected 'rgb8' or 'bgr8'")

    packed_row_bytes = width * 3
    if step < packed_row_bytes:
        raise Ros2ContractError(
            f"step must be at least width*3 ({packed_row_bytes}), got {step}"
        )
    if isinstance(data, np.ndarray):
        if data.dtype != np.uint8:
            raise Ros2ContractError(f"image data array must have dtype uint8, got {data.dtype}")
        flat = np.ascontiguousarray(data).reshape(-1)
    else:
        try:
            flat = np.frombuffer(data, dtype=np.uint8)
        except (TypeError, ValueError) as error:
            raise Ros2ContractError("image data must expose a byte buffer") from error
    expected_bytes = height * step
    if flat.size != expected_bytes:
        raise Ros2ContractError(
            f"image data length must equal height*step ({expected_bytes}), got {flat.size}"
        )

    rows = flat.reshape(height, step)[:, :packed_row_bytes]
    image = rows.reshape(height, width, 3)
    if normalized_encoding == "bgr8":
        image = image[..., ::-1]
    result = np.ascontiguousarray(image).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ImageSample:
    """One decoded RGB image and its source/arrival timestamps."""

    stamp_ns: int
    received_monotonic_ns: int
    image: RgbImage
    frame_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", _positive_integer(self.stamp_ns, name="stamp_ns"))
        object.__setattr__(
            self,
            "received_monotonic_ns",
            _positive_integer(self.received_monotonic_ns, name="received_monotonic_ns"),
        )
        object.__setattr__(self, "image", _rgb_image(self.image, name="image"))
        if not isinstance(self.frame_id, str):
            raise Ros2ContractError("frame_id must be a string")

    @classmethod
    def from_encoded(
        cls,
        *,
        stamp_ns: int,
        received_monotonic_ns: int,
        height: int,
        width: int,
        encoding: str,
        step: int,
        data: bytes | bytearray | memoryview | NDArray[np.uint8],
        frame_id: str = "",
    ) -> ImageSample:
        return cls(
            stamp_ns=stamp_ns,
            received_monotonic_ns=received_monotonic_ns,
            image=decode_ros_image(
                height=height,
                width=width,
                encoding=encoding,
                step=step,
                data=data,
            ),
            frame_id=frame_id,
        )


@dataclass(frozen=True, slots=True)
class PoseSample:
    """Measured EEF pose in the configured robot base frame."""

    stamp_ns: int
    received_monotonic_ns: int
    position: FloatArray
    quaternion_xyzw: FloatArray
    frame_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", _positive_integer(self.stamp_ns, name="stamp_ns"))
        object.__setattr__(
            self,
            "received_monotonic_ns",
            _positive_integer(self.received_monotonic_ns, name="received_monotonic_ns"),
        )
        object.__setattr__(self, "position", _float_array(self.position, shape=(3,), name="position"))
        object.__setattr__(
            self,
            "quaternion_xyzw",
            _unit_quaternion(self.quaternion_xyzw),
        )
        object.__setattr__(self, "frame_id", _nonempty_string(self.frame_id, name="frame_id"))


@dataclass(frozen=True, slots=True)
class JointStateSample:
    """One named Franka qpos sample; canonical ordering is applied by the cache."""

    stamp_ns: int
    received_monotonic_ns: int
    names: tuple[str, ...]
    positions: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", _positive_integer(self.stamp_ns, name="stamp_ns"))
        object.__setattr__(
            self,
            "received_monotonic_ns",
            _positive_integer(self.received_monotonic_ns, name="received_monotonic_ns"),
        )
        names = tuple(self.names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise Ros2ContractError("joint names must be non-empty strings")
        if len(set(names)) != len(names):
            raise Ros2ContractError("joint names contain duplicates")
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self,
            "positions",
            _float_array(self.positions, shape=(len(names),), name="joint positions"),
        )


@dataclass(frozen=True, slots=True)
class _GripperSample:
    stamp_ns: int
    received_monotonic_ns: int
    closed_0_1: float


class RosObservationSnapshot:
    """Immutable synchronized observation with qpos retained as sideband data.

    Array-returning properties and projection methods return copies.  This is
    intentional: image preparation downstream may mutate its inputs.
    """

    __slots__ = (
        "anchor_stamp_ns",
        "created_monotonic_ns",
        "frame_id",
        "gripper_closed_0_1",
        "_camera1",
        "_camera2",
        "_eef_position",
        "_eef_quaternion_xyzw",
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
        eef_position: ArrayLike,
        eef_quaternion_xyzw: ArrayLike,
        qpos: ArrayLike,
        gripper_closed_0_1: float,
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
        self._eef_position = _float_array(eef_position, shape=(3,), name="eef_position")
        self._eef_quaternion_xyzw = _unit_quaternion(eef_quaternion_xyzw, name="eef_quaternion_xyzw")
        self._qpos = _float_array(qpos, shape=(_QPOS_DIM,), name="qpos")
        if isinstance(gripper_closed_0_1, bool) or not isinstance(gripper_closed_0_1, Real):
            raise Ros2ContractError("gripper_closed_0_1 must be a real scalar")
        gripper = float(gripper_closed_0_1)
        if not math.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
            raise Ros2ContractError(f"gripper_closed_0_1 must be finite and in [0, 1], got {gripper}")
        self.gripper_closed_0_1 = gripper

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
    def eef_position(self) -> FloatArray:
        return self._eef_position.copy()

    @property
    def eef_quaternion_xyzw(self) -> FloatArray:
        return self._eef_quaternion_xyzw.copy()

    @property
    def eef(self) -> FloatArray:
        return np.concatenate((self._eef_position, self._eef_quaternion_xyzw))

    @property
    def qpos(self) -> FloatArray:
        return self._qpos.copy()

    @property
    def source_stamps_ns(self) -> dict[str, int]:
        return self._source_stamps_ns.copy()

    @property
    def state(self) -> FloatArray:
        rotation_6d = _quaternion_xyzw_to_rotation_6d(self._eef_quaternion_xyzw)
        return np.concatenate(
            (self._eef_position, rotation_6d, np.asarray([self.gripper_closed_0_1]))
        )

    def as_policy_observation(self) -> dict[str, float | RgbImage]:
        state = self.state
        observation: dict[str, float | RgbImage] = {
            name: float(value) for name, value in zip(STATE_NAMES, state, strict=True)
        }
        observation[CAMERA_NAMES[0]] = self._camera1.copy()
        observation[CAMERA_NAMES[1]] = self._camera2.copy()
        return observation

    def as_robot_observation(self) -> dict[str, float | RgbImage]:
        """Alias matching LeRobot's ``Robot.get_observation`` vocabulary."""

        return self.as_policy_observation()

    def as_sideband(self) -> dict[str, Any]:
        """Return monitoring data excluded from the canonical Franka feature contract."""

        return {
            "anchor_stamp_ns": self.anchor_stamp_ns,
            "created_monotonic_ns": self.created_monotonic_ns,
            "frame_id": self.frame_id,
            "source_stamps_ns": self._source_stamps_ns.copy(),
            "eef": self.eef,
            "qpos": self._qpos.copy(),
            "gripper.closed_0_1": self.gripper_closed_0_1,
        }

    def copy(self) -> RosObservationSnapshot:
        return RosObservationSnapshot(
            anchor_stamp_ns=self.anchor_stamp_ns,
            created_monotonic_ns=self.created_monotonic_ns,
            camera1=self._camera1,
            camera2=self._camera2,
            eef_position=self._eef_position,
            eef_quaternion_xyzw=self._eef_quaternion_xyzw,
            qpos=self._qpos,
            gripper_closed_0_1=self.gripper_closed_0_1,
            frame_id=self.frame_id,
            source_stamps_ns=self._source_stamps_ns,
        )


class RosObservationCache:
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

    def update_camera1(self, sample: ImageSample) -> None:
        self._validate_image(sample, name="camera1")
        self._insert("camera1", sample)

    def update_camera2(self, sample: ImageSample) -> None:
        self._validate_image(sample, name="camera2")
        self._insert("camera2", sample)

    def _validate_image(self, sample: ImageSample, *, name: str) -> None:
        if not isinstance(sample, ImageSample):
            raise Ros2ContractError(f"{name} must be an ImageSample")
        if sample.image.shape != CAMERA_SHAPE:
            raise Ros2ContractError(f"{name} must have shape {CAMERA_SHAPE}, got {sample.image.shape}")

    def update_eef(self, sample: PoseSample) -> None:
        if not isinstance(sample, PoseSample):
            raise Ros2ContractError("eef must be a PoseSample")
        if sample.frame_id != self.base_frame:
            raise Ros2ContractError(
                f"eef frame_id must be {self.base_frame!r}, got {sample.frame_id!r}"
            )
        self._insert("eef", sample)

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
        closed_0_1: float,
        *,
        stamp_ns: int,
        received_monotonic_ns: int,
    ) -> None:
        if isinstance(closed_0_1, bool) or not isinstance(closed_0_1, Real):
            raise Ros2ContractError("gripper closed_0_1 must be a real scalar")
        value = float(closed_0_1)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise Ros2ContractError(f"gripper closed_0_1 must be finite and in [0, 1], got {value}")
        sample = _GripperSample(
            stamp_ns=_positive_integer(stamp_ns, name="stamp_ns"),
            received_monotonic_ns=_positive_integer(
                received_monotonic_ns,
                name="received_monotonic_ns",
            ),
            closed_0_1=value,
        )
        self._insert("gripper", sample)

    def _now_ns(self) -> int:
        return _positive_integer(self._monotonic_ns(), name="monotonic clock")

    def _require_fresh(self, source: str, sample: Any, *, now_ns: int) -> None:
        age_ns = now_ns - sample.received_monotonic_ns
        if age_ns < 0:
            raise Ros2ContractError(
                f"{source} arrival timestamp is in the future by {-age_ns} ns"
            )
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

    def get_snapshot(self) -> RosObservationSnapshot:
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
            eef = self._select("eef", anchor_stamp_ns=anchor.stamp_ns, now_ns=now_ns)
            qpos = self._select("qpos", anchor_stamp_ns=anchor.stamp_ns, now_ns=now_ns)
            gripper = self._select("gripper", anchor_stamp_ns=anchor.stamp_ns, now_ns=now_ns)
            source_stamps_ns = {
                "camera1": anchor.stamp_ns,
                "camera2": camera2.stamp_ns,
                "eef": eef.stamp_ns,
                "qpos": qpos.stamp_ns,
                "gripper": gripper.stamp_ns,
            }
            return RosObservationSnapshot(
                anchor_stamp_ns=anchor.stamp_ns,
                created_monotonic_ns=now_ns,
                camera1=anchor.image,
                camera2=camera2.image,
                eef_position=eef.position,
                eef_quaternion_xyzw=eef.quaternion_xyzw,
                qpos=qpos.positions,
                gripper_closed_0_1=gripper.closed_0_1,
                frame_id=eef.frame_id,
                source_stamps_ns=source_stamps_ns,
            )

    def clear(self) -> None:
        with self._lock:
            for buffer in self._buffers.values():
                buffer.clear()


class AbsoluteActionChunk:
    """Validated client-side representation of one absolute Cartesian plan."""

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
        canonical_actions = np.asarray(raw_actions, dtype=np.float32)
        if not np.isfinite(canonical_actions).all():
            raise Ros2ContractError("actions overflow float32")

        quaternions = canonical_actions[:, 3:7].astype(np.float64)
        norms = np.linalg.norm(quaternions, axis=1)
        invalid_quaternion = np.flatnonzero(
            ~np.isclose(norms, 1.0, rtol=0.0, atol=_QUATERNION_NORM_TOLERANCE)
        )
        if invalid_quaternion.size:
            index = int(invalid_quaternion[0])
            raise Ros2ContractError(
                f"action quaternion at row {index} must be unit length; norm={norms[index]:.9g}"
            )
        gripper = canonical_actions[:, 7]
        invalid_gripper = np.flatnonzero((gripper < 0.0) | (gripper > 1.0))
        if invalid_gripper.size:
            index = int(invalid_gripper[0])
            raise Ros2ContractError(
                f"action gripper at row {index} must be in [0, 1], got {gripper[index]:.9g}"
            )

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
    def actions(self) -> NDArray[np.float32]:
        return self._actions.copy()

    @property
    def timesteps(self) -> tuple[int, ...]:
        return self._timesteps

    @property
    def poses(self) -> NDArray[np.float32]:
        """Copied ``[K,7]`` xyz+xyzw rows matching the ROS message field."""

        return self._actions[:, :7].copy()

    @property
    def gripper(self) -> NDArray[np.float32]:
        """Copied ``[K]`` normalized gripper values matching the ROS message field."""

        return self._actions[:, 7].copy()

    @property
    def client_observation_stamp_ns(self) -> int:
        """ROS-message name for the source observation timestamp."""

        return self.source_observation_timestamp_ns

    @property
    def source_observation_stamp_ns(self) -> int:
        """Compatibility spelling used by the ROS message encoder."""

        return self.source_observation_timestamp_ns

    @property
    def server_send_stamp_ns(self) -> int:
        """ROS-message spelling for ``server_send_timestamp_ns``."""

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
        """Return a copy-safe representation for a later ROS message encoder."""

        return {
            "schema_version": self.schema_version,
            "actions": self._actions.copy(),
            "poses": self._actions[:, :7].copy(),
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
    "AbsoluteActionChunk",
    "ImageSample",
    "JointStateSample",
    "PoseSample",
    "Ros2ContractError",
    "RosObservationCache",
    "RosObservationSnapshot",
    "decode_ros_image",
]
