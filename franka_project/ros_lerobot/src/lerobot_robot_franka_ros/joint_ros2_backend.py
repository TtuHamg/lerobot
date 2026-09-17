"""Non-actuating ROS2 backend for the Franka joint-space (FastWAM) plugin."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from lerobot.types import RobotAction, RobotObservation

from .eef_ik import (
    EefIkError,
    PinocchioEefIk,
    closed_fraction_to_physical,
    decode_eef_actions,
)
from .joint_config_franka_ros import FrankaJointRosConfig
from .joint_ros2_contract import JointActionChunk, JointObservationCache, JointObservationSnapshot
from .ros2_contract import RosObservationCache, RosObservationSnapshot

_NS_PER_SECOND = 1_000_000_000


class JointRos2BackendError(RuntimeError):
    """Raised when the isolated ROS2 interface cannot preserve its contract."""


class _Ros2Runtime(Protocol):
    @property
    def is_running(self) -> bool: ...

    def start(self) -> None: ...

    def publish_action_chunk(self, chunk: JointActionChunk) -> None: ...

    def ready_for_next_observation(self) -> bool: ...

    def close(self, *, timeout_s: float | None = None) -> None: ...


class _EefIk(Protocol):
    def solve_actions(
        self,
        actions: Any,
        *,
        measured_q: Any,
        measured_eef_position: Any,
        measured_eef_quaternion_xyzw: Any,
    ) -> np.ndarray: ...


ObservationCache = JointObservationCache | RosObservationCache
RuntimeFactory = Callable[[FrankaJointRosConfig, ObservationCache], _Ros2Runtime]
EefIkFactory = Callable[[FrankaJointRosConfig], _EefIk]


def _default_runtime_factory(
    config: FrankaJointRosConfig,
    cache: ObservationCache,
) -> _Ros2Runtime:
    # This import is the only path from normal plugin code to rclpy/ROS
    # messages. Plugin discovery and dry-run mode therefore stay ROS-free.
    from .joint_ros2_runtime import JointRos2Runtime

    return JointRos2Runtime(config, cache)


def _default_eef_ik_factory(config: FrankaJointRosConfig) -> _EefIk:
    return PinocchioEefIk(config)


def _seconds_to_ns(value: float, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JointRos2BackendError(f"{name} must be a real timestamp")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise JointRos2BackendError(f"{name} must be finite and greater than zero")
    result = round(seconds * _NS_PER_SECOND)
    if result <= 0:
        raise JointRos2BackendError(f"{name} is too small to represent in nanoseconds")
    return result


class JointRos2Backend:
    """Bridge measured ROS joint state and complete joint chunks without actuation.

    ``send_action`` intentionally publishes nothing. The stock RobotClient calls
    it once per local queue step to advance bookkeeping, while complete plans are
    published exactly once through :meth:`publish_action_chunk`.
    """

    def __init__(
        self,
        *,
        config: FrankaJointRosConfig,
        runtime_factory: RuntimeFactory | None = None,
        eef_ik_factory: EefIkFactory | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._eef_ik_factory = eef_ik_factory or _default_eef_ik_factory
        self._monotonic_ns = monotonic_ns
        if config.action_space == "eef":
            self._cache: ObservationCache = RosObservationCache(
                joint_names=config.arm_joint_names,
                base_frame=config.base_frame,
                max_skew_ns={
                    "camera2": round(config.camera2_max_skew_s * _NS_PER_SECOND),
                    "eef": round(config.eef_max_skew_s * _NS_PER_SECOND),
                    "qpos": round(config.qpos_max_skew_s * _NS_PER_SECOND),
                    "gripper": round(config.gripper_max_skew_s * _NS_PER_SECOND),
                },
                max_age_ns=round(config.max_observation_age_s * _NS_PER_SECOND),
                buffer_size=config.observation_buffer_size,
                monotonic_ns=monotonic_ns,
            )
        else:
            self._cache = JointObservationCache(
                joint_names=config.arm_joint_names,
                base_frame=config.base_frame,
                max_skew_ns={
                    "camera2": round(config.camera2_max_skew_s * _NS_PER_SECOND),
                    "qpos": round(config.qpos_max_skew_s * _NS_PER_SECOND),
                    "gripper": round(config.gripper_max_skew_s * _NS_PER_SECOND),
                },
                max_age_ns=round(config.max_observation_age_s * _NS_PER_SECOND),
                buffer_size=config.observation_buffer_size,
                monotonic_ns=monotonic_ns,
            )
        self._runtime: _Ros2Runtime | None = None
        self._eef_ik: _EefIk | None = None
        self._session_id: str | None = None
        self._next_plan_id = 0
        self._last_bookkept_action: RobotAction | None = None
        self._lock = threading.RLock()

    @property
    def is_connected(self) -> bool:
        runtime = self._runtime
        return runtime is not None and runtime.is_running

    def _require_runtime(self) -> _Ros2Runtime:
        runtime = self._runtime
        if runtime is None or not runtime.is_running:
            raise JointRos2BackendError("ROS2 interface backend is not connected")
        return runtime

    def connect(self) -> None:
        with self._lock:
            if self._runtime is not None:
                raise JointRos2BackendError("ROS2 interface backend is already connected")
            eef_ik = self._eef_ik_factory(self.config) if self.config.action_space == "eef" else None
            runtime = self._runtime_factory(self.config, self._cache)
            try:
                runtime.start()
                if not runtime.is_running:
                    raise JointRos2BackendError("ROS2 executor did not enter its running state")
            except Exception:
                try:
                    runtime.close(timeout_s=self.config.ros2_shutdown_timeout_s)
                except Exception:
                    pass
                raise
            self._runtime = runtime
            self._eef_ik = eef_ik
            self._session_id = uuid.uuid4().hex
            self._next_plan_id = 0
            self._last_bookkept_action = None

    def get_snapshot(self) -> JointObservationSnapshot | RosObservationSnapshot:
        self._require_runtime()
        if isinstance(self._cache, RosObservationCache):
            return self._cache.get_snapshot()
        return self._cache.get_joint_snapshot()

    def get_observation(self) -> RobotObservation:
        return self.get_snapshot().as_policy_observation()

    def get_qpos(self) -> np.ndarray:
        """Return the canonical seven-joint sideband."""

        return self.get_snapshot().qpos

    def get_sideband(self) -> dict[str, Any]:
        return self.get_snapshot().as_sideband()

    def ready_for_next_observation(self) -> bool:
        return self._require_runtime().ready_for_next_observation()

    def send_action(self, action: RobotAction) -> RobotAction:
        """Record local queue progress without creating a second ROS command."""

        self._require_runtime()
        copied = {name: float(value) for name, value in action.items()}
        with self._lock:
            self._last_bookkept_action = copied
        return copied

    def publish_action_chunk(
        self,
        timed_actions: Sequence[Any],
        *,
        source_observation_timestep: int,
        source_observation_timestamp: float,
        period_s: float,
    ) -> JointActionChunk:
        from lerobot.async_inference.helpers import TimedAction

        runtime = self._require_runtime()
        if not timed_actions:
            raise JointRos2BackendError("Cannot publish an empty action chunk")
        if not all(isinstance(action, TimedAction) for action in timed_actions):
            raise JointRos2BackendError("Action chunk must contain TimedAction values")

        timesteps = tuple(int(action.get_timestep()) for action in timed_actions)
        if timesteps[0] < int(source_observation_timestep):
            raise JointRos2BackendError(
                "source_observation_timestep must not be later than the first published action timestep"
            )
        if any(current != previous + 1 for previous, current in zip(timesteps, timesteps[1:])):
            raise JointRos2BackendError("Published action timesteps must be contiguous")

        period = float(period_s)
        if not math.isfinite(period) or period <= 0.0:
            raise JointRos2BackendError("period_s must be finite and greater than zero")
        source_timestamp = float(source_observation_timestamp)
        if not math.isfinite(source_timestamp) or source_timestamp <= 0.0:
            raise JointRos2BackendError("source_observation_timestamp must be finite and positive")
        for timed_action in timed_actions:
            expected = source_timestamp + (
                timed_action.get_timestep() - source_observation_timestep
            ) * period
            if not math.isclose(
                float(timed_action.get_timestamp()),
                expected,
                rel_tol=0.0,
                abs_tol=max(1e-6, period * 1e-4),
            ):
                raise JointRos2BackendError(
                    "TimedAction timestamps do not match timestep/period metadata"
                )

        action_rows: list[np.ndarray] = []
        server_send_timestamps: list[float] = []
        for timed_action in timed_actions:
            value = timed_action.get_action()
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            action_rows.append(np.asarray(value))
            server_send = timed_action.server_send_timestamp
            if server_send is None:
                raise JointRos2BackendError("TimedAction is missing server_send_timestamp")
            server_send_timestamps.append(float(server_send))
        if not all(
            math.isclose(value, server_send_timestamps[0], rel_tol=0.0, abs_tol=1e-9)
            for value in server_send_timestamps
        ):
            raise JointRos2BackendError("TimedAction server_send_timestamp values differ within one chunk")

        # Start the local transport TTL before potentially expensive IK. The
        # runtime will reject the converted plan if IK consumed its validity.
        received_monotonic_ns = int(self._monotonic_ns())
        if received_monotonic_ns <= 0:
            raise JointRos2BackendError("monotonic clock returned a non-positive value")

        stacked_actions = np.stack(action_rows)
        if self.config.action_space == "eef":
            stacked_actions = self._eef_actions_to_joint_actions(stacked_actions)

        with self._lock:
            session_id = self._session_id
            if session_id is None:
                raise JointRos2BackendError("ROS2 session id is unavailable")
            plan_id = self._next_plan_id
            chunk = JointActionChunk(
                actions=stacked_actions,
                timesteps=timesteps,
                source_timestep=int(source_observation_timestep),
                source_observation_timestamp_ns=_seconds_to_ns(
                    source_timestamp,
                    name="source_observation_timestamp",
                ),
                server_send_timestamp_ns=_seconds_to_ns(
                    server_send_timestamps[0],
                    name="server_send_timestamp",
                ),
                received_monotonic_ns=received_monotonic_ns,
                period_ns=round(period * _NS_PER_SECOND),
                frame_id=self.config.base_frame,
                session_id=session_id,
                plan_id=plan_id,
                valid_for_ns=round(self.config.action_chunk_validity_s * _NS_PER_SECOND),
            )
            runtime.publish_action_chunk(chunk)
            self._next_plan_id += 1
        return chunk

    def _eef_actions_to_joint_actions(self, actions: np.ndarray) -> np.ndarray:
        """Decode absolute EEF rows and solve them into gateway-ready joint rows."""

        eef_ik = self._eef_ik
        if eef_ik is None:
            raise JointRos2BackendError("EEF IK is unavailable while action_space='eef'")
        try:
            _, _, closed_fractions = decode_eef_actions(actions)
            snapshot = self.get_snapshot()
            if not isinstance(snapshot, RosObservationSnapshot):
                raise JointRos2BackendError("EEF action_space requires a Cartesian observation snapshot")
            positions = np.asarray(
                eef_ik.solve_actions(
                    actions,
                    measured_q=snapshot.qpos,
                    measured_eef_position=snapshot.eef_position,
                    measured_eef_quaternion_xyzw=snapshot.eef_quaternion_xyzw,
                ),
                dtype=np.float64,
            )
            expected_shape = (len(actions), len(self.config.arm_joint_names))
            if positions.shape != expected_shape or not np.isfinite(positions).all():
                raise EefIkError(f"IK result must be finite shape {expected_shape}, got {positions.shape}")
            gripper, _, _, _ = closed_fraction_to_physical(
                closed_fractions,
                minimum=float(self.config.gripper_command_min_position),
                maximum=float(self.config.gripper_command_max_position),
            )
        except JointRos2BackendError:
            raise
        except (EefIkError, ValueError, TypeError) as error:
            raise JointRos2BackendError(f"Could not convert EEF action chunk: {error}") from error
        return np.column_stack((positions, gripper))

    def disconnect(self) -> None:
        with self._lock:
            runtime = self._runtime
            self._runtime = None
            self._eef_ik = None
            self._session_id = None
            self._last_bookkept_action = None
        try:
            if runtime is not None:
                runtime.close(timeout_s=self.config.ros2_shutdown_timeout_s)
        finally:
            self._cache.clear()


__all__ = ["JointRos2Backend", "JointRos2BackendError", "RuntimeFactory"]
