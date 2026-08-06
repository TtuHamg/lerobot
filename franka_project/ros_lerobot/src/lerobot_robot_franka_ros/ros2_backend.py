"""Non-actuating ROS2 backend for the Franka LeRobot plugin."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from lerobot.types import RobotAction, RobotObservation

from .config_franka_ros import FrankaRosConfig
from .ros2_contract import AbsoluteActionChunk, RosObservationCache, RosObservationSnapshot


_NS_PER_SECOND = 1_000_000_000


class Ros2BackendError(RuntimeError):
    """Raised when the isolated ROS2 interface cannot preserve its contract."""


class _Ros2Runtime(Protocol):
    @property
    def is_running(self) -> bool: ...

    def start(self) -> None: ...

    def publish_action_chunk(self, chunk: AbsoluteActionChunk) -> None: ...

    def set_gateway_armed(self, armed: bool, *, timeout_s: float) -> tuple[bool, str]: ...

    def get_plan_execution_state(
        self, session_id: str, plan_id: int, *, status_freshness_ns: int
    ) -> Any: ...

    def close(self, *, timeout_s: float | None = None) -> None: ...


RuntimeFactory = Callable[[FrankaRosConfig, RosObservationCache], _Ros2Runtime]


def _default_runtime_factory(
    config: FrankaRosConfig,
    cache: RosObservationCache,
) -> _Ros2Runtime:
    # This import is the only path from normal plugin code to rclpy/ROS
    # messages. Plugin discovery and dry-run mode therefore stay ROS-free.
    from .ros2_runtime import Ros2Runtime

    return Ros2Runtime(config, cache)


def _seconds_to_ns(value: float, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Ros2BackendError(f"{name} must be a real timestamp")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise Ros2BackendError(f"{name} must be finite and greater than zero")
    result = round(seconds * _NS_PER_SECOND)
    if result <= 0:
        raise Ros2BackendError(f"{name} is too small to represent in nanoseconds")
    return result


class Ros2Backend:
    """Bridge measured ROS state and complete Cartesian chunks without actuation.

    ``send_action`` intentionally publishes nothing. The stock RobotClient calls
    it once per local queue step to advance bookkeeping, while complete plans are
    published exactly once through :meth:`publish_action_chunk`.
    """

    def __init__(
        self,
        *,
        config: FrankaRosConfig,
        runtime_factory: RuntimeFactory | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._monotonic_ns = monotonic_ns
        self._cache = RosObservationCache(
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
        self._runtime: _Ros2Runtime | None = None
        self._session_id: str | None = None
        self._next_plan_id = 0
        self._last_bookkept_action: RobotAction | None = None
        # (plan_id, waypoint_count, period_ns, published_monotonic_ns) of the
        # newest published plan; used to gate the next observation on the
        # gateway's real execution progress.
        self._last_published_plan: tuple[int, int, int, int] | None = None
        self._lock = threading.RLock()

    @property
    def is_connected(self) -> bool:
        runtime = self._runtime
        return runtime is not None and runtime.is_running

    def _require_runtime(self) -> _Ros2Runtime:
        runtime = self._runtime
        if runtime is None or not runtime.is_running:
            raise Ros2BackendError("ROS2 interface backend is not connected")
        return runtime

    def connect(self) -> None:
        with self._lock:
            if self._runtime is not None:
                raise Ros2BackendError("ROS2 interface backend is already connected")
            runtime = self._runtime_factory(self.config, self._cache)
            try:
                runtime.start()
                if not runtime.is_running:
                    raise Ros2BackendError("ROS2 executor did not enter its running state")
            except Exception:
                try:
                    runtime.close(timeout_s=self.config.ros2_shutdown_timeout_s)
                except Exception:
                    pass
                raise
            self._runtime = runtime
            self._session_id = uuid.uuid4().hex
            self._next_plan_id = 0
            self._last_bookkept_action = None

    def get_snapshot(self) -> RosObservationSnapshot:
        self._require_runtime()
        return self._cache.get_snapshot()

    def get_observation(self) -> RobotObservation:
        return self.get_snapshot().as_policy_observation()

    def get_qpos(self) -> np.ndarray:
        """Return the seven-joint sideband, excluded from canonical Franka state10."""

        return self.get_snapshot().qpos

    def get_sideband(self) -> dict[str, Any]:
        return self.get_snapshot().as_sideband()

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
    ) -> AbsoluteActionChunk:
        from lerobot.async_inference.helpers import TimedAction

        runtime = self._require_runtime()
        if not timed_actions:
            raise Ros2BackendError("Cannot publish an empty action chunk")
        if not all(isinstance(action, TimedAction) for action in timed_actions):
            raise Ros2BackendError("Action chunk must contain TimedAction values")

        timesteps = tuple(int(action.get_timestep()) for action in timed_actions)
        if timesteps[0] < int(source_observation_timestep):
            raise Ros2BackendError(
                "source_observation_timestep must not be later than the first published action timestep"
            )
        if any(current != previous + 1 for previous, current in zip(timesteps, timesteps[1:])):
            raise Ros2BackendError("Published action timesteps must be contiguous")

        period = float(period_s)
        if not math.isfinite(period) or period <= 0.0:
            raise Ros2BackendError("period_s must be finite and greater than zero")
        source_timestamp = float(source_observation_timestamp)
        if not math.isfinite(source_timestamp) or source_timestamp <= 0.0:
            raise Ros2BackendError("source_observation_timestamp must be finite and positive")
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
                raise Ros2BackendError("TimedAction timestamps do not match timestep/period metadata")

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
                raise Ros2BackendError("TimedAction is missing server_send_timestamp")
            server_send_timestamps.append(float(server_send))
        if not all(
            math.isclose(value, server_send_timestamps[0], rel_tol=0.0, abs_tol=1e-9)
            for value in server_send_timestamps
        ):
            raise Ros2BackendError("TimedAction server_send_timestamp values differ within one chunk")

        received_monotonic_ns = int(self._monotonic_ns())
        if received_monotonic_ns <= 0:
            raise Ros2BackendError("monotonic clock returned a non-positive value")
        with self._lock:
            session_id = self._session_id
            if session_id is None:
                raise Ros2BackendError("ROS2 session id is unavailable")
            plan_id = self._next_plan_id
            chunk = AbsoluteActionChunk(
                actions=np.stack(action_rows),
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
            self._last_published_plan = (
                plan_id,
                len(timesteps),
                chunk.period_ns,
                int(self._monotonic_ns()),
            )
        return chunk

    def plan_execution_complete(self) -> bool:
        """True when the newest published plan no longer gates a fresh observation.

        The stock client replaces plans on its own nominal clock. When the
        gateway retimes execution below real time, that clock runs ahead and
        every replacement starts with a catch-up jump. This check aligns the
        next observation with the gateway's actual progress instead.

        Fail-open on every abnormal path (no ack, stale status, disarmed
        gateway, missing runtime): the gateway rejects unsafe plans itself, so
        blocking the observation stream can only add deadlock risk.
        """

        with self._lock:
            last = self._last_published_plan
            session_id = self._session_id
        if last is None or session_id is None:
            return True
        runtime = self._runtime
        if runtime is None or not runtime.is_running:
            return True
        plan_id, waypoint_count, period_ns, published_ns = last

        elapsed_ns = int(self._monotonic_ns()) - published_ns
        state = runtime.get_plan_execution_state(
            session_id,
            plan_id,
            status_freshness_ns=_NS_PER_SECOND,
        )
        if state is None:
            # No ack yet: wait briefly for the gateway round trip, then fail open.
            return elapsed_ns > _NS_PER_SECOND
        if not state.accepted:
            return True
        if not state.status_fresh or not state.armed:
            return True
        if state.status_plan_matches:
            if not state.has_active_plan:
                return True
            if state.applied_waypoint_index >= waypoint_count - 1:
                return True
        elif elapsed_ns > waypoint_count * period_ns:
            # The status no longer references this plan although its nominal
            # duration has passed: it completed (or was superseded) between
            # 10 Hz status samples.
            return True
        # Absolute ceiling: nominal duration times the gateway's maximum
        # retiming scale (8), plus settle margin. Beyond this something is
        # wedged and the observation stream must not deadlock.
        return elapsed_ns > 8 * waypoint_count * period_ns + 2 * _NS_PER_SECOND

    def set_gateway_armed(self, armed: bool, *, timeout_s: float) -> tuple[bool, str]:
        runtime = self._require_runtime()
        success, message = runtime.set_gateway_armed(armed, timeout_s=timeout_s)
        if not success:
            raise Ros2BackendError(f"Gateway rejected armed={armed}: {message}")
        if not armed:
            with self._lock:
                self._last_published_plan = None
        return success, message

    def disconnect(self) -> None:
        with self._lock:
            runtime = self._runtime
            self._runtime = None
            self._session_id = None
            self._last_bookkept_action = None
        try:
            if runtime is not None:
                runtime.close(timeout_s=self.config.ros2_shutdown_timeout_s)
        finally:
            self._cache.clear()


__all__ = ["Ros2Backend", "Ros2BackendError", "RuntimeFactory"]
