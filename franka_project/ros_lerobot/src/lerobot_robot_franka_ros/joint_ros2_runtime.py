"""The optional ROS 2 boundary for the Franka joint-space (FastWAM) plugin.

Importing this module is side-effect free.  In particular, the ROS Python
packages and the generated project message are imported only by
``JointRos2Runtime.start()``.  This keeps plugin discovery and dry-run tests
usable in a plain LeRobot environment where ROS has not been sourced.

The runtime deliberately stops at an isolated ROS interface:

* callbacks copy measured ROS state into a transport-neutral observation cache;
* complete absolute joint-space chunks are published under ``/lerobot``;
* no controller, action client, IK solver, or hardware command is created here.

Unlike the Cartesian runtime, this variant subscribes to the raw joint state
(seven arm joints plus one raw gripper joint position) and does not consume the
end-effector pose.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .joint_ros2_contract import JointActionChunk, JointObservationCache
from .ros2_contract import ImageSample, JointStateSample

if TYPE_CHECKING:
    from .joint_config_franka_ros import FrankaJointRosConfig


_NANOSECONDS_PER_SECOND = 1_000_000_000
_SCHEMA_VERSION = 1


class JointRos2RuntimeUnavailableError(RuntimeError):
    """Raised when the optional ROS 2 runtime has not been built or sourced."""


class JointRos2RuntimeStateError(RuntimeError):
    """Raised when an operation is incompatible with the runtime lifecycle."""


@dataclass(frozen=True, slots=True)
class _RosBindings:
    rclpy: Any
    context_type: type
    node_type: type
    executor_type: type
    qos_profile_type: type
    history_policy: Any
    reliability_policy: Any
    durability_policy: Any
    image_type: type
    joint_state_type: type
    action_chunk_type: type


def _load_ros_bindings() -> _RosBindings:
    """Load ROS only for an explicitly started live interface."""

    try:
        import rclpy
        from lerobot_franka_interfaces.msg import JointActionChunk as JointActionChunkMsg
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import Image, JointState
    except (ImportError, ModuleNotFoundError) as error:
        raise JointRos2RuntimeUnavailableError(
            "ROS 2 Python bindings or lerobot_franka_interfaces are unavailable. "
            "Source /opt/ros/jazzy/setup.bash, build franka_project/ros2_ws with colcon, "
            "and source its install/setup.bash before enabling the ROS2 interface."
        ) from error

    return _RosBindings(
        rclpy=rclpy,
        context_type=Context,
        node_type=Node,
        executor_type=SingleThreadedExecutor,
        qos_profile_type=QoSProfile,
        history_policy=HistoryPolicy,
        reliability_policy=ReliabilityPolicy,
        durability_policy=DurabilityPolicy,
        image_type=Image,
        joint_state_type=JointState,
        action_chunk_type=JointActionChunkMsg,
    )


def _message_stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if seconds < 0 or not 0 <= nanoseconds < _NANOSECONDS_PER_SECOND:
        raise ValueError(f"invalid ROS timestamp: sec={seconds}, nanosec={nanoseconds}")
    return seconds * _NANOSECONDS_PER_SECOND + nanoseconds


def _set_time_message(message: Any, nanoseconds: int) -> None:
    if nanoseconds < 0:
        raise ValueError(f"ROS timestamp must be non-negative, got {nanoseconds}")
    seconds, remainder = divmod(int(nanoseconds), _NANOSECONDS_PER_SECOND)
    if seconds > 2**31 - 1:
        raise ValueError(f"ROS timestamp seconds exceed int32 range: {seconds}")
    message.sec = seconds
    message.nanosec = remainder


def _set_duration_message(message: Any, nanoseconds: int) -> None:
    if nanoseconds <= 0:
        raise ValueError(f"action period must be positive, got {nanoseconds} ns")
    seconds, remainder = divmod(int(nanoseconds), _NANOSECONDS_PER_SECOND)
    if seconds > 2**31 - 1:
        raise ValueError(f"ROS duration seconds exceed int32 range: {seconds}")
    message.sec = seconds
    message.nanosec = remainder


class JointRos2Runtime:
    """Own one private rclpy context and a bounded executor thread."""

    def __init__(self, config: FrankaJointRosConfig, cache: JointObservationCache):
        self.config = config
        self.cache = cache
        self._lifecycle_lock = threading.RLock()
        self._context: Any | None = None
        self._node: Any | None = None
        self._executor: Any | None = None
        self._thread: threading.Thread | None = None
        self._publisher: Any | None = None
        self._subscriptions: list[Any] = []
        self._bindings: _RosBindings | None = None
        self._closing = False
        self._spin_error: BaseException | None = None
        self._last_callback_error: dict[str, float] = {}

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            thread = self._thread
            context = self._context
            return bool(
                not self._closing
                and thread is not None
                and thread.is_alive()
                and context is not None
                and context.ok()
            )

    @property
    def spin_error(self) -> BaseException | None:
        return self._spin_error

    def start(self, *, ros_args: Sequence[str] | None = None) -> None:
        """Create the ROS node and start its executor; never called by import."""

        with self._lifecycle_lock:
            if self._context is not None or self._thread is not None:
                raise JointRos2RuntimeStateError("ROS2 runtime is already started")

            bindings = _load_ros_bindings()
            context = bindings.context_type()
            node = None
            executor = None
            try:
                bindings.rclpy.init(
                    args=list(ros_args) if ros_args is not None else None,
                    context=context,
                )
                node = bindings.node_type(
                    self.config.ros2_node_name,
                    context=context,
                    enable_rosout=True,
                )
                executor = bindings.executor_type(context=context)

                sensor_qos = bindings.qos_profile_type(
                    history=bindings.history_policy.KEEP_LAST,
                    depth=5,
                    reliability=bindings.reliability_policy.BEST_EFFORT,
                    durability=bindings.durability_policy.VOLATILE,
                )
                action_qos = bindings.qos_profile_type(
                    history=bindings.history_policy.KEEP_LAST,
                    depth=1,
                    reliability=bindings.reliability_policy.RELIABLE,
                    durability=bindings.durability_policy.VOLATILE,
                )

                subscriptions = [
                    node.create_subscription(
                        bindings.image_type,
                        self.config.camera1_topic,
                        lambda message: self._camera_callback("camera1", message),
                        sensor_qos,
                    ),
                    node.create_subscription(
                        bindings.image_type,
                        self.config.camera2_topic,
                        lambda message: self._camera_callback("camera2", message),
                        sensor_qos,
                    ),
                    node.create_subscription(
                        bindings.joint_state_type,
                        self.config.qpos_topic,
                        self._arm_joint_state_callback,
                        sensor_qos,
                    ),
                    node.create_subscription(
                        bindings.joint_state_type,
                        self.config.gripper_topic,
                        self._gripper_joint_state_callback,
                        sensor_qos,
                    ),
                ]
                publisher = node.create_publisher(
                    bindings.action_chunk_type,
                    self.config.action_chunk_topic,
                    action_qos,
                )
                executor.add_node(node)

                self._bindings = bindings
                self._context = context
                self._node = node
                self._executor = executor
                self._publisher = publisher
                self._subscriptions = subscriptions
                self._closing = False
                self._spin_error = None
                thread = threading.Thread(
                    target=self._spin,
                    name=f"{self.config.ros2_node_name}-executor",
                    daemon=True,
                )
                self._thread = thread
                thread.start()
            except BaseException:
                if executor is not None:
                    executor.shutdown(timeout_sec=0.0)
                if node is not None:
                    node.destroy_node()
                context.try_shutdown()
                self._clear_lifecycle_state()
                raise

    connect = start

    def _spin(self) -> None:
        executor = self._executor
        if executor is None:
            return
        try:
            executor.spin()
        except BaseException as error:  # Preserve the failure for the owning backend.
            self._spin_error = error

    def _log_callback_error(self, source: str, error: Exception) -> None:
        # A malformed high-rate stream must not flood logs or kill the executor.
        now = time.monotonic()
        previous = self._last_callback_error.get(source, -math.inf)
        if now - previous < 5.0:
            return
        self._last_callback_error[source] = now
        node = self._node
        if node is not None:
            node.get_logger().error(f"Rejected {source} message: {error}")

    def _camera_callback(self, name: str, message: Any) -> None:
        try:
            sample = ImageSample.from_encoded(
                stamp_ns=_message_stamp_ns(message),
                received_monotonic_ns=time.monotonic_ns(),
                height=int(message.height),
                width=int(message.width),
                encoding=str(message.encoding),
                step=int(message.step),
                data=memoryview(message.data),
                frame_id=str(message.header.frame_id),
            )
            if name == "camera1":
                self.cache.update_camera1(sample)
            elif name == "camera2":
                self.cache.update_camera2(sample)
            else:
                raise ValueError(f"unknown camera source {name!r}")
        except Exception as error:  # A bad sample is fail-closed, not an executor failure.
            self._log_callback_error(name, error)

    @staticmethod
    def _joint_state_sample(message: Any) -> JointStateSample:
        return JointStateSample(
            stamp_ns=_message_stamp_ns(message),
            received_monotonic_ns=time.monotonic_ns(),
            names=tuple(str(name) for name in message.name),
            positions=tuple(float(value) for value in message.position),
        )

    def _arm_joint_state_callback(self, message: Any) -> None:
        try:
            self.cache.update_qpos(self._joint_state_sample(message))
        except Exception as error:
            self._log_callback_error("arm joint state", error)

    def _gripper_joint_state_callback(self, message: Any) -> None:
        try:
            names = tuple(str(name) for name in message.name)
            positions = tuple(float(value) for value in message.position)
            if len(names) != len(positions):
                raise ValueError(
                    f"gripper name/position length mismatch: {len(names)} != {len(positions)}"
                )
            try:
                joint_index = names.index(self.config.gripper_joint_name)
            except ValueError as error:
                raise ValueError(
                    f"gripper JointState does not contain {self.config.gripper_joint_name!r}"
                ) from error

            joint_position = positions[joint_index]
            if not math.isfinite(joint_position):
                raise ValueError("gripper joint position is NaN or Inf")
            # FastWAM consumes the raw gripper joint position; unlike the Cartesian
            # runtime, do not normalize it to a [0, 1] closed fraction.
            self.cache.update_gripper(
                joint_position,
                stamp_ns=_message_stamp_ns(message),
                received_monotonic_ns=time.monotonic_ns(),
            )
        except Exception as error:
            self._log_callback_error("gripper joint state", error)

    def publish_action_chunk(self, chunk: JointActionChunk) -> None:
        """Validate and publish one complete absolute joint-space plan."""

        with self._lifecycle_lock:
            if not self.is_running or self._publisher is None or self._bindings is None:
                if self._spin_error is not None:
                    raise JointRos2RuntimeStateError("ROS2 executor failed") from self._spin_error
                raise JointRos2RuntimeStateError("ROS2 runtime is not running")
            message = self._action_chunk_message(chunk)
            self._publisher.publish(message)

    publish_chunk = publish_action_chunk

    def _action_chunk_message(self, chunk: JointActionChunk) -> Any:
        bindings = self._bindings
        node = self._node
        if bindings is None or node is None:
            raise JointRos2RuntimeStateError("ROS2 runtime is not initialized")

        if not isinstance(chunk, JointActionChunk):
            raise TypeError("chunk must be a JointActionChunk")
        now_monotonic_ns = time.monotonic_ns()
        chunk.require_valid_at(now_monotonic_ns)
        remaining_validity_ns = max(
            0,
            chunk.received_monotonic_ns + chunk.valid_for_ns - now_monotonic_ns,
        )

        now_ros_ns = int(node.get_clock().now().nanoseconds)

        message = bindings.action_chunk_type()
        _set_time_message(message.header.stamp, now_ros_ns)
        message.header.frame_id = chunk.frame_id
        if message.header.frame_id != self.config.base_frame:
            raise ValueError(
                f"action frame must be {self.config.base_frame!r}, got {message.header.frame_id!r}"
            )
        if chunk.schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported action chunk schema {chunk.schema_version}; expected {_SCHEMA_VERSION}"
            )
        message.schema_version = chunk.schema_version
        message.session_id = chunk.session_id
        message.plan_id = chunk.plan_id
        message.source_timestep = chunk.source_timestep
        _set_time_message(
            message.client_observation_stamp,
            chunk.source_observation_timestamp_ns,
        )
        _set_time_message(
            message.server_send_stamp,
            chunk.server_send_timestamp_ns,
        )
        _set_time_message(message.valid_until, now_ros_ns + remaining_validity_ns)
        _set_duration_message(message.period, chunk.period_ns)

        message.timesteps = list(chunk.timesteps)
        # positions is a flattened row-major [K*7] array of absolute joint angles.
        message.positions = [float(value) for value in chunk.positions.reshape(-1)]
        message.gripper = [float(value) for value in chunk.gripper]
        return message

    def close(self, *, timeout_s: float | None = None) -> None:
        """Stop callbacks and release the private context within a bounded time."""

        timeout = (
            float(self.config.ros2_shutdown_timeout_s) if timeout_s is None else float(timeout_s)
        )
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("ROS2 shutdown timeout must be finite and positive")

        with self._lifecycle_lock:
            if self._context is None and self._thread is None:
                return
            if self._closing:
                raise JointRos2RuntimeStateError("ROS2 runtime shutdown is already in progress")
            self._closing = True
            executor = self._executor
            context = self._context
            node = self._node
            thread = self._thread

        deadline = time.monotonic() + timeout
        try:
            if executor is not None:
                executor.shutdown(timeout_sec=max(0.0, deadline - time.monotonic()))
            if context is not None:
                context.try_shutdown()
            if thread is not None:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread is not None and thread.is_alive():
                raise JointRos2RuntimeStateError(
                    f"ROS2 executor did not stop within {timeout:.3f} seconds"
                )
            if executor is not None and node is not None:
                executor.remove_node(node)
            if node is not None:
                node.destroy_node()
        finally:
            with self._lifecycle_lock:
                self._clear_lifecycle_state()

    disconnect = close
    shutdown = close

    def _clear_lifecycle_state(self) -> None:
        self._publisher = None
        self._subscriptions = []
        self._executor = None
        self._node = None
        self._context = None
        self._thread = None
        self._bindings = None
        self._closing = False

    def __enter__(self) -> JointRos2Runtime:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "JointRos2Runtime",
    "JointRos2RuntimeStateError",
    "JointRos2RuntimeUnavailableError",
]
