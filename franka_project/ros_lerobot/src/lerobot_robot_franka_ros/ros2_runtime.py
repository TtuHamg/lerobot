"""The optional ROS 2 boundary for the Franka LeRobot plugin.

Importing this module is side-effect free.  In particular, the ROS Python
packages and the generated project message are imported only by
``Ros2Runtime.start()``.  This keeps plugin discovery and dry-run tests usable
in a plain LeRobot environment where ROS has not been sourced.

The runtime deliberately stops at an isolated ROS interface:

* callbacks copy measured ROS state into a transport-neutral observation cache;
* complete absolute Cartesian chunks are published under ``/lerobot``;
* no controller, action client, IK solver, or hardware command is created here.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .ros2_contract import (
    AbsoluteActionChunk,
    ImageSample,
    JointStateSample,
    PoseSample,
    RosObservationCache,
)

if TYPE_CHECKING:
    from .config_franka_ros import FrankaRosConfig


_NANOSECONDS_PER_SECOND = 1_000_000_000
_SCHEMA_VERSION = 1

# The gateway publishes these under fixed names (see gateway_node.cpp).
_GATEWAY_ACK_TOPIC = "/lerobot/franka/action_chunk_ack"
_GATEWAY_STATUS_TOPIC = "/lerobot/franka/safety_gateway_status"
_GATEWAY_ARM_SERVICE = "/franka_cartesian_safety_gateway/set_armed"
_MAX_TRACKED_ACKS = 8


class Ros2RuntimeUnavailableError(RuntimeError):
    """Raised when the optional ROS 2 runtime has not been built or sourced."""


class Ros2RuntimeStateError(RuntimeError):
    """Raised when an operation is incompatible with the runtime lifecycle."""


@dataclass(frozen=True, slots=True)
class _RosBindings:
    rclpy: Any
    context_type: type
    node_type: type
    executor_type: type
    parameter_client_type: type
    parameter_value_to_python: Any
    qos_profile_type: type
    history_policy: Any
    reliability_policy: Any
    durability_policy: Any
    image_type: type
    pose_stamped_type: type
    franka_robot_state_type: type
    joint_state_type: type
    pose_type: type
    action_chunk_type: type
    action_chunk_ack_type: type
    gateway_status_type: type
    set_bool_type: type | None = None


@dataclass(frozen=True, slots=True)
class GatewayPlanState:
    """Snapshot of the gateway's progress on one published plan."""

    accepted: bool
    waypoint_count: int
    status_fresh: bool
    # True only while the gateway status message refers to this exact plan.
    status_plan_matches: bool
    applied_waypoint_index: int
    has_active_plan: bool
    armed: bool
    detail: str


def _load_ros_bindings() -> _RosBindings:
    """Load ROS only for an explicitly started live interface."""

    try:
        import rclpy
        from franka_msgs.msg import FrankaRobotState
        from geometry_msgs.msg import Pose, PoseStamped
        from lerobot_franka_interfaces.msg import (
            CartesianActionChunk,
            CartesianActionChunkAck,
            SafetyGatewayStatus,
        )
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import parameter_value_to_python
        from rclpy.parameter_client import AsyncParameterClient
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import Image, JointState
        from std_srvs.srv import SetBool
    except (ImportError, ModuleNotFoundError) as error:
        raise Ros2RuntimeUnavailableError(
            "ROS 2 Python bindings or lerobot_franka_interfaces are unavailable. "
            "Source /opt/ros/jazzy/setup.bash, build franka_project/ros2_ws with colcon, "
            "and source its install/setup.bash before enabling the ROS2 interface."
        ) from error

    return _RosBindings(
        rclpy=rclpy,
        context_type=Context,
        node_type=Node,
        executor_type=SingleThreadedExecutor,
        parameter_client_type=AsyncParameterClient,
        parameter_value_to_python=parameter_value_to_python,
        qos_profile_type=QoSProfile,
        history_policy=HistoryPolicy,
        reliability_policy=ReliabilityPolicy,
        durability_policy=DurabilityPolicy,
        image_type=Image,
        pose_stamped_type=PoseStamped,
        franka_robot_state_type=FrankaRobotState,
        joint_state_type=JointState,
        pose_type=Pose,
        action_chunk_type=CartesianActionChunk,
        action_chunk_ack_type=CartesianActionChunkAck,
        gateway_status_type=SafetyGatewayStatus,
        set_bool_type=SetBool,
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


def _validate_gripper_endpoints(open_position: Any, closed_position: Any) -> tuple[float, float]:
    """Validate endpoint values received from the gripper follower."""

    endpoints = (open_position, closed_position)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in endpoints
    ):
        raise Ros2RuntimeStateError(
            "gripper follower open_position and closed_position must be finite real values"
        )
    resolved_open, resolved_closed = (float(value) for value in endpoints)
    if resolved_open == resolved_closed:
        raise Ros2RuntimeStateError(
            "gripper follower open_position and closed_position must differ"
        )
    return resolved_open, resolved_closed


def _read_gripper_endpoint_parameters(
    *,
    node: Any,
    executor: Any,
    parameter_client_type: type,
    parameter_value_to_python: Any,
    remote_node_name: str,
    timeout_s: float,
) -> tuple[float, float]:
    """Read and freeze the gripper follower's endpoint calibration."""

    client = parameter_client_type(node, remote_node_name)
    if not client.wait_for_services(timeout_sec=timeout_s):
        raise Ros2RuntimeStateError(
            f"gripper endpoint parameter service for {remote_node_name!r} "
            f"was not available within {timeout_s:.3f}s"
        )
    future = client.get_parameters(["open_position", "closed_position"])
    executor.spin_until_future_complete(future, timeout_sec=timeout_s)
    if not future.done():
        future.cancel()
        raise Ros2RuntimeStateError(
            f"timed out reading gripper endpoints from {remote_node_name!r} after {timeout_s:.3f}s"
        )
    error = future.exception()
    if error is not None:
        raise Ros2RuntimeStateError(
            f"failed to read gripper endpoints from {remote_node_name!r}: {error}"
        ) from error
    response = future.result()
    parameters = getattr(response, "values", response)
    if parameters is None or len(parameters) != 2:
        raise Ros2RuntimeStateError(
            f"{remote_node_name!r} returned an incomplete gripper endpoint response"
        )
    if hasattr(response, "values"):
        values = [parameter_value_to_python(value) for value in parameters]
    else:
        values = [parameter.value for parameter in parameters]
    return _validate_gripper_endpoints(values[0], values[1])


def _normalize_quaternion_xyzw(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must be finite shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    return quaternion / norm


def _quaternion_xyzw_to_matrix(value: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(value)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation matrix must be finite shape (3,3)")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = 2.0 * math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0))
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0))
            quaternion = np.asarray(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0))
            quaternion = np.asarray(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    return _normalize_quaternion_xyzw(quaternion)


def _compose_pose(
    left_position: Sequence[float],
    left_quaternion: Sequence[float],
    right_position: Sequence[float],
    right_quaternion: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    left_rotation = _quaternion_xyzw_to_matrix(left_quaternion)
    right_rotation = _quaternion_xyzw_to_matrix(right_quaternion)
    position = np.asarray(left_position, dtype=np.float64) + left_rotation @ np.asarray(
        right_position, dtype=np.float64
    )
    quaternion = _matrix_to_quaternion_xyzw(left_rotation @ right_rotation)
    return position, quaternion


def _invert_pose(position: Sequence[float], quaternion: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    rotation_inverse = _quaternion_xyzw_to_matrix(quaternion).T
    inverse_position = -(rotation_inverse @ np.asarray(position, dtype=np.float64))
    inverse_quaternion = _matrix_to_quaternion_xyzw(rotation_inverse)
    return inverse_position, inverse_quaternion


class Ros2Runtime:
    """Own one private rclpy context and a bounded executor thread."""

    def __init__(self, config: FrankaRosConfig, cache: RosObservationCache):
        self.config = config
        self.cache = cache
        self._lifecycle_lock = threading.RLock()
        self._context: Any | None = None
        self._node: Any | None = None
        self._executor: Any | None = None
        self._thread: threading.Thread | None = None
        self._publisher: Any | None = None
        self._gateway_arm_client: Any | None = None
        self._subscriptions: list[Any] = []
        self._robot_state_subscription: Any | None = None
        self._f_t_ee: tuple[np.ndarray, np.ndarray] | None = None
        self._gripper_open_position: float | None = None
        self._gripper_closed_position: float | None = None
        self._bindings: _RosBindings | None = None
        self._closing = False
        self._spin_error: BaseException | None = None
        self._last_callback_error: dict[str, float] = {}
        # Gateway execution progress, keyed under the same lock:
        # plan acks (accepted/rejected + waypoint count) and the latest status.
        self._gateway_lock = threading.Lock()
        self._plan_acks: dict[int, tuple[bool, int]] = {}
        self._gateway_status: Any | None = None
        self._gateway_status_monotonic_ns: int | None = None

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
                raise Ros2RuntimeStateError("ROS2 runtime is already started")

            bindings = _load_ros_bindings()
            context = bindings.context_type()
            node = None
            executor = None
            robot_state_subscription = None
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
                executor.add_node(node)

                gripper_open_position, gripper_closed_position = (
                    _read_gripper_endpoint_parameters(
                        node=node,
                        executor=executor,
                        parameter_client_type=bindings.parameter_client_type,
                        parameter_value_to_python=bindings.parameter_value_to_python,
                        remote_node_name=self.config.gripper_endpoint_parameter_node,
                        timeout_s=float(self.config.gripper_endpoint_parameter_timeout_s),
                    )
                )
                node.get_logger().info(
                    "Frozen gripper endpoints from "
                    f"{self.config.gripper_endpoint_parameter_node}: "
                    f"open={gripper_open_position:.6f}, closed={gripper_closed_position:.6f}"
                )

                sensor_qos = bindings.qos_profile_type(
                    history=bindings.history_policy.KEEP_LAST,
                    depth=5,
                    reliability=bindings.reliability_policy.BEST_EFFORT,
                    durability=bindings.durability_policy.VOLATILE,
                )
                fixed_transform_qos = bindings.qos_profile_type(
                    history=bindings.history_policy.KEEP_LAST,
                    depth=1,
                    reliability=bindings.reliability_policy.RELIABLE,
                    durability=bindings.durability_policy.TRANSIENT_LOCAL,
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
                        bindings.pose_stamped_type,
                        self.config.eef_pose_topic,
                        self._eef_callback,
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
                    node.create_subscription(
                        bindings.action_chunk_ack_type,
                        _GATEWAY_ACK_TOPIC,
                        self._gateway_ack_callback,
                        action_qos,
                    ),
                    node.create_subscription(
                        bindings.gateway_status_type,
                        _GATEWAY_STATUS_TOPIC,
                        self._gateway_status_callback,
                        action_qos,
                    ),
                ]
                if self.config.policy_eef_frame == "link8":
                    robot_state_subscription = node.create_subscription(
                        bindings.franka_robot_state_type,
                        self.config.robot_state_topic,
                        self._robot_state_callback,
                        fixed_transform_qos,
                    )
                    subscriptions.append(robot_state_subscription)
                publisher = node.create_publisher(
                    bindings.action_chunk_type,
                    self.config.action_chunk_topic,
                    action_qos,
                )
                gateway_arm_client = node.create_client(
                    bindings.set_bool_type,
                    _GATEWAY_ARM_SERVICE,
                )
                self._bindings = bindings
                self._context = context
                self._node = node
                self._executor = executor
                self._publisher = publisher
                self._gateway_arm_client = gateway_arm_client
                self._subscriptions = subscriptions
                self._robot_state_subscription = robot_state_subscription
                self._gripper_open_position = gripper_open_position
                self._gripper_closed_position = gripper_closed_position
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

    def _robot_state_callback(self, message: Any) -> None:
        """Capture the fixed flange->EEF transform once, then drop the 1 kHz stream."""

        try:
            pose = message.f_t_ee.pose
            transform = (
                np.asarray(
                    [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
                    dtype=np.float64,
                ),
                _normalize_quaternion_xyzw(
                    (
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                    )
                ),
            )
            with self._lifecycle_lock:
                self._f_t_ee = transform
                node = self._node
                subscription = self._robot_state_subscription
                self._robot_state_subscription = None
                if subscription in self._subscriptions:
                    self._subscriptions.remove(subscription)
            if node is not None and subscription is not None:
                node.destroy_subscription(subscription)
                node.get_logger().info(
                    "F_T_EE captured for policy_eef_frame=link8; removed 1 kHz RobotState subscription"
                )
        except Exception as error:
            self._log_callback_error("F_T_EE", error)

    def _eef_callback(self, message: Any) -> None:
        try:
            position = message.pose.position
            orientation = message.pose.orientation
            policy_position = np.asarray(
                [float(position.x), float(position.y), float(position.z)],
                dtype=np.float64,
            )
            policy_quaternion = _normalize_quaternion_xyzw(
                (
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                )
            )
            if self.config.policy_eef_frame == "link8":
                with self._lifecycle_lock:
                    transform = self._f_t_ee
                if transform is None:
                    raise Ros2RuntimeStateError("F_T_EE has not been received yet")
                inverse_position, inverse_quaternion = _invert_pose(*transform)
                policy_position, policy_quaternion = _compose_pose(
                    policy_position,
                    policy_quaternion,
                    inverse_position,
                    inverse_quaternion,
                )
            self.cache.update_eef(
                PoseSample(
                    stamp_ns=_message_stamp_ns(message),
                    received_monotonic_ns=time.monotonic_ns(),
                    position=tuple(float(value) for value in policy_position),
                    quaternion_xyzw=tuple(float(value) for value in policy_quaternion),
                    frame_id=str(message.header.frame_id),
                )
            )
        except Exception as error:
            self._log_callback_error("eef", error)

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

    def _gateway_ack_callback(self, message: Any) -> None:
        try:
            key = (str(message.session_id), int(message.plan_id))
            with self._gateway_lock:
                self._plan_acks[key] = (bool(message.accepted), int(message.waypoint_count))
                while len(self._plan_acks) > _MAX_TRACKED_ACKS:
                    self._plan_acks.pop(next(iter(self._plan_acks)))
        except Exception as error:
            self._log_callback_error("gateway ack", error)

    def _gateway_status_callback(self, message: Any) -> None:
        try:
            with self._gateway_lock:
                self._gateway_status = message
                self._gateway_status_monotonic_ns = time.monotonic_ns()
        except Exception as error:
            self._log_callback_error("gateway status", error)

    def get_plan_execution_state(
        self, session_id: str, plan_id: int, *, status_freshness_ns: int
    ) -> GatewayPlanState | None:
        """Combine the plan's ack with live gateway progress; None before the ack."""

        with self._gateway_lock:
            ack = self._plan_acks.get((str(session_id), int(plan_id)))
            status = self._gateway_status
            status_ns = self._gateway_status_monotonic_ns
        if ack is None:
            return None
        accepted, waypoint_count = ack
        status_fresh = (
            status is not None
            and status_ns is not None
            and time.monotonic_ns() - status_ns <= status_freshness_ns
        )
        if not status_fresh:
            return GatewayPlanState(
                accepted=accepted,
                waypoint_count=waypoint_count,
                status_fresh=False,
                status_plan_matches=False,
                applied_waypoint_index=0,
                has_active_plan=False,
                armed=False,
                detail="gateway status is stale",
            )
        status_plan_matches = str(status.session_id) == str(session_id) and int(status.plan_id) == int(
            plan_id
        )
        return GatewayPlanState(
            accepted=accepted,
            waypoint_count=waypoint_count,
            status_fresh=True,
            status_plan_matches=status_plan_matches,
            applied_waypoint_index=(int(status.applied_waypoint_index) if status_plan_matches else 0),
            has_active_plan=bool(status.has_active_plan) and status_plan_matches,
            armed=bool(status.armed),
            detail=str(status.detail),
        )

    def _gripper_joint_state_callback(self, message: Any) -> None:
        try:
            names = tuple(str(name) for name in message.name)
            positions = tuple(float(value) for value in message.position)
            if len(names) != len(positions):
                raise ValueError(f"gripper name/position length mismatch: {len(names)} != {len(positions)}")
            try:
                joint_index = names.index(self.config.gripper_joint_name)
            except ValueError as error:
                raise ValueError(
                    f"gripper JointState does not contain {self.config.gripper_joint_name!r}"
                ) from error

            joint_position = positions[joint_index]
            if not math.isfinite(joint_position):
                raise ValueError("gripper joint position is NaN or Inf")
            open_position = self._gripper_open_position
            closed_position = self._gripper_closed_position
            if open_position is None or closed_position is None:
                raise Ros2RuntimeStateError("gripper endpoints have not been resolved")
            span = closed_position - open_position
            closed_0_1 = (joint_position - open_position) / span
            closed_0_1 = min(1.0, max(0.0, closed_0_1))
            self.cache.update_gripper(
                closed_0_1,
                stamp_ns=_message_stamp_ns(message),
                received_monotonic_ns=time.monotonic_ns(),
            )
        except Exception as error:
            self._log_callback_error("gripper joint state", error)

    def canonicalize_policy_actions(self, actions: np.ndarray) -> np.ndarray:
        """Convert policy-frame targets to canonical ROS O_T_EE targets."""

        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
            raise ValueError(f"policy actions must be finite shape [K,8], got {values.shape}")
        if self.config.policy_eef_frame == "eef":
            return values.copy()
        with self._lifecycle_lock:
            transform = self._f_t_ee
        if transform is None:
            raise Ros2RuntimeStateError("F_T_EE has not been received yet")
        f_t_ee_position, f_t_ee_quaternion = transform
        canonical = values.copy()
        for index, row in enumerate(values):
            position, quaternion = _compose_pose(
                row[:3],
                row[3:7],
                f_t_ee_position,
                f_t_ee_quaternion,
            )
            canonical[index, :3] = position
            canonical[index, 3:7] = quaternion
        return canonical

    def publish_action_chunk(self, chunk: AbsoluteActionChunk) -> None:
        """Validate and publish one complete absolute Cartesian plan."""

        with self._lifecycle_lock:
            if not self.is_running or self._publisher is None or self._bindings is None:
                if self._spin_error is not None:
                    raise Ros2RuntimeStateError("ROS2 executor failed") from self._spin_error
                raise Ros2RuntimeStateError("ROS2 runtime is not running")
            message = self._action_chunk_message(chunk)
            self._publisher.publish(message)

    publish_chunk = publish_action_chunk

    def set_gateway_armed(self, armed: bool, *, timeout_s: float) -> tuple[bool, str]:
        """Call the existing external gateway service; this runtime owns no controller."""

        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("gateway arm timeout must be finite and positive")
        with self._lifecycle_lock:
            client = self._gateway_arm_client
            bindings = self._bindings
            if not self.is_running or client is None or bindings is None or bindings.set_bool_type is None:
                raise Ros2RuntimeStateError("ROS2 runtime is not running")
        if not client.wait_for_service(timeout_sec=timeout):
            raise Ros2RuntimeStateError(f"Gateway arm service unavailable: {_GATEWAY_ARM_SERVICE}")
        request = bindings.set_bool_type.Request()
        request.data = bool(armed)
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise Ros2RuntimeStateError(f"Gateway arm service timed out after {timeout:.3f}s")
        response = future.result()
        if response is None:
            raise Ros2RuntimeStateError("Gateway arm service returned no response")
        return bool(response.success), str(response.message)

    def _action_chunk_message(self, chunk: AbsoluteActionChunk) -> Any:
        bindings = self._bindings
        node = self._node
        if bindings is None or node is None:
            raise Ros2RuntimeStateError("ROS2 runtime is not initialized")

        if not isinstance(chunk, AbsoluteActionChunk):
            raise TypeError("chunk must be an AbsoluteActionChunk")
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
        message.gripper = [float(value) for value in chunk.gripper]
        poses = []
        for row in chunk.poses:
            pose = bindings.pose_type()
            pose.position.x = float(row[0])
            pose.position.y = float(row[1])
            pose.position.z = float(row[2])
            pose.orientation.x = float(row[3])
            pose.orientation.y = float(row[4])
            pose.orientation.z = float(row[5])
            pose.orientation.w = float(row[6])
            poses.append(pose)
        message.poses = poses
        return message

    def close(self, *, timeout_s: float | None = None) -> None:
        """Stop callbacks and release the private context within a bounded time."""

        timeout = float(self.config.ros2_shutdown_timeout_s) if timeout_s is None else float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("ROS2 shutdown timeout must be finite and positive")

        with self._lifecycle_lock:
            if self._context is None and self._thread is None:
                return
            if self._closing:
                raise Ros2RuntimeStateError("ROS2 runtime shutdown is already in progress")
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
                raise Ros2RuntimeStateError(f"ROS2 executor did not stop within {timeout:.3f} seconds")
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
        self._gateway_arm_client = None
        self._subscriptions = []
        self._robot_state_subscription = None
        self._f_t_ee = None
        self._gripper_open_position = None
        self._gripper_closed_position = None
        self._executor = None
        self._node = None
        self._context = None
        self._thread = None
        self._bindings = None
        self._closing = False

    def __enter__(self) -> Ros2Runtime:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "Ros2Runtime",
    "Ros2RuntimeStateError",
    "Ros2RuntimeUnavailableError",
]
