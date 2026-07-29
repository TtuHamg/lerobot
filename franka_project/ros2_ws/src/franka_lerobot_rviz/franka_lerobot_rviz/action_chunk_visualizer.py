# Copyright 2026 pnp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only ROS 2 visualization of LeRobot Cartesian plans in RViz."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict, deque
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from franka_safety_interfaces.msg import SafetyCommandFeedback
from lerobot_franka_interfaces.msg import (
    CartesianActionChunk,
    CartesianActionChunkAck,
    SafetyGatewayStatus,
)

from .visualization_model import (
    EXECUTION_COLOR,
    HISTORY_COLOR,
    HOLD_COLOR,
    REJECTED_COLOR,
    SHADOW_COLOR,
    AckVisualState,
    PlanKey,
    abbreviated_session,
    classify_ack,
    duration_to_nanoseconds,
    gripper_color,
    plan_key,
    scheduled_pose_nanoseconds,
    split_nanoseconds,
    validate_chunk_shape,
)


@dataclass
class PlanRecord:
    """Cached plan plus its independently arriving ACK."""

    chunk: CartesianActionChunk
    ack: CartesianActionChunkAck | None = None


@dataclass(frozen=True)
class FeedbackProgress:
    """Latest positively applied controller feedback for one plan."""

    sequence: int
    waypoint_index: int
    source_timestep: int


def _color(rgba: tuple[float, float, float, float]) -> ColorRGBA:
    message = ColorRGBA()
    message.r, message.g, message.b, message.a = rgba
    return message


def _point(pose: Pose) -> Point:
    point = Point()
    point.x = pose.position.x
    point.y = pose.position.y
    point.z = pose.position.z
    return point


def _topic_qos(*, depth: int, reliable: bool, transient_local: bool = False) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=(ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT),
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if transient_local else DurabilityPolicy.VOLATILE),
    )


class FrankaActionChunkVisualizer(Node):
    """Observe the control chain and publish visualization-only messages."""

    def __init__(self) -> None:
        super().__init__("franka_action_chunk_visualizer")

        self.declare_parameter("fixed_frame", "base")
        self.declare_parameter("action_chunk_topic", "/lerobot/franka/action_chunk")
        self.declare_parameter("ack_topic", "/lerobot/franka/action_chunk_ack")
        self.declare_parameter("gateway_status_topic", "/lerobot/franka/safety_gateway_status")
        self.declare_parameter("safety_feedback_topic", "/franka/safety_command_feedback")
        self.declare_parameter(
            "current_pose_topic", "/franka_robot_state_broadcaster/current_pose"
        )
        self.declare_parameter("planned_path_topic", "/lerobot/franka/viz/planned_path")
        self.declare_parameter("active_path_topic", "/lerobot/franka/viz/active_path")
        self.declare_parameter("planned_poses_topic", "/lerobot/franka/viz/planned_poses")
        self.declare_parameter("actual_path_topic", "/lerobot/franka/viz/actual_path")
        self.declare_parameter("actual_pose_topic", "/lerobot/franka/viz/actual_pose")
        self.declare_parameter("markers_topic", "/lerobot/franka/viz/markers")
        self.declare_parameter("orientation_stride", 5)
        self.declare_parameter("actual_publish_hz", 15.0)
        self.declare_parameter("actual_trail_seconds", 20.0)
        self.declare_parameter("actual_max_points", 600)
        self.declare_parameter("max_cached_plans", 32)

        self._fixed_frame = str(self.get_parameter("fixed_frame").value)
        self._orientation_stride = max(1, int(self.get_parameter("orientation_stride").value))
        actual_publish_hz = max(0.1, float(self.get_parameter("actual_publish_hz").value))
        self._actual_publish_period_ns = int(1_000_000_000 / actual_publish_hz)
        self._actual_trail_ns = int(
            max(0.1, float(self.get_parameter("actual_trail_seconds").value)) * 1_000_000_000
        )
        self._actual_max_points = max(2, int(self.get_parameter("actual_max_points").value))
        self._max_cached_plans = max(2, int(self.get_parameter("max_cached_plans").value))

        self._plans: OrderedDict[PlanKey, PlanRecord] = OrderedDict()
        self._early_acks: OrderedDict[PlanKey, CartesianActionChunkAck] = OrderedDict()
        self._feedback: dict[PlanKey, FeedbackProgress] = {}
        self._latest_candidate_key: PlanKey | None = None
        self._active_key: PlanKey | None = None
        self._gateway_status: SafetyGatewayStatus | None = None
        self._last_status_signature: tuple[object, ...] | None = None
        self._actual_samples: deque[tuple[int, PoseStamped]] = deque(
            maxlen=self._actual_max_points
        )
        self._last_actual_pose: PoseStamped | None = None
        self._last_actual_publish_ns = 0

        # Visualization publishers are transient-local snapshots so RViz can be
        # started after this node and still receive the complete current scene.
        output_qos = _topic_qos(depth=1, reliable=True, transient_local=True)
        self._planned_path_publisher = self.create_publisher(
            Path, str(self.get_parameter("planned_path_topic").value), output_qos
        )
        self._active_path_publisher = self.create_publisher(
            Path, str(self.get_parameter("active_path_topic").value), output_qos
        )
        self._planned_poses_publisher = self.create_publisher(
            PoseArray, str(self.get_parameter("planned_poses_topic").value), output_qos
        )
        self._actual_path_publisher = self.create_publisher(
            Path, str(self.get_parameter("actual_path_topic").value), output_qos
        )
        self._actual_pose_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("actual_pose_topic").value), output_qos
        )
        self._markers_publisher = self.create_publisher(
            MarkerArray, str(self.get_parameter("markers_topic").value), output_qos
        )

        # High-rate streams are observed with best-effort QoS. A visualizer is
        # allowed to lose a sample and must never add reliable backpressure to
        # the control path. ACK is one-shot and therefore kept reliable.
        self._subscriptions = [
            self.create_subscription(
                CartesianActionChunk,
                str(self.get_parameter("action_chunk_topic").value),
                self._on_chunk,
                _topic_qos(depth=1, reliable=False),
            ),
            self.create_subscription(
                CartesianActionChunkAck,
                str(self.get_parameter("ack_topic").value),
                self._on_ack,
                _topic_qos(depth=10, reliable=True),
            ),
            self.create_subscription(
                SafetyGatewayStatus,
                str(self.get_parameter("gateway_status_topic").value),
                self._on_gateway_status,
                _topic_qos(depth=1, reliable=False),
            ),
            self.create_subscription(
                SafetyCommandFeedback,
                str(self.get_parameter("safety_feedback_topic").value),
                self._on_safety_feedback,
                _topic_qos(depth=8, reliable=False),
            ),
            self.create_subscription(
                PoseStamped,
                str(self.get_parameter("current_pose_topic").value),
                self._on_current_pose,
                _topic_qos(depth=5, reliable=False),
            ),
        ]

        # Publish explicit empty snapshots once so a previous transient RViz
        # scene cannot survive a visualizer restart.
        self._publish_scene()
        self.get_logger().info(
            "Read-only visualization ready; publishing only under "
            "'/lerobot/franka/viz/*' by default"
        )

    def _on_chunk(self, message: CartesianActionChunk) -> None:
        valid, reason = validate_chunk_shape(message.timesteps, message.poses, message.gripper)
        if valid:
            valid, reason = self._validate_chunk_values(message)
        if not valid:
            self.get_logger().warning(
                f"Ignoring malformed visualization chunk "
                f"{message.session_id}/{message.plan_id}: {reason}"
            )
            return

        key = plan_key(message.session_id, message.plan_id)
        ack = self._early_acks.pop(key, None)
        self._plans[key] = PlanRecord(chunk=copy.deepcopy(message), ack=ack)
        self._plans.move_to_end(key)
        self._latest_candidate_key = key
        self._evict_old_plans()
        self._publish_scene()

    def _on_ack(self, message: CartesianActionChunkAck) -> None:
        key = plan_key(message.session_id, message.plan_id)
        ack = copy.deepcopy(message)
        record = self._plans.get(key)
        if record is not None:
            record.ack = ack
        else:
            self._early_acks[key] = ack
            self._early_acks.move_to_end(key)
            while len(self._early_acks) > self._max_cached_plans * 2:
                self._early_acks.popitem(last=False)

        if key == self._latest_candidate_key or key == self._active_key:
            self._publish_scene()

    def _on_gateway_status(self, message: SafetyGatewayStatus) -> None:
        new_active_key: PlanKey | None = None
        if message.has_active_plan and message.session_id:
            new_active_key = plan_key(message.session_id, message.plan_id)

        signature = (
            int(message.state),
            bool(message.shadow),
            bool(message.armed),
            bool(message.state_fresh),
            bool(message.preflight_available),
            bool(message.robot_ready),
            bool(message.controller_ready),
            bool(message.applied_feedback_fresh),
            bool(message.has_active_plan),
            message.session_id,
            int(message.plan_id),
            int(message.next_waypoint_index),
            int(message.accepted_waypoint_index),
            int(message.applied_waypoint_index),
            int(message.measured_waypoint_index),
            int(message.last_applied_sequence),
            int(message.applied_source_timestep),
            round(float(message.tcp_position_error_m), 5),
            round(float(message.tcp_orientation_error_rad), 5),
            message.detail,
        )
        active_changed = new_active_key != self._active_key
        self._active_key = new_active_key
        self._gateway_status = copy.deepcopy(message)
        if active_changed:
            self._evict_old_plans()
        if active_changed or signature != self._last_status_signature:
            self._last_status_signature = signature
            self._publish_scene()

    def _on_safety_feedback(self, message: SafetyCommandFeedback) -> None:
        if not (
            message.accepted
            and message.applied
            and message.status == SafetyCommandFeedback.STATUS_APPLIED
        ):
            return

        key = plan_key(message.session_id, message.plan_id)
        previous = self._feedback.get(key)
        if previous is not None and int(message.sequence) <= previous.sequence:
            return

        progress = FeedbackProgress(
            sequence=int(message.sequence),
            waypoint_index=int(message.waypoint_index),
            source_timestep=int(message.source_timestep),
        )
        self._feedback[key] = progress
        if key == self._active_key:
            self._publish_scene()

    def _on_current_pose(self, message: PoseStamped) -> None:
        pose = copy.deepcopy(message)
        if not pose.header.frame_id:
            pose.header.frame_id = self._fixed_frame

        # Mixing coordinates from different frames would draw a plausible but
        # false trail. Start a new trail instead; no transform is attempted.
        if (
            self._last_actual_pose is not None
            and self._last_actual_pose.header.frame_id != pose.header.frame_id
        ):
            self._actual_samples.clear()

        receive_ns = self.get_clock().now().nanoseconds
        self._last_actual_pose = pose
        if receive_ns - self._last_actual_publish_ns < self._actual_publish_period_ns:
            return

        # Downsample the potentially 1 kHz state broadcaster at the same rate
        # used for RViz. This makes a 20-second trail actually span 20 seconds
        # instead of filling the point cap in a fraction of a second.
        self._last_actual_publish_ns = receive_ns
        self._actual_samples.append((receive_ns, pose))
        cutoff_ns = receive_ns - self._actual_trail_ns
        while self._actual_samples and self._actual_samples[0][0] < cutoff_ns:
            self._actual_samples.popleft()
        self._publish_scene()

    def _validate_chunk_values(self, message: CartesianActionChunk) -> tuple[bool, str]:
        period_ns = duration_to_nanoseconds(message.period.sec, message.period.nanosec)
        if period_ns <= 0:
            return False, f"period must be positive, got {period_ns} ns"
        for index, (pose, gripper) in enumerate(zip(message.poses, message.gripper, strict=True)):
            values = (
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
                gripper,
            )
            if not all(math.isfinite(float(value)) for value in values):
                return False, f"waypoint {index} contains a non-finite value"
            quaternion_norm = math.sqrt(
                pose.orientation.x**2
                + pose.orientation.y**2
                + pose.orientation.z**2
                + pose.orientation.w**2
            )
            if quaternion_norm < 1e-9:
                return False, f"waypoint {index} has a zero quaternion"
        return True, ""

    def _evict_old_plans(self) -> None:
        protected = {self._latest_candidate_key, self._active_key}
        protected.discard(None)
        while len(self._plans) > self._max_cached_plans:
            evicted = False
            for key in list(self._plans):
                if key not in protected:
                    del self._plans[key]
                    self._feedback.pop(key, None)
                    evicted = True
                    break
            if not evicted:
                break

    def _publish_scene(self) -> None:
        candidate = (
            self._plans.get(self._latest_candidate_key)
            if self._latest_candidate_key is not None
            else None
        )
        active = self._plans.get(self._active_key) if self._active_key is not None else None

        if candidate is None:
            self._planned_path_publisher.publish(self._empty_path())
            self._planned_poses_publisher.publish(self._empty_pose_array())
        else:
            self._planned_path_publisher.publish(self._path_from_chunk(candidate.chunk))
            self._planned_poses_publisher.publish(self._pose_array_from_chunk(candidate.chunk))

        if active is None:
            self._active_path_publisher.publish(self._empty_path())
        else:
            self._active_path_publisher.publish(self._path_from_chunk(active.chunk))

        self._actual_path_publisher.publish(self._actual_path())
        if self._last_actual_pose is not None:
            self._actual_pose_publisher.publish(copy.deepcopy(self._last_actual_pose))
        self._markers_publisher.publish(self._build_markers(candidate, active))

    def _empty_path(self) -> Path:
        message = Path()
        message.header.frame_id = self._fixed_frame
        message.header.stamp = self.get_clock().now().to_msg()
        return message

    def _empty_pose_array(self) -> PoseArray:
        message = PoseArray()
        message.header.frame_id = self._fixed_frame
        message.header.stamp = self.get_clock().now().to_msg()
        return message

    def _path_from_chunk(self, chunk: CartesianActionChunk) -> Path:
        message = Path()
        message.header = copy.deepcopy(chunk.header)
        if not message.header.frame_id:
            message.header.frame_id = self._fixed_frame
        for index, pose in enumerate(chunk.poses):
            stamped = PoseStamped()
            stamped.header.frame_id = message.header.frame_id
            target_ns = scheduled_pose_nanoseconds(
                chunk.header.stamp.sec,
                chunk.header.stamp.nanosec,
                chunk.period.sec,
                chunk.period.nanosec,
                index,
            )
            sec, nanosec = split_nanoseconds(target_ns)
            stamped.header.stamp.sec = sec
            stamped.header.stamp.nanosec = nanosec
            stamped.pose = copy.deepcopy(pose)
            message.poses.append(stamped)
        return message

    def _pose_array_from_chunk(self, chunk: CartesianActionChunk) -> PoseArray:
        message = PoseArray()
        message.header = copy.deepcopy(chunk.header)
        if not message.header.frame_id:
            message.header.frame_id = self._fixed_frame
        message.poses = [copy.deepcopy(pose) for pose in chunk.poses]
        return message

    def _actual_path(self) -> Path:
        if not self._actual_samples:
            return self._empty_path()
        message = Path()
        message.header = copy.deepcopy(self._actual_samples[-1][1].header)
        message.poses = [copy.deepcopy(sample) for _, sample in self._actual_samples]
        return message

    def _new_marker(
        self,
        namespace: str,
        marker_id: int,
        marker_type: int,
        frame_id: str,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id or self._fixed_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _build_markers(
        self, candidate: PlanRecord | None, active: PlanRecord | None
    ) -> MarkerArray:
        array = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self._fixed_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        if candidate is not None:
            array.markers.extend(self._candidate_markers(candidate))
        if active is not None:
            array.markers.extend(self._active_markers(active))
        if self._gateway_status is not None:
            array.markers.append(self._gateway_text_marker(candidate, active))
        return array

    def _candidate_markers(self, record: PlanRecord) -> list[Marker]:
        chunk = record.chunk
        frame_id = chunk.header.frame_id or self._fixed_frame
        state = self._ack_visual_state(record.ack)
        markers: list[Marker] = []

        line = self._new_marker("candidate/path", 0, Marker.LINE_STRIP, frame_id)
        line.scale.x = 0.007
        line.color = _color(state.color)
        line.points = [_point(pose) for pose in chunk.poses]
        markers.append(line)

        waypoints = self._new_marker("candidate/gripper", 0, Marker.SPHERE_LIST, frame_id)
        waypoints.scale.x = 0.018
        waypoints.scale.y = 0.018
        waypoints.scale.z = 0.018
        waypoints.points = [_point(pose) for pose in chunk.poses]
        waypoints.colors = [_color(gripper_color(value)) for value in chunk.gripper]
        markers.append(waypoints)

        for index in range(0, len(chunk.poses), self._orientation_stride):
            arrow = self._new_marker(
                "candidate/orientation", index, Marker.ARROW, frame_id
            )
            arrow.pose = copy.deepcopy(chunk.poses[index])
            arrow.scale.x = 0.065
            arrow.scale.y = 0.010
            arrow.scale.z = 0.014
            arrow.color = _color(state.color)
            markers.append(arrow)

        start = self._new_marker("candidate/start", 0, Marker.SPHERE, frame_id)
        start.pose = copy.deepcopy(chunk.poses[0])
        start.scale.x = start.scale.y = start.scale.z = 0.030
        start.color = _color((0.95, 0.95, 0.95, 0.95))
        markers.append(start)

        end = self._new_marker("candidate/end", 0, Marker.SPHERE, frame_id)
        end.pose = copy.deepcopy(chunk.poses[-1])
        end.scale.x = end.scale.y = end.scale.z = 0.035
        end.color = _color(state.color)
        markers.append(end)

        text = self._new_marker("candidate/label", 0, Marker.TEXT_VIEW_FACING, frame_id)
        text.pose.position = copy.deepcopy(chunk.poses[-1].position)
        text.pose.position.z += 0.060
        text.scale.z = 0.025
        text.color = _color(state.color)
        key_text = f"{abbreviated_session(chunk.session_id)}/{int(chunk.plan_id)}"
        detail = self._clean_text(record.ack.detail, 72) if record.ack is not None else ""
        text.text = f"candidate {key_text} | {len(chunk.poses)} wp\n{state.label}"
        if detail:
            text.text += f"\n{detail}"
        markers.append(text)
        return markers

    def _active_markers(self, record: PlanRecord) -> list[Marker]:
        chunk = record.chunk
        frame_id = chunk.header.frame_id or self._fixed_frame
        status = self._gateway_status
        if status is None:
            return []
        color = self._gateway_color(status)
        count = len(chunk.poses)
        next_index = int(status.next_waypoint_index)
        markers: list[Marker] = []

        # The full active path remains visible in gray; the unconsumed suffix is
        # overlaid in the live gateway-state color.
        full = self._new_marker("active/full", 0, Marker.LINE_STRIP, frame_id)
        full.scale.x = 0.010
        full.color = _color(HISTORY_COLOR)
        full.points = [_point(pose) for pose in chunk.poses]
        markers.append(full)

        if 0 <= next_index < count:
            remaining = self._new_marker("active/remaining", 0, Marker.LINE_STRIP, frame_id)
            remaining.scale.x = 0.014
            remaining.color = _color(color)
            start_index = max(0, next_index - 1)
            remaining.points = [_point(pose) for pose in chunk.poses[start_index:]]
            markers.append(remaining)

            target = self._new_marker("active/next", 0, Marker.SPHERE, frame_id)
            target.pose = copy.deepcopy(chunk.poses[next_index])
            target.scale.x = target.scale.y = target.scale.z = 0.045
            target.color = _color((1.00, 0.95, 0.10, 0.98))
            markers.append(target)

            if (
                self._last_actual_pose is not None
                and self._last_actual_pose.header.frame_id == frame_id
            ):
                tracking = self._new_marker("active/tracking_error", 0, Marker.LINE_LIST, frame_id)
                tracking.scale.x = 0.006
                tracking.color = _color((1.00, 0.35, 0.05, 0.90))
                tracking.points = [
                    _point(self._last_actual_pose.pose),
                    _point(chunk.poses[next_index]),
                ]
                markers.append(tracking)

        applied_index = self._applied_waypoint_index(status)
        if applied_index is not None and 0 <= applied_index < count:
            applied = self._new_marker("active/applied", 0, Marker.CUBE, frame_id)
            applied.pose = copy.deepcopy(chunk.poses[applied_index])
            applied.scale.x = applied.scale.y = applied.scale.z = 0.032
            applied.color = _color((0.75, 0.25, 1.00, 0.90))
            markers.append(applied)
        return markers

    def _gateway_text_marker(
        self, candidate: PlanRecord | None, active: PlanRecord | None
    ) -> Marker:
        status = self._gateway_status
        assert status is not None

        anchor: Pose | None = None
        frame_id = self._fixed_frame
        if self._last_actual_pose is not None:
            anchor = self._last_actual_pose.pose
            frame_id = self._last_actual_pose.header.frame_id or self._fixed_frame
        elif active is not None:
            anchor = active.chunk.poses[0]
            frame_id = active.chunk.header.frame_id or self._fixed_frame
        elif candidate is not None:
            anchor = candidate.chunk.poses[0]
            frame_id = candidate.chunk.header.frame_id or self._fixed_frame

        marker = self._new_marker("gateway/status", 0, Marker.TEXT_VIEW_FACING, frame_id)
        if anchor is not None:
            marker.pose.position = copy.deepcopy(anchor.position)
            marker.pose.position.z += 0.160
        else:
            marker.pose.position.x = 0.35
            marker.pose.position.z = 0.65
        marker.scale.z = 0.024
        marker.color = _color(self._gateway_color(status))

        state_name = self._gateway_state_name(status)
        if status.has_active_plan and status.session_id:
            active_text = (
                f"{abbreviated_session(status.session_id)}/{int(status.plan_id)} "
                f"next={int(status.next_waypoint_index)} "
                f"applied={int(status.applied_waypoint_index)}"
            )
        else:
            active_text = "none"
        readiness = (
            f"fresh={int(status.state_fresh)} robot={int(status.robot_ready)} "
            f"controller={int(status.controller_ready)} feedback={int(status.applied_feedback_fresh)}"
        )
        errors = (
            f"tcp error: {float(status.tcp_position_error_m):.4f} m / "
            f"{float(status.tcp_orientation_error_rad):.4f} rad"
        )
        marker.text = f"gateway {state_name} | active {active_text}\n{readiness}\n{errors}"
        detail = self._clean_text(status.detail, 96)
        if detail:
            marker.text += f"\n{detail}"
        return marker

    def _ack_visual_state(self, ack: CartesianActionChunkAck | None) -> AckVisualState:
        if ack is None:
            return classify_ack(None, None)
        return classify_ack(bool(ack.accepted), int(ack.result))

    def _applied_waypoint_index(self, status: SafetyGatewayStatus) -> int | None:
        key = self._active_key
        if key is None:
            return None
        feedback = self._feedback.get(key)
        if feedback is not None and feedback.sequence >= int(status.last_applied_sequence):
            return feedback.waypoint_index
        if int(status.last_applied_sequence) > 0:
            return int(status.applied_waypoint_index)
        return None

    @staticmethod
    def _gateway_state_name(status: SafetyGatewayStatus) -> str:
        names = {
            SafetyGatewayStatus.DISABLED: "DISABLED",
            SafetyGatewayStatus.SHADOW: "SHADOW",
            SafetyGatewayStatus.ARMED: "ARMED",
            SafetyGatewayStatus.HOLD: "HOLD",
            SafetyGatewayStatus.FAULT: "FAULT",
        }
        return names.get(int(status.state), f"UNKNOWN({int(status.state)})")

    @staticmethod
    def _gateway_color(status: SafetyGatewayStatus) -> tuple[float, float, float, float]:
        if status.shadow or status.state == SafetyGatewayStatus.SHADOW:
            return SHADOW_COLOR
        if status.state == SafetyGatewayStatus.ARMED and status.armed:
            return EXECUTION_COLOR
        if status.state == SafetyGatewayStatus.HOLD:
            return HOLD_COLOR
        if status.state == SafetyGatewayStatus.FAULT:
            return REJECTED_COLOR
        return HISTORY_COLOR

    @staticmethod
    def _clean_text(value: str, limit: int) -> str:
        compact = " ".join(value.split())
        return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FrankaActionChunkVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
