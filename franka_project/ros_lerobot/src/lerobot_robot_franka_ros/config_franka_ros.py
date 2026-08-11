"""Configuration for the out-of-tree Franka Robot plugin."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("franka_ros")
@dataclass(kw_only=True)
class FrankaRosConfig(RobotConfig):
    """Dry-run and non-actuating ROS2-interface configuration."""

    dry_run: bool = True
    fixture_path: Path | None = None
    action_log_path: Path | None = None
    quaternion_norm_tolerance: float = 1e-3

    # ``dry_run=false`` only enables the isolated ROS interface.  It does not
    # authorize a controller, IK implementation, or Franka actuation path.
    ros2_interface_only: bool = True
    ros2_node_name: str = "lerobot_franka_interface"
    action_chunk_topic: str = "/lerobot/franka/action_chunk"
    # Maximum number of fresh actions committed to both the local queue and a
    # single ROS action chunk. ``None`` preserves the complete fresh suffix.
    max_action_chunk_waypoints: int | None = None
    camera1_topic: str = "/camera1/camera1/color/image_raw"
    camera2_topic: str = "/camera2/camera2/color/image_raw"
    eef_pose_topic: str = "/franka_robot_state_broadcaster/current_pose"
    robot_state_topic: str = "/franka_robot_state_broadcaster/robot_state"
    # Frame semantics used by the policy checkpoint. ``eef`` is the Haply/ROS
    # configured TCP (O_T_EE); ``link8`` is the Frankateleop/Polymetis flange
    # pose (O_T_panda_link8 / O_T_fr3_link8).
    policy_eef_frame: str = "eef"
    qpos_topic: str = "/franka/joint_states"
    gripper_topic: str = "/gripper/joint_states"
    base_frame: str = "base"
    arm_joint_names: tuple[str, ...] = (
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    )
    gripper_joint_name: str = "robotiq_85_left_knuckle_joint"
    # Live endpoint calibration is owned by the gripper follower.  The ROS2
    # runtime reads ``open_position`` and ``closed_position`` from this node
    # once during startup and then freezes them for the lifetime of the client.
    gripper_endpoint_parameter_node: str = "/franka_gripper_follower"
    gripper_endpoint_parameter_timeout_s: float = 5.0
    # These values remain the non-ROS/dry-run defaults.  A live ROS2 runtime
    # never uses them to normalize JointState messages.
    gripper_open_position: float = 0.0
    gripper_closed_position: float = 0.8
    observation_buffer_size: int = 32
    max_observation_age_s: float = 0.25
    camera2_max_skew_s: float = 0.05
    eef_max_skew_s: float = 0.05
    qpos_max_skew_s: float = 0.1
    gripper_max_skew_s: float = 0.05
    action_chunk_validity_s: float = 0.5
    ros2_shutdown_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            isinstance(self.quaternion_norm_tolerance, bool)
            or not isinstance(self.quaternion_norm_tolerance, Real)
            or not math.isfinite(float(self.quaternion_norm_tolerance))
            or self.quaternion_norm_tolerance <= 0.0
            or self.quaternion_norm_tolerance > 0.1
        ):
            raise ValueError("quaternion_norm_tolerance must be in (0, 0.1]")
        if not self.dry_run:
            if not self.ros2_interface_only:
                raise ValueError(
                    "Franka actuation is not implemented; dry_run=false requires ros2_interface_only=true"
                )
            if self.fixture_path is not None or self.action_log_path is not None:
                raise ValueError("fixture_path/action_log_path are only valid when dry_run=true")

        topic_fields = {
            "action_chunk_topic": self.action_chunk_topic,
            "camera1_topic": self.camera1_topic,
            "camera2_topic": self.camera2_topic,
            "eef_pose_topic": self.eef_pose_topic,
            "robot_state_topic": self.robot_state_topic,
            "qpos_topic": self.qpos_topic,
            "gripper_topic": self.gripper_topic,
        }
        for name, topic in topic_fields.items():
            if not isinstance(topic, str) or not topic.startswith("/") or " " in topic:
                raise ValueError(f"{name} must be an absolute ROS topic without spaces")
        if len(set(topic_fields.values())) != len(topic_fields):
            raise ValueError("ROS topic names must be distinct")
        if not self.action_chunk_topic.startswith("/lerobot/"):
            raise ValueError("action_chunk_topic must stay under the isolated /lerobot/ namespace")
        if self.policy_eef_frame not in {"eef", "link8"}:
            raise ValueError("policy_eef_frame must be 'eef' or 'link8'")
        if self.max_action_chunk_waypoints is not None and (
            isinstance(self.max_action_chunk_waypoints, bool)
            or not isinstance(self.max_action_chunk_waypoints, Integral)
            or self.max_action_chunk_waypoints <= 0
        ):
            raise ValueError("max_action_chunk_waypoints must be a positive integer or None")
        if (
            not isinstance(self.ros2_node_name, str)
            or not self.ros2_node_name
            or "/" in self.ros2_node_name
            or " " in self.ros2_node_name
        ):
            raise ValueError("ros2_node_name must be a non-empty ROS node basename")
        if not isinstance(self.base_frame, str) or not self.base_frame:
            raise ValueError("base_frame must not be empty")
        if len(self.arm_joint_names) != 7 or len(set(self.arm_joint_names)) != 7:
            raise ValueError("arm_joint_names must contain seven unique joint names")
        if any(not name for name in self.arm_joint_names):
            raise ValueError("arm_joint_names must not contain empty names")
        if not self.gripper_joint_name:
            raise ValueError("gripper_joint_name must not be empty")
        if (
            not isinstance(self.gripper_endpoint_parameter_node, str)
            or not self.gripper_endpoint_parameter_node.startswith("/")
            or " " in self.gripper_endpoint_parameter_node
        ):
            raise ValueError(
                "gripper_endpoint_parameter_node must be an absolute ROS node name without spaces"
            )
        gripper_positions = (self.gripper_open_position, self.gripper_closed_position)
        if any(
            isinstance(position, bool)
            or not isinstance(position, Real)
            or not math.isfinite(float(position))
            for position in gripper_positions
        ):
            raise ValueError("gripper open and closed positions must be finite real values")
        if self.gripper_open_position == self.gripper_closed_position:
            raise ValueError("gripper open and closed positions must differ")
        if (
            isinstance(self.observation_buffer_size, bool)
            or not isinstance(self.observation_buffer_size, Integral)
            or self.observation_buffer_size < 2
        ):
            raise ValueError("observation_buffer_size must be at least 2")

        positive_durations = {
            "max_observation_age_s": self.max_observation_age_s,
            "camera2_max_skew_s": self.camera2_max_skew_s,
            "eef_max_skew_s": self.eef_max_skew_s,
            "qpos_max_skew_s": self.qpos_max_skew_s,
            "gripper_max_skew_s": self.gripper_max_skew_s,
            "gripper_endpoint_parameter_timeout_s": self.gripper_endpoint_parameter_timeout_s,
            "action_chunk_validity_s": self.action_chunk_validity_s,
            "ros2_shutdown_timeout_s": self.ros2_shutdown_timeout_s,
        }
        for name, value in positive_durations.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and greater than zero")
