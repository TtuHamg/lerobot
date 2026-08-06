"""Configuration for the out-of-tree Franka joint-space (FastWAM) Robot plugin."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("franka_ros_joint")
@dataclass(kw_only=True)
class FrankaJointRosConfig(RobotConfig):
    """Dry-run and non-actuating ROS2-interface configuration for the joint-space policy.

    This mirrors :class:`FrankaRosConfig` but drops the Cartesian end-effector
    inputs. FastWAM consumes the raw joint-space state (seven Franka joints plus
    one raw gripper position) and produces absolute joint-space targets, so no
    ``eef_pose`` topic or quaternion tolerance is required here.
    """

    dry_run: bool = True
    fixture_path: Path | None = None
    action_log_path: Path | None = None

    # ``dry_run=false`` only enables the isolated ROS interface.  It does not
    # authorize a controller, IK implementation, or Franka actuation path.
    ros2_interface_only: bool = True
    ros2_node_name: str = "lerobot_franka_joint_interface"
    action_chunk_topic: str = "/lerobot/franka/joint_action_chunk"
    action_chunk_ack_topic: str = "/lerobot/franka/joint_action_chunk_ack"
    safety_gateway_status_topic: str = "/lerobot/franka/joint_safety_gateway_status"
    camera1_topic: str = "/camera1/camera1/color/image_raw"
    camera2_topic: str = "/camera2/camera2/color/image_raw"
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
    observation_buffer_size: int = 32
    max_observation_age_s: float = 0.25
    camera2_max_skew_s: float = 0.05
    qpos_max_skew_s: float = 0.1
    gripper_max_skew_s: float = 0.05
    # Forward inverse-normalized model gripper targets continuously, saturating
    # finite out-of-range predictions to this hardware safety envelope.
    gripper_command_min_position: float = 0.0
    gripper_command_max_position: float = 0.8
    action_chunk_validity_s: float = 0.5
    action_execution_timeout_s: float = 30.0
    ros2_shutdown_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.dry_run:
            if not self.ros2_interface_only:
                raise ValueError(
                    "Franka actuation is not implemented; dry_run=false requires ros2_interface_only=true"
                )
            if self.fixture_path is not None or self.action_log_path is not None:
                raise ValueError("fixture_path/action_log_path are only valid when dry_run=true")

        topic_fields = {
            "action_chunk_topic": self.action_chunk_topic,
            "action_chunk_ack_topic": self.action_chunk_ack_topic,
            "safety_gateway_status_topic": self.safety_gateway_status_topic,
            "camera1_topic": self.camera1_topic,
            "camera2_topic": self.camera2_topic,
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
        gripper_limits = (
            self.gripper_command_min_position,
            self.gripper_command_max_position,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in gripper_limits
        ):
            raise ValueError("gripper command limits must be finite real values")
        if self.gripper_command_min_position >= self.gripper_command_max_position:
            raise ValueError("gripper command minimum must be less than maximum")
        if (
            isinstance(self.observation_buffer_size, bool)
            or not isinstance(self.observation_buffer_size, Integral)
            or self.observation_buffer_size < 2
        ):
            raise ValueError("observation_buffer_size must be at least 2")

        positive_durations = {
            "max_observation_age_s": self.max_observation_age_s,
            "camera2_max_skew_s": self.camera2_max_skew_s,
            "qpos_max_skew_s": self.qpos_max_skew_s,
            "gripper_max_skew_s": self.gripper_max_skew_s,
            "action_chunk_validity_s": self.action_chunk_validity_s,
            "ros2_shutdown_timeout_s": self.ros2_shutdown_timeout_s,
            "action_execution_timeout_s": self.action_execution_timeout_s,
        }
        for name, value in positive_durations.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and greater than zero")
