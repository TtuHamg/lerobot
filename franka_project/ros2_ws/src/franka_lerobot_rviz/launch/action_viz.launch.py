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

"""Launch the read-only Franka/LeRobot action visualizer and RViz."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("franka_lerobot_rviz"))
    default_rviz_config = package_share / "rviz" / "action_viz.rviz"

    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    fixed_frame = LaunchConfiguration("fixed_frame")
    action_chunk_topic = LaunchConfiguration("action_chunk_topic")
    ack_topic = LaunchConfiguration("ack_topic")
    gateway_status_topic = LaunchConfiguration("gateway_status_topic")
    safety_feedback_topic = LaunchConfiguration("safety_feedback_topic")
    current_pose_topic = LaunchConfiguration("current_pose_topic")
    camera1_topic = LaunchConfiguration("camera1_topic")
    camera2_topic = LaunchConfiguration("camera2_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("rviz_config", default_value=str(default_rviz_config)),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("fixed_frame", default_value="base"),
            DeclareLaunchArgument(
                "action_chunk_topic", default_value="/lerobot/franka/action_chunk"
            ),
            DeclareLaunchArgument(
                "ack_topic", default_value="/lerobot/franka/action_chunk_ack"
            ),
            DeclareLaunchArgument(
                "gateway_status_topic",
                default_value="/lerobot/franka/safety_gateway_status",
            ),
            DeclareLaunchArgument(
                "safety_feedback_topic", default_value="/franka/safety_command_feedback"
            ),
            DeclareLaunchArgument(
                "current_pose_topic",
                default_value="/franka_robot_state_broadcaster/current_pose",
            ),
            DeclareLaunchArgument(
                "camera1_topic",
                default_value="/camera1/camera1/color/image_raw",
            ),
            DeclareLaunchArgument(
                "camera2_topic",
                default_value="/camera2/camera2/color/image_raw",
            ),
            Node(
                package="franka_lerobot_rviz",
                executable="franka_action_chunk_visualizer",
                name="franka_action_chunk_visualizer",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                        "fixed_frame": fixed_frame,
                        "action_chunk_topic": action_chunk_topic,
                        "ack_topic": ack_topic,
                        "gateway_status_topic": gateway_status_topic,
                        "safety_feedback_topic": safety_feedback_topic,
                        "current_pose_topic": current_pose_topic,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="franka_lerobot_action_rviz",
                arguments=["-d", rviz_config],
                remappings=[
                    ("/camera1/camera1/color/image_raw", camera1_topic),
                    ("/camera2/camera2/color/image_raw", camera2_topic),
                ],
                output="screen",
                condition=IfCondition(launch_rviz),
            ),
        ]
    )
