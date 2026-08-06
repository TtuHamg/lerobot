import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def load_yaml(package_name, relative_path):
    path = os.path.join(get_package_share_directory(package_name), relative_path)
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def generate_launch_description():
    config = PathJoinSubstitution(
        [FindPackageShare("franka_joint_safety_gateway"), "config", "joint_safety_gateway.yaml"]
    )
    hand = LaunchConfiguration("hand")
    description_share = get_package_share_directory("franka_description")
    urdf = os.path.join(description_share, "robots", "fr3", "fr3.urdf.xacro")
    srdf = os.path.join(description_share, "robots", "fr3", "fr3.srdf.xacro")
    robot_description = ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"),
                " ",
                urdf,
                " robot_type:=fr3 ros2_control:=false hand:=",
                hand,
                " use_fake_hardware:=false",
            ]
        ),
        value_type=str,
    )
    robot_description_semantic = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", srdf, " robot_type:=fr3 hand:=", hand]),
        value_type=str,
    )
    joint_limits = load_yaml("franka_fr3_moveit_config", "config/fr3_joint_limits.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("hand", default_value="false"),
            DeclareLaunchArgument("enabled", default_value="false"),
            DeclareLaunchArgument("shadow", default_value="false"),
            DeclareLaunchArgument("preflight_only", default_value="false"),
            DeclareLaunchArgument("max_waypoints", default_value="32"),
            DeclareLaunchArgument(
                "chunk_topic", default_value="/lerobot/franka/joint_action_chunk"
            ),
            DeclareLaunchArgument("require_chunk_publisher", default_value="true"),
            DeclareLaunchArgument("hold_after_plan_completion", default_value="false"),
            DeclareLaunchArgument("require_robot_readiness", default_value="false"),
            DeclareLaunchArgument("require_controller_readiness", default_value="false"),
            DeclareLaunchArgument("require_environment_scene", default_value="false"),
            DeclareLaunchArgument("launch_readiness_monitor", default_value="false"),
            DeclareLaunchArgument("launch_site_scene", default_value="false"),
            DeclareLaunchArgument(
                "site_scene_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("franka_joint_safety_gateway"),
                        "config",
                        "site_collision_geometry.yaml",
                    ]
                ),
            ),
            Node(
                package="franka_joint_safety_gateway",
                executable="planning_scene_from_yaml.py",
                name="franka_joint_site_planning_scene",
                output="screen",
                condition=IfCondition(LaunchConfiguration("launch_site_scene")),
                parameters=[{"scene_config": LaunchConfiguration("site_scene_config")}],
            ),
            Node(
                package="franka_cartesian_safety_gateway",
                executable="readiness_monitor.py",
                name="franka_joint_safety_readiness_monitor",
                output="screen",
                condition=IfCondition(LaunchConfiguration("launch_readiness_monitor")),
            ),
            Node(
                package="franka_joint_safety_gateway",
                executable="gateway_node",
                name="franka_joint_safety_gateway",
                output="screen",
                parameters=[
                    config,
                    {"robot_description": robot_description},
                    {"robot_description_semantic": robot_description_semantic},
                    {"robot_description_planning": joint_limits},
                    {
                        "enabled": ParameterValue(
                            LaunchConfiguration("enabled"), value_type=bool
                        ),
                        "shadow": ParameterValue(
                            LaunchConfiguration("shadow"), value_type=bool
                        ),
                        "preflight_only": ParameterValue(
                            LaunchConfiguration("preflight_only"), value_type=bool
                        ),
                        "max_waypoints": ParameterValue(
                            LaunchConfiguration("max_waypoints"), value_type=int
                        ),
                        "chunk_topic": LaunchConfiguration("chunk_topic"),
                        "require_chunk_publisher": ParameterValue(
                            LaunchConfiguration("require_chunk_publisher"), value_type=bool
                        ),
                        "hold_after_plan_completion": ParameterValue(
                            LaunchConfiguration("hold_after_plan_completion"), value_type=bool
                        ),
                        "require_robot_readiness": ParameterValue(
                            LaunchConfiguration("require_robot_readiness"), value_type=bool
                        ),
                        "require_controller_readiness": ParameterValue(
                            LaunchConfiguration("require_controller_readiness"), value_type=bool
                        ),
                        "require_environment_scene": ParameterValue(
                            LaunchConfiguration("require_environment_scene"), value_type=bool
                        ),
                    },
                ],
            ),
        ]
    )
