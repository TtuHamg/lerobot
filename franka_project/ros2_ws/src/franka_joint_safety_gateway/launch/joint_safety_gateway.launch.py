from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution(
        [FindPackageShare("franka_joint_safety_gateway"), "config", "joint_safety_gateway.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("enabled", default_value="false"),
            DeclareLaunchArgument("shadow", default_value="false"),
            DeclareLaunchArgument("preflight_only", default_value="false"),
            DeclareLaunchArgument("max_waypoints", default_value="32"),
            DeclareLaunchArgument(
                "chunk_topic", default_value="/lerobot/franka/joint_action_chunk"
            ),
            DeclareLaunchArgument("require_chunk_publisher", default_value="true"),
            DeclareLaunchArgument("hold_after_plan_completion", default_value="false"),
            Node(
                package="franka_joint_safety_gateway",
                executable="gateway_node",
                name="franka_joint_safety_gateway",
                output="screen",
                parameters=[
                    config,
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
                    },
                ],
            ),
        ]
    )
