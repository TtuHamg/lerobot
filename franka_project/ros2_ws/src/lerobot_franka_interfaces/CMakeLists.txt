cmake_minimum_required(VERSION 3.10)
project(lerobot_franka_interfaces)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

set(interface_files
  "msg/CartesianActionChunk.msg"
  "msg/CartesianActionChunkAck.msg"
  "msg/IkJointActionChunk.msg"
  "msg/SafetyGatewayStatus.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${interface_files}
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
