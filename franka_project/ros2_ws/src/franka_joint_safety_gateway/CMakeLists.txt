cmake_minimum_required(VERSION 3.10)
project(franka_joint_safety_gateway)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(franka_cartesian_safety_gateway REQUIRED)
find_package(franka_safety_interfaces REQUIRED)
find_package(lerobot_franka_interfaces REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_library(SYSTEM_CURL NAMES libcurl.so.4
  PATHS /usr/lib/x86_64-linux-gnu
  NO_DEFAULT_PATH REQUIRED)

add_library(joint_validation src/validation.cpp)
target_include_directories(joint_validation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(joint_validation franka_cartesian_safety_gateway)

add_library(joint_preflight src/joint_preflight.cpp)
target_include_directories(joint_preflight PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(joint_preflight joint_validation)
ament_target_dependencies(joint_preflight
  franka_cartesian_safety_gateway moveit_core moveit_ros_planning rclcpp)

add_executable(gateway_node src/gateway_node.cpp)
target_link_libraries(gateway_node joint_preflight joint_validation)
ament_target_dependencies(gateway_node
  control_msgs franka_cartesian_safety_gateway franka_safety_interfaces
  lerobot_franka_interfaces moveit_core moveit_ros_planning rclcpp
  rclcpp_action sensor_msgs std_msgs std_srvs)
target_link_libraries(gateway_node ${SYSTEM_CURL})

install(TARGETS joint_validation joint_preflight gateway_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(PROGRAMS scripts/planning_scene_from_yaml.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_joint_validation test/test_validation.cpp)
  target_link_libraries(test_joint_validation joint_validation)
  target_include_directories(test_joint_validation PRIVATE include)
endif()

ament_export_include_directories(include)
ament_export_libraries(joint_validation joint_preflight)
ament_package()
