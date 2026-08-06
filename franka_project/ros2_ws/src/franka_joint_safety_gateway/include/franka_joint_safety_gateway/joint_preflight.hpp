#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <string>

#include "franka_cartesian_safety_gateway/execution.hpp"
#include "franka_joint_safety_gateway/validation.hpp"
#include "rclcpp/rclcpp.hpp"

namespace planning_scene_monitor {
class PlanningSceneMonitor;
using PlanningSceneMonitorPtr = std::shared_ptr<PlanningSceneMonitor>;
} // namespace planning_scene_monitor

namespace robot_model_loader {
class RobotModelLoader;
using RobotModelLoaderPtr = std::shared_ptr<RobotModelLoader>;
} // namespace robot_model_loader

namespace franka_joint_safety_gateway {

struct PreflightSettings {
  std::string group_name{"fr3_arm"};
  std::string planning_scene_topic{"/monitored_planning_scene"};
  std::string collision_object_topic{"/collision_object"};
  std::string planning_scene_world_topic{"/planning_scene_world"};
  bool require_environment_scene{true};
  double joint_position_margin_rad{0.03};
  double minimum_jacobian_singular_value{0.02};
  double maximum_jacobian_condition{150.0};
  double collision_interpolation_step_rad{0.03};
  // 1.0 preserves the policy-provided waypoint period. Native-period motion
  // violations are rejected instead of silently slowing the plan down.
  double max_retiming_scale{1.0};
  franka_cartesian_safety_gateway::JointMotionLimits motion_limits;
};

struct PreflightResult {
  bool accepted{false};
  ResultCode failure_code{ResultCode::kRejectedPreflightUnavailable};
  std::string detail;
  std::int64_t period_ns{0};
};

class JointPreflightProvider {
public:
  JointPreflightProvider(const rclcpp::Node::SharedPtr &node,
                         const PreflightSettings &settings);
  ~JointPreflightProvider();

  [[nodiscard]] bool ready() const;
  [[nodiscard]] bool planning_scene_ready() const;
  [[nodiscard]] const std::string &detail() const;
  [[nodiscard]] const std::array<std::string, 7> &joint_names() const;
  PreflightResult validate(const JointPlan &plan,
                           const JointArray &current_joints) const;

private:
  rclcpp::Node::SharedPtr node_;
  PreflightSettings settings_;
  robot_model_loader::RobotModelLoaderPtr model_loader_;
  planning_scene_monitor::PlanningSceneMonitorPtr scene_monitor_;
  std::array<std::string, 7> joint_names_{};
  std::atomic<bool> received_world_scene_{false};
  bool ready_{false};
  std::string detail_;
};

} // namespace franka_joint_safety_gateway
