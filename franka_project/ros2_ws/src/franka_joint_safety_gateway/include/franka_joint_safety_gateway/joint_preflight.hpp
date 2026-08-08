#pragma once

#include <array>
#include <string>

#include "franka_cartesian_safety_gateway/execution.hpp"
#include "franka_joint_safety_gateway/validation.hpp"

namespace franka_joint_safety_gateway {

struct PreflightSettings {
  double joint_position_margin_rad{0.03};
  double execution_slowdown_scale{1.0};
  double max_retiming_scale{1.0};
  double max_plan_excursion_rad{0.30};
  franka_cartesian_safety_gateway::JointMotionLimits motion_limits;
};

struct PreflightResult {
  bool accepted{false};
  ResultCode failure_code{ResultCode::kRejectedPreflightUnavailable};
  std::string detail;
  std::int64_t period_ns{0};
};

// Lightweight joint-space preflight.
//
// It enforces joint position hard-limits (with margin) plus the velocity /
// acceleration / step / retiming checks. It deliberately does NOT load MoveIt:
// there is no PlanningSceneMonitor, no collision checking, and no
// Jacobian/singularity check.
//
// NOTE: this preflight does NOT provide collision safety. Self/world collision
// avoidance must be guaranteed by other means (workspace clearing, low-speed
// limits, controller-local fail-safe behavior, and the operator's e-stop).
class JointPreflightProvider {
public:
  explicit JointPreflightProvider(const PreflightSettings &settings);
  ~JointPreflightProvider();

  [[nodiscard]] bool ready() const;
  [[nodiscard]] const std::string &detail() const;
  [[nodiscard]] const std::array<std::string, 7> &joint_names() const;
  PreflightResult validate(const JointPlan &plan,
                           const JointArray &current_joints) const;

private:
  PreflightSettings settings_;
  std::array<std::string, 7> joint_names_{};
  JointArray lower_limits_{};
  JointArray upper_limits_{};
  bool ready_{true};
  std::string detail_;
};

} // namespace franka_joint_safety_gateway
