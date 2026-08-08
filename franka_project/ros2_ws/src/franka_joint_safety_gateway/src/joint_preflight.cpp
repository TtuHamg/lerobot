#include "franka_joint_safety_gateway/joint_preflight.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace franka_joint_safety_gateway {
namespace {

// Franka FR3 joint position limits [rad], canonical order fr3_joint1..7.
constexpr std::array<double, 7> kFr3Lower = {-2.7437, -1.7837, -2.9007, -3.0421,
                                             -2.8065, 0.5445,  -3.0159};
constexpr std::array<double, 7> kFr3Upper = {2.7437, 1.7837, 2.9007, -0.1518,
                                             2.8065, 4.5169, 3.0159};

constexpr std::array<const char *, 7> kFr3JointNames = {
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7"};

} // namespace

JointPreflightProvider::JointPreflightProvider(
    const PreflightSettings &settings)
    : settings_(settings) {
  for (std::size_t index = 0; index < joint_names_.size(); ++index) {
    joint_names_[index] = kFr3JointNames[index];
  }
  std::copy(kFr3Lower.begin(), kFr3Lower.end(), lower_limits_.begin());
  std::copy(kFr3Upper.begin(), kFr3Upper.end(), upper_limits_.begin());
  ready_ = true;
  detail_ = "lightweight joint preflight ready (hard-limit + velocity/"
            "acceleration/step checks; collision checks disabled)";
}

JointPreflightProvider::~JointPreflightProvider() = default;

bool JointPreflightProvider::ready() const { return ready_; }

const std::string &JointPreflightProvider::detail() const { return detail_; }

const std::array<std::string, 7> &JointPreflightProvider::joint_names() const {
  return joint_names_;
}

PreflightResult
JointPreflightProvider::validate(const JointPlan &plan,
                                 const JointArray &current_joints) const {
  PreflightResult result;
  const double margin = settings_.joint_position_margin_rad;

  const auto within_limits = [&](const JointArray &joints,
                                 std::string *detail) -> bool {
    for (std::size_t index = 0; index < joints.size(); ++index) {
      const double value = joints[index];
      if (!std::isfinite(value) || value < lower_limits_[index] + margin ||
          value > upper_limits_[index] - margin) {
        *detail = "fr3_joint" + std::to_string(index + 1) +
                  " violates the configured hard-limit margin";
        return false;
      }
    }
    return true;
  };

  std::string check_detail;
  if (!within_limits(current_joints, &check_detail)) {
    result.failure_code = ResultCode::kRejectedJointLimit;
    result.detail = "current state: " + check_detail;
    return result;
  }
  for (std::size_t point = 0; point < plan.waypoints.size(); ++point) {
    if (!within_limits(plan.waypoints[point], &check_detail)) {
      result.failure_code = ResultCode::kRejectedJointLimit;
      result.detail = "waypoint " + std::to_string(point) + ": " + check_detail;
      return result;
    }
  }

  double max_start_delta = 0.0;
  double max_step = 0.0;
  double max_excursion = 0.0;
  JointArray previous = current_joints;
  for (std::size_t point = 0; point < plan.waypoints.size(); ++point) {
    for (std::size_t joint = 0; joint < current_joints.size(); ++joint) {
      if (point == 0) {
        max_start_delta =
            std::max(max_start_delta, std::abs(plan.waypoints[point][joint] -
                                               current_joints[joint]));
      }
      max_step = std::max(
          max_step, std::abs(plan.waypoints[point][joint] - previous[joint]));
      max_excursion =
          std::max(max_excursion, std::abs(plan.waypoints[point][joint] -
                                           current_joints[joint]));
    }
    previous = plan.waypoints[point];
  }
  double final_delta = 0.0;
  for (std::size_t joint = 0; joint < current_joints.size(); ++joint) {
    final_delta = std::max(final_delta, std::abs(plan.waypoints.back()[joint] -
                                                 current_joints[joint]));
  }
  if (max_excursion > settings_.max_plan_excursion_rad) {
    result.failure_code = ResultCode::kRejectedJointMotion;
    result.detail = "plan excursion " + std::to_string(max_excursion) +
                    " rad exceeds limit " +
                    std::to_string(settings_.max_plan_excursion_rad) +
                    " rad relative to the measured start state";
    return result;
  }

  const long double scaled_period = static_cast<long double>(plan.period_ns) *
                                    settings_.execution_slowdown_scale;
  if (!std::isfinite(scaled_period) ||
      scaled_period >
          static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
    result.failure_code = ResultCode::kRejectedJointMotion;
    result.detail = "forced execution slowdown overflows the plan period";
    return result;
  }
  const auto base_period_ns =
      static_cast<std::int64_t>(std::ceil(scaled_period));

  // Velocity, acceleration, per-step and retiming checks. Start from the
  // mandatory slowdown period, then slow further if a motion limit requires it.
  const auto timing = franka_cartesian_safety_gateway::retime_joint_motion(
      current_joints, plan.waypoints, base_period_ns,
      settings_.max_retiming_scale, settings_.motion_limits);
  if (!timing.valid) {
    result.failure_code = ResultCode::kRejectedJointMotion;
    result.detail = timing.detail;
    return result;
  }

  result.accepted = true;
  result.period_ns = timing.period_ns;
  result.detail =
      std::string("joint hard-limit, step, velocity and acceleration checks "
                  "passed; requested period=") +
      std::to_string(plan.period_ns) + " ns, forced slowdown=" +
      std::to_string(settings_.execution_slowdown_scale) +
      "x, effective period=" + std::to_string(timing.period_ns) + " ns" +
      (timing.period_ns > base_period_ns ? " (motion-limit retimed)" : "") +
      ", max_start_delta=" + std::to_string(max_start_delta) +
      " rad, max_step=" + std::to_string(max_step) +
      " rad, max_excursion=" + std::to_string(max_excursion) +
      " rad, final_delta=" + std::to_string(final_delta) + " rad";
  return result;
}

} // namespace franka_joint_safety_gateway
