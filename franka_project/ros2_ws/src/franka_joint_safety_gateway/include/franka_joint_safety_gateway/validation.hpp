#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "franka_cartesian_safety_gateway/execution.hpp"

namespace franka_joint_safety_gateway {

using JointArray = franka_cartesian_safety_gateway::JointArray;

struct JointPlan {
  std::string frame_id;
  std::int64_t start_ns{0};
  std::uint32_t schema_version{0};
  std::string session_id;
  std::uint64_t plan_id{0};
  std::int64_t source_timestep{0};
  std::int64_t valid_until_ns{0};
  std::int64_t period_ns{0};
  std::vector<std::int64_t> timesteps;
  std::vector<JointArray> waypoints;
  std::vector<float> gripper;
};

struct ValidationLimits {
  std::uint32_t schema_version{1};
  std::string frame_id{"base"};
  std::size_t max_waypoints{32};
  std::int64_t schedule_lateness_ns{100'000'000};
  std::int64_t settle_window_ns{250'000'000};
  bool validate_gripper_range{false};
  float gripper_min{0.0F};
  float gripper_max{1.0F};
};

enum class ResultCode : std::uint8_t {
  kAcceptedShadow = 0,
  kAcceptedForExecution = 1,
  kAcceptedPreflightOnly = 2,
  kRejectedSchema = 10,
  kRejectedFrame = 11,
  kRejectedShape = 12,
  kRejectedNonfinite = 13,
  kRejectedGripper = 15,
  kRejectedOrdering = 16,
  kRejectedExpired = 17,
  kRejectedSchedule = 18,
  kRejectedJointLimit = 19,
  kRejectedJointMotion = 20,
  kRejectedNotArmed = 21,
  kRejectedStateStale = 22,
  kRejectedPreflightUnavailable = 23,
  kRejectedCollision = 24,
  kRejectedSingularity = 25,
};

struct ValidationResult {
  ResultCode code{ResultCode::kRejectedShape};
  std::string detail;

  [[nodiscard]] bool accepted() const {
    return code == ResultCode::kAcceptedShadow ||
           code == ResultCode::kAcceptedForExecution ||
           code == ResultCode::kAcceptedPreflightOnly;
  }
};

ValidationResult validate_plan(const JointPlan &plan, std::int64_t now_ns,
                               const ValidationLimits &limits);

class PlanStateMachine {
public:
  [[nodiscard]] ValidationResult
  validate_candidate(const JointPlan &candidate, std::int64_t now_ns,
                     const ValidationLimits &limits) const;
  ValidationResult accept(const JointPlan &candidate, std::int64_t now_ns,
                          const ValidationLimits &limits);
  void reset_expired(std::int64_t now_ns, const ValidationLimits &limits);
  void clear();
  [[nodiscard]] const std::optional<JointPlan> &active_plan() const;

private:
  std::optional<JointPlan> active_plan_;
};

} // namespace franka_joint_safety_gateway
