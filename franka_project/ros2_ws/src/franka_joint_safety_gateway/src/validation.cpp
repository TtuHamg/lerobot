#include "franka_joint_safety_gateway/validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace franka_joint_safety_gateway {
namespace {

ValidationResult reject(ResultCode code, const std::string &detail) {
  return {code, detail};
}

bool schedule_time(std::int64_t start_ns, std::int64_t period_ns,
                   std::size_t count, std::int64_t *result_ns) {
  if (period_ns <= 0 ||
      count > static_cast<std::size_t>(
                  std::numeric_limits<std::int64_t>::max() / period_ns)) {
    return false;
  }
  const auto offset = static_cast<std::int64_t>(count) * period_ns;
  if (start_ns > std::numeric_limits<std::int64_t>::max() - offset) {
    return false;
  }
  *result_ns = start_ns + offset;
  return true;
}

std::int64_t retention_deadline(const JointPlan &plan,
                                const ValidationLimits &limits) {
  std::int64_t final_target_ns = 0;
  if (plan.waypoints.empty() || limits.settle_window_ns < 0 ||
      !schedule_time(plan.start_ns, plan.period_ns, plan.waypoints.size(),
                     &final_target_ns) ||
      final_target_ns >
          std::numeric_limits<std::int64_t>::max() - limits.settle_window_ns) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return final_target_ns + limits.settle_window_ns;
}

} // namespace

ValidationResult validate_plan(const JointPlan &plan, std::int64_t now_ns,
                               const ValidationLimits &limits) {
  if (plan.schema_version != limits.schema_version) {
    return reject(ResultCode::kRejectedSchema, "unsupported schema_version");
  }
  if (plan.frame_id != limits.frame_id) {
    return reject(ResultCode::kRejectedFrame, "unexpected joint command frame");
  }
  if (plan.session_id.empty()) {
    return reject(ResultCode::kRejectedShape, "session_id is empty");
  }
  const auto count = plan.waypoints.size();
  if (count == 0 || count > limits.max_waypoints ||
      plan.timesteps.size() != count || plan.gripper.size() != count) {
    return reject(ResultCode::kRejectedShape,
                  "waypoint arrays have invalid sizes");
  }
  if (plan.valid_until_ns <= now_ns) {
    return reject(ResultCode::kRejectedExpired,
                  "plan transport TTL has expired");
  }
  if (plan.period_ns <= 0) {
    return reject(ResultCode::kRejectedSchedule, "period must be positive");
  }
  if (plan.start_ns <= 0 ||
      plan.start_ns + limits.schedule_lateness_ns < now_ns) {
    return reject(ResultCode::kRejectedSchedule, "plan start is already late");
  }
  if (plan.timesteps.front() < plan.source_timestep) {
    return reject(ResultCode::kRejectedOrdering,
                  "first timestep precedes source_timestep");
  }

  std::int64_t final_target_ns = 0;
  if (!schedule_time(plan.start_ns, plan.period_ns, count, &final_target_ns)) {
    return reject(ResultCode::kRejectedSchedule, "waypoint schedule overflows");
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (index > 0 && plan.timesteps[index] != plan.timesteps[index - 1] + 1) {
      return reject(ResultCode::kRejectedOrdering,
                    "timesteps are not contiguous");
    }
    if (!std::all_of(plan.waypoints[index].begin(), plan.waypoints[index].end(),
                     [](double value) { return std::isfinite(value); })) {
      return reject(ResultCode::kRejectedNonfinite,
                    "joint target contains a non-finite value");
    }
    const auto gripper = plan.gripper[index];
    if (!std::isfinite(gripper)) {
      return reject(ResultCode::kRejectedGripper,
                    "gripper target is non-finite");
    }
    if (limits.validate_gripper_range &&
        (gripper < limits.gripper_min || gripper > limits.gripper_max)) {
      return reject(ResultCode::kRejectedGripper,
                    "gripper target at waypoint " + std::to_string(index) +
                        " is " + std::to_string(gripper) +
                        ", outside configured range [" +
                        std::to_string(limits.gripper_min) + ", " +
                        std::to_string(limits.gripper_max) + "]");
    }
  }
  return {ResultCode::kAcceptedShadow, "joint plan wire contract validated"};
}

ValidationResult
PlanStateMachine::validate_candidate(const JointPlan &candidate,
                                     std::int64_t now_ns,
                                     const ValidationLimits &limits) const {
  const auto validation = validate_plan(candidate, now_ns, limits);
  if (!validation.accepted()) {
    return validation;
  }
  if (active_plan_.has_value() &&
      retention_deadline(*active_plan_, limits) > now_ns) {
    if (candidate.session_id != active_plan_->session_id) {
      return reject(ResultCode::kRejectedOrdering,
                    "session cannot change while a retained plan is active");
    }
    if (candidate.plan_id <= active_plan_->plan_id) {
      return reject(ResultCode::kRejectedOrdering, "plan_id is not increasing");
    }
  }
  return validation;
}

ValidationResult PlanStateMachine::accept(const JointPlan &candidate,
                                          std::int64_t now_ns,
                                          const ValidationLimits &limits) {
  const auto validation = validate_candidate(candidate, now_ns, limits);
  if (validation.accepted()) {
    active_plan_ = candidate;
  }
  return validation;
}

void PlanStateMachine::reset_expired(std::int64_t now_ns,
                                     const ValidationLimits &limits) {
  if (active_plan_.has_value() &&
      retention_deadline(*active_plan_, limits) <= now_ns) {
    active_plan_.reset();
  }
}

void PlanStateMachine::clear() { active_plan_.reset(); }

const std::optional<JointPlan> &PlanStateMachine::active_plan() const {
  return active_plan_;
}

} // namespace franka_joint_safety_gateway
