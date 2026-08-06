#include <limits>

#include "franka_joint_safety_gateway/validation.hpp"
#include "gtest/gtest.h"

namespace gateway = franka_joint_safety_gateway;

namespace {

gateway::JointPlan valid_plan() {
  gateway::JointPlan plan;
  plan.frame_id = "base";
  plan.start_ns = 1'000'000'000;
  plan.schema_version = 1;
  plan.session_id = "session";
  plan.plan_id = 1;
  plan.source_timestep = 4;
  plan.valid_until_ns = 2'000'000'000;
  plan.period_ns = 33'333'333;
  plan.timesteps = {5, 6};
  plan.waypoints.resize(2);
  plan.gripper = {0.0F, 0.0F};
  return plan;
}

} // namespace

TEST(JointValidation, AcceptsFiniteContiguousPlan) {
  const auto result = gateway::validate_plan(valid_plan(), 1'000'000'000,
                                             gateway::ValidationLimits{});
  EXPECT_TRUE(result.accepted()) << result.detail;
}

TEST(JointValidation, RejectsMalformedFlattenedResult) {
  auto plan = valid_plan();
  plan.waypoints.clear();
  const auto result =
      gateway::validate_plan(plan, 1'000'000'000, gateway::ValidationLimits{});
  EXPECT_EQ(result.code, gateway::ResultCode::kRejectedShape);
}

TEST(JointValidation, RejectsNonfiniteJoint) {
  auto plan = valid_plan();
  plan.waypoints[1][3] = std::numeric_limits<double>::quiet_NaN();
  const auto result =
      gateway::validate_plan(plan, 1'000'000'000, gateway::ValidationLimits{});
  EXPECT_EQ(result.code, gateway::ResultCode::kRejectedNonfinite);
}

TEST(JointValidation, AcceptsRobotiqGripperRangeEndpoints) {
  auto plan = valid_plan();
  plan.gripper = {0.0F, 0.8F};
  gateway::ValidationLimits limits;
  limits.validate_gripper_range = true;
  limits.gripper_min = 0.0F;
  limits.gripper_max = 0.8F;
  const auto result = gateway::validate_plan(plan, 1'000'000'000, limits);
  EXPECT_TRUE(result.accepted()) << result.detail;
}

TEST(JointValidation, RejectsGripperOutsideRobotiqRange) {
  auto plan = valid_plan();
  plan.gripper[1] = 0.81F;
  gateway::ValidationLimits limits;
  limits.validate_gripper_range = true;
  limits.gripper_min = 0.0F;
  limits.gripper_max = 0.8F;
  const auto result = gateway::validate_plan(plan, 1'000'000'000, limits);
  EXPECT_EQ(result.code, gateway::ResultCode::kRejectedGripper);
}

TEST(JointValidation, RejectsReplayWhilePlanRetained) {
  gateway::PlanStateMachine plans;
  auto first = valid_plan();
  ASSERT_TRUE(plans.accept(first, 1'000'000'000, gateway::ValidationLimits{})
                  .accepted());
  auto replay = first;
  replay.valid_until_ns = 3'000'000'000;
  const auto result =
      plans.accept(replay, 1'010'000'000, gateway::ValidationLimits{});
  EXPECT_EQ(result.code, gateway::ResultCode::kRejectedOrdering);
}
