#include <string>

#include "franka_joint_safety_gateway/joint_preflight.hpp"
#include "gtest/gtest.h"

namespace gateway = franka_joint_safety_gateway;

namespace {

gateway::JointArray current_joints() {
  return {0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0};
}

gateway::PreflightSettings conservative_settings() {
  gateway::PreflightSettings settings;
  settings.joint_position_margin_rad = 0.03;
  settings.max_retiming_scale = 1.0;
  settings.motion_limits.max_position_step.fill(0.15);
  settings.motion_limits.max_velocity.fill(0.5);
  settings.motion_limits.max_acceleration.fill(3.0);
  return settings;
}

gateway::JointPlan two_point_plan() {
  gateway::JointPlan plan;
  plan.period_ns = 100'000'000;
  plan.waypoints.resize(2, current_joints());
  for (std::size_t joint = 0; joint < plan.waypoints[0].size(); ++joint) {
    plan.waypoints[0][joint] += 0.01;
    plan.waypoints[1][joint] += 0.02;
  }
  return plan;
}

} // namespace

TEST(JointPreflight, AcceptsHardLimitAndMotionCompliantPlan) {
  gateway::JointPreflightProvider provider(conservative_settings());
  const auto result = provider.validate(two_point_plan(), current_joints());

  EXPECT_TRUE(provider.ready());
  EXPECT_TRUE(result.accepted) << result.detail;
  EXPECT_EQ(result.period_ns, 100'000'000);
  EXPECT_NE(result.detail.find("hard-limit"), std::string::npos);
}

TEST(JointPreflight, RejectsFr3HardLimitViolation) {
  gateway::JointPreflightProvider provider(conservative_settings());
  auto plan = two_point_plan();
  plan.waypoints[0][3] = 0.0;

  const auto result = provider.validate(plan, current_joints());

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.failure_code, gateway::ResultCode::kRejectedJointLimit);
}

TEST(JointPreflight, RejectsPositionStepAndNativePeriodVelocity) {
  auto settings = conservative_settings();
  gateway::JointPreflightProvider provider(settings);
  auto plan = two_point_plan();
  plan.waypoints[0][0] = current_joints()[0] + 0.20;
  auto result = provider.validate(plan, current_joints());
  EXPECT_FALSE(result.accepted);
  EXPECT_NE(result.detail.find("position step"), std::string::npos);

  plan = two_point_plan();
  plan.period_ns = 33'333'333;
  plan.waypoints[0][0] = current_joints()[0] + 0.10;
  result = provider.validate(plan, current_joints());
  EXPECT_FALSE(result.accepted);
  EXPECT_NE(result.detail.find("velocity"), std::string::npos);
}

TEST(JointPreflight, RejectsAccelerationOrRetimesWhenAllowed) {
  auto settings = conservative_settings();
  settings.motion_limits.max_velocity.fill(10.0);
  settings.motion_limits.max_acceleration.fill(0.5);
  auto plan = two_point_plan();
  plan.waypoints[0] = current_joints();
  plan.waypoints[1] = current_joints();
  plan.waypoints[0][2] += 0.01;
  plan.waypoints[1][2] += 0.03;

  gateway::JointPreflightProvider strict_provider(settings);
  auto result = strict_provider.validate(plan, current_joints());
  EXPECT_FALSE(result.accepted);
  EXPECT_NE(result.detail.find("acceleration"), std::string::npos);

  settings.max_retiming_scale = 2.0;
  gateway::JointPreflightProvider retiming_provider(settings);
  result = retiming_provider.validate(plan, current_joints());
  ASSERT_TRUE(result.accepted) << result.detail;
  EXPECT_GT(result.period_ns, plan.period_ns);
  EXPECT_LE(result.period_ns, 2 * plan.period_ns);
}

TEST(JointPreflight, AlwaysAppliesConfiguredExecutionSlowdown) {
  auto settings = conservative_settings();
  settings.execution_slowdown_scale = 3.0;
  gateway::JointPreflightProvider provider(settings);

  const auto result = provider.validate(two_point_plan(), current_joints());

  ASSERT_TRUE(result.accepted) << result.detail;
  EXPECT_EQ(result.period_ns, 300'000'000);
  EXPECT_NE(result.detail.find("forced slowdown=3.000000x"), std::string::npos);
}

TEST(JointPreflight, AcceptsProjectsStyleStepToleranceButStillRetimesSpeed) {
  auto settings = conservative_settings();
  settings.execution_slowdown_scale = 3.0;
  settings.max_retiming_scale = 8.0;
  settings.motion_limits.max_position_step.fill(0.065);
  settings.motion_limits.max_velocity.fill(0.15);
  settings.motion_limits.max_acceleration.fill(0.5);
  gateway::JointPreflightProvider provider(settings);

  auto plan = two_point_plan();
  plan.period_ns = 33'333'333;
  plan.waypoints.resize(1, current_joints());
  plan.waypoints[0] = current_joints();
  plan.waypoints[0][5] += 0.060671;
  auto result = provider.validate(plan, current_joints());

  ASSERT_TRUE(result.accepted) << result.detail;
  EXPECT_GT(result.period_ns, 3 * plan.period_ns);

  plan.waypoints[0][5] = current_joints()[5] + 0.066;
  result = provider.validate(plan, current_joints());
  EXPECT_FALSE(result.accepted);
  EXPECT_NE(result.detail.find("position step"), std::string::npos);
}
