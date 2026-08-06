#include "franka_joint_safety_gateway/joint_preflight.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>

#include "moveit/collision_detection/collision_common.hpp"
#include "moveit/planning_scene_monitor/planning_scene_monitor.hpp"
#include "moveit/robot_model_loader/robot_model_loader.hpp"
#include "moveit/robot_state/robot_state.hpp"
#include <Eigen/SVD>

namespace franka_joint_safety_gateway {
namespace {

bool within_position_margins(const moveit::core::RobotModel &model,
                             const std::vector<std::string> &names,
                             const JointArray &values, double margin,
                             std::string *detail) {
  for (std::size_t index = 0; index < names.size(); ++index) {
    const auto &bounds = model.getVariableBounds(names[index]);
    if (!bounds.position_bounded_ ||
        values[index] < bounds.min_position_ + margin ||
        values[index] > bounds.max_position_ - margin) {
      *detail = names[index] + " violates the configured hard-limit margin";
      return false;
    }
  }
  return true;
}

bool jacobian_is_well_conditioned(moveit::core::RobotState &state,
                                  const moveit::core::JointModelGroup *group,
                                  double minimum_singular_value,
                                  double maximum_condition,
                                  std::string *detail) {
  const Eigen::MatrixXd jacobian = state.getJacobian(group);
  if (jacobian.rows() != 6 || jacobian.cols() != 7 || !jacobian.allFinite()) {
    *detail = "Jacobian is unavailable or non-finite";
    return false;
  }
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto singular = svd.singularValues();
  if (singular.size() < 6 || singular[0] <= 0.0) {
    *detail = "Jacobian singular values are invalid";
    return false;
  }
  const double smallest = singular[singular.size() - 1];
  const double condition =
      singular[0] / std::max(smallest, std::numeric_limits<double>::min());
  if (smallest < minimum_singular_value || condition > maximum_condition) {
    *detail =
        "Jacobian quality rejected (sigma_min=" + std::to_string(smallest) +
        ", condition=" + std::to_string(condition) + ")";
    return false;
  }
  return true;
}

bool collision_free(const planning_scene::PlanningSceneConstPtr &scene,
                    moveit::core::RobotState &state,
                    const std::string &group_name, std::string *detail) {
  collision_detection::CollisionRequest request;
  request.group_name = group_name;
  collision_detection::CollisionResult self_result;
  scene->checkSelfCollision(request, self_result, state);
  if (self_result.collision) {
    *detail = "self-collision detected";
    return false;
  }
  collision_detection::CollisionResult full_result;
  scene->checkCollision(request, full_result, state);
  if (full_result.collision) {
    *detail = "robot/world collision detected";
    return false;
  }
  return true;
}

} // namespace

JointPreflightProvider::JointPreflightProvider(
    const rclcpp::Node::SharedPtr &node, const PreflightSettings &settings)
    : node_(node), settings_(settings) {
  try {
    robot_model_loader::RobotModelLoader::Options options("robot_description");
    options.load_kinematics_solvers = false;
    model_loader_ =
        std::make_shared<robot_model_loader::RobotModelLoader>(node_, options);
    const auto model = model_loader_->getModel();
    if (!model) {
      detail_ =
          "robot_description/robot_description_semantic could not be loaded";
      return;
    }
    const auto *group = model->getJointModelGroup(settings_.group_name);
    if (group == nullptr || group->getVariableCount() != 7) {
      detail_ = "MoveIt arm group must contain exactly seven variables";
      return;
    }
    const auto &variables = group->getVariableNames();
    for (std::size_t index = 0; index < joint_names_.size(); ++index) {
      joint_names_[index] = variables[index];
      const auto &bounds = model->getVariableBounds(variables[index]);
      if (!bounds.position_bounded_) {
        detail_ = variables[index] + " has no finite position bounds";
        return;
      }
      if (bounds.velocity_bounded_) {
        settings_.motion_limits.max_velocity[index] =
            std::min(settings_.motion_limits.max_velocity[index],
                     std::min(std::abs(bounds.min_velocity_),
                              std::abs(bounds.max_velocity_)));
      }
      if (bounds.acceleration_bounded_) {
        settings_.motion_limits.max_acceleration[index] =
            std::min(settings_.motion_limits.max_acceleration[index],
                     std::min(std::abs(bounds.min_acceleration_),
                              std::abs(bounds.max_acceleration_)));
      }
    }
    scene_monitor_ =
        std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
            node_, model_loader_, "franka_joint_safety_preflight");
    if (!scene_monitor_ || !scene_monitor_->getPlanningScene()) {
      detail_ = "MoveIt PlanningSceneMonitor could not be created";
      return;
    }
    scene_monitor_->addUpdateCallback(
        [this](planning_scene_monitor::PlanningSceneMonitor::SceneUpdateType
                   update) {
          if ((static_cast<int>(update) &
               static_cast<int>(planning_scene_monitor::PlanningSceneMonitor::
                                    UPDATE_GEOMETRY)) != 0) {
            received_world_scene_.store(true, std::memory_order_release);
          }
        });
    scene_monitor_->startSceneMonitor(settings_.planning_scene_topic);
    scene_monitor_->startWorldGeometryMonitor(
        settings_.collision_object_topic, settings_.planning_scene_world_topic,
        false);
    ready_ = true;
    detail_ = settings_.require_environment_scene
                  ? "MoveIt joint bounds/collision checker loaded; awaiting "
                    "world scene"
                  : "MoveIt joint bounds/self-collision checker loaded; world "
                    "scene disabled";
  } catch (const std::exception &error) {
    detail_ = std::string("MoveIt joint preflight initialization failed: ") +
              error.what();
  }
}

JointPreflightProvider::~JointPreflightProvider() {
  if (scene_monitor_) {
    scene_monitor_->stopSceneMonitor();
    scene_monitor_->stopWorldGeometryMonitor();
    scene_monitor_->clearUpdateCallbacks();
    scene_monitor_.reset();
  }
  model_loader_.reset();
}

bool JointPreflightProvider::ready() const { return ready_; }

bool JointPreflightProvider::planning_scene_ready() const {
  if (!ready_) {
    return false;
  }
  if (!settings_.require_environment_scene) {
    planning_scene_monitor::LockedPlanningSceneRO scene(scene_monitor_);
    return static_cast<bool>(scene);
  }
  if (!received_world_scene_.load(std::memory_order_acquire) ||
      node_->count_publishers(settings_.planning_scene_topic) == 0) {
    return false;
  }
  planning_scene_monitor::LockedPlanningSceneRO scene(scene_monitor_);
  return scene && scene->getWorld() && scene->getWorld()->size() > 0;
}

const std::string &JointPreflightProvider::detail() const { return detail_; }

const std::array<std::string, 7> &JointPreflightProvider::joint_names() const {
  return joint_names_;
}

PreflightResult
JointPreflightProvider::validate(const JointPlan &plan,
                                 const JointArray &current_joints) const {
  PreflightResult result;
  if (!ready_ || !planning_scene_ready()) {
    result.detail = "MoveIt model or monitored world scene is unavailable";
    return result;
  }
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(scene_monitor_);
  const auto scene =
      static_cast<planning_scene::PlanningSceneConstPtr>(locked_scene);
  const auto model = model_loader_->getModel();
  const auto *group = model->getJointModelGroup(settings_.group_name);
  if (!scene || group == nullptr) {
    result.detail = "planning scene or arm group became unavailable";
    return result;
  }

  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  state.setJointGroupPositions(group, current_joints.data());
  state.update();
  std::string check_detail;
  if (!within_position_margins(
          *model, group->getVariableNames(), current_joints,
          settings_.joint_position_margin_rad, &check_detail)) {
    result.failure_code = ResultCode::kRejectedJointLimit;
    result.detail = "current state: " + check_detail;
    return result;
  }
  if (!collision_free(scene, state, settings_.group_name, &check_detail)) {
    result.failure_code = ResultCode::kRejectedCollision;
    result.detail = "current state: " + check_detail;
    return result;
  }

  JointArray previous = current_joints;
  for (std::size_t point = 0; point < plan.waypoints.size(); ++point) {
    const auto &target = plan.waypoints[point];
    if (!within_position_margins(*model, group->getVariableNames(), target,
                                 settings_.joint_position_margin_rad,
                                 &check_detail)) {
      result.failure_code = ResultCode::kRejectedJointLimit;
      result.detail = "waypoint " + std::to_string(point) + ": " + check_detail;
      return result;
    }
    double max_delta = 0.0;
    for (std::size_t joint = 0; joint < target.size(); ++joint) {
      max_delta =
          std::max(max_delta, std::abs(target[joint] - previous[joint]));
    }
    const auto samples = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(
               max_delta / settings_.collision_interpolation_step_rad)));
    for (std::size_t sample = 1; sample <= samples; ++sample) {
      JointArray intermediate{};
      const double alpha =
          static_cast<double>(sample) / static_cast<double>(samples);
      for (std::size_t joint = 0; joint < target.size(); ++joint) {
        intermediate[joint] =
            previous[joint] + alpha * (target[joint] - previous[joint]);
      }
      state.setJointGroupPositions(group, intermediate.data());
      state.update();
      if (!collision_free(scene, state, settings_.group_name, &check_detail)) {
        result.failure_code = ResultCode::kRejectedCollision;
        result.detail = "collision before waypoint " + std::to_string(point) +
                        ": " + check_detail;
        return result;
      }
    }
    if (!jacobian_is_well_conditioned(
            state, group, settings_.minimum_jacobian_singular_value,
            settings_.maximum_jacobian_condition, &check_detail)) {
      result.failure_code = ResultCode::kRejectedSingularity;
      result.detail = "waypoint " + std::to_string(point) + ": " + check_detail;
      return result;
    }
    previous = target;
  }

  const auto timing = franka_cartesian_safety_gateway::retime_joint_motion(
      current_joints, plan.waypoints, plan.period_ns,
      settings_.max_retiming_scale, settings_.motion_limits);
  if (!timing.valid) {
    result.failure_code = ResultCode::kRejectedJointMotion;
    result.detail = timing.detail;
    return result;
  }
  result.accepted = true;
  result.period_ns = timing.period_ns;
  result.detail = "joint bounds, Jacobian, full-path collision, step, velocity "
                  "and acceleration checks passed; " +
                  std::string(timing.period_ns == plan.period_ns ? "native period="
                                                      : "retimed period=") +
                  std::to_string(timing.period_ns) + " ns";
  return result;
}

} // namespace franka_joint_safety_gateway
