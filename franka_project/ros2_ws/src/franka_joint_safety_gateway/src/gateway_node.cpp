#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "builtin_interfaces/msg/duration.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "control_msgs/action/gripper_command.hpp"
#include "franka_cartesian_safety_gateway/execution.hpp"
#include "franka_joint_safety_gateway/joint_preflight.hpp"
#include "franka_joint_safety_gateway/validation.hpp"
#include "franka_safety_interfaces/msg/safe_joint_command.hpp"
#include "franka_safety_interfaces/msg/safety_command_feedback.hpp"
#include "lerobot_franka_interfaces/msg/joint_action_chunk.hpp"
#include "lerobot_franka_interfaces/msg/joint_action_chunk_ack.hpp"
#include "lerobot_franka_interfaces/msg/safety_gateway_status.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/set_bool.hpp"

namespace franka_joint_safety_gateway {
namespace {

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
constexpr std::size_t kJointCount = 7;
using GatewayState = franka_cartesian_safety_gateway::GatewayState;
using SafetyStateMachine = franka_cartesian_safety_gateway::SafetyStateMachine;
using ArmGates = franka_cartesian_safety_gateway::ArmGates;
using GripperAction = control_msgs::action::GripperCommand;
using GripperGoalHandle = rclcpp_action::ClientGoalHandle<GripperAction>;

std::int64_t to_nanoseconds(const builtin_interfaces::msg::Time &value) {
  return static_cast<std::int64_t>(value.sec) * kNanosecondsPerSecond +
         value.nanosec;
}

std::optional<std::int64_t>
to_nanoseconds(const builtin_interfaces::msg::Duration &value) {
  if (value.sec < 0) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(value.sec) * kNanosecondsPerSecond +
         value.nanosec;
}

builtin_interfaces::msg::Time from_nanoseconds(std::int64_t value) {
  builtin_interfaces::msg::Time result;
  if (value <= 0) {
    return result;
  }
  result.sec = static_cast<std::int32_t>(value / kNanosecondsPerSecond);
  result.nanosec = static_cast<std::uint32_t>(value % kNanosecondsPerSecond);
  return result;
}

builtin_interfaces::msg::Duration
duration_from_nanoseconds(std::int64_t value) {
  builtin_interfaces::msg::Duration result;
  value = std::max<std::int64_t>(0, value);
  result.sec = static_cast<std::int32_t>(std::min<std::int64_t>(
      value / kNanosecondsPerSecond, std::numeric_limits<std::int32_t>::max()));
  result.nanosec = static_cast<std::uint32_t>(value % kNanosecondsPerSecond);
  return result;
}

} // namespace

class GatewayNode final : public rclcpp::Node {
public:
  explicit GatewayNode(const rclcpp::NodeOptions &options)
      : Node("franka_joint_safety_gateway", options),
        enabled_(parameter<bool>("enabled", false)),
        shadow_(parameter<bool>("shadow", true)),
        preflight_only_(parameter<bool>("preflight_only", false)),
        safety_state_(enabled_ || preflight_only_, shadow_ || preflight_only_) {
    state_timeout_ns_ =
        seconds_to_ns(parameter<double>("state_timeout_s", 0.15));
    state_stamp_timeout_ns_ =
        seconds_to_ns(parameter<double>("state_stamp_timeout_s", 0.5));
    readiness_timeout_ns_ =
        seconds_to_ns(parameter<double>("readiness_timeout_s", 0.5));
    command_deadline_ns_ =
        seconds_to_ns(parameter<double>("command_deadline_s", 0.030));
    publisher_watchdog_ns_ =
        seconds_to_ns(parameter<double>("publisher_watchdog_s", 0.030));
    settle_window_ns_ =
        seconds_to_ns(parameter<double>("settle_window_s", 0.25));
    feedback_timeout_ns_ =
        seconds_to_ns(parameter<double>("applied_feedback_timeout_s", 0.05));
    feedback_grace_ns_ =
        seconds_to_ns(parameter<double>("applied_feedback_grace_s", 0.10));
    tracking_grace_ns_ =
        seconds_to_ns(parameter<double>("tracking_grace_s", 0.5));
    command_period_ms_ = parameter<int>("command_period_ms", 5);
    max_state_drift_rad_ =
        parameter<double>("max_preflight_state_drift_rad", 0.01);
    max_tracking_error_rad_ =
        parameter<double>("max_joint_tracking_error_rad", 0.25);
    max_applied_sequence_lag_ = static_cast<std::uint64_t>(
        parameter<int>("max_applied_sequence_lag", 20));
    require_command_subscriber_ =
        parameter<bool>("require_command_subscriber", true);
    require_unique_command_publisher_ =
        parameter<bool>("require_unique_command_publisher", true);
    require_chunk_publisher_ = parameter<bool>("require_chunk_publisher", true);
    require_robot_readiness_ = parameter<bool>("require_robot_readiness", true);
    require_controller_readiness_ =
        parameter<bool>("require_controller_readiness", true);
    hold_after_plan_completion_ =
        parameter<bool>("hold_after_plan_completion", false);
    execute_gripper_ = parameter<bool>("execute_gripper", true);
    gripper_action_name_ = parameter<std::string>(
        "gripper_action_name", "/gripper/robotiq_gripper_controller/gripper_cmd");
    gripper_max_effort_ = parameter<double>("gripper_max_effort", 16.0);
    gripper_command_deadband_ =
        parameter<double>("gripper_command_deadband", 0.02);
    gripper_min_command_interval_ns_ = seconds_to_ns(
        parameter<double>("gripper_min_command_interval_s", 0.15));

    limits_.schema_version =
        static_cast<std::uint32_t>(parameter<int>("schema_version", 1));
    limits_.frame_id = parameter<std::string>("frame_id", "base");
    limits_.max_waypoints =
        static_cast<std::size_t>(parameter<int>("max_waypoints", 32));
    limits_.schedule_lateness_ns =
        seconds_to_ns(parameter<double>("schedule_lateness_s", 0.10));
    limits_.settle_window_ns = settle_window_ns_;
    limits_.validate_gripper_range =
        parameter<bool>("validate_gripper_range", false);
    limits_.gripper_min =
        static_cast<float>(parameter<double>("gripper_min", 0.0));
    limits_.gripper_max =
        static_cast<float>(parameter<double>("gripper_max", 1.0));

    preflight_settings_.group_name =
        parameter<std::string>("moveit_group", "fr3_arm");
    preflight_settings_.planning_scene_topic = parameter<std::string>(
        "planning_scene_topic", "/monitored_planning_scene");
    preflight_settings_.collision_object_topic =
        parameter<std::string>("collision_object_topic", "/collision_object");
    preflight_settings_.planning_scene_world_topic = parameter<std::string>(
        "planning_scene_world_topic", "/planning_scene_world");
    preflight_settings_.require_environment_scene =
        parameter<bool>("require_environment_scene", true);
    preflight_settings_.joint_position_margin_rad =
        parameter<double>("joint_position_margin_rad", 0.03);
    preflight_settings_.minimum_jacobian_singular_value =
        parameter<double>("minimum_jacobian_singular_value", 0.02);
    preflight_settings_.maximum_jacobian_condition =
        parameter<double>("maximum_jacobian_condition", 150.0);
    preflight_settings_.collision_interpolation_step_rad =
        parameter<double>("collision_interpolation_step_rad", 0.03);
    preflight_settings_.max_retiming_scale =
        parameter<double>("max_retiming_scale", 1.0);
    preflight_settings_.motion_limits.max_position_step = read_joint_array(
        "max_joint_step_rad", {0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08});
    preflight_settings_.motion_limits.max_velocity = read_joint_array(
        "max_joint_velocity_rad_s", {0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15});
    preflight_settings_.motion_limits.max_acceleration = read_joint_array(
        "max_joint_acceleration_rad_s2", {0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75});

    joint_state_topic_ =
        parameter<std::string>("joint_state_topic", "/franka/joint_states");
    chunk_topic_ = parameter<std::string>("chunk_topic",
                                          "/lerobot/franka/joint_action_chunk");
    ack_topic_ = parameter<std::string>(
        "ack_topic", "/lerobot/franka/joint_action_chunk_ack");
    status_topic_ = parameter<std::string>(
        "status_topic", "/lerobot/franka/joint_safety_gateway_status");
    safe_command_topic_ = parameter<std::string>("safe_command_topic",
                                                 "/franka/safe_joint_command");
    feedback_topic_ = parameter<std::string>("safety_feedback_topic",
                                             "/franka/safety_command_feedback");
    robot_readiness_topic_ =
        parameter<std::string>("robot_readiness_topic", "/franka/robot_ready");
    controller_readiness_topic_ = parameter<std::string>(
        "controller_readiness_topic", "/franka/controller_ready");

    if (command_period_ms_ < 1 || command_period_ms_ > 10 ||
        publisher_watchdog_ns_ <= command_period_ms_ * 1'000'000LL ||
        !std::isfinite(max_state_drift_rad_) || max_state_drift_rad_ <= 0.0 ||
        !std::isfinite(max_tracking_error_rad_) ||
        max_tracking_error_rad_ <= 0.0 ||
        !std::isfinite(preflight_settings_.collision_interpolation_step_rad) ||
        preflight_settings_.collision_interpolation_step_rad <= 0.0 ||
        !std::isfinite(preflight_settings_.max_retiming_scale) ||
        preflight_settings_.max_retiming_scale < 1.0 ||
        !std::isfinite(limits_.gripper_min) ||
        !std::isfinite(limits_.gripper_max) ||
        limits_.gripper_min > limits_.gripper_max ||
        !std::isfinite(gripper_max_effort_) || gripper_max_effort_ <= 0.0 ||
        !std::isfinite(gripper_command_deadband_) ||
        gripper_command_deadband_ < 0.0 ||
        (gripper_actuation_enabled() && !limits_.validate_gripper_range)) {
      throw rclcpp::exceptions::InvalidParametersException(
          "unsafe joint gateway parameters");
    }

    const auto qos =
        rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    const auto state_qos = rclcpp::SensorDataQoS().keep_last(1);
    chunk_group_ =
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    state_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    publisher_group_ =
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    gripper_client_ = rclcpp_action::create_client<GripperAction>(
        this, gripper_action_name_, publisher_group_);
    rclcpp::SubscriptionOptions chunk_options;
    chunk_options.callback_group = chunk_group_;
    rclcpp::SubscriptionOptions state_options;
    state_options.callback_group = state_group_;

    chunk_subscription_ =
        create_subscription<lerobot_franka_interfaces::msg::JointActionChunk>(
            chunk_topic_, qos,
            std::bind(&GatewayNode::on_chunk, this, std::placeholders::_1),
            chunk_options);
    joint_state_subscription_ =
        create_subscription<sensor_msgs::msg::JointState>(
            joint_state_topic_, state_qos,
            std::bind(&GatewayNode::on_joint_state, this,
                      std::placeholders::_1),
            state_options);
    feedback_subscription_ = create_subscription<
        franka_safety_interfaces::msg::SafetyCommandFeedback>(
        feedback_topic_, qos,
        std::bind(&GatewayNode::on_feedback, this, std::placeholders::_1),
        state_options);
    robot_readiness_subscription_ = create_subscription<std_msgs::msg::Bool>(
        robot_readiness_topic_, qos,
        [this](const std_msgs::msg::Bool::SharedPtr message) {
          std::lock_guard<std::mutex> lock(mutex_);
          robot_ready_ = message->data;
          last_robot_readiness_ns_ = now_ns();
        },
        state_options);
    controller_readiness_subscription_ =
        create_subscription<std_msgs::msg::Bool>(
            controller_readiness_topic_, qos,
            [this](const std_msgs::msg::Bool::SharedPtr message) {
              std::lock_guard<std::mutex> lock(mutex_);
              controller_ready_ = message->data;
              last_controller_readiness_ns_ = now_ns();
            },
            state_options);

    ack_publisher_ =
        create_publisher<lerobot_franka_interfaces::msg::JointActionChunkAck>(
            ack_topic_, qos);
    status_publisher_ =
        create_publisher<lerobot_franka_interfaces::msg::SafetyGatewayStatus>(
            status_topic_, qos);
    safe_command_publisher_ =
        create_publisher<franka_safety_interfaces::msg::SafeJointCommand>(
            safe_command_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    arm_service_ = create_service<std_srvs::srv::SetBool>(
        "~/set_armed", std::bind(&GatewayNode::on_set_armed, this,
                                 std::placeholders::_1, std::placeholders::_2));
    command_timer_ = create_wall_timer(
        std::chrono::milliseconds(command_period_ms_),
        std::bind(&GatewayNode::publish_command_tick, this), publisher_group_);
    status_timer_ =
        create_wall_timer(std::chrono::milliseconds(100),
                          std::bind(&GatewayNode::publish_status, this));
  }

  void initialize_preflight() {
    auto provider = std::make_shared<JointPreflightProvider>(
        std::static_pointer_cast<rclcpp::Node>(shared_from_this()),
        preflight_settings_);
    std::lock_guard<std::mutex> lock(mutex_);
    preflight_ = std::move(provider);
    if (!preflight_->ready() && enabled_ && !shadow_ && !preflight_only_) {
      safety_state_.fault(preflight_->detail());
    }
  }

  void shutdown_preflight() {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_execution_locked();
    preflight_.reset();
  }

private:
  struct ExecutablePlan {
    JointPlan plan;
    JointArray initial_joints{};
  };

  template <typename T>
  T parameter(const std::string &name, const T &default_value) {
    if (has_parameter(name)) {
      return get_parameter(name).get_value<T>();
    }
    return declare_parameter<T>(name, default_value);
  }

  static std::int64_t seconds_to_ns(double seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0 ||
        seconds >
            static_cast<double>(std::numeric_limits<std::int64_t>::max()) *
                1e-9) {
      throw rclcpp::exceptions::InvalidParametersException(
          "duration must be finite and positive");
    }
    return static_cast<std::int64_t>(seconds * 1e9);
  }

  JointArray read_joint_array(const std::string &name,
                              const std::vector<double> &defaults) {
    const auto values = parameter<std::vector<double>>(name, defaults);
    if (values.size() != kJointCount ||
        !std::all_of(values.begin(), values.end(), [](double value) {
          return std::isfinite(value) && value > 0.0;
        })) {
      throw rclcpp::exceptions::InvalidParametersException(
          name + " must contain seven finite positive values");
    }
    JointArray result{};
    std::copy(values.begin(), values.end(), result.begin());
    return result;
  }

  std::int64_t now_ns() const { return get_clock()->now().nanoseconds(); }

  bool state_fresh_locked(std::int64_t time_ns) const {
    return current_joints_.has_value() && last_joint_state_ns_.has_value() &&
           time_ns >= *last_joint_state_ns_ &&
           time_ns - *last_joint_state_ns_ <= state_timeout_ns_;
  }

  bool input_ready_locked(bool required, bool value,
                          const std::optional<std::int64_t> &stamp,
                          std::int64_t time_ns) const {
    return !required || (value && stamp.has_value() && time_ns >= *stamp &&
                         time_ns - *stamp <= readiness_timeout_ns_);
  }

  bool robot_ready_locked(std::int64_t time_ns) const {
    return input_ready_locked(require_robot_readiness_, robot_ready_,
                              last_robot_readiness_ns_, time_ns);
  }

  bool controller_ready_locked(std::int64_t time_ns) const {
    return input_ready_locked(require_controller_readiness_, controller_ready_,
                              last_controller_readiness_ns_, time_ns);
  }

  bool unique_safe_command_publisher_locked() const {
    return !require_unique_command_publisher_ ||
           count_publishers(safe_command_topic_) == 1;
  }

  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr message) {
    static const std::array<std::string, 7> required_names = {
        "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
        "fr3_joint5", "fr3_joint6", "fr3_joint7"};
    if (message->name.size() != message->position.size()) {
      return;
    }
    std::unordered_map<std::string, double> positions;
    for (std::size_t index = 0; index < message->name.size(); ++index) {
      positions.emplace(message->name[index], message->position[index]);
    }
    JointArray ordered{};
    for (std::size_t index = 0; index < required_names.size(); ++index) {
      const auto found = positions.find(required_names[index]);
      if (found == positions.end() || !std::isfinite(found->second)) {
        return;
      }
      ordered[index] = found->second;
    }
    const auto stamp = to_nanoseconds(message->header.stamp);
    const auto receive = now_ns();
    if (stamp <= 0 || stamp > receive || receive - stamp > state_stamp_timeout_ns_) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    current_joints_ = ordered;
    last_joint_state_ns_ = receive;
  }

  void on_feedback(
      const franka_safety_interfaces::msg::SafetyCommandFeedback::SharedPtr
          message) {
    const auto receive = now_ns();
    const auto stamp = to_nanoseconds(message->header.stamp);
    if (stamp <= 0 || stamp > receive ||
        receive - stamp > feedback_timeout_ns_) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!execution_ || !command_watchdog_active_) {
      return;
    }
    const auto &plan = execution_->plan;
    if (message->session_id != plan.session_id ||
        message->plan_id != plan.plan_id) {
      hold_locked("controller feedback identifies a different plan");
      return;
    }
    if (!message->accepted || !message->applied) {
      hold_locked("controller rejected command: " + message->reason);
      return;
    }
    if (message->sequence < last_applied_sequence_ ||
        message->sequence > sequence_ ||
        message->waypoint_index >= plan.waypoints.size()) {
      hold_locked("controller feedback progression mismatch");
      return;
    }
    if (message->sequence > last_applied_sequence_) {
      last_applied_sequence_ = message->sequence;
      last_applied_waypoint_index_ = message->waypoint_index;
      last_applied_source_timestep_ = message->source_timestep;
    }
    last_applied_feedback_ns_ = stamp;
  }

  JointPlan
  convert_plan(const lerobot_franka_interfaces::msg::JointActionChunk &message,
               bool *duration_valid) const {
    JointPlan plan;
    plan.frame_id = message.header.frame_id;
    plan.start_ns = to_nanoseconds(message.header.stamp);
    plan.schema_version = message.schema_version;
    plan.session_id = message.session_id;
    plan.plan_id = message.plan_id;
    plan.source_timestep = message.source_timestep;
    plan.valid_until_ns = to_nanoseconds(message.valid_until);
    const auto period = to_nanoseconds(message.period);
    *duration_valid = period.has_value();
    plan.period_ns = period.value_or(0);
    plan.timesteps = message.timesteps;
    plan.gripper = message.gripper;
    if (message.positions.size() == message.timesteps.size() * kJointCount) {
      plan.waypoints.resize(message.timesteps.size());
      for (std::size_t point = 0; point < plan.waypoints.size(); ++point) {
        std::copy_n(message.positions.begin() +
                        static_cast<std::ptrdiff_t>(point * kJointCount),
                    kJointCount, plan.waypoints[point].begin());
      }
    }
    return plan;
  }

  void
  on_chunk(const lerobot_franka_interfaces::msg::JointActionChunk::SharedPtr
               message) {
    bool duration_valid = false;
    auto candidate = convert_plan(*message, &duration_valid);
    const auto receive = now_ns();
    ValidationResult result;
    bool replaced = false;
    std::shared_ptr<JointPreflightProvider> provider;
    JointArray current{};

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!duration_valid) {
        result = {ResultCode::kRejectedSchedule, "period duration is negative"};
      } else if ((!enabled_ && !preflight_only_) ||
                 (shadow_ && !preflight_only_)) {
        result = plans_.accept(candidate, receive, limits_);
        replaced = result.accepted();
        if (result.accepted()) {
          accepted_waypoint_index_ =
              static_cast<std::uint32_t>(candidate.waypoints.size() - 1);
          accepted_source_timestep_ = candidate.timesteps.back();
        }
      } else {
        if (execution_.has_value()) {
          result = {ResultCode::kRejectedOrdering,
                    "an action plan is still executing; wait for completion"};
        } else {
          result = plans_.validate_candidate(candidate, receive, limits_);
        }
        if (result.accepted() && !preflight_only_ && !safety_state_.armed()) {
          result = {ResultCode::kRejectedNotArmed,
                    "joint gateway is not ARMED"};
        } else if (result.accepted() && !state_fresh_locked(receive)) {
          result = {ResultCode::kRejectedStateStale, "joint state is stale"};
        } else if (result.accepted() && !robot_ready_locked(receive)) {
          result = {ResultCode::kRejectedStateStale,
                    "robot readiness is absent/stale/false"};
        } else if (result.accepted() && !preflight_only_ &&
                   !controller_ready_locked(receive)) {
          result = {ResultCode::kRejectedNotArmed,
                    "controller readiness is absent/stale/false"};
        } else if (result.accepted() && (!preflight_ || !preflight_->ready() ||
                                         !preflight_->planning_scene_ready())) {
          result = {
              ResultCode::kRejectedPreflightUnavailable,
              "MoveIt joint preflight or monitored world scene is unavailable"};
        }
        if (result.accepted()) {
          provider = preflight_;
          current = *current_joints_;
        }
      }
      if (safety_state_.armed() || result.accepted()) {
        detail_ = result.detail;
      }
    }

    if (result.accepted() && provider) {
      const auto preflight_result = provider->validate(candidate, current);
      std::lock_guard<std::mutex> lock(mutex_);
      const auto finish = now_ns();
      bool state_drifted = !current_joints_.has_value();
      if (current_joints_) {
        for (std::size_t joint = 0; joint < kJointCount; ++joint) {
          state_drifted = state_drifted ||
                          std::abs((*current_joints_)[joint] - current[joint]) >
                              max_state_drift_rad_;
        }
      }
      if ((!preflight_only_ && !safety_state_.armed()) ||
          !state_fresh_locked(finish) || !robot_ready_locked(finish) ||
          (!preflight_only_ && !controller_ready_locked(finish)) ||
          !provider->planning_scene_ready() || state_drifted) {
        result = {ResultCode::kRejectedStateStale,
                  "safety gates or measured joints changed during preflight"};
      } else if (!preflight_result.accepted) {
        result = {preflight_result.failure_code, preflight_result.detail};
      } else {
        auto accepted = candidate;
        accepted.start_ns = finish;
        accepted.period_ns = preflight_result.period_ns;
        result = plans_.accept(accepted, finish, limits_);
        if (result.accepted()) {
          result = {preflight_only_ ? ResultCode::kAcceptedPreflightOnly
                                    : ResultCode::kAcceptedForExecution,
                    (preflight_only_ ? "preflight-only: " : "") +
                        preflight_result.detail};
          if (!preflight_only_) {
            execution_ = ExecutablePlan{accepted, current};
            last_gripper_waypoint_index_.reset();
            last_command_tick_ = std::chrono::steady_clock::now();
            command_watchdog_active_ = false;
            first_command_ns_.reset();
            last_applied_feedback_ns_.reset();
            last_applied_sequence_ = 0;
            last_applied_waypoint_index_ = 0;
            last_applied_source_timestep_ = 0;
          }
          replaced = true;
          accepted_waypoint_index_ =
              static_cast<std::uint32_t>(accepted.waypoints.size() - 1);
          accepted_source_timestep_ = accepted.timesteps.back();
        }
      }
      detail_ = result.detail;
    }

    lerobot_franka_interfaces::msg::JointActionChunkAck ack;
    ack.header.stamp = get_clock()->now();
    ack.header.frame_id = limits_.frame_id;
    ack.schema_version = limits_.schema_version;
    ack.session_id = message->session_id;
    ack.plan_id = message->plan_id;
    ack.result = static_cast<std::uint8_t>(result.code);
    ack.accepted = result.accepted();
    ack.replaced_active_plan = replaced;
    ack.waypoint_count = static_cast<std::uint32_t>(message->timesteps.size());
    ack.detail = result.detail;
    ack_publisher_->publish(std::move(ack));
    publish_status();
  }

  void on_set_armed(const std_srvs::srv::SetBool::Request::SharedPtr request,
                    std_srvs::srv::SetBool::Response::SharedPtr response) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!request->data) {
      stop_execution_locked();
      safety_state_.disarm();
      response->success = true;
      response->message = safety_state_.detail();
      detail_ = response->message;
      return;
    }
    const auto now = now_ns();
    ArmGates gates{enabled_,
                   shadow_ || preflight_only_,
                   preflight_ && preflight_->ready(),
                   state_fresh_locked(now),
                   preflight_ && preflight_->planning_scene_ready(),
                   robot_ready_locked(now),
                   controller_ready_locked(now)};
    response->success = safety_state_.arm(gates, &response->message);
    if (response->success && require_command_subscriber_ &&
        safe_command_publisher_->get_subscription_count() == 0) {
      safety_state_.hold("no SafeJointCommand controller subscriber");
      response->success = false;
      response->message = safety_state_.detail();
    }
    if (response->success && !unique_safe_command_publisher_locked()) {
      safety_state_.hold("another /franka/safe_joint_command publisher exists");
      response->success = false;
      response->message = safety_state_.detail();
    }
    if (response->success && gripper_actuation_enabled() &&
        (!gripper_client_ || !gripper_client_->action_server_is_ready())) {
      safety_state_.hold("Robotiq gripper action server is unavailable: " +
                         gripper_action_name_);
      response->success = false;
      response->message = safety_state_.detail();
    }
    detail_ = response->message;
  }

  bool gripper_actuation_enabled() const {
    return execute_gripper_ && enabled_ && !shadow_ && !preflight_only_;
  }

  void cancel_gripper_locked() {
    if (gripper_actuation_enabled() && gripper_client_ &&
        gripper_client_->action_server_is_ready()) {
      try {
        (void)gripper_client_->async_cancel_all_goals();
      } catch (const std::exception &error) {
        RCLCPP_ERROR(get_logger(), "Failed to cancel gripper goals: %s",
                     error.what());
      }
    }
    last_gripper_waypoint_index_.reset();
    last_gripper_command_position_.reset();
    last_gripper_command_ns_.reset();
  }

  void stop_execution_locked(bool cancel_gripper = true) {
    execution_.reset();
    plans_.clear();
    command_watchdog_active_ = false;
    first_command_ns_.reset();
    if (cancel_gripper) {
      cancel_gripper_locked();
    } else {
      last_gripper_waypoint_index_.reset();
    }
  }

  void hold_locked(const std::string &reason) {
    RCLCPP_WARN(get_logger(), "Entering HOLD: %s", reason.c_str());

    stop_execution_locked();
    safety_state_.hold(reason);
    detail_ = safety_state_.detail();
  }

  bool maybe_send_gripper_goal_locked(const JointPlan &plan,
                                      std::uint32_t waypoint_index,
                                      std::int64_t now) {
    if (!gripper_actuation_enabled()) {
      return true;
    }
    if (!gripper_client_ || !gripper_client_->action_server_is_ready()) {
      hold_locked("Robotiq gripper action server became unavailable: " +
                  gripper_action_name_);
      return false;
    }
    if (waypoint_index >= plan.gripper.size()) {
      hold_locked("accepted plan has no gripper target for current waypoint");
      return false;
    }
    if (last_gripper_waypoint_index_ == waypoint_index) {
      return true;
    }

    const double position = plan.gripper[waypoint_index];
    if (last_gripper_command_position_.has_value() &&
        std::abs(position - *last_gripper_command_position_) <
            gripper_command_deadband_) {
      last_gripper_waypoint_index_ = waypoint_index;
      return true;
    }
    if (last_gripper_command_ns_.has_value() &&
        now - *last_gripper_command_ns_ < gripper_min_command_interval_ns_) {
      return true;
    }

    GripperAction::Goal goal;
    goal.command.position = position;
    goal.command.max_effort = gripper_max_effort_;
    rclcpp_action::Client<GripperAction>::SendGoalOptions options;
    options.goal_response_callback =
        [this, position](const GripperGoalHandle::SharedPtr &goal_handle) {
          if (goal_handle) {
            RCLCPP_INFO(get_logger(), "Robotiq gripper goal accepted: %.4f",
                        position);
            return;
          }
          std::lock_guard<std::mutex> lock(mutex_);
          if (safety_state_.armed()) {
            hold_locked("Robotiq gripper rejected target " +
                        std::to_string(position));
          }
        };
    options.result_callback =
        [this, position](const GripperGoalHandle::WrappedResult &result) {
          if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_DEBUG(get_logger(), "Robotiq gripper reached target: %.4f",
                         position);
            return;
          }
          if (result.code == rclcpp_action::ResultCode::CANCELED) {
            RCLCPP_DEBUG(get_logger(), "Robotiq gripper goal canceled: %.4f",
                         position);
            return;
          }
          std::lock_guard<std::mutex> lock(mutex_);
          if (safety_state_.armed()) {
            hold_locked("Robotiq gripper action failed for target " +
                        std::to_string(position));
          }
        };
    try {
      (void)gripper_client_->async_send_goal(goal, options);
    } catch (const std::exception &error) {
      hold_locked("failed to send Robotiq gripper goal: " +
                  std::string(error.what()));
      return false;
    }
    last_gripper_waypoint_index_ = waypoint_index;
    last_gripper_command_position_ = position;
    last_gripper_command_ns_ = now;
    return true;
  }

  void publish_command_tick() {
    const auto steady_now = std::chrono::steady_clock::now();
    const auto ros_now = get_clock()->now();
    franka_safety_interfaces::msg::SafeJointCommand command;
    bool publish = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!safety_state_.armed()) {
        return;
      }
      if (command_watchdog_active_) {
        const auto gap = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             steady_now - last_command_tick_)
                             .count();
        if (gap > publisher_watchdog_ns_) {
          hold_locked("command publisher watchdog expired");
          return;
        }
      }
      last_command_tick_ = steady_now;
      const auto now = ros_now.nanoseconds();
      if (!state_fresh_locked(now)) {
        hold_locked("joint-state watchdog expired");
        return;
      }
      if (!robot_ready_locked(now)) {
        hold_locked("robot readiness watchdog expired");
        return;
      }
      if (!controller_ready_locked(now)) {
        hold_locked("controller readiness watchdog expired");
        return;
      }
      if (!preflight_ || !preflight_->ready() ||
          !preflight_->planning_scene_ready()) {
        hold_locked("MoveIt joint preflight became unavailable");
        return;
      }
      if (require_command_subscriber_ &&
          safe_command_publisher_->get_subscription_count() == 0) {
        hold_locked("SafeJointCommand controller subscriber disappeared");
        return;
      }
      if (!unique_safe_command_publisher_locked()) {
        hold_locked("another SafeJointCommand publisher appeared");
        return;
      }
      if (require_chunk_publisher_ && count_publishers(chunk_topic_) == 0) {
        hold_locked("joint action publisher disappeared");
        return;
      }
      if (!execution_) {
        return;
      }
      if (first_command_ns_.has_value() &&
          now - *first_command_ns_ > feedback_grace_ns_) {
        if (!last_applied_feedback_ns_.has_value() ||
            now < *last_applied_feedback_ns_ ||
            now - *last_applied_feedback_ns_ > feedback_timeout_ns_) {
          hold_locked("controller applied-feedback watchdog expired");
          return;
        }
        if (sequence_ >= last_applied_sequence_ &&
            sequence_ - last_applied_sequence_ > max_applied_sequence_lag_) {
          hold_locked("controller applied-feedback progression lag exceeded");
          return;
        }
      }

      const auto &plan = execution_->plan;
      std::int64_t execution_end_ns = 0;
      if (!franka_cartesian_safety_gateway::execution_deadline(
              plan.start_ns, plan.waypoints.size(), plan.period_ns,
              settle_window_ns_, nullptr, &execution_end_ns)) {
        safety_state_.fault("accepted joint plan horizon overflow");
        stop_execution_locked();
        detail_ = safety_state_.detail();
        return;
      }
      if (now >= execution_end_ns) {
        if (hold_after_plan_completion_) {
          stop_execution_locked();
          safety_state_.hold(
              "accepted joint plan completed after settle window");
          detail_ = safety_state_.detail();
        } else {
          stop_execution_locked(false);
          detail_ =
              "ARMED: joint plan completed; holding while awaiting next plan";
        }
        return;
      }
      if (sequence_ == std::numeric_limits<std::uint64_t>::max()) {
        safety_state_.fault("SafeJointCommand sequence exhausted");
        stop_execution_locked();
        detail_ = safety_state_.detail();
        return;
      }

      const auto target =
          franka_cartesian_safety_gateway::interpolate_joint_plan(
              execution_->initial_joints, plan.waypoints, plan.timesteps,
              plan.start_ns, plan.period_ns, now);
      tracking_error_rad_ = 0.0;
      for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        tracking_error_rad_ =
            std::max(tracking_error_rad_, std::abs(target.positions[joint] -
                                                   (*current_joints_)[joint]));
      }
      measured_waypoint_index_ = target.waypoint_index;
      measured_source_timestep_ = target.source_timestep;
      if (now >= plan.start_ns && now - plan.start_ns >= tracking_grace_ns_ &&
          tracking_error_rad_ > max_tracking_error_rad_) {
        hold_locked("joint tracking watchdog exceeded: " +
                    std::to_string(tracking_error_rad_) + " rad");
        return;
      }
      if (!maybe_send_gripper_goal_locked(plan, target.waypoint_index, now)) {
        return;
      }

      command.header.stamp = ros_now;
      command.header.frame_id = limits_.frame_id;
      command.deadline = from_nanoseconds(now + command_deadline_ns_);
      command.session_id = plan.session_id;
      command.plan_id = plan.plan_id;
      command.waypoint_index = target.waypoint_index;
      command.source_timestep = target.source_timestep;
      command.sequence = ++sequence_;
      command.joint_names = preflight_->joint_names();
      command.positions = target.positions;
      command.safety_state =
          franka_safety_interfaces::msg::SafeJointCommand::ARMED;
      command.status = "MoveIt-preflighted interpolated FastWAM joint target; "
                       "gripper scheduled through Robotiq action";
      command_watchdog_active_ = true;
      if (!first_command_ns_.has_value()) {
        first_command_ns_ = now;
      }
      publish = true;
    }
    if (publish) {
      safe_command_publisher_->publish(std::move(command));
    }
  }

  void publish_status() {
    lerobot_franka_interfaces::msg::SafetyGatewayStatus status;
    const auto time = get_clock()->now();
    const auto time_ns = time.nanoseconds();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      plans_.reset_expired(time_ns, limits_);
      status.header.stamp = time;
      status.header.frame_id = limits_.frame_id;
      status.schema_version = limits_.schema_version;
      const auto state = safety_state_.state();
      status.state = static_cast<std::uint8_t>(state);
      status.shadow = state == GatewayState::kShadow;
      status.armed = state == GatewayState::kArmed;
      status.state_fresh = state_fresh_locked(time_ns);
      status.preflight_available = preflight_ && preflight_->ready() &&
                                   preflight_->planning_scene_ready();
      status.robot_ready = robot_ready_locked(time_ns);
      status.controller_ready = controller_ready_locked(time_ns);
      status.applied_feedback_fresh =
          last_applied_feedback_ns_.has_value() &&
          time_ns >= *last_applied_feedback_ns_ &&
          time_ns - *last_applied_feedback_ns_ <= feedback_timeout_ns_;
      status.has_active_plan =
          execution_.has_value() || plans_.active_plan().has_value();
      status.last_command_sequence = sequence_;
      status.last_applied_sequence = last_applied_sequence_;
      status.accepted_waypoint_index = accepted_waypoint_index_;
      status.applied_waypoint_index = last_applied_waypoint_index_;
      status.measured_waypoint_index = measured_waypoint_index_;
      status.accepted_source_timestep = accepted_source_timestep_;
      status.applied_source_timestep = last_applied_source_timestep_;
      status.measured_source_timestep = measured_source_timestep_;
      status.joint_tracking_error_rad = tracking_error_rad_;
      status.tcp_position_error_m = 0.0;
      status.tcp_orientation_error_rad = 0.0;
      status.applied_feedback_age = duration_from_nanoseconds(
          last_applied_feedback_ns_ ? time_ns - *last_applied_feedback_ns_
                                    : std::numeric_limits<std::int32_t>::max() *
                                          kNanosecondsPerSecond);
      status.robot_readiness_age = duration_from_nanoseconds(
          last_robot_readiness_ns_ ? time_ns - *last_robot_readiness_ns_
                                   : std::numeric_limits<std::int32_t>::max() *
                                         kNanosecondsPerSecond);
      if (plans_.active_plan()) {
        const auto &plan = *plans_.active_plan();
        status.session_id = plan.session_id;
        status.plan_id = plan.plan_id;
        status.valid_until = from_nanoseconds(plan.valid_until_ns);
        if (execution_) {
          const auto target =
              franka_cartesian_safety_gateway::interpolate_joint_plan(
                  execution_->initial_joints, plan.waypoints, plan.timesteps,
                  plan.start_ns, plan.period_ns, time_ns);
          status.next_waypoint_index = target.waypoint_index;
          status.next_waypoint_due = from_nanoseconds(
              plan.start_ns +
              (static_cast<std::int64_t>(target.waypoint_index) + 1) *
                  plan.period_ns);
        } else {
          status.next_waypoint_due = from_nanoseconds(plan.start_ns);
        }
      }
      status.detail = detail_.empty() ? safety_state_.detail() : detail_;
    }
    status_publisher_->publish(std::move(status));
  }

  ValidationLimits limits_;
  PreflightSettings preflight_settings_;
  bool enabled_;
  bool shadow_;
  bool preflight_only_;
  SafetyStateMachine safety_state_;
  bool require_command_subscriber_{true};
  bool require_unique_command_publisher_{true};
  bool require_chunk_publisher_{true};
  bool require_robot_readiness_{true};
  bool require_controller_readiness_{true};
  bool hold_after_plan_completion_{false};
  bool execute_gripper_{true};
  int command_period_ms_{5};
  std::int64_t state_timeout_ns_{150'000'000};
  std::int64_t state_stamp_timeout_ns_{500'000'000};
  std::int64_t readiness_timeout_ns_{500'000'000};
  std::int64_t command_deadline_ns_{30'000'000};
  std::int64_t publisher_watchdog_ns_{30'000'000};
  std::int64_t settle_window_ns_{250'000'000};
  std::int64_t feedback_timeout_ns_{50'000'000};
  std::int64_t feedback_grace_ns_{100'000'000};
  std::int64_t tracking_grace_ns_{500'000'000};
  std::uint64_t max_applied_sequence_lag_{20};
  double max_state_drift_rad_{0.01};
  double max_tracking_error_rad_{0.25};
  double gripper_max_effort_{16.0};
  double gripper_command_deadband_{0.02};
  std::int64_t gripper_min_command_interval_ns_{150'000'000};
  std::string joint_state_topic_;
  std::string chunk_topic_;
  std::string ack_topic_;
  std::string status_topic_;
  std::string safe_command_topic_;
  std::string feedback_topic_;
  std::string robot_readiness_topic_;
  std::string controller_readiness_topic_;
  std::string gripper_action_name_;
  std::string detail_;
  std::mutex mutex_;
  PlanStateMachine plans_;
  std::shared_ptr<JointPreflightProvider> preflight_;
  std::optional<JointArray> current_joints_;
  std::optional<std::int64_t> last_joint_state_ns_;
  std::optional<std::int64_t> last_robot_readiness_ns_;
  std::optional<std::int64_t> last_controller_readiness_ns_;
  std::optional<std::int64_t> last_applied_feedback_ns_;
  std::optional<std::int64_t> first_command_ns_;
  std::optional<std::int64_t> last_gripper_command_ns_;
  std::optional<double> last_gripper_command_position_;
  std::optional<std::uint32_t> last_gripper_waypoint_index_;
  std::optional<ExecutablePlan> execution_;
  bool robot_ready_{false};
  bool controller_ready_{false};
  std::uint64_t sequence_{0};
  std::uint64_t last_applied_sequence_{0};
  std::uint32_t accepted_waypoint_index_{0};
  std::uint32_t last_applied_waypoint_index_{0};
  std::uint32_t measured_waypoint_index_{0};
  std::int64_t accepted_source_timestep_{0};
  std::int64_t last_applied_source_timestep_{0};
  std::int64_t measured_source_timestep_{0};
  double tracking_error_rad_{0.0};
  std::chrono::steady_clock::time_point last_command_tick_{};
  bool command_watchdog_active_{false};

  rclcpp::CallbackGroup::SharedPtr chunk_group_;
  rclcpp::CallbackGroup::SharedPtr state_group_;
  rclcpp::CallbackGroup::SharedPtr publisher_group_;
  rclcpp_action::Client<GripperAction>::SharedPtr gripper_client_;
  rclcpp::Subscription<lerobot_franka_interfaces::msg::JointActionChunk>::
      SharedPtr chunk_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
      joint_state_subscription_;
  rclcpp::Subscription<franka_safety_interfaces::msg::SafetyCommandFeedback>::
      SharedPtr feedback_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr
      robot_readiness_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr
      controller_readiness_subscription_;
  rclcpp::Publisher<lerobot_franka_interfaces::msg::JointActionChunkAck>::
      SharedPtr ack_publisher_;
  rclcpp::Publisher<lerobot_franka_interfaces::msg::SafetyGatewayStatus>::
      SharedPtr status_publisher_;
  rclcpp::Publisher<franka_safety_interfaces::msg::SafeJointCommand>::SharedPtr
      safe_command_publisher_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr arm_service_;
  rclcpp::TimerBase::SharedPtr command_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;
};

} // namespace franka_joint_safety_gateway

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node =
      std::make_shared<franka_joint_safety_gateway::GatewayNode>(options);
  node->initialize_preflight();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),
                                                    4);
  executor.add_node(node);
  executor.spin();
  node->shutdown_preflight();
  executor.remove_node(node);
  node.reset();
  rclcpp::shutdown();
  return 0;
}
