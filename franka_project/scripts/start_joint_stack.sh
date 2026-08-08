#!/bin/bash
# Bring up the Franka controller/observations with the joint safety gateway.
# Defaults to SHADOW without cameras and never arms automatically.
set -euo pipefail

MODE=shadow
WITH_CAM=false
FCI_CPU_AFFINITY="10"
GATEWAY_CPU_AFFINITY="9"
QPOS_RELAY_CPU_AFFINITY="7"
for arg in "$@"; do
  case "$arg" in
    --shadow) MODE=shadow ;;
    --execute) MODE=execute ;;
    --no-cam) WITH_CAM=false ;;
    --cam) WITH_CAM=true ;;
    *)
      echo "Unknown option: $arg"
      echo "Use --shadow, --execute, --cam, or --no-cam"
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROS_WS="$REPO_ROOT/franka_project/ros2_ws"
LOG_DIR="$HOME/franka/logs/joint_validation"
REALTIME_CONFIG="$SCRIPT_DIR/configure_fci_realtime.sh"
mkdir -p "$LOG_DIR"

# ROS/ament setup scripts probe optional variables such as
# AMENT_TRACE_SETUP_FILES and are not compatible with nounset.
source_ros_setup() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

source_ros_setup /opt/ros/jazzy/setup.bash
source_ros_setup "$HOME/franka/franka_ros2_ws/install/setup.bash"
source_ros_setup "$HOME/franka/robotiq_ws/install/setup.bash"
source_ros_setup "$HOME/franka/haply_ros/install/setup.bash"
source_ros_setup "$ROS_WS/install/setup.bash"

wait_for_topic_publisher() {
  local topic="$1"
  local timeout_s="$2"
  local attempts=$((timeout_s * 2))
  local info
  for _ in $(seq 1 "$attempts"); do
    info="$(ros2 topic info "$topic" 2>/dev/null || true)"
    if [[ "$info" =~ Publisher\ count:\ ([1-9][0-9]*) ]]; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

wait_for_topic_sample() {
  local topic="$1"
  local timeout_s="$2"
  timeout "$timeout_s" ros2 topic echo "$topic" --once >/dev/null 2>&1
}

# stop_all predates this package, so explicitly remove an earlier joint gateway.
pkill -INT -f "python -m [l]erobot_robot_franka_ros.joint_ros2_client" \
  2>/dev/null || true
pkill -INT -f "python -m [l]erobot_robot_franka_ros.ros2_client" \
  2>/dev/null || true
pkill -INT -f "[c]artesian_ik_gateway.py" 2>/dev/null || true
pkill -INT -f "joint_safety_gateway.launch.py" 2>/dev/null || true
pkill -INT -f "franka_joint_safety_gateway.*/gateway_node" 2>/dev/null || true
sleep 0.5
bash "$HOME/franka/stop_all.sh" --keep-haply-manager
echo "NOTE: stop_all fault analysis above may describe a historical run."
echo "Current startup logs: $LOG_DIR"

if ! bash "$REALTIME_CONFIG" --check; then
  echo "Applying Franka FCI host tuning (sudo may request your password)..."
  sudo FRANKA_NIC=enp1s0 bash "$REALTIME_CONFIG" --apply
fi

STARTUP_COMPLETE=false
cleanup_failed_start() {
  local status=$?
  if [[ "$status" -ne 0 && "$STARTUP_COMPLETE" != "true" ]]; then
    echo "ERROR: startup failed; cleaning the partial Franka stack"
    set +e
    pkill -INT -f "joint_safety_gateway.launch.py" 2>/dev/null
    pkill -INT -f "franka_joint_safety_gateway.*/gateway_node" 2>/dev/null
    bash "$HOME/franka/stop_all.sh" --keep-haply-manager
  fi
}
trap cleanup_failed_start EXIT

nohup ros2 launch franka_gripper_manager robotiq_gripper_controller_client.launch.py \
  >"$LOG_DIR/gripper.log" 2>&1 &
if ! wait_for_topic_publisher /gripper/joint_states 20; then
  echo "ERROR: /gripper/joint_states is unavailable"
  exit 1
fi

if [[ "$WITH_CAM" == "true" ]]; then
  nohup bash "$HOME/franka/start_cameras.sh" >"$LOG_DIR/cameras.log" 2>&1 &
  CAMERA_START_PID=$!
  # start_cameras.sh deliberately kills stale RealSense nodes before launching
  # the two configured serial numbers. Waiting only for a publisher here races
  # with that cleanup and can mistake a stale camera graph entry for readiness.
  if ! wait "$CAMERA_START_PID"; then
    echo "ERROR: camera startup/stream validation failed"
    echo "Logs: $LOG_DIR/cameras.log and $HOME/franka/logs/04_cameras.log"
    exit 1
  fi
  for topic in /camera1/camera1/color/image_raw /camera2/camera2/color/image_raw; do
    if ! wait_for_topic_publisher "$topic" 5 || ! wait_for_topic_sample "$topic" 5; then
      echo "ERROR: required camera stream is unavailable: $topic"
      exit 1
    fi
  done
fi

# Start the 1 kHz Franka loop only after gripper/camera initialization has
# settled; process and USB startup spikes can violate the FCI deadline.
nohup ros2 launch franka_arm_controllers joint_impedance_ik_controller.launch.py \
  control_mode:=real_validation use_rviz:=false \
  fci_cpu_affinity:="$FCI_CPU_AFFINITY" \
  >"$LOG_DIR/arm_controller.log" 2>&1 &
ARM_LAUNCH_PID=$!

CONTROLLER_ACTIVE=false
for _ in $(seq 1 40); do
  STATUS="$(timeout 2 ros2 control list_controllers 2>/dev/null || true)"
  if awk '$1 == "joint_impedance_ik_controller" && $NF == "active" {ok=1} END {exit !ok}' \
      <<<"$STATUS"; then
    CONTROLLER_ACTIVE=true
    break
  fi
  sleep 0.5
done
if [[ "$CONTROLLER_ACTIVE" != "true" ]]; then
  echo "ERROR: joint_impedance_ik_controller did not become active"
  echo "Log: $LOG_DIR/arm_controller.log"
  exit 1
fi

COMMAND_SOURCE="$(ros2 param get /joint_impedance_ik_controller command_source 2>/dev/null || true)"
if [[ "$COMMAND_SOURCE" != *"safety_gateway"* ]]; then
  echo "ERROR: controller command_source is not safety_gateway"
  exit 1
fi

# taskset already constrains every controller-manager thread to CPU10 before
# the first 1 kHz cycle. Add the persistent systemd cpuset when this host has it.
if [[ -x /usr/local/sbin/franka-fci-pin.sh ]]; then
  sudo /usr/local/sbin/franka-fci-pin.sh
fi

# The Python 1 kHz -> 30 Hz relay is CPU-heavy enough to suffer long gaps on an
# E-core under camera/client load. Keep all of its threads on one P-core.
mapfile -t QPOS_RELAY_PIDS < <(pgrep -f "[j]oint_states_30hz.py" || true)
if ((${#QPOS_RELAY_PIDS[@]} == 0)); then
  echo "ERROR: joint_states_30hz.py is not running"
  exit 1
fi
for pid in "${QPOS_RELAY_PIDS[@]}"; do
  taskset -apc "$QPOS_RELAY_CPU_AFFINITY" "$pid" >/dev/null
done

echo "Verifying Franka controller stability for 10 seconds..."
sleep 10
if grep -Eq \
    'communication_constraints_violation|motion aborted by reflex|ros2_control_node-.*process has died' \
    "$LOG_DIR/arm_controller.log"; then
  echo "ERROR: Franka real-time control failed during the startup stability window"
  grep -E \
    'communication_constraints_violation|motion aborted by reflex|ros2_control_node-.*process has died' \
    "$LOG_DIR/arm_controller.log" || true
  echo "Check: $LOG_DIR/arm_controller.log"
  exit 1
fi
if ! kill -0 "$ARM_LAUNCH_PID" 2>/dev/null; then
  echo "ERROR: Franka controller launch exited during the startup stability window"
  echo "Check: $LOG_DIR/arm_controller.log"
  exit 1
fi
if [[ "$WITH_CAM" == "true" ]]; then
  for topic in /camera1/camera1/color/image_raw /camera2/camera2/color/image_raw; do
    if ! wait_for_topic_publisher "$topic" 3 || ! wait_for_topic_sample "$topic" 5; then
      echo "ERROR: camera stream disappeared during Franka startup: $topic"
      echo "Check: $HOME/franka/logs/04_cameras.log"
      exit 1
    fi
  done
fi
ENABLED=false
SHADOW=true
PREFLIGHT_ONLY=true
if [[ "$MODE" == "execute" ]]; then
  ENABLED=true
  SHADOW=false
  PREFLIGHT_ONLY=false
fi
nohup taskset -c "$GATEWAY_CPU_AFFINITY" \
  ros2 launch franka_joint_safety_gateway joint_safety_gateway.launch.py \
  enabled:="$ENABLED" shadow:="$SHADOW" preflight_only:="$PREFLIGHT_ONLY" \
  require_chunk_publisher:=true \
  hold_after_plan_completion:=false \
  >"$LOG_DIR/joint_safety_gateway.log" 2>&1 &

GATEWAY_READY=false
GATEWAY_STATUS=""
for _ in $(seq 1 30); do
  GATEWAY_STATUS="$(
    timeout 2 ros2 topic echo /lerobot/franka/joint_safety_gateway_status --once \
      2>/dev/null || true
  )"
  GATEWAY_READY=true
  for gate in "state_fresh: true" "preflight_available: true"; do
    if [[ "$GATEWAY_STATUS" != *"$gate"* ]]; then
      GATEWAY_READY=false
      break
    fi
  done
  if [[ "$GATEWAY_READY" == "true" ]]; then
    break
  fi
  sleep 0.5
done
if [[ "$GATEWAY_READY" != "true" ]]; then
  echo "ERROR: joint gateway readiness gates did not become ready"
  echo "$GATEWAY_STATUS"
  echo "Log: $LOG_DIR/joint_safety_gateway.log"
  exit 1
fi

if pgrep -af "franka_cartesian_safety_gateway.*/gateway_node" >/dev/null; then
  echo "ERROR: Cartesian gateway is still running; refusing a dual-gateway stack"
  exit 1
fi

STARTUP_COMPLETE=true
printf '\n==============================================\n'
printf ' JOINT VALIDATION READY — %s / NOT ARMED\n' "${MODE^^}"
printf '==============================================\n'
echo "Action input : /lerobot/franka/joint_action_chunk"
echo "ACK          : /lerobot/franka/joint_action_chunk_ack"
echo "Status       : /lerobot/franka/joint_safety_gateway_status"
echo "Logs         : $LOG_DIR"
echo "FCI CPU      : $FCI_CPU_AFFINITY (Franka NIC IRQs: CPU8)"
echo "Gateway CPU  : $GATEWAY_CPU_AFFINITY"
echo "qpos relay CPU: $QPOS_RELAY_CPU_AFFINITY"
echo "Safety mode  : 2x, 0.30 rad/s, 1.0 rad/s², 0.80 rad/chunk; no collision model"
echo
if [[ "$MODE" == "shadow" ]]; then
  echo "SHADOW cannot actuate. Start the client, then inspect ACK/status."
  echo "For real execution, stop this stack and rerun with --execute."
else
  echo "Start the FastWAM client first. Verify one joint subscriber and one safe-command publisher."
  echo "Only after clearing the workspace, arm explicitly with:"
  echo "  ros2 service call /franka_joint_safety_gateway/set_armed std_srvs/srv/SetBool '{data: true}'"
fi
