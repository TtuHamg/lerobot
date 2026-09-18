#!/bin/bash
# Start the q-pos or absolute-EEF LeRobot client with machine-local defaults.
set -euo pipefail

ACTION_SPACE="${ACTION_SPACE:-qpos}"
PROMPT="Pick up the cup."
PROMPT_SET=false
RECORD_VIDEO=true
for arg in "$@"; do
  case "$arg" in
    --qpos) ACTION_SPACE=qpos ;;
    --eef) ACTION_SPACE=eef ;;
    --record-video) RECORD_VIDEO=true ;;
    --no-record-video) RECORD_VIDEO=false ;;
    --help)
      echo "Usage: $0 [--qpos|--eef] [--record-video|--no-record-video] [prompt]"
      exit 0
      ;;
    --*)
      echo "Unknown option: $arg"
      echo "Usage: $0 [--qpos|--eef] [--record-video|--no-record-video] [prompt]"
      exit 2
      ;;
    *)
      if [[ "$PROMPT_SET" == "true" ]]; then
        echo "Prompt must be passed as one quoted argument"
        exit 2
      fi
      PROMPT="$arg"
      PROMPT_SET=true
      ;;
  esac
done
if [[ "$ACTION_SPACE" != "qpos" && "$ACTION_SPACE" != "eef" ]]; then
  echo "ERROR: ACTION_SPACE must be qpos or eef"
  exit 2
fi

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8081}"
if [[ "$ACTION_SPACE" == "eef" ]]; then
  DEFAULT_CONDA_ENV=lerobot
  CLIENT_LOG_NAME=lerobot_client_eef.log
else
  DEFAULT_CONDA_ENV=lerobot_ght
  CLIENT_LOG_NAME=lerobot_client.log
fi
CONDA_ENV="${CONDA_ENV:-$DEFAULT_CONDA_ENV}"
IK_URDF="${IK_URDF:-$HOME/franka/config/fr3_ik.urdf}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROS_WS="$REPO_ROOT/franka_project/ros2_ws"
LOG_DIR="$HOME/franka/logs/joint_validation"
VIDEO_ROOT="${FRANKA_VIDEO_DIR:-$HOME/franka/videos/joint_validation}"
VIDEO_RECORDER="$SCRIPT_DIR/record_camera_videos.py"
VIDEO_RECORDER_CPU_AFFINITY="${VIDEO_RECORDER_CPU_AFFINITY:-12-15}"
COLOR_PROFILE="${FRANKA_COLOR_PROFILE:-640,480,30}"
VIDEO_FPS="${FRANKA_VIDEO_FPS:-${COLOR_PROFILE##*,}}"
VIDEO_CODEC="${FRANKA_VIDEO_CODEC:-avc1}"
VIDEO_RECORDER_PID=""
VIDEO_SESSION_DIR=""
VIDEO_RECORDER_LOG=""

mkdir -p "$LOG_DIR"

# Keep one client/recorder lifecycle owner at a time.
exec 9>"$LOG_DIR/.joint_client.lock"
if ! flock -n 9; then
  echo "ERROR: another run_joint_client.sh instance is already active"
  exit 1
fi

set +u
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$HOME/franka/franka_ros2_ws/install/setup.bash"
# shellcheck disable=SC1091
source "$HOME/franka/robotiq_ws/install/setup.bash"
# shellcheck disable=SC1091
source "$HOME/franka/haply_ros/install/setup.bash"
# shellcheck disable=SC1091
source "$ROS_WS/install/setup.bash"
set -u

export PYTHONPATH="$REPO_ROOT/franka_project/ros_lerobot/src:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

TOPIC_INFO="$(
  ros2 topic info /lerobot/franka/joint_action_chunk --verbose 2>/dev/null || true
)"
if [[ "$TOPIC_INFO" != *"lerobot_franka_interfaces/msg/JointActionChunk"* ]]; then
  echo "ERROR: Joint Gateway action topic has the wrong type or is unavailable"
  echo "$TOPIC_INFO"
  echo "Start it first: bash franka_project/scripts/start_joint_stack.sh --execute --cam"
  exit 1
fi

GATEWAY_SUBSCRIBERS="$(
  awk '$1 == "Node" && $2 == "name:" && $3 == "franka_joint_safety_gateway" {
         count += 1
       }
       END { print count + 0 }' <<<"$TOPIC_INFO"
)"
if [[ "$GATEWAY_SUBSCRIBERS" != "1" ]]; then
  echo "ERROR: expected exactly one franka_joint_safety_gateway subscriber"
  echo "$TOPIC_INFO"
  echo "Start it first: bash franka_project/scripts/start_joint_stack.sh --execute --cam"
  exit 1
fi

TOTAL_SUBSCRIBERS="$(
  awk '$1 == "Subscription" && $2 == "count:" {print $3; exit}' <<<"$TOPIC_INFO"
)"
if [[ -n "$TOTAL_SUBSCRIBERS" && "$TOTAL_SUBSCRIBERS" -gt 1 ]]; then
  echo "INFO: allowing $((TOTAL_SUBSCRIBERS - 1)) read-only action observer(s)"
fi

ROBOT_ARGS=(
  --robot.type=franka_ros_joint
  --robot.dry_run=false
  --robot.ros2_interface_only=true
  --robot.action_space="$ACTION_SPACE"
  --robot.max_observation_age_s=0.5
  --robot.camera2_max_skew_s=0.1
  --robot.qpos_max_skew_s=0.2
  --robot.gripper_max_skew_s=0.1
  --robot.observation_buffer_size=64
  --robot.action_chunk_validity_s=1.0
  --robot.action_execution_timeout_s=120
)
if [[ "$ACTION_SPACE" == "eef" ]]; then
  if [[ ! -f "$IK_URDF" ]]; then
    echo "ERROR: EEF IK URDF does not exist: $IK_URDF"
    echo "Generate it with:"
    echo "  FRANKA_IK_GATEWAY_SELF_CHECK_ONLY=1 bash $HOME/franka/run_cartesian_ik_gateway.sh"
    exit 1
  fi
  if ! python -c "import pinocchio" >/dev/null 2>&1; then
    echo "ERROR: EEF mode requires Pinocchio in conda environment '$CONDA_ENV'"
    exit 1
  fi
  if ! timeout 5 ros2 topic echo \
      /franka_robot_state_broadcaster/current_pose --once >/dev/null 2>&1; then
    echo "ERROR: measured EEF pose is unavailable"
    exit 1
  fi
  ROBOT_ARGS+=(
    --robot.eef_max_skew_s=0.1
    --robot.ik_urdf_path="$IK_URDF"
  )
else
  ROBOT_ARGS+=(
    --robot.gripper_model_open_high=true
    --robot.gripper_model_open_position=0.944
  )
fi

stop_video_recorder_pid() {
  local pid="${1:-}"
  if [[ -z "$pid" ]]; then
    return
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    wait "$pid" 2>/dev/null || true
    return
  fi
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 100); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 0.1
  done
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 0.1
  done
  echo "WARNING: camera recorder did not stop cleanly; forcing it to exit"
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup_client_run() {
  local status=$?
  trap - EXIT INT TERM HUP
  set +e
  if [[ -n "$VIDEO_RECORDER_PID" ]]; then
    printf '\nFinalizing camera videos...\n'
    stop_video_recorder_pid "$VIDEO_RECORDER_PID"
    if [[ -f "$VIDEO_SESSION_DIR/camera1_color.mp4" \
      && -f "$VIDEO_SESSION_DIR/camera2_color.mp4" ]]; then
      echo "Camera videos saved:"
      echo "  $VIDEO_SESSION_DIR/camera1_color.mp4"
      echo "  $VIDEO_SESSION_DIR/camera2_color.mp4"
    else
      echo "WARNING: one or both camera videos were not finalized"
      echo "Recorder log: $VIDEO_RECORDER_LOG"
    fi
  fi
  exit "$status"
}

trap cleanup_client_run EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if [[ "$RECORD_VIDEO" == "true" ]]; then
  if [[ ! -f "$VIDEO_RECORDER" ]]; then
    echo "ERROR: camera video recorder is missing: $VIDEO_RECORDER"
    exit 1
  fi
  if ! /usr/bin/python3 -c "import cv2, numpy, rclpy" >/dev/null 2>&1; then
    echo "ERROR: camera recording requires system Python with cv2, numpy, and rclpy"
    exit 1
  fi

  mapfile -t STALE_VIDEO_RECORDER_PIDS < <(
    pgrep -f "[r]ecord_camera_videos.py" 2>/dev/null || true
  )
  if ((${#STALE_VIDEO_RECORDER_PIDS[@]} > 0)); then
    echo "Finalizing stale camera recorder(s)..."
    for pid in "${STALE_VIDEO_RECORDER_PIDS[@]}"; do
      stop_video_recorder_pid "$pid"
    done
  fi

  mkdir -p "$VIDEO_ROOT"
  VIDEO_SESSION_DIR="$VIDEO_ROOT/run_$(date +%Y%m%d_%H%M%S)_$$"
  VIDEO_RECORDER_LOG="$LOG_DIR/camera_video_recorder_$(basename "$VIDEO_SESSION_DIR").log"
  (
    exec 9>&-
    exec taskset -c "$VIDEO_RECORDER_CPU_AFFINITY" \
      /usr/bin/python3 -u "$VIDEO_RECORDER" \
      --output-dir "$VIDEO_SESSION_DIR" \
      --fps "$VIDEO_FPS" \
      --codec "$VIDEO_CODEC"
  ) >"$VIDEO_RECORDER_LOG" 2>&1 &
  VIDEO_RECORDER_PID=$!

  VIDEO_RECORDER_READY=false
  for _ in $(seq 1 150); do
    if [[ -f "$VIDEO_SESSION_DIR/READY" ]]; then
      VIDEO_RECORDER_READY=true
      break
    fi
    if ! kill -0 "$VIDEO_RECORDER_PID" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if [[ "$VIDEO_RECORDER_READY" != "true" ]]; then
    echo "ERROR: dual-camera video recorder did not become ready"
    echo "Recorder log: $VIDEO_RECORDER_LOG"
    exit 1
  fi
  echo "Camera recording: $VIDEO_SESSION_DIR"
  echo "Video codec: $VIDEO_CODEC"
fi

cd "$REPO_ROOT"
python -m lerobot_robot_franka_ros.joint_ros2_client \
  "${ROBOT_ARGS[@]}" \
  --server_address="$SERVER_ADDRESS" \
  --policy_type=act \
  --pretrained_name_or_path=/dummy \
  --task="$PROMPT" \
  --actions_per_chunk=32 \
  --chunk_size_threshold=0.0 \
  --enable_pending_observation=true \
  --pending_observation_timeout_s=15 \
  --aggregate_fn_name=latest_only \
  --client_device=cpu \
  --fps=30 \
  2>&1 | tee "$LOG_DIR/$CLIENT_LOG_NAME"
