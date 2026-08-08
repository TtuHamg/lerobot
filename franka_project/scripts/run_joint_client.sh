#!/bin/bash
# Start the q-pos LeRobot client with the validated machine-local defaults.
set -euo pipefail

PROMPT="${1:-Pick up the cup.}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8081}"
CONDA_ENV="${CONDA_ENV:-lerobot_ght}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROS_WS="$REPO_ROOT/franka_project/ros2_ws"
LOG_DIR="$HOME/franka/logs/joint_validation"

mkdir -p "$LOG_DIR"

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

export PYTHONPATH="$REPO_ROOT/franka_project/ros_lerobot/src${PYTHONPATH:+:$PYTHONPATH}"

TOPIC_INFO="$(
  ros2 topic info /lerobot/franka/joint_action_chunk --verbose 2>/dev/null || true
)"
if [[ "$TOPIC_INFO" != *"lerobot_franka_interfaces/msg/JointActionChunk"* ]]; then
  echo "ERROR: q-pos Joint Gateway action topic has the wrong type or is unavailable"
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

cd "$REPO_ROOT"
python -m lerobot_robot_franka_ros.joint_ros2_client \
  --robot.type=franka_ros_joint \
  --robot.dry_run=false \
  --robot.ros2_interface_only=true \
  --robot.max_observation_age_s=0.5 \
  --robot.camera2_max_skew_s=0.1 \
  --robot.qpos_max_skew_s=0.2 \
  --robot.gripper_max_skew_s=0.1 \
  --robot.observation_buffer_size=64 \
  --robot.gripper_model_open_high=true \
  --robot.gripper_model_open_position=0.944 \
  --robot.action_chunk_validity_s=1.0 \
  --robot.action_execution_timeout_s=120 \
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
  2>&1 | tee "$LOG_DIR/lerobot_client.log"
