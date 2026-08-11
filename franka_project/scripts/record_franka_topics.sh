#!/usr/bin/env bash
# Run the topic recorder with the ROS overlays used by the Franka client.
#
# Interactive multi-session recording (key-driven start/stop/save):
#   ./record_franka_topics.sh --interactive --topics camera1 camera2 qpos eef gripper
#     SPACE/ENTER  start a recording, press again to stop and save
#     q            quit (saves the current recording first)
#   With --interactive, --output is the parent dir for timestamped recordings.
set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROS_PYTHON="${ROS_PYTHON:-/usr/bin/python3}"

SETUP_FILES=(
  /opt/ros/jazzy/setup.bash
  /home/pnp/franka/franka_ros2_ws/install/setup.bash
  /home/pnp/franka/haply_ros/install/setup.bash
  "$LEROBOT_ROOT/franka_project/ros2_ws/install/setup.bash"
)

for setup_file in "${SETUP_FILES[@]}"; do
  if [[ -f "$setup_file" ]]; then
    # ROS-generated setup files read optional variables without default expansion.
    # shellcheck disable=SC1090
    source "$setup_file"
  elif [[ "$setup_file" == "/opt/ros/jazzy/setup.bash" ]]; then
    echo "ERROR: ROS Jazzy setup file is missing: $setup_file" >&2
    exit 1
  fi
done

if [[ ! -x "$ROS_PYTHON" ]]; then
  echo "ERROR: ROS Python interpreter is not executable: $ROS_PYTHON" >&2
  exit 1
fi

exec "$ROS_PYTHON" "$SCRIPT_DIR/record_franka_topics.py" "$@"
