#!/usr/bin/env bash
# Source the exact ROS overlays required by the LeRobot/Franka log recorder.
# ROS-generated setup files read optional variables without default expansion,
# so nounset cannot remain enabled while sourcing the overlays.
set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROS_PYTHON="${ROS_PYTHON:-/usr/bin/python3}"

SETUP_FILES=(
  /opt/ros/jazzy/setup.bash
  "$LEROBOT_ROOT/franka_project/ros2_ws/install/setup.bash"
  "$HOME/franka/franka_ros2_ws/install/setup.bash"
  "$HOME/franka/haply_ros/install/setup.bash"
)

for setup_file in "${SETUP_FILES[@]}"; do
  if [[ ! -f "$setup_file" ]]; then
    echo "ERROR: required ROS setup file is missing: $setup_file" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$setup_file"
done

if [[ ! -x "$ROS_PYTHON" ]]; then
  echo "ERROR: ROS Python interpreter is not executable: $ROS_PYTHON" >&2
  exit 1
fi

exec "$ROS_PYTHON" "$SCRIPT_DIR/record_lerobot_ros_logs.py" "$@"
