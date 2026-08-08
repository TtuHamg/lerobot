#!/bin/bash
# Monitor raw q-pos model action chunks from a fresh terminal.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROS_WS="$(cd -- "$SCRIPT_DIR/../ros2_ws" && pwd)"

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$ROS_WS/install/setup.bash"
set -u

exec ros2 topic echo /lerobot/franka/joint_action_chunk "$@"
