#!/bin/bash
# Confirm the local safety conditions, then arm as soon as the q-pos client appears.
set -euo pipefail

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$HOME/franka/franka_ros2_ws/install/setup.bash"
# shellcheck disable=SC1091
source "$HOME/franka/haply_ros/install/setup.bash"
# shellcheck disable=SC1091
source "$HOME/ght_wsp/lerobot/franka_project/ros2_ws/install/setup.bash"
set -u

echo "Waiting up to 120 seconds for the Joint Gateway..."
STACK_READY=false
DEADLINE=$((SECONDS + 120))
STATUS=""
while ((SECONDS < DEADLINE)); do
  STATUS="$(
    timeout 1 ros2 topic echo /lerobot/franka/joint_safety_gateway_status --once \
      2>/dev/null || true
  )"
  STACK_READY=true
  for gate in "armed: false" "state_fresh: true" "preflight_available: true"; do
    if [[ "$STATUS" != *"$gate"* ]]; then
      STACK_READY=false
      break
    fi
  done
  if [[ "$STACK_READY" == "true" ]]; then
    break
  fi
  sleep 0.5
done
if [[ "$STACK_READY" != "true" ]]; then
  echo "ERROR: Joint Gateway did not become ready"
  echo "$STATUS"
  exit 1
fi

echo "Waiting up to 30 seconds for the arm controller..."
CONTROLLER_READY=false
DEADLINE=$((SECONDS + 30))
CONTROLLERS=""
while ((SECONDS < DEADLINE)); do
  CONTROLLERS="$(timeout 2 ros2 control list_controllers 2>/dev/null || true)"
  if awk '$1 == "joint_impedance_ik_controller" && $NF == "active" {ok=1} END {exit !ok}' \
    <<<"$CONTROLLERS"; then
    CONTROLLER_READY=true
    break
  fi
  sleep 0.5
done
if [[ "$CONTROLLER_READY" != "true" ]]; then
  echo "ERROR: joint_impedance_ik_controller is not active"
  echo "$CONTROLLERS"
  exit 1
fi

echo "ARM will allow immediate model-driven robot motion when the client connects."
read -r -p "Confirm workspace clear and e-stop reachable; type ARM: " CONFIRM
if [[ "$CONFIRM" != "ARM" ]]; then
  echo "Canceled; Gateway remains NOT ARMED"
  exit 1
fi

ros2 service call /franka_joint_safety_gateway/set_armed \
  std_srvs/srv/SetBool '{data: true}'
