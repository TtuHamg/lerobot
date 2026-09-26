# LeRobot Franka ROS plugin

This is an out-of-tree LeRobot `Robot` plugin for the staged Franka deployment project. It provides
two explicit modes:

- `dry_run=true`: load a validated NPZ observation and write absolute Cartesian actions to JSONL;
- `dry_run=false, ros2_interface_only=true`: subscribe to measured ROS2 observations and publish one
  validated action chunk to the isolated `/lerobot/franka/action_chunk` topic.

The ROS2 mode is an interface only. It does not contain a controller, IK, safety gateway, action
client, or any Franka actuation path. `ros2_interface_only=false` is rejected.

The distribution and import package are both named `lerobot_robot_franka_ros`, so LeRobot's existing
third-party plugin discovery can import it without a core change. The registered robot type is
`franka_ros`.

## Contracts

The full ROS2 topic, synchronization, qpos sideband, custom message, build, client, and safety-boundary
documentation is in [`../ROS2_INTERFACE.md`](../ROS2_INTERFACE.md).

### Dry-run fixture

The NPZ file must contain exactly:

- `state`: `float32`, shape `(10,)`, finite;
- `camera1`: `uint8`, shape `(480, 640, 3)`, RGB;
- `camera2`: `uint8`, shape `(480, 640, 3)`, RGB.

The frozen Phase 1 fixture is committed at
`../fixtures/async/franka_observation_v1.npz` relative to `franka_project`; its companion JSON records
the source dataset frame and SHA-256. Tests also create temporary malformed fixtures for fail-closed
coverage.

### Action

Both backends accept finite absolute targets in this exact feature order:

```text
target.x, target.y, target.z,
target.qx, target.qy, target.qz, target.qw,
target.gripper.closed_0_1
```

The quaternion must already be unit length and the gripper target must be in `[0, 1]`. The plugin does
not decode or execute an action. In ROS2 mode, the project-specific client publishes the accepted
server chunk once; per-waypoint `send_action()` remains bookkeeping only.

## Development install

From the LeRobot checkout:

```bash
uv pip install --no-deps -e franka_project/ros_lerobot
```

Once installed, the stock asynchronous client can select `--robot.type=franka_ros` for dry-run. The
ROS2 interface must use the chunk-aware entry point:

```bash
python -m lerobot_robot_franka_ros.ros2_client ... \
  --robot.type=franka_ros \
  --robot.dry_run=false \
  --robot.ros2_interface_only=true \
  --aggregate_fn_name=latest_only
```

ROS imports remain lazy. Build and source `../ros2_ws` only before starting the ROS2 mode; importing
the plugin or running dry-run does not require `rclpy`.

## Joint Gateway client: q-pos and EEF

The `franka_ros_joint` plugin and `joint_ros2_client` share the execution ACK,
timeout, and Joint Gateway path for two policy contracts:

- `--robot.action_space=qpos` (default): state/action are seven joint positions
  plus the checkpoint-specific gripper value.
- `--robot.action_space=eef`: state/action are
  `[xyz, rot6d.col0, rot6d.col1, gripper.closed_0_1]`. The client applies
  Gram-Schmidt to each network rotation, derives the live link8-to-TCP
  transform from measured q-pos/EEF pose, solves the complete chunk with
  warm-started Pinocchio IK, and publishes the resulting `JointActionChunk`.

The machine-local wrapper keeps q-pos behavior unchanged:

```bash
bash franka_project/scripts/run_joint_client.sh --qpos "Pick up the cup."
bash franka_project/scripts/run_joint_client.sh --eef "stack the red block"
```

### Client-side replanning

Keep the server prediction horizon at 32 actions and choose a shorter execution
prefix on the client:

```bash
SERVER_ADDRESS=127.0.0.1:15174 bash franka_project/scripts/run_joint_client.sh \
  --qpos --replan 8 "stack the yellow rectangular block on the tower"
```

`--replan N` (also `--replan=N` or `REPLAN_STEPS=N`) accepts integers from 1 to 32.
The CLI flag overrides the environment variable. The client retains only the first
N actions of each response in both its local queue and the chunk published to the
Joint Gateway; the unused suffix is discarded, not executed later. A shorter
server response executes at most N actions. The next observation waits for both
local queue depletion and gateway completion/ACK (or rejection), preserving all
existing safety gates, limits, and timeouts. This does not interrupt an active plan.
It works for both qpos and EEF; EEF IK sees only the retained prefix.

The Python entry point accepts `--replan_steps=8` with
`--chunk_size_threshold=0.0 --aggregate_fn_name=latest_only
--enable_pending_observation=true`. In this mode, fresh observations use the next
unused logical action tick. Servers that number their first prediction with the
observation tick therefore no longer lose that first target on subsequent chunks.
Observation capture timestamps and transport retry identifiers remain unchanged.
Prefix selection happens before stale-action filtering, so a replay cannot promote
the discarded tail into a new execution chunk.

Without the flag/environment variable, the previous full-chunk behavior is
unchanged. Explicit `--replan 32` uses the new sequential tick convention while
retaining the full 32-action horizon. No policy weights or server restart are needed;
restart the client with the desired setting. This is a startup parameter, not a
live interactive control. Logs report `Replan: received=32 selected=8 accepted=8
discarded=24` for an eight-action prefix.

Smaller prefixes allow earlier reobservation after execution, but do not make
inference faster or guarantee higher control frequency: inference, gateway
retiming, and diagnostics still add latency. Validate in the actual gateway
shadow mode before armed operation; the launch command does not itself enable
shadow or arm hardware.

Run the no-ROS contract regressions from the repository root:

```bash
python franka_project/tests/test_joint_client_replan.py -v
```

These execute the client/config method bodies with fake observations and a fake
gateway, covering prefix selection, repeated replans, retries, CLI validation,
and the physical-completion gate without importing optional robot dependencies.

EEF mode requires `~/franka/config/fr3_ik.urdf` (override with `IK_URDF`) and a
Python environment containing `pinocchio`. It still requires the separately
started and explicitly armed Joint Gateway; the client never arms hardware.

Use the complete, fixed 15 Hz client/server commands in
[`../ASYNC_CLIENT_SERVER_RUNBOOK.md`](../ASYNC_CLIENT_SERVER_RUNBOOK.md). In particular, the camera
`rename_map` and `aggregate_fn_name=latest_only` remain required parts of the inference contract.
