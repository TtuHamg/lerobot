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

### Policy wire profiles

The ROS2 client remains policy-agnostic after the server boundary: both PI0 and FastWAM return the
same canonical absolute8 action. At startup it only selects and validates the camera wire profile:

| policy | client `rename_map` |
|---|---|
| `pi0` | `camera1 -> base_0_rgb`, `camera2 -> left_wrist_0_rgb` (auto-filled when omitted) |
| `fastwam` | `{}` |

The ROS2 client also requires `action_offset=1` and the frozen Cartesian `base_frame=base` used by
both checkpoint families. In live ROS2 mode the client reads `open_position` and
`closed_position` once from `/franka_gripper_follower` during startup and freezes that calibration;
startup fails if the parameter service or either endpoint is invalid. The current FastWAM
move-cups artifact additionally requires `gripper_max_skew_s<=0.01` for the live Robotiq joint.
Its recorded
camera, EEF, and gripper topics/joint name are also fixed, with `camera2_max_skew_s<=0.1` and
`eef_max_skew_s<=0.05`; startup rejects CLI overrides that drift from this sensor contract.

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
  --robot.base_frame=base \
  --aggregate_fn_name=latest_only
```

### Optional action visualization

The chunk-aware client can own the read-only action visualizer and RViz lifecycle. The switch is a
top-level client option, not a `robot` option, and defaults to disabled:

```bash
python -m lerobot_robot_franka_ros.ros2_client ... \
  --robot.type=franka_ros \
  --visualize_action=true
```

Use `--visualization_launch_rviz=false` to run only the visualization/marker node without opening
RViz. The client passes its resolved action chunk, current EEF pose, camera1, camera2, and base-frame
configuration to the launch file. If the requested launch cannot start, the client exits with a clear
error; on normal exit or interruption it stops and reaps the launch process group.

Build and source the in-repository ROS workspace before using the switch. Source the Franka and Haply
underlays first; the Haply overlay supplies the observed `franka_safety_interfaces` messages:

```bash
source /opt/ros/jazzy/setup.bash
source /home/pnp/franka/franka_ros2_ws/install/local_setup.bash
source /home/pnp/franka/haply_ros/install/local_setup.bash
cd /home/pnp/Projects/lerobot/franka_project/ros2_ws
colcon build --symlink-install --packages-up-to franka_lerobot_rviz
source install/local_setup.bash
```

The standalone launch remains supported when the client switch is left disabled, for example from a
separate terminal:

```bash
ros2 launch franka_lerobot_rviz action_viz.launch.py
```

Use either client-owned or standalone launch for a run so duplicate visualizer/RViz nodes are not
started accidentally.

### Low-load web dashboard

The client can alternatively own a localhost-only rolling web dashboard:

```bash
python -m lerobot_robot_franka_ros.ros2_client ... \
  --robot.type=franka_ros \
  --visualize_action_web=true
```

The browser opens at `http://127.0.0.1:8768`. The dashboard keeps a bounded rolling history and
shows Cartesian chunks, quaternion/gripper values, and the lightweight gateway's associated
seven-joint IK result. Both camera subscriptions are absent by default; enabling a camera in the
page creates a throttled thumbnail subscription, and disabling it destroys that subscription.
`--visualize_action` and `--visualize_action_web` are mutually exclusive.
The pose-frame selector switches both planned and measured curves between canonical `O_T_EE`
(`EEF / TCP`) and `O_T_link8` (`link8 / flange`). It captures `F_T_EE` once and immediately removes
the 1 kHz RobotState subscription. The initial selector follows `robot.policy_eef_frame`.

Useful options:

```bash
--web_dashboard_port=8768 \
--web_dashboard_history_seconds=60 \
--web_dashboard_camera_fps=3 \
--web_dashboard_open_browser=true
```

### Policy EEF frame compatibility

Choose the Cartesian frame semantics used by the checkpoint explicitly:

```bash
# Haply/ROS datasets: configured physical TCP O_T_EE
--robot.policy_eef_frame=eef

# Frankateleop/Polymetis datasets: panda_link8 / fr3_link8 flange pose
--robot.policy_eef_frame=link8
```

In `link8` mode the client captures the fixed live `F_T_EE` once, removes the 1 kHz RobotState
subscription, converts the observed `O_T_EE` to `O_T_link8` before inference, and converts returned
link8 targets back to canonical `O_T_EE` before ROS publication. The gateway therefore retains one
unambiguous IK contract. Do not infer this mode from pose values; a checkpoint trained on mixed
frame semantics should be regenerated when possible.

### Selective MCAP / MP4 recording

After the client and ROS publishers are running, the read-only recorder can select aliases or any
absolute ROS topic. Mixed/non-image selections use MCAP; a camera-only selection uses one MP4 per
camera:

```bash
# camera1.mp4 + camera2.mp4
franka_project/scripts/record_franka_topics.sh --topics camera1 camera2

# MCAP containing raw camera, joint-position, and EEF streams
franka_project/scripts/record_franka_topics.sh \
  --topics camera1 camera2 current_pos eef
```

See [`../markdown/FRANKA_TOPIC_RECORDING.md`](../markdown/FRANKA_TOPIC_RECORDING.md) for aliases,
custom topics, duration/output options, QoS overrides, and the distinction between raw ROS streams
and the synchronized observation snapshot selected inside the client.

ROS imports remain lazy. Build and source `../ros2_ws` only before starting the ROS2 mode; importing
the plugin or running dry-run does not require `rclpy`.

Use the complete PI0 and FastWAM client/server commands in
[`../ASYNC_CLIENT_SERVER_RUNBOOK.md`](../ASYNC_CLIENT_SERVER_RUNBOOK.md). In particular, the camera
wire profile, `action_offset=1`, matching FPS/chunk size, and `aggregate_fn_name=latest_only` remain
required parts of the inference contract.
