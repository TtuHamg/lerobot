# LeRobot Franka ROS plugin

This is an out-of-tree LeRobot `Robot` plugin for the Franka deployment project.

Phase 1 is deliberately **dry-run only**:

- observations are loaded from one validated NPZ fixture;
- absolute Cartesian actions are written to a JSONL sink;
- no ROS package, controller, or robot is imported or contacted;
- `dry_run=false` is rejected rather than silently falling back to the sink.

The distribution and import package are both named `lerobot_robot_franka_ros`, so LeRobot's existing
third-party plugin discovery can import it without a core change. The registered robot type is
`franka_ros`.

## Fixture contract

The NPZ file must contain exactly:

- `state`: `float32`, shape `(10,)`, finite;
- `camera1`: `uint8`, shape `(480, 640, 3)`, RGB;
- `camera2`: `uint8`, shape `(480, 640, 3)`, RGB.

The frozen Phase 1 fixture is committed at
`../fixtures/async/franka_observation_v1.npz` relative to `franka_project`; its companion JSON records
the source dataset frame and SHA-256. Tests also create temporary malformed fixtures for fail-closed
coverage.

## Action contract

The JSONL sink accepts finite absolute targets in this exact feature order:

```text
target.x, target.y, target.z,
target.qx, target.qy, target.qz, target.qw,
target.gripper.closed_0_1
```

The quaternion must already be unit length and the gripper target must be in `[0, 1]`. The plugin never
normalizes, clips, decodes, or executes an action.

## Development install

From the LeRobot checkout:

```bash
uv pip install --no-deps -e franka_project/ros_lerobot
```

Once installed, the stock asynchronous client can select `--robot.type=franka_ros`. A fixture and log
path are mandatory. This package intentionally does not provide another client, tunnel, or inference
implementation.

Use the complete, fixed 15 Hz client/server commands in
[`../ASYNC_CLIENT_SERVER_RUNBOOK.md`](../ASYNC_CLIENT_SERVER_RUNBOOK.md). In particular, the camera
`rename_map` and `aggregate_fn_name=latest_only` are required parts of the Phase 1 contract.
