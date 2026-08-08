# Franka joint safety gateway

Machine-local consumer for `JointActionChunk`. It validates absolute seven-axis
FastWAM q-pos trajectories against fixed FR3 hard limits, per-waypoint step,
velocity and acceleration limits, optionally retimes a complete plan only by
slowing it down, and publishes controller-authorized `SafeJointCommand` at
100 Hz.

The installed defaults are `enabled=false`, `shadow=true`; they cannot actuate.
In execute mode the eighth model value is range-checked and sent to the Robotiq
gripper action server. The command range is 0.0=open to 0.8=closed with effort
20. An ABORTED close goal at or above 0.5 is treated as nonfatal object contact
while the final close target remains commanded; opening-side failures still
HOLD.

## Low-speed joint safety boundary

This gateway deliberately does **not** load MoveIt and does not check
self-collision, environment collision, or Jacobian singularity. It requires a
fresh measured joint state while arming and while accepting each plan so the
first segment is checked from the real start pose.

Every plan is forced to at least 2x its model period, then retimed further when
needed to satisfy a 0.065 rad step, 0.30 rad/s velocity, 1.0 rad/s²
acceleration, and 0.80 rad total excursion envelope. Two consecutive 30 Hz
samples above 0.55 rad/s,
joint-state loss, non-stale controller rejection, or endpoint loss immediately
HOLD and clear the active plan. Isolated stale/expired feedback and command
timer gaps only warn; a fresh feedback/normal timer tick resets its respective
counter, and three consecutive failures HOLD. The 1 kHz controller
independently limits target slew to 0.35 rad/s.

This still does not prevent a slow collision with a box, table, camera, or the
robot itself. Operators must clear the workspace and keep the physical e-stop
reachable until a measured collision scene is enabled.

The execute workflow is intentionally reduced to three terminal commands:

```bash
# Terminal 1: host tuning, old-stack cleanup, controller, gateway and cameras.
bash franka_project/scripts/start_joint_stack.sh --execute --cam

# Terminal 2: confirm safety and ARM the idle gateway.
bash franka_project/scripts/arm_joint_gateway.sh

# Terminal 3: start the q-pos client with validated defaults.
bash franka_project/scripts/run_joint_client.sh "Pick up the cup."
```

The startup script applies the host tuning with sudo when needed, verifies that
Franka NIC IRQs are on CPU8 with interrupt coalescing disabled, pins the
non-realtime Gateway to CPU9, and starts the FCI controller manager on CPU10.
The 30 Hz qpos relay runs on P-core CPU7. CPU11 remains idle so the FCI
physical core has no SMT competitor. Cameras remain opt-in through `--cam`.

This gateway and `franka_cartesian_safety_gateway` are mutually exclusive for
execution. ARM fails when another `/franka/safe_joint_command` publisher exists.

Topics/services:

- input: `/lerobot/franka/joint_action_chunk`
- ACK: `/lerobot/franka/joint_action_chunk_ack`
- status: `/lerobot/franka/joint_safety_gateway_status`
- controller output: `/franka/safe_joint_command`
- local ARM: `/franka_joint_safety_gateway/set_armed`
