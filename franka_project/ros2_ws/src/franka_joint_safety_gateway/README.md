# Franka joint safety gateway

Fail-closed machine-local consumer for `JointActionChunk`. It validates absolute
seven-axis FastWAM q-pos trajectories, performs direct MoveIt joint bounds,
Jacobian and full-path collision preflight, retimes unsafe velocity/acceleration
only by slowing down, and publishes controller-authorized `SafeJointCommand` at
200 Hz.

The installed defaults are `enabled=false`, `shadow=true`; they cannot actuate.
The eighth gripper model value is validated as finite and audited but is not
sent to any gripper controller.

This gateway and `franka_cartesian_safety_gateway` are mutually exclusive for
execution. ARM fails when another `/franka/safe_joint_command` publisher exists.

Topics/services:

- input: `/lerobot/franka/joint_action_chunk`
- ACK: `/lerobot/franka/joint_action_chunk_ack`
- status: `/lerobot/franka/joint_safety_gateway_status`
- controller output: `/franka/safe_joint_command`
- local ARM: `/franka_joint_safety_gateway/set_armed`
