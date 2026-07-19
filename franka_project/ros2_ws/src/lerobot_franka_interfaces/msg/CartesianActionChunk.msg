# One replaceable, absolute Cartesian plan produced by the remote policy.
#
# header.frame_id is the Cartesian reference frame (normally "base").
# header.stamp is the local ROS time at which the plan becomes eligible.
std_msgs/Header header

uint32 schema_version
string session_id
uint64 plan_id
# Timestep of the observation used for inference. If a stale action prefix was
# removed client-side, this may be smaller than timesteps[0].
int64 source_timestep

# Observation and transport provenance. These are metadata; the controller
# must schedule points from header.stamp + i * period, not from these stamps.
builtin_interfaces/Time client_observation_stamp
builtin_interfaces/Time server_send_stamp

# Absolute transport TTL for accepting this plan. The gateway rejects a newly
# received/replayed plan when local time is after valid_until. Once accepted,
# this short TTL does not truncate the already validated waypoint horizon.
#
# Point i is due at header.stamp + i * period. Gateways must independently drop
# stale points and enforce execution watchdogs; valid_until never authorizes a
# late waypoint. It is normal for later points in a multi-second chunk to be
# scheduled after this sub-second transport TTL.
builtin_interfaces/Time valid_until
builtin_interfaces/Duration period

# Poses are Franka O_T_EE targets: the measured/configured end-effector (TCP)
# in header.frame_id. They are not fr3_link8/flange poses. A gateway using a
# flange IK tip must remove the live F_T_EE transform before solving.
#
# All three sequences must be non-empty and have identical lengths.
int64[] timesteps
geometry_msgs/Pose[] poses

# Normalized Robotiq convention: 0.0 is open and 1.0 is configured closed.
float32[] gripper
