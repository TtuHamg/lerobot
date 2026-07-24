# One replaceable, absolute joint-space plan produced by the remote policy.
#
# header.frame_id is the reference frame name (normally "base").
# header.stamp begins interpolation from the measured state to the first target.
std_msgs/Header header

uint32 schema_version
string session_id
uint64 plan_id
# Timestep of the observation used for inference. If a stale action prefix was
# removed client-side, this may be smaller than timesteps[0].
int64 source_timestep

# Observation and transport provenance. These are metadata; the controller
# must schedule target i from header.stamp + (i + 1) * period, not these stamps.
builtin_interfaces/Time client_observation_stamp
builtin_interfaces/Time server_send_stamp

# Absolute transport TTL for accepting this plan. The gateway rejects a newly
# received/replayed plan when local time is after valid_until. Once accepted,
# this short TTL does not truncate the already validated waypoint horizon.
builtin_interfaces/Time valid_until
builtin_interfaces/Duration period

# All sequences must be non-empty. timesteps has length K.
int64[] timesteps

# Absolute Franka joint targets in radians, canonical order (fr3_joint1..7).
# Flattened row-major [K*7]; row i is the seven joint angles for timestep i.
# Reshape with K = len(timesteps).
float64[] positions

# Raw gripper joint position target, one value per timestep (length K).
float32[] gripper
