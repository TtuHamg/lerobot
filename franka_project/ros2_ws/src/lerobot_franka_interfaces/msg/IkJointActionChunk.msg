# Lightweight IK result corresponding to one CartesianActionChunk.
#
# The tuple (session_id, plan_id) identifies the source Cartesian plan.
# positions is row-major [waypoint_count, 7] in joint_names order.
std_msgs/Header header

uint32 schema_version
string session_id
uint64 plan_id
int64 source_timestep
builtin_interfaces/Duration period

uint32 waypoint_count
string[7] joint_names
float64[] positions
