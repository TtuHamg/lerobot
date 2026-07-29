# Periodic and event-driven state of the machine-local safety gateway.
std_msgs/Header header

uint32 schema_version

uint8 DISABLED=0
uint8 SHADOW=1
uint8 ARMED=2
uint8 HOLD=3
uint8 FAULT=4
uint8 state

bool shadow
bool armed
bool state_fresh
bool preflight_available
bool robot_ready
bool controller_ready
bool applied_feedback_fresh
bool has_active_plan

string session_id
uint64 plan_id
uint32 next_waypoint_index
uint32 accepted_waypoint_index
uint32 applied_waypoint_index
uint32 measured_waypoint_index
uint64 last_command_sequence
uint64 last_applied_sequence
int64 accepted_source_timestep
int64 applied_source_timestep
int64 measured_source_timestep

# Live watchdog diagnostics. Joint error is max(abs(q_measured-q_commanded)).
float64 joint_tracking_error_rad
float64 tcp_position_error_m
float64 tcp_orientation_error_rad
builtin_interfaces/Duration applied_feedback_age
builtin_interfaces/Duration robot_readiness_age

# valid_until is the plan's transport acceptance deadline. A plan received
# after this time is rejected. An already accepted plan may contain later
# waypoints; each remains governed by header.stamp + index * period and the
# gateway/controller watchdogs.
builtin_interfaces/Time valid_until
builtin_interfaces/Time next_waypoint_due

string detail
