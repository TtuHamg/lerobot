# Periodic and event-driven state of the machine-local safety gateway.
std_msgs/Header header

uint32 schema_version

uint8 DISABLED=0
uint8 SHADOW=1
uint8 ARMED=2
uint8 FAULT=3
uint8 state

bool shadow
bool armed
bool state_fresh
bool preflight_available
bool has_active_plan

string session_id
uint64 plan_id
uint32 next_waypoint_index

# valid_until is the plan's transport acceptance deadline. A plan received
# after this time is rejected. An already accepted plan may contain later
# waypoints; each remains governed by header.stamp + index * period and the
# gateway/controller watchdogs.
builtin_interfaces/Time valid_until
builtin_interfaces/Time next_waypoint_due

string detail
