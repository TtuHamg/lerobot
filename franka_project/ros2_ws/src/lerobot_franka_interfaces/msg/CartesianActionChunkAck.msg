# Result of the local safety gateway's handling of one CartesianActionChunk.
# This is an acknowledgement of validation/plan replacement, not proof that
# any robot command was sent. Inspect SafetyGatewayStatus for actuation state.
std_msgs/Header header

uint32 schema_version
string session_id
uint64 plan_id

uint8 ACCEPTED_SHADOW=0
uint8 ACCEPTED_FOR_EXECUTION=1
uint8 REJECTED_SCHEMA=10
uint8 REJECTED_FRAME=11
uint8 REJECTED_SHAPE=12
uint8 REJECTED_NONFINITE=13
uint8 REJECTED_QUATERNION=14
uint8 REJECTED_GRIPPER=15
uint8 REJECTED_ORDERING=16
uint8 REJECTED_EXPIRED=17
uint8 REJECTED_SCHEDULE=18
uint8 REJECTED_WORKSPACE=19
uint8 REJECTED_CARTESIAN_STEP=20
uint8 REJECTED_NOT_ARMED=21
uint8 REJECTED_STATE_STALE=22
uint8 REJECTED_PREFLIGHT_UNAVAILABLE=23
uint8 result

bool accepted
bool replaced_active_plan
uint32 waypoint_count
string detail
