# Result of the local joint safety gateway's handling of one JointActionChunk.
# Acceptance means validation/plan replacement succeeded; it is not proof that
# the controller applied a command. Inspect SafetyGatewayStatus and controller
# feedback for actual execution progress.
std_msgs/Header header

uint32 schema_version
string session_id
uint64 plan_id

uint8 ACCEPTED_SHADOW=0
uint8 ACCEPTED_FOR_EXECUTION=1
uint8 ACCEPTED_PREFLIGHT_ONLY=2
uint8 REJECTED_SCHEMA=10
uint8 REJECTED_FRAME=11
uint8 REJECTED_SHAPE=12
uint8 REJECTED_NONFINITE=13
uint8 REJECTED_GRIPPER=15
uint8 REJECTED_ORDERING=16
uint8 REJECTED_EXPIRED=17
uint8 REJECTED_SCHEDULE=18
uint8 REJECTED_JOINT_LIMIT=19
uint8 REJECTED_JOINT_MOTION=20
uint8 REJECTED_NOT_ARMED=21
uint8 REJECTED_STATE_STALE=22
uint8 REJECTED_PREFLIGHT_UNAVAILABLE=23
uint8 REJECTED_COLLISION=24
uint8 REJECTED_SINGULARITY=25
uint8 result

bool accepted
bool replaced_active_plan
uint32 waypoint_count
string detail
