# Franka LeRobot ROS2 接口

> 状态：ROS2 interface runtime 已实现，本轮未启动 ROS graph，未驱动 Franka。
>
> ROS 2、LeRobot Client、KML gateway、Franka safety gateway 以及
> `DISABLED/SHADOW/HOLD/ARMED/FAULT` 的综合说明，请优先阅读
> [`FRANKA_LEROBOT_ROS2_COMMUNICATION_GUIDE_CN.md`](./FRANKA_LEROBOT_ROS2_COMMUNICATION_GUIDE_CN.md)。

本文档描述当前已实现的“非执行 ROS2 边界”：它从 ROS2 读取两路图像、
EEF pose、Franka qpos 和 gripper joint state，并把 PolicyServer 返回的完整
absolute Cartesian action chunk 发布到隔离的 `/lerobot/` topic。

这个 runtime 不包含 controller、IK、trajectory interpolation、safety gateway、watchdog
或硬件 action client。发布 action chunk 不等于授权机器人运动。

## 1. 已实现的模块边界

| 模块 | 已实现职责 |
|---|---|
| `ros2_contract.py` | 纯 Python 图像解码、因果同步 cache、10D policy projection、qpos sideband 和 action chunk 验证 |
| `ros2_runtime.py` | 延迟加载 ROS bindings，创建 private `rclpy` context/node/executor，订阅 observation，发布 `CartesianActionChunk` |
| `ros2_backend.py` | 管理 ROS session/plan ID、observation cache、chunk TTL；逐步 `send_action()` 只记录 bookkeeping，不发布 ROS 命令 |
| `franka_ros.py` | 在 `dry_run=true` 和非执行 ROS2 backend 之间选择，保持 LeRobot `Robot` 特征契约 |
| `ros2_client.py` | 继承 stock `RobotClient`，按 PI0/FastWAM 选择 camera wire profile，并在 `latest_only` queue aggregation 边界将完整 accepted chunk 发布一次 |
| `lerobot_franka_interfaces` | 提供 `CartesianActionChunk.msg`，以及供未来 gateway 使用的 `CartesianActionChunkAck.msg`、`SafetyGatewayStatus.msg` wire schema |

ROS imports 是 lazy 的：import `lerobot_robot_franka_ros` 或执行 dry-run 时不会 import
`rclpy`/ROS messages。只有 `dry_run=false` 后 `Ros2Runtime.start()` 被显式调用时，
runtime 才导入：

- `rclpy`;
- `geometry_msgs.msg.Pose` / `PoseStamped`;
- `sensor_msgs.msg.Image` / `JointState`;
- `lerobot_franka_interfaces.msg.CartesianActionChunk`。

因此 LeRobot plugin discovery 和无 ROS 单元测试仍可在未 source ROS 的 Python 环境中运行。

## 2. 启用模式与安全门

### 2.1 dry-run

```yaml
robot.type: franka_ros
robot.dry_run: true
robot.fixture_path: <npz>
robot.action_log_path: <jsonl>
```

此模式不创建 ROS runtime，保持现有 fixture -> PI0 -> JSONL 验证链路。

### 2.2 非执行 ROS2 interface

```yaml
robot.type: franka_ros
robot.dry_run: false
robot.ros2_interface_only: true
```

`dry_run=false + ros2_interface_only=true` 会启动已实现的 private rclpy runtime，但只订阅
observation topics 并向 `/lerobot/franka/action_chunk` 发布计划。

下列组合必须 fail closed：

```yaml
robot.dry_run: false
robot.ros2_interface_only: false
```

当前 `FrankaRosConfig` 会报错 `Franka actuation is not implemented`。另外，ROS2 interface
模式不允许同时配置 `fixture_path` 或 `action_log_path`，防止两种 backend 被混用。

### 2.3 当前明确不会做的事

- 不创建 controller publisher 或 ROS action client；
- 不调用 Franka controller、gripper driver 或硬件 service；
- 不做 IK、collision check 或轨迹插值；
- `FrankaRos.send_action()` 在 ROS2 模式中只更新本地 bookkeeping，不产生第二条 ROS command path；
- `/lerobot/franka/action_chunk` 不是 controller topic，不得 remap/relay 到真机 controller；
- 本轮实现和测试没有启动 ROS graph，没有发布真实 ROS 消息，没有连接硬件。

## 3. `FrankaRosConfig` 的实际默认值

下列名称和字符串与当前 `FrankaRosConfig` 一致。

### 3.1 node、frame 和 topics

| 配置字段 | 默认值 | ROS type | 方向 |
|---|---|---|---|
| `ros2_node_name` | `lerobot_franka_interface` | - | private runtime node |
| `base_frame` | `base` | 必须保持 `base` | EEF/action Cartesian reference frame；PI0/FastWAM checkpoint 均冻结在该坐标系 |
| `camera1_topic` | `/camera1/camera1/color/image_raw` | `sensor_msgs/msg/Image` | ROS -> client |
| `camera2_topic` | `/camera2/camera2/color/image_raw` | `sensor_msgs/msg/Image` | ROS -> client |
| `eef_pose_topic` | `/franka_robot_state_broadcaster/current_pose` | `geometry_msgs/msg/PoseStamped` | ROS -> client |
| `qpos_topic` | `/franka/joint_states` | `sensor_msgs/msg/JointState` | ROS -> client |
| `gripper_topic` | `/gripper/joint_states` | `sensor_msgs/msg/JointState` | ROS -> client |
| `action_chunk_topic` | `/lerobot/franka/action_chunk` | `lerobot_franka_interfaces/msg/CartesianActionChunk` | client -> isolated ROS boundary |

所有 topic 必须是无空格的 absolute ROS topic，且互不相同。`action_chunk_topic`
还必须保持在 `/lerobot/` namespace 下。

当前 runtime 不做 TF lookup/transform，也没有 `eef_frame` 配置。EEF `PoseStamped.header.frame_id`
必须已经等于 `base_frame`（默认 `base`），否则样本被拒绝。

### 3.2 joints、gripper 和时间参数

```yaml
arm_joint_names:
  - fr3_joint1
  - fr3_joint2
  - fr3_joint3
  - fr3_joint4
  - fr3_joint5
  - fr3_joint6
  - fr3_joint7

gripper_joint_name: robotiq_85_left_knuckle_joint
gripper_endpoint_parameter_node: /franka_gripper_follower
gripper_endpoint_parameter_timeout_s: 5.0
gripper_open_position: 0.0
gripper_closed_position: 0.8

observation_buffer_size: 32
max_observation_age_s: 0.25
camera2_max_skew_s: 0.05
eef_max_skew_s: 0.05
qpos_max_skew_s: 0.1
gripper_max_skew_s: 0.05
action_chunk_validity_s: 0.5
ros2_shutdown_timeout_s: 5.0
quaternion_norm_tolerance: 0.001
```

live ROS2 client 启动时从 `/franka_gripper_follower` 一次性读取 `open_position` 和
`closed_position` 并冻结。本机 Robotiq 2F-85 的真实边界为 `0.0/0.8 rad`，同时要求
`gripper_max_skew_s<=0.01`。参数服务不可用或端点非法时，client fail-closed。

gripper callback 在 `JointState.name` 中定位 `gripper_joint_name`，用下式转换并 clip 到
`[0,1]`：

```text
closed_0_1 =
    (joint_position - gripper_open_position)
    / (gripper_closed_position - gripper_open_position)
```

反向执行时，当前 `CartesianActionChunk.gripper` 仍只承载 canonical `closed_0_1`；未来唯一的
真机 gateway 应在驱动边界做：

```text
joint_target = gripper_open_position
             + closed_0_1 * (gripper_closed_position - gripper_open_position)
```

对已确认的 `0.0/0.8` 标定即 `joint_target=0.8*closed_0_1`。这里没有 FastWAM 的 `0.04`：
`0.04` 仅属于 server 输入 proprio 的 pseudo-finger 编码，绝不能用于输出关节命令。

## 4. Observation 同步和 policy projection

### 4.1 camera1 因果 anchor

`RosObservationCache` 是 thread-safe bounded cache，用最新 `camera1.header.stamp` 作为 anchor。
对 camera2、EEF、qpos 和 gripper，cache 只选择：

1. source timestamp `<= camera1 anchor` 的样本；
2. 满足该 source 的 `*_max_skew_s` 限制的最新样本；
3. 按本地 `received_monotonic_ns` 计算，年龄不超过 `max_observation_age_s=0.25`
   的样本。

这保证 observation 不用“未来 robot state”配对过去 image。任一 required source
缺失、过期或超出 skew，`get_observation()` 都 fail closed，不返回部分 snapshot。

ROS callbacks 在 private `SingleThreadedExecutor` daemon thread 中只做解码/验证并更新 cache。
LeRobot 15 Hz control loop 不在读 observation 时阻塞等待多 topic。

### 4.2 图像契约

- ROS `rgb8` 和 `bgr8` 被支持；`bgr8` 会显式转 RGB；
- 支持 `step > width*3` 的 row padding，解码后去掉 padding；
- 其他 encoding 和 compressed image 被拒绝；
- camera1/camera2 必须为 HWC `uint8 [480,640,3]`；
- 返回的图像和 sideband 都是 copy，不会将 policy preprocessing 的可变操作传回 callback cache；
- PI0 wire profile 将 camera1 映射为 `base_0_rgb`、camera2 映射为 `left_wrist_0_rgb`；
- FastWAM wire profile 保持原始 `camera1`/`camera2` key，`rename_map={}`。

### 4.3 冻结的 10D policy state

EEF quaternion 使用 `xyzw`，验证并归一化后转成 rotation matrix 前两列。
policy-facing scalar 顺序仍为：

```text
0  eef.x
1  eef.y
2  eef.z
3  eef.rot6d.col0.x
4  eef.rot6d.col0.y
5  eef.rot6d.col0.z
6  eef.rot6d.col1.x
7  eef.rot6d.col1.y
8  eef.rot6d.col1.z
9  gripper.closed_0_1
```

`RosObservationSnapshot.as_policy_observation()` 仅返回：

```text
10 scalar state fields
camera1: uint8 [480,640,3], RGB, HWC
camera2: uint8 [480,640,3], RGB, HWC
```

task 仍由 RobotClient 添加，normalization 仍由 checkpoint processor 执行。

### 4.4 qpos sideband

`/franka/joint_states` 的 `JointState` 必须精确包含配置的 7 个不重复 arm joint name。
cache 按 `arm_joint_names` 重排 position，生成 finite `float64 [7]` qpos（rad）。

qpos 可通过：

```python
robot.get_qpos()
robot.get_ros_sideband()
```

获取。sideband 还包含 anchor/source timestamps、EEF xyz+xyzw、gripper 和 frame。qpos
不加入 `FrankaRos.observation_features`，也不进入 PI0/FastWAM 的 canonical wire state10。
如果把 qpos 直接加入 policy features，state 将从 10D 变为 17D，并被
Franka PolicyServer 的 exact contract 拒绝。

## 5. `CartesianActionChunk.msg`

消息定义位于：

```text
franka_project/ros2_ws/src/lerobot_franka_interfaces/msg/CartesianActionChunk.msg
```

字段与当前 `.msg` 逐字一致：

```text
std_msgs/Header header

uint32 schema_version
string session_id
uint64 plan_id
int64 source_timestep

builtin_interfaces/Time client_observation_stamp
builtin_interfaces/Time server_send_stamp

builtin_interfaces/Time valid_until
builtin_interfaces/Duration period

int64[] timesteps
geometry_msgs/Pose[] poses
float32[] gripper
```

### 5.1 字段语义

- `header.frame_id` 等于 `base_frame`，默认 `base`；
- `header.stamp` 是 runtime 开始发布该计划时的本地 ROS time；
- `schema_version` 当前只允许 `1`；
- `session_id` 在 ROS2 backend 每次 connect 时生成；
- `plan_id` 在该 session 内从 `0` 单调增加；
- `source_timestep` 是 PolicyServer 本次推理所用 observation 的 timestep；若 client
  丢弃了已执行的 stale prefix，它可以小于 `timesteps[0]`；
- `client_observation_stamp` 是该 source observation 的 wall-clock timestamp，属于
  provenance，不是未来 execution deadline；
- `server_send_stamp` 是 PolicyServer 在整个 chunk 上标注的同一
  `server_send_timestamp`；
- `valid_until` 是发布时将剩余 monotonic TTL 映射到本地 ROS clock 得到的过期时间；
- `period` 当前等于 RobotClient `environment_dt`，PI0 15 Hz 示例约为 `66,666,667 ns`，
  FastWAM 30 Hz 示例约为 `33,333,333 ns`；
- `timesteps` 必须与 actions 一一对应且严格连续；
- `poses` 是 `[x,y,z,qx,qy,qz,qw]`，位置单位 meter，quaternion 使用 `xyzw`；
- `gripper` 与 `poses` 等长，`0.0=open`、`1.0=configured closed`。

`timesteps`/`poses`/`gripper` 必须非空且长度一致。每个 action 在转消息前均被验证：

- floating `float32 [K,8]` 且 finite；
- quaternion norm 在 `0.001` tolerance 内为 1；
- gripper 在 `[0,1]`；
- timestep 连续；
- 同一 chunk 的 `server_send_timestamp` 相同；
- 每个 `TimedAction.timestamp` 满足
  `client_observation_stamp + (timestep-source_timestep) * period`。

`action_chunk_validity_s=0.5` 使用本地 monotonic clock 检查。如果 chunk 在转 ROS
消息前已过期，runtime 拒绝发布。

## 6. custom RobotClient chunk hook

启动入口是：

```bash
python -m lerobot_robot_franka_ros.ros2_client [RobotClientConfig arguments]
```

该模块中的 `FrankaRos2RobotClient` 继承 stock `RobotClient`。它强制：

```text
robot.type = franka_ros
robot.dry_run = false
robot.ros2_interface_only = true
aggregate_fn_name = latest_only
```

hook 实现在 `_aggregate_action_queues()` 边界：

```text
GetActions
  -> pickle.loads(): list[TimedAction]
  -> stock receive_actions device handling
  -> FrankaRos2RobotClient._aggregate_action_queues()
       1. 丢弃 timestep <= latest_action 的 stale prefix
       2. 可选地截取前 max_action_chunk_waypoints 个 fresh actions
       3. 保留原始 source observation 元数据
       4. 用同一份 accepted actions 替换 stock 本地 queue
       5. 将 accepted actions 作为一个完整 ROS chunk 发布一次
  -> stock control loop 逐点 pop，仅更新 bookkeeping
```

当前 `_ready_to_send_observation()` 会等最新 ROS plan 报告执行完成后才采下一帧并开始下一次
推理。因此这是第一阶段的 stop-and-go 策略：FastWAM 的 32 steps / 30 Hz chunk 约执行
`1.067 s`，chunk 之间还会停顿一次完整 inference latency。若后续要求连续运动，必须先在
safety gateway 中实现带 provenance/TTL 的安全 plan replacement，再让推理与当前 plan 重叠；
不能只删除这个 gate。

新 observation 的逻辑 timestep 在发送 RPC 前按下式生成：

```text
observation_timestep = max(latest_action + action_offset, 0)
```

`action_offset` 是 RobotClient 顶层配置；stock 默认值仍为 `0`，但
`FrankaRos2RobotClient` 必须显式使用 `--action_offset=1`，否则会在每个新 chunk 中固定丢掉
第一条 waypoint。设置为 `1` 后，PolicyServer 的第一条 action 从本地已消费 cursor 的下一 timestep
开始编号。假设创建 observation 时 `latest_action=L`，且等待返回期间该 cursor 没有继续推进，
50-step server chunk 会从 `L+1` 到 `L+50`，因而 50 条都会通过 fresh filter。若推理/传输期间
cursor 又推进了 `m` 步，前 `m` 条仍属于真正 stale action，会按现有规则裁剪；因此该参数消除
固定的一步重叠，但不承诺任何时延下都无条件得到 50 条。

该偏移只改变 observation/action 的逻辑编号。Pending retry 仍复用同一 observation、timestep
和 `request_id`，gRPC ACK 状态机、ROS chunk schema 以及 gateway 的验证/武装协议均不变。

因此 ROS 看到的是本地 client 接受后的 fresh suffix，不是包含已执行 stale prefix
的原始 server payload。如果 chunk 转换/发布失败，client 立即设置 shutdown event
并重新抛出异常；不 fallback 为逐 waypoint ROS 发布。

`robot.max_action_chunk_waypoints` 是 Franka ROS2 专属的可选上限，默认 `None`，即保留
完整 fresh suffix。配置为 `30` 时，Server 仍可按 `--actions_per_chunk=50` 生成并发送
50 条；Client 先去掉 stale prefix，再取最靠近当前 observation 的前 30 条 fresh action。
这同一份最多 30 条的数据既替换本地 action queue，也发布到 ROS，避免 `latest_action`
消费 Gateway 从未收到的尾部 waypoint。若 fresh suffix 本来不足 30 条，则不会补齐。

gRPC action-delivery ACK 仍确认整个 Server delivery 已按 Client 配置完成 commit；被上限明确
丢弃的尾部不会稍后重发。用于 `chunk_size_threshold` 的有效 chunk size 也同步按 30
归一化。健康且 armed 的 Gateway 下，下一次 observation 仍受
`plan_execution_complete()` 门控：30 点在 15 Hz 下名义执行约 2 秒，并不会因为 threshold
达到 0.5 就在第 15 点自动开始下一次推理。

不得把 `aggregate_fn_name` 改成 `weighted_average`：absolute quaternion 不能做逐元素线性平均。

## 7. 已实现 QoS

runtime 当前只构造两个 QoS profile：

| 用途 | Reliability | History | Depth | Durability |
|---|---|---|---:|---|
| camera1、camera2、EEF、qpos、gripper 全部 subscription | Best effort | Keep last | 5 | Volatile |
| `CartesianActionChunk` publisher | Reliable | Keep last | 1 | Volatile |

action publisher 使用 `Volatile` ，避免新 subscriber 自动获得已经过期的旧计划。
当前没有配置 DDS deadline 或 liveliness lease；它们也不能代替后续本地 watchdog。

## 8. 构建和测试

### 8.1 无 ROS graph 单元/集成测试

下列测试使用 pure-Python contracts 和 fake runtime/message，不启动 ROS graph：

```bash
cd /home/pnp/Projects/lerobot

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/pnp/miniconda3/envs/lerobot/bin/python -m pytest -q \
  franka_project/tests/test_franka_ros_dry_run.py \
  franka_project/tests/test_franka_ros2_contract.py \
  franka_project/tests/test_franka_ros2_backend.py \
  franka_project/tests/test_franka_ros2_runtime.py \
  franka_project/tests/test_franka_async_lightweight_e2e.py
```

覆盖点包括：

- lazy ROS import；
- `rgb8`/`bgr8`/row padding 解码；
- camera1 因果同步、skew、freshness 和 thread-safe snapshot；
- exact 10D policy projection 和 qpos sideband copy；
- absolute8 shape/finite/quaternion/gripper/timing/session/TTL 契约；
- `.msg` 字段转换；
- custom client 只发布 fresh accepted chunk；
- publication failure fail closed；
- `dry_run=false + ros2_interface_only=false` 被拒绝。

### 8.2 ROSIDL 构建

ROS Jazzy 生成的 Python extension 必须使用系统 Python 构建，不得让 colcon 拾取
Miniconda Python：

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pnp/Projects/lerobot/franka_project/ros2_ws

colcon build --symlink-install \
  --packages-select lerobot_franka_interfaces \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

source install/setup.bash
ros2 interface show lerobot_franka_interfaces/msg/CartesianActionChunk
```

`ros2_ws` 当前只包含 `lerobot_franka_interfaces`；Python node/runtime 由
`lerobot_robot_franka_ros` plugin 提供，不存在一个另外的 `lerobot_franka_bridge` colcon package。

## 9. 未来 ROS graph smoke（本轮未执行）

下列步骤只验证 observation 与隔离的 chunk topic，仍不得连接 controller。

### 9.1 预检

```bash
source /home/pnp/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
source /opt/ros/jazzy/setup.bash
source /home/pnp/Projects/lerobot/franka_project/ros2_ws/install/setup.bash

python -c 'import rclpy; from lerobot_franka_interfaces.msg import CartesianActionChunk; print("ROS imports OK")'

ros2 topic info --verbose /camera1/camera1/color/image_raw
ros2 topic info --verbose /camera2/camera2/color/image_raw
ros2 topic info --verbose /franka_robot_state_broadcaster/current_pose
ros2 topic info --verbose /franka/joint_states
ros2 topic info --verbose /gripper/joint_states
```

必须确认 type、QoS、header stamp、EEF `frame_id=base`、相机实际 encoding/分辨率、
7 个 arm joint name，以及 Robotiq joint 名与 open/closed 标定。

### 9.2 启动自定义 client

PolicyServer 和 KML tunnel 已启动后，使用专用 client，不再使用 stock
`lerobot.async_inference.robot_client` 入口：

PI0 示例：

```bash
cd /home/pnp/Projects/lerobot

python -m lerobot_robot_franka_ros.ros2_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=franka_ros \
  --robot.id=franka_ros2_interface \
  --robot.dry_run=false \
  --robot.ros2_interface_only=true \
  --robot.max_action_chunk_waypoints=30 \
  --robot.base_frame=base \
  --task='stack the cups' \
  --policy_type=pi0 \
  --pretrained_name_or_path=server-owned \
  --policy_device=cpu \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --action_offset=1 \
  --fps=15 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only \
  --enable_pending_observation=true \
  --pending_observation_timeout_s=10 \
  '--rename_map={"observation.images.camera1":"observation.images.base_0_rgb","observation.images.camera2":"observation.images.left_wrist_0_rgb"}'
```

当前 FastWAM move-cups checkpoint 示例（注意 task、30 Hz、32 steps、空 rename map 和
实时 Robotiq 的 `0.8/0.01` gripper contract）：

```bash
python -m lerobot_robot_franka_ros.ros2_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=franka_ros \
  --robot.id=franka_fastwam_ros2 \
  --robot.dry_run=false \
  --robot.ros2_interface_only=true \
  --robot.base_frame=base \
  --robot.gripper_max_skew_s=0.01 \
  --task='move the paper cup from one end of the can to the other.' \
  --policy_type=fastwam \
  --pretrained_name_or_path=server-owned \
  --policy_device=cpu \
  --client_device=cpu \
  --actions_per_chunk=32 \
  --action_offset=1 \
  --fps=30 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only \
  --enable_pending_observation=true \
  --pending_observation_timeout_s=30 \
  '--rename_map={}'
```

client/server 握手会双向拒绝 protocol、FPS、policy type 或 chunk size 不一致：server 校验
client 请求，client 再校验 server 返回的 `PolicySetupAck`。task 在第一帧 observation 到达
server 时与 checkpoint 的固定 instruction 做 exact match。

可通过下列命令检查这一隔离边界：

```bash
ros2 node info /lerobot_franka_interface
ros2 topic info --verbose /lerobot/franka/action_chunk
ros2 topic echo --once /lerobot/franka/action_chunk
```

预期只出现 5 个 observation subscriptions 和 1 个 `/lerobot/franka/action_chunk` publisher。
不应出现 controller command publisher、action client 或 hardware service call。

需要把 action chunk、Gateway ACK/status 和过滤后的 Controller applied feedback 按本次运行
时间分目录、分文件保存时，使用
[`LEROBOT_ROS_LOGGING.md`](./LEROBOT_ROS_LOGGING.md) 中的 recorder。它同时区分
`--actions_per_chunk` 配置值与 ROS 实际 waypoint 数，且不会订阅 `/franka/safe_joint_command`。

## 10. 当前未实现，必须保留到后续阶段的事项

### 10.1 robot status、TF 和执行 ack

当前 runtime 没有 robot mode/error subscription、bridge status topic、TF lookup、`eef_frame`
配置或 action ack topic。后续需增加，但不得伪装成已实现能力。

ROSIDL package 已预先定义 `CartesianActionChunkAck` 和 `SafetyGatewayStatus`，便于未来 gateway
对接；当前 LeRobot runtime 不发布、订阅或消费这两种消息，它们的存在不代表 execution ACK
已经实现。

未来 ack 至少需要关联：

```text
session_id
plan_id
status: RECEIVED | REJECTED | ACCEPTED | EXECUTING | COMPLETED | HOLD | FAULT
reason/detail
last_accepted_index
last_applied_index
last_applied_timestep
gateway monotonic timestamp
robot state timestamp
```

RobotClient 当前通过本地 queue pop 更新 `latest_action`，这不能证明机器人已执行。
真机 chunk mode 必须使用 safety gateway/controller ack 推进 applied timestep。

### 10.2 本地 safety gateway（shadow 骨架已实现，执行仍锁定）

`/lerobot/franka/action_chunk` 的唯一合法后继消费者应是机器人主机上的独立
safety gateway。`haply_ros/src/franka_cartesian_safety_gateway` 现已提供默认
`enabled=false + shadow=true` 的 fail-closed 验证边界、ack/status、session/plan
排序、transport TTL、waypoint schedule、workspace/Cartesian step 检查和 robot-state
freshness 检查。由于可信的 sequential IK、奇异性和碰撞 preflight 尚未接入，
arm service 会拒绝启用，且当前不会发布 `SafeJointCommand`。

进入执行模式前仍必须完成：

- schema/session/plan ID/TTL/过期 waypoint 检查；
- finite、单位 quaternion、workspace、单步平移/旋转限制；
- 速度、加速度、jerk、joint/torque limits；
- 使用 qpos sideband 作为 seed 的 IK；
- 奇异位形、自碰、环境碰撞检查；
- robot mode/error、controller state、observation freshness；
- `DISABLED -> ARMED -> RUNNING -> HOLD -> FAULT` 本地状态机；
- E-stop、deadman、断线 HOLD、controlled stop 和 stale plan purge；
- plan replacement、旧 session 丢弃和重复 plan 幂等。

KML WebSocket heartbeat、gRPC 连接状态和 DDS publisher 存活都不能作为真机 watchdog。

### 10.3 IK、1 kHz controller 和 gripper driver

Python/LeRobot/KML 链路只提供 15 Hz 低频 absolute Cartesian waypoints。后续本地
C++/ros2_control 路径需要：

- 用 qpos 和当前 robot state 做 IK/singularity/joint-limit 检查；
- 对 accepted pose plan 做 SE(3) 连续插值；
- 在本地 1 kHz 控制周期中更新 target；
- 拥有独立 watchdog 并在计划过期/网络断开时 HOLD/stop；
- 分开记录 policy desired、gateway accepted、controller applied 和 measured trajectory；
- 对 Robotiq gripper 单独做 mapping、滞回、限频、force/speed limit。

### 10.4 进入真机的 review gate

必须按顺序完成：

1. 在目标主机确认实际 topic type/QoS/frame/joint/gripper convention；
2. live ROS observation 与离线 dataset sample 做 golden comparison；
3. 完成 fake publisher/subscriber 的 ROS graph smoke，仍不连 controller；
4. 完成 safety gateway、ack、watchdog、IK 和本地 controller；
5. shadow inference -> fake controller -> 单 waypoint -> 低速短 chunk；
6. 现场确认 E-stop、deadman、低 stiffness、空旷 workspace 和人工 enable；
7. 全部 review 通过后才考虑连续异步 rollout。
