# Franka、ROS 2 与 LeRobot Client 通信详解

> 状态基线：2026-07-20（Asia/Shanghai）
>
> 适用路径：`/home/pnp/Projects/lerobot` 与 `/home/pnp/franka`
>
> 目标：说明 ROS 2 observation、LeRobot 异步 client/server、KML WebSocket tunnel、
> Franka safety gateway 和本地 controller 之间的职责、协议、状态与安全边界。

## 0. 先看结论

当前系统包含两个名称中都有 `gateway`、但职责完全不同的组件：

1. **KML gateway** 是平台托管的网络入口，把外部 `wss://.../ws` 路由到 KML
   开发机上的 WebSocket tunnel server。它只转发字节，不理解机器人动作。
2. **Franka Cartesian safety gateway** 是机器人主机上的 ROS 2/C++ 安全闸门，位于
   LeRobot policy 输出与 1 kHz Franka controller 之间。它会验证、做 MoveIt preflight、
   转换为关节计划，并在本地条件持续满足时发布短时效关节命令。

`enabled`、`shadow` 和 `armed` 不是同一个概念：

- `enabled`：启动时的总执行开关；
- `shadow`：启动时的“只做基础验证、不执行”开关；
- `armed`：运行时状态，表示本地安全闸门已被显式打开。

最关键的关系是：

```text
shadow=false  !=  armed=true
enabled=true  !=  armed=true
ARMED         !=  当前一定有计划在执行
ROS publish 成功 != gateway 接受 != controller 应用 != Franka 已运动
```

截至本文状态基线，实际运行的是两条互不相连的验证链：

```text
fixture -> dry-run RobotClient -> KML PI0 -> JSONL

validation_action_chunk（0 publisher）
  -> safety gateway（HOLD、未 armed）
  -> inactive arm controller
```

因此当前**没有 LeRobot 到 Franka 真机的闭环执行链**。

---

## 1. 组件和信任边界

### 1.1 远端推理通信组件

| 组件 | 位置/地址 | 职责 |
|---|---|---|
| LeRobot RobotClient | 机器人主机 | 采集 observation、发送推理请求、接收 action chunk |
| tunnel client | `127.0.0.1:8080` | 把本地 gRPC/TCP 字节封装进 WebSocket |
| KML gateway | `wss://<KML 域名>/ws`，通常走 443 | KML 平台托管路由、TLS/AccessProxy/SSO 入口 |
| tunnel server | KML `0.0.0.0:16782` | 把 WebSocket 字节还原到本地 TCP |
| FrankaPI0PolicyServer | KML `127.0.0.1:15173` | PI0 预处理、推理、后处理、Franka action 解码 |

RobotClient 只连接本机 `127.0.0.1:8080`，不会直接连接 `16782`。
`16782` 是 KML gateway 映射到开发机的目标端口，不是公开 raw gRPC 端口。

网络拓扑的现有启动说明见
[`ASYNC_CLIENT_SERVER_RUNBOOK.md`](./ASYNC_CLIENT_SERVER_RUNBOOK.md)。

### 1.2 机器人主机 ROS 2 组件

| 组件 | 主要职责 |
|---|---|
| `lerobot_robot_franka_ros` | LeRobot 的 out-of-tree Franka Robot 插件 |
| `FrankaRos2RobotClient` | 在 stock RobotClient 上增加完整 ROS action chunk 发布 hook |
| `lerobot_franka_interfaces` | 定义 Cartesian chunk、gateway ACK 和 gateway status |
| `franka_cartesian_safety_gateway` | 本机 plan 验证、MoveIt preflight、状态机和 200 Hz 安全命令 |
| `franka_safety_interfaces` | 定义 gateway 到 controller 的 `SafeJointCommand` |
| `joint_impedance_ik_controller` | 1 kHz 二次校验、限速/限扭矩和 Franka effort control |

Safety gateway 没有固定 TCP 端口；它通过 ROS 2/DDS topic 和 service 通信。

### 1.3 完整设计拓扑

```text
机器人主机

  camera1 / camera2 -------------------------------+
  Franka EEF PoseStamped --------------------------|
  Franka 7D JointState ----------------------------|--> RosObservationCache
  Robotiq JointState ------------------------------+          |
                                                              v
                                                    FrankaRos2RobotClient
                                                              |
                                                    local gRPC/TCP :8080
                                                              |
                                                       WebSocket tunnel
                                                              |
KML                                                           v

  KML managed WSS gateway -> :16782 tunnel server -> :15173 PolicyServer
                                                              |
                                                       PI0 inference
                                                              |
                                                       absolute8 chunk
                                                              |
机器人主机                                                    v

  FrankaRos2RobotClient
         |
         v
  /lerobot/franka/action_chunk
         |
         v
  Franka Cartesian safety gateway
         |  schema / TTL / schedule / workspace / ordering
         |  MoveIt IK / limits / singularity / collision
         |  local state machine / watchdog / explicit arm
         v
  /franka/safe_joint_command（200 Hz）
         |
         v
  joint_impedance_ik_controller（1 kHz）
         |
         v
        FR3
```

---

## 2. ROS 2 observation 如何进入 LeRobot Client

### 2.1 默认输入 topic

默认配置位于
[`config_franka_ros.py`](./ros_lerobot/src/lerobot_robot_franka_ros/config_franka_ros.py)。

| 数据 | 默认 topic | ROS type | 用途 |
|---|---|---|---|
| camera1 | `/camera1/camera1/color/image_raw` | `sensor_msgs/msg/Image` | 同步 anchor、PI0 base image |
| camera2 | `/camera2/camera2/color/image_raw` | `sensor_msgs/msg/Image` | PI0 wrist image |
| EEF pose | `/franka_robot_state_broadcaster/current_pose` | `geometry_msgs/msg/PoseStamped` | state10 的 xyz 和 rotation6d |
| arm qpos | `/franka/joint_states` | `sensor_msgs/msg/JointState` | 7D sideband、诊断和本地安全参考 |
| gripper | `/gripper/joint_states` | `sensor_msgs/msg/JointState` | state10 的 `closed_0_1` |

ROS runtime 还创建一个 action publisher：

```text
/lerobot/franka/action_chunk
  type: lerobot_franka_interfaces/msg/CartesianActionChunk
```

当前 QoS：

| 用途 | Reliability | History | Depth | Durability |
|---|---|---|---:|---|
| 五类 observation subscription | Best effort | Keep last | 5 | Volatile |
| action chunk publisher | Reliable | Keep last | 1 | Volatile |

创建位置见
[`ros2_runtime.py`](./ros_lerobot/src/lerobot_robot_franka_ros/ros2_runtime.py)。

### 2.2 Callback 解码和拒绝行为

- 图像支持 `rgb8`、`bgr8` 和行 padding，最终统一为 HWC RGB；
- 图像必须是 `uint8 [480,640,3]`；
- EEF pose 保存 source timestamp、接收 monotonic timestamp、`frame_id`、xyz 和 xyzw quaternion；
- arm joint state 按配置中的七个 joint name 重新排序；
- gripper joint position 按 open/closed 标定线性映射并 clip 到 `[0,1]`；
- 畸形样本会被拒绝并限频记录错误，不会杀死 ROS executor。

ROS import 是 lazy 的：只有 `dry_run=false` 且显式启动 `Ros2Runtime` 时才加载
`rclpy` 和 ROS message。插件发现和 dry-run 不要求 source ROS 环境。

### 2.3 因果同步

`RosObservationCache` 使用最新 camera1 source timestamp 作为 anchor。对 camera2、EEF、qpos
和 gripper，它只选择：

1. source timestamp 不晚于 camera1 anchor；
2. 满足对应最大 skew；
3. 本地 arrival age 未超过上限；
4. 满足条件的最新样本。

所有 required source 必须同时满足条件，才会生成一个 snapshot。任一来源缺失、过期或
skew 超限，`get_observation()` 都 fail closed，不返回部分 observation。

默认约束：

| 参数 | 默认值 |
|---|---:|
| cache buffer size | 32 |
| observation 最大 arrival age | 0.25 s |
| camera2 最大 skew | 0.05 s |
| EEF 最大 skew | 0.05 s |
| qpos 最大 skew | 0.10 s |
| gripper 最大 skew | 0.05 s |

### 2.4 Policy-facing state10

EEF quaternion 先验证、归一化，再转换成 rotation matrix 的前两列。最终 state 顺序固定为：

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

Robot observation 最终包含：

```text
10 个具名 float state feature
camera1: uint8 [480,640,3], RGB, HWC
camera2: uint8 [480,640,3], RGB, HWC
```

七关节 qpos 只作为 sideband 保存，不进入当前 PI0 state10。若直接把 qpos 加进 policy
features，state 会变成 17D，并被 Franka PolicyServer 的 frozen contract 拒绝。

---

## 3. LeRobot Client 的线程、握手和 observation 发送

### 3.1 为什么 ROS 模式必须使用专用 client

Dry-run 可以使用 stock 入口：

```text
python -m lerobot.async_inference.robot_client
```

ROS action chunk 模式必须使用：

```text
python -m lerobot_robot_franka_ros.ros2_client
```

专用 `FrankaRos2RobotClient` 继承 stock `RobotClient`，但强制：

```text
robot.type = franka_ros
robot.dry_run = false
robot.ros2_interface_only = true
aggregate_fn_name = latest_only
```

它的唯一关键扩展是在 action queue 聚合边界发布一次完整 ROS chunk。不能用 stock client
代替，否则 action 只会逐步进入 Robot 的 `send_action()`，不会形成 gateway 所需的完整计划。

实现见
[`ros2_client.py`](./ros_lerobot/src/lerobot_robot_franka_ros/ros2_client.py)。

### 3.2 Client 初始化和握手

`RobotClient.__init__()` 会：

1. 根据 `FrankaRosConfig` 创建 `FrankaRos`；
2. 启动 private ROS runtime 并连接 Robot backend；
3. 将 Robot features 转为 LeRobot feature schema；
4. 建立 `grpc.insecure_channel(server_address)`；
5. 初始化 action queue、pending-observation gate、client session UUID 和线程 barrier。

`start()` 依次调用：

```text
Ready()
SendPolicyInstructions(RemotePolicyConfig)
```

- `Ready()` 会重置 PolicyServer 当前 session 状态；
- policy instructions 携带 policy type、robot feature contract、action chunk size、device 和
  camera rename map；
- Franka server 会对 feature 名称、顺序、shape 和 rename map 做精确检查。

### 3.3 两个 client 线程

| 线程 | 职责 |
|---|---|
| main/control thread | 以 15 Hz pop 本地 action，并判断是否采集/发送新 observation |
| action receiver thread | 循环 `GetActions()`、校验、queue commit、ROS chunk publish、gRPC ACK |

两线程通过 barrier 同时开始。

### 3.4 TimedObservation

每个 observation 包含：

```text
task
timestamp                 # time.time()，供跨机器时间关联
timestep                  # max(latest_action + action_offset, 0)
must_go                   # action queue 空时强制推理
client_send_timestamp
request_id                # <client-session-uuid>:<monotonic-sequence>
```

`request_id` 在同一次逻辑 observation 的 transport retry 中保持不变。

`action_offset` 是 RobotClient 顶层 CLI 参数，默认值为 `0`，保持旧版“以当前
`latest_action` 编号”的行为。显式设置 `--action_offset=1` 时，新 observation 从本地已消费
cursor 的下一 timestep 开始编号。启动阶段 `latest_action=-1`，经 `max(..., 0)` 后 timestep
仍为 `0`。

Observation 使用 pickle 序列化，再通过 `SendObservations(stream Observation)` 分片发送。
当前传输是受信任内部协议，不应把 raw gRPC 或 tunnel server 暴露给不受信任客户端，因为
服务端会反序列化 pickle。

### 3.5 Pending observation gate

启用 pending gate 后，同一时刻通常只有一个等待 action 的 observation：

- `SendObservations` 成功，只证明服务端完整收到请求；
- 发送 RPC 失败时，client 保留原 observation 和原 `request_id`，稍后重试；
- observation 成功发送但没有 action 时，等待 `pending_observation_timeout_s` 后采集一帧新的
  observation；
- action 成功 commit 后清除 pending gate。

---

## 4. TCP-over-WebSocket 和 KML gateway

### 4.1 字节流路径

```text
RobotClient gRPC/HTTP2 bytes
  -> local TCP 127.0.0.1:8080
  -> tunnel client binary WebSocket frames
  -> KML WSS gateway / AccessProxy
  -> KML :16782 tunnel server
  -> local TCP 127.0.0.1:15173
  -> PolicyServer
```

[`tools/ws_tcp_tunnel.py`](../tools/ws_tcp_tunnel.py) 不解析 protobuf 或 action：

- client 从本地 TCP 读取最多 64 KiB，作为 WebSocket binary frame 发送；
- server 把 WebSocket binary/text payload 原样写给目标 TCP；
- 任一方向结束时，另一方向 task 被取消并关闭连接；
- WebSocket heartbeat 默认 20 秒。

### 4.2 KML gateway 是什么

KML gateway 是平台托管路由，不是仓库内需要启动的进程。它通常负责：

- 对外 `wss://.../ws`；
- TLS；
- AccessProxy/SSO；
- 将请求映射到开发机 `0.0.0.0:16782`。

仓库内需要启动的是两端的 `ws_tcp_tunnel.py`，不是 KML gateway 本身。

### 4.3 认证边界

当前有人值守流程从已完成 SSO/MFA 的 Firefox profile 读取仅适用于目标 host/path 的 Cookie，
再通过进程环境传给 tunnel。该方式适合验证，不适合作为长期无人值守机器身份。

长期部署应使用平台正式支持的 machine token、mTLS 或等价 machine-to-machine 身份。
WebSocket heartbeat 和 gRPC connected 状态都不能充当机器人安全 watchdog。

---

## 5. PolicyServer 和 Franka action 解码

### 5.1 Feature contract

Franka server 固定要求：

```text
observation.state: float32 [10]，名称和顺序精确匹配
observation.images.camera1: image [480,640,3]
observation.images.camera2: image [480,640,3]
```

Camera rename 固定为：

```text
observation.images.camera1
  -> observation.images.base_0_rgb

observation.images.camera2
  -> observation.images.left_wrist_0_rgb
```

服务端还严格检查 PI0 checkpoint manifest、文件 hash、几何 contract、stats、15 Hz 和
50-step action chunk。

实现见
[`async_server.py`](./src/franka_eef_pipeline/async_server.py)。

### 5.2 Observation queue

PolicyServer observation queue 大小为 1：

- 新 observation 可以替换尚未处理的旧 observation；
- 已排队、正在推理、已生成或已交付的 timestep 会被过滤；
- 相对上一推理 observation 过于相似且不是 `must_go` 的 observation 可以被过滤；
- `GetActions` 被串行化，任一时刻最多有一个待 ACK action chunk。

### 5.3 PI0 relative7 到 absolute8

PI0 postprocessor 输出每个 waypoint 的 relative7：

```text
[delta_x_base, delta_y_base, delta_z_base,
 body_rotvec_x, body_rotvec_y, body_rotvec_z,
 gripper_target_0_1]
```

Franka adapter 使用触发本次推理的原始 state10 作为统一 anchor：

```text
p_target = p_anchor + delta_p_base
R_target = R_anchor @ Exp(body_rotvec)
g_target = clip(g_prediction, 0, 1)
```

整个 chunk 的 waypoint 都使用同一个 observation anchor，不递推累加。网络输出为：

```text
[target.x, target.y, target.z,
 target.qx, target.qy, target.qz, target.qw,
 target.gripper.closed_0_1]
```

这些 pose 是 absolute Franka `O_T_EE` TCP targets。Safety gateway 不应再次把它们解释成
relative action。

### 5.4 TimedAction

每个 action 的逻辑时间是：

```text
timestamp = observation.timestamp + i * environment_dt
timestep  = observation.timestep + i
```

15 Hz 时 `environment_dt` 约为 `66,666,667 ns`。服务端还给同一个 chunk 的所有 action
写入相同的 `server_send_timestamp`。

---

## 6. Action 返回、queue commit 和 ROS chunk

### 6.1 gRPC Actions payload

```text
data                # pickled list[TimedAction]
request_id
chunk_id
source_timestep
```

Receiver thread 会检查：

- `data` 是非空 `list[TimedAction]`；
- ACK-capable response 同时具有 request/chunk ID；
- `source_timestep` 等于首个 TimedAction timestep；
- duplicate `chunk_id` 不会再次本地 commit；
- action device 按 client 配置转换。

### 6.2 Fresh suffix

`FrankaRos2RobotClient` 丢弃：

```text
timestep <= latest_action
```

的 stale prefix，然后：

1. 使用 stock `latest_only` 更新本地 queue；
2. 保留原始 source observation timestep/timestamp；
3. 把 fresh suffix 作为一个完整 ROS chunk 发布一次；
4. ROS 转换或 publish 失败时设置 client shutdown，不降级为逐 waypoint publish。

Absolute quaternion 不能使用逐元素 `weighted_average`，因此必须使用 `latest_only`。

令创建 observation 时的 `latest_action=L`。在 `--action_offset=1` 下，observation timestep
为 `max(L+1, 0)`，PolicyServer 按第 5.4 节从该值连续生成 action。对于正常的 `L>=0`：

- action 返回前 cursor 仍为 `L` 时，50-step chunk 编号为 `L+1 ... L+50`，50 条全部保留；
- 推理/传输期间 cursor 又推进 `m` 步时，`L+1 ... L+m` 已真正过期，fresh filter 仍会裁剪
  这 `m` 条；
- 因此 offset 解决的是固定首项与 `latest_action` 重叠造成的 49/50，不是“任何时延下固定
  发布 50 条”的承诺。

Pending transport retry 会复用原 observation、偏移后的 timestep 和 `request_id`，不会按新的
cursor 再算一次。Action delivery ACK 的去重/重发状态机、ROS `CartesianActionChunk` wire
schema，以及 gateway 的 `enabled`/`shadow`/`armed` 语义均未改变；gateway 也不直接接收
`action_offset` 参数。

### 6.3 `send_action()` 为什么不控制 ROS

Stock RobotClient main loop 仍然会逐点从本地 queue pop action，并调用 Robot
`send_action()`。但 ROS backend 的 `send_action()` 只记录 bookkeeping，不发布第二条 ROS command
path。

因此：

```text
latest_action = 本地 queue pop 进度
latest_action != gateway accepted timestep
latest_action != controller applied timestep
latest_action != Franka measured timestep
```

---

## 7. `CartesianActionChunk` 协议

消息定义见
[`CartesianActionChunk.msg`](./ros2_ws/src/lerobot_franka_interfaces/msg/CartesianActionChunk.msg)。

| 字段 | 语义 |
|---|---|
| `header.stamp` | client 开始发布该计划时的本地 ROS time，也是 waypoint 0 的 schedule anchor |
| `header.frame_id` | 默认 `base` |
| `schema_version` | 当前只允许 1 |
| `session_id` | ROS backend 每次 connect 时生成 |
| `plan_id` | session 内从 0 递增 |
| `source_timestep` | 触发推理的 observation timestep |
| `client_observation_stamp` | observation wall-clock provenance，不是 execution deadline |
| `server_send_stamp` | PolicyServer 发出 chunk 的 wall-clock provenance |
| `valid_until` | 新到达/replay plan 可以被 gateway 接受的 transport TTL |
| `period` | waypoint 周期，15 Hz 时约 66.67 ms |
| `timesteps[]` | fresh suffix 的连续 LeRobot timestep |
| `poses[]` | absolute xyz + quaternion xyzw |
| `gripper[]` | `[0,1]`，0=open，1=configured closed |

### 7.1 TTL 和执行 horizon 必须分开理解

默认 `action_chunk_validity_s=0.5` 只限制一个新 plan 是否还可以被 gateway 接纳。
它不会把已经接纳的 50-step、约 3.33 秒计划在 0.5 秒处截断。

已经接纳的 waypoint 仍由：

```text
due(i) = header.stamp + i * period
```

以及 plan horizon、state freshness 和本地 watchdog 管理。TTL 永远不授权迟到 waypoint。

### 7.2 TCP pose 与 flange pose

`poses[]` 表示 Franka 实测/配置的 `O_T_EE` TCP pose，不是 `fr3_link8` flange pose。
当前 MoveIt provider 从新鲜实测状态推导固定 `F_T_EE`，把目标 TCP pose 转为 flange target
再求 IK，并用 FK round-trip 检查解出的 TCP 是否匹配请求。

---

## 8. Franka Cartesian safety gateway

源码位于：

```text
/home/pnp/franka/haply_ros/src/franka_cartesian_safety_gateway
```

设计说明见
[`Franka gateway README`](../../../franka/haply_ros/src/franka_cartesian_safety_gateway/README.md)。

### 8.1 为什么需要 safety gateway

远端 policy、gRPC、WebSocket、Python 和 KML 都不是硬实时控制器，也不应被当作安全可信输入。
Safety gateway 将远端 Cartesian plan 视为“不可信候选计划”，在机器人主机上重新判断：

- 消息是否完整、有限、未过期、顺序正确；
- 计划是否在 workspace 和步长限制内；
- 当前 robot state 是否新鲜；
- MoveIt IK、关节限制、奇异性和碰撞是否通过；
- 本地 operator 是否显式 arm；
- controller、planning scene、action publisher 和 timers 是否持续存活。

只有完整通过的计划才能产生 `SafeJointCommand`。

### 8.2 第一层：纯 plan validation

所有模式都会创建 action chunk subscription。纯验证检查：

- `schema_version`；
- Cartesian `frame_id`；
- 非空 `session_id`；
- waypoint 数量和三组数组长度；默认最多 50，匹配 PI0 的 50-step chunk；client 丢弃
  stale prefix 后发布的 fresh suffix 可以少于 50；
- transport TTL；
- 正 period 和 schedule overflow；
- 连续 timesteps，且首 timestep 不早于 source timestep；
- finite pose；
- 单位 quaternion；
- gripper `[0,1]`；
- workspace box；
- 相邻 Cartesian translation/rotation step；
- 首 waypoint 相对当前 pose 的 step；
- active horizon 内 session 不切换、`plan_id` 严格增加。

### 8.3 第二层：MoveIt preflight

MoveIt preflight 只在 `enabled=true`、`shadow=false`、gateway 已 ARMED 且基础验证通过时运行。
它会：

1. 用新鲜七关节实测状态作为首个 IK seed；
2. 用上一 waypoint 的解作为下一点 seed；
3. 从实测 `O_T_EE` 和 MoveIt flange pose 推导 `F_T_EE`；
4. 将每个 TCP target 转换成 flange target；
5. 检查 IK、FK round-trip、URDF joint bound inset margin；
6. 检查 Jacobian 最小奇异值和 condition number；
7. 检查当前状态、每个 waypoint 和关节插值样本的自碰/环境碰撞；
8. 检查关节位置步长、速度和加速度；
9. preflight 结束后重新检查 plan TTL/schedule、状态新鲜度、scene 和 state drift；
10. 全部通过后原子替换 executable plan。

Planning scene 被认为 ready 至少要求：

- MoveIt model/SRDF/IK provider 已加载；
- 收到外部 geometry/full-scene update；
- `/monitored_planning_scene` 仍有 publisher；
- world 非空。

本机自己构造的空 world 不被视为可信环境安全证明。

---

## 9. `enabled`、`shadow`、`armed` 分别是什么意思

### 9.1 三个概念

#### `enabled`

`enabled` 是节点启动时读取的总执行开关。

- `false`：状态为 `DISABLED`，不能武装、不能发布安全关节命令；
- `true`：允许进入 `SHADOW` 或可武装的 `HOLD` 模式。

`DISABLED` 不等于“不接收消息”。当前实现仍会订阅 chunk、做纯验证并发布验证 ACK。

#### `shadow`

`shadow` 是节点启动时读取的“只验证、不执行”开关，仅在 `enabled=true` 时有意义。

- `enabled=true, shadow=true`：进入 `SHADOW`；
- `enabled=true, shadow=false`：进入 `HOLD`，等待显式 arm。

当前 `SHADOW` 只运行第 8.2 节的纯 plan validation，不运行完整 MoveIt IK/碰撞 preflight。
因此它是“transport/Cartesian validation shadow”，不是“完整模拟一次将要执行的 preflight”。

#### `armed`

`armed` 是运行时状态，不是配置参数。只有内部状态为 `ARMED` 时才为 true。

`armed` 不是接收 action chunk 的前提。Gateway 在所有模式都会创建 chunk subscription：
SHADOW 下合法 chunk 返回 `ACCEPTED_SHADOW`；execution HOLD 下合法 chunk 通常返回
`REJECTED_NOT_ARMED`；只有 ARMED 才会继续 MoveIt preflight 并可能输出关节命令。

武装的准确含义是：

> 在本地 gates 当前全部满足的前提下，允许一个之后到达并完整通过 preflight 的计划成为
> executable plan，并允许 timer 发布 `SafeJointCommand`。

它不表示：

- Franka 已上电或关节已解锁；
- controller 已 active；
- 当前已有可执行 plan；
- 当前正在发布命令；
- 任意 waypoint 已被实际应用。

### 9.2 启动配置真值表

| `enabled` | `shadow` | 初始内部状态 | 能否调用 arm 成功 | 能否输出命令 |
|---:|---:|---|---|---|
| false | false/true | `DISABLED` | 否 | 否 |
| true | true | `SHADOW` | 否 | 否 |
| true | false | `HOLD` | gates 满足时可以 | arm 且有 executable plan 后可以 |

出厂配置是：

```yaml
enabled: false
shadow: true
```

实际初始状态是 `DISABLED`，不是 `SHADOW`，因为 `enabled=false` 优先。

当前源码把 `enabled` 和 `shadow` 只在构造时复制到成员变量，没有动态更新路径。即使
`ros2 param set` 改变参数存储值，也不会可靠切换当前运行模式；应通过明确的启动参数和重启
切换模式。

### 9.3 五个内部状态

| 状态 | 典型进入条件 | Chunk 行为 | `SafeJointCommand` | 离开方式 |
|---|---|---|---|---|
| `DISABLED` | `enabled=false` | 仍做纯验证；通过时 ACK 为 `ACCEPTED_SHADOW` | 不发布 | 以 `enabled=true` 重启 |
| `SHADOW` | `enabled=true, shadow=true` | 仍做纯验证；不跑 MoveIt preflight | 不发布 | 以 `shadow=false` 重启，先进入 HOLD |
| `HOLD` | execution mode 初始态、显式 disarm、arm gate 失败、watchdog 触发 | 非法 chunk 返回具体错误；基础合法 chunk 通常返回 `REJECTED_NOT_ARMED` | 不发布 | 修复 gates 后再次显式 arm |
| `ARMED` | 显式 arm 且所有 arm gates 通过 | 基础验证 + MoveIt preflight；成功则替换 executable plan | 有 plan 且运行 gates 持续成立时发布 | disarm、watchdog 或内部 fault |
| `FAULT` | provider 异常、horizon overflow、sequence 耗尽等内部异常 | 不执行 | 不发布 | 修复根因后 disarm/re-arm；部分原因需要重启 |

当前没有独立 `RUNNING` 状态。`ARMED + has_active_plan=true` 才最接近“已持有执行计划”；
是否已有 waypoint 被 controller 应用仍无法由该状态证明。

在 `ARMED` 下，一个新 candidate chunk 被基础验证或 preflight 拒绝，通常只会返回拒绝 ACK，
不会自动 disarm，也不会自动撤销此前已经接纳的 active plan。拒绝新计划不等于 emergency stop；
需要立即停止输出时，应走本地 disarm/急停路径。

### 9.4 Arm gates

`set_armed(true)` 至少要求：

1. `enabled=true`；
2. `shadow=false`；
3. MoveIt preflight provider ready；
4. 完整 joint state 和 EEF pose 新鲜，默认不超过 0.15 s；
5. monitored world planning scene ready；
6. 默认存在 `SafeJointCommand` subscriber。

注意：subscriber count 只证明 ROS endpoint 存在，不证明 lifecycle controller 已经 active。
`require_chunk_publisher` 也不是 arm-time gate，而是在已有 execution plan 后由 command timer
持续检查的 watchdog。

Arm service 是：

```text
/franka_cartesian_safety_gateway/set_armed
type: std_srvs/srv/SetBool
```

`data=false` 会 disarm、清空当前 execution/plan，并回到由启动配置决定的非 armed 状态。
`data=true` 只有所有 gates 满足才会成功。

### 9.5 HOLD 和 FAULT 在线上 status 中的编码限制

内部状态机有五态：

```text
DISABLED=0, SHADOW=1, ARMED=2, HOLD=3, FAULT=4
```

但当前 `SafetyGatewayStatus.msg` 只有：

```text
DISABLED=0, SHADOW=1, ARMED=2, FAULT=3
```

因此内部 `HOLD` 和 `FAULT` 在线上都编码为：

```text
state = 3
armed = false
shadow = false
```

通常 `detail` 会带 `HOLD: ...` 或 `FAULT: ...`，但 chunk validation 结果也会覆盖同一 detail
字段，所以不能把自由文本作为可靠控制协议。消费端应把 `state=3` 一律视为“未武装、无命令
输出”。如果必须可靠区分 HOLD 和 FAULT，应升级 wire schema。

### 9.6 状态迁移简图

```text
                 enabled=false
                      |
                      v
                  DISABLED

enabled=true, shadow=true
          |
          v
       SHADOW

enabled=true, shadow=false
          |
          v
        HOLD <---------------- disarm / watchdog
          |                            ^
          | set_armed(true),            |
          | all gates pass              |
          v                            |
        ARMED --------------------------+
          |
          | internal unrecoverable/invalid runtime condition
          v
        FAULT
```

`enabled`/`shadow` 当前应通过重启改变；`armed` 通过 service 动态改变。

---

## 10. Gateway 如何生成 200 Hz 安全关节命令

一个 plan 通过 MoveIt preflight 后，gateway 保存：

```text
原始 CartesianPlan
接收/preflight 时的 initial_joints[7]
每个 Cartesian waypoint 对应的 IK joint_waypoint[7]
```

每 5 ms 的 command timer：

1. 确认仍为 ARMED；
2. 检查 timer 自身 gap；
3. 检查 robot state freshness；
4. 检查 MoveIt provider 和 planning scene；
5. 检查 controller subscriber；
6. 检查 action publisher；
7. 检查 plan horizon；
8. 在相邻 15 Hz IK joint waypoint 之间做关节空间线性插值；
9. 生成递增 `sequence`；
10. 发布 deadline 为 `now + 30 ms` 的 `SafeJointCommand`。

消息定义见
[`SafeJointCommand.msg`](../../../franka/haply_ros/src/franka_safety_interfaces/msg/SafeJointCommand.msg)。

关键字段包括：

```text
deadline
session_id
plan_id
waypoint_index
source_timestep
sequence
joint_names[7]
positions[7]
safety_state=ARMED
```

以下情况会 HOLD 并停止输出：

- command timer watchdog expired；
- robot state watchdog expired；
- monitored planning scene 或其 publisher 丢失；
- controller subscriber 丢失；
- action publisher 丢失；
- accepted plan horizon 到期。

Gripper 数组当前只在 Cartesian chunk 中验证，不会由 gateway 下发给 Robotiq controller。

---

## 11. 1 kHz controller 的第二道门

Controller 当前配置的独占输入源是：

```yaml
command_source: safety_gateway
safety_command_topic: /franka/safe_joint_command
```

配置见
[`controllers.yaml`](../../../franka/haply_ros/src/franka_arm_controllers/config/controllers.yaml)。

Callback 再检查：

- 七个 joint name 和顺序精确匹配；
- positions 全部 finite；
- frame 匹配；
- message `safety_state` 为 ARMED；
- ROS deadline 尚未过期；
- session/plan/waypoint progression 合法；
- 全局 sequence 单调增加。

1 kHz update loop 再检查：

- ROS deadline；
- steady-clock receive age，默认 50 ms；
- 新 sequence；
- joint target slew：默认 0.25 rad/s 且每 cycle 不超过 0.001 rad；
- torque rate 和 torque step；
- controller 原有 soft joint/effort/force 保护。

坏、缺失或过期的 gateway command 只能使 controller HOLD measured joints，不会 fallback 到
Haply、position teleop 或 replay。这是启动时选择的独占 command source。

---

## 12. ACK 分层：每一级到底证明什么

### 12.1 LeRobot gRPC transport ACK

```text
PolicyServer
  READY_UNACKED
       |
       | Actions(request_id, chunk_id, source_timestep)
       v
RobotClient deserialize / validate / queue commit / ROS publish call
       |
       | AckActions(ids)
       v
PolicyServer ACKED
```

它保证：

- server 在返回前缓存完整 action response；
- ACK 前重复 `GetActions` 会返回完全相同的 chunk；
- action response 丢失时不重复推理；
- ACK 丢失时 client 用 `chunk_id` 去重，只重发 ACK；
- ROS chunk 转换或 publisher 调用失败时不发送 ACK。

它最多证明：

```text
PolicyServer -> RobotClient -> ROS publisher API 调用成功
```

它不证明 DDS subscriber 收到，更不证明 gateway 接受、controller 应用或 Franka 执行。

详细协议见
[`ACTION_DELIVERY_ACK_PROTOCOL.md`](./ACTION_DELIVERY_ACK_PROTOCOL.md)。

### 12.2 ROS `CartesianActionChunkAck`

Gateway 对每个 chunk 发布：

```text
/lerobot/franka/action_chunk_ack
```

结果包括：

- `ACCEPTED_SHADOW`；
- `ACCEPTED_FOR_EXECUTION`；
- schema/frame/shape/nonfinite/quaternion/gripper/order/expired/schedule/workspace/step；
- not armed；
- state stale；
- preflight unavailable。

该 ACK 只表示 gateway 对 plan 的验证/替换结果。即使是 `ACCEPTED_FOR_EXECUTION`，也不证明
某个 waypoint 已被 controller 应用。

### 12.3 `SafetyGatewayStatus`

Gateway 以 10 Hz 和事件驱动方式发布：

```text
/lerobot/franka/safety_gateway_status
```

其中包括 state、shadow、armed、state freshness、preflight、active plan、session/plan 和
next waypoint 等状态。

### 12.4 当前缺失的执行 ACK

当前 LeRobot ROS runtime 不订阅 `CartesianActionChunkAck` 或 `SafetyGatewayStatus`。因此：

- gateway reject 不会撤回已经完成的 gRPC transport ACK；
- RobotClient 不知道 plan 是否被 gateway 接受；
- local queue 仍可能继续 pop，推进 `latest_action`；
- PolicyServer 也不知道 controller 实际应用到哪个 timestep。

当前还缺少 controller-applied / robot-measured ACK，例如：

```text
session_id
plan_id
last_gateway_accepted_index
last_controller_applied_index/timestep
robot mode/error
watchdog state
ROS + monotonic timestamps
```

在这条反馈闭环完成前，不能把 client queue 进度当成真机执行进度。

---

## 13. 失败和恢复语义

| 故障 | 当前行为 |
|---|---|
| ROS image encoding/frame/joint name 非法 | Callback 拒绝该样本并限频日志 |
| 任一 observation 缺失、stale 或 skew 超限 | Cache 不生成 snapshot，该轮不发送 observation |
| `SendObservations` transport error | 保留原 observation/request ID，稍后重试 |
| Observation 成功发送但被 similarity filter | Pending timeout 后采集新 observation、新 request ID |
| Policy inference/serialization 失败 | Server 回滚 reservation，返回 Empty |
| Action response 在网络中丢失 | Server 保持 `READY_UNACKED`，重发原 chunk，不重复推理 |
| Action payload 非法 | Client 不 commit、不 ACK，server 保留 pending chunk |
| ROS chunk 转换/TTL/publish 失败 | ROS client shutdown，不做逐点 fallback，不发 transport ACK |
| gRPC ACK 丢失 | Server 重发；client 识别 duplicate，仅重发 ACK |
| Gateway 拒绝 chunk | 发布 ROS ACK；当前 LeRobot 不消费，transport ACK 不回滚 |
| Gateway robot state/scene/endpoint/watchdog 失效 | HOLD、清 plan、停止 SafeJointCommand |
| SafeJointCommand 非法或过期 | Controller HOLD measured joints，等待新的合法 sequence |
| Client/server 进程重启或重新 `Ready()` | 内存 delivery/session 状态清空，不能恢复旧 chunk |
| Plan horizon 到期 | Gateway HOLD，需要重新 arm/提交新计划，不能继续旧 plan |

---

## 14. 推荐的分阶段验证方式

### 阶段 A：纯网络 dry-run

```text
fixture observation
  -> stock RobotClient
  -> KML PI0
  -> JSONL sink
```

验收重点：网络、checkpoint、模型 shape、relative7/absolute8、transport ACK。该阶段不 source
ROS、不连接 controller、不驱动 Franka。

### 阶段 B：ROS observation/interface

```text
live ROS observation
  -> FrankaRos2RobotClient
  -> remote PI0
  -> /lerobot/franka/action_chunk
```

Gateway 保持 `DISABLED`，或者不要启动 executable gateway。验收重点：topic type/QoS、因果同步、
state10 golden comparison、chunk schema 和时延。

注意：`ros2_interface_only=true` 只表示 LeRobot plugin 自身没有 controller API。如果一个已
ARMED 的 downstream gateway 正在消费相同 topic，系统整体仍可能具有执行副作用。

### 阶段 C：Gateway SHADOW

建议配置语义：

```text
enabled=true
shadow=true
```

验收基础消息、TTL、ordering、workspace 和 Cartesian step，并确认没有
`/franka/safe_joint_command` 输出。

当前 SHADOW 不运行完整 MoveIt preflight。如果希望 shadow 真正回答“这份计划如果执行，是否会
通过 IK/碰撞检查”，需要先扩展实现，不能根据当前 `ACCEPTED_SHADOW` 推断。

### 阶段 D：Fake controller / 单 waypoint

在 fake hardware 或明确隔离的 controller 上验证：

- planning scene；
- arm/disarm；
- preflight accept/reject；
- 200 Hz sequence/deadline；
- timeout/HOLD；
- session/plan replacement；
- gateway/controller ACK。

### 阶段 E：真机低速执行

只有完成前述阶段，并补足本文第 17 节的关键缺口后，才进入现场 E-stop、deadman、低 stiffness、
空旷 workspace、人工 enable 和单 waypoint 验证。连续 50-step rollout 应最后考虑。

---

## 15. 启动和编排现状

当前没有单一脚本可以完整启动/停止：

```text
KML PolicyServer
KML tunnel server
local tunnel client
ROS observation sources
LeRobot ROS2 client
MoveIt planning scene
safety gateway
controller/hardware
```

现有 `/home/pnp/franka/start_data_collection.sh` 和 `manual_start_collect.sh` 仍主要面向
Haply teleop/data collection：

- 会启动 arm、gripper、Haply 和 cameras；
- 不启动 LeRobot ROS2 client；
- 不启动 safety gateway；
- 不完整管理 MoveIt/planning scene；
- `stop_all.sh` 也不完整停止 gateway、MoveIt、tunnel 或 RobotClient。

与此同时，当前 `controllers.yaml` 已改成独占 `command_source=safety_gateway`。因此用旧脚本
启动 controller、但没有 gateway command 时，controller 会 HOLD，不会继续使用 Haply。

部署时应把每个进程的 owner、启动顺序、ready probe、stop 顺序和日志位置显式编排，不能依赖
多个残留 shell 手工拼接。

---

## 16. 只读诊断命令

先 source 对应环境：

```bash
source /opt/ros/jazzy/setup.bash
source /home/pnp/Projects/lerobot/franka_project/ros2_ws/install/setup.bash
source /home/pnp/franka/franka_ros2_ws/install/setup.bash
source /home/pnp/franka/haply_ros/install/setup.bash
```

检查 LeRobot ROS client 是否存在：

```bash
ros2 node list | grep lerobot
ros2 topic info --verbose /lerobot/franka/action_chunk
```

检查 gateway 启动模式：

```bash
ros2 param get /franka_cartesian_safety_gateway enabled
ros2 param get /franka_cartesian_safety_gateway shadow
ros2 param get /franka_cartesian_safety_gateway chunk_topic
ros2 param get /franka_cartesian_safety_gateway max_waypoints
```

检查 gateway 状态和 ACK：

```bash
ros2 topic echo --once /lerobot/franka/safety_gateway_status
ros2 topic echo --once /lerobot/franka/action_chunk_ack
```

检查安全命令 endpoint 和 controller：

```bash
ros2 topic info --verbose /franka/safe_joint_command
ros2 control list_controllers
ros2 param get /joint_impedance_ik_controller command_source
```

检查本地 tunnel：

```bash
ss -lntp | grep ':8080'
ps -ef | grep -E 'ws_tcp_tunnel|robot_client|ros2_client|serve_franka_pi0'
```

安全 disarm 命令：

```bash
ros2 service call /franka_cartesian_safety_gateway/set_armed \
  std_srvs/srv/SetBool '{data: false}'
```

本文不提供可直接复制的真机 arm 命令。执行 `data=true` 前必须先确认 controller lifecycle、
robot mode/error、state freshness、planning scene、publisher/subscriber、E-stop/deadman 和现场人员。

---

## 17. 当前已知缺口和风险

### 17.1 尚未形成执行反馈闭环

Gateway ACK/status 没有回到 LeRobot，controller applied timestep 也没有反馈。当前 client queue
会在不知道 gateway 是否拒绝的情况下继续推进。

### 17.2 Gripper 没有执行路径

PI0 和 Cartesian chunk 中存在 gripper target，但 gateway 只校验，不下发 Robotiq 命令。

### 17.3 SHADOW 不等于完整 preflight dry-run

当前 SHADOW 不运行 sequential IK、Jacobian 和 collision preflight。`ACCEPTED_SHADOW` 不能证明
同一计划在 ARMED 下会被接受。

### 17.4 Robot readiness 不完整

Gateway 当前主要看 joint/TCP freshness、MoveIt scene 和 endpoint count，没有完整消费
Franka robot mode/error、controller lifecycle、E-stop 或独立 deadman 状态。存在 subscriber
也不代表 controller active。

### 17.5 “Machine-local” 没有由 DDS 身份强制保证

当前 safety boundary 是架构假设，不是来源认证边界。Fast DDS 使用子网发现时，同 ROS domain
中的其他节点理论上可以发布同名 action/safe-command topic、planning scene，或调用 arm service。

真机部署需要至少采用网络/ROS domain 隔离、指定 discovery peers、接口绑定或 SROS2/DDS
Security，确保只有授权节点可以：

- 发布 `/lerobot/franka/action_chunk`；
- 发布 `/franka/safe_joint_command`；
- 发布 monitored planning scene；
- 调用 `set_armed`。

### 17.6 Candidate race：disarm 与 publish 之间

当前 timer 在 mutex 内构造 `SafeJointCommand`，释放 mutex 后调用 ROS publisher。Arm service 位于
另一 callback group，MultiThreadedExecutor 允许 `set_armed(false)` 在“解锁但尚未 publish”窗口
插入。理论上可能多发布一条已经构造、最长仍有约 30 ms deadline 的 ARMED command。

这需要在真机前专门审计，例如在 publish 前再次检查 generation/armed token，或把 disarm 和
command publish 放进同一互斥执行序列。

### 17.7 Wire status 无法无损区分 HOLD/FAULT

当前 status schema 把两者都编码成 `state=3`。自由文本 detail 也可能被后续 chunk validation
覆盖，不应作为控制判断条件。

### 17.8 启动编排和配置漂移

现有采集脚本不启动 gateway，但 controller 已切到 gateway-only 输入；旧进程也可能被
`stop_all.sh` 遗留。需要统一 orchestration。

### 17.9 代码和版本状态

截至本文基线：

- LeRobot 分支为 `agent/franka-async-client-server`，相关代码实现已提交；加入本文档之前工作树为 clean；
- `/home/pnp/franka` 中 gateway/interfaces 仍是未跟踪 WIP；
- controller gateway 接入仍是未提交修改；
- 本机 build artifact 新于对应 gateway C++ source，但因为源码未提交，仍无法从 Git 独立复现；
- 部分旧文档仍称 KML SSO 被 302 阻塞、gateway 只有 shadow skeleton，已经落后于当前代码和现场。

---

## 18. 2026-07-20 当前运行态快照

快照时间：2026-07-20 00:17（Asia/Shanghai）。运行态会变化，本节不是静态配置保证。

### 18.1 网络 dry-run

```text
tunnel client: 127.0.0.1:8080 LISTEN
local RobotClient -> 8080: ESTABLISHED
tunnel -> KML :443: ESTABLISHED
RobotClient mode: dry_run=true
input: franka_observation_v1.npz
output: /tmp/franka_async_actions.jsonl
```

检查时 JSONL 已有 11014 行且仍在增长。因此当前：

```text
fixture -> RobotClient -> KML/WSS -> PI0 -> JSONL
```

已经实际工作。旧文档中的 SSO 302 阻塞结论不再代表当前现场。

### 18.2 ROS/gateway

当前没有 `lerobot_robot_franka_ros.ros2_client` 进程，也没有 `/lerobot_franka_interface` 节点。
两路 camera topic 有 publisher，但没有 LeRobot subscriber；默认
`/lerobot/franka/action_chunk` 不存在。

当前 gateway 是单 waypoint 验证实例：

```text
enabled=true
shadow=false
chunk_topic=/lerobot/franka/validation_action_chunk
max_waypoints=1
require_chunk_publisher=false
command_period_ms=5
```

Validation topic 为 0 publisher / 1 subscriber。Gateway status：

```text
state=3                 # wire 上 HOLD/FAULT 共用值
armed=false
shadow=false
state_fresh=false
preflight_available=true
has_active_plan=false
detail=HOLD: robot state watchdog expired
```

Arm controller lifecycle 均为 inactive。虽然 `/franka/safe_joint_command` graph 中存在一个
publisher 和一个 subscriber，这不代表存在有效 command 流或 controller 正在应用命令。

ROS graph 还报告了两个完全相同的 `/franka_cartesian_safety_gateway` node name，尽管 OS 层只看到
一个 `gateway_node` 进程。重复 node name 会让 parameter/service 诊断产生歧义，应在正式编排前
定位是同进程重复建 node、残留 DDS participant，还是 launch 命名冲突。

结论：网络 dry-run 和本地 gateway validation 当前彼此断开，没有真机执行。

### 18.3 2026-07-20 10:55 接收验证更新

Gateway 已从隔离单 waypoint topic 切回模型主 topic，并保持无执行副作用的 SHADOW：

```text
enabled=true
shadow=true
chunk_topic=/lerobot/franka/action_chunk
max_waypoints=50
```

恢复 Franka 新鲜 joint/TCP observation 后，实测收到：

```text
plan_id=5
waypoints=49
period_ns=66666667
ACK accepted=true
ACK result=ACCEPTED_SHADOW
detail=plan validated
```

这是当时尚无显式 offset、等价于 `action_offset=0` 的现场记录，故保留 `waypoints=49`，不以
新配置倒推改写历史结果。现在显式使用 `--action_offset=1` 时，如果 action 返回前本地 cursor
没有继续推进，预期可发布完整 50 points；若 cursor 已推进，仍按第 6.2 节裁剪真正 stale 的前缀。

服务端 contract 仍是 50-step；本次 ROS chunk 为 49 points 是因为 client 按第 6.2 节移除了
已经过期的 stale prefix。该结果证明 model chunk 已到达 Gateway 并通过纯验证，不证明已武装
或执行。

---

## 19. 源码索引

### LeRobot 仓库

| 内容 | 路径 |
|---|---|
| 完整网络 runbook | `franka_project/ASYNC_CLIENT_SERVER_RUNBOOK.md` |
| gRPC ACK 协议 | `franka_project/ACTION_DELIVERY_ACK_PROTOCOL.md` |
| 早期 ROS interface 说明 | `franka_project/ROS2_INTERFACE.md` |
| Franka plugin config | `franka_project/ros_lerobot/src/lerobot_robot_franka_ros/config_franka_ros.py` |
| ROS observation/cache contract | `franka_project/ros_lerobot/src/lerobot_robot_franka_ros/ros2_contract.py` |
| ROS runtime | `franka_project/ros_lerobot/src/lerobot_robot_franka_ros/ros2_runtime.py` |
| ROS backend/chunk builder | `franka_project/ros_lerobot/src/lerobot_robot_franka_ros/ros2_backend.py` |
| 专用 chunk-aware client | `franka_project/ros_lerobot/src/lerobot_robot_franka_ros/ros2_client.py` |
| Franka PI0 server adapter | `franka_project/src/franka_eef_pipeline/async_server.py` |
| PolicyServer | `src/lerobot/async_inference/policy_server.py` |
| RobotClient | `src/lerobot/async_inference/robot_client.py` |
| gRPC schema | `src/lerobot/transport/services.proto` |
| TCP-over-WebSocket tunnel | `tools/ws_tcp_tunnel.py` |
| ROS Cartesian message | `franka_project/ros2_ws/src/lerobot_franka_interfaces/msg/CartesianActionChunk.msg` |
| ROS gateway ACK | `franka_project/ros2_ws/src/lerobot_franka_interfaces/msg/CartesianActionChunkAck.msg` |
| ROS gateway status | `franka_project/ros2_ws/src/lerobot_franka_interfaces/msg/SafetyGatewayStatus.msg` |

### Franka 仓库

| 内容 | 路径 |
|---|---|
| Gateway 设计说明 | `/home/pnp/franka/haply_ros/src/franka_cartesian_safety_gateway/README.md` |
| Gateway node | `/home/pnp/franka/haply_ros/src/franka_cartesian_safety_gateway/src/gateway_node.cpp` |
| 纯 plan validation | `/home/pnp/franka/haply_ros/src/franka_cartesian_safety_gateway/src/validation.cpp` |
| 状态机/插值/关节运动检查 | `/home/pnp/franka/haply_ros/src/franka_cartesian_safety_gateway/src/execution.cpp` |
| MoveIt preflight | `/home/pnp/franka/haply_ros/src/franka_cartesian_safety_gateway/src/moveit_preflight.cpp` |
| Gateway 参数 | `/home/pnp/franka/haply_ros/src/franka_cartesian_safety_gateway/config/safety_gateway.yaml` |
| Gateway launch | `/home/pnp/franka/haply_ros/src/franka_cartesian_safety_gateway/launch/safety_gateway.launch.py` |
| SafeJointCommand schema | `/home/pnp/franka/haply_ros/src/franka_safety_interfaces/msg/SafeJointCommand.msg` |
| Controller | `/home/pnp/franka/haply_ros/src/franka_arm_controllers/src/joint_impedance_ik_controller.cpp` |
| Controller 参数 | `/home/pnp/franka/haply_ros/src/franka_arm_controllers/config/controllers.yaml` |

---

## 20. 一句话判断系统是否真的闭环

只有同时能证明下面每一段都成立，才能称为 LeRobot 到 Franka 的闭环：

```text
实时 ROS observation
  -> remote PI0 inference
  -> fresh Cartesian chunk
  -> gateway accepted
  -> controller applied
  -> measured robot state 更新
  -> 下一次 observation
```

当前已经证明的是远端 dry-run 通信和若干本地安全组件分项行为；尚未证明的是 gateway ACK 回到
LeRobot、controller-applied feedback、gripper 执行，以及真机低速端到端验收。

---

## 21. 统一编排与分阶段验收入口

新增的安全编排和验收说明见
[`CLOSED_LOOP_STAGED_RUNBOOK.md`](./CLOSED_LOOP_STAGED_RUNBOOK.md)。入口包括：

```text
/home/pnp/franka/scripts/franka_closed_loop.py
/home/pnp/franka/scripts/franka_realtime_health.py
franka_project/scripts/closed_loop_validation_probe.py
```

编排器只接受 `shadow`、`preflight-only` 和
`isolated-single-waypoint` 三种显式模式，启动前检查重复的 arm/gateway/MoveIt/client/camera
进程，并且不会调用 `set_armed(true)`。`preflight-only` 禁止启动 model client；
`isolated-single-waypoint` 还禁止 arm、真 controller、PolicyServer 和 model client。

只读 health artifact 记录 CPU/load、controller lifecycle、EEF/joint source stamp、FCI/reflex
文本指标、gateway mode/status freshness 与 tunnel/client socket。离线 validation artifact 覆盖
50-point contract、timeout/HOLD、expired/replay、session restart 和 2/5-point short chunk。
这些 artifact 是分阶段证据，不是后续真机运行的 arming authorization。
