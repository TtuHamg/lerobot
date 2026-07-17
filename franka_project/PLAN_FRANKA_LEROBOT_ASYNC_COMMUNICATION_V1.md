# Franka + LeRobot PI0 异步通信适配计划

> 状态：**PLANNING APPROVED；C1--C3 与本机 WebSocket C4 已通过，远端 C4 被 KML SSO 302 阻塞**
> 版本：v0.2-implementation
> 日期：2026-07-16
> 工作目录：`/m2v_intern/tujiahang/Projects/lerobot`
> 当前阶段：仅解决 KML 上的 PolicyServer 与机器人侧 RobotClient 通信；不连接 ROS2，不驱动 Franka

## 0. 已确认约束

本计划按以下约束设计；若任一项需要调整，应先修改本文，再开始实现。

- [x] KML 新开放端口使用 `16782`，沿用现有 SO-101 TCP-over-WebSocket 方案，只调整 KML tunnel server 的监听端口。
- [x] KML 内部 PolicyServer 继续绑定 `127.0.0.1:15173`，不直接暴露 gRPC。
- [x] 机器人侧 tunnel client 继续绑定 `127.0.0.1:8080`，RobotClient 只连接这个本地端口。
- [x] 第一阶段只验证 client 和 server 的真实 PI0 异步通信，不读取或发布任何 ROS2 topic。
- [x] 第一阶段可以创建 `ros_lerobot`/Franka Robot 插件，但插件只使用 fixture observation 和本地 action sink。
- [x] 不修改 `src/lerobot/`、现有 protobuf、`TimedAction` 或 LeRobot async queue 逻辑。
- [x] 优先复用现有 PolicyServer、RobotClient、WebSocket tunnel、processor、Franka checkpoint loader 和几何函数。
- [x] 不复制现有 gRPC/WebSocket、图像打包、状态打包、normalization 或 queue 代码。
- [x] Franka offset/timestep 语义暂不处理，也不以修改 LeRobot core 的方式处理。
- [x] ROS2 observation、ROS2 command、controller、安全限制、watchdog、deadline 和真机 rollout 全部保留为后续显式阶段。

### 0.1 关于端口 `16782` 的解释

本文假设 `16782` 是 **KML 平台映射到开发机的 WebSocket tunnel server 端口**，不是 PolicyServer 的 raw gRPC 端口。

KML 对外通常仍提供类似下面的 URL：

```text
wss://<kml-assigned-domain>/ws
```

平台再把该 URL 映射到开发机的 `0.0.0.0:16782`。除非 KML 明确提供 `host:16782` raw 访问，否则 client 的 URL 不手工追加 `:16782`。

## 1. 本阶段目标、非目标与完成定义

### 1.1 目标

第一阶段建立并验证下面这条真实链路：

```text
Franka dry-run observation
  -> stock LeRobot RobotClient
  -> local gRPC 127.0.0.1:8080
  -> WebSocket tunnel client
  -> KML WebSocket gateway
  -> tunnel server 0.0.0.0:16782
  -> local gRPC 127.0.0.1:15173
  -> Franka-aware PI0 PolicyServer
  -> real checkpoint inference
  -> server-side relative7 -> absolute8 decode
  -> 原路返回 RobotClient
  -> dry-run action sink
```

第一阶段需要证明：

1. KML `16782` WebSocket 路由确实双向可用；
2. 当前 Franka PI0 checkpoint 能在 server 端严格加载；
3. client 发出的 10D state、双目图像和 task 能到达 server；
4. server 能产生真实 PI0 action chunk；
5. server 能用触发本次推理的未归一化 state 将 relative 7D chunk 解码为 absolute 8D chunk；
6. stock RobotClient 能接收并消费 absolute 8D action；
7. 整条链路不连接 ROS2，也不会产生任何真机副作用。

### 1.2 本阶段明确不包含

- 不 `import rclpy`，不 source ROS 环境，不创建 ROS node。
- 不订阅相机、EEF、gripper、joint state 或 robot state topic。
- 不发布 pose、joint trajectory、gripper、torque 或 controller command。
- 不实现 IK、Cartesian controller、trajectory interpolation 或 gripper driver。
- 不实现真机 safety guard、deadman、E-stop、watchdog 或 fault recovery。
- 不修改 action offset、client queue、aggregation 算法或 timestamp 协议。
- 不新增自定义 protobuf、WebSocket 协议、消息总线、服务编排或 dashboard。
- 不把通信成功解释为模型具备真机控制能力。

### 1.3 完成定义

本阶段只有同时满足以下条件才算完成：

1. `src/lerobot/` 没有因为本阶段产生新的修改；
2. KML tunnel 的 `16782` 路由通过远端实际 WebSocket 连接验证；
3. 第三方 Franka 插件能被 stock RobotClient 自动发现；
4. 本地直连和 KML WebSocket 两种路径都完成 observation -> inference -> action 的往返；
5. server 使用真实 Franka checkpoint，而不是 mock policy；
6. server 内部模型输出为 `[K,7]`，网络发送给 client 的 action 为 `[K,8]`；
7. dry-run sink 至少连续接收 10 个 finite absolute action；
8. 全程没有 ROS import、ROS graph 连接或机械臂命令；
9. 产出一次可复核的命令、日志、checkpoint hash、shape 和延迟报告；
10. 完成后停在 review gate，未经用户确认不进入 ROS2 阶段。

## 2. 通信框架

### 2.1 进程拓扑

```text
KML 开发机

  [1] Franka PI0 PolicyServer
      bind: 127.0.0.1:15173
      responsibilities:
        - strict checkpoint load
        - observation preprocessing
        - PI0 inference
        - action unnormalization
        - relative7 -> absolute8 decode
                ^
                | local gRPC/TCP
                v
  [2] tools/ws_tcp_tunnel.py server
      bind:   0.0.0.0:16782
      target: 127.0.0.1:15173
                ^
                | KML managed WebSocket gateway
                | wss://<assigned-domain>/ws
                v

机器人侧控制机（第一阶段不连接 ROS/Franka）

  [3] tools/ws_tcp_tunnel.py client
      bind:   127.0.0.1:8080
      target: KML WebSocket URL
                ^
                | local gRPC/TCP
                v
  [4] stock lerobot.async_inference.robot_client
      robot: third-party franka_ros plugin in dry-run mode
                |
                v
  [5] fixture observation + JSONL/debug action sink
      no ROS, no controller, no hardware side effect
```

KML gateway 是平台托管路由，不是需要自行启动的进程。

### 2.2 端口表

| 位置 | 地址 | 用途 | 是否外部暴露 |
|---|---|---|---|
| KML PolicyServer | `127.0.0.1:15173` | LeRobot gRPC server | 否 |
| KML tunnel server | `0.0.0.0:16782` | 接收 KML gateway 转入的 WebSocket | 是，由 KML URL 映射 |
| 机器人侧 tunnel client | `127.0.0.1:8080` | RobotClient 的本地 gRPC 入口 | 否 |
| ROS2/DDS | 本阶段无 | 明天再设计 | 否 |

RobotClient **不会**连接 `16782`，只连接 `127.0.0.1:8080`。

### 2.3 Observation 数据流

第一阶段插件从一个冻结 fixture 读取：

```text
state:   float32 [10]
camera1: uint8 [480,640,3], RGB, HWC
camera2: uint8 [480,640,3], RGB, HWC
task:    "stack the cups"
```

10D state 顺序固定为：

```text
[x, y, z,
 rot6d_c1_x, rot6d_c1_y, rot6d_c1_z,
 rot6d_c2_x, rot6d_c2_y, rot6d_c2_z,
 gripper_0_1]
```

规则：

- fixture 必须来自现有转换数据或由现有 pipeline 导出，不重新实现 MCAP/dataset 解析；
- fixture 使用 `npz`，`allow_pickle=False`；
- 插件在 `connect()` 时一次性验证 shape、dtype、finite、合法 rot6d 和图像布局；
- 插件将 fixture 的 state 展开为 10 个有序 scalar feature，让 LeRobot 现有 feature 工具统一组装 `observation.state`；插件不自行构造 policy tensor；
- 图像 raw key 固定为 `camera1`、`camera2`，client 复用 `rename_map` 映射到：

```text
observation.images.base_0_rgb
observation.images.left_wrist_0_rgb
```

- 不伪造第三路 `right_wrist_0_rgb`；继续复用 PI0 对缺失 camera slot 的 mask 逻辑；
- normalization 只由 checkpoint processor 执行，client/plugin 不重复实现。

### 2.4 Action 数据流

Policy 的内部输出仍是训练契约的 7D action：

```text
[delta_x_base, delta_y_base, delta_z_base,
 rotvec_body_x, rotvec_body_y, rotvec_body_z,
 gripper_target_0_1]
```

server 在 postprocessor 反归一化之后，用本次 observation 的原始 10D state 解码整个 chunk：

```text
p_target = p_anchor + delta_p_base
R_target = R_anchor @ Exp(rotvec_body)
g_target = clip(g_prediction, 0, 1)
```

chunk 中所有 waypoint 使用同一个 observation anchor，不递推累加。几何实现直接复用：

```text
franka_eef_pipeline.geometry.rotation_6d_to_matrix
franka_eef_pipeline.geometry.decode_relative_action
franka_eef_pipeline.geometry.matrix_to_quaternion_xyzw
```

网络与 client 插件之间的 action schema 固定为 absolute 8D：

```text
[x, y, z, qx, qy, qz, qw, gripper_0_1]
```

第一阶段 action sink 只做：

- exact key/order/dimension 检查；
- finite 检查；
- quaternion norm 检查；
- 写入本地 JSONL/debug log；
- 返回实际接收的 action；
- 不发布 ROS，不执行运动。

### 2.5 本阶段时序配置

第一阶段固定：

```yaml
fps: 15
model_chunk_size: 50
actions_per_chunk: 50
aggregate_fn_name: latest_only
enable_pending_observation: true
pending_observation_timeout_s: 10
client_device: cpu
```

说明：

- dry-run 使用完整 50 步，目的是验证真实 payload、网络带宽和完整 shape；
- 真机阶段会重新评审 `actions_per_chunk`，初期预计缩短到 5--10；
- 不使用 `weighted_average`，避免对 absolute quaternion 逐元素线性平均；
- 本阶段保持 stock `TimedAction`、timestep 和 timestamp 行为不变；
- Franka offset、deadline、plan age 和 session 语义登记在后续事项中，不在本阶段处理。

## 3. 复用原则与最小新增边界

| 能力 | 直接复用 | 本阶段只新增什么 |
|---|---|---|
| gRPC service/protobuf | `lerobot.transport`、`PolicyServer`、`RobotClient` | 无 |
| KML 网络穿透 | `tools/ws_tcp_tunnel.py` | server listen port 改为 CLI 参数 `16782` |
| async queue/timing | `lerobot.async_inference` | 无 |
| PI0 model | 当前 Franka checkpoint | 严格加载 adapter |
| pre/post processor | checkpoint processor JSON/stats | 只校验，不重新实现 |
| EEF 几何 | `franka_eef_pipeline.geometry` | server 调用已有函数 |
| Robot 抽象 | LeRobot `Robot` API 和 plugin discovery | out-of-tree Franka dry-run plugin |
| observation 数据 | 现有转换 pipeline | 导出一份小型冻结 fixture |
| action 消费 | `RobotClient.control_loop_action()` | 无副作用 sink |

禁止通过复制 `policy_server.py`、`robot_client.py` 或 tunnel 脚本来实现 Franka 版本。

## 4. 计划新增文件（用户确认后才创建）

建议的最小文件结构：

```text
franka_project/
├── PLAN_FRANKA_LEROBOT_ASYNC_COMMUNICATION_V1.md
├── src/franka_eef_pipeline/
│   └── async_server.py
├── scripts/
│   └── serve_franka_pi0_async.py
├── ros_lerobot/
│   ├── pyproject.toml
│   ├── README.md
│   └── src/lerobot_robot_franka_ros/
│       ├── __init__.py
│       ├── config_franka_ros.py
│       └── franka_ros.py
├── fixtures/async/
│   └── franka_observation_v1.npz
├── tests/
│   ├── test_async_policy_server.py
│   └── test_franka_ros_dry_run.py
└── reports/
    └── C1_CLIENT_SERVER_DRY_RUN.md       # 验收后生成
```

### 4.1 `async_server.py`

职责严格限制为：

1. 继承并复用 `lerobot.async_inference.policy_server.PolicyServer`；
2. 使用项目已有的严格 PI0 checkpoint 加载逻辑，加载失败直接退出，不 fallback 到随机初始化；
3. 复用 checkpoint processor；
4. 在 postprocess 后调用现有 geometry 函数，将 `[K,7]` 转成 `[K,8]`；
5. 对输入 state、模型输出和 wire action 做 shape/finite 断言；
6. 其余 Ready、PolicyInstructions、Observation、GetActions、queue 和 gRPC handler 全部复用基类。

不在该文件实现 WebSocket、ROS、controller、安全状态机或新协议。

### 4.2 `serve_franka_pi0_async.py`

只负责：

- 解析/复用 `PolicyServerConfig`；
- 实例化 Franka PolicyServer adapter；
- 注册到现有 gRPC service；
- 启动和退出日志。

它是薄 launcher，不复制 inference handler。

### 4.3 `lerobot_robot_franka_ros` 插件

插件 distribution 和 import package 都使用合法且可自动发现的名称：

```text
lerobot_robot_franka_ros
```

注册：

```text
Robot type: franka_ros
Config:     FrankaRosConfig
Robot:      FrankaRos
```

第一阶段只支持：

```yaml
dry_run: true
fixture_path: <npz>
action_log_path: <jsonl>
```

如果设置 `dry_run: false`，必须 fail closed 并提示 ROS backend 尚未实现。第一阶段不声明 `rclpy` 依赖。

插件只实现 LeRobot `Robot` 的标准生命周期：

```text
connect             加载并验证 fixture，打开 sink
get_observation     返回 fixture 的 copy
send_action         校验 absolute8 并写 sink
disconnect          flush/关闭 sink
is_calibrated       True
calibrate/configure no-op
```

不额外创建 client launcher；继续使用 stock：

```text
python -m lerobot.async_inference.robot_client
```

## 5. 分阶段执行计划

### C0：Planning review（当前阶段）

交付：

- 本 planning 文档；
- 明确范围、端口、wire schema、复用点、阶段和验收条件。

退出条件：

- 用户明确确认 planning；
- 未确认前不实现任何 adapter/plugin/runtime code。

### C1：Preflight 与纯 transport probe

目的：先证明 KML `16782` 通路，不把模型和插件问题混进网络排障。

检查：

1. KML 平台 hostname/URL 已实际映射到开发机 `16782`；
2. KML `16782` 未被其他进程占用；
3. KML `15173` 只在 loopback 监听；
4. client `8080` 只在 loopback 监听；
5. 两端环境能 import `aiohttp` 和 `grpc`；当前 tunnel 的 `aiohttp` 不假定由 `lerobot[async]` 自动安装；
6. 保存开始实现前已有的 `src/lerobot/` diff 作为只读 baseline，本阶段不得扩大该 diff；
7. 使用 stock `Ready` RPC 做一次不加载模型的探针。

拟用命令：

```bash
# KML：临时 stock server，只用于 Ready probe
uv run python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=15173

# KML：只把旧 SO-101 tunnel 端口换成 16782
uv run python tools/ws_tcp_tunnel.py server \
  --listen-host=0.0.0.0 \
  --listen-port=16782 \
  --target-host=127.0.0.1 \
  --target-port=15173

# 机器人侧控制机
uv run python tools/ws_tcp_tunnel.py client \
  --listen-host=127.0.0.1 \
  --listen-port=8080 \
  --ws-url="$KML_WS_URL"
```

验收：

- Ready RPC 在 10 秒内成功；
- tunnel 两端分别出现 local TCP、WebSocket 和 target TCP 连接日志；
- 重复短连接至少 3 次均可重新建立；
- 无 `websocket connection failed` 或 `failed to connect target`；
- 明确记录：这一阶段只证明字节流，不证明模型或 ROS。

### C2：Server adapter 本地验证

目的：在不经过 KML 网络时先消除 checkpoint 和 geometry 问题。

步骤：

1. 冻结 checkpoint 路径、model/config/processor/stats/geometry manifest hash；
2. 使用 strict loader 加载真实 checkpoint，并执行 `policy.eval()`；
3. 用 fixture observation 完成一次真实 PI0 inference；
4. 断言内部输出 `[1,50,7]` 且 finite；
5. 断言 server decode 后输出 `[50,8]`；
6. 用已有 geometry 函数做独立 golden cross-check；
7. 错 checkpoint、错 stats、错 state shape、NaN/Inf 均 fail closed。

退出条件：

- 本地 GPU smoke 通过；
- 不修改 LeRobot core；
- 不启动 WebSocket 或 ROS。

### C3：Franka dry-run plugin 与本地 gRPC E2E

目的：验证 stock RobotClient 能使用 out-of-tree 插件完成完整闭环。

步骤：

1. editable install `lerobot_robot_franka_ros`；
2. 在全新 Python 进程调用 plugin discovery，确认 `franka_ros` 已注册；
3. RobotClient 直连 `127.0.0.1:15173`；
4. fixture observation 经真实 gRPC 到达 server；
5. server 返回 absolute8 chunk；
6. plugin sink 连续记录 action；
7. 检查 feature 顺序、shape、finite、quaternion norm 和日志 flush。

退出条件：

- 本地 direct gRPC observation -> real PI0 -> absolute8 -> sink 成功；
- no-ROS gate 通过；
- action sink 没有外部副作用。

### C4：KML `16782` 真实 E2E

目的：只把 C3 的 gRPC 字节流换成现有 WebSocket tunnel，验证最终 client-server 通信。

启动顺序：

1. KML Franka PolicyServer：`127.0.0.1:15173`；
2. KML tunnel server：`0.0.0.0:16782 -> 127.0.0.1:15173`；
3. 机器人侧 tunnel client：`127.0.0.1:8080 -> $KML_WS_URL`；
4. stock RobotClient：`server_address=127.0.0.1:8080`、`robot.type=franka_ros`、`dry_run=true`。

验收：

- 至少 10 个 absolute8 action 到达 sink；
- server 日志包含 checkpoint load、observation、inference 和 chunk shape；
- client 日志包含 plugin discovery、observation send、chunk receive 和 action sink；
- 记录 cold/warm inference latency、client->server、server->client、queue depth；
- 使用真实 480x640 双目 payload，确认 KML gateway 不因帧大小/带宽断开；
- 主动停止 tunnel 后错误可见、进程可清理；不为本阶段另写重连状态机。

### C5：报告与用户 review gate

生成：

```text
franka_project/reports/C1_CLIENT_SERVER_DRY_RUN.md
```

内容至少包括：

- 实际命令与环境；
- 三端口和 KML URL 映射；
- checkpoint/config/processor/stats hash；
- observation/action schema；
- direct gRPC 与 KML WS 日志摘要；
- chunk shape、数量和 finite 检查；
- cold/warm latency；
- 断线现象；
- 明确的 out-of-scope 和遗留问题。

完成后停止，不自动进入 ROS2 或真机阶段。

## 6. 测试计划

### 6.1 自动测试

1. **Plugin registration**：editable install 后在独立进程自动发现 `franka_ros`。
2. **Plugin contract**：精确检查 10D state、两路 HWC RGB image 和 absolute8 action key 顺序。
3. **Plugin lifecycle**：未连接时拒绝 get/send；connect/disconnect 可重复；fixture/sink 正确关闭。
4. **Fixture validation**：坏 shape/dtype、NaN/Inf、错误 rot6d、错误图像布局全部拒绝。
5. **Server strict load**：错误 checkpoint/processor/stats 直接失败，无随机初始化 fallback。
6. **Geometry decode**：relative7 -> absolute8 与现有 geometry golden 结果一致。
7. **Lightweight async E2E**：复用现有 async test pattern，用轻量 policy 验证真实 gRPC handshake、observation、action queue 和 sink，不在常规 pytest 中加载 3B 模型。
8. **No-core gate**：与 preflight baseline 比较，检查本阶段没有给 `src/lerobot/` 增加或改变 diff；不误删用户已有改动。
9. **No-ROS gate**：纯 LeRobot 环境能 import 和运行 plugin，且依赖中没有 `rclpy`。

### 6.2 手动 GPU/KML smoke

- 真实 PI0 checkpoint GPU load/inference；
- 本机 direct gRPC；
- KML `16782` WebSocket E2E；
- 双目真实大小 payload；
- tunnel 断开和重新启动；
- 日志和报告归档。

## 7. 失败策略

第一阶段虽然不连接真机，仍采用 fail-closed：

- checkpoint、config、stats、processor 或 manifest 不匹配：server 不启动；
- observation/action dimension 或 feature order 不匹配：拒绝该次运行；
- state/action 出现 NaN/Inf：不写 sink；
- quaternion 非法：不写 sink；
- KML route 未映射到 `16782`：停在 C1，不绕过平台网络模型；
- tunnel 断开：记录并结束当前 smoke，不新增自定义重连逻辑；
- PolicyServer 重启：重启 RobotClient，重新完成 Ready/PolicyInstructions；
- `dry_run=false`：plugin 直接拒绝启动，防止误以为已实现 ROS。

## 8. 后续阶段必须保留的问题清单

以下内容不在本阶段实现，但不得在后续遗漏。

### R1：ROS2 observation backend

- 订阅 current EEF、两路相机、gripper、joint state、robot mode/error；
- 以 camera1 15 Hz 为 anchor，按 header stamp 做因果同步和 freshness gate；
- 构造严格 10D state、RGB image、task；
- configured EE/TCP、base frame、相机顺序、图像方向和 gripper convention 与训练一致；
- `get_observation()` 读取原子 cache，不在 LeRobot 15 Hz loop 内阻塞等待多个 topic。

### R2：ROS2 command 与 controller boundary

- ROS 只接收 server 已解码的 absolute Cartesian pose + gripper；
- 不在 ROS 侧用“收到时当前 pose”重新解释 relative chunk；
- 明确现有 Franka controller 类型、command topic/action、frame 和 QoS；
- Python/LeRobot 只给低频 waypoint，本地 controller 做高频插值；
- Robotiq gripper 使用独立 driver、映射、滞回和限频。

### R3：本地安全与 watchdog

- 独立 safety gateway，而不是依赖 KML/server heartbeat；
- workspace、单步平移/旋转、速度/加速度/jerk、joint limits、IK、奇异位形、碰撞、NaN/Inf 检查；
- robot/controller/state/command freshness；
- DISABLED/ARMED/RUNNING/HOLD/FAULT 状态机；
- deadman、E-stop、断线 HOLD、长超时 controlled stop；
- 旧 queue 不得在失联后长期继续执行。

### R4：异步时序与计划有效性

- Franka action offset 的最终语义；
- command step 与 target arrival time 的区别；
- source observation、plan/session sequence、deadline/TTL；
- 迟到 chunk 和过期 waypoint 的丢弃；
- queue overlap、absolute SE(3) aggregation；
- 真机阶段缩短 `actions_per_chunk`，重新测量 p95/p99 latency。

### R5：图像与模型一致性

- ROS encoding 到 RGB 的转换；
- 480x640 到模型输入的 resize/pad 必须与训练一致，不能无意拉伸；
- 两个真实 camera slot 与缺失第三 slot 的 mask；
- live observation 与离线 dataset sample 做 golden comparison。

### R6：部署与安全通信

- WSS、KML access control、最小暴露面；
- 当前 pickle/gRPC 链路只在可信网络使用；
- 进程监督、日志轮转、health check；
- PolicyServer 重启后的 client handshake/restart 策略；
- WebSocket 20 秒 heartbeat 不作为机器人 watchdog。

### R7：真机 rollout 能力边界

- 训练 action 是 future measured EEF waypoint proxy，不是原 controller desired command；
- 先 shadow inference，再 fake controller，再单 waypoint；
- 低速度、低 stiffness、空旷 workspace、人工 enable/deadman；
- 记录 desired、accepted、applied 和 measured trajectory；
- 未完成独立真机安全 review 前，不允许连续异步 rollout。

## 9. 用户 review 清单

开始实现前，请确认以下决策：

1. `16782` 是 KML WebSocket tunnel server 的映射端口；内部 `15173` 和 client 本地 `8080` 保持不变。
2. 第一阶段包含两级验证：先 Ready transport probe，再跑真实 checkpoint E2E。
3. 第一阶段插件名为 `lerobot_robot_franka_ros` / robot type `franka_ros`，但只支持 `dry_run=true`。
4. 第一阶段 observation 来自冻结 fixture，不读取 ROS。
5. server 内部 action 为 relative7，client-server wire 为 absolute8，client 只写 debug sink。
6. 使用 stock RobotClient、PolicyServer 基类、gRPC/protobuf 和 tunnel，不改 LeRobot core。
7. 第一阶段固定 `fps=15`、`actions_per_chunk=50`、`latest_only`；真机参数后续重新评审。
8. C1--C5 通过并提交报告后停止，等用户在真机旁再次确认才开始 ROS2 适配。
