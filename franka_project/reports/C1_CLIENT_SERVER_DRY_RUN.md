# C1 Franka PI0 client-server dry-run report

> 日期：2026-07-17
> 结论：**本机真实 PI0 + WebSocket 拓扑通过；远端 KML gateway 被 SSO 302 阻塞**
> 安全边界：fixture observation -> JSONL sink；没有 ROS import、ROS graph 或真机命令

## 1. 验收状态

| Gate | 结果 | 证据 |
|---|---|---|
| Out-of-tree plugin 自动发现 | PASS | wheel 安装后的新进程发现 `franka_ros` |
| 严格 client observation 握手 | PASS | exact state names/order、双目 shape、rename map；漂移测试 fail closed |
| Lightweight stock client/gRPC E2E | PASS | RobotClient -> protobuf/gRPC -> adapter -> absolute8 -> JSONL |
| 真实 Franka PI0 checkpoint load | PASS | manifest、全文件 SHA-256、graph/tensor strict load |
| 真实双目 fixture + PI0 inference | PASS | model 输出 `[1,50,7]`，server wire 输出 `[50,8]` |
| 本机 8080 -> WS -> 16782 -> 15173 | PASS | `/ws` 返回 `101`，5 个真实 action chunk 往返 |
| 远端 KML URL -> 16782 | **BLOCKED** | AccessProxy 对 WebSocket Upgrade 返回 `302` SSO；16782 无连接日志 |
| No-core | PASS | 实现前后 `git diff -- src/lerobot` SHA-256 相同 |
| No-ROS / no-hardware | PASS | plugin 无 `rclpy` 依赖，且 `dry_run=false` fail closed |

远端 gate 未通过，所以本报告不把 C4/KML E2E 标记为完成，也不进入 ROS2 阶段。

## 2. 实现边界

- `FrankaPI0PolicyServer` 继承 stock `PolicyServer`，只增加严格 checkpoint loader、client
  schema 握手和 postprocessor 后的 `relative7 -> absolute8` 解码。
- gRPC service、protobuf、queue、`TimedAction`、timestamp/timestep 和 RobotClient 都继续使用
  LeRobot 原实现。
- `lerobot_robot_franka_ros` 通过现有 third-party plugin discovery 注册；Phase 1 backend 只读
  NPZ 并写 JSONL。
- fixture 从现有 `LeRobotDataset` 导出，未复制 MCAP/dataset parser；没有 action label。
- 没有修改 `tools/ws_tcp_tunnel.py`，只通过 CLI 把监听端口设置为 `16782`。

完整启动命令见
[`../ASYNC_CLIENT_SERVER_RUNBOOK.md`](../ASYNC_CLIENT_SERVER_RUNBOOK.md)。

## 3. 环境与冻结输入

```text
git commit: a869295ffadf3e82054fb8f6abdd7d31551ac55f
branch: main
Python: 3.12.13
LeRobot: 0.5.2
PyTorch: 2.11.0+cu128
grpcio: 1.81.1
aiohttp: 3.14.1
GPU: NVIDIA A800-SXM4-80GB（CUDA_VISIBLE_DEVICES=1）
```

Checkpoint：

```text
franka_project/experiments/pi0_full_eef_obs15_act15_v1/
20260715T145931Z_pi0_full_eef_f0_obs15_act15_22of25_b33ece09/
checkpoints/step-037330/pretrained_model
```

| 文件 | bytes | manifest SHA-256 |
|---|---:|---|
| `config.json` | 2,513 | `e2c25de9683b5a82ed9ebbc9e3871a89a79af5e99381ccb884ff82e2a6855a56` |
| `model.safetensors` | 7,312,556,248 | `299cad2d370d1b57d231de5ee553ac534ee9a3b09869023fef47f1e621cc7d21` |
| `policy_preprocessor.json` | 2,106 | `2d1d8cab5f52d43f1ad097309624dfaaf914d33464ff48fde1c48239276f30e1` |
| `policy_postprocessor.json` | 777 | `1c34e47ea2101b993b6c379edb679519fd06ac14a69e8b3f361aec7dcaf2c502` |
| `franka_eef_geometry_manifest.json` | 2,324 | `e9bee10d74e692565222e16e73e61a33df1c1f96ed12177debdaa6ae025f8945` |
| `pi0_eef_stats.json` | 5,995 | `4cd544adeb17bb594131e6507141504e834cb0a1ac76d822c6f36260ee96aed6` |

Fixture：

```text
path: franka_project/fixtures/async/franka_observation_v1.npz
SHA-256: 788109ef6956b687af60062c9751ac06344c61027e3d4c84594dd277d8a35b1c
source: episode 0, frame 0, task "stack the cups"
state: float32 [10]
camera1/camera2: uint8 [480,640,3]
```

## 4. 自动测试

```text
franka_project/tests: 206 passed
LeRobot async unit tests: 24 passed, 1 skipped
wheel build + isolated --no-deps install + plugin discovery: PASS
git diff --check: PASS
```

自动覆盖包括：manifest/hash/geometry/stats、strict load adapter、state/camera/rename
handshake、relative7 decode、gripper clip、退化 rot6d、absolute8/quaternion、plugin lifecycle、
fixture ledger，以及真实 loopback gRPC serialization/sink。

## 5. 真实 PI0 本机 WebSocket E2E

实测拓扑：

```text
stock RobotClient
  -> 127.0.0.1:8080
  -> ws://127.0.0.1:16782/ws
  -> 127.0.0.1:15173
  -> real PI0 checkpoint
  -> JSONL sink
```

Tunnel 证据：

```text
websocket connected from 127.0.0.1 path=/ws
GET /ws HTTP/1.1 101
websocket disconnected from 127.0.0.1
```

模型与时延：

| 指标 | 实测 |
|---|---:|
| checkpoint integrity + construct/load + CUDA cold setup | 200.1425 s |
| cold preprocess + inference | 1.0925 s |
| cold total observation -> serialized chunk | 1.2953 s |
| warm preprocess + inference | 0.2903--0.3004 s |
| warm total observation -> serialized chunk | 0.3241--0.4050 s |
| inferred observation client -> server | 5.20--11.66 ms |
| action server -> client | 4.41--5.42 ms |

5 个真实 chunk 的 server log 均为：

```text
action shape: torch.Size([1, 50, 7])
```

adapter 随后把每个 chunk 转换成 CPU float32 absolute `[50,8]`。最终 sink 验证：

```text
records: 246
all exact absolute8 keys: true
all finite: true
quaternion norm range: [0.9999999706, 1.0000000281]
gripper range: [0.0, 0.0052585006]
```

为什么 5×50 最终是 246：保持未修改的 stock timestep 行为时，后续 chunk 的第一个
timestep 与刚执行的 timestep 重合，RobotClient 会丢弃 `<= latest_action` 的动作。因此首
chunk 消费 50 个，后四个各消费 49 个。该 offset 问题按用户要求暂不修改 LeRobot core。

## 6. KML URL 实测阻塞

目标 URL：

```text
wss://kml-dtmachine-27353-prod-0.kmlhb2az1l3-2.corp.kuaishou.com/ws
```

在 `0.0.0.0:16782` 确实有 tunnel server 监听时，对目标发起 HTTP/1.1 WebSocket Upgrade，
返回：

```text
HTTP/1.1 302 Moved Temporarily
Server: openresty
Location: company SSO login
```

同时 tunnel server 没有出现 `websocket connected`，证明 Upgrade 在 AccessProxy 层被拦截，
不是 PI0、gRPC、16782 进程或 `/ws` path 的问题。需要 KML route 的 machine-to-machine
WebSocket 权限或官方 machine auth；不应把临时浏览器 cookie 固化到 tunnel。

## 7. 保留到真机阶段的问题

- stock timestep 的 1-step overlap/offset；本阶段没有改 core。
- identical observation 会被 `observations_similar()` 过滤，但 client pending flag 只有收到
  action 才清除；本次 fixture 测试因此每隔 10 秒 timeout 后才发送 must-go observation。
- 真机将把 chunk 从 50 重新评审为约 5--10，并定义 plan age/deadline/watchdog。
- absolute quaternion 禁止使用 `weighted_average`；当前固定 `latest_only`。
- ROS topic、frame/TCP、controller command、gripper、safety state、fault/E-stop 和 shutdown
  顺序都尚未实现。
- KML route 鉴权解决后，需要在机器人控制机上重复远端 16782 E2E 和断线测试。

到此停在用户 review gate，不启动 ROS2 适配或真机控制。
