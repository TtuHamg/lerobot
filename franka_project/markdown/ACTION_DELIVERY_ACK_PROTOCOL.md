# LeRobot 异步 action chunk 可靠交付协议

本文说明 LeRobot `RobotClient` 与 `PolicyServer` 之间新增的 action chunk ACK、重投递和
幂等语义。它修复以下故障：服务端已经完成某个 observation（例如 `#588`）的推理，但
`GetActions` 响应恰好在 WebSocket/gRPC 断线窗口丢失；客户端重发 observation 时，旧服务端
又因为 timestep 已被标记为 predicted 而将其过滤，最终双方永久等待。

本文只描述 LeRobot 传输层。ROS2 topic 的字段、observation 同步和当前非执行接口见
[`ROS2_INTERFACE.md`](./ROS2_INTERFACE.md)。KML tunnel 和完整启动命令见
[`ASYNC_CLIENT_SERVER_RUNBOOK.md`](./ASYNC_CLIENT_SERVER_RUNBOOK.md)。

## 1. 结论与边界

现在一个 ACK-capable action chunk 按以下状态迁移：

```text
observation QUEUED
        |
        v
     PREDICTING
        |
        v
  READY_UNACKED  -- 相同 GetActions 重复返回同一 chunk --> READY_UNACKED
        |
        | ActionDeliveryAck(request_id, chunk_id, source_timestep)
        v
      ACKED
```

关键规则：

- 服务端完成推理时只进入 `READY_UNACKED`，不会提前把 timestep 标记为 delivered/predicted；
- 服务端在返回 RPC 之前缓存完整 `Actions` protobuf；未收到匹配 ACK 时，每次
  `GetActions` 都逐字节重发同一个响应，不重复推理；
- 客户端只有在反序列化、校验和本地提交都成功后才发送 ACK；
- ACK 丢失时，客户端识别重复 `chunk_id`，跳过第二次本地提交，只重发 ACK；
- 只有与当前 pending chunk 三个关联字段完全一致的 ACK 才能推进服务端状态；
- `GetActions` 在服务端串行执行，任一时刻最多存在一个 `READY_UNACKED` chunk。

这里的 ACK 是“action chunk 已提交到客户端通信边界”，不是“Franka 已执行完成”。

## 2. 协议字段

`src/lerobot/transport/services.proto` 增加：

```proto
rpc AckActions(ActionDeliveryAck) returns (Empty);

message Actions {
  bytes data = 1;
  string request_id = 2;
  string chunk_id = 3;
  int64 source_timestep = 4;
}

message ActionDeliveryAck {
  string request_id = 1;
  string chunk_id = 2;
  int64 source_timestep = 3;
}
```

三个关联字段的语义：

| 字段 | 生成方 | 稳定范围 | 用途 |
|---|---|---|---|
| `request_id` | client | 同一次逻辑 observation 的发送与传输重试保持不变 | 区分“同 timestep 的不同 observation”和真正重复请求 |
| `chunk_id` | server | 同一个缓存 action response 的所有重投递保持不变 | client 本地 exactly-once/idempotency key |
| `source_timestep` | client observation / server response | 等于推理所用 observation timestep，也等于首个 `TimedAction.timestep` | 可读关联和一致性校验 |

`request_id` 的格式当前为 `<client_session_uuid>:<monotonic_sequence>`。不能只使用 timestep
作为 request ID，因为机器人尚未消费新 action 时，多帧 observation 可能具有同一个 timestep。

## 3. 正常通信时序

```text
RobotClient                    PolicyServer                    ROS2 boundary
    |                               |                                |
    | SendObservations(request_id)  |                                |
    |------------------------------>| QUEUED                         |
    |                               | PREDICTING                     |
    | GetActions                    |                                |
    |------------------------------>| cache READY_UNACKED            |
    | Actions(request_id, chunk_id) |                                |
    |<------------------------------|                                |
    | validate + aggregate queue    |                                |
    | publish_action_chunk() --------------------------------------->|
    | local commit succeeds         |                                |
    | AckActions(ids)               |                                |
    |------------------------------>| ACKED / mark predicted         |
```

stock `RobotClient` 的本地 commit 边界是 action queue 聚合成功。`FrankaRos2RobotClient`
覆盖同一个 `_aggregate_action_queues()` hook，并且只有 `publish_action_chunk()` 返回成功后
该调用才返回，所以 ROS2 模式下 ACK 位于完整 chunk 发布成功之后。如果转换、TTL 校验或 ROS
publish 抛异常，client 不发送 ACK，server 保留 `READY_UNACKED`。

## 4. 断线与重复消息

### 4.1 action response 丢失

1. server 完成推理并先缓存 `Actions`；
2. WebSocket/gRPC 在响应到达 client 前断开；
3. server 保持 `READY_UNACKED`，不会把 observation 当作已交付；
4. client 的 `GetActions` 调用恢复后再次轮询；
5. server 返回完全相同的 `request_id/chunk_id/data`；
6. client 本地提交成功并 ACK。

因此不需要再次推理，也不会发生旧版的“predicted 过滤后永久无响应”。

### 4.2 ACK response 丢失或 ACK RPC 失败

client 在发 ACK 前先记录已提交的 `chunk_id`。若 server 没收到 ACK，下一次仍会返回同一
chunk。client 看到 `chunk_id` 已提交后：

- 不再次聚合 queue；
- 不再次发布 ROS action chunk；
- 直接重发 `ActionDeliveryAck`。

ACK RPC 使用 `pending_observation_timeout_s` 作为 deadline，避免断链时永久占住 action
receiver 线程。

### 4.3 observation 发送 RPC 失败

`SendObservations` 抛出 gRPC transport error 时，client 无法确定 server 是否已经收到完整
payload。此时它保留原 `TimedObservation` 和原 `request_id`，并原样重试同一个逻辑请求。
retry claim 与 action commit 使用同一把锁协调；如果 action 恰好先到并完成，已清除的旧
pending request 不会被重试线程重新激活。

### 4.4 observation 已发送成功，但没有生成 action

server 仍可能按正常策略过滤一帧相似且非 `must_go` 的 observation。`SendObservations` RPC
明确成功后，client 不再保留该帧作为 transport retry payload；如果在
`pending_observation_timeout_s` 内没有收到 action，client 打开 gate 并采集一帧新的
observation，新帧获得新的 `request_id`。

这与 action response 丢失不冲突：若 server 已有 `READY_UNACKED` chunk，`GetActions` 会优先
重放该 chunk，再处理后来排队的新 observation。

## 5. 服务端状态和幂等规则

服务端分别记录：

- queued timestep/request ID；
- inflight timestep/request ID；
- generated-but-unacknowledged timestep；
- 单个 `_pending_delivery`；
- 最近 1024 个已 ACK `chunk_id` tombstone，用于重复 ACK 幂等。

队列 reservation 会在 observation 对 `Queue` 可见之前登记，避免 `GetActions` 刚取走数据，
producer 又写入过期 queued 标记的竞态。推理或序列化失败时，inflight/predicted reservation
会回滚，允许 observation 后续重新进入队列。

每次 `Ready()` 还会递增 server session generation。已经在旧 session 中开始、但直到 reset
之后才完成接收的 `SendObservations` RPC 会因 generation 不匹配被丢弃，不能污染新 session
的 observation queue。

以下消息不会清除 pending chunk：

- 空 `request_id` 或 `chunk_id` 的 ACK；
- request ID、chunk ID 或 source timestep 任一不匹配的 ACK；
- 同 timestep 的重复 observation，包括 `must_go=true`；
- 同 request ID 的传输重试。

## 6. ROS ACK 与未来 controller ACK 的区别

当前 `ActionDeliveryAck` 最多证明：

```text
PolicyServer action
  -> RobotClient 校验/queue
  -> FrankaRos2RobotClient 调用 ROS publisher 成功
```

ROS publisher 返回不证明 DDS subscriber 已收到，更不证明 safety gateway、controller 或
Franka 已消费任何 waypoint。当前 `/lerobot/franka/action_chunk` 仍是隔离、非执行 topic，
不得把本协议的 ACK 当作真机执行进度。

未来接真机时必须增加独立的 ROS execution ACK/status，至少关联：

- ROS session ID 和 plan/chunk ID；
- controller 已接受/拒绝状态；
- `last_applied_timestep` 或明确的已消费 waypoint 数；
- robot mode/error/watchdog；
- ACK 的 ROS/monotonic 时间。

本地 action queue 的 pop 以及本协议的 transport ACK 都不能替代该反馈。

## 7. 兼容性和恢复范围

protobuf 字段是 additive，允许滚动部署：

- 新 client 连接旧 server：收到 bytes-only `Actions`，继续旧行为，不发送 ACK；
- 旧 client 连接新 server：observation 没有 `request_id`，server 走 legacy 行为；
- 只有 client 和 server 都更新后，才具备本文的可靠重投递保护。

当前缓存只在 server 内存中，且 `Ready()` 会重置 session 状态。因此本轮保证：

- 同一 client/server 进程存活期间的 WebSocket 或 gRPC 断线重连；
- action response/ACK 的重复、丢失和重投递。

本轮不保证：

- PolicyServer 进程重启后的恢复；
- client 进程重启或重新调用 `Ready()` 后恢复旧 chunk；
- 多 client 共享一个 PolicyServer；
- 跨进程持久化 exactly-once；
- ROS/controller/Franka 实际执行 exactly-once。

## 8. 部署与运行

client 和 server 都更新到包含本协议的同一提交后，先重启 PolicyServer，再重启 RobotClient。
KML WebSocket tunnel 命令和 ROS2 topic 配置不需要为 ACK 协议修改，继续按两个现有文档启动。

运行中可关注以下日志：

```text
Redelivering unacknowledged action chunk #... (chunk_id=...)
Received duplicate action chunk ...; skipping local commit and retrying ACK
Action chunk #... acknowledged by client (chunk_id=...)
Pending observation timed out after ...; allowing a fresh observation.
Retrying pending observation #... (request_id=...)
```

若持续看到第一行而没有第三行，应检查 client 的反序列化、queue/ROS publish 错误和 ACK RPC
连通性。若 `publish_action_chunk()` 失败，保持未 ACK 是预期的 fail-closed 行为。

## 9. 回归测试

无需 ROS graph 的协议测试：

```bash
cd /home/pnp/Projects/lerobot

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=franka_project/ros_lerobot/src:franka_project/src \
/home/pnp/miniconda3/envs/lerobot/bin/python -m pytest -q \
  tests/async_inference/test_action_delivery_retry.py \
  tests/async_inference/test_policy_server.py \
  tests/async_inference/test_helpers.py \
  franka_project/tests/test_async_policy_server.py
```

真实本地 gRPC、stock client、Franka action sink 的轻量 fault-injection 测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=franka_project/ros_lerobot/src:franka_project/src \
/home/pnp/miniconda3/envs/lerobot/bin/python -m pytest -q \
  franka_project/tests/test_franka_async_lightweight_e2e.py
```

第二个测试会故意丢弃第一次 `GetActions` response，验证第二次拿到逐字节相同的 cached chunk，
并且只有 ACK 后 server 才将 source timestep 标记为 predicted。

## 10. 本次相关文件

- `src/lerobot/transport/services.proto`
- `src/lerobot/transport/services_pb2.py`
- `src/lerobot/transport/services_pb2_grpc.py`
- `src/lerobot/async_inference/helpers.py`
- `src/lerobot/async_inference/policy_server.py`
- `src/lerobot/async_inference/robot_client.py`
- `tests/async_inference/test_action_delivery_retry.py`
- `tests/async_inference/test_policy_server.py`
- `franka_project/tests/test_franka_async_lightweight_e2e.py`
