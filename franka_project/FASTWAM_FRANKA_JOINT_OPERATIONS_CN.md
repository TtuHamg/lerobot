# FastWAM ↔ Franka 关节控制完整运行手册

> 更新时间：2026-07-25
> 工作区：`/home/pnp/ght_wsp/lerobot`
> 适用对象：FastWAM 绝对关节位置策略、Franka FR3、ROS 2 Jazzy、30 FPS 训练/推理
> 当前结论：SHADOW 和通信验证可用；在“物理计划完成 ACK”闭环修复前，禁止再次进入真机 ARMED 执行。

## 1. 安全边界

1. `Publisher count: 1 / Subscription count: 1` 只证明 ROS 拓扑连通，不证明 action 安全。
2. `set_armed(true)` 成功只证明调用瞬间的本地门控满足，不证明后续轨迹连续。
3. `RobotClient` 本地 action queue 清空不等于 Franka 已执行完计划。gateway 可能把 30 Hz 轨迹最多放慢 8 倍。
4. 任何大幅晃动、异常声音、碰撞风险或不可预测运动：先按急停，再排查日志。
5. 每次真机前必须有人在急停旁值守，清空工作空间，确认 Franka Desk 无错误。
6. 不得把 SO-101 checkpoint、stats 或关节顺序直接用于 Franka。

### 1.1 当前真机阻断项

以下项目全部完成并经过 SHADOW/测试验证之前，不得执行本文第 9 节的 ARM 操作：

- client 订阅 `/lerobot/franka/joint_action_chunk_ack` 和
  `/lerobot/franka/joint_safety_gateway_status`；
- client 仅在匹配 `session_id/plan_id` 的物理计划完成后采下一帧 observation；
- 计划完成条件使用 gateway status，而不是本地 queue；
- gateway 拒绝在旧计划仍 active 时用新 chunk 替换；
- gateway 校验跨 chunk 的位置、速度和加速度连续性；
- 初期真机阈值收紧到低速验证范围；
- ACK、计划完成、拒绝替换和 tracking HOLD 均有自动测试。

2026-07-25 的真机事件中，gateway 在最大关节跟踪误差达到 `0.250972 rad`
（约 14.4°）后进入 HOLD。停止保护生效，但阈值过宽，不能用相同配置再次执行。

## 2. 端到端链路和固定端口

```text
Franka ROS observations
  -> joint_ros2_client
  -> local TCP 127.0.0.1:8081
  -> WebSocket/KML gateway
  -> policy-host tunnel 0.0.0.0:15645
  -> FastWAM gRPC 127.0.0.1:15174
  -> 16 x absolute joint action
  -> /lerobot/franka/joint_action_chunk
  -> joint safety gateway
  -> /franka/safe_joint_command
  -> joint_impedance_ik_controller
```

| 项目 | 固定值 |
|---|---:|
| 训练/推理 FPS | 30 |
| FastWAM gRPC | 15174 |
| policy-host tunnel listen | 15645 |
| robot-host local tunnel | 8081 |
| action horizon | 32 |
| actions per chunk | 16 |
| camera keys | `camera1 camera2` |
| action 维度 | 8（7 joints + gripper） |
| 当前真机执行维度 | 7 joints；gateway 不执行 gripper action |

## 3. 终端分工

| 终端 | 主机 | 任务 |
|---|---|---|
| P1 | policy/GPU | FastWAM server |
| P2 | policy/GPU | WebSocket tunnel server |
| R1 | robot | KML tunnel client |
| R2 | robot | Franka stack |
| R3 | robot | cameras |
| R4 | robot | LeRobot client |
| R5 | robot | status/ACK/急停监控 |

关键进程不要全部后台堆在同一个终端，否则无法确认哪个进程已经退出。

## 4. 一次性准备

### 4.1 Policy/GPU 主机

FastWAM server 不依赖 ROS，只激活 FastWAM/PyTorch 环境：

```bash
conda activate fastwam
cd /m2v_intern/genghaotian/FastWAM

test -f runs/frank3_finetune/<RUN>/config.yaml
test -f runs/frank3_finetune/<RUN>/checkpoints/weights/step_NNNNNN.pt
test -f runs/frank3_finetune/<RUN>/dataset_stats.json
test -d data/text_embeds_cache/frank3
```

必须核对：

- state/action 顺序是 `fr3_joint1..fr3_joint7, gripper`；
- checkpoint、config、stats 来自同一次 Franka 训练；
- 数据集、server 和 robot client 的 FPS 都是 30；
- `camera1/camera2` 与训练时的相机角色一致。

### 4.2 Robot 主机 ROS overlay

修改 ROS message 或 gateway 后：

```bash
cd /home/pnp/ght_wsp/lerobot/franka_project/ros2_ws
source /opt/ros/jazzy/setup.bash
source /home/pnp/franka/franka_ros2_ws/install/setup.bash
source /home/pnp/franka/haply_ros/install/setup.bash

colcon build --packages-select \
  lerobot_franka_interfaces \
  franka_joint_safety_gateway

source install/setup.bash
```

安装当前工作区插件：

```bash
cd /home/pnp/ght_wsp/lerobot
uv pip install --no-deps -e franka_project/ros_lerobot
```

### 4.3 主机实时性

```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
```

应全部为 `performance`。Franka 日志若提示非实时内核，说明仍不能保证 1 kHz deadline；
长期部署应使用 PREEMPT_RT 内核并完成网卡 IRQ/EEE 调优。

## 5. 每次运行前：清理与 Desk 检查

Robot 主机：

```bash
cd /home/pnp/ght_wsp/lerobot
bash /home/pnp/franka/stop_all.sh --keep-haply-manager
pgrep -af 'ros2_control_node|joint_safety_gateway|joint_ros2_client|robot_client'
```

Franka Desk：

- Acknowledge 上一次 reflex；
- 急停已释放但操作员手边可立即触达；
- robot 已解锁并允许 FCI；
- 工作空间清空；
- 不存在另一个 controller/gateway/client。

`stop_all.sh` 打印的 `/home/pnp/franka/logs/01_arm_controller.log` 可能是历史故障。
本轮日志以 `/home/pnp/franka/logs/joint_validation/` 为准。

## 6. 启动 Policy server 与隧道

### 6.1 P1：FastWAM Franka server

```bash
conda activate fastwam
cd /m2v_intern/genghaotian/FastWAM
export PROMPT='Pick up the cup.'

python scripts/franka_fastwam_async_server.py \
  --run-config runs/frank3_finetune/<RUN>/config.yaml \
  --checkpoint runs/frank3_finetune/<RUN>/checkpoints/weights/step_NNNNNN.pt \
  --stats runs/frank3_finetune/<RUN>/dataset_stats.json \
  --text-cache data/text_embeds_cache/frank3 \
  --prompt "$PROMPT" \
  --cam-keys camera1 camera2 \
  --action-horizon 32 \
  --actions-per-chunk 16 \
  --fps 30 \
  --num-inference-steps 10 \
  --port 15174 \
  --device cuda:0 \
  --diagnose-n 3 \
  --dump-images-dir /m2v_intern/genghaotian/FastWAM/franka_fastwam_dump
```

首次推理检查：state raw 为 8 维、proprio pad 后为 16 维、两路图像统计合理、
raw action 为 8 维 finite 值、`action[0]-state` 无明显跳变、没有 tensor 8/10 维度错误。

### 6.2 P2：policy-host tunnel server

```bash
cd /m2v_intern/genghaotian/FastWAM
python tools/ws_tcp_tunnel.py server \
  --listen-host=0.0.0.0 \
  --listen-port=15645 \
  --target-host=127.0.0.1 \
  --target-port=15174
```

不要把 `target-port` 写成 `1`、`15173` 或被换行截断的端口。

### 6.3 R1：robot-host tunnel client

先在 Firefox 正常完成 KML SSO/MFA，然后：

```bash
cd /home/pnp/ght_wsp/lerobot
export KML_WS_URL='wss://<KML-HOST>/ws'
python tools/start_kml_tunnel_client.py \
  --listen-host=127.0.0.1 \
  --listen-port=8081 \
  --ws-url="$KML_WS_URL"
```

验证：

```bash
ss -lntp | rg ':8081'
```

robot client 的 `--server_address` 必须是 `127.0.0.1:8081`。

## 7. 当前允许的完整 SHADOW 验证顺序

SHADOW 不允许 actuation，但仍会连接 Franka FCI 并启动 controller，因此仍需 Desk 就绪和急停值守。

### 7.1 R2：先启动无相机 stack

```bash
cd /home/pnp/ght_wsp/lerobot
bash franka_project/scripts/start_joint_stack.sh --shadow --no-cam
```

必须看到：

```text
JOINT VALIDATION READY — SHADOW / NOT ARMED
```

若出现 `communication_constraints_violation`，停止并在 Desk Acknowledge；不要继续启动 client。

### 7.2 R3：stack 稳定后启动相机

```bash
nohup bash /home/pnp/franka/start_cameras.sh \
  > /home/pnp/franka/logs/joint_validation/cameras_after_stack.log 2>&1 &
```

等待至少 20 秒：

```bash
ros2 control list_controllers
ros2 topic info /camera1/camera1/color/image_raw
ros2 topic info /camera2/camera2/color/image_raw
ros2 topic info /franka/joint_states
ros2 topic info /gripper/joint_states
```

三个 controller 必须全部为 `active`，四个 observation topic 必须各有 publisher。

### 7.3 R4：启动 client

```bash
cd /home/pnp/ght_wsp/lerobot
source /home/pnp/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_ght
source /opt/ros/jazzy/setup.bash
source /home/pnp/franka/franka_ros2_ws/install/setup.bash
source /home/pnp/franka/haply_ros/install/setup.bash
source /home/pnp/ght_wsp/lerobot/franka_project/ros2_ws/install/setup.bash
export PROMPT='Pick up the cup.'

python -m lerobot_robot_franka_ros.joint_ros2_client \
  --robot.type=franka_ros_joint \
  --robot.dry_run=false \
  --robot.ros2_interface_only=true \
  --robot.max_observation_age_s=0.5 \
  --robot.camera2_max_skew_s=0.1 \
  --robot.qpos_max_skew_s=0.2 \
  --robot.gripper_max_skew_s=0.1 \
  --robot.observation_buffer_size=64 \
  --robot.action_chunk_validity_s=1.0 \
  --server_address=127.0.0.1:8081 \
  --policy_type=act \
  --pretrained_name_or_path=/dummy \
  --task="$PROMPT" \
  --actions_per_chunk=16 \
  --chunk_size_threshold=0.0 \
  --enable_pending_observation=true \
  --pending_observation_timeout_s=120 \
  --aggregate_fn_name=latest_only \
  --fps=30
```

`chunk_size_threshold=0.0` 只保证本地 queue 清空后才采新 observation，当前不能保证物理计划完成。
这是 SHADOW 可观察但 EXECUTE 被阻断的原因。

### 7.4 R5：SHADOW 验收

```bash
ros2 topic info /lerobot/franka/joint_action_chunk
ros2 topic info /franka/safe_joint_command
ros2 topic echo /lerobot/franka/joint_action_chunk_ack
ros2 topic echo /lerobot/franka/joint_safety_gateway_status
```

验收条件：

- action chunk：`Publisher count: 1`、`Subscription count: 1`；
- SHADOW 下不允许产生可执行 ARMED command；
- ACK 无 schema/shape/nonfinite/ordering 错误；
- server 连续输出 finite action；
- client 无 camera/qpos/gripper stale/skew 风暴；
- controller 保持 active，无 FCI/reflex；
- 记录每个 ACK detail 中的 gateway retimed period。

SHADOW 完成后按第 10 节停机，不要直接切换 EXECUTE。

## 8. 真机前必须完成的顺序执行协议

```text
capture observation N
  -> infer chunk N
  -> gateway uses fresh qpos to validate hard limits/step/velocity/acceleration
  -> gateway ACK accepts and optionally retimes plan N
  -> execute all retimed waypoints of plan N
  -> gateway's scheduled plan horizon completes
  -> capture observation N+1
```

注意：当前 motion-limits-only 模式不以 controller feedback 或 tracking
error 判定物理执行完成；`has_active_plan:false` 只表示网关计划时域已结束。

禁止继续使用：

```text
local queue empty
  -> capture next observation
  -> publish new chunk
  -> replace still-running retimed plan
```

建议初期真机安全配置：

```yaml
max_joint_step_rad: [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
max_joint_velocity_rad_s: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
max_joint_acceleration_rad_s2: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
hold_after_plan_completion: true
```

这些值是低速验证上限，不是最终部署参数。修改后必须重新 build、SHADOW 和测试。

## 9. EXECUTE 与 ARM 顺序（当前阻断，修复后使用）

> 当前代码尚未实现第 8 节的物理计划完成 ACK 闭环。本节不得在当前版本直接执行。

修复并通过评审后：

1. 重复第 5、6 节；
2. `bash franka_project/scripts/start_joint_stack.sh --execute --no-cam`；
3. 等待 `EXECUTE / NOT ARMED`；
4. 单独启动相机并稳定至少 20 秒；
5. 启动具备 execution ACK gating 的 client；
6. 检查 topic 拓扑和全部 controller；
7. status 必须为：

```text
armed: false
state_fresh: true
preflight_available: true
robot_ready: true
controller_ready: true
has_active_plan: false
```
pick up the cup
bash franka_project/scripts/run_joint_client.sh "pick up the cup"
`robot_ready/controller_ready:true` 在本模式中表示这些 gateway gate 已禁用，
不是实时 readiness 证明。

8. 确认工作空间无碰撞风险且急停可达；
9. 由现场操作员显式 ARM；
10. 首次只允许单个低速 chunk；
11. 人工监控 ACK、controller active、tracking error 和 controller feedback；
12. chunk 完成并回到 `has_active_plan:false` 后，才允许下一次 observation。

ARM service：

```text
/franka_joint_safety_gateway/set_armed
std_srvs/srv/SetBool
```

操作员必须先掌握第 10.1 节 DISARM。本文不会把 ARM 命令放入可整段复制的启动脚本。

## 10. 停机顺序

### 10.1 紧急停止

机械臂仍在异常运动：先按物理急停。

机械臂已静止且 ROS service 可用：

```bash
ros2 service call /franka_joint_safety_gateway/set_armed \
  std_srvs/srv/SetBool '{data: false}'
```

停止 client，然后：

```bash
bash /home/pnp/franka/stop_all.sh --keep-haply-manager
```

### 10.2 正常停机

```text
DISARM
  -> 确认 armed:false / has_active_plan:false
  -> Ctrl-C robot client
  -> stop_all.sh
  -> Ctrl-C robot tunnel client
  -> Ctrl-C policy tunnel server
  -> Ctrl-C FastWAM server
```

不要先杀 gateway 再尝试 disarm。

## 11. 运行中监控

```bash
ros2 topic echo /lerobot/franka/joint_safety_gateway_status
ros2 topic echo /lerobot/franka/joint_action_chunk_ack
ros2 control list_controllers
```

以下条件不会再由 gateway 自动 HOLD；监控者发现后必须立即 DISARM：

- `joint_tracking_error_rad > 0.05`（低速验证阶段）；
- `state_fresh/robot_ready/controller_ready` 任一变 false；
- active plan 中 `applied_feedback_fresh` 变 false；
- ACK rejected；
- session/plan/sequence 不连续；
- controller 不是 active；
- 出现 `communication_constraints_violation`；
- 相机或 qpos 时间戳持续 stale/skew；
- server action 相对当前 qpos 出现异常跳变。

## 12. 日志位置

Robot 主机：

```text
/home/pnp/franka/logs/joint_validation/arm_controller.log
/home/pnp/franka/logs/joint_validation/joint_safety_gateway.log
/home/pnp/franka/logs/joint_validation/gripper.log
/home/pnp/franka/logs/joint_validation/cameras.log
/home/pnp/franka/logs/joint_validation/cameras_after_stack.log
```

Policy 主机：

```text
/m2v_intern/genghaotian/FastWAM/franka_fastwam_dump
FastWAM server 终端日志
ws_tcp_tunnel server 终端日志
```

## 13. 常见错误定位

| 现象 | 含义 | 处理 |
|---|---|---|
| `Socket closed` | 8081 隧道或远端 target 未连通 | 检查 8081 → KML → 15645 → 15174 |
| protocol `server=2, client=None` | client/server checkout 不一致 | 同步协议版本 |
| `no camera1 anchor` | camera1 尚未发布/订阅 QoS 不匹配 | 检查 camera topic |
| `no camera2 sample...` | 两路相机时间戳不匹配 | 检查频率和 timestamp |
| `qpos skew exceeds limit` | joint state 停止或时间戳异常 | 检查 controller 和 topic hz |
| `TimedAction missing server_send_timestamp` | server 没返回当前协议字段 | 更新 server |
| `communication_constraints_violation` | 1 kHz FCI deadline 失败 | Desk Acknowledge；检查 RT/CPU/网卡/启动负载 |
| tracking error 持续增大 | 目标与实测偏差过大；gateway 不会自动 HOLD | 人工 DISARM；禁止重试；检查 retiming/chunk 连续性 |
| `Publisher 1, Subscription 0` | gateway 未启动或 overlay 错误 | 检查 gateway/build/source |

## 14. 2026-07-25 事件结论

- 先 `--execute --no-cam`，controller 稳定后再启动相机，可以通过启动阶段；
- 相机启动 wrapper 自身退出是正常的，只要 camera publisher 仍存在；
- topic 拓扑、controller active 和 ARM 成功不能替代 execution ACK；
- 当前 client 只按本地 30 Hz queue 推进 bookkeeping；
- gateway retiming 后的物理计划可能明显长于本地 0.53 秒 chunk；
- active plan 被新 chunk 替换可能造成跨 chunk 不连续；
- gateway 最终因 `0.250972 rad` tracking error 进入 HOLD。

因此当前下一步不是再次 ARM，而是实现第 8 节的顺序执行协议。


运行顺序：
 bash /home/pnp/franka/stop_all.sh --keep-haply-manager

 