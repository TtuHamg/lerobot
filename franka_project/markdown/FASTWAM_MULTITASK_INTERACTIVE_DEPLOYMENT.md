# FastWAM 多任务交互式部署设计

## 目标与边界

本设计让一个已经加载到 GPU 的 FastWAM 多任务模型在以下三个训练任务之间切换：

- `pick up the cup`
- `pick up the chips`
- `pick up the tape`

Checkpoint、Wan VAE、normalizer 和模型参数在 server 生命周期内只加载一次。Server 启动时校验并预加载每个任务的 text embedding cache；每个 observation 携带规范化 task 和单调递增的 `task_generation`，server 按该 task 选择本次推理使用的 context。

LeRobot 只负责 policy transport、任务代际隔离和 ROS action-chunk 边界。本仓库不实现 Franka Cartesian safety gateway、IK 或 controller。任务中止复用机器人主机已有的：

```text
/franka_cartesian_safety_gateway/set_armed
```

`set_armed(false)` 必须清空外部 gateway 的 active plan 并进入 HOLD；硬件 E-stop、deadman 和现场安全系统仍是独立且更高优先级的边界。

## Server 数据流

Server 从冻结 dataset contract 指向的 source manifest 中提取实际参与训练的 task allowlist，再按 FastWAM prompt template 计算对应 cache 文件名。每个 cache 都必须通过 dtype、shape、mask 和 finite 检查，否则 server 在模型推理前 fail closed。

```text
checkpoint contract
  -> allowed task instructions
  -> task -> cached (context, context_mask)
  -> one resident FastWAM model

TimedObservation(task, task_generation)
  -> exact allowlist lookup
  -> local context selection
  -> infer_action / infer_joint
  -> Actions(task_generation)
```

推理路径不得修改共享的 `runtime.context`。每次推理只持有本地的 `(context, context_mask)` 引用，从而避免任务切换和并发 RPC 修改全局文本条件。

## Client 状态机

交互式 Franka client 使用独立键盘线程；控制循环不直接阻塞在 `input()`。支持：

```text
1..N  选择 server 握手返回的 allowlist task
s     中止当前任务并请求 gateway disarm/HOLD
a     显式 arm；成功后恢复 observation
q     disarm 后退出
h     重印帮助与状态
```

状态迁移：

```text
HOLD/IDLE --select task--> HOLD/READY
HOLD/READY --arm ACK--> RUNNING(task, generation)
RUNNING --stop--> HOLDING --disarm ACK--> HOLD/IDLE
RUNNING --select another task--> HOLDING -> HOLD/READY
```

选择任务本身绝不自动 arm。新任务必须经过独立的 `a` 操作，且只有 gateway arm service 成功后 client 才恢复发送 observation。

## Task generation 与旧结果处理

每次选择、重新指派或中止任务都会增加 `task_generation`。Generation 随 observation 和 action delivery 传播。Client 仅向本地 action queue 和 ROS publisher 提交当前 generation 的结果。

旧 generation 的 action delivery 会被安全丢弃，但仍发送 transport ACK，使 server 不会永久重发一个已被主动废弃的 chunk。该 ACK 只表示传输结果已被 client 处理，不表示 gateway 接受、controller 应用或机器人执行。

任务切换顺序固定为：

```text
暂停 observation
  -> generation + 1
  -> 清空 client action queue 与 pending observation
  -> 记录新任务（仍为 paused）
  -> gateway set_armed(false)
  -> 确认 service 成功
  -> operator 显式 set_armed(true)
  -> arm 成功
  -> must-go observation
```

GPU inference 本身不要求可抢占。旧 inference 可以自然结束，但其 generation 已经过期，结果不会进入 ROS action-chunk publisher。

## Failure semantics

- 未知或任意自由文本 task：server 拒绝，不回退到默认 embedding。
- task cache 缺失或损坏：server 启动失败，不加载一个错误的默认 context。
- gateway arm/disarm service 失败或超时：client 保持 observation paused，不自动继续。
- 旧 generation action：client 丢弃并仅完成 transport ACK。
- gateway 未运行：交互式执行模式不能 arm；可继续使用既有 shadow/dry-run 路径。
- 键盘线程退出：不会隐式 arm；主程序退出前尝试 disarm。

## 启动命令

Server 机器使用 `franka-fastwam-serve` 环境；这里的 task 不在命令行固定，而是在每个 observation 中由 client 发送：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot
conda activate franka-fastwam-serve
CUDA_VISIBLE_DEVICES=1 python franka_project/scripts/serve_franka_pi0_async.py \
  --host=0.0.0.0 \
  --port=15173 \
  --fps=30 \
  --inference_latency=0 \
  --obs_queue_timeout=1 \
  --observation_similarity_mode=none \
  --policy_type=fastwam \
  --pretrained_name_or_path=/m2v_intern/tujiahang/Projects/FastWAM/franka_project/runs/franka_eef_mix3_0804_joint_lora_after_warmup_pretrained_xt/lora_after_warmup_bs256_lr1e-4_r32a64/checkpoints/weights/step_005000.pt \
  --actions_per_chunk=32 \
  --policy_device=cuda
```

机器人机器先启动仓库外已有的 Franka Cartesian safety gateway，并确认服务存在：

```bash
ros2 service type /franka_cartesian_safety_gateway/set_armed
```

然后启动交互式 client；将 `<SERVER_IP>` 替换为 server 地址：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot
source /opt/ros/jazzy/setup.bash
source franka_project/ros2_ws/install/setup.bash
conda activate franka-fastwam-serve
export PYTHONPATH="$PWD/src:$PWD/franka_project/ros_lerobot/src:${PYTHONPATH:-}"
python -m lerobot_robot_franka_ros.ros2_client \
  --server_address=<SERVER_IP>:15173 \
  --robot.type=franka_ros \
  --robot.id=franka_fastwam_multitask \
  --robot.dry_run=false \
  --robot.ros2_interface_only=true \
  --robot.base_frame=base \
  --robot.gripper_open_position=0.0 \
  --robot.gripper_closed_position=0.8 \
  --robot.gripper_max_skew_s=0.01 \
  --robot.camera2_max_skew_s=0.1 \
  --robot.eef_max_skew_s=0.05 \
  --robot.max_action_chunk_waypoints=32 \
  --policy_type=fastwam \
  --pretrained_name_or_path=server-owned \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=32 \
  --action_offset=1 \
  --fps=30 \
  --chunk_size_threshold=0.0 \
  --aggregate_fn_name=latest_only \
  --enable_pending_observation=true \
  --pending_observation_timeout_s=30 \
  --rename_map={} \
  --interactive_task_control=true \
  --gateway_arm_timeout_s=5
```

Client 握手后会保持 HOLD 并打印编号任务。操作顺序是 `1/2/3` 选择任务，再按 `a` 开始；切换时先按 `s`，或直接选择另一个编号（仍会先 disarm，之后必须再次按 `a`）。

## 验收标准

1. 多任务 checkpoint contract 能解析三个 task，且三个 cache 均被加载一次。
2. 同一模型实例对不同 observation task 选择不同 context。
3. 握手向 client 返回 server allowlist。
4. 任务 generation 在 observation、action response 和 ACK 中一致。
5. 切换任务后旧 action chunk 不进入 local queue 或 ROS publisher。
6. `s` 使 client 暂停并请求 disarm；`a` 仅在 arm service 成功后恢复。
7. 单任务 checkpoint 和非交互式 stock client 保持兼容。
