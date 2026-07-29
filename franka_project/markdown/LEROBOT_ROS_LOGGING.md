# LeRobot / ROS / Franka 分文件运行日志

本工具用于记录一次 LeRobot 真机运行中从 Client action chunk 到 Gateway、Controller
feedback 的完整 ROS 侧链路。每次启动都会创建一个由本地启动时间构成的独立目录，
不会和上一次运行混写。

## 1. 启动与停止

建议在一个独立终端运行：

```bash
cd /home/pnp/Projects/lerobot

franka_project/scripts/record_lerobot_ros_logs.sh
```

脚本会自动 source ROS Jazzy、LeRobot interface、Franka 和 Haply overlay。可以先启动 recorder
再启动 Client/Gateway；subscription 会预先建立，不会因为 publisher 尚未出现而退出。

按 `Ctrl+C` 停止并写回结束时间、停止原因和每类消息数量。也可以限定时长：

```bash
franka_project/scripts/record_lerobot_ros_logs.sh \
  --duration-s 600 \
  --label cup_move
```

默认输出根目录：

```text
~/franka/logs/lerobot_runs/
```

每次运行生成例如：

```text
~/franka/logs/lerobot_runs/20260722_184910_240_cup_move/
```

`~/franka/logs/lerobot_runs/latest` 会指向最新一次目录。可用
`--output-root=/other/path` 修改根目录。

## 2. 文件说明

```text
00_events.log
01_manifest.json
02_runtime_snapshot_start.log
03_runtime_snapshot_end.log  # 仅在 --snapshot-at-end 时生成
10_action_chunk.jsonl
11_action_chunk_summary.jsonl
12_action_chunk_ack.jsonl
13_safety_gateway_status.jsonl
14_safety_command_feedback_applied.jsonl
15_safety_command_feedback_other.jsonl
```

- `01_manifest.json`：开始/结束时间、host、ROS 环境白名单、LeRobot 与 Franka Git
  SHA/dirty 状态、运行期间发现的相关进程 argv、Client/Server 关键 CLI 参数及最终出现值、
  各文件消息数量。
  Recorder 每 5 秒扫描一次相关进程，因此先开 Recorder、后开 Client 也能记录其配置。
- `02_runtime_snapshot_start.log`：ROS node/topic graph、Gateway/Controller/Client ROS 参数、
  controller 状态和核心 topic endpoint/QoS 快照。需要停止时再采一份时可增加
  `--snapshot-at-end`，届时会生成 `03_runtime_snapshot_end.log`。
- `10_action_chunk.jsonl`：`/lerobot/franka/action_chunk` 全字段，包括完整 poses、gripper、
  timesteps、period 和 provenance 时间戳。
- `11_action_chunk_summary.jsonl`：每个 plan 的实际 ROS waypoint 数、首尾 timestep 和名义
  horizon，适合快速查看 Client 实际给 ROS 的 25/30/50 点。
- `12_action_chunk_ack.jsonl`：Gateway validation/replacement ACK，包括 `waypoint_count`、
  result、accepted、replaced 和 detail。
- `13_safety_gateway_status.jsonl`：Gateway 10 Hz 状态、plan ID、accepted/applied/measured
  waypoint、source timestep、tracking error 和 HOLD/FAULT detail。
- `14_safety_command_feedback_applied.jsonl`：严格等价于：

  ```bash
  ros2 topic echo /franka/safety_command_feedback \
    --filter 'm.applied and m.status == 2'
  ```

  `status == 2` 即 `STATUS_APPLIED`。文件保留 commanded/measured joints、sequence、plan、
  waypoint 和 controller receipt/application timestamp。
- `15_safety_command_feedback_other.jsonl`：未通过上述 filter 的 reject/stale/holding 等反馈，
  用于定位安全中断。

JSONL 是“一行一个完整 JSON 对象”。每条记录同时包含 logger wall-clock receive time 和
monotonic receive time；消息自身 header/source 时间也单独保留。

只有几秒的快速检查可增加 `--skip-runtime-snapshot`，避免 ROS graph/parameter 查询占用退出
时间；topic 日志本身不受影响。

## 3. `actions_per_chunk` 与 ROS 实际点数

不存在 `/lerobot/action_per_chunk` 或 `/lerobot/actions_per_chunk` ROS topic。

```text
--actions_per_chunk=50
```

是 Client/Server CLI 与模型合同配置，因此从 `01_manifest.json` 的 Client 进程 argv 和
`client_options` 读取。Server 侧对应值也会保存在该进程的 `selected_options` 中。重复
出现的 CLI 参数会按顺序全部保留，并在 `*_option_last_values` 中额外记录最后一个值。

真正发布给 ROS/Gateway 的数量可能因为 stale prefix 或
`--robot.max_action_chunk_waypoints` 而更少，应查看：

- `11_action_chunk_summary.jsonl` 的 `actual_waypoint_count`；
- `12_action_chunk_ack.jsonl` 的 `waypoint_count`。

`11_action_chunk_summary.jsonl` 还分别记录 timestep/pose/gripper 长度以及三者是否一致。这些
运行数值与配置的 `actions_per_chunk` 必须分开理解。

## 4. 安全和容量边界

- Recorder 不订阅 `/franka/safe_joint_command`，不会改变 Gateway 用于判断 controller
  subscriber 是否存在的安全拓扑。
- 不把两路 camera image 用文本写入日志；使用
  [`record_franka_topics.sh`](./FRANKA_TOPIC_RECORDING.md) 可选择任意 topic 写 MCAP，
  仅选择相机时自动写 MP4。
- applied feedback 执行期间约为 200 Hz，长时间运行前应检查磁盘空间。
- Recorder 只观察 topic，不调用 arm service、不发布 command，也不会改变 Gateway
  `enabled/shadow/armed` 状态。
- Client Python 配置不是 ROS parameter；`ros2 param dump /lerobot_franka_interface` 不能替代
  Client argv/config 日志。

## 5. 快速查看

```bash
RUN_DIR="$HOME/franka/logs/lerobot_runs/latest"

tail -f "$RUN_DIR/11_action_chunk_summary.jsonl"
tail -f "$RUN_DIR/12_action_chunk_ack.jsonl"
tail -f "$RUN_DIR/14_safety_command_feedback_applied.jsonl"
tail -f "$RUN_DIR/15_safety_command_feedback_other.jsonl"
```

若某个 JSONL 文件为空，先查看 `01_manifest.json` 的 `message_counts`，再查看
`02_runtime_snapshot_start.log` 中对应 topic 的 publisher count 和 QoS。
