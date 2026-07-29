# Franka × LeRobot RViz 只读可视化

这个包不修改 `/home/pnp/franka` 的控制端，也不增加任何控制输出；它只旁路观察
LeRobot client → safety gateway → Franka controller 的计划与执行状态，并在 RViz 中显示：

- client 最新发送的 TCP action chunk；
- gateway ACK 对该计划的处理结果；
- gateway 当前 active plan 及 next/applied waypoint；
- Franka 实测 TCP 当前位姿和最近 20 秒轨迹；
- 实测 TCP 到 next waypoint 的误差连线；
- gateway 的 armed/shadow/hold/fault、ready/fresh 和跟踪误差信息；
- client observation 使用的 camera1/base RGB 与 camera2/left-wrist RGB 实时画面。

它只创建订阅者和 `/lerobot/franka/viz/*` 下的可视化发布者，**不会**发布
`/franka/safe_joint_command`，也不会调用 `set_armed` 或其他机器人控制服务。

## 数据流

```text
LeRobot client
  └─ /lerobot/franka/action_chunk ───────────────┐
                                                  │
Safety gateway                                    ├─> franka_action_chunk_visualizer
  ├─ /lerobot/franka/action_chunk_ack ───────────┤       └─ /lerobot/franka/viz/* ─> RViz
  └─ /lerobot/franka/safety_gateway_status ──────┤
                                                  │
Franka controller / state broadcaster             │
  ├─ /franka/safety_command_feedback ────────────┤
  └─ /franka_robot_state_broadcaster/current_pose┘

RealSense camera1 ── /camera1/.../image_raw ──┬─> LeRobot observation camera1/base_0_rgb
                                              └─> RViz Image pane
RealSense camera2 ── /camera2/.../image_raw ──┬─> LeRobot observation camera2/left_wrist_0_rgb
                                              └─> RViz Image pane
```

计划必须用 `(session_id, plan_id)` 联合识别。`plan_id=0` 本身不代表同一个计划，
因为 client 重启后新的 session 也可以从 0 开始。

## 颜色和图形语义

| 颜色/图形 | 含义 |
|---|---|
| 青色计划线 | 已收到 client candidate，尚未收到匹配 ACK |
| 绿色计划线 | ACK 为 `ACCEPTED_FOR_EXECUTION` / gateway 正在 ARMED 执行 |
| 蓝色计划线 | ACK 为 `ACCEPTED_SHADOW`；只验证，不下发机器人 |
| 琥珀色计划线 | `ACCEPTED_PREFLIGHT_ONLY`；只做预检，不执行 |
| 红色计划线 | gateway 拒绝或处于 fault |
| 灰色细线 | active plan 已消费部分或非执行状态 |
| 黄球 | gateway 报告的 next waypoint |
| 紫色方块 | controller/status 确认的 applied waypoint |
| 白色轨迹 | Franka 实测 TCP 历史轨迹 |
| 橙色连线 | 实测 TCP 到 next waypoint 的空间误差 |
| waypoint 青→紫渐变 | gripper 从 open (`0.0`) 到 closed (`1.0`) |

最新 candidate 和当前 active plan 会同时显示。因此，即使一个更新的 candidate 被拒绝，
旧 active plan 仍保留在画面中，不会产生“机器人已经切换到 rejected plan”的错觉。

`/lerobot/franka/viz/planned_path` 在 RViz 的 Path display 中固定为淡青色；ACK 的动态颜色
以 `/lerobot/franka/viz/markers` 中的线和文字为准。

## 构建

首次构建：

```bash
source /opt/ros/jazzy/setup.bash
source /home/pnp/franka/haply_ros/install/local_setup.bash

cd /home/pnp/Projects/lerobot/franka_project/ros2_ws
colcon build --symlink-install --packages-up-to franka_lerobot_rviz
source install/setup.bash
```

`lerobot_franka_interfaces` 与本包位于同一工作空间，`--packages-up-to` 会按依赖顺序构建；
`franka_safety_interfaces` 则由 `/home/pnp/franka/haply_ros` underlay 提供。构建阶段使用
`local_setup.bash`，避免该外部工作空间历史上记录的其他 underlay 把当前目标工作空间提前加载。

## 启动

推荐让 Franka 专用 LeRobot client 管理可视化生命周期。在已 source 本工作空间的 client
启动命令末尾增加顶层参数（不是 `--robot.*` 参数）：

```bash
python -m lerobot_robot_franka_ros.ros2_client ... \
  --visualize_action=true
```

Client 会把当前配置中的 action chunk、EEF、camera1、camera2 topic 和 base frame 传给
launch；client 结束时也会关闭可视化进程。若只需要 marker/path topic 而不打开 RViz，再加：

```bash
--visualization_launch_rviz=false
```

也可以保持 `--visualize_action=false`（默认值），先按原流程启动 Franka ROS、gateway 和
LeRobot client，再在另一个终端独立启动：

```bash
source /opt/ros/jazzy/setup.bash
source /home/pnp/franka/franka_ros2_ws/install/setup.bash
source /home/pnp/franka/haply_ros/install/setup.bash
source /home/pnp/Projects/lerobot/franka_project/ros2_ws/install/setup.bash

ros2 launch franka_lerobot_rviz action_viz.launch.py
```

只运行转换节点、不启动 RViz：

```bash
ros2 launch franka_lerobot_rviz action_viz.launch.py launch_rviz:=false
```

如果实际的实测 TCP 话题不同，可在启动时覆盖：

```bash
ros2 launch franka_lerobot_rviz action_viz.launch.py \
  current_pose_topic:=/your/current_pose_topic
```

也可以覆盖 `action_chunk_topic`、`ack_topic`、`gateway_status_topic`、
`safety_feedback_topic`、`camera1_topic`、`camera2_topic` 和 `fixed_frame`。例如相机话题被重命名时：

```bash
ros2 launch franka_lerobot_rviz action_viz.launch.py \
  camera1_topic:=/your/base/image_raw \
  camera2_topic:=/your/wrist/image_raw
```

两个 Image display 会在 RViz 中创建各自的 dock pane；可以拖动为左右并排或放到 3D 视图下方。

## 输入与输出话题

### 只读输入

| 默认话题 | 类型 | 用途 |
|---|---|---|
| `/lerobot/franka/action_chunk` | `lerobot_franka_interfaces/CartesianActionChunk` | client 发送的未来 TCP 计划 |
| `/lerobot/franka/action_chunk_ack` | `lerobot_franka_interfaces/CartesianActionChunkAck` | candidate 是否通过 gateway |
| `/lerobot/franka/safety_gateway_status` | `lerobot_franka_interfaces/SafetyGatewayStatus` | active plan、消费进度和 gateway 状态 |
| `/franka/safety_command_feedback` | `franka_safety_interfaces/SafetyCommandFeedback` | controller 真正 applied 的 waypoint |
| `/franka_robot_state_broadcaster/current_pose` | `geometry_msgs/PoseStamped` | Franka 实测 TCP 位姿 |
| `/camera1/camera1/color/image_raw` | `sensor_msgs/Image` | RViz 与 client 共用的 camera1/base 原始图像源 |
| `/camera2/camera2/color/image_raw` | `sensor_msgs/Image` | RViz 与 client 共用的 camera2/left-wrist 原始图像源 |

action/status/pose/feedback 使用 best-effort observer QoS，避免这个非关键节点给控制链增加可靠传输反压；
一次性的 ACK 使用 reliable QoS。两路 RViz Image display 使用 best-effort + volatile + keep-last(1)，只取最新帧，
不会让相机等待 RViz 的可靠传输确认。可视化输出使用 reliable + transient-local，RViz 晚启动也能立即拿到完整轨迹场景。

当前映射与 client/server 的 rename map 一致：

```text
camera1 raw → observation.images.camera1 → observation.images.base_0_rgb
camera2 raw → observation.images.camera2 → observation.images.left_wrist_0_rgb
```

RViz 显示的是 client 的两路**原始 ROS 图像输入源**。Client 内部会以 camera1 为时间锚点，选择不晚于
camera1 且时间差不超过 50 ms 的 camera2 帧，再进行 processor/resize 并发送给 server；这个“某次请求最终选中的
精确同步帧对”目前没有独立 ROS topic，所以在不修改 client 的约束下无法由 RViz 严格复现。

### RViz 输出

| 话题 | 类型 | 内容 |
|---|---|---|
| `/lerobot/franka/viz/planned_path` | `nav_msgs/Path` | 最新 client candidate 的完整 TCP 路径 |
| `/lerobot/franka/viz/active_path` | `nav_msgs/Path` | gateway 当前 active plan 的完整 TCP 路径 |
| `/lerobot/franka/viz/planned_poses` | `geometry_msgs/PoseArray` | candidate 全部姿态；RViz 默认关闭以减少遮挡 |
| `/lerobot/franka/viz/actual_path` | `nav_msgs/Path` | 实测 TCP 的有界历史轨迹 |
| `/lerobot/franka/viz/actual_pose` | `geometry_msgs/PoseStamped` | 最新实测 TCP 位姿 |
| `/lerobot/franka/viz/markers` | `visualization_msgs/MarkerArray` | 动态颜色、进度、标签和误差连线 |

每次 marker 消息均为 `DELETEALL + 当前完整场景`，所以短计划替换长计划时不会残留旧 waypoint。

## 快速检查

确认可视化节点没有控制输出：

```bash
ros2 node info /franka_action_chunk_visualizer
```

`Publishers` 应只有 `/lerobot/franka/viz/*`（以及 ROS 标准日志/参数相关端点），
`Subscriptions` 应只有上表的五个输入；不应出现 `/franka/safe_joint_command` publisher。

确认数据正在更新：

```bash
ros2 topic hz /lerobot/franka/viz/planned_path
ros2 topic echo /lerobot/franka/viz/markers --once --no-arr
ros2 topic echo /lerobot/franka/safety_gateway_status --once
ros2 topic hz /camera1/camera1/color/image_raw
ros2 topic hz /camera2/camera2/color/image_raw
```

若 Image pane 显示 `No Image received`，先确认两路 RealSense topic 已启动；RViz 配置本身不会启动相机。

若能看到路径但看不到机械臂模型，检查：

```bash
ros2 topic info /robot_description -v
ros2 topic info /joint_states -v
ros2 topic list | grep '^/tf'
```

RobotModel 依赖原 Franka bringup 提供 `/robot_description`、`/joint_states` 和 TF；这些未启动时，
`base` 坐标系中的 action 路径仍可显示，但机械臂模型不会出现或不会更新。

## 能力边界

`CartesianActionChunk.poses` 是 `header.frame_id`（通常为 `base`）中的 Franka `O_T_EE`
TCP 目标。因此这个包能忠实显示未来 TCP 的位置、方向、gripper 值、gateway 接纳状态和实际跟踪效果。

它不会伪造“未来整条机械臂”的姿态。要显示每个 action 对应的整臂 ghost/animation，必须为每个 TCP pose
重新做 IK 并发布 MoveIt `DisplayTrajectory`；旁路 IK 的 seed、碰撞场景或解分支可能与 safety gateway
内部预检不同，容易产生看起来合理但实际不会执行的机械臂姿态。当前版本刻意只展示可由现有消息确定的事实。

另外，`valid_until` 只是 gateway 接受一个新 plan 的传输截止时间。计划一旦被接受，不能因为这个短 TTL
过期就在 RViz 中消失；active plan 的生命周期和进度以 `SafetyGatewayStatus` / controller feedback 为准。
