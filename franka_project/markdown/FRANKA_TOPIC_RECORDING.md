# Franka ROS 2 自定义 topic 录制（MCAP / MP4）

`record_franka_topics.sh` 是一个只订阅、不发布控制命令的独立录制工具。它可以在 LeRobot
client 启动后单独运行：

- 相机与 EEF、qpos、action 等混合数据写入 rosbag2 MCAP；
- 只选择 camera1/camera2 时，自动为每路相机写一个 MP4；
- 既接受常用别名，也接受任意绝对 ROS 2 topic。

## 常用命令

```bash
cd /home/pnp/Projects/lerobot

# 两路视频：输出 recordings/<时间>_mp4/camera1.mp4 和 camera2.mp4
franka_project/scripts/record_franka_topics.sh \
  --topics camera1 camera2

# 相机 + 关节位置 + EEF：自动写 MCAP
franka_project/scripts/record_franka_topics.sh \
  --topics camera1 camera2 current_pos eef \
  --duration-s 120

# 任意自定义 topic；显式指定 MCAP 输出目录
franka_project/scripts/record_franka_topics.sh \
  --format mcap \
  --output /data/franka/run_001 \
  --topics /my/custom/topic /lerobot/franka/action_chunk

# 单路任意 sensor_msgs/msg/Image 保存为 MP4
franka_project/scripts/record_franka_topics.sh \
  --format mp4 \
  --output /data/franka/wrist.mp4 \
  --topics /my/wrist/image_raw
```

按 `Ctrl+C` 会让 MCAP 正常写回 metadata 并关闭 MP4。也可以用 `--duration-s` 自动停止。
默认输出位于 `franka_project/recordings/`，该目录已加入 `.gitignore`。如果用 `--output`
写到仓库内的其他位置，该路径不会被自动忽略，应避免把大文件加入 Git。

## 别名

| 别名 | ROS 2 topic |
|---|---|
| `camera1` | `/camera1/camera1/color/image_raw` |
| `camera2` | `/camera2/camera2/color/image_raw` |
| `current_pos` / `qpos` | `/franka/joint_states` |
| `eef` / `current_pose` | `/franka_robot_state_broadcaster/current_pose` |
| `gripper` | `/gripper/joint_states` |
| `action` | `/lerobot/franka/action_chunk` |
| `action_ack` | `/lerobot/franka/action_chunk_ack` |
| `gateway_status` | `/lerobot/franka/safety_gateway_status` |

运行 `record_franka_topics.sh --list-aliases` 可以查看当前映射。

## 格式规则与依赖

`--format=auto` 是默认值。只有全部 topic 都是内置 camera1/camera2 图像源时使用 MP4；只要
混入 EEF、qpos 或其他 topic 就使用 MCAP。自定义图像 topic 请显式传 `--format=mp4`。

录制器读取的是 ROS 2 总线上的原始流。Client 内部会以 camera1 为时间锚点，从缓存中选择
camera2/EEF/qpos/gripper，并派生发送给 policy 的 10D state；这些“某次请求实际选中的同步
snapshot”当前没有单独 topic，因此不能由旁路 recorder 严格复现。若需要保留精确的
policy observation，需要后续在 Client 内增加 observation tap，而不是从进程私有
`action_queue` 读取。

MCAP 模式调用 ROS Jazzy 的 `ros2 bag record --storage mcap`，输出路径是一个 rosbag2
目录，其中包含 `.mcap` 数据文件与 `metadata.yaml`。默认使用 `zstd_fast`，可用
`--mcap-storage-profile` 调整。特殊 publisher 的 QoS 需要固定时，可传
`--qos-profile-overrides-path /path/to/qos.yaml`；未指定时由 rosbag2 根据已发现的
publisher 自动协商。

MP4 模式要求 topic 类型为 `sensor_msgs/msg/Image`，支持 `rgb8`、`bgr8`、`rgba8`、
`bgra8` 和 `mono8`，并使用系统 `python3-opencv`。默认会分别用每路 ROS image
header timestamp 的中位间隔估算恒定 MP4 帧率（时间戳不可用时回退为 15 FPS）；也可以用
`--fps` 显式覆盖。`--codec` 默认 `mp4v`；两个参数只影响 MP4。MP4 只保留图像帧和恒定
播放帧率，不保留逐帧 ROS timestamp、CameraInfo 或可回放消息；需要这些信息时应选 MCAP。

启动脚本会 source ROS Jazzy、已有的 Franka/Haply overlay（若存在），最后 source
`franka_project/ros2_ws/install/setup.bash`。若刚迁入或修改 ROS package，请先构建该
workspace。
