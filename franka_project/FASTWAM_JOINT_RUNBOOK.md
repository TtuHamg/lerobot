# FastWAM ↔ Franka 关节空间异步交互 Runbook

> 本文是早期接口说明，端口和执行假设已过期。当前完整、安全运行顺序请阅读
> [`FASTWAM_FRANKA_JOINT_OPERATIONS_CN.md`](./FASTWAM_FRANKA_JOINT_OPERATIONS_CN.md)。
> 在 execution ACK/计划完成闭环修复前，禁止按本文旧命令进行真机 ARM。

本文档记录 FastWAM（关节空间策略）通过 lerobot async-inference 框架控制 Franka（FR3，7 关节 + gripper）
的端到端启动流程。它是 so-arm101 版（`FastWAM/scripts/fastwam_async_server.py` +
`--robot.type=so101_follower`）的 Franka 关节空间对应版本。

数据链路与 so101 一致，只是 robot 类型换成隔离的 `franka_ros_joint`，动作走关节空间 `JointActionChunk`：

```
Franka observation (ROS2: camera1/camera2 + /franka/joint_states + /gripper/joint_states)
  -> FrankaJointRos2RobotClient (stock RobotClient + 完整 chunk 发布 hook)
  -> local gRPC 127.0.0.1:8080 -> WebSocket tunnel -> KML gateway
  -> franka_fastwam_async_server.py (FastWAM 关节空间扩散推理)
  -> list[TimedAction]（8 维：7 关节 + gripper，绝对关节角）原路返回
  -> ROS2 topic /lerobot/franka/joint_action_chunk 发布（positions[K*7] + gripper[K]）
```

关键约束：
- `--fps=30`（Franka 训练与控制帧率，必须与数据和 checkpoint 一致）。
- `--aggregate_fn_name=latest_only`（关节空间完整 chunk 发布语义要求）。
- `--robot.dry_run=false --robot.ros2_interface_only=true`（非执行 ROS2 接口，不含 controller/IK/actuation）。
- 图像布局与 so101 相同：RobotWin 384×320（camera1→上 256×320，camera2→左下 128×160，右下黑）。

---

## 0. 需要对齐训练的 server 资产（严格匹配）

- `--run-config`  训练保存的 `config.yaml`（frank3_finetune，`action_state_merger.*_target_dim=16`）。
- `--checkpoint`  权重 `step_NNNNNN.pt`。
- `--stats`       `dataset_stats.json`（含 `global_mean/global_std`，key=`default`）。
- `--text-cache`  预计算的 T5 文本 embedding 目录（见步骤 1）。
- `--motor-keys`  默认 8 个：`fr3_joint1.pos … fr3_joint7.pos gripper.pos`（顺序 = convert_frank3 的 state 拼接）。
- `--cam-keys`    默认 `camera1 camera2`（Franka 机器人 observation 的相机 key）。

---

## 1. 预计算文本 embedding（KML，fastwam 环境，仅需一次）

```bash
conda activate fastwam
cd /m2v_intern/genghaotian/FastWAM
python scripts/precompute_text_embeds.py \
    --prompt "Stack three paper cups." \
    --out-dir data/text_embeds_cache/frank3
```
prompt 用数据集 `meta/tasks.jsonl` 里的原始 `task` 文本；server 会用训练时的 DEFAULT_PROMPT_TEMPLATE
包装后再哈希查缓存。

## 2. 启动 FastWAM Franka server（KML，GPU 主机）

```bash
conda activate fastwam
cd /m2v_intern/genghaotian/FastWAM
python scripts/franka_fastwam_async_server.py \
    --run-config runs/frank3_finetune/<RUN>/config.yaml \
    --checkpoint runs/frank3_finetune/<RUN>/checkpoints/weights/step_NNNNNN.pt \
    --stats      runs/frank3_finetune/<RUN>/dataset_stats.json \
    --text-cache data/text_embeds_cache/frank3 \
    --prompt     "Stack three paper cups." \
    --port 15173 \
    --action-horizon 32 --actions-per-chunk 16 --fps 30 \
    --cam-keys camera1 camera2 \
    --diagnose-n 3
```
`--diagnose-n 3` 会打印前 3 次推理的 proprio(8→pad16)、图像 [-1,1] 统计、raw action(8) 用于核对维度。

## 3. WebSocket 隧道 server（KML）

```bash
python /m2v_intern/tujiahang/Projects/lerobot/tools/ws_tcp_tunnel.py server \
    --listen-port 15174 --target-port 15173
```

---

## 4. ROS2 消息包编译（Franka 机器人主机，仅需一次/改消息后）

```bash
source /opt/ros/jazzy/setup.bash
cd /m2v_intern/genghaotian/lerobot/franka_project/ros2_ws
colcon build --packages-select lerobot_franka_interfaces
source install/setup.bash
python -c "from lerobot_franka_interfaces.msg import JointActionChunk; print('msg OK')"
```

## 5. 安装 Franka 插件（Franka 机器人主机，py3.12 + lerobot 环境）

```bash
uv pip install --no-deps -e /m2v_intern/genghaotian/lerobot/franka_project/ros_lerobot
```

## 6. WebSocket 隧道 client（Franka 机器人主机）

```bash
python /m2v_intern/tujiahang/Projects/lerobot/tools/ws_tcp_tunnel.py client \
    --listen-port 8080 --ws-url ws://kml-dtmachine-XXXXX.../ws
```

## 7. 启动 Franka 关节空间 async client（Franka 机器人主机）

先 `source install/setup.bash`（让 rclpy + lerobot_franka_interfaces 可用），再：

```bash
python -m lerobot_robot_franka_ros.joint_ros2_client \
    --robot.type=franka_ros_joint \
    --robot.dry_run=false \
    --robot.ros2_interface_only=true \
    --server_address=127.0.0.1:8080 \
    --policy_type=act --pretrained_name_or_path=/dummy \
    --task="Stack three paper cups." \
    --actions_per_chunk=16 --chunk_size_threshold=0.6 \
    --aggregate_fn_name=latest_only --fps=30
```

核对动作输出：
```bash
ros2 topic echo /lerobot/franka/joint_action_chunk
# positions: 长度 K*7 的绝对关节角（fr3_joint1..7，行主序）
# gripper:   长度 K 的原始 gripper 关节目标
```

---

## Dry-run 快速自检（无需 ROS / GPU，机器人侧联调前验证接线）

```bash
python -m lerobot_robot_franka_ros.joint_ros2_client \
    --robot.type=franka_ros_joint --robot.dry_run=true \
    --robot.fixture_path=/path/joint_observation.npz \
    --robot.action_log_path=/tmp/joint_actions.jsonl \
    --server_address=127.0.0.1:8080 \
    --policy_type=act --pretrained_name_or_path=/dummy \
    --aggregate_fn_name=latest_only --fps=30
```
dry-run fixture 是含 `state`(8, float32) + `camera1`/`camera2`(480,640,3, uint8) 的 npz。

---

## 单元测试（py3.12 + `uv sync --extra dev` 环境）

```bash
cd /m2v_intern/genghaotian/lerobot/franka_project
uv run --project /m2v_intern/genghaotian/lerobot pytest \
    tests/test_franka_joint_ros_dry_run.py \
    tests/test_franka_joint_ros2_backend.py -svv
```

> 说明：仓库 `lerobot` 需要 Python 3.12+（PEP 695 语法）。若当前机器只有 3.10 / 无 uv，可只验证纯契约层
> （`joint_contract` / `joint_ros2_contract`）与 `py_compile` 语法检查。

---

## 与 Cartesian/PI0 路径的关系

Franka 关节空间通道（`franka_ros_joint` / `joint_*` 文件 / `JointActionChunk`）与既有 Cartesian/PI0
路径（`franka_ros` / `Cartesian*` / `CartesianActionChunk`）完全并存、互不影响。切换只需改
`--robot.type` 与对应的 client 入口（`joint_ros2_client` vs `ros2_client`）和 server 脚本。
