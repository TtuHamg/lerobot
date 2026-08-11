# PI0 / FastWAM 异步部署启动指南

本文给出当前 Franka ROS2 client 与 policy server 的启动命令，覆盖三种 artifact：

1. PI0 schema 1，全参数 checkpoint；
2. PI0 schema 2，可配置训练组件 checkpoint；
3. FastWAM deployment schema 1 checkpoint。

当前代码以提交 `0d11d0352fb03d1e360d1a7732902e0f07663c52` 的功能树为基线，并使用
async protocol v2。PI0 schema 只作为 checkpoint provenance 保存，两种 schema 使用同一加载
路径；client 和 server 都不需要 `--schema` 参数。

如果要把一条 FastWAM `grab the paper cup.` 训练 observation 从 client 精确发送一次、保存
一个 response 后退出，见 [`FASTWAM_FIXTURE.md`](./FASTWAM_FIXTURE.md)。该流程固定使用
dry-run fixture，不执行 server 返回的动作。

## 1. 支持矩阵

| Policy artifact | manifest header | server `--policy_type` | FPS | chunk | checkpoint 参数 |
|---|---|---|---:|---:|---|
| PI0 chips | `schema_version=1`, `checkpoint_type=franka_pi0_full_parameter_eef` | `pi0` | 30 | 50 | `pretrained_model/` 目录 |
| PI0 move-cups | `schema_version=2`, `checkpoint_type=franka_pi0_configurable_parameter_eef` | `pi0` | 30 | 50 | `pretrained_model/` 目录 |
| FastWAM move-cups | `schema_version=1`, `policy_type=fastwam` | `fastwam` | 30 | 32 | `step_019650.pt` 文件 |

PI0 server 不再根据外层 `schema_version/checkpoint_type` 拒绝 checkpoint，也不审计
`parameter_training` 的组件计数、冻结状态或 signature。schema 1 和 schema 2 都直接进入原有
canonical PI0 权重加载流程。启动时不会计算 PI0 大权重文件的完整 SHA-256，也不会把 manifest
中的 `training_graph` 与运行时报告做完整字典比较；必要文件、小型 metadata hash、processor、
stats、geometry 和 strict tensor key/shape/load 检查仍保留。

注意三个不同层级不要混淆：PI0 schema 2 checkpoint 的外层
`schema_version` 是 `2`，其中 `parameter_training.schema_version` 是 `1`，
`training_graph.schema_version` 是 `2`。不要对 manifest 全局替换 `schema_version`。

FastWAM 的 deployment schema 1 是另一种 manifest，不等同于 PI0 schema 1。
它同样只作为 provenance 保存；server 不因 FastWAM 外层 `schema_version` 不同而拒绝加载，
但仍核对 `policy_type`、run、checkpoint step 和 task。

## 2. 启动前共同检查

### 2.1 确认两端都是 protocol v2

server 和 client 各自使用实际启动它们的 Python 执行：

```bash
python -c 'import lerobot; from lerobot.async_inference.helpers import ASYNC_INFERENCE_PROTOCOL_VERSION as v; print(lerobot.__file__); print(v)'
```

两端最后一行都必须是：

```text
2
```

client 侧再检查 ROS 插件来自预期 checkout：

```bash
python -c 'import lerobot_robot_franka_ros as p; print(p.__file__)'
```

更新代码后必须重启旧 server/client 进程，并在 client checkout 重新安装 editable 插件：

```bash
cd /home/pnp/Projects/lerobot
python -m pip install --no-deps -e franka_project/ros_lerobot
```

只比较 `git rev-parse HEAD` 不足以证明运行时代码一致：dirty worktree、旧 editable install 或
另一个 Python 环境都可能让进程加载旧模块。若看到
`server=2, client=None`，说明 client 实际仍在发送旧协议。

protocol v2 会通过 `PolicySetupAck` 双向核对 protocol、FPS、policy type 和 chunk size。

### 2.2 网络地址

本文按当前 tunnel 拓扑书写：

```text
server process: 127.0.0.1:15173
robot client:   127.0.0.1:8080 -> WebSocket tunnel -> server:15173
```

如果 client 与 server 同机且不经过 tunnel，把 client 的
`--server_address=127.0.0.1:8080` 改成 `--server_address=127.0.0.1:15173`。

WebSocket/KML tunnel 的启动和 SSO 说明见
[`ASYNC_CLIENT_SERVER_RUNBOOK.md`](./ASYNC_CLIENT_SERVER_RUNBOOK.md)。

### 2.3 ROS observation 契约

```text
base frame:       base
camera1:          /camera1/camera1/color/image_raw
camera2:          /camera2/camera2/color/image_raw
EEF pose:         /franka_robot_state_broadcaster/current_pose
arm qpos:         /franka/joint_states
gripper:          /gripper/joint_states
gripper joint:    robotiq_85_left_knuckle_joint
isolated output:  /lerobot/franka/action_chunk
```

EEF `PoseStamped.header.frame_id` 必须已经是 `base`；当前 client 不做 TF。两份 PI0 数据转换
配置都使用 gripper raw endpoints `open=0.0, closed=0.8`，因此下面的 PI0 client 也保持该标定。
FastWAM 还会强制检查 topic、joint name、gripper endpoints 和时间 skew。

## 3. PI0 schema 1：chips full-parameter checkpoint

固定契约：

```text
task:  pick up the potato chip
fps:   30
chunk: 50
```

### 3.1 Server

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

export PI0_SCHEMA1_CHECKPOINT=/m2v_intern/tujiahang/Projects/lerobot/franka_project/experiments/pi0_full_eef_chips_obs30_act30_k50_v1/20260717T041129Z_pi0_full_eef_chips_f0_obs30_act30_k50_46pass_255fea1e/checkpoints/step-036150/pretrained_model

CUDA_VISIBLE_DEVICES=0 python franka_project/scripts/serve_franka_pi0_async.py \
  --host=127.0.0.1 \
  --port=15173 \
  --fps=30 \
  --inference_latency=0 \
  --obs_queue_timeout=1 \
  --observation_similarity_mode=none \
  --policy_type=pi0 \
  --pretrained_name_or_path="$PI0_SCHEMA1_CHECKPOINT" \
  --actions_per_chunk=50 \
  --policy_device=cuda
```

### 3.2 Client

```bash
cd /home/pnp/Projects/lerobot

python -m lerobot_robot_franka_ros.ros2_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=franka_ros \
  --robot.id=franka_pi0_schema1 \
  --robot.dry_run=false \
  --robot.ros2_interface_only=true \
  --robot.base_frame=base \
  --robot.gripper_open_position=0.0 \
  --robot.gripper_closed_position=0.4 \
  --task='pick up the potato chip' \
  --policy_type=pi0 \
  --pretrained_name_or_path=server-owned \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --action_offset=1 \
  --fps=30 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only \
  --enable_pending_observation=true \
  --pending_observation_timeout_s=10 \
  --rename_map='{"observation.images.camera1":"observation.images.base_0_rgb","observation.images.camera2":"observation.images.left_wrist_0_rgb"}'
```

## 4. PI0 schema 2：move-cups configurable-parameter checkpoint

固定契约：

```text
task:  move the paper cup from one end of the can to the other.
fps:   30
chunk: 50
```

schema 2 与 schema 1 使用相同的 PI0 CLI；差别只在 checkpoint manifest 记录的训练 provenance。

### 4.1 Server

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

export PI0_SCHEMA2_CHECKPOINT=/m2v_intern/tujiahang/Projects/lerobot/franka_project/experiments/pi0_ActTrans_PaliGemma_eef_grab_cups_obs30_act30_k50_v1/20260723T030203Z_pi0_ActTrans_PaliGemma_eef_grab_cups_obs30_act30_k50_5be94403/checkpoints/step-014100/pretrained_model

CUDA_VISIBLE_DEVICES=0 python franka_project/scripts/serve_franka_pi0_async.py \
  --host=127.0.0.1 \
  --port=15173 \
  --fps=30 \
  --inference_latency=0 \
  --obs_queue_timeout=1 \
  --observation_similarity_mode=none \
  --policy_type=pi0 \
  --pretrained_name_or_path="$PI0_SCHEMA2_CHECKPOINT" \
  --actions_per_chunk=50 \
  --policy_device=cuda
```

### 4.2 Client

```bash
cd /home/pnp/Projects/lerobot

python -m lerobot_robot_franka_ros.ros2_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=franka_ros \
  --robot.id=franka_pi0_schema2 \
  --robot.dry_run=false \
  --robot.ros2_interface_only=true \
  --robot.base_frame=base \
  --robot.gripper_open_position=0.0 \
  --robot.gripper_closed_position=0.4 \
  --task='grab the paper cup.' \
  --policy_type=pi0 \
  --pretrained_name_or_path=server-owned \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --robot.max_action_chunk_waypoints=25 \
  --action_offset=1 \
  --fps=30 \
  --chunk_size_threshold=0.0 \
  --aggregate_fn_name=latest_only \
  --enable_pending_observation=true \
  --pending_observation_timeout_s=10 \
  --rename_map='{"observation.images.camera1":"observation.images.base_0_rgb","observation.images.camera2":"observation.images.left_wrist_0_rgb"}'
```

```bash
python -m lerobot_robot_franka_ros.ros2_client   --server_address=127.0.0.1:8080   --robot.type=franka_ros   --robot.id=franka_pi0_schema2   --robot.dry_run=false   --robot.ros2_interface_only=true   --robot.base_frame=base   --robot.gripper_open_position=0.0   --robot.gripper_closed_position=0.4   --task='grab the paper cup.'   --policy_type=pi0   --pretrained_name_or_path=server-owned   --policy_device=cpu   --client_device=cpu   --actions_per_chunk=50   --robot.max_action_chunk_waypoints=50   --action_offset=1   --fps=30   --chunk_size_threshold=0.0   --aggregate_fn_name=latest_only   --enable_pending_observation=true   --pending_observation_timeout_s=30   --rename_map='{"observation.images.camera1":"observation.images.base_0_rgb","observation.images.camera2":"observation.images.left_wrist_0_rgb"}' --observation_trigger_mode=post_action_delay --post_action_observation_delay_s=3
```


主 manifest 应保持：

```text
schema_version=2
checkpoint_type=franka_pi0_configurable_parameter_eef
parameter_training.schema_version=1
training_graph.schema_version=2
```

不要再把外层 manifest 临时伪装为 schema 1/full。

## 5. FastWAM deployment schema 1：move-cups

固定契约：

```text
task:       move the paper cup from one end of the can to the other.
fps:        30
chunk:      32
rename_map: {}
```

### 5.1 当前运行环境限制

FastWAM 的运行参数直接从 checkpoint 同级的 `fastwam_runtime.resolved.yaml`、
`dataset_contract.json`、`dataset_stats.json` 和对应的 text context 读取。服务端不再维护
deployment manifest 白名单，也不会校验 artifact/source tree 的 SHA-256；所以新训练出的
`RUN/checkpoints/weights/step_<N>.pt` 可以直接指定。它仍会检查这些文件是否存在，以及当前
server 的 fps、chunk size、state/action 维度和双相机排列是否和训练配置一致。

但是当前两个现成环境都不能完整启动真模型：

- `lerobot` 环境是 Python 3.12，但缺少 Hydra、OmegaConf、boto3 和 FastWAM runtime；
- `fastwam` 环境依赖较全，但仍是 Python 3.10，且缺少当前 LeRobot/draccus。

当前实现是在同一个 server 进程内加载 FastWAM，并没有独立 worker。因此执行下面的 server
命令前，必须先准备一个同时满足当前 LeRobot 和 FastWAM 依赖的 Python 3.12 serving 环境。
真实 12 GB FastWAM 权重尚未完成 CUDA load smoke，不能把 artifact hash 校验通过等同于真模型
已跑通。

### 5.2 可选：joint video + action 推理

默认只运行 action-only 推理。加入下面的参数后，server 会调用 `infer_joint`，按训练配置的
9 个稀疏视频帧（offset `0,4,...,32`，7.5 FPS）同时生成未来视频和 action；RPC 始终只把
action 发给机器人。

```bash
--fastwam_joint_video_inference=true
```

需要保存生成视频时，再指定输出目录（目录不存在会自动创建）：

```bash
--fastwam_joint_video_inference=true \
--fastwam_joint_video_output_dir=/m2v_intern/tujiahang/Projects/FastWAM/franka_project/runs/joint_videos
```

每次推理会在该目录写入一个 `fastwam_joint_t<timestep>_<timestamp>.mp4`，帧率为 7.5。保存是同步
I/O；joint 推理和视频编码都会显著增加耗时与显存占用，不适合维持 30 Hz 的实时控制，建议仅在
离线或低速实验中开启。

`franka_eef_mix3_0804` 的历史 gripper metadata 与视频中的物理开闭方向相反。部署该
`step_005000.pt` 时必须显式加入：

```bash
--fastwam_state_gripper_encoding=closed_0_1
--fastwam_action_gripper_encoding=closed_0_1
```

`state` 参数控制 FastWAM 8D proprio 中 pseudo-finger 的物理含义；`action` 参数控制模型
第 7 维输出的物理含义。两者必须分别验证。其他 checkpoint 保持默认 `open_0_1`，除非其
golden fixture 分别证明 state 或 action 实际使用 `closed_0_1`。

环境准备完成后先执行：

```bash
PYTHONPATH=/m2v_intern/tujiahang/Projects/FastWAM/src \
  python -c 'import torch, hydra, omegaconf, boto3; import fastwam; print(torch.__version__)'
```

### 5.2 Server

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

export FASTWAM_CHECKPOINT=/m2v_intern/tujiahang/Projects/FastWAM/franka_project/runs/franka_eef_grab_cups/checkpoints/weights/step_014150.pt
export DIFFSYNTH_MODEL_BASE_PATH=/m2v_intern/tujiahang/Projects/FastWAM/checkpoints

CUDA_VISIBLE_DEVICES=0 python franka_project/scripts/serve_franka_pi0_async.py \
  --host=127.0.0.1 \
  --port=15173 \
  --fps=30 \
  --inference_latency=0 \
  --obs_queue_timeout=1 \
  --observation_similarity_mode=none \
  --policy_type=fastwam \
  --pretrained_name_or_path="$FASTWAM_CHECKPOINT" \
  --actions_per_chunk=32 \
  --policy_device=cuda
```
```
CUDA_VISIBLE_DEVICES=0 python franka_project/scripts/serve_franka_pi0_async.py   --host=127.0.0.1   --port=15173   --fps=30   --inference_latency=0   --obs_queue_timeout=1   --observation_similarity_mode=none   --policy_type=fastwam   --pretrained_name_or_path="$FASTWAM_CHECKPOINT"   --actions_per_chunk=32   --policy_device=cuda --fastwam_joint_video_inference=true --fastwam_joint_video_output_dir=/m2v_intern/tujiahang/Projects/FastWAM/franka_project/runs/depoly_joint_videos
```

### 5.3 Client

```bash
cd /home/pnp/Projects/lerobot

python -m lerobot_robot_franka_ros.ros2_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=franka_ros \
  --robot.id=franka_fastwam_schema1 \
  --robot.dry_run=false \
  --robot.ros2_interface_only=true \
  --robot.base_frame=base \
  --robot.gripper_open_position=0.0 \
  --robot.gripper_closed_position=0.4 \
  --robot.gripper_max_skew_s=0.01 \
  --task='move the paper cup from one end of the can to the other.' \
  --policy_type=fastwam \
  --pretrained_name_or_path=server-owned \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=32 \
  --robot.max_action_chunk_waypoints=32 \
  --action_offset=1 \
  --fps=30 \
  --chunk_size_threshold=0.0 \
  --aggregate_fn_name=latest_only \
  --enable_pending_observation=true \
  --pending_observation_timeout_s=30 \
  --rename_map={}
```
```
python -m lerobot_robot_franka_ros.ros2_client   --server_address=127.0.0.1:8080   --robot.type=franka_ros   --robot.id=franka_fastwam_schema1   --robot.dry_run=false   --robot.ros2_interface_only=true   --robot.base_frame=base   --robot.gripper_open_position=0.0   --robot.gripper_closed_position=0.4   --robot.gripper_max_skew_s=0.01   --task='grab the paper cup.'   --policy_type=fastwam   --pretrained_name_or_path=server-owned   --policy_device=cuda   --client_device=cpu   --actions_per_chunk=32   --robot.max_action_chunk_waypoints=32   --action_offset=1   --fps=30   --chunk_size_threshold=0.0   --aggregate_fn_name=latest_only   --enable_pending_observation=true   --pending_observation_timeout_s=30   --rename_map={}

source /opt/ros/jazzy/setup.bash
source franka_project/ros2_ws/install/setup.bash

 bash stop_all.sh && bash ~/.cursor/skills/franka-real-validation/scripts/start_stack.sh

python -m lerobot_robot_franka_ros.ros2_client   --server_address=127.0.0.1:8080   --robot.type=franka_ros   --robot.id=franka_fastwam_multitask   --robot.dry_run=false   --robot.ros2_interface_only=true   --robot.base_frame=base   --robot.gripper_open_position=0.0   --robot.gripper_closed_position=0.4   --robot.gripper_max_skew_s=0.01   --policy_type=fastwam   --pretrained_name_or_path=server-owned   --policy_device=cuda   --client_device=cpu   --actions_per_chunk=32   --robot.max_action_chunk_waypoints=32   --action_offset=1   --fps=30   --chunk_size_threshold=0.0   --aggregate_fn_name=latest_only   --enable_pending_observation=true   --pending_observation_timeout_s=30   --rename_map={} --interactive_task_control=true --gateway_arm_timeout_s=5 --visualize_action_web=true --robot.policy_eef_frame=link8
```

FastWAM 必须使用空 `rename_map`；不要复制 PI0 的 camera rename map。

## 6. 常见 fail-fast 报错

| 报错 | 含义与处理 |
|---|---|
| `server=2, client=None` | client 实际加载了旧 wire/protobuf；同步代码、重装 ROS 插件并重启进程 |
| checkpoint 文件缺失或 size mismatch | 路径错误、文件不完整或复制中断；重新检查 checkpoint 目录 |
| strict tensor key/shape/load 错误 | 权重与当前模型代码确实不兼容；使用对应训练代码或 checkpoint |
| `observation task does not match` | client `--task` 必须与 checkpoint task 完全一致，包括句号 |
| FastWAM import/Hydra 错误 | 当前 Python 环境未同时满足 LeRobot 与 FastWAM 依赖 |

启动后如果 server 已收到 observation 却没有入队，确认 server 使用：

```text
--observation_similarity_mode=none
```

这样不会因为 state 与上一帧相似而跳过 inference。


## 7. 最新启动方式：mix3 `step_005000`（27353 + SSH 隧道）

本节是当前推荐方式。数据流为：

```text
client -> 127.0.0.1:18080 -> SSH:2222 -> server:15173
       -> CartesianActionChunk -> IK gateway -> Franka controller
```

固定配置：

```text
checkpoint: step_005000.pt
task:       pick up the cup / chips / tape
fps:        30
chunk:      32（禁止裁成 25）
pose frame: link8
gripper:    model action[6] 按 closed_0_1 解码
```

### 7.1 Client 机器：启动 KML SSH 和模型转发

先在 Edge 中登录 27353 的 KML 页面并完成 SSO/MFA，然后打开两个终端。

终端 A：

```bash
cd /home/pnp/Projects/lerobot

~/miniconda3/envs/lerobot/bin/python \
  franka_project/scripts/ws_ssh_client.py \
  --browser-cookie \
  --browser=edge
```

终端 B：

```bash
ssh -N \
  -L 127.0.0.1:18080:127.0.0.1:15173 \
  -p 2222 \
  -i ~/.ssh/id_rsa \
  -o IdentitiesOnly=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=10 \
  -o ServerAliveCountMax=3 \
  root@127.0.0.1
```

检查：

```bash
ss -ltnp '( sport = :2222 or sport = :18080 )'
```

两个端口均已监听时不要重复启动。

### 7.2 Server 机器：启动修复后的 FastWAM server

通过 `ssh -p 2222 -i ~/.ssh/id_rsa root@127.0.0.1` 登录 27353，在远端执行：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

export FASTWAM_CHECKPOINT=/m2v_intern/tujiahang/Projects/FastWAM/franka_project/runs/franka_eef_mix3_0804_joint_lora_after_warmup_pretrained_xt/lora_after_warmup_bs256_lr1e-4_r32a64/checkpoints/weights/step_005000.pt
export PYTHONPATH="$PWD/src:$PWD/franka_project/src:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 \
  /ytech_milm_intern/tujiahang/miniconda3/envs/franka-fastwam-serve/bin/python \
  franka_project/scripts/serve_franka_pi0_async.py \
  --host=127.0.0.1 \
  --port=15173 \
  --fps=30 \
  --inference_latency=0 \
  --obs_queue_timeout=1 \
  --observation_similarity_mode=none \
  --policy_type=fastwam \
  --pretrained_name_or_path="$FASTWAM_CHECKPOINT" \
  --actions_per_chunk=32 \
  --policy_device=cuda \
  --fastwam_state_gripper_encoding=closed_0_1 \
  --fastwam_action_gripper_encoding=closed_0_1 \
  --fastwam_joint_video_inference=true \
  --fastwam_joint_video_output_dir=/m2v_intern/tujiahang/Projects/lerobot/franka_project/runs/depoly_joint_videos/20260810

```

上面的基础命令默认关闭视频，只运行 action 推理。每次启动 server 时可用下列参数选择模式。

只生成 joint video、不保存：

```bash
--fastwam_joint_video_inference=true
```

生成并保存每次推理的视频：

```bash
export FASTWAM_VIDEO_OUTPUT=/m2v_intern/tujiahang/Projects/lerobot/franka_project/runs/depoly_joint_videos/20260810_mix3_step005000
mkdir -p "$FASTWAM_VIDEO_OUTPUT"

# 将这两个参数追加到基础 server 命令末尾：
--fastwam_joint_video_inference=true \
--fastwam_joint_video_output_dir="$FASTWAM_VIDEO_OUTPUT"
```

不传 `--fastwam_joint_video_inference` 即关闭视频。仅传 output dir 而未开启 joint-video 会
fail-fast。切换模式需要重启 server，不支持运行时热切换。

视频固定为 9 帧、7.5 FPS。同 seed 下 action 与 action-only 模式一致；保存视频会增加 Ceph I/O。

### 7.3 Client 机器：启动 Franka controller（保持 HOLD）

当前 controller 为 `inactive`、曾发生 FCI error，或者切换过其他同事的 joint 栈时，先完整清理：

```bash
bash ~/franka/stop_all.sh --keep-haply-manager
```

启动 Cartesian real-validation 栈：

```bash
# 不自动移动到 rest pose：
bash ~/.cursor/skills/franka-real-validation/scripts/start_stack.sh --no-rest

# 若需要移动到保存的标准起始位，确认工作区安全后去掉 --no-rest：
# bash ~/.cursor/skills/franka-real-validation/scripts/start_stack.sh
```

必须看到：

```text
REAL VALIDATION READY — HOLD / NOT ARMED
```

并确认：

```bash
source /opt/ros/jazzy/setup.bash
ros2 control list_controllers
ros2 topic echo /lerobot/franka/safety_gateway_status --once
```

三个 controller 应为 `active`，gateway 应满足：

```text
armed: false
state_fresh: true
robot_ready: true
controller_ready: true
has_active_plan: false
```

### 7.4 Client 机器：启动 LeRobot client

另开终端：

```bash
cd /home/pnp/Projects/lerobot

source /opt/ros/jazzy/setup.bash
source ~/franka/franka_ros2_ws/install/local_setup.bash
source ~/franka/haply_ros/install/local_setup.bash
source franka_project/ros2_ws/install/local_setup.bash

export PYTHONPATH="$PWD/src:$PWD/franka_project/src:$PWD/franka_project/ros_lerobot/src:${PYTHONPATH:-}"

~/miniconda3/envs/lerobot/bin/python \
  -m lerobot_robot_franka_ros.ros2_client \
  --server_address=127.0.0.1:18080 \
  --robot.type=franka_ros \
  --robot.id=franka_fastwam_mix3 \
  --robot.dry_run=false \
  --robot.ros2_interface_only=true \
  --robot.base_frame=base \
  --robot.gripper_open_position=0.0 \
  --robot.gripper_closed_position=0.4 \
  --robot.gripper_max_skew_s=0.01 \
  --robot.camera2_max_skew_s=0.1 \
  --robot.eef_max_skew_s=0.05 \
  --robot.max_action_chunk_waypoints=32 \
  --robot.policy_eef_frame=link8 \
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
  --pending_observation_timeout_s=120 \
  --rename_map={} \
  --interactive_task_control=true \
  --visualize_action_web=true \
  --gateway_arm_timeout_s=5
```

正常启动应出现：

```text
F_T_EE captured for policy_eef_frame=link8
Interactive client started in HOLD with 3 tasks
```

### 7.5 任务按键与安全停止

在 `task>` 提示符中：

```text
1  pick up the cup
2  pick up the chips
3  pick up the tape
s  停止任务并 DISARM
a  ARM 当前任务
q  DISARM 并退出
```

选择 `1/2/3` 后仍保持 HOLD。按 `a` 会开始真实机械臂和夹爪运动；只有在工作区无人、无障碍物、
急停可立即触达，并完成现场安全确认后才能执行。

随时停止：

```bash
source /opt/ros/jazzy/setup.bash
source ~/Projects/lerobot/franka_project/ros2_ws/install/setup.bash

ros2 service call /franka_cartesian_safety_gateway/set_armed \
  std_srvs/srv/SetBool '{data: false}'
```

完全停栈：

```bash
bash ~/franka/stop_all.sh --keep-haply-manager
```
