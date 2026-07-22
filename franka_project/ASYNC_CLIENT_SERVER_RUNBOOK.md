# Franka PI0 / FastWAM client-server runbook

本 runbook 覆盖 LeRobot client 与 PI0/FastWAM server 的异步通信。
本文命令让 Franka 插件读取冻结 observation fixture，并把返回动作写入 JSONL；
它不会 import ROS、连接 controller 或驱动真机。后续新增的非执行 ROS2 interface 另见
[`ROS2_INTERFACE.md`](./ROS2_INTERFACE.md)，不改变本文 dry-run 验收范围。

## 固定拓扑

```text
stock RobotClient
  -> 127.0.0.1:8080
  -> WebSocket tunnel client
  -> wss://kml-dtmachine-27353-prod-0.kmlhb2az1l3-2.corp.kuaishou.com/ws
  -> KML gateway 映射到 0.0.0.0:16782
  -> WebSocket tunnel server
  -> 127.0.0.1:15173
  -> FrankaPI0PolicyServer 或 FrankaFastWAMPolicyServer
```

KML URL 使用 `wss://.../ws`，不追加 `:16782`。`16782` 是 gateway 在 KML 机器上的
映射目标，不是 robot client 直接连接的 raw TCP 端口。

## 当前外部阻塞（2026-07-17）

对上述 `/ws` 发起实际 WebSocket Upgrade 时，KML AccessProxy 返回 HTTP `302` 并重定向
到公司 SSO；未认证的请求不会到达正在监听的 tunnel server。当前有人值守的
推荐流程是：用户先在日常 Firefox 中正常完成公司 SSO/MFA，再由
`tools/start_kml_tunnel_client.py` 只读已有 profile 的 `cookies.sqlite`，只选取对
目标 KML host/path 有效的 Cookie，供后续 gRPC 连接和重连内存复用。

这不是绕过 SSO/MFA。launcher 不会打印 Cookie，也不会把它放进命令行或写入长期
cookie 文件。它通过 `KML_COOKIE` 初始进程环境把 Cookie 传给 tunnel，tunnel 读取后立即
从 Python 的 `os.environ` 映射删除，并仅在进程内存中使用。Linux 上同一用户或特权进程
仍可能通过 `/proc/<pid>/environ` 或进程内存观察初始环境，因此这不是抵御本机同权限用户的
secret boundary。若 Firefox 数据库被锁定，launcher 会在临时目录中建立
`cookies.sqlite`/WAL 稳定快照，读取后自动删除。
该流程可用于当前有人值守的 remote dry-run 验收，但在实际验收完成前仍不能声称
远端 KML E2E 已通过。

长期无人值守运行的外部阻塞仍未解决。该场景应向 KML 平台申请正式支持的
machine token、mTLS 或等价的 machine-to-machine 身份，不应把人类 SSO Cookie 当作机器人
的长期通信凭证。

## 1. 安装 out-of-tree 插件

在机器人侧 LeRobot checkout 中执行；不会安装 ROS 依赖：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

uv pip install --no-deps -e franka_project/ros_lerobot
```

如果该机器没有 `uv`，使用其 LeRobot Python 环境：

```bash
python -m pip install --no-deps -e franka_project/ros_lerobot
```

如果 shell 报 `Run 'conda init' before 'conda activate'`，无需修改 shell 配置，可直接执行：

```bash
source /ytech_milm_intern/tujiahang/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
```

两端启动 tunnel 前确认基础通信依赖；`aiohttp` 当前不是 `lerobot[async]` 自动依赖：

```bash
python -c 'import aiohttp, grpc; print(aiohttp.__version__, grpc.__version__)'
```

若缺少 `aiohttp`，在对应的 LeRobot 环境安装：

```bash
python -m pip install aiohttp
```

确认 stock client 能自动发现插件：

```bash
python -m lerobot.async_inference.robot_client --help | grep franka_ros
```

## 2. KML：启动 policy server

`serve_franka_pi0_async.py` 保留历史文件名以兼容旧命令；实际 backend 由
`--policy_type=pi0|fastwam` 在进程启动时选择，不支持运行中热切换。

### 2.1 PI0

Checkpoint 不提交到 Git。先把完整 `pretrained_model` 目录放在 server 本地，并设置绝对路径：

```bash
export CHECKPOINT=/absolute/path/to/pretrained_model
```

目录必须包含：

```text
franka_pi0_checkpoint_manifest.json
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_6_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
franka_eef_geometry_manifest.json
pi0_eef_stats.json
```

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

test -f "$CHECKPOINT/model.safetensors"

CUDA_VISIBLE_DEVICES=1 python franka_project/scripts/serve_franka_pi0_async.py \
  --host=127.0.0.1 \
  --port=15173 \
  --fps=15 \
  --inference_latency=0 \
  --obs_queue_timeout=1 \
  --observation_similarity_mode=none \
  --policy_type=pi0 \
  --pretrained_name_or_path="$CHECKPOINT" \
  --actions_per_chunk=50 \
  --policy_device=cuda
```

Launcher 会从所选 checkpoint 的 `config.json`、geometry manifest 和 stats manifest 推导
task、profile、FPS 与 chunk size，并 fail-fast 检查它们和 CLI 输入一致；不再写死 15 Hz 或
50-step chunk。上面是杯子 checkpoint 的 15 Hz 示例；薯片 native30 checkpoint 应把 server
和 client 的 `--fps` 都改为 `30`。第一次 client 握手会校验小型 metadata 并 strict-load
模型 tensor，但不会重新计算多 GB `model.safetensors` 的完整 SHA-256，也不会完整比较 manifest
中的 `training_graph` 报告。

### 2.2 FastWAM move-cups checkpoint

当前批准的 FastWAM artifact 被
[`manifests/fastwam_move_cups_step_019650.json`](./manifests/fastwam_move_cups_step_019650.json)
锁定。server 启动时会校验 checkpoint、runtime YAML、dataset contract、训练统计、文本
embedding、Wan VAE、FastWAM Python source tree 和 pyproject。12 GB checkpoint 与 Wan VAE
只检查路径、存在性和 size；小型 metadata/source 文件仍检查 SHA-256。

现有 `lerobot` conda 环境缺少 Hydra/OmegaConf/FastWAM runtime，现有 `fastwam` 环境则是
Python 3.10，低于本仓库要求的 Python 3.12。真实模型启动前必须先准备并验证一个 Python
3.12 serving 环境；不要直接把两个项目完整依赖集合强行覆盖安装到任一现有环境。本提交已
验证 artifact 契约和无模型 inference 路径，尚未完成 12 GB 权重的真实 CUDA load smoke。

环境准备完成后先检查：

```bash
python -c 'import torch, hydra, omegaconf, boto3; import fastwam.runtime; print(torch.__version__)'
```

再启动：

```bash
export FASTWAM_CHECKPOINT=/m2v_intern/tujiahang/Projects/FastWAM/franka_project/runs/franka_eef_move_cups/checkpoints/weights/step_019650.pt
export DIFFSYNTH_MODEL_BASE_PATH=/m2v_intern/tujiahang/Projects/FastWAM/checkpoints

cd /m2v_intern/tujiahang/Projects/lerobot

CUDA_VISIBLE_DEVICES=1 python franka_project/scripts/serve_franka_pi0_async.py \
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

固定 task 是：

```text
move the paper cup from one end of the can to the other.
```

FastWAM client 必须保持 `rename_map={}`；PI0 的相机 rename map 不能复用。client/server 握手
会同时校验 protocol version、FPS、policy type 和 chunk size。server 必须返回显式
`PolicySetupAck`，client 会核对其中的 resolved contract；误连旧 server 时响应会解码成全零
默认值并被拒绝，因此新旧任一端未同步都不能进入 observation/action loop。

FastWAM 模型的第 7 维输出是 absolute `gripper.open_target_0_1`。server 在逐行累计 pose delta
后转换成 canonical `closed_0_1 = 1 - clip(open, 0, 1)`，再把 absolute8 发给 client。
`0.04` 只用于把输入 proprio 的开合比例编码成左右 pseudo-finger 米制位置
`[+0.04*open, -0.04*open]`；它绝不能乘到输出命令上。若后续 gripper driver 需要原始关节单位，
应在机器人侧按实测标定把 `closed_0_1` 映射到 `[raw_open, raw_closed]`。

## 3. KML：启动 16782 tunnel server

另开终端：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

python tools/ws_tcp_tunnel.py server \
  --listen-host=0.0.0.0 \
  --listen-port=16782 \
  --target-host=127.0.0.1 \
  --target-port=15173
```

该 tunnel server 没有应用层认证或 ACL，只能运行在由 KML AccessProxy 和网络 ACL 保护的
实验环境中，不得直接暴露到公网或宽泛可访问的公司网络。其后端 PolicyServer 使用 Python
pickle 传输内部对象，只能接受受信任的 LeRobot client；不得把 `16782` 或 `15173` 作为
公共服务端口开放。

## 4. 机器人侧：启动 tunnel client

先使用当前 Unix 用户的日常 Firefox 访问 KML 目标机器，正常完成 SSO 和 MFA。
确认该 Firefox profile 中已有有效 KML 会话后，在当前机器执行：

```bash
source /home/pnp/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/pnp/Projects/lerobot

python tools/start_kml_tunnel_client.py \
  --listen-host=127.0.0.1 \
  --listen-port=8080 \
  --ws-url=wss://kml-dtmachine-27353-prod-0.kmlhb2az1l3-2.corp.kuaishou.com/ws
```

默认会从 `~/snap/firefox/common/.mozilla/firefox` 和 `~/.mozilla/firefox` 中自动查找
Firefox 默认 profile。如果有多个 profile 并且自动选择不正确，显式指定：

```bash
python tools/start_kml_tunnel_client.py \
  --cookie-db=/absolute/path/to/firefox/profile/cookies.sqlite
```

有效 Cookie 已存在时，launcher 本身不会打开浏览器，也不要求
`DISPLAY`/`WAYLAND_DISPLAY`。Cookie 过期时，先停止 tunnel，在日常 Firefox 中重新完成
KML SSO/MFA，然后重新运行 launcher；该方式不会自动弹窗刷新。

tunnel client 出现以下日志表示认证信息已进入内存，且本地 `8080` 已就绪：

```text
loaded ... target-scoped cookie(s) from the environment
client listening on ('127.0.0.1', 8080), forwarding to wss://kml-dtmachine-.../ws
```

如果无法从已有 profile 读取有效 Cookie，可在图形桌面终端使用以下备用交互
模式。它会打开隔离的临时 Firefox，仍需要用户正常完成 SSO/MFA：

```bash
python tools/ws_tcp_tunnel.py client \
  --listen-host=127.0.0.1 \
  --listen-port=8080 \
  --ws-url=wss://kml-dtmachine-27353-prod-0.kmlhb2az1l3-2.corp.kuaishou.com/ws \
  --auth=kml-firefox \
  --geckodriver=/snap/bin/geckodriver \
  --auth-timeout=600
```

随后启动第 5 节的 RobotClient。远端 tunnel server 打印
`websocket connected ... path=/ws` 才表示 WebSocket Upgrade 已返回 `101 Switching Protocols`
并真正穿过 KML gateway。这只改变通信链路的交互式认证方式；本阶段仍然是
fixture observation 和 JSONL action sink 的 `dry_run=true`，不会连接 ROS/controller 或驱动
Franka 真机。

## 5. 机器人侧：启动 stock RobotClient（仍为 dry-run）

下面是本阶段唯一推荐的配置。尤其不要把 `latest_only` 改成
`weighted_average`，因为 absolute quaternion 不能直接逐元素平均。

```bash
cd /path/to/lerobot

python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=franka_ros \
  --robot.id=franka_async_dry_run \
  --robot.dry_run=true \
  --robot.fixture_path="$PWD/franka_project/fixtures/async/franka_observation_v1.npz" \
  --robot.action_log_path=/tmp/franka_async_actions.jsonl \
  --task='stack the cups' \
  --policy_type=pi0 \
  --pretrained_name_or_path=server-owned \
  --policy_device=cpu \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --action_offset=1 \
  --fps=15 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only \
  --enable_pending_observation=true \
  --pending_observation_timeout_s=10 \
  '--rename_map={"observation.images.camera1":"observation.images.base_0_rgb","observation.images.camera2":"observation.images.left_wrist_0_rgb"}'
```

`action_offset` 是 RobotClient 顶层参数，默认 `0` 以兼容旧行为。这里显式设置为 `1`，使新
observation 使用 `max(latest_action + 1, 0)` 作为 timestep，PolicyServer 的首条 action 因而
从下一 timestep 开始编号。若等待 action 返回期间 `latest_action` 不推进，50-step response
可完整保留；若 cursor 同期推进了 `m` 步，仍会裁剪前 `m` 条真正 stale action，所以这不是
无条件的 50 条保证。该选项不改变 pending retry、gRPC ACK 或 ROS safety gateway 协议。

实际 ROS2 client 若需要保持 Server/checkpoint 的 50-step 合同、但每次只提交最多 30 个
waypoint，可额外设置 `--robot.max_action_chunk_waypoints=30`。该选项只在专用
`lerobot_robot_franka_ros.ros2_client` 入口生效，并同时限制本地有效 queue 与 ROS chunk；
不能仅截断 publisher。完整时序和 ACK 语义见 [`ROS2_INTERFACE.md`](./ROS2_INTERFACE.md)。

这里的 `pretrained_name_or_path=server-owned` 只是满足 client 配置；server CLI 中的真实
checkpoint 路径具有优先级。插件只接受 exact 10D state、两路 `480×640×3 uint8` 图像和
absolute8 action。本文命令必须保持 `dry_run=true`；`dry_run=false` 现在只允许进入
`ros2_interface_only=true` 的隔离接口，并且必须改用 `lerobot_robot_franka_ros.ros2_client`
入口，详见 `ROS2_INTERFACE.md`。

## 6. 本机回归 KML 拓扑

在没有第二台机器时，可以把第 4 步的 URL 临时换成：

```text
ws://127.0.0.1:16782/ws
```

这会完整验证 `RobotClient -> 8080 -> WebSocket -> 16782 -> 15173 -> PI0 -> sink`，但不
验证 KML gateway/SSO。

## 7. 验收与已知现象

成功时应看到：

- server：真实 checkpoint strict load，model action shape `[1,50,7]`；
- tunnel server：`GET /ws ... 101`；
- client：插件自动发现，并收到 action chunk；
- JSONL：连续、finite、单位 quaternion 的 absolute8 action。

默认的 `--observation_similarity_mode=none` 不会根据 state 相似度过滤 observation；如需
恢复原行为，可显式设置为 `state`。若 `SendObservations` 已成功但没有产生 action，client
会在 pending timeout 后采集一帧新的 observation。若推理已完成但 action response 在断线
窗口丢失，server 会缓存并重发同一个 chunk，直到 client 本地提交后 ACK。完整状态机和故障恢复边界见
[`ACTION_DELIVERY_ACK_PROTOCOL.md`](./ACTION_DELIVERY_ACK_PROTOCOL.md)。
