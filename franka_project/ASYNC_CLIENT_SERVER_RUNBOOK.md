# Franka PI0 client-server dry-run runbook

本 runbook 只覆盖已经批准的第一阶段：LeRobot client 与 PI0 server 异步通信。Franka
插件读取冻结 observation fixture，并把返回动作写入 JSONL；它不会 import ROS、连接
controller 或驱动真机。

## 固定拓扑

```text
stock RobotClient
  -> 127.0.0.1:8080
  -> WebSocket tunnel client
  -> wss://kml-dtmachine-27353-prod-0.kmlhb2az1l3-2.corp.kuaishou.com/ws
  -> KML gateway 映射到 0.0.0.0:16782
  -> WebSocket tunnel server
  -> 127.0.0.1:15173
  -> FrankaPI0PolicyServer
```

KML URL 使用 `wss://.../ws`，不追加 `:16782`。`16782` 是 gateway 在 KML 机器上的
映射目标，不是 robot client 直接连接的 raw TCP 端口。

## 当前外部阻塞（2026-07-17）

对上述 `/ws` 发起实际 WebSocket Upgrade 时，KML AccessProxy 返回 HTTP `302` 并重定向
到公司 SSO；请求没有到达正在监听的 tunnel server，因此目前只能完成本机同拓扑验证，
不能声称远端 KML E2E 已通过。

在机器人侧运行前，需要 KML 平台提供以下任一种能力：

- 允许该 route 进行非浏览器 WebSocket/machine-to-machine 访问；或
- 提供平台正式支持的 machine token/header 认证方式。

不要把浏览器 SSO cookie 写进脚本：它会过期，也不适合作为机器人通信凭证。可用 route 的
验收标志是 WebSocket Upgrade 返回 `101 Switching Protocols`，并且 tunnel server 打印
`websocket connected ... path=/ws`。

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

## 2. KML：启动 PI0 server

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
  --policy_type=pi0 \
  --pretrained_name_or_path="$CHECKPOINT" \
  --actions_per_chunk=50 \
  --policy_device=cuda
```

Launcher 会 fail-fast 检查 loopback 地址、端口、15 Hz、50-step chunk、PI0 checkpoint
manifest/hash/geometry/stats 和 CUDA device。第一次 client 握手时才真正加载模型。

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

## 4. 机器人侧：启动 tunnel client

确认 KML route 已解决上述 SSO 问题后执行：

```bash
cd /path/to/lerobot

python tools/ws_tcp_tunnel.py client \
  --listen-host=127.0.0.1 \
  --listen-port=8080 \
  --ws-url=wss://kml-dtmachine-27353-prod-0.kmlhb2az1l3-2.corp.kuaishou.com/ws
```

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
  --fps=15 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only \
  --enable_pending_observation=true \
  --pending_observation_timeout_s=10 \
  '--rename_map={"observation.images.camera1":"observation.images.base_0_rgb","observation.images.camera2":"observation.images.left_wrist_0_rgb"}'
```

这里的 `pretrained_name_or_path=server-owned` 只是满足 client 配置；server CLI 中的真实
checkpoint 路径具有优先级。插件只接受 exact 10D state、两路 `480×640×3 uint8` 图像和
absolute8 action；`dry_run=false` 会直接拒绝启动。

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

当前 stock LeRobot 行为保持不变：相同 fixture observation 可能被 server 的
`observations_similar()` 过滤，而 client 的 pending flag 要到 10 秒 timeout 后才重发。这是
后续真机阶段需要重新评审的 observation/queue 时序问题；本阶段没有修改 LeRobot core。
