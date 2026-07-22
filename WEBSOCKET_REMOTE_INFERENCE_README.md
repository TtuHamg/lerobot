# 通过 KML WebSocket 进行 LeRobot 远程推理

> **Franka 当前入口（2026-07-17）**：端口已改为 `16782`，KML URL 为
> `wss://kml-dtmachine-27353-prod-0.kmlhb2az1l3-2.corp.kuaishou.com/ws`。Franka 的完整固定
> 配置、dry-run 命令和当前 SSO `302` 阻塞见
> [`franka_project/ASYNC_CLIENT_SERVER_RUNBOOK.md`](franka_project/ASYNC_CLIENT_SERVER_RUNBOOK.md)。
> 下文保留的是早期 SO-101/端口 `15019` 记录，不应直接用于 Franka。

这份文档记录当前的部署方案：在 KML 开发机上运行 policy inference，同时在另一台 4070 主机上连接并控制 SO-101 follower 机械臂。

## 为什么需要这个方案

KML 开发机有一个固定内网 IP：

```text
10.82.124.31
```

但是这个 IP 并不会把任意端口作为普通 raw TCP 服务暴露出来。实际情况是：

```text
ping 10.82.124.31        可以
ssh 10.82.124.31         不可以
nc 10.82.124.31 8080     不可以
```

KML 是通过 HTTP/WebSocket 网关暴露服务的。之前用于 code-server 的浏览器 URL：

```text
http://kml-dtmachine-26501-prod.kmlhb2az1l3-2.corp.kuaishou.com
```

会映射到开发机当前暴露的端口，目前是 `15019`。

LeRobot async inference 默认使用 gRPC over raw TCP：

```text
4070 robot_client  -- gRPC/TCP -->  dev-machine policy_server
```

但 KML 网络模型阻断了这条直接 TCP 路径。因此当前的解决办法是：把 gRPC 字节流包装到 WebSocket 里面传输。

## 架构

```text
4070 主机
  lerobot.async_inference.robot_client
    连接 127.0.0.1:8080
        |
        v
  tools/ws_tcp_tunnel.py client
    将本地 TCP 字节流包装成 WebSocket
        |
        v
  KML WebSocket gateway
    ws://kml-dtmachine-26501-prod.kmlhb2az1l3-2.corp.kuaishou.com/ws
        |
        v
开发机
  tools/ws_tcp_tunnel.py server
    将 WebSocket 字节流还原为 TCP
        |
        v
  lerobot.async_inference.policy_server
    监听 127.0.0.1:15173
```

端口用途：

```text
15019  KML 暴露的 HTTP/WebSocket 入口。通常被 code-server 使用，也可以临时用于 tunnel 测试。
15173  开发机上的 LeRobot policy_server 端口。
8080   4070 主机上的本地 TCP 监听端口。robot_client 连接这个端口。
```

如果 KML 后续为 `15174` 提供单独的外部 URL，建议 tunnel server 使用 `15174`，把 `15019` 留给 Jupyter/code-server。

## URL 规则

WebSocket 必须使用 `ws://`，不能使用 `http://`。

```text
http://...   普通浏览器/HTTP 流量
ws://...     WebSocket 流量
https://...  普通 TLS HTTP 流量
wss://...    TLS WebSocket 流量
```

当前 KML URL 是：

```text
http://kml-dtmachine-26501-prod.kmlhb2az1l3-2.corp.kuaishou.com
```

因此 tunnel client 应该使用：

```text
ws://kml-dtmachine-26501-prod.kmlhb2az1l3-2.corp.kuaishou.com/ws
```

tunnel server 可以接受任意 path。因此只要网关允许，`/ws` 和 `/` 都可以。

## 开发机配置

下面命令在 KML 开发机上运行。

### 1. 启动 LeRobot Policy Server

policy server 建议只绑定本机内部地址：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=15173
```

保持这个进程持续运行。

### 2. 启动 WebSocket Tunnel Server

如果使用当前 KML 暴露端口 `15019`，需要先停止占用该端口的 code-server/Jupyter：

```bash
PORT=15019
PID=$(lsof -t -i:$PORT)
if [ -n "$PID" ]; then
  kill -9 "$PID"
fi
```

然后启动 tunnel server：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

python tools/ws_tcp_tunnel.py server \
  --listen-host=0.0.0.0 \
  --listen-port=15019 \
  --target-host=127.0.0.1 \
  --target-port=15173
```

预期日志：

```text
server listening on 0.0.0.0:15019, forwarding websocket streams to 127.0.0.1:15173
```

如果 KML 为 `15174` 提供了单独的外部 URL，建议改用：

```bash
python tools/ws_tcp_tunnel.py server \
  --listen-host=0.0.0.0 \
  --listen-port=15174 \
  --target-host=127.0.0.1 \
  --target-port=15173
```

## 4070 主机配置

下面命令在连接 SO-101 follower 机械臂的 4070 主机上运行。

### 1. 准备 Tunnel 脚本

从开发机复制这个文件到 4070 主机：

```text
tools/ws_tcp_tunnel.py
```

如果缺少 Python 依赖，安装：

```bash
pip install aiohttp
```

### 2. 启动 WebSocket Tunnel Client

```bash
python ws_tcp_tunnel.py client \
  --listen-host=127.0.0.1 \
  --listen-port=8080 \
  --ws-url=ws://kml-dtmachine-26501-prod.kmlhb2az1l3-2.corp.kuaishou.com/ws
```

保持这个进程持续运行。

连接成功后，开发机 tunnel server 上应看到类似日志：

```text
websocket connected ...
```

### 3. 启动 LeRobot Robot Client

robot client 连接的是本地 tunnel，不是开发机 IP：

```bash
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist.left: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}' \
  --robot.id=tjh_follower_arm \
  --task="Place the black bottle cap into the white paper cup" \
  --policy_type=pi0 \
  --pretrained_name_or_path=/m2v_intern/tujiahang/Projects/lerobot/output_lerobot/<checkpoint>/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average
```

`--pretrained_name_or_path` 是开发机上的 checkpoint 路径，因为真正加载模型的是开发机上的 policy server。

如果 robot 端相机名称和 checkpoint 中的相机 key 不一致，需要加上 `rename_map`。例如 robot 端是：

```text
observation.images.front
observation.images.wrist.left
```

而 checkpoint 期望：

```text
observation.images.base_0_rgb
observation.images.left_wrist_0_rgb
```

则添加：

```bash
--rename_map='{
  "observation.images.front": "observation.images.base_0_rgb",
  "observation.images.wrist.left": "observation.images.left_wrist_0_rgb"
}'
```

## 预期成功日志

开发机 policy server：

```text
Client ... connected and ready
Receiving policy instructions ...
Policy type: pi0
Running inference for observation #...
Action chunk #... generated
```

开发机 tunnel server：

```text
websocket connected ...
```

4070 tunnel client：

```text
local tcp connected ...
```

4070 robot client：

```text
Robot connected and ready
Sending policy instructions to policy server
Received action chunk ...
Action #... performed
```

## 排查问题

### `nc 10.82.124.31 8080` 失败

这是预期现象。固定内网 IP 可以被 ICMP ping 到，但任意 raw TCP 端口不会被暴露。需要通过 KML 的 WebSocket gateway 访问。

### Tunnel Client 连不上

检查 URL 是否真的映射到 tunnel server 所监听的端口。如果 URL 仍然指向 code-server，就不会连到 tunnel server。

当前方案里 URL 映射到 `15019`，因此 tunnel server 必须监听 `15019`。

### 需要同时保留 Jupyter/code-server

向 KML 平台申请一个单独的外部 HTTP/WebSocket URL，例如映射到 `15174`。

然后开发机运行：

```bash
python tools/ws_tcp_tunnel.py server \
  --listen-host=0.0.0.0 \
  --listen-port=15174 \
  --target-host=127.0.0.1 \
  --target-port=15173
```

4070 主机上使用对应的 `ws://...` 或 `wss://...` URL。

### gRPC 连接成功，但 Policy 加载失败

检查：

```text
--policy_type
--pretrained_name_or_path
checkpoint config.json
开发机 Python 环境
```

对于 pi0 LoRA/PEFT checkpoint，需要确保 server 端代码支持从 adapter 目录读取 `config.json`，并用该 config 初始化 base pi0 policy。

### PI0 显存不够

PI0 对 12GB 4070 来说比较大。推荐在开发机 GPU 上推理。若必须本地运行，可以尝试支持情况下使用 BF16：

```bash
--policy.dtype=bfloat16
```
