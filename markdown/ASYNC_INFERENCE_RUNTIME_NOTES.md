# Async Inference 运行说明

这份文档总结 `lerobot.async_inference` 在远程 policy server 上运行 pi0 时的通信链路、延时来源、observation 采集频率，以及 action chunk 如何被 client 消费。

## 运行链路

```text
robot_client
  读取机械臂状态和相机图像
  构造 TimedObservation
  pickle 序列化并通过 gRPC 发送 observation
  从 server 接收 TimedAction chunk
  将 action 合并进本地 action_queue
  control loop 每个 tick 执行一个 action

policy_server
  接收 TimedObservation
  将 robot observation key 转成 LeRobot observation key
  resize 图像、tokenize task、normalize 输入
  执行 policy.predict_action_chunk()
  unnormalize action
  将 TimedAction list 返回给 client
```

如果中间使用 websocket tunnel，那么 gRPC 流量还会经过 tunnel，tunnel 的网络延时、断连和重连都会影响整体闭环。

## 默认频率

client 和 server 默认都是：

```text
fps = 30
environment_dt = 1 / 30 = 33.3 ms
```

client 的 control loop 会尝试按 30Hz 运行。要稳定达到 30Hz，机械臂状态读取、相机读取、action 写入和 Python 本地开销都需要控制在 33.3ms 内。

这里需要区分“循环机会”和“实际发送”：

- action 执行机会：client control loop 每个 tick 都会检查 action queue，如果 queue 非空，就执行一个 action。因此理想情况下 action 执行频率是 30Hz。
- observation 采集机会：client control loop 每个 tick 都会检查是否应该发送 observation，因此检查频率也是 30Hz。
- observation 实际发送频率：不一定是 30Hz，取决于 action queue 余量，见下一节。

## Observation 发送频率

client 并不是有了 action queue 之后每一帧都发送 observation。发送条件是：

```python
action_queue.qsize() / action_chunk_size <= chunk_size_threshold
```

以你当前的设置为例：

```text
actions_per_chunk = 50
chunk_size_threshold = 0.5
fps = 30
```

client 会在队列里大约剩 25 个 action 时发送新的 observation。理想情况下：

```text
25 actions / 30 Hz ~= 0.83 秒发送一次新的 observation
```

启动时 action queue 是空的，所以第一帧 observation 会被标记为：

```text
must_go=True
```

这意味着它应该被 server 立即处理，用来生成第一批 action。

日志中的 observation 编号是 timestep，不是墙上时间。例如：

```text
observation #3354 -> observation #3378
```

差了 24 个 timestep。30Hz 下约等于：

```text
24 / 30 = 0.8 秒
```

这通常表示 client 在消耗已有 action queue，等队列余量接近阈值后才发送新的 observation。

## pi0 的 Action Chunk 机制

pi0 输出的是一个 action chunk：

```text
[batch, chunk_size, action_dim]
```

server 会按你的参数裁剪：

```python
chunk[:, :actions_per_chunk, :]
```

然后把每个 action 包成一个 `TimedAction`：

```text
timestamp = observation_timestamp + i * environment_dt
timestep = observation_timestep + i
```

client 收到后，会把这些 `TimedAction` 合并进本地 `action_queue`。control loop 每 1/30 秒从 queue 里取一个 action，调用 `robot.send_action()` 发给机械臂。

## 一个 Chunk 里的 Action 都会被用到吗？

不一定。

如果没有新的 chunk 到来，当前 queue 里的 action 会按 30Hz 一个一个执行，直到执行完。

如果新的 chunk 提前到来，client 会执行 `_aggregate_action_queues()` 合并新旧队列：

- 已经过期的 action 会被跳过
- timestep 相同的 action 会用 `aggregate_fn` 合并
- 只存在于新 chunk 里的 action 会直接加入 queue

你当前使用：

```bash
--aggregate_fn_name=weighted_average
```

重叠 timestep 的 action 会这样合并：

```python
0.3 * old + 0.7 * new
```

所以这套机制不是“必须把旧 chunk 全部执行完”，而是更偏向尽快采用基于最新 observation 生成的新动作。

## 主要延时来源

常见延时来源包括：

1. client 端相机采集和机械臂状态读取。
2. `TimedObservation` 的 pickle 序列化。
3. 图像 observation 的 gRPC 分块传输。
4. websocket tunnel 延时，如果使用 tunnel。
5. server 端反序列化。
6. 图像 resize 和 observation key 映射。
7. task 文本 tokenization。
8. normalization 和数据搬到 CUDA。
9. pi0 denoising 推理，和 `num_inference_steps` 有关。
10. action postprocess 和 unnormalization。
11. action chunk 的 pickle 序列化返回。
12. client 端反序列化和 action queue 合并。
13. action 在 `action_queue` 里的等待时间。
14. Feetech 总线向电机写 action 的延时。

## 关键日志

server 端如果真的进入推理，应看到：

```text
Running inference for observation #...
Preprocessing and inference took ... action shape: ...
Observation ... | Total time: ...
Action chunk #... generated
```

client 端如果真的收到并执行 action，应看到：

```text
Received action chunk for step #...
Action #... performed
```

如果 server 一直只有：

```text
Starting receiver
```

但没有 `Running inference`，说明 server 正在接收 observation stream，但还没有成功走到“入队 observation -> 推理 -> 生成 action chunk”的流程。

### Server 端时间字段说明

典型日志：

```text
INFO ... Action chunk #3353 generated | Total time: 276.06ms
INFO ... Running inference for observation #3354 (must_go: False)
INFO ... Preprocessing and inference took 0.2586s, action shape: torch.Size([1, 50, 6])
INFO ... Observation 3354 | Total time: 278.43ms
INFO ... Action chunk #3354 generated | Total time: 283.93ms
```

含义如下：

- `2026-06-22 21:20:41`：server 打印日志的墙上时间，日志默认只显示到秒。
- `observation #3354`：client control loop 的 timestep，不是 Unix timestamp。
- `must_go: False`：这帧 observation 不是“队列空了必须处理”的强制帧，而是根据 observation 筛选逻辑进入推理。
- `Preprocessing and inference took 0.2586s`：主要是 `policy.predict_action_chunk()` 的耗时。日志名字里带 preprocessing，但实际计时点从调用 policy 推理前开始，到 action tensor 输出结束。
- `action shape: torch.Size([1, 50, 6])`：batch=1，chunk 长度=50，每个 action 维度=6。SO101 正常应该是 6 维。
- `Observation 3354 | Total time: 278.43ms`：`_predict_action_chunk()` 内部从准备 observation 到 postprocess action 完成的总耗时，大致包括 raw observation 转换、图像 resize、preprocessor、policy 推理、postprocessor/unnormalize。
- `Action chunk #3354 generated | Total time: 283.93ms`：`GetActions()` 外层统计的总耗时，约等于 `_predict_action_chunk()` 耗时加上 action chunk 的 pickle 序列化时间。

因此通常：

```text
Action chunk Total time >= Observation Total time
```

两者差值通常是 action list 打包、序列化和少量外层逻辑的开销。

## 常见问题

### 模型加载成功，但 action shape 是 32 而不是 6

pi0 的 LoRA/PEFT checkpoint 必须用 adapter 目录里的 `config.json` 初始化 base pi0 policy，而不能只用 base model 的 config。

否则 pi0 可能按：

```text
max_action_dim = 32
```

输出动作，而 SO101 的 postprocessor 只期望 6 维 action，导致类似错误：

```text
The size of tensor a (32) must match the size of tensor b (6)
```

SO101 正常 action shape 应该是：

```text
[1, 50, 6]
```

异常 shape 是：

```text
[1, 50, 32]
```

### 相机 key 不匹配

如果 robot 端发的是：

```text
observation.images.front
observation.images.wrist.left
```

但 checkpoint 期望的是：

```text
observation.images.base_0_rgb
observation.images.left_wrist_0_rgb
```

需要在 client 命令里使用：

```bash
--rename_map='{
  "observation.images.front": "observation.images.base_0_rgb",
  "observation.images.wrist.left": "observation.images.left_wrist_0_rgb"
}'
```

### 机械臂没有动作

按下面顺序排查：

1. server 日志是否有 `Action chunk #... generated`。
2. client 日志是否有 `Received action chunk`。
3. client 日志是否有 `Action #... performed`。
4. action 数值是否和当前关节位置有明显差异。
5. 电机 torque 是否启用，是否有 Feetech 通信错误。

如果第 1 步失败，优先排查 server 推理和 observation 传输。  
如果第 2 步失败，优先排查 action 返回路径和网络/tunnel。  
如果第 3 步失败，优先排查 client action queue 和 control loop。  
如果前三步都正常但机械臂仍不动，再检查 action 数值、机械臂安全限制和电机通信。
