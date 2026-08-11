# FastWAM mix3 `step_005000` 无动作部署审计

日期：2026-08-07
结论：**client/server 联调 Go；真机 ARM 仍为 No-Go，等待安全与执行门禁确认。**

## 1. 执行边界

本次审计：

- 未 ARM；
- 未向 ROS action topic 发布模型动作；
- 未停止或修改同事正在运行的 `/home/pnp/ght_wsp/lerobot` joint 栈；
- 所有模型请求均由 `dry_run=True` 的一次性 client 发出，并明确记录
  `actions_executed=false`；
- 仅启动过一个临时 `15175` action-only server 做 A/B，验证后已停止。

## 2. Artifact 与实际网络拓扑

目标 checkpoint：

```text
/m2v_intern/tujiahang/Projects/FastWAM/franka_project/runs/
franka_eef_mix3_0804_joint_lora_after_warmup_pretrained_xt/
lora_after_warmup_bs256_lr1e-4_r32a64/checkpoints/weights/step_005000.pt
```

已核对：

- 大小：`12,042,105,653` bytes；
- export 状态：`merged_and_strictly_verified`；
- export 记录 SHA-256：
  `300fb283793902758f645c1663a7df479ebc6699c5cd97a930a9ac11887ffe33`；
- model target：`fastwam.runtime.create_fastwam_joint`；
- dtype：BF16；
- checkpoint step：5000；
- action：30 Hz、32 steps；
- video：offset `0,4,...,32`，9 帧、7.5 FPS；
- task allowlist：
  - `pick up the cup`
  - `pick up the chips`
  - `pick up the tape`

本次建立的隔离路径：

```text
dry-run client
  -> 127.0.0.1:18080
  -> SSH local forward
  -> 127.0.0.1:2222
  -> SSH-over-WebSocket (远端 15590 -> sshd 220)
  -> server 127.0.0.1:15173
  -> mix3 step_005000
```

握手返回 protocol v2 和完整三任务 allowlist，证明没有进入同事的：

```text
127.0.0.1:8081 -> prod-1 -> 15645 -> 15174 -> 另一 checkpoint
```

`15599` 没有被 SSH 占用；SSH WebSocket server 实际监听 `15590`。

## 3. 训练与部署契约核验

### 3.1 相机映射：当前没有部署反序

0804 采集时物理 topic 命名反了：

```text
训练逻辑 camera1/global <- /camera2/...（采集时）
训练逻辑 camera2/wrist  <- /camera1/...（采集时）
```

实时抓图确认当前设备映射已恢复：

```text
部署 camera1 <- /camera1/... = global
部署 camera2 <- /camera2/... = wrist
```

因此训练和当前部署最终送入模型的语义顺序均为：

```text
[global, wrist]
```

结论：dataset contract 的反向 topic 映射是正确的采集修正，当前不应在 server
中再次按历史 topic 名交换图像。仍建议后续把物理 serial/角色写入握手账本。

### 3.2 Pose endpoint：训练数据确定为 `O_T_LINK8`

在 held-out cup episode 38、frame 120：

- 从 MCAP 提取同步 `/franka/joint_states`；
- 用同一 FR3 URDF 对 qpos 做 `fr3_link8` FK；
- 与训练存储 pose 比较。

结果：

```text
translation error = 0.0001050 m
rotation error    = 0.0000329 rad
```

这证明 mix3 的 `/current_pose` 数值实际是 `O_T_LINK8`，并非当前 ROS
broadcaster 语义下的 `O_T_EE`。现有 client 的转换方向正确：

```text
observation: O_T_LINK8 = O_T_EE * inverse(F_T_EE)
action:      O_T_EE    = O_T_LINK8 * F_T_EE
```

部署必须显式使用：

```text
--robot.policy_eef_frame=link8
```

### 3.3 Action 标签确定为相邻增量

为 held-out episode 38 的 frame 120、160、190 各导出一个 33-state /
32-action golden window。逐行重新计算：

```text
delta_xyz[k] = p[k+1] - p[k]
delta_rot[k] = Log(R[k]^T R[k+1])
```

与 parquet action 最大绝对误差：

| frame | max abs error |
|---:|---:|
| 120 | `1.8692e-7` |
| 160 | `1.9527e-7` |
| 190 | `1.9655e-7` |

结论：server 当前“逐步积分 32 个 adjacent delta”的 pose 解码方式正确。

### 3.4 确定根因：gripper 物理语义被反转

训练配置声明：

```text
raw_open=0.0
raw_closed=0.8
action[6]=gripper.open_target_0_1
```

但 0804 的 wrist 视频和 joint state 同步显示：

```text
raw≈0.8 -> 物理张开
raw≈0.0 -> 物理闭合
```

所以模型实际学到的是：

```text
action[6] = physical closed_target_0_1
```

旧 server 又执行：

```text
canonical_closed = 1 - model_action[6]
```

导致物理夹爪命令完全相反。三阶段 shadow 的 gripper MAE：

| frame | 旧 wire 对物理 GT MAE | 修正后 MAE |
|---:|---:|---:|
| 120 | `0.99982` | `0.00018` |
| 160 | `0.78004` | `0.08423` |
| 190 | `0.96371` | `0.00703` |

已增加显式 server 参数：

```text
--fastwam_action_gripper_encoding=closed_0_1
```

不得把该 override 无条件用于其他 checkpoint。

## 4. Held-out shadow 推理结果

样本均来自 cup validation episode 38，未执行动作。

| frame | 阶段 | xyz ADE | xyz FDE | rot mean | rot final |
|---:|---|---:|---:|---:|---:|
| 120 | approach/下降 | 20.7 mm | 35.1 mm | 0.0483 rad | 0.0717 rad |
| 160 | gripper transition | 10.7 mm | 42.4 mm | 0.0261 rad | 0.0794 rad |
| 190 | retreat/lift | 46.7 mm | 60.9 mm | 0.0365 rad | 0.0752 rad |

其他结果：

- 所有预测 action 维度均位于训练 global q01–q99 内；
- 所有预测 action 维度均位于训练 min–max 内；
- 最大单步平移为 4.4 / 9.7 / 12.0 mm；
- 最大单步旋转为 0.0039 / 0.0100 / 0.0087 rad；
- 输出平滑、尺度正常，但一秒 horizon 的 Cartesian FDE 已达到 3.5–6.1 cm。

因此 action loss 收敛不能等价为精确闭环轨迹。当前误差不是异常放大或单位错误，
更像物理任务精度不足。

## 5. Video loss 的解释

同一 golden window 的 joint 预测视频与真实未来视频比较，并加入“保持首帧不动”
persistence baseline：

| frame | generated future PSNR | persistence PSNR | 提升 |
|---:|---:|---:|---:|
| 120 | 19.45 dB | 16.15 dB | +3.30 dB |
| 160 | 19.67 dB | 17.47 dB | +2.20 dB |
| 190 | 20.38 dB | 17.27 dB | +3.11 dB |

模型确实学到一定视觉动态，不是纯复制首帧。但模型配置
`video_dit_config.action_conditioned=false`，视频质量不能证明 action 与视频一致。

## 6. `infer_joint` 与 action-only A/B

同一 frame 160、同一 seed：

- joint 与 action-only 的 32-step xyz/rotation **逐元素一致**；
- action-only 加上 `closed_0_1` override 后，gripper 等于
  `1 - old_joint_wire_gripper`，符合 golden physical target；
- warm inference roundtrip：
  - joint 经 SSH：1.58 s；
  - action-only 远端本地：1.78 s；
- 第二个模型 cold load：220.5 s。

结论：

- 关闭 joint video 不会改变该 checkpoint 的 action；
- 关闭同步 MP4 写 Ceph 的主要收益是减少存储 I/O、产物污染和失败面，不是显著降低
  action 推理时延；
- 32-step horizon 仅 1.067 s，而推理约 1.5–1.8 s，闭环频率受推理限制。

## 7. 实时场景 shadow

实时画面中存在 cup，状态通过 qpos FK 构造 `O_T_LINK8`，没有使用或发布
Cartesian action。

### 当前物理闭爪

```text
gripper raw=0.0
预测 path length=14.8 mm
预测总位移=[-4.1, -9.6, -0.7] mm
模型 physical closed=0.928 -> 0.924
```

### 同画面的物理开爪反事实

```text
预测 path length=30.6 mm
预测总位移=[+7.0, -2.0, +7.7] mm
模型 physical closed=0.000 -> 0.010
```

两种情况下模型都没有生成训练演示中的明显下降接近轨迹。state 除伪手指正好落在
训练 endpoint 的浮点边界外约 `1e-9` 外，没有实质 OOD。

结论：修复 gripper 反转是必要条件，但不足以让当前场景完成抓取；模型仍表现为
近 no-op，存在明显 scene/generalization 或训练能力问题。

## 8. 已实施修复

### 仓库代码

- `src/lerobot/async_inference/configs.py`
  - 增加 `fastwam_action_gripper_encoding` 并严格校验。
- `franka_project/src/franka_eef_pipeline/async_server.py`
  - FastWAM delta/absolute action 支持 `open_0_1` 与 `closed_0_1`；
  - 启动日志记录实际 gripper encoding。
- `franka_project/scripts/export_fastwam_async_fixture.py`
  - 支持 multi-task mix3、采集期相机 topic 置换；
  - 数值验证 adjacent action；
  - 导出 absolute8 golden target 和 pose-frame ledger。
- `franka_project/scripts/send_async_fixture_once.py`
  - 记录 inference roundtrip、request/response/server-send 时间。
- `franka_project/ros_lerobot/.../ros2_runtime.py`
  - `F_T_EE` 使用 reliable + transient-local QoS 获取固定变换。
- `franka_project/ros_lerobot/.../ros2_client.py`
  - FastWAM 禁止 32→25 静默截断；
  - HOLD 不再采集/发布可被 ARM 后执行的旧 plan；
  - ARM 前重新 discard 并显式 disarm 清 plan。

### 本机执行脚本

- `/home/pnp/franka/gripper_follower.py`
  - canonical `closed_0_1` scale 从错误的 `0.4` 改为 `1.0`。
- `/home/pnp/franka/start_real_validation.sh`
  - 显式启动 `model_closed_value:=1.0`。

### 未做的错误修复

没有交换 live camera1/camera2。实时抓图证明当前模型语义顺序已经正确，盲目按采集期
topic 名再次交换会制造新错误。

## 9. 验证

```text
ruff check: passed
bash -n start_real_validation.sh: passed
py_compile gripper_follower.py: passed
pytest: 30 passed
```

定向测试覆盖：

- FastWAM `closed_0_1` override；
- adjacent action golden round-trip；
- 32-step horizon 禁止截断；
- HOLD action delivery 阻断；
- ARM 前 discard/disarm/clear 顺序；
- link8 数学 round-trip；
- 既有 fixture 与 sender 契约。

## 10. 真机 Go/No-Go

当前为 **No-Go**，原因：

1. 正在运行的 `15173` 主进程仍是修复前已加载代码；仅磁盘代码更新不会改变该进程，
   必须按修复命令重启。
2. 当前主机实际运行的是同事的 joint client/gateway，不是目标 Cartesian EEF 栈。
3. 实时 cup shadow 在闭爪与开爪反事实下都接近 no-op，尚未证明模型具备成功接近能力。
4. Python Cartesian gateway 仍按 nominal waypoint index 提前报告 progress，且
   `applied_feedback` 不是真实 controller applied ACK；这会影响下一 observation 和
   gripper 时序。
5. 真机前必须重新启动 follower，确认日志为：

```text
model_closed=1.000, hardware_closed=0.400
```

修复后 server 的推荐控制模式：

```bash
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
  --fastwam_action_gripper_encoding=closed_0_1 \
  --policy_device=cuda
```

客户端必须包含：

```text
--server_address=127.0.0.1:18080
--actions_per_chunk=32
--robot.max_action_chunk_waypoints=32
--robot.policy_eef_frame=link8
```

在重新 ARM 前还需要：

1. 等同事结束并完整切换到目标 Cartesian 栈；
2. 重启修复后的 target server、client、gateway、gripper follower；
3. 在 HOLD 下验证真实 controller feedback 与最终 tracking settle；
4. 再次采集实时 cup shadow；若仍近 no-op，不应通过放宽 safety limit 强行试跑；
5. 最后单独取得现场人员/障碍物清空和 E-stop 可达确认。

## 11. 联调更新（2026-08-07 22:05）

审计完成后已进一步完成：

- 远端 `15173` 已重启为修复后的 action-only server：
  - `fastwam_action_gripper_encoding=closed_0_1`
  - `fastwam_joint_video_inference=false`
  - `step_005000.pt` 已加载并完成 corrected golden fixture 推理；
- 同事的 joint client/gateway 与 `8081` tunnel 已安全下线；
- 本机 Cartesian 栈已重新启动，gripper follower 确认为
  `model_closed=1.000, hardware_closed=0.400`；
- 实际 LeRobot client 已通过 `127.0.0.1:18080` 握手，捕获 `F_T_EE`，获得三个任务；
- 已选择 `pick up the cup`，但网关保持：

```text
armed=false
has_active_plan=false
detail=HOLD (not armed)
```

真实 ROS snapshot 的 source skew：

```text
camera2=29.9 ms
EEF=25.0 ms
qpos=2.0 ms
gripper=1.8 ms
```

真实 observation 到修复模型的一次 shadow 推理：

```text
roundtrip=1.188 s
predicted path length=105.5 mm
predicted gripper.closed range=0.902..0.926
actions_executed_or_published=false
```

期间曾出现一次 FCI `communication_constraints_violation`，导致 controller/broadcaster
inactive 和 EEF 时间戳落后 47 秒；完整停栈并重启后 snapshot 恢复。该事件仍需在真机
ARM 前作为稳定性门禁观察。
