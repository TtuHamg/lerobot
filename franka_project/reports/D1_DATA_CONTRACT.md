# D1 Frank3 EEF 数据契约

## 结论

D1 的表示、时间对齐和双频 schema 已冻结。原始 25 条 manifest 中有 3 条缺失 MCAP，因此仍不能宣称 25/25 可转换；根据用户对同机实验的授权，正式 D3 输入已通过独立的 `franka_scope_22of25_v1` 冻结为当前存在的 22 条 strict-PASS MCAP。原始 manifest 保持只读且未修改。本阶段不按 A/B 划分，不设独立 validation/test episode；后续 `val_loss` 只使用从训练 anchors 中固定随机抽取的子集，它只是训练分布内的诊断指标，不是泛化估计。

## 冻结范围

| 项目 | v1 决定 |
|---|---|
| 任务指令 | `stack the cups` |
| observation clock | cam1 MCAP `log_time_ns`，标称 15 Hz |
| 对齐 | `latest_not_after`，先裁剪到 cam1/cam2/EEF/gripper 公共覆盖区间 |
| 缺失相机 | 不填零图，不插值 |
| policy state | 10D = xyz + rotation-6D + normalized gripper |
| policy action | 7D = base-frame delta xyz + body-frame SO(3) rotation vector + target gripper |
| qpos | 不对 policy 可见，仅保留为 audit sidecar |
| 训练数据划分 | 不做 A/B、validation 或 test episode split |
| 统计和归一化 | 对真正送入 policy 的 10D state/7D action 计算 `min/max/mean/std/count/q01/q10/q50/q90/q99`；15 Hz 和 30 Hz 分开统计 |

## EEF frame 与 TCP 语义

22 条可读 MCAP 中，`current_pose.header.frame_id` 全部为 `base`。官方 `franka_robot_state_broadcaster` 源码将 `current_pose` 直接发布为 robot state 里的 `o_t_ee`；libfranka 对 `O_T_EE` 的定义是“base frame 中测得的 end-effector pose”。因此本数据契约中的 EEF 是：

```text
base -> configured EE frame 的 measured pose
```

采集 bag 没有保存数值化的 `F_T_EE`/TCP 配置，所以离线数据本身无法进一步证明 configured EE 的原点恰好位于夹爪的哪个物理点。这不阻断 D2 几何 round-trip，但真机 rollout 前必须满足以下任一条：

- 训练和部署时使用完全相同的 configured EE frame；
- 或者明确给出采集 frame 与部署 TCP 之间的固定变换并在 processor/controller 中统一转换。

官方源码参考：[franka_ros2 broadcaster](https://github.com/frankarobotics/franka_ros2/blob/humble/franka_robot_state_broadcaster/src/franka_robot_state_broadcaster.cpp)、[libfranka RobotState](https://frankarobotics.github.io/libfranka/latest/structfranka_1_1RobotState.html)。

## 时间对齐与有效 anchor

公共区间内共有 8,815 个 cam1 observation anchors。冻结的 source-age 上限是 cam2 100 ms、EEF 50 ms、gripper 10 ms；其中 18 个 observation（0.204%）因 source 过旧被判无效。EEF 与 gripper 都满足上限，这 18 个全部来自 cam2 的局部 gap。

只采样能提供完整 50 个 camera interval horizon 的 anchor，且 horizon 中任一 observation 无效时不采样。在当前 22 条可读数据上：

- 如果只考虑 episode 尾部，名义上有 7,715 个完整 horizon anchors；
- 加上 100 ms cam2 age gate 后保留 7,465 个，减少 250 个（3.24%）。

用户已决定 v1 不专门修正“追帧/不规则 dt”；本契约不重采样或修改真实时间，只是排除明显过旧的 cam2 观测并保留原始 source timestamp 供审计。

## 双频 action schema

### obs15/action15

- 每个 cam1 anchor 一个 observation；
- action slots 是 `c_(t+1) ... c_(t+50)` 的 future measured EEF/gripper；
- `K=50`，物理 horizon 约 3.33 s；
- 作为第一轮训练版本。

### obs15/action30

- camera 仍为 15 Hz，不复制或伪造 30 Hz 图像；
- 对每个 camera interval 依次取 midpoint 和 endpoint 的 future measured EEF/gripper；
- `K=100`，与 15 Hz 版本保持相同约 3.33 s 物理 horizon；
- 只转换、统计和 QA，本轮不训练。

30 Hz 的每个 endpoint 必须与 15 Hz 对应 target 的绝对 pose、gripper 和 source timestamp 逐项相等。D0 在全部 22 条可读 episode 上已通过该对账。

## Gripper 和归一化

原始夹爪关节为 `robotiq_85_left_knuckle_joint`，22 条可读数据总范围为 `[0.0, 0.789407]`。图像抽查显示 raw `0.0` 时夹爪张开，raw 约 `0.78` 时夹爪夹持杯子，因此冻结线性映射：

```text
normalized_gripper = clip(raw / 0.8, 0, 1)
0 = open, 1 = closed
```

state/action 在完成几何转换后分别按 mean/std 归一化。不使用 on-disk 8D absolute carrier 的默认 action stats 代替 7D policy action stats，也不混用 15 Hz/30 Hz 两版 action stats。

## 22of25 范围与下一个 gate

| 问题 | 证据 | 影响 | 级别/信心 |
|---|---|---|---|
| 3 条 PASS 缺原始 MCAP | manifest=25，可读=22；缺失 ID 见配置 | 从 `22of25_v1` 显式排除；仍禁止宣称 25/25 | Critical for original scope / high |
| 数值 TCP transform 不在 bag 中 | 只能证明 `base -> configured EE` | 不影响离线 D2；真机必须使用同一 EE 配置 | High for deployment / high |
| cam2 局部 gap | 18/8,815 observations 超过 100 ms | 丢弃 250/7,715 full-horizon anchors | Medium / high |

D3 已按第二种方式解锁：派生 manifest 保持 source manifest 原始顺序，纳入 22 条现存 MCAP，逐文件记录 byte size 与完整 SHA-256，并显式记录 3 条缺失项。D3 只能读取该派生 manifest；原始 25 条 manifest 不作为 D3 输入。

该授权基于同一机器并复用采集时 configured EE/TCP 的实验约束，不代表本阶段已经授权真机 rollout。若任一 MCAP 的 size/hash 改变，范围锁失效，必须重新冻结。

## 可复现性

- 机器可读契约：`configs/data/franka_current_eef_contract_v1.yaml`
- common config SHA256：`1d9c23d8c54aed8a0870c37d5dfc26aab95164f65c041a67a781cbe923cc4d1a`
- manifest SHA256：`d2ba4dc819eafd0163ed6673f53ea6e6cc7d492b075978c4db925c5b68dcf38b`
- D0 JSON：`artifacts/raw_audit/d0_raw_audit.json`
- 22of25 scope config：`configs/data/franka_scope_22of25_v1.yaml`
- scope config SHA256：`2dd95dcd5b9b953abd816d7863af1b38d92275923af884eadcbd07527273bd56`
- 22of25 derived manifest：`manifests/frank3_train_passed_22of25_v1.yaml`
- derived manifest SHA256：`76b839b29840714dc6a0c58261d2709b25dfd846fe7f13761787fd6d02b17a77`
- scope content SHA256：`86402a4138002b1ff6e02d95e7f434c5ef3f658c362fb88253ff8b1c44a69461`
- scope lock：`artifacts/scope/d1_scope_22of25_v1.lock.json`
- scope lock SHA256：`da999ae54a83c36bb1d977ca41623533d8806b29e39c0b9cff2ded0145c705a9`
- scope report：`reports/D1_SCOPE_22OF25_V1.md`
