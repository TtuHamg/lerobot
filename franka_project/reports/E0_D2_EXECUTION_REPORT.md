# Frank3 E0–D2 执行报告

## 结论

已严格完成批准的首批 E0–D2，未启动 D3 全量转换或 pi0 训练。E0 和 D2 通过，D1 已冻结并可支持 D2；D0 发现 manifest 中 3 条 PASS 缺失原始 MCAP，因此按计划的停止规则禁止进入 D3。

| Phase | 状态 | 主要结果 |
|---|---|---|
| E0 | PASS | Conda、官方 loader、双 A800 BF16/cuDNN/NCCL、pi0_base、W&B scalar 记录均通过 |
| D0 | BLOCKED | 25 条 manifest 只有 22 条原始 MCAP 可读 |
| D1 | CONDITIONAL PASS | EEF/time/gripper/15–30 Hz/stats 契约已冻结，可做 D2，不解锁 D3 |
| D2 | PASS | 2 条 episode、824 anchors，双频时间、几何、endpoint 和 normalization 验证全部通过 |
| D3/训练 | NOT STARTED | 没有转换全量 dataset，没有加载 pi0 训练，没有生成 checkpoint |

## E0 环境

固定启动前缀：

```bash
source /ytech_milm_intern/tujiahang/.bashrc >/dev/null 2>&1
conda activate lerobot
unset LD_LIBRARY_PATH
```

`unset LD_LIBRARY_PATH` 不能省略：系统 cuDNN 9.11 与 PyTorch 需要的 cuDNN 9.19 冲突。清除后，2 × A800 80GB 的 BF16 Conv2d 前反向和 NCCL all-reduce 通过。

官方 loader 完整解码 `run_20260714_212901_00063`：T=588，双目图像均为 `[588,480,640,3] uint8`，qpos/gripper/EEF 为 `[588,7]/[588,1]/[588,7]`。

W&B smoke：[`ttuhamg/franka_pi0_full_eef/syujv5nl`](https://wandb.ai/ttuhamg/franka_pi0_full_eef/runs/syujv5nl)，仅记录 `e0/smoke_scalar=1`，`logged_artifact_count=0`，无模型/checkpoint 上传。

## D0 原始数据完整性

Manifest 声明 25 条 PASS、698.374 s、10,308 个 cam1 帧。当前只有 22 条 MCAP 可读：608.013 s、cam1=8,987、cam2=9,007。

缺失的 3 条：

- `run_20260714_202237_00032`：29.150 s / 435 cam1。
- `run_20260714_175543_00008`：45.769 s / 685 cam1。
- `run_20260714_165338_00005`：15.442 s / 201 cam1。

搜索只找到对应 `viz_cache`，没有原始 MCAP。缓存不包含可复现的全部原始消息和 timestamp，不作为 raw 替代。

22 条可读数据的其他检查稳定：

- topic count 与 manifest 逐条一致，图像均为 480×640 RGB8；
- EEF/gripper/qpos 无 NaN/Inf 或 name/order 漂移；
- EEF quaternion 最大 norm 误差约 `2.03e-8`；
- 公共区间 8,815 个 cam1 anchors，8,797 个 observation 通过 age gate；
- 完整 50-camera-interval horizon anchors 在 15/30 Hz 版本中均为 7,465；
- 18 个 invalid observation 全部是 cam2 局部 stale；EEF/gripper/qpos 均没超过各自 gate。

## D1 冻结契约

- task instruction：`stack the cups`。
- policy state：10D `xyz + rotation6D + gripper`。
- policy action：7D `base-frame delta xyz + body-frame SO(3) rotvec + target gripper`。
- qpos：只作 audit sidecar，不输入 policy。
- 对齐：cam1 MCAP log time + `latest_not_after` + cam1/cam2/EEF/gripper 公共区间；不填零图。
- source-age gate：cam2/EEF/gripper/qpos audit = 100/50/10/10 ms。
- gripper：`clip(raw/0.8,0,1)`，0=open，1=closed。
- split：不做 A/B、held-out validation 或 test；后续从训练 anchors 固定抽样的 loss 只是 in-distribution monitor。
- stats：对真正 model-visible 10D/7D 统计 `min/max/mean/std/count/q01/q10/q50/q90/q99`，15/30 Hz 分开保存。

`current_pose.header.frame_id` 在 22 条数据中都是 `base`。官方 broadcaster 将 `current_pose` 发布为 `state.o_t_ee`，libfranka 定义其为 base 中 measured configured EE pose。但 bag 不含数值化 `F_T_EE/NE_T_EE`，真机部署需复用同一 configured EE/TCP 或显式转换。

## D2 Cartesian pipeline spike

Episode：`run_20260714_212901_00063` 和 `run_20260714_211745_00057`。总计 824 个有效完整 horizon anchors。

| Profile | State shape | Action shape | Rotation RT | Max theta | Action norm RT |
|---|---:|---:|---:|---:|---:|
| obs15/action15 | 824×10 | 824×50×7 | 5.162e-8 rad | 0.979978 rad | 1.110e-16 |
| obs15/action30 | 824×10 | 824×100×7 | 5.162e-8 rad | 0.981708 rad | 1.110e-16 |

验证结果：

- 15 Hz 第一/最后 action 精确为 `t+1/t+50`；
- 30 Hz 第一 action 是首个 camera interval midpoint，最后是 `t+50`；
- 所有 EEF/gripper source timestamp 均不晚于 target；
- 30 Hz 的 endpoint value/source timestamp 与 15 Hz 对应项 bit-exact；
- position/gripper encode→decode 最大误差均为 0；
- 所有 model arrays 和 stats finite，无跨 episode 或 padded target。

单元测试：`21 passed`；唯一 warning 来自外部 `mcap_ros2` 包的 deprecation docstring。

## 停止边界和下一步

本批没有解码/写入全量图像 dataset，没有启动 pi0 forward/backward 或训练，没有 checkpoint，也没有 W&B model artifact。

D3 只能以下列方式之一解锁：

1. 恢复 3 条原始 MCAP，重跑 D0 并达到 25/25；
2. 明确批准创建新的、可追溯的 `22of25` 派生 manifest/dataset version。

解锁后的既定顺序是 D3 双版转换 → D4 QA → 仅 obs15/action15 的 S0/S1 → 复核后再启动 pi0 全参数正式训练。

