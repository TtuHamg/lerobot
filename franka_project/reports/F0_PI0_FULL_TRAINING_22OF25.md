# Franka 22-of-25 数据转换、PI0 全参数训练与 M0 报告

> 最终状态：`F0_PASS / M0_COMPLETE`  
> 生成时间：2026-07-16 UTC  
> 任务指令：`stack the cups`

## 1. 结论

本轮已按用户确认的范围完成：只使用 22 条当前完整的 Franka 轨迹，不恢复缺失的 3 条；由于采集与部署使用同一台机器和同一套 EE/TCP 配置，本轮不把 bag 中缺失 `F_T_EE/NE_T_EE` 作为阻塞项。

- `obs15/action15` 与 `obs15/action30` 两版数据均已转换、计算 statistics 并通过 QA。
- 首轮只在 `obs15/action15` 上训练；相机与 action 均按 15 Hz，action chunk 为 50 步。
- PI0 从本地 `lerobot/pi0_base` 初始化，使用非 LoRA/PEFT 的全参数配置训练 37,330 steps（10 effective epochs）。
- 固定 747-anchor、训练集内部 monitor loss 从 `0.57267639` 降至 `0.05408542`，下降 `90.56%`；best 与 last 均为 step 37,330。
- M0 完成全部 22 episodes、7,465 anchors、3 个固定推理 seeds 的 teacher-forced in-sample 重建，所有输出均为 finite。
- W&B run 已结束且记录完整 loss 曲线；没有上传模型 checkpoint，`model_artifacts=0`。
- 这些结果证明当前数据/processor/训练链路可学习并能拟合采集轨迹，但不等同于 held-out 泛化、闭环稳定性或真机叠杯成功率。

## 2. 数据范围与版本

原始数据来自：

```text
/ytech_milm/collect_data_103/frank3/20260714/DATA
```

冻结的执行范围为 25 条 source manifest 中当前存在且完整的 22 条，scope content SHA256：

```text
86402a4138002b1ff6e02d95e7f434c5ef3f658c362fb88253ff8b1c44a69461
```

| 数据集 | observation/action | chunk | episodes | 主表 rows | 有效 anchors | model action stats count | 首轮训练 |
|---|---:|---:|---:|---:|---:|---:|---|
| `franka_current_eef_obs15_act15_v1_22of25_76b839b2` | 15/15 Hz | 50 | 22 | 8,793 | 7,465 | 373,250 | 是 |
| `franka_current_eef_obs15_act30_v1_22of25_76b839b2` | 15/30 Hz | 100 | 22 | 8,793 | 7,465 | 746,500 | 否，仅转换与 QA |

两版数据的相机都保持约 15 Hz。action30 版通过 sidecar 保存每个相机间隔内约 30 Hz 的 EEF waypoint，没有把图像复制成 30 Hz。

数据目录：

```text
franka_project/data/lerobot/franka_current_eef_obs15_act15_v1_22of25_76b839b2
franka_project/data/lerobot/franka_current_eef_obs15_act30_v1_22of25_76b839b2
```

每条 episode 都先裁剪到 cam1、cam2、EEF 和 gripper 的公共有效时间区间；没有使用零图填充，也没有跨 episode 补帧。

## 3. 模型可见的 EEF 表示

### 3.1 Proprioception：10D current measured state

```text
[current_xyz(3), current_rotation_6d(6), current_gripper(1)]
```

- `xyz` 为当前实测 EEF 平移。
- 旋转使用连续的 6D rotation representation，避免 Euler angle 奇异点和 quaternion 双覆盖问题。
- 不向 PI0 输入 qpos；qpos 只保留用于数据审计和未来底层控制。

### 3.2 Action：7D anchor-relative Cartesian waypoint

```text
[delta_xyz(3), body_delta_rotvec(3), future_gripper(1)]
```

- 平移与旋转都相对当前 anchor EEF 构造。
- 旋转 action 是 3D axis-angle/rotation-vector，而不是 6D rotation matrix。
- 标签来自 future measured EEF，是现有数据下的监督代理，不冒充遥操作端的 desired command。
- action15 每个 anchor 预测 50 个 waypoint；action30 每个 anchor 对应 100 个 waypoint。

## 4. Statistics 与归一化

两版数据分别生成 LeRobot `meta/stats.json` 以及 PI0 实际使用的 `meta/pi0_eef_stats.json`，不跨版本混用。

PI0 processor 的归一化映射为：

```yaml
VISUAL: IDENTITY
STATE: MEAN_STD
ACTION: MEAN_STD
```

action15 的 model-visible stats 关键对账值：

| 特征 | 维度 | count |
|---|---:|---:|
| `observation.state` | 10 | 7,465 |
| `action` | 7 | 373,250 = 7,465 × 50 |

action30 的 action count 为 `746,500 = 7,465 × 100`。训练 checkpoint 内同时保存 processor 和本次 EEF stats，以避免部署时误用其他数据集的归一化参数。

## 5. F0 全参数训练

### 5.1 配置

| 项目 | 值 |
|---|---|
| base model | 本地 `lerobot/pi0_base` snapshot |
| PEFT/LoRA | `null` / 未使用 |
| vision encoder | 未冻结 |
| train expert only | `false` |
| precision | BF16 |
| gradient checkpointing | 开启 |
| optimizer | AdamW |
| peak/final LR | `1e-5 / 1e-6` |
| weight decay | `0.01` |
| seed | `1000` |
| GPU | 2 × A800，DDP |
| global batch | 2 |
| steps per epoch | 3,733 |
| total | 37,330 steps / 10 effective epochs |

配置文件：`franka_project/configs/train/pi0_full_eef_f0_15hz.yaml`。

训练图中有 `3,238,048,528` 个唯一 `requires_grad` 参数、776 个参数 tensors，未插入 LoRA adapter，也未冻结视觉编码器。首次 backward 中 770 个 tensors 得到 finite gradients；另外精确登记的 6 个 PaliGemma final-prefix tensors（`104,861,696` 个 scalars）虽然保持 trainable，但因 PI0 suffix-only action loss 的结构永久不在 action-loss 路径上。该事实已写入 checkpoint ledger，而不是被静默忽略。

### 5.2 训练结果

运行目录：

```text
franka_project/experiments/pi0_full_eef_obs15_act15_v1/
20260715T145931Z_pi0_full_eef_f0_obs15_act15_22of25_b33ece09
```

训练先完成双卡 step `0→2` preflight，再从同一 immutable run 的 step 2 恢复到 37,330；两个 invocation 均 `exit_code=0`。最终本地 checkpoint：

```text
.../checkpoints/step-037330
```

最后一个记录到的 minibatch train loss 为 `0.01757070`。用于 checkpoint 比较的是固定 747-anchor 的 train-internal monitor loss：

| step | effective epoch | monitor loss |
|---:|---:|---:|
| 0 | 0 | 0.57267639 |
| 2 | preflight | 0.57278287 |
| 3,733 | 1 | 0.23009648 |
| 7,466 | 2 | 0.15960879 |
| 11,199 | 3 | 0.12414087 |
| 14,932 | 4 | 0.10117570 |
| 18,665 | 5 | 0.08290479 |
| 22,398 | 6 | 0.07361592 |
| 26,131 | 7 | 0.06595752 |
| 29,864 | 8 | 0.05972554 |
| 33,597 | 9 | 0.05645756 |
| 37,330 | 10 | **0.05408542** |

下降过程总体连续，最终 step 同时是最低 monitor loss checkpoint 和 last checkpoint。

### 5.3 W&B

```text
entity/project/run: ttuhamg/franka_pi0_full_eef/jkrz5th0
state: finished
train metric points: 748
monitor/loss points: 12
model_artifacts: 0
```

W&B 在 run 结束后自动生成了一个 51,454-byte 的 `wandb-history` 指标历史 artifact；它只包含曲线 parquet，不是模型权重。门禁确认总 artifact 数为 1、允许的 history artifact 数为 1、模型 artifact 数为 0，因此满足“不上传 model checkpoint”的要求。

## 6. M0 teacher-forced in-sample 评估

### 6.1 方法与覆盖

- 评估 checkpoint：step 37,330（best train monitor = last）。
- 全量范围：22 episodes、7,465 anchors。
- 每个 anchor：50-step action chunk。
- PI0 flow denoising：10 inference steps。
- 固定 prediction seeds：`1001000 / 2001000 / 3001000`。
- 汇总主指标：先在每条 episode 内汇总，再对 22 条 episode 等权平均；最后对 3 个 prediction seeds 取均值。
- `ADE` 为 50 个 waypoint 的平均轨迹误差，`FDE` 为第 50 个 waypoint 的误差。

### 6.2 最终 checkpoint 主指标

| 指标 | ADE/MAE | FDE |
|---|---:|---:|
| translation | **12.4826 mm** | **16.1923 mm** |
| rotation geodesic | **1.79266°** | **2.27081°** |
| gripper | **0.0190087** | **0.0290417** |

分 horizon 的 episode-macro、3-seed mean：

| waypoint horizon | 1 | 5 | 10 | 25 | 50 |
|---:|---:|---:|---:|---:|---:|
| translation (mm) | 3.4265 | 8.4006 | 10.5291 | 13.4462 | 16.1923 |
| rotation (deg) | 0.5861 | 1.3715 | 1.6278 | 1.9019 | 2.2708 |
| gripper MAE | 0.01434 | 0.01468 | 0.01590 | 0.01647 | 0.02904 |

三个推理 seed 的 population variance 较小，例如 translation ADE/FDE variance 为 `0.008877/0.006317 mm²`，rotation ADE/FDE variance 为 `0.000181/0.000523 deg²`。这只衡量 PI0 推理采样噪声，不是训练 seed 方差。

### 6.3 随 epoch 的 Cartesian 重建趋势

以下比较使用同一 747-anchor monitor 子集和同一 prediction seed：

| step | monitor loss | translation ADE/FDE (mm) | rotation ADE/FDE (deg) | gripper MAE |
|---:|---:|---:|---:|---:|
| 3,733 | 0.23010 | 45.986 / 64.647 | 6.723 / 9.388 | 0.11464 |
| 7,466 | 0.15961 | 34.924 / 48.832 | 5.098 / 6.624 | 0.07164 |
| 11,199 | 0.12414 | 26.894 / 36.742 | 3.962 / 4.962 | 0.06355 |
| 14,932 | 0.10118 | 22.458 / 29.869 | 3.206 / 4.092 | 0.04415 |
| 18,665 | 0.08290 | 19.303 / 25.203 | 2.759 / 3.518 | 0.03994 |
| 22,398 | 0.07362 | 17.658 / 23.233 | 2.496 / 3.175 | 0.03231 |
| 26,131 | 0.06596 | 15.237 / 19.803 | 2.204 / 2.825 | 0.02524 |
| 29,864 | 0.05973 | 13.949 / 18.466 | 1.999 / 2.618 | 0.02180 |
| 33,597 | 0.05646 | 12.920 / 17.501 | 1.871 / 2.454 | 0.01966 |
| 37,330 | 0.05409 | **12.533 / 17.114** | **1.814 / 2.448** | **0.01816** |

monitor loss 与 Cartesian 指标同步改善，支持“数据标签和 processor 确实被模型学到”的判断。

### 6.4 数值与经验 envelope 诊断

| 诊断 | episode-macro、3-seed mean |
|---|---:|
| nonfinite action element/waypoint fraction | 0 / 0 |
| rotvec norm > π waypoint fraction | 0 |
| decoded xyz 超出训练 state min/max fraction | 0.002474（0.247%） |
| gripper 输出超出 `[0,1]` waypoint fraction | 0.169776（16.98%） |
| 任一 action 维超出训练 min/max waypoint fraction | 0.251861（25.19%） |

后两项说明模型 raw output 不能直接视为真机安全命令。真机部署前至少需要 gripper clamp、Cartesian 增量/工作空间限幅、控制器速率与延迟检查，以及 Franka 侧碰撞、关节限位和 IK/控制器安全门禁。这些经验 envelope 指标本身也不能证明无碰撞、无奇异点或满足关节限制。

## 7. 审计与验证结果

- M0 最终状态：`M0_COMPLETE`，最终 JSON 使用原子发布。
- 32/32 全量 inference shards 完成，覆盖 `7,465 × 3 = 22,395` 个 anchor-seed predictions。
- 10/10 epoch milestone monitor artifacts 完成。
- 最终 checkpoint 文件尺寸、safetensors header 与完整文件 SHA256 复核通过。
- W&B run state、748 个 train metric points、12 个 monitor points 和 `model_artifacts=0` 复核通过。
- 最终模型曾执行 strict 3B reload、真实数据 no-grad forward；forward loss 为 `0.01568851`，processor 输出 shape 为 state `[1,10]`、action `[1,50,7]`。
- 项目完整测试：`125 passed, 1 warning`；warning 来自 `mcap_ros2` 上游 deprecation，不影响本轮结果。

M0 报告 SHA256：

```text
e017007621981ca16b70ba13e2cf524b5a6a5ddfae0a80e17fd62c82ccd7f55d
```

## 8. 主要产物

| 产物 | 路径 |
|---|---|
| 完整计划与决策历史 | `franka_project/PLAN_DATA_PROCESSING_AND_PI0_FULL_TRAINING.md` |
| action15 数据 | `franka_project/data/lerobot/franka_current_eef_obs15_act15_v1_22of25_76b839b2` |
| action30 数据 | `franka_project/data/lerobot/franka_current_eef_obs15_act30_v1_22of25_76b839b2` |
| F0 配置 | `franka_project/configs/train/pi0_full_eef_f0_15hz.yaml` |
| F0 run | `franka_project/experiments/pi0_full_eef_obs15_act15_v1/20260715T145931Z_pi0_full_eef_f0_obs15_act15_22of25_b33ece09` |
| 最终 checkpoint | 上述 run 下 `checkpoints/step-037330` |
| M0 JSON | `franka_project/artifacts/m0/20260715T145931Z_pi0_full_eef_f0_obs15_act15_22of25_b33ece09/m0_report.json` |
| 转换脚本 | `franka_project/scripts/convert_to_lerobot.py` |
| 训练脚本 | `franka_project/scripts/train_pi0_full.py` |
| M0 evaluator | `franka_project/scripts/eval_pi0_f0_m0.py` |
| F0 verifier | `franka_project/scripts/verify_pi0_f0_run.py` |

最终 checkpoint 约 21 GB，两个转换后数据集各约 386–387 MB，完整 M0 artifact 约 56 MB。

## 9. 复核命令

```bash
cd /m2v_intern/tujiahang/Projects/lerobot
source /ytech_milm_intern/tujiahang/.bashrc
conda activate lerobot
unset LD_LIBRARY_PATH
export PYTHONPATH=franka_project/src

python franka_project/scripts/verify_pi0_f0_run.py \
  --run-dir franka_project/experiments/pi0_full_eef_obs15_act15_v1/20260715T145931Z_pi0_full_eef_f0_obs15_act15_22of25_b33ece09 \
  --verify-file-hashes --check-wandb

python franka_project/scripts/eval_pi0_f0_m0.py \
  --run-dir franka_project/experiments/pi0_full_eef_obs15_act15_v1/20260715T145931Z_pi0_full_eef_f0_obs15_act15_22of25_b33ece09 \
  --mode finalize

pytest -q franka_project/tests
```

若要从头训练，使用两卡启动：

```bash
torchrun --standalone --nproc_per_node=2 \
  franka_project/scripts/train_pi0_full.py \
  --config franka_project/configs/train/pi0_full_eef_f0_15hz.yaml
```

若要在最终 `step-037330` 上保留 AdamW/RNG/data offset 并继续训练，使用新增的
`F0-CONT-15` 模式；完整合同与命令见
[`F0_CONTINUATION_MODE.md`](./F0_CONTINUATION_MODE.md)。

## 10. 结果边界与后续门槛

本轮没有独立 validation/test split；全部 22 条轨迹进入训练池，所谓 monitor 也只是训练池固定子集。因此：

- `has_held_out_validation=false`
- `has_test_set=false`
- `real_robot_rollout_performed=false`
- `generalization_claim=false`

当前可以合理得出的结论是：15/15 Hz EEF 表示、normalization、PI0 全参数训练和 checkpoint/evaluator 链路工作正常，模型对这 22 条采集轨迹具有明显拟合能力。下一阶段若要判断是否真的能完成 `stack the cups`，必须另行制定并批准真机部署与安全验证方案；action30 训练也应作为独立实验，而不是用本轮 action15 结果替代。
