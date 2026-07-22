# C0：PI0 M0 teacher-forced in-sample 评估使用说明

## 1. 唯一评估目标

本 evaluator 只复现 [`F0_PI0_FULL_TRAINING_22OF25.md`](./F0_PI0_FULL_TRAINING_22OF25.md) §6.2 的 M0 teacher-forced in-sample 指标，不构造或计算任何对比基线。

固定输入为：

| 项目 | 固定值 |
|---|---|
| checkpoint | `franka_project/experiments/pi0_full_eef_obs15_act15_v1/20260715T145931Z_pi0_full_eef_f0_obs15_act15_22of25_b33ece09/checkpoints/step-037330` |
| dataset | `franka_project/data/lerobot/franka_current_eef_obs15_act15_v1_22of25_76b839b2` |
| task | `stack the cups` |
| evaluation scope | 全量 22 episodes、7,465 anchors |
| action chunk | 每个 anchor 50 waypoints |
| PI0 flow denoising | 10 inference steps |
| prediction seeds | `1001000 / 2001000 / 3001000` |

推理链路为：

```text
双目 RGB + current EEF state10 + "stack the cups"
                         ↓
               step-037330 PI0 checkpoint
                         ↓
              50 × 7 Cartesian action chunk
                         ↓
             与示教 action chunk 逐点比较
```

这里的 teacher-forced 表示每个 anchor 都使用数据集中真实记录的图像和 current EEF state；预测 action 不会作为下一个 anchor 的 observation，也不执行开环或闭环 rollout。

### 1.1 No-leakage 与 checkpoint 加载

脚本在调用 processor 前单独保存 ground-truth action，并从模型输入中删除：

```text
action
action_is_pad
```

processor 之后允许 LeRobot canonical converter 产生空的 `action=None` 占位，但会断言
`action/action_is_pad` 都没有非空值，并在调用 policy 前显式删除这两个键。因此模型只能看到
双目图像、current EEF state 和语言指令，ground truth 只在推理完成后用于计算指标。

checkpoint 必须 strict load。evaluator 不读取训练日志、W&B、optimizer 或 checkpoint cadence，也不重新执行 F0 训练审计。

## 2. 固定覆盖与聚合规则

正式评估始终遍历数据集中的全部 22 episodes 和 7,465 个有效 anchors，不提供 anchor 抽样或截断选项。

每个固定 prediction seed 分别完成一次全量推理。对每个 seed：

1. 先在每条 episode 内对该 episode 的 anchors 汇总指标；
2. 再对 22 条 episode 等权平均，得到 episode-macro 指标。

随后对三个 seed 的 episode-macro 值计算：

- arithmetic mean，作为最终报告值；
- population variance，即对三个 seed 使用分母 `N=3` 的总体方差。

episode-macro 不按每条 episode 的 anchor 数加权。population variance 只反映固定 checkpoint 的 PI0 推理采样噪声，不是训练 seed 方差。

## 3. F0 §6.2 指标定义

50-waypoint chunk 的数组下标为 `0..49`，在报告中对应 waypoint horizon `1..50`。

### 3.1 Translation

- `translation ADE`：全部 50 个 waypoints 上，预测 EEF xyz 与示教 EEF xyz 的 L2 error 均值，单位 mm；
- `translation FDE`：waypoint 50（数组下标 49）的 EEF xyz L2 error，单位 mm。

ADE 必须包含 waypoint 1（数组下标 0），不使用只统计数组下标 `1..49` 的 future-only 口径。

### 3.2 Rotation

- `rotation geodesic ADE`：全部 50 个 waypoints 上的 SO(3) geodesic error 均值，单位 degree；
- `rotation geodesic FDE`：waypoint 50（数组下标 49）的 SO(3) geodesic error，单位 degree。

rotation-vector 不能直接逐元素相减。实现需要通过 exponential map 还原旋转，再计算 SO(3) geodesic distance。

### 3.3 Gripper

- `gripper MAE`：全部 50 个 waypoints 上的绝对误差均值；
- `gripper FDE`：waypoint 50（数组下标 49）的绝对误差。

### 3.4 固定 horizon

除 ADE/MAE 和 FDE 外，还要报告以下五个 waypoint 的逐点误差：

| waypoint horizon | 数组下标 | 相对 anchor 的时间（15 Hz） |
|---:|---:|---:|
| 1 | 0 | 0.067 s |
| 5 | 4 | 0.333 s |
| 10 | 9 | 0.667 s |
| 25 | 24 | 1.667 s |
| 50 | 49 | 3.333 s |

每个 horizon 分别报告 translation L2 error（mm）、rotation geodesic error（degree）和 gripper absolute error，并使用与主指标相同的 episode-macro、3-seed mean 与 population variance 聚合。

## 4. F0 §6.2 参考结果

使用上述唯一 checkpoint、全量数据和固定 seeds 时，F0 §6.2 的最终 3-seed mean 为：

| 指标 | ADE/MAE | FDE |
|---|---:|---:|
| translation | **12.4826 mm** | **16.1923 mm** |
| rotation geodesic | **1.79266°** | **2.27081°** |
| gripper | **0.0190087** | **0.0290417** |

固定 horizon 的 episode-macro、3-seed mean 为：

| waypoint horizon | 1 | 5 | 10 | 25 | 50 |
|---:|---:|---:|---:|---:|---:|
| translation (mm) | 3.4265 | 8.4006 | 10.5291 | 13.4462 | 16.1923 |
| rotation (deg) | 0.5861 | 1.3715 | 1.6278 | 1.9019 | 2.2708 |
| gripper absolute error | 0.01434 | 0.01468 | 0.01590 | 0.01647 | 0.02904 |

F0 §6.2 明确列出的 population variance 参考值包括：

| 指标 | ADE variance | FDE variance |
|---|---:|---:|
| translation | `0.008877 mm²` | `0.006317 mm²` |
| rotation geodesic | `0.000181 deg²` | `0.000523 deg²` |

evaluator 应从三个 seed 的实际结果计算 mean 和 population variance；这些参考数字不得作为 evaluator 的硬编码输出。

## 5. 环境准备

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

source /ytech_milm_intern/tujiahang/.bashrc
conda activate lerobot
unset LD_LIBRARY_PATH

export PYTHONPATH=franka_project/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

如果默认 Hugging Face datasets cache 没有写权限，可以额外指定：

```bash
export HF_DATASETS_CACHE=/tmp/franka_direct_capability_hf_datasets
```

## 6. 全量评估命令

```bash
python franka_project/scripts/eval_pi0_direct_capability.py \
  --checkpoint-dir \
    franka_project/experiments/pi0_full_eef_obs15_act15_v1/20260715T145931Z_pi0_full_eef_f0_obs15_act15_22of25_b33ece09/checkpoints/step-037330 \
  --dataset-root \
    franka_project/data/lerobot/franka_current_eef_obs15_act15_v1_22of25_76b839b2 \
  --output-json \
    franka_project/artifacts/direct_capability/c0_step037330_m0_full.json \
  --device cuda:0 \
  --prediction-seeds 1001000 2001000 3001000 \
  --batch-size 4 \
  --workers 2 \
  --num-inference-steps 10 \
  --save-predictions
```

该 CLI 只保留以下参数：

```text
--checkpoint-dir
--dataset-root
--output-json
--device
--prediction-seeds
--batch-size
--workers
--num-inference-steps
--save-predictions
```

全量覆盖由 evaluator 固定执行，不需要额外传入采样参数。

## 7. 输出

假设传入：

```text
--output-json .../c0_step037330_m0_full.json
```

脚本生成：

```text
c0_step037330_m0_full.json             完整结构化指标与覆盖元数据
c0_step037330_m0_full.md               自动生成的 F0 §6.2 简要结果表
c0_step037330_m0_full.predictions.npz  使用 --save-predictions 时保存的逐预测结果
```

JSON 和 Markdown 只报告：

- 全量覆盖与固定推理配置；
- translation、rotation geodesic、gripper 的 ADE/MAE 与 FDE；
- waypoint horizon `1 / 5 / 10 / 25 / 50` 的逐点误差；
- 上述指标的 episode-macro、3-seed mean 与 population variance。

JSON 中 §6.2 数值的固定入口为：

```text
metrics.mean
metrics.population_variance
```

不输出对比基线、base checkpoint 结果、micro 聚合、分位数、tolerance/threshold 命中率、gripper binary accuracy 或 future-only ADE。

## 8. 结果边界

该 evaluation 只衡量：

```text
在训练数据的真实 recorded observation 条件下，step-037330 能否复现示教 Cartesian action chunk。
```

这是 teacher-forced in-sample reconstruction，不是 held-out validation，也不是闭环能力评估。即使 ADE/FDE 很低，也不能回答：

- 新杯子位置或新相机画面是否能够泛化；
- 连续执行 action 后是否产生开环或闭环漂移；
- IK、关节限制和碰撞检查是否通过；
- 真机是否能成功完成 `stack the cups`。

最后一项只能通过另行批准的真机闭环 rollout 和明确的安全方案评估。
