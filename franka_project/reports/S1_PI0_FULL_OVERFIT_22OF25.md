# S1 PI0 全参数小样本过拟合验收：22of25 / obs15-act15

- 结论：**PASS — 小样本可学习性与 checkpoint/reload 工程门禁通过**
- Run：`20260715T142336Z_pi0_full_eef_s1_overfit_obs15_act15_22of25_b8bc6628`
- Final checkpoint：`step-000300`
- W&B：[ttuhamg/franka_pi0_full_eef/7gnnlevp](https://wandb.ai/ttuhamg/franka_pi0_full_eef/runs/7gnnlevp)
- Evaluator 输出时间：`2026-07-15T14:57:47.035052Z`
- 报告生成时间：`2026-07-15T15:00:10Z`

## 技术结论

- S1 从同一 `lerobot/pi0_base` 独立完成 `300` 个 optimizer steps，使用非 LoRA 的 `3,238,048,528` 参数全训练图。固定 in-train monitor loss 从 `0.35633501` 降至 `0.14697757`，下降 `58.75%`。
- 固定噪声 action-chunk 评估中，step 300 相对 base 的 translation ADE 从 `56.447842 mm` 降至 `35.498271 mm`，trained/base=`0.628869`；gripper MAE 从 `0.567802` 降至 `0.284258`，trained/base=`0.500628`。
- Rotation ADE 从 `5.215184°` 降至 `4.968256°`，trained/base=`0.952652`，仅改善 `4.73%`。该改善明显弱于 translation/gripper，且逐 anchor 方向不一致；S1 PASS 不应解释为旋转预测已经充分拟合。
- step 300 第二次 strict load 与磁盘 saved processors reload 后，normalized/raw action 的最大绝对差均为 `0.0`。Base、trained、trained-reload 均 `missing_keys=[]`、`unexpected_keys=[]`，使用相同 action15 effective stats。
- W&B run `7gnnlevp` 已 `finished`，`logged_artifacts=0`。S1 自身是一次 `start_step=0 → 300` 的完整 invocation，`resume=false`；checkpoint resume 能力由此前 S0 的 step `2 → 3` 独立验证，不能把 S0 resume 写成 S1 resume。

因此，S1 支持“当前 state/action 表示、normalization、PI0 全参数图和保存/重载链路能够在小训练池上产生可测的拟合改善”。它不支持未见数据泛化、22 条全量训练已完成或真机闭环成功的结论。下一阶段状态为 **F0 启动前预检/待启动**。

## 三项固定噪声误差均改善，但 rotation 改善有限

所有指标都在 postprocess 后的原始 action7 空间计算。表中的 ratio 为 `trained / base`；小于 `1` 表示 step 300 误差更低。Aggregate 先对每个 anchor 的 50-step horizon 求均值，再按 episode 做 macro-average。

| 指标 | PI0 base | Step 300 | Trained / base | 相对下降 |
| --- | ---: | ---: | ---: | ---: |
| Translation ADE | `56.447842 mm` | `35.498271 mm` | `0.628869` | `37.11%` |
| Rotation geodesic ADE | `5.215184°` | `4.968256°` | `0.952652` | `4.73%` |
| Gripper MAE | `0.567802` | `0.284258` | `0.500628` | `49.94%` |

**逐 anchor 结果揭示了 aggregate 隐藏的差异。** Translation 在 4/4 个 anchor 上改善；rotation 只在 index `11` 上显著改善、其余 3 个变差；gripper 在 index `9/29` 上大幅改善，但在 `11/24` 上变差。因此本阶段通过的是小样本 learnability gate，不是各维度、各样本都达到稳定拟合。

| Monitor index | Episode | Translation mm（base → trained） | Rotation °（base → trained） | Gripper MAE（base → trained） |
| ---: | ---: | ---: | ---: | ---: |
| `9` | `0` | `79.132885 → 42.475575` | `9.326010 → 9.489520` | `0.852650 → 0.019847` |
| `11` | `0` | `61.967413 → 37.622735` | `6.591485 → 2.799196` | `0.319667 → 0.565586` |
| `24` | `1` | `29.995669 → 28.269253` | `2.459648 → 4.467851` | `0.248391 → 0.497765` |
| `29` | `1` | `54.695404 → 33.625521` | `2.483593 → 3.116456` | `0.850500 → 0.053833` |

## 评估范围只有 4 个 train-internal 样本

`22of25` 表示数据产品的冻结 provenance，不表示 S1 使用了 22 个 episode。S1 配置只选择 episode index `[0,1]`，每条均匀抽 `16` 个有效 anchor，训练池总计 `32` 个样本。Evaluator 使用训练时已经冻结的 4 个 monitor anchor，每个 episode 恰有 2 个；`held_out=false`，没有 validation/test split。

| 项目 | 固定合同 |
| --- | --- |
| 数据 profile | observation `15 Hz`、action `15 Hz`、chunk `50` |
| S1 训练池 | 2 episodes、32 anchors；不是完整 22-episode 训练池 |
| Evaluator monitor | indices `[9,11,24,29]`，4/32 train-internal anchors |
| Raw episodes | `run_20260714_212901_00063`、`run_20260714_211745_00057` |
| State/action | current measured EEF state10；future measured EEF delta + gripper action7 |
| Translation ADE | `mean L2(pred_delta_xyz - target_delta_xyz)`，单位 mm |
| Rotation ADE | `mean geodesic(Exp(pred_rotvec), Exp(target_rotvec))`，单位 degree |
| Gripper MAE | raw `gripper_0_1` absolute error；预测值不裁剪 |
| Aggregation | episode macro-average of anchor horizon means |
| Quality threshold | evaluator 未设置自动 PASS 阈值；本报告结合方向性改善与工程门禁作阶段判断 |

每个 monitor 样本使用独立且可复现的 CPU `torch.Generator` noise，shape `[1,50,32]`、dtype `float32`，PI0 以 `num_steps=10` 预测完整 chunk。四个 seed 分别为 `101009/101011/101024/101029`；同一样本的 base、trained 和 reload 预测复用同一 noise。

## Monitor loss 总体下降，单 batch train loss 保持高波动

固定 monitor loss 在 step 0、50、100、150、200、250、300 上使用相同 4 个训练 anchor 和固定随机条件。它总体下降，但 step 150 相比 step 100 有回升，不能描述为单调收敛。

| Step | Logged train loss | Fixed monitor loss | 说明 |
| ---: | ---: | ---: | --- |
| `0` | — | `0.35633501` | 训练前基线 |
| `5` | `0.94256938` | — | 首个 train log |
| `50` | `0.18689850` | `0.23622213` | monitor 下降 |
| `100` | `0.21603332` | `0.21199043` | 发布 checkpoint |
| `150` | `0.31005377` | `0.22637497` | monitor 暂时回升 |
| `200` | `0.54885179` | `0.21451148` | 发布 checkpoint |
| `250` | `0.17745204` | `0.15539633` | 后段继续下降 |
| `300` | `0.07152425` | `0.14697757` | 发布 final checkpoint |

Train loss 是每 5 steps 记录的当前单 batch 值，并非滑动均值：日志最小值达到 `0.00514197`（step 220），最大值达到 `2.19569921`（step 235）。这种波动与 batch size `1`、小训练池和随机 flow-matching 条件一致；更可靠的阶段证据是固定 monitor 的总体变化及同噪声 Cartesian 对比，而不是任一单点 train loss。

## Strict reload、saved processors 与 W&B 均可复现

| 验收项 | 结果 | 证据 |
| --- | --- | --- |
| PI0 base strict load | PASS | `missing=[]`、`unexpected=[]`；官方 factory processor 注入 action15 stats |
| Step 300 strict load | PASS | project manifest 存在；`missing=[]`、`unexpected=[]` |
| Step 300 independent reload | PASS | 第二次从磁盘 strict load 相同 checkpoint |
| Saved processors | PASS | 第二次预测从 checkpoint 的 pre/post processor JSON 与 state safetensors 重载 |
| Normalized action consistency | EXACT | max absolute diff=`0.0`；allclose `rtol=1e-5, atol=1e-6` |
| Raw action consistency | EXACT | max absolute diff=`0.0`；固定 noise 完全复用 |
| Effective stats | IDENTICAL | 三次加载 SHA-256 均为 `2fd71a75a7005cfc76e20467c9fea294e39b1edc6718821e2e36db37ff52a4d9` |
| W&B | PASS | `7gnnlevp` state=`finished`、`logged_artifacts=0`；scalar history 到 step 300 |
| Local checkpoints | PASS | step `100/200/300` 均本地发布，日志明确 `no Hub/W&B artifact` |

S1 invocation 为 `start_step=0`、`requested_stop_step=300`、`last_step=300`、`resume=false`、`exit_code=0`。这次没有发生中断恢复；恢复路径的实测证据来自 [S0 报告](./S0_PI0_FULL_SMOKE_22OF25.md)，其中 step 2 checkpoint 成功恢复并继续到 step 3。

## 能力边界与 F0 下一步

- 本结果是 teacher-forced、fixed-noise、in-sample offline evaluation。4 个 monitor 样本全部属于训练池，不能估计泛化误差。
- S1 只训练两个 episode 的 32 个 anchor；它不能替代 F0 在完整 22 条冻结 scope 上的正式训练。
- 每个 anchor 只评估一个固定 noise seed，未估计 diffusion sampling variance；样本数也不足以建立统计置信区间。
- Rotation aggregate 只改善 `4.73%`，且 3/4 anchors 变差；F0/M0 必须继续逐 episode 监控 rotation，而不能只看 aggregate translation 或 monitor loss。
- 没有真机 rollout、IK/controller、安全限位或任务成功率证据；真机运行仍未授权。

**阶段决策：S1 标记 PASS，F0 进入启动前预检/待启动。** F0 启动前应再次冻结完整 22-episode action15 dataset、双卡 DDP 参数、step/checkpoint 周期和 no-artifact 合同；启动后保持 train-internal monitor 的解释边界，最终由 M0 给出全训练集逐 episode in-sample 报告。

尚待 F0/M0 回答的问题是：rotation 的小幅、异质改善能否随完整数据和更多 steps 稳定扩大，以及固定 monitor 改善是否能在全部 22 条训练轨迹的 teacher-forced reconstruction 中复现。

## 审计索引

- 机器可读 evaluator 结果：[pi0_s1_overfit_22of25_step300.json](../artifacts/s1_eval/pi0_s1_overfit_22of25_step300.json)
- Evaluator 实现：[eval_pi0_s1_overfit.py](../scripts/eval_pi0_s1_overfit.py)
- Run config：[resolved_config.yaml](../experiments/pi0_full_eef_obs15_act15_v1/20260715T142336Z_pi0_full_eef_s1_overfit_obs15_act15_22of25_b8bc6628/resolved_config.yaml)
- Run manifest：[run_manifest.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T142336Z_pi0_full_eef_s1_overfit_obs15_act15_22of25_b8bc6628/run_manifest.json)
- Monitor identity：[monitor_subset.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T142336Z_pi0_full_eef_s1_overfit_obs15_act15_22of25_b8bc6628/monitor_subset.json)
- Training log：[20260715T142336_r0.log](../experiments/pi0_full_eef_obs15_act15_v1/20260715T142336Z_pi0_full_eef_s1_overfit_obs15_act15_22of25_b8bc6628/logs/20260715T142336_r0.log)
- Invocation：[20260715T143040_acf7728b.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T142336Z_pi0_full_eef_s1_overfit_obs15_act15_22of25_b8bc6628/invocations/20260715T143040_acf7728b.json)
- Final model manifest：[franka_pi0_checkpoint_manifest.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T142336Z_pi0_full_eef_s1_overfit_obs15_act15_22of25_b8bc6628/checkpoints/step-000300/pretrained_model/franka_pi0_checkpoint_manifest.json)
