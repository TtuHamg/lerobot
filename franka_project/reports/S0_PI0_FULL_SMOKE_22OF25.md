# S0 PI0 全参数训练 smoke：22of25 / obs15-act15

- 结论：**PASS — S0 工程链路验收通过**
- Run：`20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2`
- W&B：[ttuhamg/franka_pi0_full_eef/vay230a2](https://wandb.ai/ttuhamg/franka_pi0_full_eef/runs/vay230a2)
- 报告生成时间：`2026-07-15T14:26:33Z`

## 技术结论

- 该 run 在冻结的 **22/25 条 Franka 数据 scope** 上使用 action15 数据产品：相机 `15 Hz`、action `15 Hz`、action chunk `50`。S0 为最小工程 smoke，实际只从 episode index `0, 1` 各取一个 anchor，训练 dataset size 为 `2`；它不是在 22 条数据上的完整收敛训练。
- 三个优化 step 全部完成。第一次 invocation 从 step `0` 训练到 step `2` 并发布 checkpoint；第二次 invocation 严格从 step `2` 恢复，完成 step `3` 并发布最终 checkpoint。两次进程均 `exit_code=0`。
- 模型是非 LoRA 的全参数图：`3,238,048,528 / 3,238,048,528` 个唯一参数 `requires_grad=True`，共 `776` 个 parameter tensors，`trainable_fraction=1.0`、`lora_parameter_count=0`。
- 首次 backward 有 `3,133,186,832` 个参数获得有限梯度。未获得梯度的 `104,861,696` 个参数精确对应预先登记的 6 项结构性不可达账本；账本外没有 `grad=None` 参数。
- 独立 verifier 已通过最终 checkpoint strict reload、已保存 pre/post processors reload、真实 action15 样本预处理/反归一化 roundtrip 和有限 policy forward；W&B 状态为 `finished`，scalar history 完整，`logged_artifacts=0`。项目回归结果为 **80 passed**。

这证明训练、保存、恢复和可重载性链路成立；三步 smoke 不足以证明模型已学会 `stack the cups`，也不构成真机部署授权。

## 三步训练与恢复链路完整

固定 monitor 来自训练数据本身，包含两个 anchor，`held_out=false`。因此 `monitor/loss` 是可重复的 in-train smoke 指标，不是独立验证集指标。

| Step | Invocation | `train/loss` | `monitor/loss` | LR | Grad norm | 结果 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | 初始 | — | `0.14048439` | — | — | 固定 monitor 基线 |
| 1 | 初始 | `0.16212599` | — | `1.000e-05` | `12.86729` | optimizer step 完成 |
| 2 | 初始 | `0.11883555` | `0.10239626` | `5.500e-06` | `12.48328` | 原子发布 `step-000002` |
| 3 | 从 step 2 恢复 | `0.01356122` | `0.09315707` | `1.000e-06` | `2.98860` | 原子发布 `step-000003` |

| Invocation | Start → stop | Resume | Exit | Published checkpoint |
| --- | --- | --- | ---: | --- |
| `20260715T141506_9c78fd00` | `0 → 2` | `false` | `0` | `step-000002` |
| `20260715T141936_cf2d8fcf` | `2 → 3` | `true` | `0` | `step-000003` |

只有三个优化 step，不适合绘制或解释为收敛趋势；这里使用精确值表做工程审计。可陈述的结论仅是 loss 均为有限值，且固定 monitor 从 `0.14048439` 降至 `0.09315707`。

## Scope、表示与采样定义

| 项目 | 冻结定义 |
| --- | --- |
| 原始 scope | 25 条 source episodes 中 22 条存在声明的 MCAP 并获准进入训练；3 条因声明的 MCAP 缺失而排除 |
| Scope identity | `franka_scope_22of25_v1`；content SHA-256 `86402a4138002b1ff6e02d95e7f434c5ef3f658c362fb88253ff8b1c44a69461` |
| 数据产品 | `franka_current_eef_obs15_act15_v1_22of25_76b839b2`；22 episodes、7,465 个有效 anchor |
| 频率 | observation `15 Hz`；action `15 Hz` |
| S0 有效样本 | episode indices `[0, 1]`，每条最多 1 个 anchor，共 2 个样本；batch size `1`、gradient accumulation `1`、world size `1` |
| Monitor | 两个训练 anchor；raw episodes `run_20260714_212901_00063`、`run_20260714_211745_00057`；不是 held-out |
| Language | `stack the cups` |
| State | 当前实测 EEF：`xyz + rotation6d + gripper_0_1`，共 10D |
| Action | 未来实测目标 EEF delta：平移 3D、body rotvec 3D、gripper 1D，共 7D；chunk size 50 |
| Normalization | state/action 为 `MEAN_STD`，visual 为 `IDENTITY` |

三条排除记录为 `run_20260714_202237_00032`、`run_20260714_175543_00008`、`run_20260714_165338_00005`。完整 22 条 episode 身份、MCAP 大小与逐文件 SHA-256 由冻结 scope manifest 保存，不在本报告重复展开。

## 3.238B 全参数图与结构账本吻合

训练图在进入 optimizer 前完成两项语义保持整理：PaliGemma output head 与 input embedding 做参数指针级共享，去除 `526,647,296` 个 tied duplicate；PI0 diffusion forward 不使用的 expert LM head 从图中移除，去除 `263,323,648` 个参数。整理后的 `3,238,048,528` 个唯一参数全部保持可训练，不使用 PEFT/LoRA。

首次 backward 的总体结果：

| 指标 | 精确值 |
| --- | ---: |
| `requires_grad` parameters | `3,238,048,528` |
| `requires_grad` tensors | `776` |
| 获得 finite gradient 的 parameters | `3,133,186,832` (`96.761577%`) |
| 获得 gradient 的 tensors | `770 / 776` (`99.226804%`) |
| 获得非零 gradient 的 tensors | `769` |
| Structural ledger | `104,861,696` parameters、`6` tensors |

按子系统检查，vision encoder、action expert、state/action projections 的 parameter/gradient coverage 均为 `100%`；只有 VLM 的 6 项结构账本不在 action loss 的反向路径上。所有已获得的梯度均为有限值。

以下名称共享前缀 `model.paligemma_with_expert.paligemma.model.language_model.`：

| # | 结构性不可达参数后缀 | Shape | Numel |
| ---: | --- | --- | ---: |
| 1 | `layers.17.self_attn.o_proj.weight` | `[2048, 2048]` | `4,194,304` |
| 2 | `layers.17.post_attention_layernorm.weight` | `[2048]` | `2,048` |
| 3 | `layers.17.mlp.gate_proj.weight` | `[16384, 2048]` | `33,554,432` |
| 4 | `layers.17.mlp.up_proj.weight` | `[16384, 2048]` | `33,554,432` |
| 5 | `layers.17.mlp.down_proj.weight` | `[2048, 16384]` | `33,554,432` |
| 6 | `norm.weight` | `[2048]` | `2,048` |
|  | **合计** |  | **`104,861,696`** |

这些参数仍为 `requires_grad=True` 并保留在 optimizer 中；账本将其分类为 `permanent_final_prefix_output_outside_action_loss`，训练配置因此需要 DDP `find_unused_parameters=true`。这里的“全参数训练”指整理后训练图全部可训练，不应误读为每个参数在 PI0 action loss 下都必然获得梯度。

## Checkpoint、processor 与 W&B 验收通过

| 验收项 | 结果 | 审计事实 |
| --- | --- | --- |
| Base strict load | PASS | `missing_keys=[]`、`unexpected_keys=[]`；canonical expert head 仅在精确验证后丢弃 |
| Step 2 checkpoint | PASS | 原子发布；包含 model、pre/post processor、optimizer、scheduler、step 和 rank-0 RNG state |
| Resume strict load | PASS | 从 `step-000002` 项目 checkpoint 恢复，project manifest 存在，`missing_keys=[]`、`unexpected_keys=[]` |
| Step 3 checkpoint | PASS | `last_checkpoint.step=3`；最终 `model.safetensors` 为 `7,312,556,248` bytes，SHA-256 `2cc105ac9d01685ba35971e35268af18905f70378cfa7974333031283f6f3ab1` |
| Saved processors | PASS | 独立从最终 checkpoint 本地重载；真实样本得到 state `[1,10]`、action `[1,50,7]`，action roundtrip 通过 |
| Finite policy forward | PASS | 独立 verifier 对真实 action15 样本完成 no-grad forward，loss 为 finite scalar |
| W&B | PASS | run id `vay230a2`、state `finished`；train scalar steps `1..3`、monitor steps `0,2,3` 完整；`logged_artifacts=0` |
| Upload policy | PASS | 两个 checkpoint 均 `hub_upload=false`、`wandb_artifact_upload=false` |
| Regression | PASS | `80 passed` |

两次 invocation 使用同一 W&B run id；resume 没有创建第二个远端实验身份。W&B 只记录 scalar，不上传 model checkpoint。

## 证据边界、下一步与未决问题

- 本报告是 S0 工程验收，不评估泛化、任务成功率或真机安全性。Monitor 是训练集固定子集；没有独立 val/test。
- 22 条冻结 scope 是数据 provenance，S0 实际只执行两个训练 anchor。不能把本次结果表述为“已在全部 22 条数据上训练完成”。
- 6 项结构账本是模型图的确定性属性，不是偶发漏梯度；后续长训必须继续做账本精确匹配，并拒绝任何账本外 `grad=None`。
- 下一阶段可在完整 22 条 action15 数据上启动正式全参数训练，保持相同 normalization、checkpoint/resume、W&B no-artifact 和 strict verifier 合同；模型质量最终仍需离线行为检查和受控真机 rollout 验证。
- 独立 verifier PASS 与 `80 passed` 来自本次验收终端结果；run 目录当前未持久化 verifier JSON 或 pytest 日志。若需要完全机器可追溯的发布包，下一次应将这两份 stdout 结果作为只读验收附件保存。
- 正式训练仍需在启动前冻结总 steps、checkpoint 周期和停止条件；真机成功率及安全门槛也尚未由本次 S0 定义。

## 审计索引

- 冻结 scope：[frank3_train_passed_22of25_v1.yaml](../manifests/frank3_train_passed_22of25_v1.yaml)
- Run config：[resolved_config.yaml](../experiments/pi0_full_eef_obs15_act15_v1/20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2/resolved_config.yaml)
- Run manifest：[run_manifest.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2/run_manifest.json)
- Gradient ledger：[first_backward_gradient_coverage.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2/first_backward_gradient_coverage.json)
- Monitor identity：[monitor_subset.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2/monitor_subset.json)
- Initial invocation：[20260715T141506_9c78fd00.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2/invocations/20260715T141506_9c78fd00.json)
- Resume invocation：[20260715T141936_cf2d8fcf.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2/invocations/20260715T141936_cf2d8fcf.json)
- Step 2 manifest：[checkpoint_manifest.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2/checkpoints/step-000002/checkpoint_manifest.json)
- Step 3 model manifest：[franka_pi0_checkpoint_manifest.json](../experiments/pi0_full_eef_obs15_act15_v1/20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2/checkpoints/step-000003/pretrained_model/franka_pi0_checkpoint_manifest.json)
- Training logs：[initial](../experiments/pi0_full_eef_obs15_act15_v1/20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2/logs/20260715T141209_r0.log) · [resume](../experiments/pi0_full_eef_obs15_act15_v1/20260715T141209Z_pi0_full_eef_s0_obs15_act15_22of25_13652be2/logs/20260715T141539_r0.log)
- Verifier implementation：[verify_pi0_s0_run.py](../scripts/verify_pi0_s0_run.py)
