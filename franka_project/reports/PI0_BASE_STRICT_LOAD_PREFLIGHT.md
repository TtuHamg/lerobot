# PI0 base 全参数加载 preflight

## 结论

本地 `lerobot/pi0_base` revision `25c379b52ba2ff8788cab921758a3cc3fe3f77f2` 的真实 strict-load smoke 已通过。此次只验证模型加载、全参数契约和 official PI0 pre/post processor，**没有启动训练、没有保存模型、没有创建 W&B run，也没有上传 Hub**。

核心结果：

- 三个 canonical image slots 原样保留；policy-visible state/action 为 `10D/7D`，内部 padding 仍为 `32D/32D`。
- 配置为 BF16、gradient checkpointing 开启，`freeze_vision_encoder=false`、`train_expert_only=false`、`use_relative_actions=false`。
- `4,028,019,472 / 4,028,019,472` 参数可训练，共 778 个 parameter tensors；PEFT/LoRA 参数为 0。
- 权重 namespace 为 `pi0_core_unprefixed`；strict load 最终 `missing=[]`、`unexpected=[]`。
- `action_in_proj.bias` 与 safetensors 原 tensor 在模型 dtype cast 后逐元素完全一致，checkpoint tensor SHA-256 为 `4c06cfc44340a7c14bfb617544fb6c49a6ee2feeba78dc02fc4418c6423c6dd6`。
- official processor 只注入 `observation.state` 和 `action` 两组 effective stats，维度严格为 `10/7`。

## Canonical embedding alias

第一次 smoke 按 fail-closed 规则停止，唯一缺失的 current-model key 为：

```text
paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
```

canonical base checkpoint 中存在同 shape 的：

```text
paligemma_with_expert.paligemma.lm_head.weight
```

这与 upstream `PI0Policy._fix_pytorch_state_dict_keys` 已实现的 `lm_head → embed_tokens` clone 规则一致。项目 loader 现在只允许这一条显式、shape-checked alias；任何其他 missing/unexpected key 或 `model.` policy-wrapper namespace 都会抛错，不会返回随机权重。第二次真实 smoke 通过。

## 资源

| 项目 | 数值 |
|---|---:|
| GPU | NVIDIA A800-SXM4-80GB，`cuda:1` |
| 完整加载与 processor 构建 | 144.36 s |
| peak allocated | 22,901,433,856 bytes（约 21.33 GiB） |
| peak reserved | 23,420,993,536 bytes（约 21.81 GiB） |
| 结束后 GPU memory | 已释放 |

## Statistics 边界

本次为了验证 processor 构建，使用 D2 spike 文件 `artifacts/stats/pi0_eef_stats_spike_action15.json`，其 state/action count 为 `824/41,200`。该文件**不用于正式训练**；S0/F0 必须注入 D3/D4 最终 action15 数据集的 `10D/7D` effective stats。

## 代码与测试

- Helper：`franka_project/src/franka_eef_pipeline/pi0_training.py`
- Tests：`franka_project/tests/test_pi0_training.py`
- 显式 resume API：`load_pi0_full_checkpoint_weights(policy, checkpoint_dir)`
- Trainer API：`load_pi0_full_policy_and_processors(pretrained_path, dataset_stats, device)`
- Checkpoint API：`save_pi0_full_checkpoint(...)`，保存 unprefixed `policy.model` weights、config、pre/post processor 和 geometry/stats manifest，始终 `push_to_hub=false`
- 全项目轻量回归：`55 passed`；唯一 warning 来自外部 `mcap_ros2` deprecated import。

机器可读结果见 `artifacts/training_preflight/pi0_base_strict_load_smoke.json`。
