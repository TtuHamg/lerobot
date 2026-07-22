# ACT and pi0 Training / Loss Analysis

本文基于当前仓库源码和 `output_lerobot` 中的本地训练日志，回答两个问题：

- ACT 和 pi0 这两次训练是否是全参数训练，是否用了预训练参数。
- 二者在 LeRobot 框架下的 loss 代码在哪里，以及 loss 的实现含义。

## 结论概览

| Model | 本次是否全参数训练 | 是否使用预训练参数 | 本次中断后是否有 checkpoint |
| --- | --- | --- | --- |
| ACT | 是。日志中 `num_learnable_params == num_total_params == 51597190` | policy 整体没有从 checkpoint 加载，`pretrained_path=null`；但视觉 backbone 是 `resnet18` 且用了 ImageNet 预训练权重 | 有，`output_lerobot/act/task0/checkpoints/last -> 020000` |
| pi0 | 是。日志中 `num_learnable_params == num_total_params == 4028019472` | 是，从本地 HF cache 的 `lerobot/pi0_base` snapshot 加载 `model.safetensors` | 没有，崩溃前只到 step 4463，低于 `save_freq=20000` |

## ACT

### 训练方式和预训练来源

本次 ACT 配置在：

- `output_lerobot/act/task0/checkpoints/020000/pretrained_model/config.json`

关键字段：

- `use_peft=false`
- `pretrained_path=null`
- `vision_backbone=resnet18`
- `pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1`
- `use_vae=true`
- `kl_weight=10.0`

因此：

1. 这不是从一个完整 ACT policy checkpoint 继续 fine-tune。`pretrained_path=null` 表示 ACT policy 主体从当前配置初始化。
2. 视觉 backbone 使用了 torchvision 的 ResNet18 ImageNet 权重。源码中 `ACT.__init__` 调用 `torchvision.models.<vision_backbone>(weights=config.pretrained_backbone_weights, ...)`。
3. 本次不是 PEFT/LoRA。日志显示 learnable 参数量等于总参数量：
   - `num_learnable_params=51597190`
   - `num_total_params=51597190`
4. ACT 的 `get_optim_params()` 会把 backbone 和非 backbone 分成两个 optimizer param group，但你的配置里 `optimizer_lr=1e-5`、`optimizer_lr_backbone=1e-5`，所以 backbone 也在训练，且学习率相同。

相关源码：

- `src/lerobot/policies/act/configuration_act.py:96-128`
- `src/lerobot/policies/act/modeling_act.py:72-91`
- `src/lerobot/policies/act/modeling_act.py:324-334`
- `src/lerobot/scripts/lerobot_train.py:375-376`

### ACT loss 代码位置

ACT 的训练 loss 在：

- `src/lerobot/policies/act/modeling_act.py:137-164`

核心逻辑：

```python
actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
num_valid = valid_mask.sum() * abs_err.shape[-1]
l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

if self.config.use_vae:
    mean_kld = (
        (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
    )
    loss = l1_loss + mean_kld * self.config.kl_weight
else:
    loss = l1_loss
```

### ACT loss 分析

ACT 的训练目标是 action chunk reconstruction，加上可选 VAE KL 正则。

在你的配置里 `use_vae=true`，所以实际 loss 是：

```text
loss = masked_L1(action, predicted_action) + kl_weight * KL(q(z | state, action_chunk) || N(0, I))
```

其中：

- `masked_L1` 是对 action chunk 的 L1 回归误差。
- `action_is_pad` 用来屏蔽 padding action。被 padding 的时间步不会进入 loss。
- `num_valid` 按有效时间步数乘以 action 维度做归一化。
- KL 项来自 VAE encoder 输出的 `mu_hat` 和 `log_sigma_x2_hat`。
- 你的 `kl_weight=10.0`，所以 KL 正则权重较强。

这意味着 ACT 日志里的 `loss` 不是单纯动作 L1，而是 `l1_loss + 10 * kld_loss`。如果要判断动作拟合质量，应同时看 `l1_loss` 和 `kld_loss`，不要只看总 loss。

## pi0

### 训练方式和预训练来源

本次 pi0 原始启动参数在：

- `output_lerobot/pi0/task0/wandb/run-20260618_180935-jp7dj0oh/files/wandb-metadata.json`

关键参数：

- `--policy.type=pi0`
- `--policy.pretrained_path=/ytech_milm_intern/tujiahang/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/25c379b52ba2ff8788cab921758a3cc3fe3f77f2`
- `--policy.freeze_vision_encoder=false`
- `--policy.train_expert_only=false`
- `--policy.gradient_checkpointing=true`
- `--policy.compile_model=true`
- `--policy.dtype=bfloat16`

训练日志进一步确认：

- `Loading model from: .../lerobot--pi0_base/...`
- `Loaded state dict from model.safetensors`
- `Remapped 777 state dict keys`
- `All keys loaded successfully`
- `num_learnable_params=4028019472`
- `num_total_params=4028019472`

因此：

1. pi0 是从 `lerobot/pi0_base` 预训练权重开始 fine-tune。
2. 本次不是只训练 action expert，也没有冻结 vision encoder。
3. 由于 `freeze_vision_encoder=false` 且 `train_expert_only=false`，冻结逻辑不会把 PaliGemma 或 vision tower 的参数 `requires_grad=False`。
4. `get_optim_params()` 对 pi0 直接返回 `self.parameters()`，日志也显示全部 4B 参数可训练。
5. `gradient_checkpointing=true` 只影响显存和计算图重算，不代表冻结参数。
6. `compile_model=true` 只影响 `torch.compile` 优化，不代表冻结参数。

相关源码：

- `src/lerobot/policies/pi0/configuration_pi0.py:79-88`
- `src/lerobot/policies/pi0/modeling_pi0.py:432-447`
- `src/lerobot/policies/pi0/modeling_pi0.py:582-590`
- `src/lerobot/policies/pi0/modeling_pi0.py:970-1086`
- `src/lerobot/policies/pi0/modeling_pi0.py:1146-1147`

### pi0 loss 代码位置

pi0 的 loss 分两层：

1. `PI0Policy.forward()` 准备 batch、采样 noise/time、调用核心模型，并做 reduction：
   - `src/lerobot/policies/pi0/modeling_pi0.py:1283-1321`
2. `PI0Pytorch.forward()` 实现 flow matching 的逐元素 MSE：
   - `src/lerobot/policies/pi0/modeling_pi0.py:762-811`

核心逻辑：

```python
noise = self.model.sample_noise(actions.shape, actions.device)
time = self.model.sample_time(actions.shape[0], actions.device)
losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
losses = losses[:, :, :original_action_dim]
loss = losses.mean()
```

核心模型内部：

```python
time_expanded = time[:, None, None]
x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions

# model predicts v_t from images/language/state/noisy actions/time
v_t = self.action_out_proj(suffix_out)

return F.mse_loss(u_t, v_t, reduction="none")
```

### pi0 loss 分析

pi0 使用 flow matching 训练目标。它不是直接回归动作 `action`，而是在随机时间 `t` 上构造一个 noisy action：

```text
x_t = t * noise + (1 - t) * action
```

目标速度场是：

```text
u_t = noise - action
```

模型根据图像、语言 token、状态、`x_t` 和 `t` 预测速度：

```text
v_t = model(obs, state, x_t, t)
```

loss 是逐元素 MSE：

```text
loss = mean((v_t - u_t)^2)
```

实现细节：

- `sample_noise()` 从标准正态采样。
- `sample_time()` 从 Beta 分布采样，再按 `time_sampling_scale` 和 `time_sampling_offset` 缩放。
- loss 先保留形状 `(batch, chunk_size, max_action_dim)`。
- 然后裁剪到真实 action 维度：`losses[:, :, :original_action_dim]`。
- 默认 `reduction="mean"`，对 batch、时间步、action 维度全部求均值。
- `loss_per_dim` 会记录每个 action 维度的平均 loss，但当前 wandb wrapper 不支持 list，所以日志里有忽略 `loss_per_dim` 的 warning。

这意味着 pi0 日志中的 `loss` 是 velocity prediction MSE，不是 action L1/MSE。它衡量模型对 flow matching 速度场的预测质量；和 ACT 的 action reconstruction loss 不是同一个量纲，不能直接横向比较数值大小。

## 二者 loss 的关键差异

| 项目 | ACT | pi0 |
| --- | --- | --- |
| 训练目标 | 直接预测 action chunk | 预测从 action 到 noise 的 flow velocity |
| 主 loss | masked L1 | MSE |
| 随机性 | VAE latent sampling/regularization | 每步采样 noise 和 time |
| padding 处理 | 显式用 `action_is_pad` mask | 裁剪到真实 action dim；当前代码片段未使用 `action_is_pad` 做时间 mask |
| 额外正则 | VAE KL，权重 `kl_weight=10.0` | 无显式 KL；依赖 flow matching 目标 |
| 日志可比性 | `loss = l1 + kl_weight * kld` | `loss = velocity MSE mean` |

## 对当前训练的建议

1. ACT 可以从 `020000` checkpoint 断点续训。由于上次实际跑到 `23526` 才崩，`20000-23526` 的更新会丢失。
2. pi0 这次没有 checkpoint，不能从 step 4463 恢复。重新训练时建议加 `--save_freq=1000` 或 `--save_freq=2000`。
3. 如果 pi0 全参数训练显存/时间压力过大，可以考虑：
   - `--policy.freeze_vision_encoder=true`：冻结 vision tower。
   - `--policy.train_expert_only=true`：冻结 PaliGemma/VLM，只训练 action expert 和相关 projection。
   - 或使用 PEFT/LoRA 配置，但这需要单独确认当前仓库对 pi0 的 PEFT 参数设置。
4. 比较 ACT 和 pi0 时，不要直接比较 loss 数值。ACT 的 loss 是 action reconstruction L1 加 KL，pi0 的 loss 是 flow velocity MSE。
