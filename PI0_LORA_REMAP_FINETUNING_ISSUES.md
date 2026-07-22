# PI0 LoRA remap fine-tuning 问题分析

本文记录 `output_lerobot/pi0/task0_lora_ablation/qkvo_mlp_r64` 这一路 LoRA fine-tuning 的主要问题。下文称它为“前者”或“remap 版”。对照实验为 `output_lerobot/pi0/task0_lora_ablation_noremap/qkvo_mlp_r64`，下文称为“后者”或“noremap 版”。

## 结论

前者训练 loss 很低，但真机效果差，主要原因不是 LoRA rank 或 target module 本身，而是训练和 async 真机推理时的输入 schema 与图像预处理路径不一致。

最核心的问题是：前者 checkpoint 继承了 `pi0_base` 的 `input_features`，里面图像 feature 被写成 `3x224x224`。async inference 在把真机 observation 发给 policy 前，会按 checkpoint 的 image feature shape 先做一次直接 `interpolate` resize。因此前者真机推理时，`1280x720` 或 `640x480` 图像会先被直接拉伸成 `224x224` 方图，而训练时图像大概率是原始 dataset 分辨率进入 Pi0 内部，再由 Pi0 的 `resize_with_pad_torch` 保持比例缩放并补黑边到 `224x224`。

两者最终虽然都是 `224x224`，但图像几何分布不同：一个是直接拉伸变形，一个是保持宽高比加 padding。对机器人抓取策略来说，这个差异足够严重。

## 两个实验的 checkpoint schema 差异

### 前者：remap 版

路径：

```text
output_lerobot/pi0/task0_lora_ablation/qkvo_mlp_r64/checkpoints/last/pretrained_model
```

训练时使用了类似：

```bash
--policy.path=/.../pi0_base/...
--rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist.left":"observation.images.left_wrist_0_rgb"}'
```

但没有清空 pretrained config 的 `input_features`。

保存后的 `config.json` 中，policy 仍然期望：

```text
observation.images.base_0_rgb         [3, 224, 224]
observation.images.left_wrist_0_rgb   [3, 224, 224]
observation.images.right_wrist_0_rgb  [3, 224, 224]
observation.state                     [32]
```

注意：这不是你的 SO101 dataset 的真实 feature schema，而是继承自 `pi0_base`。

### 后者：noremap 版

路径：

```text
output_lerobot/pi0/task0_lora_ablation_noremap/qkvo_mlp_r64/checkpoints/last/pretrained_model
```

训练时使用了：

```bash
--policy.path=/.../pi0_base/...
--policy.input_features=null
--policy.empty_cameras=0
```

保存后的 `config.json` 中，policy 期望：

```text
observation.images.front       [3, 720, 1280]
observation.images.wrist.left  [3, 720, 1280]
observation.state              [6]
```

这和 `tuuy/lerobot_so_arm101_task0_new` dataset 以及真机 observation key 更一致。

## 为什么前者 config 会写成 224x224

`make_policy` 里的逻辑是：

```python
cfg.output_features = dataset action features

if not cfg.input_features:
    cfg.input_features = dataset input features
```

前者从 `pi0_base` 加载时，`cfg.input_features` 已经有值，所以不会用当前 dataset 的 feature shape 覆盖。`rename_map` 只会改 observation key，不会同步修改 shape。

因此，虽然 dataset 里视频是 `720x1280`，前者 checkpoint 的 policy config 仍然保存为 `224x224`。

## 训练路径和 async 推理路径的关键不一致

### 训练时的可能路径

前者训练时，dataset 读出的图像仍然是原始分辨率：

```text
observation.images.front       720x1280
observation.images.wrist.left  720x1280
```

然后 preprocessor 做 key remap：

```text
front      -> base_0_rgb
wrist.left -> left_wrist_0_rgb
```

visual normalizer 是 `IDENTITY`，不会 resize 图像。之后进入 Pi0 model，Pi0 内部调用：

```python
resize_with_pad_torch(img, 224, 224)
```

这个函数会保持宽高比 resize，再补黑边，不裁剪、不拉伸。

以 `1280x720` 到 `224x224` 为例：

```text
1280x720 -> 224x126，再上下补黑边到 224x224
```

### async 真机推理时的路径

async inference 的 `raw_observation_to_observation` 会先根据 `policy_image_features[key].shape` resize 图像。前者 checkpoint 的 shape 是 `3x224x224`，所以真机图像会先被直接 resize 到 `224x224`：

```python
torch.nn.functional.interpolate(..., size=(224, 224))
```

这是直接缩放，不保持原始宽高比，也不补黑边。

以 `1280x720` 到 `224x224` 为例：

```text
1280x720 -> 224x224，画面被压成方图
```

因此 Pi0 内部再检查图像尺寸时，发现已经是 `224x224`，就不会再执行训练时的 aspect-ratio preserving padding 路径。

## 为什么这会导致真机效果差

最终输入 encoder 的 tensor shape 都是 `224x224`，但视觉内容分布不同：

| 路径 | 处理方式 | 几何效果 |
|---|---|---|
| 训练时 Pi0 内部处理 | 保持比例缩放，加黑边 | 物体比例不变 |
| 前者 async 推理 | 直接 interpolate 到方图 | 物体和夹爪比例变形 |

机器人 manipulation policy 对视觉几何很敏感。瓶盖、纸杯、夹爪之间的相对位置、比例和接触点都依赖图像几何。直接拉伸会改变这些视觉线索，即使 loss 很低，也可能在真机上表现很差。

## `observation.state` 写成 32 是否要紧

这个问题相对没那么严重。

Pi0 内部本来会把 state pad 到 `max_state_dim`：

```python
state = pad_vector(batch["observation.state"], self.config.max_state_dim)
```

SO101 的真实 state 是 6 维：

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

前者 config 写成 `[32]`，主要是继承了 `pi0_base` 的最大 state 维度。实际输入通常仍然是 6 维，归一化后再 pad 到 32 维，后 26 维是 0。这是 Pi0 支持不同机器人 state 维度的一种设计。

它需要满足两个前提：

1. 6 维 state 的顺序和训练 dataset 一致。
2. normalizer 使用的是当前 SO101 dataset 的 state stats，而不是 base model 的 stats。

从当前 checkpoint 的训练配置看，前者使用了 `tuuy/lerobot_so_arm101_task0_new` 的 dataset stats，因此 state=32 大概率不是主要问题。

## 缺失第三路相机的问题

前者继承了 `pi0_base` 的三路相机 schema：

```text
base_0_rgb
left_wrist_0_rgb
right_wrist_0_rgb
```

但你的真机和 dataset 只有两路：

```text
front
wrist.left
```

Pi0 对缺失图像 key 会补空图，并将 mask 置为 false。因此如果训练和推理都稳定缺失同一第三路相机，这通常不是最主要的问题。不过这仍然说明前者使用的是一个和 SO101 dataset 不完全一致的 policy schema。

## 当前推荐路线

优先使用 noremap 版训练方式：

```bash
--policy.path=/.../pi0_base/...
--policy.input_features=null
--policy.empty_cameras=0
```

真机 async 推理时不要传 `--rename_map`：

```bash
python -m lerobot.async_inference.robot_client \
  --policy_type=pi0 \
  --pretrained_name_or_path=/m2v_intern/tujiahang/Projects/lerobot/output_lerobot/pi0/task0_lora_ablation_noremap/qkvo_mlp_r64/checkpoints/last/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average
```

同时建议真机相机分辨率和 dataset 对齐：

```text
width: 1280
height: 720
```

如果继续用 `640x480`，后者虽然已经能抓，但仍然存在从 `4:3` 到 `16:9` 的预 resize 变形风险。

## 如果一定要验证前者 remap checkpoint

可以尝试以下方向，但它们只是为了诊断，不建议作为长期主路线：

1. 修改 async inference 图像预处理，避免在 client/server observation 准备阶段根据 checkpoint feature shape 提前 resize 到 `224x224`，而是把原始分辨率交给 Pi0 内部处理。
2. 或者制作一个 remap checkpoint 的 config 副本，把 visual feature shape 从 `[3, 224, 224]` 调整为 dataset 分辨率 `[3, 720, 1280]`，让 async 路径不要提前压成方图。
3. 真机 camera 使用 `1280x720`，保持和训练 dataset 一致。
4. 确保 `--rename_map` 只在 remap checkpoint 上使用，不要加到 noremap checkpoint 的推理命令里。

## 复盘 checklist

评估一个 Pi0 LoRA checkpoint 前，建议检查：

```bash
jq '.input_features' <checkpoint>/config.json
jq '.steps[0]' <checkpoint>/policy_preprocessor.json
jq '.normalization_mapping' <checkpoint>/config.json
```

重点确认：

1. 图像 key 是否和真机 observation key 一致。
2. 图像 shape 是否和 dataset 或真机相机分辨率一致。
3. `rename_map` 是否只在需要 remap 的 checkpoint 上使用。
4. `observation.state` 维度是否能解释为真实维度加 padding。
5. action/state 的 names 顺序是否和 SO101 dataset 一致。

## 总体判断

前者 LoRA 训练 loss 低，不能证明真机可用。它很可能学到了训练分布下的动作预测，但 async 真机推理时输入图像几何发生了明显分布偏移。

后续实验建议以 noremap 版为主线：让 policy config 直接使用 SO101 dataset schema，不依赖 `rename_map` 去伪装成 `pi0_base` schema。这样训练、checkpoint 保存和真机 async 推理三者更容易保持一致。
