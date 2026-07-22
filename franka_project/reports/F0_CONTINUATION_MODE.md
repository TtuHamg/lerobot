# PI0 F0 全参数 continuation 模式

## 结论

训练入口 `franka_project/scripts/train_pi0_full.py` 现在支持 `stage: F0-CONT-15`。本次冻结配置为：

- source：F0 `step-037330`
- continuation：额外 20 epoch
- 每个 epoch：3733 optimizer steps
- 新阶段长度：74660 steps
- 最终累计 global step：111990
- 学习率：从 source 当前值 `1e-6` 开始，无 warmup，cosine 衰减到 `1e-7`

配置文件：`franka_project/configs/train/pi0_full_eef_f0_continue20_15hz.yaml`

## continuation 到底恢复什么

新阶段会创建一个新的本地 run 和新的 W&B run；旧 F0 run 始终只读。

| 状态 | 行为 |
|---|---|
| PI0 weights | 从 F0 source checkpoint 严格加载 |
| AdamW moments / optimizer step | 保留；不会重新初始化 |
| Python / NumPy / Torch / CUDA rank RNG | 分 rank 保留 |
| DataLoader global offset | 从 global step 37330 延续 |
| global step | 从 37330 累计到 111990 |
| 旧 F0 scheduler | 不加载，因为它已经走完原 10 epoch |
| 新 scheduler | phase step 0 开始，`1e-6 -> 1e-7` |

特别处理了 PyTorch `LambdaLR` 的一个陷阱：source AdamW 中当前 `lr=1e-6`，但旧 `initial_lr` 仍是 `1e-5`。新 scheduler 创建前会验证当前 LR，并把 `lr`、`initial_lr` 和 scheduler `base_lrs` 都冻结为 `1e-6`，因此第一步不会跳回旧峰值。

## 推荐运行方式

先进入用户指定的环境：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot
source /ytech_milm_intern/tujiahang/.bashrc
conda activate lerobot
unset LD_LIBRARY_PATH
export PYTHONPATH=franka_project/src
```

可以直接完成全部额外 20 epoch：

```bash
torchrun --standalone --nproc_per_node=2 \
  franka_project/scripts/train_pi0_full.py \
  --config franka_project/configs/train/pi0_full_eef_f0_continue20_15hz.yaml
```

更稳妥的启动方式是先做 2-step smoke。这里 `--max-steps` 是累计 global step，所以 source `37330 + 2 = 37332`：

```bash
torchrun --standalone --nproc_per_node=2 \
  franka_project/scripts/train_pi0_full.py \
  --config franka_project/configs/train/pi0_full_eef_f0_continue20_15hz.yaml \
  --max-steps 37332
```

该命令会创建并打印一个新的 continuation run 路径。确认 smoke 后，从这个**新 run**继续：

```bash
torchrun --standalone --nproc_per_node=2 \
  franka_project/scripts/train_pi0_full.py \
  --config franka_project/configs/train/pi0_full_eef_f0_continue20_15hz.yaml \
  --resume-run-dir <新生成的-continuation-run-绝对路径>
```

不要把旧 F0 run 传给 `--resume-run-dir`。旧 F0 checkpoint 已经由 YAML 的 `continuation.source_checkpoint` 指定；`--resume-run-dir` 只用于恢复新 continuation run 自己的中断。

## 审计和失败保护

新 run 创建前会验证 source 是已完成且原子发布的 F0 最后 checkpoint，并核对：

- source run/config/checkpoint manifest 的冻结 SHA256；
- dataset root、scope、profile、monitor subset、steps per epoch；
- world size、batch、gradient accumulation、seed、shuffle 和 optimizer 超参数；
- 776 个有序 optimizer 参数、完整 AdamW state 文件、两个不同的 rank RNG；
- source scheduler 和 AdamW 当前 LR 都是 `1e-6`；
- full-parameter、无 LoRA 的 PI0 model ledger。

continuation checkpoint 同时记录 global step、phase step 和 source identity。同一新 run 恢复时，会加载它自己的 weights、AdamW、新 scheduler 和 rank RNG，并拒绝 source lineage 或双 step 轴不一致。

在 continuation 完成前不要移动或删除旧 F0 source checkpoint：即使新 checkpoint 本身包含完整 weights/optimizer，恢复入口仍会重新核对旧 source identity。这里保留的是确定性 `CartesianAnchorDataset` 的 global sampler/data offset；DataLoader worker 的私有随机种子没有序列化，因此该保证不应外推到未来带随机 augmentation 的 dataset。

## 存储提醒

当前一个完整 checkpoint 约包含 7.3 GB model 和 14.2 GB optimizer state，即约 21.5 GB。按每 epoch 保存一次，额外 20 epoch 约需 430 GB（不含日志和临时 staging）；W&B 只记录 metrics，不上传 checkpoint/model artifact。

## 本轮代码验证

- continuation source 只读 preflight：PASS
- source identity：`aad7ecc566e009a84f000d46ca51c32e536f8423851534a1e7573a30e7781ac7`
- continuation 专项测试：24 passed（含真实 optimizer/scheduler state roundtrip）
- `franka_project/tests` 全量 CPU 测试：159 passed
- 本轮没有加载 3B model/optimizer tensor，也没有启动 GPU 训练或创建 W&B run
