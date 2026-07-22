# PI0 task0 LoRA ablation

All runs use the same dataset, batch size 32, learning rate `1e-4`, and
approximately 10 epochs (`3000` optimizer steps). Checkpoints `001500` and
`003000` correspond to approximately 5 and 10 epochs.

Weights & Biases metric logging is enabled, while model artifact upload is
disabled with `--wandb.disable_artifact=true`. Checkpoints remain available
locally and are not pushed to either W&B artifact storage or Hugging Face Hub.

| Variant | Rank | Expert LoRA targets | Fully trained modules |
|---|---:|---|---|
| `qv_r16` | 16 | Q/V control group | state/action/time projections |
| `qkvo_r32` | 32 | Q/K/V/O | state/action/time projections |
| `qkvo_r64` | 64 | Q/K/V/O | state/action/time projections |
| `qkvo_mlp_r32` | 32 | Q/K/V/O + gate/up/down MLP | state/action/time projections |
| `qkvo_mlp_r64` | 64 | Q/K/V/O + gate/up/down MLP | state/action/time projections |

Run one experiment:

```bash
scripts/run_pi0_lora_task0_qkvo_r32.sh
```

Run all experiments sequentially:

```bash
scripts/run_pi0_lora_task0_ablation_all.sh
```

Override the GPU or run length when needed:

```bash
GPU_ID=2 STEPS=1500 SAVE_FREQ=1500 scripts/run_pi0_lora_task0_qkvo_mlp_r64.sh
```

Evaluate both the 5-epoch and 10-epoch checkpoints. Use the same initial object
poses, camera positions, lighting, episode duration, and success criterion for
every checkpoint.

```bash
scripts/eval_pi0_task0_real_robot.sh \
  output_lerobot/pi0/task0_lora_ablation/qkvo_r32/checkpoints/001500/pretrained_model \
  qkvo_r32_step1500
```

After manually reviewing the ten episodes, record the number of successful
episodes:

```bash
scripts/log_pi0_task0_eval_result.sh \
  qkvo_r32_step1500 \
  output_lerobot/pi0/task0_lora_ablation/qkvo_r32/checkpoints/001500/pretrained_model \
  7 10 \
  "Fixed initial pose; success means cap fully inside cup"
```

Results are appended to:

```text
output_lerobot/pi0/task0_lora_ablation/real_robot_results.csv
```
