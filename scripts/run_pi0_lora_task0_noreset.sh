#!/usr/bin/env bash
set -euo pipefail

cd /m2v_intern/tujiahang/Projects/lerobot
source /ytech_milm_intern/tujiahang/.bashrc
conda activate lerobot

export CUDA_VISIBLE_DEVICES=4

exec lerobot-train \
  --policy.path=/ytech_milm_intern/tujiahang/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/25c379b52ba2ff8788cab921758a3cc3fe3f77f2 \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist.left":"observation.images.left_wrist_0_rgb"}' \
  --dataset.repo_id=tuuy/lerobot_so_arm101_task0_new_noreset \
  --dataset.root=/ytech_milm_intern/tujiahang/.cache/huggingface/lerobot/tuuy/lerobot_so_arm101_task0_new_noreset \
  --dataset.streaming=false \
  --output_dir=output_lerobot/pi0/task0_noreset_lora \
  --job_name=pi0_task0_noreset_lora \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --steps=50000 \
  --save_freq=4000 \
  --eval_freq=20000 \
  --log_freq=200 \
  --batch_size=32 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --wandb.project=Lerobot_Project
