#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_pi0_lora_task0_ablation.sh <qv_r16|qkvo_r32|qkvo_r64|qkvo_mlp_r32|qkvo_mlp_r64>
#
# Optional environment overrides:
#   GPU_ID=1 STEPS=3000 SAVE_FREQ=1500 LOG_FREQ=50 WANDB_ENABLE=true

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <qv_r16|qkvo_r16|qkvo_r32|qkvo_r64|qkvo_mlp_r32|qkvo_mlp_r64>" >&2
  exit 2
fi

variant="$1"

case "${variant}" in
  qv_r16)
    rank=16
    alpha=32
    target_modules='.*\.gemma_expert\..*\.self_attn\.(q|v)_proj'
    ;;
  qkvo_r16)
    rank=16
    alpha=32
    target_modules='.*\.gemma_expert\..*\.self_attn\.(q|k|v|o)_proj'
    ;;
  qkvo_r32)
    rank=32
    alpha=64
    target_modules='.*\.gemma_expert\..*\.self_attn\.(q|k|v|o)_proj'
    ;;
  qkvo_r64)
    rank=64
    alpha=128
    target_modules='.*\.gemma_expert\..*\.self_attn\.(q|k|v|o)_proj'
    ;;
  qkvo_mlp_r32)
    rank=32
    alpha=64
    target_modules='.*\.gemma_expert\..*\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'
    ;;
  qkvo_mlp_r64)
    rank=64
    alpha=128
    target_modules='.*\.gemma_expert\..*\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

cd /m2v_intern/tujiahang/Projects/lerobot
source /ytech_milm_intern/tujiahang/.bashrc
conda activate lerobot

export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"

steps="${STEPS:-20000}"
save_freq="${SAVE_FREQ:-2000}"
log_freq="${LOG_FREQ:-200}"
wandb_enable="${WANDB_ENABLE:-true}"
run_name="pi0_task0_lora_${variant}_lr1e4"
output_dir="output_lerobot/pi0/task0_lora_ablation_noremap/${variant}"

# 9593 frames / batch size 32 ~= 300 optimizer steps per epoch.
# The default 3000 steps therefore cover approximately 10 epochs, with
# checkpoints at approximately 5 and 10 epochs.
#
# The projection modules below are intentionally excluded from target_modules.
# PEFT's modules_to_save mechanism keeps them fully trainable instead of adding
# low-rank adapters to them.
exec lerobot-train \
  --policy.path=/ytech_milm_intern/tujiahang/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/25c379b52ba2ff8788cab921758a3cc3fe3f77f2 \
  --policy.input_features=null \
  --policy.push_to_hub=false \
  --policy.empty_cameras=0 \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.optimizer_lr=1e-4 \
  --dataset.repo_id=tuuy/lerobot_so_arm101_task0_new \
  --dataset.root=/ytech_milm_intern/tujiahang/.cache/huggingface/lerobot/tuuy/lerobot_so_arm101_task0_new \
  --dataset.streaming=false \
  --output_dir="${output_dir}" \
  --job_name="${run_name}" \
  --peft.method_type=LORA \
  --peft.target_modules="${target_modules}" \
  --peft.full_training_modules='["state_proj","action_in_proj","action_out_proj","action_time_mlp_in","action_time_mlp_out"]' \
  --peft.r="${rank}" \
  --peft.lora_alpha="${alpha}" \
  --steps="${steps}" \
  --save_freq="${save_freq}" \
  --eval_freq=0 \
  --log_freq="${log_freq}" \
  --batch_size=32 \
  --wandb.enable="${wandb_enable}" \
  --wandb.project=Lerobot_Project
