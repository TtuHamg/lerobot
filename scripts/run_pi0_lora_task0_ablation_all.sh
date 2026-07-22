#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

for variant in qv_r16 qkvo_r32 qkvo_r64 qkvo_mlp_r32 qkvo_mlp_r64; do
  "${script_dir}/run_pi0_lora_task0_ablation.sh" "${variant}"
done
