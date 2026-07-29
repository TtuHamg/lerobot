#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/run_pi0_lora_task0_ablation.sh" qkvo_mlp_r32
