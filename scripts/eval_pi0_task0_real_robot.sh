#!/usr/bin/env bash
set -euo pipefail

# Records real-robot evaluation episodes for one checkpoint.
#
# Usage:
#   scripts/eval_pi0_task0_real_robot.sh <checkpoint_path> <run_name>
#
# Optional environment overrides:
#   GPU_ID=1 NUM_EPISODES=10 EPISODE_TIME_S=60 RESET_TIME_S=15
#   FOLLOWER_PORT=/dev/ttyACM1 FRONT_CAMERA=0 WRIST_CAMERA=2

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <checkpoint_path> <run_name>" >&2
  exit 2
fi

checkpoint_path="$1"
run_name="$2"
safe_run_name="${run_name//[^a-zA-Z0-9_-]/_}"

if [[ ! -d "${checkpoint_path}" ]]; then
  echo "Checkpoint directory does not exist: ${checkpoint_path}" >&2
  exit 1
fi

cd /m2v_intern/tujiahang/Projects/lerobot
source /ytech_milm_intern/tujiahang/.bashrc
conda activate lerobot

export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"

exec lerobot-rollout \
  --strategy.type=episodic \
  --policy.path="${checkpoint_path}" \
  --policy.device=cuda \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT:-/dev/ttyACM1}" \
  --robot.id=tjh_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA:-0}, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist.left: {type: opencv, index_or_path: ${WRIST_CAMERA:-2}, width: 1280, height: 720, fps: 30, fourcc: MJPG}}" \
  --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist.left":"observation.images.left_wrist_0_rgb"}' \
  --task="Place the black bottle cap into the white paper cup" \
  --dataset.repo_id="tuuy/eval_pi0_task0_${safe_run_name}" \
  --dataset.single_task="Place the black bottle cap into the white paper cup" \
  --dataset.num_episodes="${NUM_EPISODES:-10}" \
  --dataset.episode_time_s="${EPISODE_TIME_S:-60}" \
  --dataset.reset_time_s="${RESET_TIME_S:-15}" \
  --dataset.push_to_hub=false \
  --display_data=true
