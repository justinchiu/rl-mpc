#!/usr/bin/env bash
set -euo pipefail

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
VIDEO_COUNT="${VIDEO_COUNT:-5}"
VIDEO_EPISODES="${VIDEO_EPISODES:-1}"
DEVICE="${DEVICE:-cpu}"
NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_DIR_BASE="${VIDEO_DIR_BASE:-videos/sb3}"

if [[ "${VIDEO_COUNT}" -le 0 ]]; then
  echo "VIDEO_COUNT must be > 0" >&2
  exit 1
fi

VIDEO_EVERY_STEPS=$((TOTAL_TIMESTEPS / VIDEO_COUNT))
if [[ "${VIDEO_EVERY_STEPS}" -le 0 ]]; then
  echo "VIDEO_EVERY_STEPS computed as 0; increase TOTAL_TIMESTEPS or reduce VIDEO_COUNT" >&2
  exit 1
fi

for algo in dqn ppo; do
  video_dir="${VIDEO_DIR_BASE}/${algo}_${TOTAL_TIMESTEPS}"
  uv run python scripts/sb3_cartpole.py \
    algo="${algo}" \
    total_timesteps="${TOTAL_TIMESTEPS}" \
    video_dir="${video_dir}" \
    video_every_steps="${VIDEO_EVERY_STEPS}" \
    video_episodes="${VIDEO_EPISODES}" \
    device="${DEVICE}" \
    num_envs="${NUM_ENVS}"
done
