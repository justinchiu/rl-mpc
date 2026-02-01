#!/usr/bin/env bash
set -euo pipefail

ENV_ID="${ENV_ID:-CartPole-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2048}"

python third_party/cleanrl/v1.0.0/dqn.py \
  --env-id "${ENV_ID}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --learning-starts 100 \
  --train-frequency 10 \
  --buffer-size 5000 \
  --cuda False

python third_party/cleanrl/v1.0.0/ppo.py \
  --env-id "${ENV_ID}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --num-envs 1 \
  --num-steps 64 \
  --cuda False
