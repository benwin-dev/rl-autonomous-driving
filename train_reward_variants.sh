#!/usr/bin/env bash

set -euo pipefail

TIMESTEPS="${1:-50000}"

./test-venv/bin/python train_ppo.py \
  --reward-variant collision_strong \
  --timesteps "${TIMESTEPS}" \
  --model-path ppo_intersection_model_collision_strong

./test-venv/bin/python train_ppo.py \
  --reward-variant near_miss_penalty \
  --timesteps "${TIMESTEPS}" \
  --model-path ppo_intersection_model_near_miss_penalty

./test-venv/bin/python train_ppo.py \
  --reward-variant waiting_penalty \
  --timesteps "${TIMESTEPS}" \
  --model-path ppo_intersection_model_waiting_penalty
