#!/usr/bin/env bash

mkdir -p results/multiseed

for model in baseline collision_strong near_miss_penalty waiting_penalty; do
  for seed in 42 142 242; do
    python evaluate_ppo.py \
      --episodes 500 \
      --seed $seed \
      --model-path ppo_intersection_model_${model} \
      --output-json results/multiseed/eval_${model}_seed${seed}.json \
      --output-csv results/multiseed/eval_${model}_seed${seed}.csv
  done
done
