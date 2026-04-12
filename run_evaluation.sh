#!/usr/bin/env bash

set -euo pipefail

./test-venv/bin/python evaluate_ppo.py --episodes 100 \
  --output-json results/eval_report.json \
  --output-csv results/eval_report.csv
