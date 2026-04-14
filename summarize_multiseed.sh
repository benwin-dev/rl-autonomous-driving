#!/usr/bin/env bash

python - <<'PY'
import json, glob, os, statistics
from collections import defaultdict
from pathlib import Path

rows = defaultdict(lambda: {"collision":[], "success":[], "wait":[], "reward":[]})
for f in glob.glob("results/multiseed/eval_*_seed*.json"):
    d=json.load(open(f))
    name=os.path.basename(f).split("_seed")[0].replace("eval_","")
    p=d["metrics"]["ppo"]
    rows[name]["collision"].append(p["collision_rate_percent"])
    rows[name]["success"].append(p["success_rate_percent"])
    rows[name]["wait"].append(p["avg_waiting_seconds"])
    rows[name]["reward"].append(p["avg_reward"])

summary=[]
for k,v in rows.items():
    summary.append((k, statistics.mean(v["collision"]), statistics.mean(v["success"]),
                    statistics.mean(v["wait"]), statistics.mean(v["reward"])))
summary.sort(key=lambda x:(x[1], -x[2], x[3]))

print("variant,collision_mean,success_mean,wait_mean,reward_mean")
for r in summary:
    print(",".join([r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.3f}"]))

Path("results/plots").mkdir(parents=True, exist_ok=True)
table_path = Path("results/plots/variant_summary_table.md")
lines = [
    "| Variant | Collision Mean (%) | Success Mean (%) | Waiting Mean (s) | Reward Mean |",
    "|---|---:|---:|---:|---:|",
]
for r in summary:
    lines.append(
        f"| {r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} |"
    )
table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nSaved markdown table: {table_path}")
PY
