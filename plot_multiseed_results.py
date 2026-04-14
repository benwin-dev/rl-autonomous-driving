import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


METRIC_LABELS = {
    "collision_rate_percent": "Collision Rate (%)",
    "success_rate_percent": "Success Rate (%)",
    "avg_waiting_seconds": "Average Waiting Time (s)",
    "avg_reward": "Average Episode Reward",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aggregated multi-seed PPO variant results."
    )
    parser.add_argument(
        "--input-glob",
        default="results/multiseed/eval_*_seed*.json",
        help="Glob for evaluation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory to save generated PNG plots.",
    )
    return parser.parse_args()


def load_data(input_glob: str):
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_glob}")

    by_variant = defaultdict(lambda: defaultdict(list))
    for path in files:
        data = json.loads(Path(path).read_text())
        variant = Path(path).name.split("_seed")[0].replace("eval_", "")
        ppo = data["metrics"]["ppo"]
        for metric in METRIC_LABELS:
            by_variant[variant][metric].append(float(ppo[metric]))
    return by_variant


def summarize(by_variant):
    summary = {}
    for variant, metric_map in by_variant.items():
        summary[variant] = {}
        for metric, values in metric_map.items():
            summary[variant][metric] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    return summary


def make_bar_plot(summary, metric_key: str, output_dir: Path):
    variants = sorted(summary.keys())
    means = [summary[v][metric_key]["mean"] for v in variants]
    stds = [summary[v][metric_key]["std"] for v in variants]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(variants, means, yerr=stds, capsize=5)
    plt.ylabel(METRIC_LABELS[metric_key])
    plt.xlabel("Reward Variant")
    plt.title(f"{METRIC_LABELS[metric_key]} by Reward Variant (mean +/- std over seeds)")
    plt.xticks(rotation=20, ha="right")

    for bar, val in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = output_dir / f"{metric_key}_bar.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def write_summary_csv(summary, output_dir: Path):
    out_path = output_dir / "variant_summary.csv"
    headers = [
        "variant",
        "collision_mean",
        "collision_std",
        "success_mean",
        "success_std",
        "wait_mean",
        "wait_std",
        "reward_mean",
        "reward_std",
    ]
    lines = [",".join(headers)]
    for variant in sorted(summary.keys()):
        s = summary[variant]
        lines.append(
            ",".join(
                [
                    variant,
                    f"{s['collision_rate_percent']['mean']:.6f}",
                    f"{s['collision_rate_percent']['std']:.6f}",
                    f"{s['success_rate_percent']['mean']:.6f}",
                    f"{s['success_rate_percent']['std']:.6f}",
                    f"{s['avg_waiting_seconds']['mean']:.6f}",
                    f"{s['avg_waiting_seconds']['std']:.6f}",
                    f"{s['avg_reward']['mean']:.6f}",
                    f"{s['avg_reward']['std']:.6f}",
                ]
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_variant = load_data(args.input_glob)
    summary = summarize(by_variant)

    generated = []
    for metric_key in METRIC_LABELS:
        generated.append(make_bar_plot(summary, metric_key, output_dir))
    generated.append(write_summary_csv(summary, output_dir))

    print("Generated files:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
