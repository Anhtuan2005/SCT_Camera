"""Plot aggregate/per-camera throughput and AI latency from revision benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = payload["summary"]
    cameras = [row["camera_count"] for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    axes[0].errorbar(cameras, [row["aggregate_fps_mean"] for row in rows], yerr=[row["aggregate_fps_std"] for row in rows], marker="o", capsize=4, label="Aggregate FPS")
    axes[0].errorbar(cameras, [row["fps_per_camera_mean"] for row in rows], yerr=[row["fps_per_camera_std"] for row in rows], marker="s", capsize=4, label="FPS per camera")
    axes[0].set(xlabel="Concurrent cameras", ylabel="Frames per second", xticks=cameras, title="Throughput scalability")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].errorbar(cameras, [row["ai_total_p50_ms_mean"] for row in rows], yerr=[row["ai_total_p50_ms_std"] for row in rows], marker="o", capsize=4, label="p50")
    axes[1].errorbar(cameras, [row["ai_total_p95_ms_mean"] for row in rows], yerr=[row["ai_total_p95_ms_std"] for row in rows], marker="s", capsize=4, label="p95")
    axes[1].errorbar(cameras, [row["ai_total_p99_ms_mean"] for row in rows], yerr=[row["ai_total_p99_ms_std"] for row in rows], marker="^", capsize=4, label="p99")
    axes[1].set(xlabel="Concurrent cameras", ylabel="AI latency (ms)", xticks=cameras, title="End-to-end AI-stage latency")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    figure.suptitle("SCT Camera revision benchmark (mean ± SD, n=3)", fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=220)
    plt.close(figure)
    print(f"wrote={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
