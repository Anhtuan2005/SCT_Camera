"""Run repeated 1/2/4-camera benchmarks and summarize the raw JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.yaml")
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--pose", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 3),
        "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


def summarize(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in raw:
        grouped.setdefault(int(item["camera_count"]), []).append(item)
    rows: list[dict[str, Any]] = []
    for camera_count, items in sorted(grouped.items()):
        aggregate = [float(item["aggregate_fps"]) for item in items]
        per_camera = [value / camera_count for value in aggregate]
        total_p50 = [float(item["latency"]["total"]["p50_ms"]) for item in items]
        total_p95 = [float(item["latency"]["total"]["p95_ms"]) for item in items]
        total_p99 = [float(item["latency"]["total"]["p99_ms"]) for item in items]
        peak_rss = [float(item["rss_peak_mb"]) for item in items]
        row: dict[str, Any] = {"camera_count": camera_count, "repetitions": len(items)}
        for name, values in (
            ("aggregate_fps", aggregate),
            ("fps_per_camera", per_camera),
            ("ai_total_p50_ms", total_p50),
            ("ai_total_p95_ms", total_p95),
            ("ai_total_p99_ms", total_p99),
            ("rss_peak_mb", peak_rss),
        ):
            stats = _mean_std(values)
            row[f"{name}_mean"] = stats["mean"]
            row[f"{name}_std"] = stats["std"]
        rows.append(row)
    return rows


def main() -> int:
    args = _args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_runs: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    for repetition in range(1, args.repetitions + 1):
        json_out = args.out_dir / f"run_{repetition:02d}.json"
        csv_out = args.out_dir / f"run_{repetition:02d}.csv"
        command = [
            sys.executable,
            str(ROOT / "scripts/benchmark_baseline.py"),
            "--source", str(args.source),
            "--settings", str(args.settings),
            "--cameras", *[str(value) for value in args.cameras],
            "--duration", str(args.duration),
            "--warmup-frames", str(args.warmup_frames),
            "--pose", args.pose,
            "--json-out", str(json_out),
            "--csv-out", str(csv_out),
        ]
        if args.camera_config:
            command.extend(["--camera-config", str(args.camera_config)])
        subprocess.run(command, cwd=ROOT, check=True)
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        metadata = {key: payload[key] for key in ("environment", "input", "model")}
        for run in payload["runs"]:
            raw_runs.append({"repetition": repetition, **run})

    rows = summarize(raw_runs)
    summary = {
        "schema_version": 1,
        "protocol": {
            "repetitions": args.repetitions,
            "camera_counts": args.cameras,
            "duration_seconds_per_case": args.duration,
            "warmup_frames": args.warmup_frames,
            "batch_size": 1,
            "source": str(args.source.resolve()),
            "note": "aggregate_fps is total throughput; fps_per_camera is aggregate_fps/camera_count",
        },
        **(metadata or {}),
        "summary": rows,
        "raw_runs": raw_runs,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={args.out_dir / 'summary.json'}")
    print(f"wrote={args.out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
