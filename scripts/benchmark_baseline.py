"""Measure SCT Camera throughput and per-stage AI latency."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.behavior_engine import BehaviorEngine
from core.detector import YOLOv11Detector
from core.pose import PoseEstimator
from core.tracker import ByteTrackTracker


STAGES = ("track", "pose", "identity", "behavior", "total")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local benchmark video")
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.yaml")
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--pose", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--json-out", type=Path, default=ROOT / "data/benchmarks/baseline.json")
    parser.add_argument("--csv-out", type=Path, default=ROOT / "data/benchmarks/baseline.csv")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _pose_enabled(mode: str, settings: dict[str, Any], config: dict[str, Any]) -> bool:
    if mode == "on":
        return bool(settings.get("pose", {}).get("enabled", True))
    if mode == "off":
        return False
    if not bool(settings.get("pose", {}).get("enabled", True)):
        return False
    theft = settings.get("behavior", {}).get("theft", {})
    if not bool(theft.get("enabled", True)):
        return False
    return any(
        isinstance(zone, dict)
        and str(zone.get("type", zone.get("zone_type", ""))) in {"all", "asset_watch"}
        and len(zone.get("polygon", [])) >= 3
        for zone in config.get("zones", [])
    )


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    p50, p95, p99 = np.percentile(np.asarray(values), [50, 95, 99])
    return {
        "p50_ms": round(float(p50), 3),
        "p95_ms": round(float(p95), 3),
        "p99_ms": round(float(p99), 3),
    }


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout.splitlines()[0]
        name, driver, total, used, utilization = [item.strip() for item in line.split(",")]
        return {
            "name": name,
            "driver": driver,
            "memory_total_mb": int(total),
            "memory_used_mb": int(used),
            "utilization_percent": int(utilization),
        }
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return {"available": False}


def _resize(frame: np.ndarray, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if max_height <= 0 or height <= max_height:
        return frame
    scale = max_height / float(height)
    return cv2.resize(frame, (max(1, int(width * scale)), max_height), interpolation=cv2.INTER_AREA)


def _worker(
    worker_id: int,
    source: Path,
    settings: dict[str, Any],
    config_template: dict[str, Any],
    detector: YOLOv11Detector,
    pose_estimator: PoseEstimator,
    identity_resolver: Any,
    duration: float,
    max_frames: int,
    warmup_frames: int,
    run_pose: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(config_template)
    config.update(
        {
            "camera_id": f"benchmark_{worker_id + 1}",
            "name": f"Benchmark {worker_id + 1}",
            "source": str(source),
            "enabled": True,
        }
    )
    local_settings = copy.deepcopy(settings)
    local_settings.setdefault("behavior_learning", {})["enabled"] = False
    tracker = ByteTrackTracker(detector, local_settings)
    behavior = BehaviorEngine(local_settings, identity_resolver=identity_resolver)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open benchmark source: {source}")

    max_height = int(local_settings.get("pipeline", {}).get("processing_max_height", 0))
    process = psutil.Process(os.getpid())
    samples = {stage: [] for stage in STAGES}
    rss_peak = process.memory_info().rss
    measured_frames = 0
    decoded_frames = 0
    started_at: float | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker.reset()
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
            decoded_frames += 1
            frame = _resize(frame, max_height)

            t0 = time.perf_counter()
            objects = tracker.track(frame)
            t1 = time.perf_counter()
            if run_pose:
                objects = pose_estimator.attach(frame, objects)
            t2 = time.perf_counter()
            objects = behavior.label_objects(objects, config, frame)
            t3 = time.perf_counter()
            behavior.analyze(objects, config, frame.shape)
            t4 = time.perf_counter()

            if decoded_frames <= warmup_frames:
                continue
            if started_at is None:
                started_at = t0
            measured_frames += 1
            samples["track"].append((t1 - t0) * 1000.0)
            samples["pose"].append((t2 - t1) * 1000.0)
            samples["identity"].append((t3 - t2) * 1000.0)
            samples["behavior"].append((t4 - t3) * 1000.0)
            samples["total"].append((t4 - t0) * 1000.0)
            rss_peak = max(rss_peak, process.memory_info().rss)

            elapsed = t4 - started_at
            if max_frames > 0 and measured_frames >= max_frames:
                break
            if duration > 0 and elapsed >= duration:
                break
    finally:
        capture.release()

    elapsed = max(time.perf_counter() - (started_at or time.perf_counter()), 0.0)
    return {
        "worker_id": worker_id + 1,
        "frames": measured_frames,
        "elapsed_seconds": round(elapsed, 3),
        "fps": round(measured_frames / elapsed, 3) if elapsed > 0 else 0.0,
        "rss_peak_mb": round(rss_peak / (1024 * 1024), 3),
        "latency": {stage: _percentiles(samples[stage]) for stage in STAGES},
        "samples": samples,
    }


def _run_case(
    camera_count: int,
    args: argparse.Namespace,
    settings: dict[str, Any],
    config: dict[str, Any],
    detector: YOLOv11Detector,
    pose_estimator: PoseEstimator,
    identity_resolver: Any,
    run_pose: bool,
) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    rss_start = process.memory_info().rss
    gpu_start = _gpu_snapshot()
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=camera_count) as executor:
        futures = [
            executor.submit(
                _worker,
                index,
                args.source,
                settings,
                config,
                detector,
                pose_estimator,
                identity_resolver,
                args.duration,
                args.max_frames,
                args.warmup_frames,
                run_pose,
            )
            for index in range(camera_count)
        ]
        workers = [future.result() for future in futures]
    wall_elapsed = time.perf_counter() - wall_start
    worker_samples = [worker.pop("samples") for worker in workers]
    aggregate_samples = {
        stage: [value for samples in worker_samples for value in samples[stage]]
        for stage in STAGES
    }
    frames_total = sum(worker["frames"] for worker in workers)
    measured_elapsed = max((worker["elapsed_seconds"] for worker in workers), default=0.0)
    return {
        "camera_count": camera_count,
        "frames_total": frames_total,
        "wall_seconds": measured_elapsed,
        "execution_wall_seconds": round(wall_elapsed, 3),
        "aggregate_fps": round(frames_total / measured_elapsed, 3) if measured_elapsed > 0 else 0.0,
        "rss_start_mb": round(rss_start / (1024 * 1024), 3),
        "rss_end_mb": round(process.memory_info().rss / (1024 * 1024), 3),
        "rss_peak_mb": max(worker["rss_peak_mb"] for worker in workers),
        "gpu_start": gpu_start,
        "gpu_end": _gpu_snapshot(),
        "latency": {stage: _percentiles(aggregate_samples[stage]) for stage in STAGES},
        "workers": workers,
    }


def _write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "camera_count",
        "frames_total",
        "wall_seconds",
        "execution_wall_seconds",
        "aggregate_fps",
        "rss_start_mb",
        "rss_end_mb",
        "rss_peak_mb",
    ] + [f"{stage}_{percentile}" for stage in STAGES for percentile in ("p50_ms", "p95_ms", "p99_ms")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {field: run.get(field) for field in fields}
            for stage in STAGES:
                for percentile, value in run["latency"][stage].items():
                    row[f"{stage}_{percentile}"] = value
            writer.writerow(row)


def main() -> int:
    args = _parse_args()
    args.source = args.source.resolve()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if any(count not in {1, 2} for count in args.cameras):
        raise ValueError("--cameras accepts only 1 and/or 2")
    if args.duration <= 0 and args.max_frames <= 0:
        raise ValueError("Set --duration or --max-frames to a positive value")

    settings = _load_yaml(args.settings)
    settings.setdefault("behavior_learning", {})["enabled"] = False
    config = _load_yaml(args.camera_config) if args.camera_config else {
        "camera_id": "benchmark",
        "name": "Benchmark",
        "zones": [],
        "lines": [],
        "auto_global_zone": False,
    }
    run_pose = _pose_enabled(args.pose, settings, config)
    detector = YOLOv11Detector(settings)
    pose_estimator = PoseEstimator(
        settings,
        detector.inference_lock,
        detector.device,
        detector.use_half,
    )
    if run_pose:
        run_pose = pose_estimator._ensure_model()
    identity_resolver = BehaviorEngine(settings).identity_resolver

    runs = [
        _run_case(
            count,
            args,
            settings,
            config,
            detector,
            pose_estimator,
            identity_resolver,
            run_pose,
        )
        for count in args.cameras
    ]
    capture = cv2.VideoCapture(str(args.source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "ram_total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 3),
            "gpu": _gpu_snapshot(),
        },
        "input": {
            "source_type": "local_video",
            "filename": args.source.name,
            "resolution": [width, height],
            "duration_limit_seconds": args.duration,
            "max_frames": args.max_frames,
            "warmup_frames": args.warmup_frames,
            "pose_mode": args.pose,
            "pose_executed": run_pose,
        },
        "model": {
            "detection": detector.model_path,
            "pose": pose_estimator.model_path if run_pose else None,
            "device": detector.device,
            "imgsz": detector.imgsz,
        },
        "runs": runs,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(args.csv_out, runs)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.csv_out}")
    for run in runs:
        total = run["latency"]["total"]
        print(
            f"{run['camera_count']} camera(s): {run['aggregate_fps']:.2f} FPS, "
            f"AI p50/p95/p99={total['p50_ms']:.1f}/{total['p95_ms']:.1f}/{total['p99_ms']:.1f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
