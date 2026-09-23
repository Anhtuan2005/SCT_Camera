"""Measure live RTSP capture freshness and MJPEG publication throughput."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.frame_buffer import FrameBuffer
from core.pipeline import CameraPipeline, _LatestFrameCapture
from utils.drawing import draw_annotations


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=ROOT / "config/cameras/imou_camera.yaml",
    )
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.yaml")
    parser.add_argument("--duration", type=float, default=10.0)
    return parser.parse_args()


def _percentile(values: list[float], percentile: int) -> float:
    return round(float(np.percentile(values, percentile)), 2) if values else 0.0


def main() -> None:
    args = _args()
    camera = yaml.safe_load(args.camera_config.read_text(encoding="utf-8")) or {}
    settings = yaml.safe_load(args.settings.read_text(encoding="utf-8")) or {}
    source = CameraPipeline._parse_source(camera.get("source", 0))
    if not CameraPipeline._is_rtsp_source(source):
        raise SystemExit("benchmark_live_stream requires an RTSP camera config")

    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise SystemExit("cannot open RTSP camera")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    pipeline_settings = settings.get("pipeline", {})
    processing_height = int(pipeline_settings.get("processing_max_height", 720))
    buffer = FrameBuffer(max_height=int(pipeline_settings.get("stream_max_height", 720)))
    reader = _LatestFrameCapture(capture, threading.Event())
    reader.start()
    latencies: list[float] = []
    frames = 0
    sequence = 0
    started = time.monotonic()

    try:
        while time.monotonic() - started < max(0.1, args.duration):
            captured = reader.read_latest(sequence)
            if captured is None:
                break
            sequence = captured.sequence
            frame = CameraPipeline._apply_frame_rotation(captured.frame, camera)
            height, width = frame.shape[:2]
            if processing_height > 0 and height > processing_height:
                scale = processing_height / float(height)
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), processing_height),
                    interpolation=cv2.INTER_AREA,
                )
            frame = draw_annotations(frame, [], camera, {}, {}, [], [])
            buffer.update(
                frame,
                object_count=0,
                captured_at_monotonic=captured.captured_at,
                capture_fps=captured.capture_fps,
                dropped_capture_frames=captured.dropped_frames,
            )
            latencies.append(buffer.snapshot().capture_to_publish_ms)
            frames += 1
    finally:
        reader.stop()

    elapsed = max(time.monotonic() - started, 1e-6)
    snapshot = buffer.snapshot()
    print(
        json.dumps(
            {
                "camera_id": camera.get("camera_id", args.camera_config.stem),
                "source_fps": round(source_fps, 2),
                "capture_fps": round(snapshot.capture_fps, 2),
                "publish_fps": round(frames / elapsed, 2),
                "dropped_capture_frames": snapshot.dropped_capture_frames,
                "capture_to_publish_ms": {
                    "p50": _percentile(latencies, 50),
                    "p95": _percentile(latencies, 95),
                    "max": round(max(latencies), 2) if latencies else 0.0,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
