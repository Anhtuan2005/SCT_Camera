"""Run the current pipeline on labeled clips and export time-aligned behavior detections."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.behavior_engine import BehaviorEngine
from core.detector import YOLOv11Detector
from core.pose import PoseEstimator
from core.tracker import ByteTrackTracker


RAW_TO_BEHAVIOR = {
    "intrusion": "intrusion",
    "loitering": "loitering",
    "suspicious_stranger": "suspicious_behavior",
    "suspicious_theft_behavior": "theft",
}
TIME_MODULES = (
    "analytics.loitering.time",
    "analytics.suspicious_stranger.time",
    "analytics.theft_behavior.time",
    "analytics.asset_watch.time",
    "analytics.unknown_person.time",
    "analytics.fall_detection.time",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.yaml")
    parser.add_argument("--detections-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--risk-events-out", type=Path, required=True)
    parser.add_argument("--freeze-file", type=Path)
    return parser.parse_args()


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _rotate(frame: Any, rotation: str) -> Any:
    code = {
        "cw90": cv2.ROTATE_90_CLOCKWISE,
        "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }.get(str(rotation).lower())
    return cv2.rotate(frame, code) if code is not None else frame


def _resize(frame: Any, max_height: int) -> Any:
    height, width = frame.shape[:2]
    if max_height <= 0 or height <= max_height:
        return frame
    scale = max_height / float(height)
    return cv2.resize(frame, (round(width * scale), max_height), interpolation=cv2.INTER_AREA)


def _counter_snapshot(engine: BehaviorEngine, camera_id: str) -> dict[tuple[str, str], int]:
    return {
        (line_id, direction): int(value)
        for line_id, counts in engine.get_counters(camera_id).items()
        for direction, value in counts.items()
    }


def _counter_deltas(
    previous: dict[tuple[str, str], int],
    current: dict[tuple[str, str], int],
) -> list[tuple[str, str]]:
    """Return one instrumented event per new IN/OUT counter increment."""
    return [
        key
        for key, value in sorted(current.items())
        for _ in range(max(0, value - previous.get(key, 0)))
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_freeze(path: Path) -> str:
    freeze = _yaml(path)
    version = str(freeze.get("experiment_version", "")).strip()
    if not version:
        raise ValueError(f"Missing experiment_version in freeze file: {path}")
    if freeze.get("historical_logs_allowed") is not False:
        raise ValueError("Freeze file must explicitly set historical_logs_allowed: false")
    files = freeze.get("runtime_file_sha256", {})
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Missing runtime_file_sha256 in freeze file: {path}")
    for relative_path, expected in files.items():
        source = ROOT / str(relative_path)
        if not source.is_file():
            raise ValueError(f"Frozen runtime file is missing: {source}")
        actual = _sha256(source)
        if actual.lower() != str(expected).strip().lower():
            raise ValueError(f"Frozen runtime file changed: {relative_path}")
    return version


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _args()
    settings = _yaml(args.settings)
    settings.setdefault("behavior_learning", {}).update({
        "enabled": True,
        "log_candidates": False,
        "gate_alerts": False,
    })
    detector = YOLOv11Detector(settings)
    pose = PoseEstimator(settings, detector.inference_lock, detector.device, detector.use_half)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("status", "RECORDED")).strip().upper() in {"RECORDED", "READY"}
        ]
    if not manifest:
        raise ValueError("Manifest has no RECORDED/READY rows")
    versions = {
        str(row.get("experiment_version", "")).strip()
        for row in manifest
        if str(row.get("experiment_version", "")).strip()
    }
    if len(versions) > 1:
        raise ValueError(f"Manifest mixes experiment versions: {sorted(versions)}")
    experiment_version = next(iter(versions), "UNVERSIONED")
    if args.freeze_file is not None:
        frozen_version = _verify_freeze(args.freeze_file)
        if versions != {frozen_version}:
            raise ValueError(
                f"Manifest version {sorted(versions)} does not match freeze {frozen_version}"
            )
        experiment_version = frozen_version

    detections: list[dict[str, Any]] = []
    risk_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    clock = SimpleNamespace(value=0.0)
    fake_time = SimpleNamespace(monotonic=lambda: clock.value)
    with ExitStack() as stack:
        for module in TIME_MODULES:
            stack.enter_context(patch(module, fake_time))
        for item in manifest:
            clip_id = str(item.get("clip_id", "")).strip()
            group_id = str(item.get("group_id", clip_id)).strip() or clip_id
            split = str(item.get("split", "")).strip()
            video_path = Path(str(item.get("video_path", "")).strip())
            config_path = Path(str(item.get("camera_config_path", "")).strip())
            if not clip_id or not video_path.is_file() or not config_path.is_file():
                raise ValueError(f"Invalid manifest row: {item}")
            config = _yaml(config_path)
            config_version = str(config.get("experiment_version", "")).strip()
            if experiment_version != "UNVERSIONED" and config_version != experiment_version:
                raise ValueError(
                    f"Config version {config_version or 'MISSING'} does not match "
                    f"{experiment_version}: {config_path}"
                )
            config["source"] = str(video_path)
            config.setdefault("camera_id", clip_id)
            config.setdefault("name", clip_id)
            camera_id = str(config["camera_id"])
            local_settings = copy.deepcopy(settings)
            tracker = ByteTrackTracker(detector, local_settings)
            engine = BehaviorEngine(local_settings)
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {video_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
            frame_count = 0
            wall_started = time.perf_counter()
            counters = _counter_snapshot(engine, camera_id)
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                clock.value = frame_count / fps
                frame = _resize(_rotate(frame, str(config.get("frame_rotation", "none"))), int(local_settings.get("pipeline", {}).get("processing_max_height", 0)))
                started = time.perf_counter()
                objects = tracker.track(frame)
                if bool(local_settings.get("pose", {}).get("enabled", True)):
                    objects = pose.attach(frame, objects)
                objects = engine.label_objects(objects, config, frame)
                alerts = engine.analyze(objects, config, frame.shape)
                processing_ms = (time.perf_counter() - started) * 1000.0
                for alert in alerts:
                    raw_type = str(alert.get("type", ""))
                    behavior = RAW_TO_BEHAVIOR.get(raw_type)
                    if behavior is None and raw_type.startswith("loitering"):
                        behavior = "loitering"
                    if behavior is None:
                        continue
                    detections.append({
                        "experiment_version": experiment_version,
                        "clip_id": clip_id,
                        "group_id": group_id,
                        "split": split,
                        "behavior": behavior,
                        "raw_alert_type": raw_type,
                        "behavior_event_id": alert.get("behavior_event_id", ""),
                        "video_time_seconds": round(clock.value, 3),
                        "frame_index": frame_count,
                        "direction": alert.get("direction", ""),
                        "line_id": alert.get("line_id", ""),
                        "track_id": alert.get("track_id", ""),
                        "frame_processing_ms": round(processing_ms, 3),
                    })
                    risk_events.append({
                        "experiment_version": experiment_version,
                        "event_id": alert.get("behavior_event_id", ""),
                        "clip_id": clip_id,
                        "group_id": group_id,
                        "split": split,
                        "video_time_seconds": round(clock.value, 3),
                        "camera_id": camera_id,
                        "alert_type": raw_type,
                        "features": alert.get("behavior_features", {}),
                        "label": None,
                    })
                updated = _counter_snapshot(engine, camera_id)
                for event_index, key in enumerate(_counter_deltas(counters, updated), start=1):
                    detections.append({
                        "experiment_version": experiment_version,
                        "clip_id": clip_id,
                        "group_id": group_id,
                        "split": split,
                        "behavior": "line_crossing",
                        "raw_alert_type": "counter_delta",
                        "behavior_event_id": "",
                        "instrumentation_event_id": (
                            f"{clip_id}:line:{frame_count}:{key[0]}:{key[1]}:{event_index}"
                        ),
                        "video_time_seconds": round(clock.value, 3),
                        "frame_index": frame_count,
                        "direction": key[1].upper(),
                        "line_id": key[0],
                        "track_id": "",
                        "frame_processing_ms": round(processing_ms, 3),
                    })
                counters = updated
                frame_count += 1
            capture.release()
            summaries.append({
                "experiment_version": experiment_version,
                "clip_id": clip_id,
                "video_path": str(video_path.resolve()),
                "video_sha256": _sha256(video_path),
                "camera_config_path": str(config_path.resolve()),
                "camera_config_sha256": _sha256(config_path),
                "frames": frame_count,
                "fps": round(fps, 3),
                "duration_seconds": round(frame_count / fps, 3),
                "detections": sum(row["clip_id"] == clip_id for row in detections),
                "wall_seconds": round(time.perf_counter() - wall_started, 3),
            })

    detection_fields = ["experiment_version", "clip_id", "group_id", "split", "behavior", "raw_alert_type", "behavior_event_id", "instrumentation_event_id", "video_time_seconds", "frame_index", "direction", "line_id", "track_id", "frame_processing_ms"]
    _write(args.detections_out, detections, detection_fields)
    _write(args.summary_out, summaries, ["experiment_version", "clip_id", "video_path", "video_sha256", "camera_config_path", "camera_config_sha256", "frames", "fps", "duration_seconds", "detections", "wall_seconds"])
    args.risk_events_out.parent.mkdir(parents=True, exist_ok=True)
    args.risk_events_out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in risk_events),
        encoding="utf-8",
    )
    print(f"clips={len(summaries)} detections={len(detections)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
