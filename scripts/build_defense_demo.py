"""Build a self-contained backup video for the thesis defense."""

from __future__ import annotations

import copy
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.behavior_engine import BehaviorEngine
from core.detector import YOLOv11Detector
from core.pose import PoseEstimator
from core.tracker import ByteTrackTracker
from utils.drawing import draw_annotations


CONFIG_PATH = ROOT / "experiments/priority_19_08/manual/configs_pilot/P01_theft_positive_r1_daylight_side_oblique.yaml"
SETTINGS_PATH = ROOT / "config/settings.yaml"
OUTPUT_DIR = ROOT / "outputs/defense_demo"
OUTPUT_PATH = OUTPUT_DIR / "SCT_Camera_Defense_Demo.mp4"
OUTPUT_SIZE = (1280, 720)
OUTPUT_FPS = 25.0
TITLE_SECONDS = 3
SUMMARY_SECONDS = 5
TIME_MODULES = (
    "analytics.loitering.time",
    "analytics.suspicious_stranger.time",
    "analytics.theft_behavior.time",
    "analytics.asset_watch.time",
    "analytics.unknown_person.time",
    "analytics.fall_detection.time",
    "utils.drawing.time",
)


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _rotate(frame: np.ndarray, rotation: str) -> np.ndarray:
    code = {
        "cw90": cv2.ROTATE_90_CLOCKWISE,
        "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }.get(str(rotation).lower())
    return cv2.rotate(frame, code) if code is not None else frame


def _resize_height(frame: np.ndarray, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= max_height:
        return frame
    scale = max_height / float(height)
    return cv2.resize(frame, (round(width * scale), max_height), interpolation=cv2.INTER_AREA)


def _put(
    frame: np.ndarray,
    text: str,
    point: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (238, 244, 250),
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (25, 12, 5),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _title_card() -> np.ndarray:
    frame = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (66, 38, 24), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (18, OUTPUT_SIZE[1]), (214, 190, 12), -1)
    cv2.circle(frame, (108, 124), 26, (214, 190, 12), -1, cv2.LINE_AA)
    cv2.circle(frame, (108, 124), 11, (66, 38, 24), 4, cv2.LINE_AA)
    _put(frame, "SCT CAMERA", (160, 138), 1.45, (255, 255, 255), 3)
    _put(frame, "BACKUP DEMO FOR THESIS DEFENSE", (94, 245), 0.86, (214, 190, 12), 2)
    _put(frame, "Scenario: suspicious theft behavior near a watched motorcycle", (94, 320), 0.73)
    _put(frame, "YOLOv11  |  ByteTrack  |  YOLOv11-Pose  |  Behavior Analysis", (94, 375), 0.67, (190, 205, 224))
    cv2.rectangle(frame, (92, 456), (1186, 554), (50, 28, 17), -1)
    cv2.rectangle(frame, (92, 456), (1186, 554), (214, 190, 12), 2)
    _put(frame, "OFFLINE REPLAY", (124, 500), 0.72, (214, 190, 12), 2)
    _put(frame, "Recorded input, processed by the same experimental AI pipeline", (124, 535), 0.62)
    _put(frame, "No camera, Internet, or Telegram connection is required.", (94, 646), 0.58, (166, 183, 207))
    return frame


def _demo_canvas(
    annotated: np.ndarray,
    video_time: float,
    objects: list[Any],
    theft_states: list[dict[str, Any]],
    alert_time: float | None,
) -> np.ndarray:
    canvas = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (50, 29, 18), dtype=np.uint8)
    video_width = round(annotated.shape[1] * OUTPUT_SIZE[1] / annotated.shape[0])
    video = cv2.resize(annotated, (video_width, OUTPUT_SIZE[1]), interpolation=cv2.INTER_AREA)
    canvas[:, :video_width] = video
    cv2.line(canvas, (video_width, 0), (video_width, OUTPUT_SIZE[1]), (214, 190, 12), 3)

    left = video_width + 38
    _put(canvas, "SCT CAMERA", (left, 54), 0.92, (255, 255, 255), 2)
    _put(canvas, "OFFLINE REPLAY - SAME AI PIPELINE", (left, 88), 0.49, (214, 190, 12), 1)
    cv2.line(canvas, (left, 108), (OUTPUT_SIZE[0] - 36, 108), (108, 77, 57), 1)

    stages = (
        "1  YOLOv11 object detection",
        "2  ByteTrack multi-object tracking",
        "3  YOLOv11-Pose estimation",
        "4  Behavior scoring and alerting",
    )
    for index, label in enumerate(stages):
        y = 153 + (index * 48)
        cv2.circle(canvas, (left + 10, y - 6), 7, (82, 205, 135), -1, cv2.LINE_AA)
        _put(canvas, label, (left + 32, y), 0.55, (225, 235, 246), 1)

    people = sum(obj.class_name == "person" for obj in objects)
    vehicles = sum(obj.class_name in {"bicycle", "car", "motorcycle", "bus", "truck"} for obj in objects)
    _put(canvas, f"Video time  {video_time:05.1f} s", (left, 370), 0.57, (176, 194, 216))
    _put(canvas, f"Tracked     {len(objects)} objects  |  {people} person  |  {vehicles} vehicle", (left, 402), 0.53, (176, 194, 216))

    panel_top = 438
    cv2.rectangle(canvas, (left, panel_top), (OUTPUT_SIZE[0] - 38, 622), (40, 22, 12), -1)
    state = max(
        theft_states,
        key=lambda item: (bool(item.get("alerted")), int(item.get("score", 0)), float(item.get("near_seconds", 0.0))),
        default=None,
    )
    if state is None:
        _put(canvas, "BEHAVIOR STATUS", (left + 22, panel_top + 38), 0.55, (214, 190, 12), 1)
        _put(canvas, "Waiting for a person-vehicle interaction...", (left + 22, panel_top + 83), 0.57)
    else:
        score = int(state.get("score", 0))
        threshold = int(state.get("score_threshold", 1))
        near = float(state.get("near_seconds", 0.0))
        near_threshold = float(state.get("near_threshold", 0.0))
        signals = ", ".join(str(item).replace("_", " ") for item in state.get("behaviors", [])) or "collecting evidence"
        color = (55, 72, 245) if alert_time is not None else (214, 190, 12)
        _put(canvas, "THEFT BEHAVIOR SCORE", (left + 22, panel_top + 38), 0.55, color, 1)
        _put(canvas, f"Score {score}/{threshold}    Near vehicle {near:.1f}/{near_threshold:.0f} s", (left + 22, panel_top + 80), 0.61)
        _put(canvas, f"Signals: {signals[:72]}", (left + 22, panel_top + 122), 0.46, (190, 205, 224))
        if alert_time is not None:
            cv2.rectangle(canvas, (left + 18, panel_top + 140), (OUTPUT_SIZE[0] - 56, panel_top + 174), (55, 72, 245), -1)
            _put(canvas, f"ALERT TRIGGERED AT {alert_time:.2f} s", (left + 34, panel_top + 166), 0.61, (255, 255, 255), 2)

    _put(canvas, "Recorded evidence - no external service dependency", (left, 682), 0.49, (142, 162, 190))
    return canvas


def _summary_card(alert_time: float, input_frames: int, source_fps: float) -> np.ndarray:
    frame = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (66, 38, 24), dtype=np.uint8)
    _put(frame, "DEMO RESULT", (92, 112), 0.82, (214, 190, 12), 2)
    cv2.rectangle(frame, (92, 156), (1188, 278), (57, 34, 22), -1)
    cv2.rectangle(frame, (92, 156), (1188, 278), (55, 72, 245), 3)
    _put(frame, "SUSPICIOUS THEFT BEHAVIOR DETECTED", (128, 216), 0.90, (255, 255, 255), 2)
    _put(frame, f"Alert timestamp: {alert_time:.2f} seconds", (128, 254), 0.61, (205, 217, 234))
    facts = (
        "Person and vehicle were detected and assigned persistent track IDs",
        "Pose and motion cues were combined with proximity duration",
        "Behavior threshold generated a local alert without network access",
    )
    for index, fact in enumerate(facts):
        y = 362 + index * 62
        cv2.circle(frame, (111, y - 6), 7, (82, 205, 135), -1, cv2.LINE_AA)
        _put(frame, fact, (136, y), 0.60)
    _put(frame, f"Input evidence: {input_frames} frames at {source_fps:.2f} FPS", (92, 616), 0.53, (166, 183, 207))
    _put(frame, "This is an offline replay of the experimental pipeline, not a live claim.", (92, 660), 0.53, (166, 183, 207))
    return frame


def _open_writer(path: Path) -> tuple[cv2.VideoWriter, str]:
    attempts = (
        ("H.264 (Media Foundation)", cv2.CAP_MSMF, "H264"),
        ("H.264 (FFmpeg)", cv2.CAP_FFMPEG, "avc1"),
        ("MPEG-4 Part 2 (FFmpeg)", cv2.CAP_FFMPEG, "mp4v"),
    )
    for label, backend, fourcc in attempts:
        if path.exists():
            path.unlink()
        writer = cv2.VideoWriter(
            str(path),
            backend,
            cv2.VideoWriter_fourcc(*fourcc),
            OUTPUT_FPS,
            OUTPUT_SIZE,
        )
        if writer.isOpened():
            return writer, label
        writer.release()
    raise RuntimeError("No MP4 encoder is available")


def _verify_and_preview(path: Path, alert_frame: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode generated video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    sample_frames: list[np.ndarray] = []
    sample_indexes = [round(OUTPUT_FPS * 1.5), round(OUTPUT_FPS * 10), alert_frame, max(0, frames - round(OUTPUT_FPS * 2))]
    for index in sample_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot decode frame {index} from generated video")
        sample_frames.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
    capture.release()
    contact_sheet = np.vstack((np.hstack(sample_frames[:2]), np.hstack(sample_frames[2:])))
    cv2.imwrite(str(OUTPUT_DIR / "demo_contact_sheet.jpg"), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(OUTPUT_DIR / "demo_alert_frame.jpg"), sample_frames[2], [cv2.IMWRITE_JPEG_QUALITY, 95])
    if (width, height) != OUTPUT_SIZE or fps <= 0 or frames <= 0:
        raise RuntimeError("Generated video metadata is invalid")
    return {
        "video": str(path.resolve()),
        "codec_fourcc": fourcc,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frames": frames,
        "duration_seconds": round(frames / fps, 3),
        "decoded_sample_frames": sample_indexes,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = copy.deepcopy(_yaml(SETTINGS_PATH))
    settings.setdefault("behavior_learning", {}).update({
        "enabled": True,
        "log_candidates": False,
        "gate_alerts": False,
    })
    config = _yaml(CONFIG_PATH)
    source_path = ROOT / str(config["source"])
    config["source"] = str(source_path)
    config["show_theft_overlay"] = True

    detector = YOLOv11Detector(settings)
    pose = PoseEstimator(settings, detector.inference_lock, detector.device, detector.use_half)
    tracker = ByteTrackTracker(detector, settings)
    engine = BehaviorEngine(settings)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {source_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    input_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer, encoder = _open_writer(OUTPUT_PATH)

    for _ in range(round(TITLE_SECONDS * OUTPUT_FPS)):
        writer.write(_title_card())

    clock = SimpleNamespace(value=0.0)
    fake_time = SimpleNamespace(monotonic=lambda: clock.value)
    alert_time: float | None = None
    frame_index = 0
    with ExitStack() as stack:
        for module in TIME_MODULES:
            stack.enter_context(patch(module, fake_time))
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            clock.value = frame_index / source_fps
            frame = _resize_height(_rotate(frame, str(config.get("frame_rotation", "none"))), OUTPUT_SIZE[1])
            objects = tracker.track(frame)
            if bool(settings.get("pose", {}).get("enabled", True)):
                objects = pose.attach(frame, objects)
            objects = engine.label_objects(objects, config, frame)
            alerts = engine.analyze(objects, config, frame.shape)
            theft_states = engine.get_theft_states(str(config["camera_id"]))
            if alert_time is None and any(alert.get("type") == "suspicious_theft_behavior" for alert in alerts):
                alert_time = clock.value
            annotated = draw_annotations(
                frame,
                objects,
                config,
                engine.get_counters(str(config["camera_id"])),
                engine.get_person_timer_states(str(config["camera_id"])),
                alerts,
                theft_states,
            )
            writer.write(_demo_canvas(annotated, clock.value, objects, theft_states, alert_time))
            frame_index += 1
    capture.release()

    if alert_time is None:
        writer.release()
        raise RuntimeError("The expected theft alert was not reproduced; demo was not accepted")

    summary = _summary_card(alert_time, input_frames, source_fps)
    for _ in range(round(SUMMARY_SECONDS * OUTPUT_FPS)):
        writer.write(summary)
    writer.release()

    alert_frame = round((TITLE_SECONDS + alert_time + 1.0) * OUTPUT_FPS)
    metadata = _verify_and_preview(OUTPUT_PATH, alert_frame)
    metadata.update({
        "encoder": encoder,
        "scenario": "suspicious theft behavior near a watched motorcycle",
        "input_video": str(source_path.resolve()),
        "camera_config": str(CONFIG_PATH.resolve()),
        "input_frames_processed": frame_index,
        "alert_video_time_seconds": round(alert_time, 3),
        "offline_replay": True,
        "network_required": False,
    })
    (OUTPUT_DIR / "demo_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
