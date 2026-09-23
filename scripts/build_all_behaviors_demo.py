"""Build one short defense video covering line crossing, intrusion, and theft."""

from __future__ import annotations

import copy
import json
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.behavior_engine import BehaviorEngine
from core.detector import YOLOv11Detector
from core.pose import PoseEstimator
from core.tracker import ByteTrackTracker
from scripts.build_defense_demo import (
    OUTPUT_FPS,
    OUTPUT_SIZE,
    SETTINGS_PATH,
    TIME_MODULES,
    _open_writer,
    _put,
    _resize_height,
    _rotate,
    _yaml,
)
from utils.drawing import draw_annotations


OUTPUT_DIR = ROOT / "outputs/defense_demo"
OUTPUT_PATH = OUTPUT_DIR / "SCT_Camera_All_Behaviors_Demo.mp4"
TITLE_SECONDS = 3.0
SCENARIO_TITLE_SECONDS = 1.2
SUMMARY_SECONDS = 5.0


@dataclass(frozen=True)
class Scenario:
    kind: str
    title: str
    subtitle: str
    config_name: str
    display_start: float
    display_end: float


SCENARIOS = (
    Scenario(
        "line_crossing",
        "LINE CROSSING",
        "Directional counting through a configured virtual line",
        "P01_line_crossing_positive_r1_daylight_frontal_oblique.yaml",
        0.0,
        11.0,
    ),
    Scenario(
        "intrusion",
        "INTRUSION DETECTION",
        "Entry into a protected region of interest",
        "P01_intrusion_positive_r1_daylight_frontal_oblique.yaml",
        0.0,
        7.0,
    ),
    Scenario(
        "theft",
        "SUSPICIOUS THEFT BEHAVIOR",
        "Person-vehicle proximity, motion, and pose/contact cues",
        "P01_theft_positive_r1_daylight_side_oblique.yaml",
        6.0,
        22.0,
    ),
)


def _title_card() -> np.ndarray:
    frame = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (66, 38, 24), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (18, OUTPUT_SIZE[1]), (214, 190, 12), -1)
    _put(frame, "SCT CAMERA", (86, 112), 1.35, (255, 255, 255), 3)
    _put(frame, "3-BEHAVIOR BACKUP DEMO", (86, 178), 0.86, (214, 190, 12), 2)
    cards = (
        ("01", "LINE CROSSING", "Count IN / OUT"),
        ("02", "INTRUSION", "Protected ROI"),
        ("03", "THEFT", "Multi-cue scoring"),
    )
    for index, (number, title, subtitle) in enumerate(cards):
        x1 = 86 + index * 386
        x2 = x1 + 346
        cv2.rectangle(frame, (x1, 270), (x2, 470), (50, 28, 17), -1)
        cv2.rectangle(frame, (x1, 270), (x2, 470), (69, 91, 124), 2)
        _put(frame, number, (x1 + 24, 322), 0.60, (214, 190, 12), 2)
        _put(frame, title, (x1 + 24, 382), 0.66, (255, 255, 255), 2)
        _put(frame, subtitle, (x1 + 24, 430), 0.54, (176, 194, 216))
    _put(frame, "Offline replay using the same experimental AI pipeline", (86, 574), 0.68)
    _put(frame, "No camera, Internet, or Telegram connection is required.", (86, 628), 0.56, (166, 183, 207))
    return frame


def _scenario_card(scenario: Scenario) -> np.ndarray:
    frame = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (66, 38, 24), dtype=np.uint8)
    _put(frame, f"SCENARIO {SCENARIOS.index(scenario) + 1}/3", (90, 184), 0.72, (214, 190, 12), 2)
    _put(frame, scenario.title, (90, 300), 1.20, (255, 255, 255), 3)
    _put(frame, scenario.subtitle, (90, 374), 0.69, (190, 205, 224))
    cv2.line(frame, (90, 430), (1188, 430), (108, 77, 57), 2)
    _put(frame, "YOLOv11  >  ByteTrack  >  Pose  >  Behavior rule", (90, 506), 0.66, (214, 190, 12))
    return frame


def _counter_snapshot(engine: BehaviorEngine, camera_id: str) -> dict[tuple[str, str], int]:
    return {
        (line_id, direction): int(value)
        for line_id, counts in engine.get_counters(camera_id).items()
        for direction, value in counts.items()
    }


def _first_counter_delta(
    before: dict[tuple[str, str], int],
    after: dict[tuple[str, str], int],
) -> tuple[str, str] | None:
    return next((key for key, value in sorted(after.items()) if value > before.get(key, 0)), None)


def _canvas(
    scenario: Scenario,
    annotated: np.ndarray,
    video_time: float,
    objects: list[Any],
    counters: dict[str, dict[str, int]],
    theft_states: list[dict[str, Any]],
    event_time: float | None,
    event_detail: str,
) -> np.ndarray:
    canvas = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (50, 29, 18), dtype=np.uint8)
    video_width = round(annotated.shape[1] * OUTPUT_SIZE[1] / annotated.shape[0])
    canvas[:, :video_width] = cv2.resize(annotated, (video_width, OUTPUT_SIZE[1]), interpolation=cv2.INTER_AREA)
    cv2.line(canvas, (video_width, 0), (video_width, OUTPUT_SIZE[1]), (214, 190, 12), 3)
    left = video_width + 38

    _put(canvas, scenario.title, (left, 55), 0.82, (255, 255, 255), 2)
    _put(canvas, "OFFLINE REPLAY - SAME AI PIPELINE", (left, 88), 0.48, (214, 190, 12))
    cv2.line(canvas, (left, 108), (OUTPUT_SIZE[0] - 36, 108), (108, 77, 57), 1)
    for index, label in enumerate(("YOLOv11 detection", "ByteTrack IDs", "Pose estimation", "Behavior analysis")):
        y = 150 + index * 42
        cv2.circle(canvas, (left + 8, y - 6), 7, (82, 205, 135), -1, cv2.LINE_AA)
        _put(canvas, label, (left + 30, y), 0.53)

    people = sum(obj.class_name == "person" for obj in objects)
    _put(canvas, f"Video time  {video_time:05.1f} s", (left, 350), 0.57, (176, 194, 216))
    _put(canvas, f"Tracked     {len(objects)} objects  |  {people} person", (left, 382), 0.53, (176, 194, 216))
    cv2.rectangle(canvas, (left, 420), (OUTPUT_SIZE[0] - 38, 624), (40, 22, 12), -1)

    if scenario.kind == "line_crossing":
        totals = {"in": 0, "out": 0}
        for counts in counters.values():
            totals["in"] += int(counts.get("in", 0))
            totals["out"] += int(counts.get("out", 0))
        _put(canvas, "DIRECTIONAL COUNTER", (left + 22, 462), 0.58, (214, 190, 12), 1)
        _put(canvas, f"IN  {totals['in']}     OUT  {totals['out']}", (left + 22, 516), 0.78, (255, 255, 255), 2)
    elif scenario.kind == "intrusion":
        _put(canvas, "PROTECTED REGION", (left + 22, 462), 0.58, (214, 190, 12), 1)
        _put(canvas, "Person entry is evaluated against the configured ROI", (left + 22, 510), 0.52)
    else:
        state = max(theft_states, key=lambda item: (bool(item.get("alerted")), int(item.get("score", 0))), default=None)
        _put(canvas, "THEFT BEHAVIOR SCORE", (left + 22, 462), 0.58, (214, 190, 12), 1)
        if state:
            _put(
                canvas,
                f"Score {int(state.get('score', 0))}/{int(state.get('score_threshold', 1))}    Near {float(state.get('near_seconds', 0)):.1f}/{float(state.get('near_threshold', 0)):.0f} s",
                (left + 22, 510),
                0.62,
            )
            signals = ", ".join(str(item).replace("_", " ") for item in state.get("behaviors", [])) or "collecting evidence"
            _put(canvas, f"Signals: {signals[:70]}", (left + 22, 550), 0.45, (190, 205, 224))

    if event_time is not None:
        cv2.rectangle(canvas, (left + 18, 574), (OUTPUT_SIZE[0] - 56, 614), (55, 72, 245), -1)
        _put(canvas, f"{event_detail}  @ {event_time:.2f} s", (left + 32, 603), 0.58, (255, 255, 255), 2)
    else:
        _put(canvas, "Monitoring...", (left + 22, 596), 0.54, (166, 183, 207))
    _put(canvas, "Recorded evidence - no external service dependency", (left, 681), 0.48, (142, 162, 190))
    return canvas


def _summary_card(events: dict[str, float]) -> np.ndarray:
    frame = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (66, 38, 24), dtype=np.uint8)
    _put(frame, "DEMO RESULT", (88, 102), 0.82, (214, 190, 12), 2)
    _put(frame, "3/3 BEHAVIORS REPRODUCED", (88, 172), 1.04, (255, 255, 255), 3)
    rows = (
        ("LINE CROSSING", events["line_crossing"], "Directional counter updated"),
        ("INTRUSION", events["intrusion"], "Protected ROI alert generated"),
        ("THEFT", events["theft"], "Multi-cue behavior threshold reached"),
    )
    for index, (label, timestamp, detail) in enumerate(rows):
        y = 280 + index * 104
        cv2.circle(frame, (106, y - 7), 9, (82, 205, 135), -1, cv2.LINE_AA)
        _put(frame, label, (136, y), 0.72, (255, 255, 255), 2)
        _put(frame, f"{detail} at video time {timestamp:.2f} s", (136, y + 38), 0.56, (190, 205, 224))
    _put(frame, "Each segment is an offline replay produced by its original experimental configuration.", (88, 636), 0.55, (166, 183, 207))
    return frame


def _verify(path: Path, expected_events: dict[str, float]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path), cv2.CAP_MSMF)
    if not capture.isOpened():
        raise RuntimeError(f"Windows Media Foundation cannot decode {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_indexes = [round(frames * ratio) for ratio in (0.05, 0.25, 0.435, 0.78, 0.95)]
    samples: list[np.ndarray] = []
    for index in sample_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot decode frame {index}")
        samples.append(cv2.resize(frame, (426, 240), interpolation=cv2.INTER_AREA))
    capture.release()
    sheet = np.full((480, 1278, 3), (66, 38, 24), dtype=np.uint8)
    sheet[:240, :1278] = np.hstack(samples[:3])
    sheet[240:, 213:1065] = np.hstack(samples[3:])
    cv2.imwrite(str(OUTPUT_DIR / "all_behaviors_contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if (width, height) != OUTPUT_SIZE or frames <= 0 or set(expected_events) != {item.kind for item in SCENARIOS}:
        raise RuntimeError("Generated combined demo failed validation")
    return {
        "video": str(path.resolve()),
        "codec": "H.264",
        "resolution": f"{width}x{height}",
        "fps": round(fps, 3),
        "frames": frames,
        "duration_seconds": round(frames / fps, 3),
        "event_video_times_seconds": {key: round(value, 3) for key, value in expected_events.items()},
        "windows_media_foundation_decode": True,
        "offline_replay": True,
        "network_required": False,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = copy.deepcopy(_yaml(SETTINGS_PATH))
    settings.setdefault("behavior_learning", {}).update({"enabled": True, "log_candidates": False, "gate_alerts": False})
    detector = YOLOv11Detector(settings)
    pose = PoseEstimator(settings, detector.inference_lock, detector.device, detector.use_half)
    engine = BehaviorEngine(settings)
    writer, encoder = _open_writer(OUTPUT_PATH)
    for _ in range(round(TITLE_SECONDS * OUTPUT_FPS)):
        writer.write(_title_card())

    events: dict[str, float] = {}
    clock = SimpleNamespace(value=0.0)
    fake_time = SimpleNamespace(monotonic=lambda: clock.value)
    with ExitStack() as stack:
        for module in TIME_MODULES:
            stack.enter_context(patch(module, fake_time))
        for scenario in SCENARIOS:
            for _ in range(round(SCENARIO_TITLE_SECONDS * OUTPUT_FPS)):
                writer.write(_scenario_card(scenario))
            config_path = ROOT / "experiments/priority_19_08/manual/configs_pilot" / scenario.config_name
            config = _yaml(config_path)
            source_path = ROOT / str(config["source"])
            config["source"] = str(source_path)
            config["show_theft_overlay"] = scenario.kind == "theft"
            camera_id = str(config["camera_id"])
            tracker = ByteTrackTracker(detector, settings)
            capture = cv2.VideoCapture(str(source_path))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {source_path}")
            source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
            previous_counters = _counter_snapshot(engine, camera_id)
            event_time: float | None = None
            event_detail = ""
            visual_alert: dict[str, Any] | None = None
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                clock.value = frame_index / source_fps
                if clock.value > scenario.display_end:
                    break
                frame = _resize_height(_rotate(frame, str(config.get("frame_rotation", "none"))), OUTPUT_SIZE[1])
                objects = tracker.track(frame)
                if bool(settings.get("pose", {}).get("enabled", True)):
                    objects = pose.attach(frame, objects)
                objects = engine.label_objects(objects, config, frame)
                alerts = engine.analyze(objects, config, frame.shape)
                counters_snapshot = _counter_snapshot(engine, camera_id)
                delta = _first_counter_delta(previous_counters, counters_snapshot)
                previous_counters = counters_snapshot
                if event_time is None:
                    target_alert = next(
                        (
                            alert
                            for alert in alerts
                            if (scenario.kind == "intrusion" and alert.get("type") == "intrusion")
                            or (scenario.kind == "theft" and alert.get("type") == "suspicious_theft_behavior")
                        ),
                        None,
                    )
                    if scenario.kind == "line_crossing" and delta is not None:
                        event_time = clock.value
                        event_detail = f"CROSSING {delta[1].upper()}"
                    elif target_alert is not None:
                        event_time = clock.value
                        event_detail = "INTRUSION ALERT" if scenario.kind == "intrusion" else "THEFT ALERT"
                        visual_alert = dict(target_alert, started_at=clock.value, expires_at=clock.value + 3.0)
                active_alerts = [visual_alert] if visual_alert and clock.value <= float(visual_alert["expires_at"]) else []
                theft_states = engine.get_theft_states(camera_id)
                annotated = draw_annotations(
                    frame,
                    objects,
                    config,
                    engine.get_counters(camera_id),
                    engine.get_person_timer_states(camera_id),
                    active_alerts,
                    theft_states,
                )
                if clock.value >= scenario.display_start:
                    writer.write(
                        _canvas(
                            scenario,
                            annotated,
                            clock.value,
                            objects,
                            engine.get_counters(camera_id),
                            theft_states,
                            event_time,
                            event_detail,
                        )
                    )
                frame_index += 1
            capture.release()
            if event_time is None:
                writer.release()
                raise RuntimeError(f"Expected event was not reproduced for {scenario.kind}")
            events[scenario.kind] = event_time

    summary = _summary_card(events)
    for _ in range(round(SUMMARY_SECONDS * OUTPUT_FPS)):
        writer.write(summary)
    writer.release()
    metadata = _verify(OUTPUT_PATH, events)
    metadata["encoder"] = encoder
    (OUTPUT_DIR / "all_behaviors_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
