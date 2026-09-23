"""Combine a local dashboard walkthrough with the three-behavior demo."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from build_defense_demo import OUTPUT_FPS, OUTPUT_SIZE, _open_writer, _put


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/defense_demo"
CAPTURE_DIR = OUTPUT_DIR / "dashboard_capture"
PIPELINE_VIDEO = OUTPUT_DIR / "SCT_Camera_All_Behaviors_Demo.mp4"
OUTPUT_VIDEO = OUTPUT_DIR / "SCT_Camera_Complete_System_Demo.mp4"
CONTACT_SHEET = OUTPUT_DIR / "complete_demo_contact_sheet.jpg"
METADATA_PATH = OUTPUT_DIR / "complete_demo_metadata.json"
FADE_FRAMES = 12


def _load_image(name: str) -> np.ndarray:
    path = CAPTURE_DIR / name
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot load dashboard capture: {path}")
    return cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)


def _ease(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _zoom_frame(
    image: np.ndarray,
    progress: float,
    zoom_start: float,
    zoom_end: float,
    center_start: tuple[float, float],
    center_end: tuple[float, float],
) -> np.ndarray:
    progress = _ease(progress)
    zoom = zoom_start + ((zoom_end - zoom_start) * progress)
    center_x = center_start[0] + ((center_end[0] - center_start[0]) * progress)
    center_y = center_start[1] + ((center_end[1] - center_start[1]) * progress)
    width, height = OUTPUT_SIZE
    crop_width = max(1, round(width / zoom))
    crop_height = max(1, round(height / zoom))
    x0 = round((center_x * width) - (crop_width / 2))
    y0 = round((center_y * height) - (crop_height / 2))
    x0 = min(max(0, x0), width - crop_width)
    y0 = min(max(0, y0), height - crop_height)
    crop = image[y0:y0 + crop_height, x0:x0 + crop_width]
    return cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_CUBIC)


def _caption(frame: np.ndarray, title: str, detail: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (OUTPUT_SIZE[0], 86), (25, 16, 12), -1)
    cv2.rectangle(overlay, (0, 650), (OUTPUT_SIZE[0], OUTPUT_SIZE[1]), (25, 16, 12), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (0, 0), (12, 86), (214, 190, 12), -1)
    _put(frame, "SCT CAMERA  |  LOCAL DASHBOARD", (34, 36), 0.68, (255, 255, 255), 2)
    _put(frame, title, (34, 70), 0.52, (214, 190, 12), 1)
    _put(frame, detail, (34, 692), 0.56, (238, 244, 250), 1)


def _cursor(frame: np.ndarray, point: tuple[int, int], pulse: float) -> None:
    radius = 13 + round(5 * abs(np.sin(pulse * np.pi * 2)))
    cv2.circle(frame, point, radius, (214, 190, 12), 2, cv2.LINE_AA)
    cv2.circle(frame, point, 5, (255, 255, 255), -1, cv2.LINE_AA)


def _title_card() -> np.ndarray:
    frame = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (66, 38, 24), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (18, OUTPUT_SIZE[1]), (214, 190, 12), -1)
    cv2.circle(frame, (112, 126), 28, (214, 190, 12), -1, cv2.LINE_AA)
    cv2.circle(frame, (112, 126), 12, (66, 38, 24), 4, cv2.LINE_AA)
    _put(frame, "SCT CAMERA", (164, 142), 1.52, (255, 255, 255), 3)
    _put(frame, "COMPLETE SYSTEM DEMO", (94, 250), 0.92, (214, 190, 12), 2)
    _put(frame, "Dashboard walkthrough + three behavior scenarios", (94, 320), 0.72)
    _put(frame, "Line crossing  |  Intrusion  |  Suspicious theft behavior", (94, 372), 0.66, (190, 205, 224))
    cv2.rectangle(frame, (92, 460), (1186, 554), (50, 28, 17), -1)
    cv2.rectangle(frame, (92, 460), (1186, 554), (214, 190, 12), 2)
    _put(frame, "SAFE LOCAL CAPTURE", (124, 500), 0.69, (214, 190, 12), 2)
    _put(frame, "External Telegram, Discord and siren channels were disabled", (124, 536), 0.60)
    _put(frame, "The behavior clips are offline replays of the same experimental AI pipeline.", (94, 650), 0.54, (166, 183, 207))
    return frame


def _transition_card() -> np.ndarray:
    frame = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (66, 38, 24), dtype=np.uint8)
    _put(frame, "DASHBOARD WALKTHROUGH COMPLETE", (110, 250), 0.90, (255, 255, 255), 2)
    cv2.line(frame, (110, 292), (1170, 292), (214, 190, 12), 3)
    _put(frame, "NEXT: THREE-BEHAVIOR PIPELINE EVIDENCE", (110, 380), 0.82, (214, 190, 12), 2)
    _put(frame, "Detection  >  Tracking  >  Pose  >  Behavior analysis  >  Alert", (110, 440), 0.61, (190, 205, 224))
    return frame


def _write_hold(writer: cv2.VideoWriter, frame: np.ndarray, seconds: float) -> np.ndarray:
    for _ in range(round(seconds * OUTPUT_FPS)):
        writer.write(frame)
    return frame


def _write_scene(
    writer: cv2.VideoWriter,
    image: np.ndarray,
    seconds: float,
    title: str,
    detail: str,
    center_start: tuple[float, float],
    center_end: tuple[float, float],
    cursor_start: tuple[int, int],
    cursor_end: tuple[int, int],
    previous: np.ndarray,
) -> np.ndarray:
    frame_count = round(seconds * OUTPUT_FPS)
    last = image
    for index in range(frame_count):
        progress = index / max(1, frame_count - 1)
        frame = _zoom_frame(image, progress, 1.0, 1.035, center_start, center_end)
        _caption(frame, title, detail)
        cursor_progress = _ease(progress)
        cursor_point = (
            round(cursor_start[0] + ((cursor_end[0] - cursor_start[0]) * cursor_progress)),
            round(cursor_start[1] + ((cursor_end[1] - cursor_start[1]) * cursor_progress)),
        )
        _cursor(frame, cursor_point, progress)
        if index < FADE_FRAMES:
            alpha = (index + 1) / FADE_FRAMES
            frame = cv2.addWeighted(previous, 1.0 - alpha, frame, alpha, 0)
        writer.write(frame)
        last = frame
    return last


def _append_video(writer: cv2.VideoWriter, path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open behavior demo: {path}")
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if (frame.shape[1], frame.shape[0]) != OUTPUT_SIZE:
            frame = cv2.resize(frame, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        count += 1
    capture.release()
    if count == 0:
        raise RuntimeError("Behavior demo contains no decodable frames")
    return count


def _verify(path: Path, dashboard_frames: int) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode generated video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    if (width, height) != OUTPUT_SIZE or not (24.9 <= fps <= 25.1) or frames <= dashboard_frames:
        raise RuntimeError("Generated video metadata is invalid")
    if fourcc.lower() not in {"avc1", "h264"}:
        raise RuntimeError(f"Expected H.264 output, got FOURCC {fourcc!r}")

    sample_seconds = (1.0, 3.6, 7.5, 12.5, 17.5, 21.0, 35.0, (frames / fps) - 2.0)
    samples: list[np.ndarray] = []
    sample_indexes: list[int] = []
    for seconds in sample_seconds:
        index = min(frames - 1, max(0, round(seconds * fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot decode validation frame {index}")
        samples.append(cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA))
        sample_indexes.append(index)
    capture.release()
    sheet = np.vstack(tuple(np.hstack(tuple(samples[row:row + 4])) for row in (0, 4)))
    if not cv2.imwrite(str(CONTACT_SHEET), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Cannot write contact sheet: {CONTACT_SHEET}")
    return {
        "video": str(path.resolve()),
        "codec_fourcc": fourcc,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frames": frames,
        "duration_seconds": round(frames / fps, 3),
        "dashboard_duration_seconds": round(dashboard_frames / fps, 3),
        "decoded_sample_frames": sample_indexes,
        "contact_sheet": str(CONTACT_SHEET.resolve()),
    }


def main() -> int:
    required = (
        CAPTURE_DIR / "01_login.png",
        CAPTURE_DIR / "02_live_wall.png",
        CAPTURE_DIR / "03_camera_detail.png",
        CAPTURE_DIR / "04_intrusion_detail.png",
        PIPELINE_VIDEO,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing source files: {missing}")

    writer, encoder = _open_writer(OUTPUT_VIDEO)
    try:
        previous = _write_hold(writer, _title_card(), 2.0)
        previous = _write_scene(
            writer, _load_image("01_login.png"), 3.0,
            "SESSION-PROTECTED ACCESS", "Open the local dashboard and enter the monitored workspace.",
            (0.50, 0.50), (0.50, 0.54), (1120, 110), (642, 516), previous,
        )
        previous = _write_scene(
            writer, _load_image("02_live_wall.png"), 5.0,
            "MULTI-CAMERA LIVE WALL", "Camera status, detected objects, AI throughput and recent alerts in one view.",
            (0.52, 0.47), (0.50, 0.52), (1158, 101), (364, 424), previous,
        )
        previous = _write_scene(
            writer, _load_image("04_intrusion_detail.png"), 5.0,
            "INTRUSION AND LINE-CROSSING VIEW", "Tracking IDs, virtual line counters, ROI evidence and alert history.",
            (0.51, 0.48), (0.53, 0.52), (1128, 174), (1000, 480), previous,
        )
        previous = _write_scene(
            writer, _load_image("03_camera_detail.png"), 5.0,
            "THEFT MONITORING VIEW", "Watched-asset ROI, behavior overlay and persistent recent-alert records.",
            (0.50, 0.49), (0.52, 0.54), (1125, 174), (1030, 492), previous,
        )
        transition = _transition_card()
        for index in range(round(2.0 * OUTPUT_FPS)):
            frame = transition
            if index < FADE_FRAMES:
                alpha = (index + 1) / FADE_FRAMES
                frame = cv2.addWeighted(previous, 1.0 - alpha, transition, alpha, 0)
            writer.write(frame)
        dashboard_frames = round(22.0 * OUTPUT_FPS)
        pipeline_frames = _append_video(writer, PIPELINE_VIDEO)
    finally:
        writer.release()

    metadata = _verify(OUTPUT_VIDEO, dashboard_frames)
    metadata.update({
        "encoder": encoder,
        "dashboard_capture": "real local dashboard state; external alert channels disabled",
        "pipeline_video": str(PIPELINE_VIDEO.resolve()),
        "pipeline_frames_appended": pipeline_frames,
        "behavior_scenarios": ["line_crossing", "intrusion", "suspicious_theft_behavior"],
        "network_required_for_playback": False,
    })
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
