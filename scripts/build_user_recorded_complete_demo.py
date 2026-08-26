"""Trim the user's dashboard recording and prepend it to the accepted demo tail."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from build_defense_demo import OUTPUT_FPS, OUTPUT_SIZE, _open_writer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/defense_demo"
USER_RECORDING = Path(
    r"C:\Users\Admin\AppData\Local\Packages\Microsoft.ScreenSketch_8wekyb3d8bbwe"
    r"\TempState\Recordings\20260823-1017-57.0088274.mp4"
)
ACCEPTED_DEMO = OUTPUT_DIR / "SCT_Camera_Complete_System_Demo.mp4"
OUTPUT_VIDEO = OUTPUT_DIR / "SCT_Camera_Final_Defense_Demo.mp4"
CONTACT_SHEET = OUTPUT_DIR / "final_demo_contact_sheet.jpg"
METADATA_PATH = OUTPUT_DIR / "final_demo_metadata.json"

# Remove the login/loading portion and stop before the recorder UI can appear.
DASHBOARD_START_SECONDS = 8.0
DASHBOARD_END_SECONDS = 35.5
# The accepted composite begins its transition into the behavior evidence here.
ACCEPTED_DEMO_START_SECONDS = 20.0


def _letterbox(frame: np.ndarray) -> np.ndarray:
    target_width, target_height = OUTPUT_SIZE
    height, width = frame.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return canvas


def _append_range(
    writer: cv2.VideoWriter,
    path: Path,
    start_seconds: float,
    end_seconds: float | None = None,
) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or source_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid input metadata: {path}")

    source_duration = source_frames / source_fps
    stop_seconds = source_duration if end_seconds is None else min(end_seconds, source_duration)
    if not (0 <= start_seconds < stop_seconds):
        capture.release()
        raise RuntimeError(f"Invalid time range {start_seconds}-{stop_seconds}: {path}")

    start_frame = round(start_seconds * source_fps)
    stop_frame = min(source_frames, round(stop_seconds * source_fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_count = 0
    source_index = start_frame
    while source_index < stop_frame:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        source_elapsed = (source_index - start_frame) / source_fps
        while source_elapsed + 1e-9 >= output_count / OUTPUT_FPS:
            writer.write(_letterbox(frame))
            output_count += 1
        source_index += 1
    capture.release()
    if output_count == 0:
        raise RuntimeError(f"No frames appended from {path}")
    return output_count, source_fps


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

    duration = frames / fps
    dashboard_duration = dashboard_frames / fps
    sample_seconds = (
        0.5,
        5.0,
        12.0,
        20.0,
        max(0.0, dashboard_duration - 1.0),
        dashboard_duration + 1.0,
        dashboard_duration + 15.0,
        duration - 2.0,
    )
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
    sheet = np.vstack((np.hstack(samples[:4]), np.hstack(samples[4:])))
    if not cv2.imwrite(str(CONTACT_SHEET), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Cannot write contact sheet: {CONTACT_SHEET}")
    return {
        "video": str(path.resolve()),
        "codec_fourcc": fourcc,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frames": frames,
        "duration_seconds": round(duration, 3),
        "dashboard_duration_seconds": round(dashboard_duration, 3),
        "decoded_sample_frames": sample_indexes,
        "contact_sheet": str(CONTACT_SHEET.resolve()),
    }


def main() -> int:
    missing = [str(path) for path in (USER_RECORDING, ACCEPTED_DEMO) if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing input files: {missing}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer, encoder = _open_writer(OUTPUT_VIDEO)
    try:
        dashboard_frames, dashboard_source_fps = _append_range(
            writer,
            USER_RECORDING,
            DASHBOARD_START_SECONDS,
            DASHBOARD_END_SECONDS,
        )
        accepted_frames, accepted_source_fps = _append_range(
            writer,
            ACCEPTED_DEMO,
            ACCEPTED_DEMO_START_SECONDS,
        )
    finally:
        writer.release()

    metadata = _verify(OUTPUT_VIDEO, dashboard_frames)
    metadata.update({
        "encoder": encoder,
        "dashboard_recording": str(USER_RECORDING),
        "dashboard_trim_seconds": [DASHBOARD_START_SECONDS, DASHBOARD_END_SECONDS],
        "dashboard_source_fps": dashboard_source_fps,
        "accepted_demo": str(ACCEPTED_DEMO.resolve()),
        "accepted_demo_start_seconds": ACCEPTED_DEMO_START_SECONDS,
        "accepted_demo_source_fps": accepted_source_fps,
        "accepted_demo_frames_appended": accepted_frames,
        "behavior_scenarios": ["line_crossing", "intrusion", "suspicious_theft_behavior"],
        "network_required_for_playback": False,
    })
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
