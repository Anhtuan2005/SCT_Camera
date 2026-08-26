import json
from pathlib import Path

import cv2
import numpy as np


OUTPUT = Path(r"E:\SCT_Camera\tmp\loitering_line_review_20260819_1")
LOITERING = Path(r"E:\20260819005233907_F6B80AMPBVCDD6C_L_0_L0120819005233_1.mp4")
VIDEOS = [(LOITERING, "loitering_negative", False)] + [
    (path, "line_crossing", True) for path in sorted(OUTPUT.glob("*.mp4"))
]


def main() -> None:
    results = []
    for video_path, role, draw_line in VIDEOS:
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = frame_count / fps if fps > 0 else 0.0

        sample_count = 20 if role == "loitering_negative" else 16
        columns = 5 if role == "loitering_negative" else 4
        tiles = []
        brightness = []
        sample_times = np.linspace(0.0, max(duration_s - 0.05, 0.0), sample_count)
        for time_s in sample_times:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(time_s) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if draw_line:
                frame_height, frame_width = frame.shape[:2]
                point1 = (round(0.30 * frame_width), round(0.16 * frame_height))
                point2 = (round(0.75 * frame_width), round(0.16 * frame_height))
                cv2.line(frame, point1, point2, (0, 255, 255), 8, cv2.LINE_AA)
            tile_width = 225
            tile_height = round(frame.shape[0] * tile_width / frame.shape[1])
            tile = cv2.resize(frame, (tile_width, tile_height))
            cv2.putText(
                tile,
                f"{time_s:05.1f}s",
                (7, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        capture.release()

        rows = []
        for start in range(0, len(tiles), columns):
            row_tiles = tiles[start : start + columns]
            while len(row_tiles) < columns:
                row_tiles.append(np.zeros_like(tiles[0]))
            rows.append(cv2.hconcat(row_tiles))
        sheet_path = OUTPUT / f"contact_{video_path.stem}.jpg"
        cv2.imwrite(str(sheet_path), cv2.vconcat(rows))

        results.append(
            {
                "file": video_path.name,
                "role": role,
                "size_bytes": video_path.stat().st_size,
                "fps": fps,
                "frame_count": frame_count,
                "duration_s": duration_s,
                "stored_width": width,
                "stored_height": height,
                "upright_width_after_cw90": height,
                "upright_height_after_cw90": width,
                "sample_brightness_mean": float(np.mean(brightness)),
                "sample_brightness_min": float(np.min(brightness)),
                "sample_brightness_max": float(np.max(brightness)),
                "contact_sheet": str(sheet_path),
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
