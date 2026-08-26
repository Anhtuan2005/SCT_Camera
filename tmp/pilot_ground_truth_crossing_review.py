from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"E:\SCT_Camera")
OUTPUT = ROOT / "tmp" / "pilot_ground_truth_crossings"
OUTPUT.mkdir(parents=True, exist_ok=True)
CLIPS = {
    "intrusion_dev_in": (
        ROOT / "experiments/priority_19_08/manual/videos_pilot/development/development_r1_daylight_frontal_oblique/P01_intrusion_positive_r1_daylight_frontal_oblique.mp4",
        6.0,
        12.0,
    ),
    "intrusion_test_in": (
        ROOT / "experiments/priority_19_08/manual/videos_pilot/test/test_r2_low_light_side_oblique/P01_intrusion_positive_r2_low_light_side_oblique.mp4",
        12.0,
        20.0,
    ),
    "line_dev_out": (
        ROOT / "experiments/priority_19_08/manual/videos_pilot/development/development_r1_daylight_frontal_oblique/P01_line_crossing_positive_r1_daylight_frontal_oblique.mp4",
        6.0,
        14.0,
    ),
    "line_test_out": (
        ROOT / "experiments/priority_19_08/manual/videos_pilot/test/test_r2_low_light_side_oblique/P01_line_crossing_positive_r2_low_light_side_oblique.mp4",
        6.0,
        14.0,
    ),
}


for name, (path, start, end) in CLIPS.items():
    capture = cv2.VideoCapture(str(path))
    tiles = []
    for time_s in np.linspace(start, end, 24):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(time_s) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        height, width = frame.shape[:2]
        point1 = (round(0.30 * width), round(0.16 * height))
        point2 = (round(0.75 * width), round(0.16 * height))
        cv2.line(frame, point1, point2, (0, 255, 255), 8, cv2.LINE_AA)
        tile_width = 210
        tile_height = round(frame.shape[0] * tile_width / frame.shape[1])
        tile = cv2.resize(frame, (tile_width, tile_height))
        cv2.putText(tile, f"{time_s:05.2f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    capture.release()
    rows = [cv2.hconcat(tiles[index : index + 6]) for index in range(0, 24, 6)]
    cv2.imwrite(str(OUTPUT / f"{name}.jpg"), cv2.vconcat(rows))
