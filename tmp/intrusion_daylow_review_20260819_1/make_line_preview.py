from pathlib import Path

import cv2


ROOT = Path(r"E:\SCT_Camera\tmp\intrusion_daylow_review_20260819_1")
SAMPLES = {
    "58606849029842367887.mp4": [5.5, 7.0, 8.5],
    "91014796315081936208.mp4": [13.5, 15.3, 17.0],
}


tiles = []
for filename, times in SAMPLES.items():
    capture = cv2.VideoCapture(str(ROOT / filename))
    for time_s in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        height, width = frame.shape[:2]
        point1 = (round(0.30 * width), round(0.16 * height))
        point2 = (round(0.75 * width), round(0.16 * height))
        cv2.line(frame, point1, point2, (0, 255, 255), 8, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{filename[:6]}  {time_s:.1f}s",
            (18, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        tile_width = 300
        tile_height = round(frame.shape[0] * tile_width / frame.shape[1])
        tiles.append(cv2.resize(frame, (tile_width, tile_height)))
    capture.release()

preview = cv2.hconcat(tiles[:3])
preview2 = cv2.hconcat(tiles[3:])
cv2.imwrite(str(ROOT / "frozen_doorway_line_preview_out.jpg"), preview)
cv2.imwrite(str(ROOT / "frozen_doorway_line_preview_in.jpg"), preview2)
