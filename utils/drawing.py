"""Drawing helpers for tracked objects, zones, lines, and counters."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from analytics.line_counter import CountingLine
from analytics.zone import Zone
from core.tracker import TrackedObject

COLOR_BY_ZONE = {
    "all": (75, 210, 145),
    "intrusion": (64, 96, 255),
    "loitering": (0, 190, 255),
    "counting": (255, 190, 70),
    "stranger_watch": (245, 120, 120),
    "asset_watch": (112, 215, 235),
}


def draw_annotations(
    frame: np.ndarray,
    tracked_objects: list[TrackedObject],
    camera_config: dict[str, Any],
    counters: dict[str, dict[str, int]] | None = None,
    person_timer_states: dict[int, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Draw zones, lines, track boxes, histories, and counters on a frame."""
    counters = counters or {}
    person_timer_states = person_timer_states or {}
    zones = [Zone.from_config(item) for item in camera_config.get("zones", []) if len(item.get("polygon", [])) >= 3]
    lines = [CountingLine.from_config(item) for item in camera_config.get("lines", [])]

    annotated = frame.copy()
    overlay = annotated.copy()
    for zone in zones:
        _draw_zone(overlay, annotated, zone)
    annotated = cv2.addWeighted(overlay, 0.24, annotated, 0.76, 0)

    for line in lines:
        _draw_line(annotated, line, counters.get(line.id, {"in": 0, "out": 0}))

    for obj in tracked_objects:
        _draw_object(annotated, obj, person_timer_states.get(obj.track_id))

    _draw_frame_hud(annotated, camera_config, tracked_objects)
    return annotated


def _draw_zone(overlay: np.ndarray, base: np.ndarray, zone: Zone) -> None:
    polygon = zone.pixel_polygon(base.shape)
    color = COLOR_BY_ZONE.get(zone.zone_type, (150, 170, 190))
    cv2.fillPoly(overlay, [polygon], color)
    cv2.polylines(base, [polygon], isClosed=True, color=color, thickness=_line_thickness(base))
    label_point = tuple(polygon[0])
    _draw_label(base, zone.name, label_point, color)


def _draw_line(frame: np.ndarray, line: CountingLine, counter: dict[str, int]) -> None:
    p1, p2 = line.pixel_points(frame.shape)
    color = (112, 215, 235)
    thickness = _line_thickness(frame)
    cv2.line(frame, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    normal = (-dy / length, dx / length)
    if line.direction.lower() in {"reverse", "backward", "out"}:
        normal = (-normal[0], -normal[1])
    arrow_tip = (int(mid[0] + normal[0] * 42), int(mid[1] + normal[1] * 42))
    cv2.arrowedLine(frame, mid, arrow_tip, color, thickness, tipLength=0.35)
    _draw_label(
        frame,
        f"{line.name} IN:{counter.get('in', 0)} OUT:{counter.get('out', 0)}",
        (mid[0] + 8, mid[1] - 8),
        color,
    )


def _draw_object(
    frame: np.ndarray,
    obj: TrackedObject,
    person_timer_state: dict[str, Any] | None = None,
) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in obj.bbox_xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    color = _color_for_id(obj.track_id)
    thickness = _line_thickness(frame)
    shadow = max(thickness + 1, 3)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 14, 18), shadow, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    label = _object_label(obj)
    _draw_label(frame, label, (x1, y1 - max(8, int(8 * _visual_scale(frame)))), color)
    if person_timer_state:
        _draw_timer_badge(frame, person_timer_state, (x1, y1, x2, y2))

    if obj.pose_keypoints:
        _draw_pose(frame, obj.pose_keypoints, color)

    if len(obj.center_history) > 1:
        points = np.array(obj.center_history, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
    cx, cy = obj.center
    radius = max(3, int(round(3 * _visual_scale(frame))))
    cv2.circle(frame, (int(cx), int(cy)), radius + 1, (10, 14, 18), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (int(cx), int(cy)), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_pose(
    frame: np.ndarray,
    keypoints: list[tuple[float, float, float]],
    color: tuple[int, int, int],
) -> None:
    skeleton = (
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    )
    for first, second in skeleton:
        first_point = _visible_keypoint(keypoints, first)
        second_point = _visible_keypoint(keypoints, second)
        if first_point is None or second_point is None:
            continue
        cv2.line(frame, first_point, second_point, color, 1, lineType=cv2.LINE_AA)
    for index in (5, 6, 7, 8, 9, 10, 11, 12):
        point = _visible_keypoint(keypoints, index)
        if point is not None:
            cv2.circle(frame, point, 2, color, -1, lineType=cv2.LINE_AA)


def _visible_keypoint(
    keypoints: list[tuple[float, float, float]],
    index: int,
    min_confidence: float = 0.25,
) -> tuple[int, int] | None:
    if index >= len(keypoints):
        return None
    x, y, confidence = keypoints[index]
    if confidence < min_confidence or x <= 0 or y <= 0:
        return None
    return int(round(x)), int(round(y))


def _draw_frame_hud(
    frame: np.ndarray,
    camera_config: dict[str, Any],
    tracked_objects: list[TrackedObject],
) -> None:
    text = f"{camera_config.get('name', camera_config.get('camera_id', 'Camera'))} | Objects: {len(tracked_objects)}"
    _draw_label(frame, text, (18, 34), (75, 210, 145))


def _draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    visual_scale = _visual_scale(frame)
    scale = 0.56 * visual_scale
    thickness = max(1, int(round(1.15 * visual_scale)))
    pad_x = max(5, int(round(6 * visual_scale)))
    pad_y = max(4, int(round(4 * visual_scale)))
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(pad_x, min(x, frame.shape[1] - width - (pad_x * 2)))
    y = max(height + (pad_y * 2), min(y, frame.shape[0] - pad_y))
    top_left = (x - pad_x, y - height - pad_y - baseline)
    bottom_right = (x + width + pad_x, y + baseline + pad_y)
    cv2.rectangle(frame, top_left, bottom_right, (10, 14, 18), -1)
    cv2.rectangle(frame, top_left, bottom_right, color, max(1, thickness), lineType=cv2.LINE_AA)
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (8, 10, 12),
        thickness + 2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(frame, text, (x, y), font, scale, (242, 246, 248), thickness, lineType=cv2.LINE_AA)


def _draw_timer_badge(
    frame: np.ndarray,
    state: dict[str, Any],
    bbox_xyxy: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        return

    duration = float(state.get("duration", 0.0))
    threshold = float(state.get("threshold_seconds", 0.0))
    alert_ready = bool(state.get("alert_ready", False))
    progress = 0.0 if threshold <= 0 else max(0.0, min(1.0, duration / threshold))
    color = (64, 96, 255) if alert_ready else (245, 120, 120)

    visual_scale = _visual_scale(frame)
    bar_height = max(4, int(round(4 * visual_scale)))
    bar_top = max(y1, y2 - bar_height)
    _draw_filled_rect_alpha(frame, (x1, bar_top), (x2, y2), (8, 10, 12), 0.82)
    fill_right = x1 + int(round((x2 - x1) * progress))
    if fill_right > x1:
        cv2.rectangle(frame, (x1, bar_top), (fill_right, y2), color, -1)

    if x2 - x1 < 58:
        return

    text = "ALERT" if alert_ready else _format_seconds(duration)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42 * visual_scale
    thickness = max(1, int(round(1.0 * visual_scale)))
    pad_x = max(4, int(round(5 * visual_scale)))
    pad_y = max(2, int(round(3 * visual_scale)))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    badge_width = text_width + (pad_x * 2)
    badge_height = text_height + baseline + (pad_y * 2)
    badge_x2 = min(x2, frame.shape[1] - 2)
    badge_x1 = max(x1, badge_x2 - badge_width)
    badge_y1 = max(y1, min(y2 - badge_height, y1 + max(2, int(round(3 * visual_scale)))))
    badge_y2 = badge_y1 + badge_height

    _draw_filled_rect_alpha(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (8, 10, 12), 0.72)
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, lineType=cv2.LINE_AA)
    text_x = badge_x1 + pad_x
    text_y = badge_y2 - pad_y - baseline
    cv2.putText(frame, text, (text_x, text_y), font, scale, (8, 10, 12), thickness + 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, text, (text_x, text_y), font, scale, (242, 246, 248), thickness, lineType=cv2.LINE_AA)


def _draw_filled_rect_alpha(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1 = max(0, min(frame.shape[1], x1))
    x2 = max(0, min(frame.shape[1], x2))
    y1 = max(0, min(frame.shape[0], y1))
    y2 = max(0, min(frame.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    frame[y1:y2, x1:x2] = cv2.addWeighted(
        overlay[y1:y2, x1:x2],
        alpha,
        frame[y1:y2, x1:x2],
        1.0 - alpha,
        0,
    )


def _visual_scale(frame: np.ndarray) -> float:
    height = frame.shape[0]
    return max(1.0, min(1.55, height / 720.0))


def _line_thickness(frame: np.ndarray) -> int:
    return max(2, int(round(1.8 * _visual_scale(frame))))


def _object_label(obj: TrackedObject) -> str:
    if obj.class_name == "person":
        label = obj.identity_label or "person"
        return f"{label} #{obj.track_id}"
    return f"{obj.class_name} #{obj.track_id}"


def _format_seconds(value: float) -> str:
    total_seconds = max(0, int(round(value)))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    palette = [
        (75, 210, 145),
        (112, 215, 235),
        (255, 190, 70),
        (170, 125, 240),
        (245, 120, 120),
    ]
    return palette[track_id % len(palette)]
