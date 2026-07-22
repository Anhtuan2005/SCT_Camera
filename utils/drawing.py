"""Drawing helpers for tracked objects, zones, lines, and counters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from analytics.identity_status import KNOWN_PERSON_KIND, PENDING_PERSON_KIND, STRANGER_KIND
from analytics.intrusion import CountingLine
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
ALERT_COLOR = (48, 59, 255)
ALERT_FLASH_INTERVAL_SECONDS = 0.25
EMERGENCY_COLOR = (32, 48, 255)
EMERGENCY_DIM_COLOR = (38, 44, 138)
EMERGENCY_FLASH_INTERVAL_SECONDS = 0.5

_EMERGENCY_ALERT_LABELS = {
    "possible_fall": "SOS | POSSIBLE FALL",
    "possible_unresponsive": "SOS | POSSIBLE EMERGENCY",
}
_EMERGENCY_OBJECT_LABELS = {
    "possible_fall": "possible fall",
    "possible_unresponsive": "possible emergency",
}
_EMERGENCY_ALERT_PRIORITY = {
    "possible_fall": 1,
    "possible_unresponsive": 2,
}
_POSE_DISPLAY_LABELS = {
    "getting_up": "getting up",
    "sitting_down": "sitting down",
    "changing_posture": "changing posture",
    "changing_to_lying": "changing posture",
}


@dataclass(frozen=True)
class _AlertVisualState:
    zone_ids: set[str]
    line_ids: set[str]
    track_ids: set[int]
    flash_on: bool
    emergency_labels_by_track: dict[int, str]
    emergency_flash_on: bool
    emergency_label: str | None


def draw_annotations(
    frame: np.ndarray,
    tracked_objects: list[TrackedObject],
    camera_config: dict[str, Any],
    counters: dict[str, dict[str, int]] | None = None,
    person_timer_states: dict[int, dict[str, Any]] | None = None,
    active_alerts: list[dict[str, Any]] | None = None,
    theft_states: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Draw zones, lines, track boxes, histories, and counters on a frame."""
    counters = counters or {}
    person_timer_states = person_timer_states or {}
    theft_states = theft_states or []
    alert_state = _alert_visual_state(active_alerts or [], theft_states)
    zones = [Zone.from_config(item) for item in camera_config.get("zones", []) if len(item.get("polygon", [])) >= 3]
    lines = [CountingLine.from_config(item) for item in camera_config.get("lines", [])]

    annotated = frame.copy()
    for zone in zones:
        _draw_zone(
            annotated,
            zone,
            alerting=zone.id in alert_state.zone_ids,
            flash_on=alert_state.flash_on,
        )

    for line in lines:
        _draw_line(
            annotated,
            line,
            counters.get(line.id, {"in": 0, "out": 0}),
            alerting=line.id in alert_state.line_ids,
            flash_on=alert_state.flash_on,
        )

    for obj in tracked_objects:
        _draw_object(
            annotated,
            obj,
            person_timer_states.get(obj.track_id),
            alerting=obj.track_id in alert_state.track_ids,
            flash_on=alert_state.flash_on,
            emergency_label=alert_state.emergency_labels_by_track.get(obj.track_id),
            emergency_flash_on=alert_state.emergency_flash_on,
        )

    _draw_frame_hud(annotated, camera_config, tracked_objects)
    if camera_config.get("show_theft_overlay", False) and theft_states:
        _draw_theft_overlay(annotated, theft_states)
    if alert_state.emergency_label:
        _draw_emergency_overlay(
            annotated,
            alert_state.emergency_label,
            alert_state.emergency_flash_on,
        )
    return annotated


def _draw_zone(
    frame: np.ndarray,
    zone: Zone,
    alerting: bool = False,
    flash_on: bool = False,
) -> None:
    polygon = zone.pixel_polygon(frame.shape)
    color = ALERT_COLOR if alerting and flash_on else COLOR_BY_ZONE.get(zone.zone_type, (150, 170, 190))
    thickness = _line_thickness(frame) + (2 if alerting and flash_on else 0)
    shadow = max(thickness + 1, 3)
    cv2.polylines(frame, [polygon], isClosed=True, color=(10, 14, 18), thickness=shadow, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    label_point = tuple(polygon[0])
    label = f"ALERT {zone.name}" if alerting and flash_on else zone.name
    _draw_label(frame, label, label_point, color)


def _draw_line(
    frame: np.ndarray,
    line: CountingLine,
    counter: dict[str, int],
    alerting: bool = False,
    flash_on: bool = False,
) -> None:
    p1, p2 = line.pixel_points(frame.shape)
    color = ALERT_COLOR if alerting and flash_on else (112, 215, 235)
    thickness = _line_thickness(frame) + (2 if alerting and flash_on else 0)
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
    alerting: bool = False,
    flash_on: bool = False,
    emergency_label: str | None = None,
    emergency_flash_on: bool = False,
) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in obj.bbox_xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    # Suspicious alerts only flash confirmed strangers. Fall emergencies must
    # remain visible for every identity state, including known people.
    is_stranger = obj.identity_kind == STRANGER_KIND
    emergency = emergency_label is not None
    should_flash = not emergency and alerting and flash_on and is_stranger
    if emergency:
        color = EMERGENCY_COLOR if emergency_flash_on else EMERGENCY_DIM_COLOR
        emphasis = 3 if emergency_flash_on else 1
    else:
        color = ALERT_COLOR if should_flash else _color_for_object(obj)
        emphasis = 2 if should_flash else 0
    thickness = _line_thickness(frame) + emphasis
    shadow = max(thickness + 1, 3)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 14, 18), shadow, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    label = _object_label(obj, emergency_label)
    if emergency:
        label = f"SOS {label}"
    elif should_flash:
        label = f"ALERT {label}"
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
    visual_scale = _visual_scale(frame)
    scale = 0.56 * visual_scale
    thickness = max(1, int(round(1.15 * visual_scale)))
    (text_width, _), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    centered_x = (frame.shape[1] - text_width) // 2
    _draw_label(frame, text, (centered_x, 34), (75, 210, 145))


def _draw_theft_overlay(frame: np.ndarray, states: list[dict[str, Any]]) -> None:
    state = max(
        states,
        key=lambda item: (
            bool(item.get("alerted", False)),
            int(item.get("score", 0)),
            float(item.get("near_seconds", 0.0)),
        ),
    )
    score = int(state.get("score", 0))
    score_threshold = max(1, int(state.get("score_threshold", 1)))
    near_seconds = float(state.get("near_seconds", 0.0))
    near_threshold = float(state.get("near_threshold", 0.0))
    passes = int(state.get("pacing_passes", 0))
    pass_threshold = max(1, int(state.get("pacing_threshold", 1)))
    alerted = bool(state.get("alerted", False))
    person_id = int(state.get("person_track_id", 0))
    vehicle_id = int(state.get("vehicle_track_id", 0))
    signal_names = {
        "near_vehicle_duration": "near",
        "pacing_near_vehicle": "pacing",
        "vehicle_started_moving": "moved",
        "moving_same_direction": "same-dir",
        "pose_push_contact": "contact",
    }
    signals = [
        signal_names.get(str(item), str(item))
        for item in state.get("behaviors", [])
    ]
    signal_text = ", ".join(signals) if signals else "none"
    title = f"THEFT ALERT {score}/{score_threshold}" if alerted else f"THEFT SCORE {score}/{score_threshold}"
    lines = [
        title,
        f"P#{person_id} + V#{vehicle_id}   Near {near_seconds:.1f}/{near_threshold:.0f}s   Pass {passes}/{pass_threshold}",
        f"Signals: {signal_text}",
    ]

    visual_scale = _visual_scale(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.44 * visual_scale
    title_scale = 0.50 * visual_scale
    thickness = max(1, int(round(1.0 * visual_scale)))
    pad_x = max(8, int(round(9 * visual_scale)))
    pad_y = max(6, int(round(7 * visual_scale)))
    line_gap = max(5, int(round(6 * visual_scale)))
    sizes = [
        cv2.getTextSize(line, font, title_scale if index == 0 else text_scale, thickness)
        for index, line in enumerate(lines)
    ]
    panel_width = max(size[0][0] for size in sizes) + (pad_x * 2)
    panel_height = sum(size[0][1] + size[1] for size in sizes) + (pad_y * 2) + (line_gap * 2)
    margin = max(10, int(round(12 * visual_scale)))
    x2 = frame.shape[1] - margin
    x1 = max(margin, x2 - panel_width)
    y1 = max(56, int(round(70 * visual_scale)))
    y2 = min(frame.shape[0] - margin, y1 + panel_height)
    color = ALERT_COLOR if alerted else COLOR_BY_ZONE["asset_watch"]

    _draw_filled_rect_alpha(frame, (x1, y1), (x2, y2), (8, 10, 12), 0.76)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, thickness), lineType=cv2.LINE_AA)
    cursor_y = y1 + pad_y
    for index, line in enumerate(lines):
        (_, text_height), baseline = sizes[index]
        cursor_y += text_height
        scale = title_scale if index == 0 else text_scale
        cv2.putText(
            frame,
            line,
            (x1 + pad_x, cursor_y),
            font,
            scale,
            (8, 10, 12),
            thickness + 2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x1 + pad_x, cursor_y),
            font,
            scale,
            (242, 246, 248),
            thickness,
            lineType=cv2.LINE_AA,
        )
        cursor_y += baseline + line_gap


def _draw_emergency_overlay(frame: np.ndarray, label: str, flash_on: bool) -> None:
    """Draw a persistent SOS frame whose emphasis pulses for fall emergencies."""
    color = EMERGENCY_COLOR if flash_on else EMERGENCY_DIM_COLOR
    visual_scale = _visual_scale(frame)
    inset = max(3, int(round(4 * visual_scale)))
    thickness = max(3, int(round((6 if flash_on else 3) * visual_scale)))
    cv2.rectangle(
        frame,
        (inset, inset),
        (frame.shape[1] - inset - 1, frame.shape[0] - inset - 1),
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56 * visual_scale
    text_thickness = max(1, int(round(1.15 * visual_scale)))
    (text_width, _), _ = cv2.getTextSize(label, font, scale, text_thickness)
    origin_x = max(8, (frame.shape[1] - text_width) // 2)
    origin_y = max(70, int(round(76 * visual_scale)))
    _draw_label(frame, label, (origin_x, origin_y), color)


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


def _alert_visual_state(
    active_alerts: list[dict[str, Any]],
    theft_states: list[dict[str, Any]] | None = None,
) -> _AlertVisualState:
    now = time.monotonic()
    zone_ids: set[str] = set()
    line_ids: set[str] = set()
    track_ids: set[int] = set()
    emergency_types_by_track: dict[int, str] = {}
    flash_on = False
    emergency_flash_on = False
    emergency_type: str | None = None
    for alert in active_alerts:
        expires_at = _float_or_none(alert.get("expires_at"))
        if expires_at is not None and expires_at <= now:
            continue
        alert_type = str(alert.get("type", ""))
        if alert_type == "fall_recovery":
            continue
        started_at = _float_or_none(alert.get("started_at"))
        started_at = now if started_at is None else started_at
        elapsed = max(0.0, now - started_at)
        track_id = _int_or_none(alert.get("track_id"))
        if alert_type in _EMERGENCY_ALERT_LABELS:
            current_type = emergency_types_by_track.get(track_id) if track_id is not None else None
            if track_id is not None and (
                current_type is None
                or _EMERGENCY_ALERT_PRIORITY[alert_type]
                > _EMERGENCY_ALERT_PRIORITY[current_type]
            ):
                emergency_types_by_track[track_id] = alert_type
            emergency_flash_on = emergency_flash_on or (
                int(elapsed / EMERGENCY_FLASH_INTERVAL_SECONDS) % 2 == 0
            )
            if emergency_type is None or (
                _EMERGENCY_ALERT_PRIORITY[alert_type]
                > _EMERGENCY_ALERT_PRIORITY[emergency_type]
            ):
                emergency_type = alert_type
            continue
        zone_id = alert.get("zone_id")
        if zone_id is not None:
            zone_ids.add(str(zone_id))
        line_id = alert.get("line_id")
        if line_id is not None:
            line_ids.add(str(line_id))
        if track_id is not None:
            track_ids.add(track_id)
        flash_on = flash_on or int(elapsed / ALERT_FLASH_INTERVAL_SECONDS) % 2 == 0
    alerted_theft_states = [state for state in theft_states or [] if bool(state.get("alerted", False))]
    for state in alerted_theft_states:
        zone_id = state.get("zone_id")
        if zone_id is not None:
            zone_ids.add(str(zone_id))
    if alerted_theft_states:
        flash_on = flash_on or int(now / ALERT_FLASH_INTERVAL_SECONDS) % 2 == 0
    emergency_label = _EMERGENCY_ALERT_LABELS.get(emergency_type) if emergency_type else None
    emergency_labels_by_track = {
        track_id: _EMERGENCY_OBJECT_LABELS[alert_type]
        for track_id, alert_type in emergency_types_by_track.items()
    }
    return _AlertVisualState(
        zone_ids,
        line_ids,
        track_ids,
        flash_on,
        emergency_labels_by_track,
        emergency_flash_on,
        emergency_label,
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _visual_scale(frame: np.ndarray) -> float:
    height = frame.shape[0]
    return max(1.0, min(1.55, height / 720.0))


def _line_thickness(frame: np.ndarray) -> int:
    return max(2, int(round(1.8 * _visual_scale(frame))))


def _object_label(obj: TrackedObject, pose_label: str | None = None) -> str:
    if obj.class_name == "person":
        label = obj.identity_label or "person"
        if obj.identity_kind != KNOWN_PERSON_KIND:
            label = f"{label} #{obj.track_id}"
        pose_label = obj.pose_label if pose_label is None else pose_label
        if pose_label and pose_label not in {"unknown", "upright"}:
            label = f"{label} - {_POSE_DISPLAY_LABELS.get(pose_label, pose_label)}"
        return label
    return f"{obj.class_name} #{obj.track_id}"


def _format_seconds(value: float) -> str:
    total_seconds = max(0, int(round(value)))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _color_for_object(obj: TrackedObject) -> tuple[int, int, int]:
    """Pick a color based on identity status.

    - Known person: stable green
    - Pending/Identifying: amber/yellow
    - Stranger: magenta/pink (high visibility)
    - Non-person objects: palette by track_id
    """
    if obj.class_name == "person":
        if obj.identity_kind == KNOWN_PERSON_KIND:
            return (75, 210, 145)      # green — safe, recognized
        if obj.identity_kind == PENDING_PERSON_KIND:
            return (0, 190, 255)       # amber — still identifying
        if obj.identity_kind == STRANGER_KIND:
            return (180, 80, 255)      # magenta — stranger, attention
    return _color_for_id(obj.track_id)


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    palette = [
        (75, 210, 145),
        (112, 215, 235),
        (255, 190, 70),
        (170, 125, 240),
        (245, 120, 120),
    ]
    return palette[track_id % len(palette)]
