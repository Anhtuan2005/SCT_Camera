"""ByteTrack wrapper using Ultralytics built-in tracking."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.detector import YOLOv11Detector
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrackedObject:
    """A tracked object with identity and center-point history."""

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    center_history: list[tuple[float, float]]
    identity_label: str | None = None
    identity_kind: str | None = None
    identity_score: float | None = None
    pose_keypoints: list[tuple[float, float, float]] | None = None

    @property
    def center(self) -> tuple[float, float]:
        """Return current box center."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class ByteTrackTracker:
    """Track objects with Ultralytics ByteTrack and keep per-track history."""

    def __init__(self, detector: YOLOv11Detector, settings: dict[str, Any]) -> None:
        tracking_settings = settings.get("tracking", {})
        detection_settings = settings.get("detection", {})
        self.detector = detector
        self.tracker_config = str(tracking_settings.get("tracker", "bytetrack.yaml"))
        self.history_length = int(tracking_settings.get("track_history_length", 50))
        self.grace_frames = int(tracking_settings.get("track_grace_frames", 8))
        self.confidence = float(detection_settings.get("confidence", detector.confidence))
        self.class_ids = [int(item) for item in detection_settings.get("classes", detector.class_ids)]
        self.iou = float(detection_settings.get("iou", detector.iou))
        self.imgsz = int(detection_settings.get("imgsz", detector.imgsz))
        self.duplicate_iou_threshold = float(tracking_settings.get("duplicate_iou_threshold", 0.85))
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self._last_objects: dict[int, TrackedObject] = {}
        self._missing_frames: dict[int, int] = {}
        self._next_track_id = 1

    def track(self, frame_bgr: np.ndarray) -> list[TrackedObject]:
        """Track configured classes in a BGR frame."""
        with self.detector.inference_lock:
            results = self.detector.model.track(
                frame_bgr,
                persist=True,
                tracker=self.tracker_config,
                classes=self.class_ids,
                conf=self.confidence,
                iou=self.iou,
                device=self.detector.device,
                half=self.detector.use_half and self.detector.device.startswith("cuda"),
                imgsz=self.imgsz,
                verbose=False,
            )

        if not results:
            return self._mark_missing_and_stale(set())

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return self._mark_missing_and_stale(set())

        xyxys = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = [int(value) for value in boxes.cls.cpu().tolist()]
        raw_ids = getattr(boxes, "id", None)
        if raw_ids is None:
            track_ids = self._fallback_track_ids(xyxys, frame_bgr.shape)
        else:
            track_ids = [int(value) for value in raw_ids.cpu().tolist()]
            if track_ids:
                self._next_track_id = max(self._next_track_id, max(track_ids) + 1)

        active_ids = set(track_ids)
        active_objects: list[TrackedObject] = []
        for track_id, xyxy, confidence, class_id in zip(track_ids, xyxys, confidences, class_ids):
            bbox = tuple(float(value) for value in xyxy)
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self._history[track_id].append(center)
            obj = TrackedObject(
                track_id=track_id,
                bbox_xyxy=bbox,
                class_id=class_id,
                class_name=self.detector.class_name(class_id),
                confidence=float(confidence),
                center_history=list(self._history[track_id]),
            )
            self._last_objects[track_id] = obj
            self._missing_frames[track_id] = 0
            active_objects.append(obj)

        active_objects = self._dedupe_same_class(active_objects)
        stale_objects = self._mark_missing_and_stale(active_ids)
        stale_objects = [
            obj
            for obj in stale_objects
            if not self._overlaps_any_same_class(obj, active_objects)
        ]
        return self._dedupe_same_class([*active_objects, *stale_objects])

    def _fallback_track_ids(
        self,
        xyxys: list[list[float]],
        frame_shape: tuple[int, ...],
    ) -> list[int]:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        diagonal = max((height * height + width * width) ** 0.5, 1.0)
        max_distance = max(48.0, diagonal * 0.08)
        available_ids = set(self._last_objects)
        track_ids: list[int] = []

        for xyxy in xyxys:
            x1, y1, x2, y2 = [float(value) for value in xyxy]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            best_id: int | None = None
            best_distance = max_distance
            for track_id in list(available_ids):
                previous = self._last_objects[track_id].center
                distance = ((center[0] - previous[0]) ** 2 + (center[1] - previous[1]) ** 2) ** 0.5
                if distance < best_distance:
                    best_id = track_id
                    best_distance = distance
            if best_id is None:
                best_id = self._allocate_track_id()
            else:
                available_ids.discard(best_id)
            track_ids.append(best_id)
        return track_ids

    def _allocate_track_id(self) -> int:
        while self._next_track_id in self._last_objects:
            self._next_track_id += 1
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _mark_missing_and_stale(self, active_ids: set[int]) -> list[TrackedObject]:
        stale_objects: list[TrackedObject] = []
        for track_id in list(self._last_objects):
            if track_id in active_ids:
                continue
            missing_frames = self._missing_frames.get(track_id, 0) + 1
            if missing_frames <= self.grace_frames:
                self._missing_frames[track_id] = missing_frames
                stale_objects.append(self._last_objects[track_id])
                continue
            self._history.pop(track_id, None)
            self._last_objects.pop(track_id, None)
            self._missing_frames.pop(track_id, None)
        return stale_objects

    def _dedupe_same_class(self, objects: list[TrackedObject]) -> list[TrackedObject]:
        if self.duplicate_iou_threshold <= 0 or len(objects) < 2:
            return objects

        kept: list[TrackedObject] = []
        for obj in sorted(objects, key=lambda item: item.confidence, reverse=True):
            if self._overlaps_any_same_class(obj, kept):
                continue
            kept.append(obj)
        kept_ids = {obj.track_id for obj in kept}
        return [obj for obj in objects if obj.track_id in kept_ids]

    def _overlaps_any_same_class(self, obj: TrackedObject, others: list[TrackedObject]) -> bool:
        return any(
            obj.class_id == other.class_id
            and _bbox_iou(obj.bbox_xyxy, other.bbox_xyxy) >= self.duplicate_iou_threshold
            for other in others
        )


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    intersection = width * height
    if intersection <= 0:
        return 0.0

    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union
