"""ByteTrack wrapper using Ultralytics built-in tracking."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.detector import Detection, YOLOv11Detector
from utils.logger import get_logger

logger = get_logger(__name__)


_TRACKER_ARG_TYPES: dict[str, type] = {
    "track_high_thresh": float,
    "track_low_thresh": float,
    "new_track_thresh": float,
    "track_buffer": int,
    "match_thresh": float,
}


@dataclass(frozen=True)
class _TrackerDetections:
    """NumPy detection view expected by Ultralytics BYTETracker."""

    xywh: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


class _SafeGMC:
    """Guard Ultralytics GMC against low-texture frames and invalid transforms."""

    def __init__(self, method: str, downscale: int) -> None:
        from ultralytics.trackers.utils.gmc import GMC

        self._gmc = GMC(method=method, downscale=downscale)

    def apply(self, frame: np.ndarray, detections: Any = None) -> np.ndarray:
        try:
            transform = self._gmc.apply(frame, detections)
        except Exception as exc:
            logger.debug("Camera motion estimation skipped: %s", exc)
            self._gmc.reset_params()
            return np.eye(2, 3, dtype=np.float32)

        if transform is None:
            return np.eye(2, 3, dtype=np.float32)
        transform = np.asarray(transform, dtype=np.float32)
        if transform.shape != (2, 3) or not np.isfinite(transform).all():
            return np.eye(2, 3, dtype=np.float32)
        return transform

    def reset_params(self) -> None:
        self._gmc.reset_params()


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
        self.detector = detector
        self.tracker_config = str(tracking_settings.get("tracker", "bytetrack.yaml"))
        self.history_length = int(tracking_settings.get("track_history_length", 50))
        self.grace_frames = int(tracking_settings.get("track_grace_frames", 8))
        self.duplicate_iou_threshold = float(tracking_settings.get("duplicate_iou_threshold", 0.85))
        self.duplicate_containment_threshold = float(
            tracking_settings.get("duplicate_containment_threshold", 0.7)
        )
        self.tracker_arg_overrides = _parse_tracker_arg_overrides(tracking_settings)
        self.tracker_arg_overrides.setdefault("track_buffer", 90)
        cmc_settings = tracking_settings.get("camera_motion_compensation", {})
        self.cmc_enabled = bool(cmc_settings.get("enabled", False))
        self.cmc_method = str(cmc_settings.get("method", "sparseOptFlow"))
        self.cmc_downscale = max(1, int(cmc_settings.get("downscale", 2)))
        self._tracker = self._build_tracker()
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self._last_objects: dict[int, TrackedObject] = {}
        self._missing_frames: dict[int, int] = {}
        self._track_id_aliases: dict[tuple[int, int], int] = {}
        self._track_alias_keys: dict[int, tuple[int, int]] = {}
        self._next_track_alias = 1

    def track(self, frame_bgr: np.ndarray) -> list[TrackedObject]:
        """Track configured classes in a BGR frame."""
        detections = self.detector.detect(frame_bgr)
        tracks = self._tracker.update(self._to_tracker_detections(detections), frame_bgr)
        if tracks is None or len(tracks) == 0:
            return self._mark_missing_and_stale(set())

        active_ids = {
            self._app_track_id(int(row[4]), int(row[6]))
            for row in tracks
        }
        active_objects: list[TrackedObject] = []
        for row in tracks:
            bbox = tuple(float(value) for value in row[:4])
            raw_track_id = int(row[4])
            confidence = float(row[5])
            class_id = int(row[6])
            track_id = self._app_track_id(raw_track_id, class_id)
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

    def reset(self) -> None:
        """Reset ByteTrack, GMC, and local object history for a new stream."""
        self._tracker.reset()
        gmc = getattr(self._tracker, "gmc", None)
        if gmc is not None:
            gmc.reset_params()
        self._history.clear()
        self._last_objects.clear()
        self._missing_frames.clear()
        self._track_id_aliases.clear()
        self._track_alias_keys.clear()
        self._next_track_alias = 1

    def update_settings(self, settings: dict[str, Any]) -> None:
        """Apply mutable tracker settings without rebuilding the pipeline."""
        tracking_settings = settings.get("tracking", {})
        self.grace_frames = int(
            tracking_settings.get("track_grace_frames", self.grace_frames)
        )
        self.duplicate_iou_threshold = float(
            tracking_settings.get(
                "duplicate_iou_threshold",
                self.duplicate_iou_threshold,
            )
        )
        self.duplicate_containment_threshold = float(
            tracking_settings.get(
                "duplicate_containment_threshold",
                self.duplicate_containment_threshold,
            )
        )
        self.tracker_arg_overrides = _parse_tracker_arg_overrides(tracking_settings)
        self.tracker_arg_overrides.setdefault("track_buffer", 90)
        self._apply_tracker_arg_overrides(self._tracker)

    def _app_track_id(self, raw_track_id: int, class_id: int) -> int:
        key = (class_id, raw_track_id)
        existing = self._track_id_aliases.get(key)
        if existing is not None:
            return existing

        if raw_track_id not in self._track_alias_keys:
            self._track_id_aliases[key] = raw_track_id
            self._track_alias_keys[raw_track_id] = key
            return raw_track_id

        while self._next_track_alias in self._track_alias_keys:
            self._next_track_alias += 1
        app_track_id = self._next_track_alias
        self._next_track_alias += 1
        self._track_id_aliases[key] = app_track_id
        self._track_alias_keys[app_track_id] = key
        return app_track_id

    def _build_tracker(self) -> Any:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, yaml_load
        from ultralytics.utils.checks import check_yaml

        config_path = check_yaml(self.tracker_config)
        tracker_args = yaml_load(config_path)
        tracker_args.update(self.tracker_arg_overrides)
        args = IterableSimpleNamespace(**tracker_args)
        if str(args.tracker_type).lower() != "bytetrack":
            raise ValueError(
                f"ByteTrackTracker requires tracker_type=bytetrack, got {args.tracker_type}"
            )

        if self.tracker_arg_overrides:
            logger.info("ByteTrack threshold overrides: %s", self.tracker_arg_overrides)
        tracker = BYTETracker(args=args, frame_rate=30)
        if self.cmc_enabled:
            tracker.gmc = _SafeGMC(self.cmc_method, self.cmc_downscale)
            logger.info(
                "ByteTrack camera motion compensation enabled: method=%s downscale=%d",
                self.cmc_method,
                self.cmc_downscale,
            )
        return tracker

    def _apply_tracker_arg_overrides(self, tracker: Any) -> None:
        args = getattr(tracker, "args", None)
        if args is None:
            return
        for key, value in self.tracker_arg_overrides.items():
            setattr(args, key, value)

    @staticmethod
    def _to_tracker_detections(detections: list[Detection]) -> _TrackerDetections:
        xywh: list[tuple[float, float, float, float]] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            xywh.append(
                (
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                    max(0.0, x2 - x1),
                    max(0.0, y2 - y1),
                )
            )
            confidences.append(detection.confidence)
            class_ids.append(detection.class_id)
        return _TrackerDetections(
            xywh=np.asarray(xywh, dtype=np.float32).reshape(-1, 4),
            conf=np.asarray(confidences, dtype=np.float32),
            cls=np.asarray(class_ids, dtype=np.float32),
        )

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
            and (
                _bbox_iou(obj.bbox_xyxy, other.bbox_xyxy)
                >= self.duplicate_iou_threshold
                or _bbox_containment_ratio(obj.bbox_xyxy, other.bbox_xyxy)
                >= self.duplicate_containment_threshold
            )
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


def _bbox_containment_ratio(
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
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0:
        return 0.0
    return intersection / smaller_area


def _parse_tracker_arg_overrides(
    tracking_settings: dict[str, Any],
) -> dict[str, float | int]:
    overrides: dict[str, float | int] = {}
    for key, value_type in _TRACKER_ARG_TYPES.items():
        if key not in tracking_settings:
            continue
        try:
            parsed = value_type(tracking_settings[key])
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, (float, int)) and parsed < 0:
            continue
        overrides[key] = parsed
    return overrides
