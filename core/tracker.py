"""ByteTrack wrapper using Ultralytics built-in tracking."""

from __future__ import annotations

import threading
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.detector import Detection, YOLOv11Detector
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_LOST_TRACKS = 20
_LOST_TRACK_DISTANCE_THRESHOLD = 0.08
_LOST_TRACK_AREA_RATIO_MIN = 0.5
_LOST_TRACK_AREA_RATIO_MAX = 2.0
_LOST_PERSON_TRACK_AREA_RATIO_MIN = 0.3
_LOST_PERSON_TRACK_AREA_RATIO_MAX = 3.0
_CLASS_SMOOTHING_HISTORY_LENGTH = 8
_CLASS_SWITCH_CONFIRM_FRAMES = 5
_PERSON_CLASS_ID = 0
_GHOST_ANIMAL_CLASS_IDS = {15, 16}
_PERSON_ANIMAL_FLIP_IOU_THRESHOLD = 0.5
_GHOST_ANIMAL_PERSON_IOU_THRESHOLD = 0.4


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
class _LostTrack:
    """Recently lost track for position-based re-identification."""

    app_track_id: int
    class_id: int
    center: tuple[float, float]
    bbox_area: float


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
    pose_label: str | None = None
    class_confirmed: bool = True

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
        self._settings_lock = threading.RLock()
        self.tracker_arg_overrides = _parse_tracker_arg_overrides(tracking_settings)
        self.tracker_arg_overrides.setdefault("track_buffer", 90)
        self._grace_frames = int(
            tracking_settings.get(
                "track_grace_frames",
                self.tracker_arg_overrides["track_buffer"],
            )
        )
        self._duplicate_iou_threshold = float(tracking_settings.get("duplicate_iou_threshold", 0.85))
        self._duplicate_containment_threshold = float(
            tracking_settings.get("duplicate_containment_threshold", 0.7)
        )
        self.lost_track_reid_enabled = bool(
            tracking_settings.get("lost_track_reid_enabled", False)
        )
        self._class_smoothing_history_length = max(
            1,
            int(
                tracking_settings.get(
                    "class_smoothing_history_length",
                    _CLASS_SMOOTHING_HISTORY_LENGTH,
                )
            ),
        )
        self._class_switch_confirm_frames = max(
            1,
            int(
                tracking_settings.get(
                    "class_switch_confirm_frames",
                    _CLASS_SWITCH_CONFIRM_FRAMES,
                )
            ),
        )
        self._person_animal_flip_iou_threshold = float(
            tracking_settings.get(
                "person_animal_flip_iou_threshold",
                _PERSON_ANIMAL_FLIP_IOU_THRESHOLD,
            )
        )
        self._ghost_animal_person_iou_threshold = float(
            tracking_settings.get(
                "ghost_animal_person_iou_threshold",
                _GHOST_ANIMAL_PERSON_IOU_THRESHOLD,
            )
        )
        self._warn_tracker_memory_mismatch()
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
        self._raw_track_ids_seen: set[int] = set()
        self._next_track_alias = 1
        self._recycled_ids: deque[int] = deque()
        self._lost_track_positions: deque[_LostTrack] = deque(
            maxlen=_MAX_LOST_TRACKS
        )
        self._raw_class_votes: dict[int, deque[int]] = {}
        self._raw_stable_class_ids: dict[int, int] = {}
        self._raw_confirmed_class_ids: set[int] = set()
        self._raw_class_switch_candidates: dict[int, tuple[int, int]] = {}
        self._raw_last_bboxes: dict[int, tuple[float, float, float, float]] = {}

    @property
    def grace_frames(self) -> int:
        with self._settings_lock:
            return self._grace_frames

    @grace_frames.setter
    def grace_frames(self, value: int) -> None:
        with self._settings_lock:
            self._grace_frames = value

    @property
    def duplicate_iou_threshold(self) -> float:
        with self._settings_lock:
            return self._duplicate_iou_threshold

    @duplicate_iou_threshold.setter
    def duplicate_iou_threshold(self, value: float) -> None:
        with self._settings_lock:
            self._duplicate_iou_threshold = value

    @property
    def duplicate_containment_threshold(self) -> float:
        with self._settings_lock:
            return self._duplicate_containment_threshold

    @duplicate_containment_threshold.setter
    def duplicate_containment_threshold(self, value: float) -> None:
        with self._settings_lock:
            self._duplicate_containment_threshold = value

    def track(self, frame_bgr: np.ndarray) -> list[TrackedObject]:
        """Track configured classes in a BGR frame."""
        detections = self.detector.detect(frame_bgr)
        tracks = self._tracker.update(self._to_tracker_detections(detections), frame_bgr)
        if tracks is None or len(tracks) == 0:
            return self._mark_missing_and_stale(set())

        raw_track_counts = Counter(int(row[4]) for row in tracks)
        active_ids: set[int] = set()
        active_objects: list[TrackedObject] = []
        for row in tracks:
            bbox = tuple(float(value) for value in row[:4])
            raw_track_id = int(row[4])
            confidence = float(row[5])
            raw_class_id = int(row[6])
            class_id = raw_class_id
            class_confirmed = True
            if raw_track_counts[raw_track_id] == 1:
                class_id, class_confirmed = self._stable_class_id(
                    raw_track_id,
                    raw_class_id,
                    confidence,
                    bbox,
                )
                if class_id == raw_class_id:
                    self._raw_last_bboxes[raw_track_id] = bbox
            track_id = self._app_track_id(raw_track_id, class_id, row)
            active_ids.add(track_id)
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
                class_confirmed=class_confirmed,
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
        self._raw_track_ids_seen.clear()
        self._next_track_alias = 1
        self._recycled_ids.clear()
        self._lost_track_positions.clear()
        self._raw_class_votes.clear()
        self._raw_stable_class_ids.clear()
        self._raw_confirmed_class_ids.clear()
        self._raw_class_switch_candidates.clear()
        self._raw_last_bboxes.clear()

    def update_settings(self, settings: dict[str, Any]) -> None:
        """Apply mutable tracker settings without rebuilding the pipeline."""
        tracking_settings = settings.get("tracking", {})
        tracker_arg_overrides = _parse_tracker_arg_overrides(tracking_settings)
        tracker_arg_overrides.setdefault("track_buffer", 90)
        with self._settings_lock:
            if "track_grace_frames" in tracking_settings:
                self._grace_frames = int(tracking_settings["track_grace_frames"])
            elif "track_buffer" in tracking_settings:
                self._grace_frames = int(tracker_arg_overrides["track_buffer"])
            self._duplicate_iou_threshold = float(
                tracking_settings.get(
                    "duplicate_iou_threshold",
                    self._duplicate_iou_threshold,
                )
            )
            self._duplicate_containment_threshold = float(
                tracking_settings.get(
                    "duplicate_containment_threshold",
                    self._duplicate_containment_threshold,
                )
            )
            self.lost_track_reid_enabled = bool(
                tracking_settings.get(
                    "lost_track_reid_enabled",
                    self.lost_track_reid_enabled,
                )
            )
            new_history_length = max(
                1,
                int(
                    tracking_settings.get(
                        "class_smoothing_history_length",
                        self._class_smoothing_history_length,
                    )
                ),
            )
            if new_history_length != self._class_smoothing_history_length:
                self._class_smoothing_history_length = new_history_length
                self._raw_class_votes = {
                    raw_id: deque(votes, maxlen=new_history_length)
                    for raw_id, votes in self._raw_class_votes.items()
                }
            self._class_switch_confirm_frames = max(
                1,
                int(
                    tracking_settings.get(
                        "class_switch_confirm_frames",
                        self._class_switch_confirm_frames,
                    )
                ),
            )
            self._person_animal_flip_iou_threshold = float(
                tracking_settings.get(
                    "person_animal_flip_iou_threshold",
                    self._person_animal_flip_iou_threshold,
                )
            )
            self._ghost_animal_person_iou_threshold = float(
                tracking_settings.get(
                    "ghost_animal_person_iou_threshold",
                    self._ghost_animal_person_iou_threshold,
                )
            )
        self.tracker_arg_overrides = tracker_arg_overrides
        self._warn_tracker_memory_mismatch()
        self._apply_tracker_arg_overrides(self._tracker)

    def _warn_tracker_memory_mismatch(self) -> None:
        track_buffer = int(self.tracker_arg_overrides.get("track_buffer", 0))
        if track_buffer <= 0:
            return
        if self.grace_frames < track_buffer and not self.lost_track_reid_enabled:
            logger.warning(
                "Tracker wrapper expires tracks after %d frames before ByteTrack "
                "buffer %d frames, and lost_track_reid_enabled is false; "
                "occlusions may cause app track_id churn.",
                self.grace_frames,
                track_buffer,
            )
        elif self.grace_frames > track_buffer:
            logger.warning(
                "Tracker wrapper grace %d frames exceeds ByteTrack buffer %d "
                "frames; stale app objects may outlive ByteTrack memory.",
                self.grace_frames,
                track_buffer,
            )

    def _stable_class_id(
        self,
        raw_track_id: int,
        raw_class_id: int,
        raw_confidence: float,
        bbox: tuple[float, float, float, float],
    ) -> tuple[int, bool]:
        votes = self._raw_class_votes.setdefault(
            raw_track_id,
            deque(maxlen=self._class_smoothing_history_length),
        )
        votes.append(raw_class_id)
        stable_class_id = self._raw_stable_class_ids.get(raw_track_id)
        if stable_class_id is None:
            self._raw_stable_class_ids[raw_track_id] = raw_class_id
            if self._class_switch_confirm_frames <= 1:
                self._raw_confirmed_class_ids.add(raw_track_id)
                return raw_class_id, True
            return raw_class_id, False

        class_confirmed = raw_track_id in self._raw_confirmed_class_ids

        if raw_class_id == stable_class_id:
            self._raw_class_switch_candidates.pop(raw_track_id, None)
            if (
                not class_confirmed
                and _recent_class_vote_count(votes, stable_class_id)
                >= self._class_switch_confirm_frames
                and _majority_class_id(votes) == stable_class_id
            ):
                self._raw_confirmed_class_ids.add(raw_track_id)
                class_confirmed = True
            return stable_class_id, class_confirmed

        previous_bbox = self._raw_last_bboxes.get(raw_track_id)
        previous_iou = (
            _bbox_iou(bbox, previous_bbox) if previous_bbox is not None else 0.0
        )
        if (
            stable_class_id == _PERSON_CLASS_ID
            and raw_class_id in _GHOST_ANIMAL_CLASS_IDS
            and previous_bbox is not None
            and previous_iou >= self._person_animal_flip_iou_threshold
        ):
            logger.debug(
                (
                    "Suppressing person-animal class flip raw_track_id=%d "
                    "raw_class=%s(%d) confidence=%.3f stable_class=%s(%d) "
                    "iou=%.3f"
                ),
                raw_track_id,
                self.detector.class_name(raw_class_id),
                raw_class_id,
                raw_confidence,
                self.detector.class_name(stable_class_id),
                stable_class_id,
                previous_iou,
            )
            self._raw_class_switch_candidates.pop(raw_track_id, None)
            return stable_class_id, class_confirmed

        candidate_class_id, candidate_count = self._raw_class_switch_candidates.get(
            raw_track_id,
            (raw_class_id, 0),
        )
        if candidate_class_id != raw_class_id:
            candidate_count = 0
        candidate_count += 1
        self._raw_class_switch_candidates[raw_track_id] = (
            raw_class_id,
            candidate_count,
        )
        if (
            candidate_count >= self._class_switch_confirm_frames
            and _majority_class_id(votes) == raw_class_id
        ):
            self._retire_app_track_for_alias((stable_class_id, raw_track_id))
            self._raw_stable_class_ids[raw_track_id] = raw_class_id
            self._raw_confirmed_class_ids.add(raw_track_id)
            self._raw_class_switch_candidates.pop(raw_track_id, None)
            return raw_class_id, True

        return stable_class_id, class_confirmed

    def _retire_app_track_for_alias(self, alias_key: tuple[int, int]) -> None:
        track_id = self._track_id_aliases.pop(alias_key, None)
        if track_id is None:
            return
        self._track_alias_keys.pop(track_id, None)
        self._history.pop(track_id, None)
        self._last_objects.pop(track_id, None)
        self._missing_frames.pop(track_id, None)

    def _app_track_id(
        self,
        raw_track_id: int,
        class_id: int,
        row: Any = None,
    ) -> int:
        key = (class_id, raw_track_id)
        existing = self._track_id_aliases.get(key)
        if existing is not None:
            return existing

        if self.lost_track_reid_enabled and row is not None:
            reused = self._try_reuse_lost_track_id(class_id, row)
            if reused is not None:
                self._track_id_aliases[key] = reused
                self._track_alias_keys[reused] = key
                self._raw_track_ids_seen.add(raw_track_id)
                return reused

        # Use recycled ID if available, otherwise increment
        app_track_id: int | None = None
        while self._recycled_ids:
            candidate = self._recycled_ids.popleft()
            if candidate not in self._track_alias_keys:
                app_track_id = candidate
                break
        if app_track_id is None:
            while self._next_track_alias in self._track_alias_keys:
                self._next_track_alias += 1
            app_track_id = self._next_track_alias
            self._next_track_alias += 1

        self._track_id_aliases[key] = app_track_id
        self._track_alias_keys[app_track_id] = key
        self._raw_track_ids_seen.add(raw_track_id)
        return app_track_id

    def _try_reuse_lost_track_id(
        self,
        class_id: int,
        row: Any,
    ) -> int | None:
        """Find a recently lost track at a similar position and reuse its ID."""
        if not self._lost_track_positions:
            return None
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        area = max(1.0, (x2 - x1) * (y2 - y1))
        # Use bbox diagonal as distance reference
        diag = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        threshold = diag * (_LOST_TRACK_DISTANCE_THRESHOLD / 0.08) * 5.0

        best_lost: _LostTrack | None = None
        best_dist = float("inf")
        for lost in self._lost_track_positions:
            if lost.class_id != class_id:
                continue
            if lost.app_track_id in self._track_alias_keys:
                continue
            area_ratio = area / max(1.0, lost.bbox_area)
            min_area_ratio, max_area_ratio = _lost_track_area_ratio_bounds(class_id)
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            dist = ((center[0] - lost.center[0]) ** 2 + (center[1] - lost.center[1]) ** 2) ** 0.5
            if dist < threshold and dist < best_dist:
                best_dist = dist
                best_lost = lost

        if best_lost is not None:
            self._lost_track_positions.remove(best_lost)
            logger.debug(
                "Reusing lost track ID %d for class %d (dist=%.1f)",
                best_lost.app_track_id,
                class_id,
                best_dist,
            )
            return best_lost.app_track_id
        return None

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
            # Track expired — save position for re-ID, recycle ID
            lost_obj = self._last_objects[track_id]
            cx, cy = lost_obj.center
            x1, y1, x2, y2 = lost_obj.bbox_xyxy
            bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
            self._lost_track_positions.append(
                _LostTrack(
                    app_track_id=track_id,
                    class_id=lost_obj.class_id,
                    center=(cx, cy),
                    bbox_area=bbox_area,
                )
            )
            # Recycle the app track ID
            alias_key = self._track_alias_keys.pop(track_id, None)
            if alias_key is not None:
                self._track_id_aliases.pop(alias_key, None)
                self._forget_raw_class_state_if_unused(alias_key[1])
            self._recycled_ids.append(track_id)
            self._history.pop(track_id, None)
            self._last_objects.pop(track_id, None)
            self._missing_frames.pop(track_id, None)
        return stale_objects

    def _forget_raw_class_state_if_unused(self, raw_track_id: int) -> None:
        if any(key[1] == raw_track_id for key in self._track_id_aliases):
            return
        self._raw_class_votes.pop(raw_track_id, None)
        self._raw_stable_class_ids.pop(raw_track_id, None)
        self._raw_confirmed_class_ids.discard(raw_track_id)
        self._raw_class_switch_candidates.pop(raw_track_id, None)
        self._raw_last_bboxes.pop(raw_track_id, None)
        self._raw_track_ids_seen.discard(raw_track_id)

    def _dedupe_same_class(self, objects: list[TrackedObject]) -> list[TrackedObject]:
        if (
            self.duplicate_iou_threshold <= 0
            and self.duplicate_containment_threshold <= 0
        ) or len(objects) < 2:
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
                (
                    self.duplicate_iou_threshold > 0
                    and _bbox_iou(obj.bbox_xyxy, other.bbox_xyxy)
                    >= self.duplicate_iou_threshold
                )
                or (
                    self.duplicate_containment_threshold > 0
                    and _bbox_containment_ratio(obj.bbox_xyxy, other.bbox_xyxy)
                    >= self.duplicate_containment_threshold
                )
            )
            for other in others
        )

    def suppress_ghost_animal_over_person(
        self,
        objects: list[TrackedObject],
    ) -> list[TrackedObject]:
        return self._suppress_ghost_animal_over_person(objects)

    def _suppress_ghost_animal_over_person(
        self,
        objects: list[TrackedObject],
    ) -> list[TrackedObject]:
        persons = [obj for obj in objects if _is_person_object(obj)]
        if not persons:
            return objects

        kept: list[TrackedObject] = []
        for obj in objects:
            if not _is_ghost_animal_object(obj):
                kept.append(obj)
                continue
            suppressing_person: TrackedObject | None = None
            suppressing_iou = 0.0
            for person in persons:
                iou = _bbox_iou(obj.bbox_xyxy, person.bbox_xyxy)
                if iou >= self._ghost_animal_person_iou_threshold:
                    suppressing_person = person
                    suppressing_iou = iou
                    break
            if suppressing_person is None:
                kept.append(obj)
                continue
            logger.debug(
                (
                    "Suppressing ghost animal track_id=%d class=%s over "
                    "person_track_id=%d person_bbox=%s iou=%.3f"
                ),
                obj.track_id,
                obj.class_name,
                suppressing_person.track_id,
                suppressing_person.bbox_xyxy,
                suppressing_iou,
            )
        return kept


def _majority_class_id(votes: deque[int]) -> int:
    return Counter(votes).most_common(1)[0][0]


def _recent_class_vote_count(votes: deque[int], class_id: int) -> int:
    count = 0
    for vote in reversed(votes):
        if vote != class_id:
            break
        count += 1
    return count


def _is_person_object(obj: TrackedObject) -> bool:
    return obj.class_id == _PERSON_CLASS_ID or obj.class_name == "person"


def _is_ghost_animal_object(obj: TrackedObject) -> bool:
    return obj.class_id in _GHOST_ANIMAL_CLASS_IDS or obj.class_name in {"cat", "dog"}


def _lost_track_area_ratio_bounds(class_id: int) -> tuple[float, float]:
    if class_id == _PERSON_CLASS_ID:
        return _LOST_PERSON_TRACK_AREA_RATIO_MIN, _LOST_PERSON_TRACK_AREA_RATIO_MAX
    return _LOST_TRACK_AREA_RATIO_MIN, _LOST_TRACK_AREA_RATIO_MAX


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
