"""Behavior analytics orchestrator."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from analytics.asset_watch import AssetWatchDetector
from analytics.behavior_learning import BehaviorLearningService
from analytics.intrusion import IntrusionDetector
from analytics.line_counter import CountingLine, LineCounter
from analytics.loitering import LoiteringDetector
from analytics.person_identity import PersonIdentityResolver
from analytics.suspicious_stranger import SuspiciousStrangerDetector
from analytics.theft_behavior import SuspiciousTheftDetector
from analytics.unknown_person import UnknownPersonDetector
from analytics.zone import Zone
from core.tracker import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


_AUTO_GLOBAL_ZONE_BEHAVIORS = ("intrusion", "loitering", "stranger_watch")
_VIDEO_FILE_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


class BehaviorEngine:
    """Run all configured behavior rules for a camera frame."""

    def __init__(
        self,
        settings: dict[str, Any],
        identity_resolver: PersonIdentityResolver | None = None,
    ) -> None:
        behavior = settings.get("behavior", {})
        self.intrusion = IntrusionDetector(
            reset_frames=int(behavior.get("intrusion_reset_frames", 30)),
            allowed_classes=behavior.get("intrusion_classes", ["person"]),
        )
        self.loitering = LoiteringDetector(
            default_threshold_seconds=float(
                behavior.get("loitering_threshold_seconds", 30)
            )
        )
        self.suspicious_stranger = SuspiciousStrangerDetector(
            default_threshold_seconds=float(behavior.get("stranger_watch_seconds", 180)),
            settings=behavior.get("suspicious", {}),
        )
        self.unknown_person = UnknownPersonDetector()
        self.asset_watch = AssetWatchDetector(
            default_missing_seconds=float(behavior.get("asset_missing_seconds", 6)),
            settings=behavior.get("asset_watch", {}),
        )
        self.theft_behavior = SuspiciousTheftDetector(
            settings=behavior.get("theft", {}),
        )
        self.line_counter = LineCounter()
        self.identity_resolver = identity_resolver or PersonIdentityResolver(settings)
        self.learning = BehaviorLearningService(settings)
        self._person_entry_times: dict[tuple[str, int], float] = {}
        self._person_timer_states: dict[tuple[str, int], dict[str, Any]] = {}
        self._timer_visible_track_ids: dict[str, set[int]] = {}

    def label_objects(
        self,
        tracked_objects: list[TrackedObject],
        camera_config: dict[str, Any],
        frame_bgr: Any,
    ) -> list[TrackedObject]:
        """Attach person/animal labels before analytics and drawing."""
        camera_id = str(camera_config.get("camera_id", "unknown"))
        unknown_policy = str(
            camera_config.get("unknown_person_policy", "face_match")
        ).strip().lower()
        assume_unknown_persons = self._is_video_file_source(
            camera_config.get("source")
        ) or unknown_policy in {
            "assume_stranger",
            "unknown_by_default",
            "all_unknown",
        }
        return self.identity_resolver.label_objects(
            camera_id,
            tracked_objects,
            frame_bgr,
            assume_unknown_persons=assume_unknown_persons,
        )

    def analyze(
        self,
        tracked_objects: list[TrackedObject],
        camera_config: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> list[dict[str, Any]]:
        """Run intrusion, loitering, and line crossing rules."""
        camera_id = str(camera_config.get("camera_id", "unknown"))
        camera_name = str(camera_config.get("name", camera_id))
        timestamp = datetime.now().astimezone()
        zones = self._load_zones(camera_config)
        lines = self._load_lines(camera_config)
        self._update_person_presence_timers(camera_id, tracked_objects)

        alerts: list[dict[str, Any]] = []
        alerts.extend(
            self.intrusion.analyze(
                camera_id, camera_name, tracked_objects, zones, frame_shape, timestamp
            )
        )
        alerts.extend(
            self.loitering.analyze(
                camera_id, camera_name, tracked_objects, zones, frame_shape, timestamp
            )
        )
        stranger_candidates = self._objects_outside_intrusion_zones(
            tracked_objects,
            zones,
            frame_shape,
        )
        alerts.extend(
            self.unknown_person.analyze(
                camera_id, camera_name, stranger_candidates, timestamp
            )
        )
        alerts.extend(
            self.suspicious_stranger.analyze(
                camera_id, camera_name, tracked_objects, zones, frame_shape, timestamp
            )
        )
        alerts.extend(
            self.asset_watch.analyze(
                camera_id, camera_name, tracked_objects, zones, frame_shape, timestamp
            )
        )
        alerts.extend(
            self.theft_behavior.analyze(
                camera_id, camera_name, tracked_objects, zones, frame_shape, timestamp
            )
        )
        alerts.extend(
            self.line_counter.analyze(
                camera_id, camera_name, tracked_objects, lines, frame_shape, timestamp
            )
        )
        return self.learning.enrich_alerts(
            alerts,
            tracked_objects,
            camera_config,
            frame_shape,
            timestamp,
        )

    def get_counters(self, camera_id: str) -> dict[str, dict[str, int]]:
        """Return line counters for a camera."""
        return self.line_counter.get_counters(camera_id)

    def get_person_timer_states(self, camera_id: str) -> dict[int, dict[str, Any]]:
        """Return bbox timers for non-known people, upgraded with loitering state."""
        visible_track_ids = self._timer_visible_track_ids.get(camera_id, set())
        states = {
            track_id: dict(state)
            for (state_camera_id, track_id), state in self._person_timer_states.items()
            if state_camera_id == camera_id and track_id in visible_track_ids
        }
        for track_id, state in self.loitering.get_active_states(camera_id).items():
            if track_id in visible_track_ids:
                states[track_id] = state
        return states

    def get_stranger_watch_states(self, camera_id: str) -> dict[int, dict[str, Any]]:
        """Return current stranger-watch timer states."""
        return self.suspicious_stranger.get_active_states(camera_id)

    @classmethod
    def _load_zones(cls, camera_config: dict[str, Any]) -> list[Zone]:
        zones: list[Zone] = []
        for raw_zone in camera_config.get("zones", []):
            try:
                zone = Zone.from_config(raw_zone)
                if len(zone.polygon) >= 3:
                    zones.append(zone)
            except (TypeError, ValueError) as exc:
                logger.warning("Invalid zone config skipped: %s", exc)
        if cls._auto_global_zones_enabled(camera_config):
            zones.extend(cls._missing_auto_global_zones(zones))
        return zones

    @staticmethod
    def _auto_global_zones_enabled(camera_config: dict[str, Any]) -> bool:
        return bool(camera_config.get("auto_global_zone", True))

    @staticmethod
    def _is_video_file_source(source: Any) -> bool:
        if isinstance(source, int):
            return False
        text = str(source or "").strip()
        if not text or text.isdigit():
            return False
        if text.lower().startswith(("rtsp://", "http://", "https://")):
            return False
        return Path(text).suffix.lower() in _VIDEO_FILE_EXTENSIONS

    def _update_person_presence_timers(
        self,
        camera_id: str,
        objects: list[TrackedObject],
    ) -> None:
        now = time.monotonic()
        visible_track_ids: set[int] = set()
        for obj in objects:
            if not self._should_show_bbox_timer(obj):
                continue
            visible_track_ids.add(obj.track_id)
            key = (camera_id, obj.track_id)
            first_seen = self._person_entry_times.setdefault(key, now)
            self._person_timer_states[key] = {
                "camera_id": camera_id,
                "track_id": obj.track_id,
                "duration": now - first_seen,
                "threshold_seconds": 0.0,
                "remaining_seconds": 0.0,
                "alert_ready": False,
                "timer_kind": "presence",
            }
        self._timer_visible_track_ids[camera_id] = visible_track_ids

        active_keys = {(camera_id, track_id) for track_id in visible_track_ids}
        stale_keys = [
            key
            for key in self._person_entry_times
            if key[0] == camera_id and key not in active_keys
        ]
        for key in stale_keys:
            self._person_entry_times.pop(key, None)
            self._person_timer_states.pop(key, None)

    @staticmethod
    def _should_show_bbox_timer(obj: TrackedObject) -> bool:
        return obj.class_name == "person" and obj.identity_kind != "known_person"

    @staticmethod
    def _missing_auto_global_zones(zones: list[Zone]) -> list[Zone]:
        missing = [
            behavior_type
            for behavior_type in _AUTO_GLOBAL_ZONE_BEHAVIORS
            if not any(zone.applies_to(behavior_type) for zone in zones)
        ]
        return [
            Zone(
                id=f"__global_{behavior_type}__",
                name="Full Frame",
                zone_type=behavior_type,
                polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                auto_generated=True,
            )
            for behavior_type in missing
        ]

    @staticmethod
    def _load_lines(camera_config: dict[str, Any]) -> list[CountingLine]:
        lines: list[CountingLine] = []
        for raw_line in camera_config.get("lines", []):
            try:
                line = CountingLine.from_config(raw_line)
                lines.append(line)
            except (TypeError, ValueError) as exc:
                logger.warning("Invalid line config skipped: %s", exc)
        return lines

    @staticmethod
    def _objects_outside_intrusion_zones(
        objects: list[TrackedObject],
        zones: list[Zone],
        frame_shape: tuple[int, int, int],
    ) -> list[TrackedObject]:
        intrusion_zones = [
            zone
            for zone in zones
            if zone.applies_to("intrusion") and not zone.auto_generated
        ]
        if not intrusion_zones:
            return objects
        return [
            obj
            for obj in objects
            if not any(
                zone.contains_point(obj.center[0], obj.center[1], frame_shape)
                for zone in intrusion_zones
            )
        ]
