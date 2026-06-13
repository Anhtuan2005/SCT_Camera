"""Behavior analytics orchestrator."""

from __future__ import annotations

from datetime import datetime
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


class BehaviorEngine:
    """Run all configured behavior rules for a camera frame."""

    def __init__(self, settings: dict[str, Any]) -> None:
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
        self.identity_resolver = PersonIdentityResolver(settings)
        self.learning = BehaviorLearningService(settings)

    def label_objects(
        self,
        tracked_objects: list[TrackedObject],
        camera_config: dict[str, Any],
        frame_bgr: Any,
    ) -> list[TrackedObject]:
        """Attach person/animal labels before analytics and drawing."""
        camera_id = str(camera_config.get("camera_id", "unknown"))
        return self.identity_resolver.label_objects(camera_id, tracked_objects, frame_bgr)

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
        alerts.extend(
            self.unknown_person.analyze(
                camera_id, camera_name, tracked_objects, timestamp
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
        """Return current full-frame loitering timers for drawing."""
        return self.loitering.get_active_states(camera_id)

    def get_stranger_watch_states(self, camera_id: str) -> dict[int, dict[str, Any]]:
        """Return current stranger-watch timer states."""
        return self.suspicious_stranger.get_active_states(camera_id)

    @staticmethod
    def _load_zones(camera_config: dict[str, Any]) -> list[Zone]:
        zones: list[Zone] = []
        for raw_zone in camera_config.get("zones", []):
            try:
                zone = Zone.from_config(raw_zone)
                if len(zone.polygon) >= 3:
                    zones.append(zone)
            except (TypeError, ValueError) as exc:
                logger.warning("Invalid zone config skipped: %s", exc)
        return zones

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
