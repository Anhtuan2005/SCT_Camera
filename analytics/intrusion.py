"""Intrusion detection rule for restricted zones."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

from analytics.zone import Zone
from core.tracker import TrackedObject


class IntrusionDetector:
    """Alert once while a qualifying person occupies an intrusion zone."""

    def __init__(
        self,
        reset_frames: int = 30,
        allowed_classes: Iterable[str] | str | None = None,
    ) -> None:
        self.reset_frames = reset_frames
        if isinstance(allowed_classes, str):
            allowed_classes = [allowed_classes]
        self.allowed_classes = {
            str(class_name)
            for class_name in (allowed_classes or ["person"])
        }
        self._occupied_zones: set[tuple[str, str]] = set()
        self._empty_frames: dict[tuple[str, str], int] = defaultdict(int)

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        zones: list[Zone],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return intrusion alert payloads for objects inside restricted zones."""
        alerts: list[dict[str, Any]] = []
        intrusion_zones = [zone for zone in zones if zone.applies_to("intrusion")]
        active_zone_keys = {(camera_id, zone.id) for zone in intrusion_zones}

        for zone in intrusion_zones:
            zone_key = (camera_id, zone.id)
            people_inside = [
                obj
                for obj in objects
                if obj.class_name in self.allowed_classes
                and zone.contains_point(obj.center[0], obj.center[1], frame_shape)
            ]

            if people_inside:
                self._empty_frames[zone_key] = 0
                if zone_key in self._occupied_zones:
                    continue
                self._occupied_zones.add(zone_key)
                actor = max(people_inside, key=lambda obj: obj.confidence)
                alerts.append(
                    {
                        "type": "intrusion",
                        "camera_id": camera_id,
                        "camera_name": camera_name,
                        "track_id": actor.track_id,
                        "class_id": actor.class_id,
                        "class_name": actor.class_name,
                        "zone_id": zone.id,
                        "zone_name": zone.name,
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "details": "Person entered restricted zone",
                    }
                )
                continue

            if zone_key not in self._occupied_zones:
                continue
            self._empty_frames[zone_key] += 1
            if self._empty_frames[zone_key] > self.reset_frames:
                self._occupied_zones.discard(zone_key)
                self._empty_frames.pop(zone_key, None)

        for zone_key in list(self._occupied_zones):
            if zone_key[0] == camera_id and zone_key not in active_zone_keys:
                self._occupied_zones.discard(zone_key)
                self._empty_frames.pop(zone_key, None)

        return alerts
