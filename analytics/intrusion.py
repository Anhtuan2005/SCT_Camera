"""Intrusion detection rule for restricted zones."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from analytics.zone import Zone
from core.tracker import TrackedObject


class IntrusionDetector:
    """Alert once when a tracked object first enters an intrusion zone."""

    def __init__(self, reset_frames: int = 30) -> None:
        self.reset_frames = reset_frames
        self._alerted: dict[tuple[str, str], set[int]] = defaultdict(set)
        self._missing_frames: dict[tuple[str, str, int], int] = defaultdict(int)

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

        for zone in intrusion_zones:
            zone_key = (camera_id, zone.id)
            current_inside: set[int] = set()

            for obj in objects:
                if zone.contains_point(obj.center[0], obj.center[1], frame_shape):
                    current_inside.add(obj.track_id)
                    self._missing_frames[(camera_id, zone.id, obj.track_id)] = 0
                    if obj.track_id not in self._alerted[zone_key]:
                        self._alerted[zone_key].add(obj.track_id)
                        alerts.append(
                            {
                                "type": "intrusion",
                                "camera_id": camera_id,
                                "camera_name": camera_name,
                                "track_id": obj.track_id,
                                "class_id": obj.class_id,
                                "class_name": obj.class_name,
                                "zone_id": zone.id,
                                "zone_name": zone.name,
                                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "details": "Object entered restricted zone",
                            }
                        )

            for track_id in list(self._alerted[zone_key]):
                if track_id in current_inside:
                    continue
                missing_key = (camera_id, zone.id, track_id)
                self._missing_frames[missing_key] += 1
                if self._missing_frames[missing_key] > self.reset_frames:
                    self._alerted[zone_key].discard(track_id)
                    self._missing_frames.pop(missing_key, None)

        return alerts
