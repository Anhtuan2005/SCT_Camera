"""Loitering behavior rule."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from analytics.zone import Zone
from core.tracker import TrackedObject


class LoiteringDetector:
    """Alert when an object remains inside a zone beyond a threshold."""

    def __init__(self, default_threshold_seconds: float = 30.0) -> None:
        self.default_threshold_seconds = default_threshold_seconds
        self._entry_times: dict[tuple[str, str, int], float] = {}
        self._alerted: set[tuple[str, str, int]] = set()

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        zones: list[Zone],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return loitering alert payloads for objects exceeding dwell time."""
        now = time.monotonic()
        alerts: list[dict[str, Any]] = []
        loitering_zones = [zone for zone in zones if zone.applies_to("loitering")]

        for zone in loitering_zones:
            threshold = zone.threshold_seconds or self.default_threshold_seconds
            inside_now: set[tuple[str, str, int]] = set()

            for obj in objects:
                key = (camera_id, zone.id, obj.track_id)
                if zone.contains_point(obj.center[0], obj.center[1], frame_shape):
                    inside_now.add(key)
                    self._entry_times.setdefault(key, now)
                    duration = now - self._entry_times[key]
                    if duration >= threshold and key not in self._alerted:
                        self._alerted.add(key)
                        alerts.append(
                            {
                                "type": "loitering",
                                "camera_id": camera_id,
                                "camera_name": camera_name,
                                "track_id": obj.track_id,
                                "class_id": obj.class_id,
                                "class_name": obj.class_name,
                                "zone_id": zone.id,
                                "zone_name": zone.name,
                                "duration": round(duration, 1),
                                "threshold_seconds": threshold,
                                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "details": (
                                    f"Lingered for {duration:.0f} seconds "
                                    f"(threshold: {threshold:.0f}s)"
                                ),
                            }
                        )

            stale_keys = [
                key
                for key in self._entry_times
                if key[0] == camera_id and key[1] == zone.id and key not in inside_now
            ]
            for key in stale_keys:
                self._entry_times.pop(key, None)
                self._alerted.discard(key)

        return alerts
