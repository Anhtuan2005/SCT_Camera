"""Loitering behavior rule."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from analytics.zone import Zone
from core.tracker import TrackedObject


class LoiteringDetector:
    """Alert when a person remains in a loitering ROI beyond a threshold."""

    def __init__(self, default_threshold_seconds: float = 20.0) -> None:
        self.default_threshold_seconds = default_threshold_seconds
        self._entry_times: dict[tuple[str, str, int], float] = {}
        self._alerted: set[tuple[str, str, int]] = set()
        self._zone_context: dict[tuple[str, str], tuple[str, float]] = {}

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        zones: list[Zone],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return alerts for people remaining too long inside loitering zones."""
        loitering_zones = [zone for zone in zones if zone.applies_to("loitering")]
        if not loitering_zones:
            self._clear_camera(camera_id)
            return []

        now = time.monotonic()
        alerts: list[dict[str, Any]] = []
        people = [obj for obj in objects if obj.class_name == "person"]
        active_keys: set[tuple[str, str, int]] = set()
        for zone in loitering_zones:
            active_keys.update(
                (camera_id, zone.id, obj.track_id)
                for obj in people
                if zone.contains_point(obj.center[0], obj.center[1], frame_shape)
            )
        self._clear_missing(camera_id, active_keys)

        for zone in loitering_zones:
            threshold = zone.threshold_seconds or self.default_threshold_seconds
            self._zone_context[(camera_id, zone.id)] = (zone.name, threshold)
            people_inside = [
                obj
                for obj in people
                if (camera_id, zone.id, obj.track_id) in active_keys
            ]
            for obj in people_inside:
                key = (camera_id, zone.id, obj.track_id)
                self._entry_times.setdefault(key, now)
                duration = now - self._entry_times[key]
                if duration < threshold or key in self._alerted:
                    continue
                self._alerted.add(key)
                alerts.append(
                    {
                        "type": "loitering",
                        "camera_id": camera_id,
                        "camera_name": camera_name,
                        "track_id": obj.track_id,
                        "class_id": obj.class_id,
                        "class_name": obj.class_name,
                        "identity_label": obj.identity_label,
                        "identity_kind": obj.identity_kind,
                        "zone_id": zone.id,
                        "zone_name": zone.name,
                        "duration": round(duration, 1),
                        "threshold_seconds": threshold,
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "details": (
                            f"Person remained in {zone.name} for {duration:.0f} seconds "
                            f"(threshold: {threshold:.0f}s)"
                        ),
                    }
                )

        return alerts

    def get_active_states(self, camera_id: str) -> dict[int, dict[str, Any]]:
        """Return live loitering-zone timers keyed by track id for drawing."""
        now = time.monotonic()
        states: dict[int, dict[str, Any]] = {}
        for key, entry_time in self._entry_times.items():
            if key[0] != camera_id:
                continue
            duration = now - entry_time
            zone_name, threshold = self._zone_context.get(
                (camera_id, key[1]),
                (key[1], self.default_threshold_seconds),
            )
            state = {
                "camera_id": camera_id,
                "zone_id": key[1],
                "zone_name": zone_name,
                "track_id": key[2],
                "duration": duration,
                "threshold_seconds": threshold,
                "remaining_seconds": max(0.0, threshold - duration),
                "alert_ready": duration >= threshold,
            }
            previous = states.get(key[2])
            if previous is None or (
                state["alert_ready"],
                state["duration"],
            ) > (
                bool(previous.get("alert_ready", False)),
                float(previous.get("duration", 0.0)),
            ):
                states[key[2]] = state
        return states

    def _clear_camera(self, camera_id: str) -> None:
        stale_keys = [key for key in self._entry_times if key[0] == camera_id]
        for key in stale_keys:
            self._entry_times.pop(key, None)
            self._alerted.discard(key)
        self._zone_context = {
            key: value
            for key, value in self._zone_context.items()
            if key[0] != camera_id
        }

    def _clear_missing(
        self,
        camera_id: str,
        active_keys: set[tuple[str, str, int]],
    ) -> None:
        stale_keys = [
            key
            for key in self._entry_times
            if key[0] == camera_id and key not in active_keys
        ]
        for key in stale_keys:
            self._entry_times.pop(key, None)
            self._alerted.discard(key)
