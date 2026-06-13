"""Loitering behavior rule."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from analytics.zone import Zone
from core.tracker import TrackedObject


class LoiteringDetector:
    """Alert when a person remains visible beyond a threshold."""

    def __init__(self, default_threshold_seconds: float = 30.0) -> None:
        self.default_threshold_seconds = default_threshold_seconds
        self._entry_times: dict[tuple[str, int], float] = {}
        self._alerted: set[tuple[str, int]] = set()

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        zones: list[Zone],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return alerts for people visible anywhere in the frame too long."""
        now = time.monotonic()
        alerts: list[dict[str, Any]] = []
        people = [obj for obj in objects if obj.class_name == "person"]
        active_keys = {(camera_id, obj.track_id) for obj in people}
        self._clear_missing(camera_id, active_keys)

        for obj in people:
            key = (camera_id, obj.track_id)
            self._entry_times.setdefault(key, now)
            duration = now - self._entry_times[key]
            if duration < self.default_threshold_seconds or key in self._alerted:
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
                    "duration": round(duration, 1),
                    "threshold_seconds": self.default_threshold_seconds,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "details": (
                        f"Person remained visible for {duration:.0f} seconds "
                        f"(threshold: {self.default_threshold_seconds:.0f}s)"
                    ),
                }
            )

        return alerts

    def get_active_states(self, camera_id: str) -> dict[int, dict[str, Any]]:
        """Return live full-frame loitering timers keyed by track id."""
        now = time.monotonic()
        states: dict[int, dict[str, Any]] = {}
        for key, entry_time in self._entry_times.items():
            if key[0] != camera_id:
                continue
            duration = now - entry_time
            states[key[1]] = {
                "camera_id": camera_id,
                "track_id": key[1],
                "duration": duration,
                "threshold_seconds": self.default_threshold_seconds,
                "remaining_seconds": max(0.0, self.default_threshold_seconds - duration),
                "alert_ready": duration >= self.default_threshold_seconds,
            }
        return states

    def _clear_missing(
        self,
        camera_id: str,
        active_keys: set[tuple[str, int]],
    ) -> None:
        stale_keys = [
            key
            for key in self._entry_times
            if key[0] == camera_id and key not in active_keys
        ]
        for key in stale_keys:
            self._entry_times.pop(key, None)
            self._alerted.discard(key)
