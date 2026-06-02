"""Suspicious stranger rule for front-yard and doorway monitoring."""

from __future__ import annotations

import time
from datetime import datetime
from math import hypot
from typing import Any

from analytics.zone import Zone
from core.tracker import TrackedObject


class SuspiciousStrangerDetector:
    """Alert when an unknown person remains in a watched zone and behaves suspiciously."""

    def __init__(
        self,
        default_threshold_seconds: float = 180.0,
        settings: dict[str, Any] | None = None,
    ) -> None:
        settings = settings or {}
        self.default_threshold_seconds = default_threshold_seconds
        self.min_history_points = int(settings.get("min_history_points", 5))
        self.stationary_max_displacement_ratio = float(
            settings.get("stationary_max_displacement_ratio", 0.04)
        )
        self.pacing_path_min_ratio = float(settings.get("pacing_path_min_ratio", 0.18))
        self.pacing_net_max_ratio = float(settings.get("pacing_net_max_ratio", 0.08))
        self._entry_times: dict[tuple[str, str, int], float] = {}
        self._alerted: set[tuple[str, str, int]] = set()
        self._active_states: dict[tuple[str, str, int], dict[str, Any]] = {}

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        zones: list[Zone],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return suspicious-stranger alerts for watched zones."""
        watch_zones = [zone for zone in zones if zone.applies_to("stranger_watch")]
        if not watch_zones:
            self._clear_camera(camera_id)
            return []

        now = time.monotonic()
        alerts: list[dict[str, Any]] = []
        for zone in watch_zones:
            threshold = zone.threshold_seconds or self.default_threshold_seconds
            inside_now: set[tuple[str, str, int]] = set()

            for obj in objects:
                if not self._is_stranger(obj):
                    continue
                key = (camera_id, zone.id, obj.track_id)
                if not zone.contains_point(obj.center[0], obj.center[1], frame_shape):
                    continue

                inside_now.add(key)
                self._entry_times.setdefault(key, now)
                duration = now - self._entry_times[key]
                reason = self._suspicious_reason(obj, frame_shape)
                self._active_states[key] = {
                    "camera_id": camera_id,
                    "track_id": obj.track_id,
                    "zone_id": zone.id,
                    "zone_name": zone.name,
                    "duration": duration,
                    "threshold_seconds": threshold,
                    "remaining_seconds": max(0.0, threshold - duration),
                    "suspicious_reason": reason,
                    "alert_ready": duration >= threshold and reason is not None,
                }
                if duration >= threshold and reason and key not in self._alerted:
                    self._alerted.add(key)
                    label = obj.identity_label or "Stranger"
                    alerts.append(
                        {
                            "type": "suspicious_stranger",
                            "camera_id": camera_id,
                            "camera_name": camera_name,
                            "track_id": obj.track_id,
                            "class_id": obj.class_id,
                            "class_name": obj.class_name,
                            "identity_label": label,
                            "identity_kind": obj.identity_kind or "stranger",
                            "identity_score": obj.identity_score,
                            "zone_id": zone.id,
                            "zone_name": zone.name,
                            "duration": round(duration, 1),
                            "threshold_seconds": threshold,
                            "suspicious_reason": reason,
                            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "siren": True,
                            "details": (
                                f"{label} stayed for {duration:.0f}s "
                                f"(threshold: {threshold:.0f}s), reason: {reason}"
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
                self._active_states.pop(key, None)
                self._alerted.discard(key)

        return alerts

    def get_active_states(self, camera_id: str) -> dict[int, dict[str, Any]]:
        """Return current stranger-watch timer states keyed by track id."""
        now = time.monotonic()
        states: dict[int, dict[str, Any]] = {}
        for key, state in list(self._active_states.items()):
            if key[0] != camera_id:
                continue
            entry_time = self._entry_times.get(key)
            if entry_time is None:
                self._active_states.pop(key, None)
                continue
            duration = now - entry_time
            threshold = float(state.get("threshold_seconds", self.default_threshold_seconds))
            current = {
                **state,
                "duration": duration,
                "remaining_seconds": max(0.0, threshold - duration),
                "alert_ready": (
                    duration >= threshold
                    and state.get("suspicious_reason") is not None
                ),
            }
            track_id = int(state["track_id"])
            previous = states.get(track_id)
            if previous is None or current["duration"] > previous["duration"]:
                states[track_id] = current
        return states

    def _clear_camera(self, camera_id: str) -> None:
        stale_keys = [key for key in self._active_states if key[0] == camera_id]
        for key in stale_keys:
            self._active_states.pop(key, None)
            self._entry_times.pop(key, None)
            self._alerted.discard(key)

    @staticmethod
    def _is_stranger(obj: TrackedObject) -> bool:
        return obj.class_name == "person" and obj.identity_kind != "known_person"

    def _suspicious_reason(
        self,
        obj: TrackedObject,
        frame_shape: tuple[int, int, int],
    ) -> str | None:
        points = obj.center_history
        if len(points) < self.min_history_points:
            return None

        height, width = frame_shape[:2]
        diagonal = max(hypot(width, height), 1.0)
        displacement = max(hypot(x - points[0][0], y - points[0][1]) for x, y in points)
        path_length = sum(
            hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
            for index in range(1, len(points))
        )
        net_distance = hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])

        if displacement <= diagonal * self.stationary_max_displacement_ratio:
            return "standing_still"
        if (
            path_length >= diagonal * self.pacing_path_min_ratio
            and net_distance <= diagonal * self.pacing_net_max_ratio
        ):
            return "pacing_near_area"
        return None
