"""Suspicious stranger rule for front-yard and doorway monitoring."""

from __future__ import annotations

import time
from datetime import datetime
from math import hypot
from typing import Any

from analytics.dwell_policy import (
    DwellSession,
    current_dwell_duration,
    should_purge_missing_session,
    update_dwell_session,
)
from analytics.identity_status import is_confirmed_stranger
from analytics.zone import Zone
from core.tracker import TrackedObject

_DwellKey = tuple[str, str, int]


class SuspiciousStrangerDetector:
    """Alert when an unknown person remains in a watched zone and behaves suspiciously."""

    def __init__(
        self,
        default_threshold_seconds: float = 180.0,
        settings: dict[str, Any] | None = None,
        state_grace_seconds: float = 3.0,
    ) -> None:
        settings = settings or {}
        self.default_threshold_seconds = default_threshold_seconds
        self.state_grace_seconds = max(
            0.0,
            float(settings.get("state_grace_seconds", state_grace_seconds)),
        )
        self.min_history_points = int(settings.get("min_history_points", 5))
        self.stationary_max_displacement_ratio = float(
            settings.get("stationary_max_displacement_ratio", 0.04)
        )
        self.pacing_path_min_ratio = float(settings.get("pacing_path_min_ratio", 0.18))
        self.pacing_net_max_ratio = float(settings.get("pacing_net_max_ratio", 0.08))
        self._sessions: dict[_DwellKey, DwellSession] = {}
        self._alerted_tier_index: dict[_DwellKey, int] = {}
        self._active_states: dict[_DwellKey, dict[str, Any]] = {}
        self._session_gaps: dict[_DwellKey, float] = {}

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
            inside_now: set[_DwellKey] = set()

            for obj in objects:
                if not self._is_stranger(obj) or self._is_excluded(zone, obj):
                    continue
                key = (camera_id, zone.id, obj.track_id)
                if not zone.contains_point(obj.center[0], obj.center[1], frame_shape):
                    continue

                inside_now.add(key)
                threshold = self._threshold_for(zone, obj, timestamp)
                duration = self._duration_for(key, zone, now)
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
                if reason is None:
                    continue

                tiers = (
                    zone.dwell_policy.effective_tiers(obj.identity_kind, timestamp)
                    if zone.dwell_policy is not None
                    else []
                )
                last_fired = self._alerted_tier_index.get(key, -1)
                if tiers:
                    for index, tier in enumerate(tiers):
                        if index <= last_fired or duration < tier.after_seconds:
                            continue
                        self._alerted_tier_index[key] = index
                        alerts.append(
                            self._alert(
                                camera_id,
                                camera_name,
                                zone,
                                obj,
                                duration,
                                tier.after_seconds,
                                reason,
                                timestamp,
                                alert_type=tier.alert_type,
                                siren=tier.siren,
                                channels=tier.channels,
                            )
                        )
                    continue

                if duration < threshold or last_fired >= 0:
                    continue
                self._alerted_tier_index[key] = 0
                alerts.append(
                    self._alert(
                        camera_id,
                        camera_name,
                        zone,
                        obj,
                        duration,
                        threshold,
                        reason,
                        timestamp,
                        alert_type="suspicious_stranger",
                        siren=True,
                    )
                )

            self._clear_missing(camera_id, zone.id, inside_now, now)

        return alerts

    def get_active_states(self, camera_id: str) -> dict[int, dict[str, Any]]:
        """Return current stranger-watch timer states keyed by track id."""
        now = time.monotonic()
        states: dict[int, dict[str, Any]] = {}
        for key, state in list(self._active_states.items()):
            if key[0] != camera_id:
                continue
            session = self._sessions.get(key)
            if session is None:
                self._active_states.pop(key, None)
                continue
            if not session.active and now - session.last_seen > self.state_grace_seconds:
                continue
            duration = current_dwell_duration(session, now, self.state_grace_seconds)
            threshold = float(
                state.get("threshold_seconds", self.default_threshold_seconds)
            )
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

    def _duration_for(self, key: _DwellKey, zone: Zone, now: float) -> float:
        session_gap = self._session_gap_for(zone)
        self._session_gaps[key] = session_gap
        return update_dwell_session(
            self._sessions,
            key,
            now,
            self.state_grace_seconds,
            session_gap,
        )

    def _clear_missing(
        self,
        camera_id: str,
        zone_id: str,
        inside_now: set[_DwellKey],
        now: float,
    ) -> None:
        for key, session in list(self._sessions.items()):
            if key[:2] != (camera_id, zone_id) or key in inside_now:
                continue
            gap = max(0.0, now - session.last_seen)
            should_purge = should_purge_missing_session(
                session,
                now,
                self.state_grace_seconds,
                self._session_gaps.get(key, 0.0),
            )
            if gap > self.state_grace_seconds:
                self._active_states.pop(key, None)
            if should_purge:
                self._purge_key(key)

    def _clear_camera(self, camera_id: str) -> None:
        for key in [key for key in self._sessions if key[0] == camera_id]:
            self._purge_key(key)

    def _purge_key(self, key: _DwellKey) -> None:
        self._sessions.pop(key, None)
        self._alerted_tier_index.pop(key, None)
        self._active_states.pop(key, None)
        self._session_gaps.pop(key, None)

    def _threshold_for(
        self,
        zone: Zone,
        obj: TrackedObject,
        timestamp: datetime,
    ) -> float:
        if zone.dwell_policy is not None:
            return zone.dwell_policy.effective_threshold(obj.identity_kind, timestamp)
        return zone.threshold_seconds or self.default_threshold_seconds

    @staticmethod
    def _session_gap_for(zone: Zone) -> float:
        if zone.dwell_policy is None:
            return 0.0
        return zone.dwell_policy.session_gap_seconds

    @staticmethod
    def _is_excluded(zone: Zone, obj: TrackedObject) -> bool:
        return (
            zone.dwell_policy is not None
            and zone.dwell_policy.is_excluded(obj.identity_kind)
        )

    @staticmethod
    def _is_stranger(obj: TrackedObject) -> bool:
        return is_confirmed_stranger(obj)

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

    @staticmethod
    def _alert(
        camera_id: str,
        camera_name: str,
        zone: Zone,
        obj: TrackedObject,
        duration: float,
        threshold: float,
        reason: str,
        timestamp: datetime,
        alert_type: str,
        siren: bool = False,
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        label = obj.identity_label or "Stranger"
        alert = {
            "type": alert_type,
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
            "details": (
                f"{label} stayed for {duration:.0f}s "
                f"(threshold: {threshold:.0f}s), reason: {reason}"
            ),
        }
        if siren:
            alert["siren"] = True
        if channels is not None:
            alert["notification_channels"] = channels
        return alert
