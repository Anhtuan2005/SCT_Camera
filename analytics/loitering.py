"""Loitering behavior rule."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from analytics.dwell_policy import (
    DwellSession,
    current_dwell_duration,
    should_purge_missing_session,
    update_dwell_session,
)
from analytics.zone import Zone
from core.tracker import TrackedObject

_DwellKey = tuple[str, str, int]


class LoiteringDetector:
    """Alert when a person remains in a loitering ROI beyond a threshold."""

    def __init__(
        self,
        default_threshold_seconds: float = 20.0,
        state_grace_seconds: float = 3.0,
    ) -> None:
        self.default_threshold_seconds = default_threshold_seconds
        self.state_grace_seconds = max(0.0, float(state_grace_seconds))
        self._sessions: dict[_DwellKey, DwellSession] = {}
        self._alerted_tier_index: dict[_DwellKey, int] = {}
        self._timer_context: dict[_DwellKey, tuple[str, float]] = {}
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
        """Return alerts for people remaining too long inside loitering zones."""
        loitering_zones = [zone for zone in zones if zone.applies_to("loitering")]
        if not loitering_zones:
            self._clear_camera(camera_id)
            return []

        now = time.monotonic()
        alerts: list[dict[str, Any]] = []
        people = [obj for obj in objects if obj.class_name == "person"]
        active_keys: set[_DwellKey] = set()
        for zone in loitering_zones:
            for obj in people:
                if self._is_excluded(zone, obj):
                    continue
                if zone.contains_point(obj.center[0], obj.center[1], frame_shape):
                    key = (camera_id, zone.id, obj.track_id)
                    active_keys.add(key)
        self._clear_missing(camera_id, active_keys, now)

        for zone in loitering_zones:
            people_inside = [
                obj
                for obj in people
                if (camera_id, zone.id, obj.track_id) in active_keys
            ]
            for obj in people_inside:
                key = (camera_id, zone.id, obj.track_id)
                threshold = self._threshold_for(zone, obj, timestamp)
                duration = self._duration_for(key, zone, now)
                self._timer_context[key] = (zone.name, threshold)

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
                        timestamp,
                        alert_type="loitering",
                    )
                )

        return alerts

    def get_active_states(self, camera_id: str) -> dict[int, dict[str, Any]]:
        """Return live loitering-zone timers keyed by track id for drawing."""
        now = time.monotonic()
        states: dict[int, dict[str, Any]] = {}
        for key, session in self._sessions.items():
            if key[0] != camera_id:
                continue
            if not session.active and now - session.last_seen > self.state_grace_seconds:
                continue
            duration = current_dwell_duration(session, now, self.state_grace_seconds)
            zone_name, threshold = self._timer_context.get(
                key,
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

    def _clear_camera(self, camera_id: str) -> None:
        for key in [key for key in self._sessions if key[0] == camera_id]:
            self._purge_key(key)

    def _clear_missing(
        self,
        camera_id: str,
        active_keys: set[_DwellKey],
        now: float,
    ) -> None:
        for key, session in list(self._sessions.items()):
            if key[0] != camera_id or key in active_keys:
                continue
            gap = max(0.0, now - session.last_seen)
            should_purge = should_purge_missing_session(
                session,
                now,
                self.state_grace_seconds,
                self._session_gaps.get(key, 0.0),
            )
            if gap > self.state_grace_seconds:
                self._timer_context.pop(key, None)
            if should_purge:
                self._purge_key(key)

    def _purge_key(self, key: _DwellKey) -> None:
        self._sessions.pop(key, None)
        self._alerted_tier_index.pop(key, None)
        self._timer_context.pop(key, None)
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
    def _alert(
        camera_id: str,
        camera_name: str,
        zone: Zone,
        obj: TrackedObject,
        duration: float,
        threshold: float,
        timestamp: datetime,
        alert_type: str,
        siren: bool = False,
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        alert = {
            "type": alert_type,
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
        if siren:
            alert["siren"] = True
        if channels is not None:
            alert["notification_channels"] = channels
        return alert
