"""Fall/faint detection based on sudden pose transitions to lying."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.tracker import TrackedObject

_UPRIGHT_LABELS = {"standing_still", "walking_slow", "running", "sitting"}


@dataclass
class _FallCandidate:
    pre_lying_label: str
    upright_since: float
    became_lying_at: float | None = None


class FallDetector:
    """Detect a sudden collapse to lying and sustained inactivity afterward."""

    def __init__(self, settings: dict[str, Any]) -> None:
        cfg = settings.get("fall_detection", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.max_transition_seconds = float(cfg.get("max_transition_seconds", 1.5))
        self.lying_alert_seconds = float(cfg.get("lying_alert_seconds", 8))
        self._states: dict[tuple[str, int], _FallCandidate] = {}
        self._alerted: set[tuple[str, int]] = set()

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        tracked_objects: list[TrackedObject],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        active_keys = {
            (camera_id, obj.track_id)
            for obj in tracked_objects
            if obj.class_name == "person"
        }
        for key in [key for key in self._states if key[0] == camera_id and key not in active_keys]:
            self._states.pop(key, None)
        self._alerted = {
            key for key in self._alerted if key[0] != camera_id or key in active_keys
        }

        now = time.monotonic()
        alerts: list[dict[str, Any]] = []
        for obj in tracked_objects:
            if obj.class_name != "person" or obj.pose_label is None:
                continue
            key = (camera_id, obj.track_id)

            if obj.pose_label in _UPRIGHT_LABELS:
                self._states[key] = _FallCandidate(
                    pre_lying_label=obj.pose_label,
                    upright_since=now,
                )
                continue

            if obj.pose_label != "lying":
                continue
            state = self._states.get(key)
            if state is None:
                continue

            if state.became_lying_at is None:
                state.became_lying_at = now

            transition_seconds = state.became_lying_at - state.upright_since
            sudden_drop = transition_seconds <= self.max_transition_seconds
            skipped_sitting = state.pre_lying_label != "sitting"
            sustained = (now - state.became_lying_at) >= self.lying_alert_seconds

            if sustained and (sudden_drop or skipped_sitting) and key not in self._alerted:
                self._alerted.add(key)
                alerts.append(
                    {
                        "type": "possible_fall",
                        "camera_id": camera_id,
                        "camera_name": camera_name,
                        "track_id": obj.track_id,
                        "class_name": obj.class_name,
                        "identity_label": obj.identity_label or "Unknown",
                        "transition_seconds": round(transition_seconds, 2),
                        "sudden_drop": sudden_drop,
                        "skipped_sitting": skipped_sitting,
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "siren": True,
                        "details": (
                            f"Possible fall: lying for {now - state.became_lying_at:.0f}s "
                            f"after {transition_seconds:.1f}s transition"
                        ),
                    }
                )
        return alerts
