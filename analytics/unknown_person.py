"""Full-frame unknown-person alert rule."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from analytics.identity_status import is_confirmed_stranger
from core.tracker import TrackedObject


class UnknownPersonDetector:
    """Alert once when an unknown person appears anywhere in the frame."""

    def __init__(self) -> None:
        self._alerted: set[tuple[str, int]] = set()

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return alerts for newly visible unknown people."""
        strangers = [obj for obj in objects if self._is_stranger(obj)]
        active_keys = {(camera_id, obj.track_id) for obj in strangers}
        self._alerted = {
            key
            for key in self._alerted
            if key[0] != camera_id or key in active_keys
        }

        alerts: list[dict[str, Any]] = []
        for obj in strangers:
            key = (camera_id, obj.track_id)
            if key in self._alerted:
                continue
            self._alerted.add(key)
            label = obj.identity_label or "Stranger"
            alerts.append(
                {
                    "type": "stranger_detected",
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "track_id": obj.track_id,
                    "class_id": obj.class_id,
                    "class_name": obj.class_name,
                    "identity_label": label,
                    "identity_kind": obj.identity_kind or "stranger",
                    "identity_score": obj.identity_score,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "details": f"Unknown person detected: {label} (Track #{obj.track_id})",
                }
            )
        return alerts

    @staticmethod
    def _is_stranger(obj: TrackedObject) -> bool:
        return is_confirmed_stranger(obj)
