"""Unknown-person alert rule for candidate tracks."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from analytics.identity_status import is_confirmed_stranger
from core.tracker import TrackedObject

FULL_FRAME_ZONE_ID = "__global_stranger_watch__"
FULL_FRAME_ZONE_NAME = "Full Frame"


class UnknownPersonDetector:
    """Alert once per continuous stranger presence on each camera."""

    def __init__(self, absence_grace_seconds: float = 2.0) -> None:
        self.absence_grace_seconds = max(0.0, float(absence_grace_seconds))
        self._active_cameras: set[str] = set()
        self._last_seen_at: dict[str, float] = {}

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return alerts for newly visible unknown people."""
        now = time.monotonic()
        strangers = [obj for obj in objects if self._is_stranger(obj)]
        if not strangers:
            last_seen_at = self._last_seen_at.get(camera_id)
            if (
                last_seen_at is None
                or now - last_seen_at >= self.absence_grace_seconds
            ):
                self.reset_camera(camera_id)
            return []
        self._last_seen_at[camera_id] = now
        if camera_id in self._active_cameras:
            return []

        self._active_cameras.add(camera_id)
        obj = strangers[0]
        label = obj.identity_label or "Stranger"
        return [
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
                "zone_id": FULL_FRAME_ZONE_ID,
                "zone_name": FULL_FRAME_ZONE_NAME,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "details": f"Unknown person detected: {label} (Track #{obj.track_id})",
            }
        ]

    def reset_camera(self, camera_id: str) -> None:
        """Forget continuous-presence state for one camera."""
        self._active_cameras.discard(camera_id)
        self._last_seen_at.pop(camera_id, None)

    @staticmethod
    def _is_stranger(obj: TrackedObject) -> bool:
        return is_confirmed_stranger(obj)
