"""Regression tests for suspicious theft behavior."""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

from analytics.identity_status import STRANGER_KIND
from analytics.theft_behavior import SuspiciousTheftDetector
from analytics.zone import Zone
from core.tracker import TrackedObject


def tracked_object(
    track_id: int,
    class_id: int,
    class_name: str,
    bbox: tuple[float, float, float, float],
    history: list[tuple[float, float]] | None = None,
) -> TrackedObject:
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    identity_label = "Stranger" if class_name == "person" else None
    identity_kind = STRANGER_KIND if class_name == "person" else None
    return TrackedObject(
        track_id=track_id,
        bbox_xyxy=bbox,
        class_id=class_id,
        class_name=class_name,
        confidence=0.8,
        center_history=history or [center],
        identity_label=identity_label,
        identity_kind=identity_kind,
    )


class SuspiciousTheftDetectorTests(unittest.TestCase):
    def test_theft_alert_runs_without_configured_zone(self) -> None:
        detector = SuspiciousTheftDetector(
            settings={
                "proximity_seconds": 1,
                "vehicle_move_min_ratio": 0.02,
                "score_threshold": 2,
                "require_near_duration": True,
                "require_vehicle_signal": True,
            }
        )
        frame_shape = (100, 100, 3)
        timestamp = datetime(2026, 6, 13, 14, 0, 0)
        args = ("cam-1", "Camera 1")
        person = tracked_object(20, 0, "person", (55, 35, 85, 90))
        bike = tracked_object(10, 1, "bicycle", (35, 40, 65, 70))
        bike_moved = tracked_object(
            10,
            1,
            "bicycle",
            (43, 48, 73, 78),
            history=[(50, 55), (58, 63)],
        )

        with patch("analytics.theft_behavior.time.monotonic", return_value=0.0):
            self.assertEqual(
                [],
                detector.analyze(*args, [person, bike], [], frame_shape, timestamp),
            )
        with patch("analytics.theft_behavior.time.monotonic", return_value=2.0):
            alerts = detector.analyze(
                *args,
                [person, bike_moved],
                [],
                frame_shape,
                timestamp,
            )

        self.assertEqual(1, len(alerts))
        self.assertEqual("suspicious_theft_behavior", alerts[0]["type"])
        self.assertEqual("global-asset-watch", alerts[0]["zone_id"])
        self.assertEqual("Full Frame", alerts[0]["zone_name"])

    def test_theft_alert_dedupes_when_vehicle_track_id_changes(self) -> None:
        detector = SuspiciousTheftDetector(
            settings={
                "proximity_seconds": 1,
                "vehicle_move_min_ratio": 0.02,
                "score_threshold": 2,
                "require_near_duration": True,
                "require_vehicle_signal": True,
            }
        )
        zone = Zone(
            id="bike-zone",
            name="Bike Zone",
            zone_type="asset_watch",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
        frame_shape = (100, 100, 3)
        timestamp = datetime(2026, 6, 13, 14, 0, 0)
        args = ("cam-1", "Camera 1")

        first_person = tracked_object(20, 0, "person", (55, 35, 85, 90))
        first_bike = tracked_object(10, 1, "bicycle", (35, 40, 65, 70))
        first_bike_moved = tracked_object(
            10,
            1,
            "bicycle",
            (43, 48, 73, 78),
            history=[(50, 55), (58, 63)],
        )
        second_person = tracked_object(21, 0, "person", (55, 35, 85, 90))
        second_bike = tracked_object(11, 1, "bicycle", (35, 40, 65, 70))
        second_bike_moved = tracked_object(
            11,
            1,
            "bicycle",
            (43, 48, 73, 78),
            history=[(50, 55), (58, 63)],
        )

        with patch("analytics.theft_behavior.time.monotonic", return_value=0.0):
            self.assertEqual(
                [],
                detector.analyze(*args, [first_person, first_bike], [zone], frame_shape, timestamp),
            )
        with patch("analytics.theft_behavior.time.monotonic", return_value=2.0):
            first_alerts = detector.analyze(
                *args,
                [first_person, first_bike_moved],
                [zone],
                frame_shape,
                timestamp,
            )
        with patch("analytics.theft_behavior.time.monotonic", return_value=4.0):
            detector.analyze(*args, [second_person, second_bike], [zone], frame_shape, timestamp)
        with patch("analytics.theft_behavior.time.monotonic", return_value=6.0):
            second_alerts = detector.analyze(
                *args,
                [second_person, second_bike_moved],
                [zone],
                frame_shape,
                timestamp,
            )

        self.assertEqual(1, len(first_alerts))
        self.assertEqual("suspicious_theft_behavior", first_alerts[0]["type"])
        self.assertEqual([], second_alerts)


if __name__ == "__main__":
    unittest.main()
