"""Regression tests for asset-watch behavior."""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

from analytics.asset_watch import AssetWatchDetector
from analytics.zone import Zone
from core.tracker import TrackedObject


def tracked_object(
    track_id: int,
    class_id: int,
    class_name: str,
    bbox: tuple[float, float, float, float],
) -> TrackedObject:
    x1, y1, x2, y2 = bbox
    return TrackedObject(
        track_id=track_id,
        bbox_xyxy=bbox,
        class_id=class_id,
        class_name=class_name,
        confidence=0.8,
        center_history=[((x1 + x2) / 2, (y1 + y2) / 2)],
    )


class AssetWatchDetectorTests(unittest.TestCase):
    def test_missing_asset_alert_survives_temporary_person_detection_loss(self) -> None:
        detector = AssetWatchDetector(
            default_missing_seconds=6,
            settings={
                "person_window_seconds": 12,
                "min_presence_seconds": 2,
                "interaction_distance_ratio": 0.3,
            },
        )
        zone = Zone(
            id="bike-zone",
            name="Bike Zone",
            zone_type="asset_watch",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            threshold_seconds=6,
        )
        bicycle = tracked_object(10, 1, "bicycle", (40, 40, 70, 75))
        person = tracked_object(20, 0, "person", (60, 25, 85, 80))
        args = ("cam-1", "Camera 1")
        frame_shape = (100, 100, 3)
        timestamp = datetime(2026, 6, 10, 22, 0, 0)

        with patch("analytics.asset_watch.time.monotonic", return_value=0.0):
            self.assertEqual(
                detector.analyze(*args, [bicycle], [zone], frame_shape, timestamp),
                [],
            )
        with patch("analytics.asset_watch.time.monotonic", return_value=3.0):
            self.assertEqual(
                detector.analyze(*args, [bicycle, person], [zone], frame_shape, timestamp),
                [],
            )
        with patch("analytics.asset_watch.time.monotonic", return_value=8.0):
            self.assertEqual(
                detector.analyze(*args, [bicycle], [zone], frame_shape, timestamp),
                [],
            )
        with patch("analytics.asset_watch.time.monotonic", return_value=14.1):
            alerts = detector.analyze(*args, [], [zone], frame_shape, timestamp)

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["type"], "asset_missing")
        self.assertEqual(alerts[0]["actor_track_id"], person.track_id)
        self.assertEqual(alerts[0]["class_name"], "bicycle")


if __name__ == "__main__":
    unittest.main()
