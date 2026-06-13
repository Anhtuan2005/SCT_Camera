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

    def test_asset_missing_alert_dedupes_when_asset_track_id_changes(self) -> None:
        detector = AssetWatchDetector(
            default_missing_seconds=2,
            settings={
                "person_window_seconds": 12,
                "min_presence_seconds": 1,
                "interaction_distance_ratio": 0.3,
            },
        )
        zone = Zone(
            id="bike-zone",
            name="Bike Zone",
            zone_type="asset_watch",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            threshold_seconds=2,
        )
        first_bicycle = tracked_object(10, 1, "bicycle", (40, 40, 70, 75))
        second_bicycle = tracked_object(11, 1, "bicycle", (42, 42, 72, 77))
        person = tracked_object(20, 0, "person", (60, 25, 85, 80))
        args = ("cam-1", "Camera 1")
        frame_shape = (100, 100, 3)
        timestamp = datetime(2026, 6, 10, 22, 0, 0)

        with patch("analytics.asset_watch.time.monotonic", return_value=0.0):
            detector.analyze(*args, [first_bicycle, person], [zone], frame_shape, timestamp)
        with patch("analytics.asset_watch.time.monotonic", return_value=1.1):
            detector.analyze(*args, [second_bicycle, person], [zone], frame_shape, timestamp)
        with patch("analytics.asset_watch.time.monotonic", return_value=3.2):
            first_alerts = detector.analyze(*args, [], [zone], frame_shape, timestamp)
        with patch("analytics.asset_watch.time.monotonic", return_value=4.5):
            second_alerts = detector.analyze(*args, [], [zone], frame_shape, timestamp)

        self.assertEqual(1, len(first_alerts))
        self.assertEqual([], second_alerts)

    def test_missing_asset_alert_runs_without_configured_zone(self) -> None:
        detector = AssetWatchDetector(
            default_missing_seconds=2,
            settings={
                "person_window_seconds": 12,
                "min_presence_seconds": 1,
                "interaction_distance_ratio": 0.3,
            },
        )
        bicycle = tracked_object(10, 1, "bicycle", (40, 40, 70, 75))
        person = tracked_object(20, 0, "person", (60, 25, 85, 80))
        args = ("cam-1", "Camera 1")
        frame_shape = (100, 100, 3)
        timestamp = datetime(2026, 6, 10, 22, 0, 0)

        with patch("analytics.asset_watch.time.monotonic", return_value=0.0):
            detector.analyze(*args, [bicycle, person], [], frame_shape, timestamp)
        with patch("analytics.asset_watch.time.monotonic", return_value=3.0):
            alerts = detector.analyze(*args, [], [], frame_shape, timestamp)

        self.assertEqual(1, len(alerts))
        self.assertEqual("asset_missing", alerts[0]["type"])
        self.assertEqual("global-asset-watch", alerts[0]["zone_id"])
        self.assertEqual("Full Frame", alerts[0]["zone_name"])


if __name__ == "__main__":
    unittest.main()
