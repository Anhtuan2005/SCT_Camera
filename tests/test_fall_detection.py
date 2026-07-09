"""Regression tests for fall and faint detection."""

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

from analytics.fall_detection import FallDetector
from core.tracker import TrackedObject


def _person(track_id: int = 1, pose_label: str = "standing_still") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox_xyxy=(20.0, 20.0, 80.0, 120.0),
        class_id=0,
        class_name="person",
        confidence=0.9,
        center_history=[(50.0, 70.0)],
        identity_label="Stranger",
        identity_kind="stranger",
        pose_label=pose_label,
    )


class FallDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = FallDetector(
            {
                "fall_detection": {
                    "max_transition_seconds": 1.5,
                    "lying_alert_seconds": 8,
                }
            }
        )
        self.timestamp = datetime(2026, 7, 5, 12, 0, 0)

    def test_alerts_after_sudden_transition_to_sustained_lying(self) -> None:
        person = _person()

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=0.5):
            self.detector.analyze(
                "cam",
                "Camera",
                [replace(person, pose_label="lying")],
                self.timestamp,
            )
        with patch("analytics.fall_detection.time.monotonic", return_value=8.6):
            alerts = self.detector.analyze(
                "cam",
                "Camera",
                [replace(person, pose_label="lying")],
                self.timestamp,
            )

        self.assertEqual(1, len(alerts))
        self.assertEqual("possible_fall", alerts[0]["type"])
        self.assertTrue(alerts[0]["siren"])
        self.assertTrue(alerts[0]["sudden_drop"])

    def test_existing_lying_track_does_not_alert(self) -> None:
        lying = _person(pose_label="lying")

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=10.0):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual([], alerts)

    def test_slow_sitting_to_lying_does_not_alert(self) -> None:
        sitting = _person(pose_label="sitting")
        lying = replace(sitting, pose_label="lying")

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [sitting], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=2.0):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=11.0):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual([], alerts)

    def test_alerts_once_per_track_presence(self) -> None:
        person = _person()
        lying = replace(person, pose_label="lying")

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=0.2):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=9.0):
            first = self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=20.0):
            second = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual(1, len(first))
        self.assertEqual([], second)


if __name__ == "__main__":
    unittest.main()
