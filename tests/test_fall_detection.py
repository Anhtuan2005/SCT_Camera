"""Regression tests for fall and possible-unresponsive detection."""

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

from analytics.behavior_learning import BehaviorLearningService
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


def _lying(person: TrackedObject) -> TrackedObject:
    """Return the same track after a visible downward transition."""
    return replace(
        person,
        bbox_xyxy=(10.0, 70.0, 110.0, 130.0),
        center_history=[*person.center_history, (60.0, 100.0)],
        pose_label="lying",
    )


class FallDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = FallDetector(
            {
                "fall_detection": {
                    "min_upright_seconds": 1,
                    "max_transition_seconds": 3,
                    "min_vertical_drop_ratio": 0.2,
                    "lying_alert_seconds": 10,
                    "pose_grace_seconds": 1,
                    "escalation_seconds": 45,
                    "urgent_reminder_seconds": 60,
                    "recovery_confirm_seconds": 3,
                    "emergency_number": "115",
                }
            }
        )
        self.timestamp = datetime(2026, 7, 5, 12, 0, 0)

    def _arm_and_fall(self, person: TrackedObject | None = None) -> TrackedObject:
        person = person or _person()
        lying = _lying(person)
        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.2):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        return lying

    def _keep_lying_until(
        self,
        lying: TrackedObject,
        start: float,
        stop: float,
        step: float = 0.8,
    ) -> None:
        now = start
        while now < stop:
            with patch("analytics.fall_detection.time.monotonic", return_value=now):
                self.detector.analyze("cam", "Camera", [lying], self.timestamp)
            now += step

    def test_alerts_after_sudden_transition_to_sustained_lying(self) -> None:
        lying = self._arm_and_fall()
        self._keep_lying_until(lying, 2.0, 11.3)

        with patch("analytics.fall_detection.time.monotonic", return_value=11.3):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual(1, len(alerts))
        self.assertEqual("possible_fall", alerts[0]["type"])
        self.assertEqual("critical", alerts[0]["severity"])
        self.assertTrue(alerts[0]["safety_critical"])
        self.assertTrue(alerts[0]["siren"])
        self.assertIn("115", alerts[0]["recommended_action"])
        self.assertAlmostEqual(0.3, alerts[0]["vertical_drop_ratio"])

    def test_existing_lying_track_does_not_alert(self) -> None:
        lying = _lying(_person())

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=20.0):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual([], alerts)

    def test_sitting_to_lying_does_not_alert(self) -> None:
        sitting = _person(pose_label="sitting")
        lying = _lying(sitting)

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [sitting], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=0.5):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=20.0):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual([], alerts)

    def test_short_sitting_transition_still_alerts(self) -> None:
        person = _person()
        sitting = replace(person, pose_label="sitting")
        lying = _lying(person)

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=2.0):
            self.detector.analyze("cam", "Camera", [sitting], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=3.9):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        self._keep_lying_until(lying, 4.0, 13.9)
        with patch("analytics.fall_detection.time.monotonic", return_value=13.9):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual("possible_fall", alerts[0]["type"])
        self.assertEqual(2.9, alerts[0]["transition_seconds"])

    def test_long_sitting_transition_does_not_alert(self) -> None:
        person = _person()
        sitting = replace(person, pose_label="sitting")
        lying = _lying(person)

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=2.0):
            self.detector.analyze("cam", "Camera", [sitting], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=4.1):
            self.detector.analyze("cam", "Camera", [sitting], self.timestamp)
        self._keep_lying_until(lying, 4.2, 15.0)
        with patch("analytics.fall_detection.time.monotonic", return_value=15.0):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual([], alerts)

    def test_slow_transition_to_lying_does_not_alert(self) -> None:
        person = _person()
        lying = _lying(person)

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=4.1):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=20.0):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual([], alerts)

    def test_transition_without_visible_drop_does_not_alert(self) -> None:
        person = _person()
        lying_without_drop = replace(person, pose_label="lying")

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.2):
            self.detector.analyze("cam", "Camera", [lying_without_drop], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=12.0):
            alerts = self.detector.analyze(
                "cam", "Camera", [lying_without_drop], self.timestamp
            )

        self.assertEqual([], alerts)

    def test_pose_smoothing_does_not_overwrite_last_upright_geometry(self) -> None:
        person = _person()
        lying = _lying(person)
        smoothed_upright = replace(lying, pose_label="standing_still")

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.1):
            self.detector.analyze(
                "cam", "Camera", [smoothed_upright], self.timestamp
            )
        with patch("analytics.fall_detection.time.monotonic", return_value=1.3):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        self._keep_lying_until(lying, 2.0, 11.4)
        with patch("analytics.fall_detection.time.monotonic", return_value=11.4):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual("possible_fall", alerts[0]["type"])

    def test_changing_to_lying_starts_fall_timer_without_display_delay(self) -> None:
        person = _person()
        changing = replace(_lying(person), pose_label="changing_to_lying")
        lying = _lying(person)

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.3):
            self.detector.analyze("cam", "Camera", [changing], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.8):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        self._keep_lying_until(lying, 2.0, 11.3)
        with patch("analytics.fall_detection.time.monotonic", return_value=11.3):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual("possible_fall", alerts[0]["type"])
        self.assertEqual(0.3, alerts[0]["transition_seconds"])

    def test_walking_label_during_collapse_preserves_upright_geometry(self) -> None:
        person = _person()
        collapsed_walking = replace(
            person,
            bbox_xyxy=(20.0, 60.0, 80.0, 120.0),
            center_history=[*person.center_history, (50.0, 90.0)],
            pose_label="walking_slow",
        )
        lying = _lying(person)

        with patch("analytics.fall_detection.time.monotonic", return_value=0.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.0):
            self.detector.analyze("cam", "Camera", [person], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=1.5):
            self.detector.analyze(
                "cam", "Camera", [collapsed_walking], self.timestamp
            )
        with patch("analytics.fall_detection.time.monotonic", return_value=1.8):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        self._keep_lying_until(lying, 2.0, 11.9)
        with patch("analytics.fall_detection.time.monotonic", return_value=11.9):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual(1, len(alerts))
        self.assertEqual("possible_fall", alerts[0]["type"])
        self.assertEqual("walking_slow", alerts[0]["pre_lying_label"])

    def test_recovery_before_ten_seconds_cancels_candidate(self) -> None:
        lying = self._arm_and_fall()
        standing = replace(lying, pose_label="standing_still")

        with patch("analytics.fall_detection.time.monotonic", return_value=5.0):
            self.detector.analyze("cam", "Camera", [standing], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=12.0):
            alerts = self.detector.analyze("cam", "Camera", [standing], self.timestamp)

        self.assertEqual([], alerts)

    def test_brief_sitting_after_lying_does_not_cancel_candidate(self) -> None:
        lying = self._arm_and_fall()
        sitting = replace(lying, pose_label="sitting")

        self._keep_lying_until(lying, 2.0, 4.1)
        with patch("analytics.fall_detection.time.monotonic", return_value=4.1):
            self.detector.analyze("cam", "Camera", [sitting], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=6.0):
            self.detector.analyze("cam", "Camera", [sitting], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=6.1):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        self._keep_lying_until(lying, 6.2, 11.3)
        with patch("analytics.fall_detection.time.monotonic", return_value=11.3):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual("possible_fall", alerts[0]["type"])

    def test_long_unknown_pose_gap_cancels_candidate(self) -> None:
        lying = self._arm_and_fall()
        unknown = replace(lying, pose_label="unknown")

        with patch("analytics.fall_detection.time.monotonic", return_value=2.0):
            self.detector.analyze("cam", "Camera", [unknown], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=2.3):
            self.detector.analyze("cam", "Camera", [unknown], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=12.0):
            alerts = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual([], alerts)

    def test_escalates_and_repeats_urgent_alert_while_person_remains_down(self) -> None:
        lying = self._arm_and_fall()
        self._keep_lying_until(lying, 2.0, 11.3)
        with patch("analytics.fall_detection.time.monotonic", return_value=11.3):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        settled = replace(lying, center_history=[(60.0, 100.0)] * 8)
        self._keep_lying_until(settled, 12.0, 46.3)
        with patch("analytics.fall_detection.time.monotonic", return_value=46.3):
            urgent = self.detector.analyze("cam", "Camera", [settled], self.timestamp)
        self._keep_lying_until(settled, 47.0, 106.4)
        with patch("analytics.fall_detection.time.monotonic", return_value=106.4):
            reminder = self.detector.analyze("cam", "Camera", [settled], self.timestamp)

        self.assertEqual("possible_unresponsive", urgent[0]["type"])
        self.assertEqual("urgent", urgent[0]["alert_stage"])
        self.assertEqual(
            "KHẨN CẤP: NGƯỜI NGÃ VẪN NẰM BẤT ĐỘNG", urgent[0]["title"]
        )
        self.assertIn("tình trạng y tế khẩn cấp", urgent[0]["details"])
        self.assertEqual("possible_unresponsive", reminder[0]["type"])
        self.assertEqual("reminder", reminder[0]["alert_stage"])
        self.assertEqual(
            "NHẮC LẠI: NGƯỜI NGÃ VẪN NẰM BẤT ĐỘNG", reminder[0]["title"]
        )

    def test_sends_recovery_after_stable_upright_pose(self) -> None:
        lying = self._arm_and_fall()
        self._keep_lying_until(lying, 2.0, 11.3)
        with patch("analytics.fall_detection.time.monotonic", return_value=11.3):
            self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        sitting = replace(lying, pose_label="sitting")
        with patch("analytics.fall_detection.time.monotonic", return_value=12.0):
            first = self.detector.analyze("cam", "Camera", [sitting], self.timestamp)
        with patch("analytics.fall_detection.time.monotonic", return_value=15.1):
            recovered = self.detector.analyze("cam", "Camera", [sitting], self.timestamp)

        self.assertEqual([], first)
        self.assertEqual("fall_recovery", recovered[0]["type"])
        self.assertEqual("info", recovered[0]["severity"])
        self.assertFalse(recovered[0]["siren"])

    def test_alerts_once_per_stage_before_reminder(self) -> None:
        lying = self._arm_and_fall()
        self._keep_lying_until(lying, 2.0, 11.3)
        with patch("analytics.fall_detection.time.monotonic", return_value=11.3):
            first = self.detector.analyze("cam", "Camera", [lying], self.timestamp)
        self._keep_lying_until(lying, 12.0, 20.0)
        with patch("analytics.fall_detection.time.monotonic", return_value=20.0):
            second = self.detector.analyze("cam", "Camera", [lying], self.timestamp)

        self.assertEqual(1, len(first))
        self.assertEqual([], second)

    def test_safety_critical_alert_bypasses_behavior_model_gate(self) -> None:
        service = BehaviorLearningService(
            {
                "behavior_learning": {
                    "enabled": True,
                    "log_candidates": False,
                    "gate_alerts": True,
                    "min_risk_score": 0.9,
                    "model_path": "missing-fall-test-model.npz",
                }
            }
        )
        alert = {
            "type": "possible_fall",
            "camera_id": "cam",
            "track_id": 1,
            "class_name": "person",
            "safety_critical": True,
        }

        with patch.object(service, "_score", return_value=0.1):
            result = service.enrich_alerts(
                [alert], [_person()], {"camera_id": "cam"}, (100, 100, 3), self.timestamp
            )

        self.assertEqual([alert], result)
        self.assertNotIn("behavior_suppressed", alert)


if __name__ == "__main__":
    unittest.main()
