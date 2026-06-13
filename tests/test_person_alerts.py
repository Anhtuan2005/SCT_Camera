from datetime import datetime
import unittest
from unittest.mock import patch

from analytics.identity_status import PENDING_PERSON_KIND
from analytics.behavior_engine import BehaviorEngine
from analytics.loitering import LoiteringDetector
from analytics.unknown_person import UnknownPersonDetector
from core.tracker import TrackedObject
from notifications.alert_manager import AlertManager


def person(track_id: int, identity_kind: str = "stranger") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox_xyxy=(20.0, 20.0, 80.0, 80.0),
        class_id=0,
        class_name="person",
        confidence=0.9,
        center_history=[(50.0, 50.0)],
        identity_label=(
            "Identifying"
            if identity_kind == PENDING_PERSON_KIND
            else "Stranger"
            if identity_kind == "stranger"
            else "Known person"
        ),
        identity_kind=identity_kind,
    )


class PersonAlertTests(unittest.TestCase):
    def test_stranger_alerts_without_roi_once_per_presence(self) -> None:
        detector = UnknownPersonDetector()
        now = datetime.now()

        first = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(1)],
            now,
        )
        repeated = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(1)],
            now,
        )

        self.assertEqual(1, len(first))
        self.assertEqual("stranger_detected", first[0]["type"])
        self.assertEqual([], repeated)

        detector.analyze("cam", "RTSP Camera", [], now)
        returned = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(1)],
            now,
        )
        self.assertEqual(1, len(returned))

    def test_known_person_does_not_trigger_stranger_alert(self) -> None:
        detector = UnknownPersonDetector()

        alerts = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(1, identity_kind="known_person")],
            datetime.now(),
        )

        self.assertEqual([], alerts)

    def test_pending_identity_does_not_trigger_stranger_alert(self) -> None:
        detector = UnknownPersonDetector()

        alerts = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(1, identity_kind=PENDING_PERSON_KIND)],
            datetime.now(),
        )

        self.assertEqual([], alerts)

    def test_full_frame_alert_cooldown_is_per_track(self) -> None:
        first = AlertManager._cooldown_key(
            {"camera_id": "cam", "type": "stranger_detected", "track_id": 1}
        )
        second = AlertManager._cooldown_key(
            {"camera_id": "cam", "type": "stranger_detected", "track_id": 2}
        )

        self.assertNotEqual(first, second)

    def test_behavior_engine_emits_stranger_alert_without_zones(self) -> None:
        engine = BehaviorEngine(
            {
                "behavior_learning": {"enabled": False},
                "identity": {"enabled": False},
            }
        )

        alerts = engine.analyze(
            [person(3)],
            {"camera_id": "cam", "name": "RTSP Camera", "zones": [], "lines": []},
            (100, 100, 3),
        )

        self.assertIn("stranger_detected", [alert["type"] for alert in alerts])

    @patch("analytics.loitering.time.monotonic", side_effect=[100.0, 110.0, 131.0])
    def test_loitering_uses_full_screen_and_exposes_timer(self, _monotonic) -> None:
        detector = LoiteringDetector(default_threshold_seconds=30)
        now = datetime.now()

        first = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(7)],
            [],
            (100, 100, 3),
            now,
        )
        states = detector.get_active_states("cam")
        final = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(7)],
            [],
            (100, 100, 3),
            now,
        )

        self.assertEqual([], first)
        self.assertAlmostEqual(10.0, states[7]["duration"])
        self.assertEqual(30.0, states[7]["threshold_seconds"])
        self.assertEqual(1, len(final))
        self.assertEqual("loitering", final[0]["type"])
        self.assertNotIn("zone_id", final[0])


if __name__ == "__main__":
    unittest.main()
