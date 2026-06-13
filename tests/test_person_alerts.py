from datetime import datetime
import unittest
from unittest.mock import patch

from analytics.identity_status import PENDING_PERSON_KIND
from analytics.behavior_engine import BehaviorEngine
from analytics.loitering import LoiteringDetector
from analytics.unknown_person import UnknownPersonDetector
from analytics.zone import Zone
from core.tracker import TrackedObject
from notifications.alert_manager import AlertManager


class RecordingIdentityResolver:
    def __init__(self) -> None:
        self.assume_unknown_persons: list[bool] = []

    def label_objects(
        self,
        _camera_id,
        tracked_objects,
        _frame_bgr,
        assume_unknown_persons=False,
    ):
        self.assume_unknown_persons.append(assume_unknown_persons)
        return tracked_objects


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

    def test_video_sources_auto_label_people_as_strangers(self) -> None:
        resolver = RecordingIdentityResolver()
        engine = BehaviorEngine(
            {
                "behavior_learning": {"enabled": False},
                "identity": {"enabled": False},
            },
            identity_resolver=resolver,
        )

        engine.label_objects(
            [person(1)],
            {
                "camera_id": "video",
                "source": "E:\\SUS.mp4",
                "unknown_person_policy": "face_match",
            },
            object(),
        )

        self.assertEqual([True], resolver.assume_unknown_persons)

    def test_camera_sources_keep_face_recognition_policy(self) -> None:
        resolver = RecordingIdentityResolver()
        engine = BehaviorEngine(
            {
                "behavior_learning": {"enabled": False},
                "identity": {"enabled": False},
            },
            identity_resolver=resolver,
        )

        engine.label_objects(
            [person(1)],
            {
                "camera_id": "rtsp",
                "source": "rtsp://camera/live",
                "unknown_person_policy": "face_match",
            },
            object(),
        )

        self.assertEqual([False], resolver.assume_unknown_persons)

    def test_behavior_engine_adds_default_warning_zones_without_configured_roi(self) -> None:
        zones = BehaviorEngine._load_zones({"zones": []})

        generated = {zone.zone_type: zone for zone in zones if zone.auto_generated}
        self.assertEqual(
            {"intrusion", "loitering", "stranger_watch"},
            set(generated),
        )
        self.assertTrue(all(zone.name == "Full Frame" for zone in generated.values()))
        self.assertNotIn("asset_watch", generated)

    def test_auto_global_zone_can_be_disabled_per_camera(self) -> None:
        zones = BehaviorEngine._load_zones(
            {
                "zones": [],
                "auto_global_zone": False,
            }
        )

        self.assertEqual([], zones)

    def test_behavior_engine_loitering_runs_on_default_full_frame_zone(self) -> None:
        engine = BehaviorEngine(
            {
                "behavior": {"loitering_threshold_seconds": 30},
                "behavior_learning": {"enabled": False},
                "identity": {"enabled": False},
            }
        )
        config = {"camera_id": "cam", "name": "RTSP Camera", "zones": [], "lines": []}
        now = [100.0]

        with patch("analytics.loitering.time.monotonic", side_effect=lambda: now[0]):
            engine.analyze([person(7)], config, (100, 100, 3))
            now[0] = 131.0
            alerts = engine.analyze([person(7)], config, (100, 100, 3))

        self.assertIn("loitering", [alert["type"] for alert in alerts])

    def test_bbox_timer_runs_for_non_known_people_only(self) -> None:
        engine = BehaviorEngine(
            {
                "behavior": {"loitering_threshold_seconds": 30},
                "behavior_learning": {"enabled": False},
                "identity": {"enabled": False},
            }
        )
        config = {"camera_id": "cam", "name": "Camera", "zones": [], "lines": []}
        now = [100.0]

        with patch("analytics.behavior_engine.time.monotonic", side_effect=lambda: now[0]):
            engine.analyze(
                [
                    person(1, identity_kind="stranger"),
                    person(2, identity_kind="known_person"),
                ],
                config,
                (100, 100, 3),
            )
            now[0] = 112.0
            engine.analyze(
                [
                    person(1, identity_kind="stranger"),
                    person(2, identity_kind="known_person"),
                ],
                config,
                (100, 100, 3),
            )

        states = engine.get_person_timer_states("cam")

        self.assertIn(1, states)
        self.assertGreaterEqual(states[1]["duration"], 12.0)
        self.assertNotIn(2, states)

    @patch("analytics.loitering.time.monotonic", side_effect=[100.0, 131.0])
    def test_loitering_requires_roi_zone(self, _monotonic) -> None:
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

        self.assertEqual([], first)
        self.assertEqual({}, states)

    @patch("analytics.loitering.time.monotonic", side_effect=[100.0, 110.0, 131.0])
    def test_loitering_uses_roi_zone_and_exposes_timer(self, _monotonic) -> None:
        detector = LoiteringDetector(default_threshold_seconds=30)
        now = datetime.now()
        zone = Zone(
            id="porch",
            name="Porch",
            zone_type="loitering",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )

        first = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(7)],
            [zone],
            (100, 100, 3),
            now,
        )
        states = detector.get_active_states("cam")
        final = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(7)],
            [zone],
            (100, 100, 3),
            now,
        )

        self.assertEqual([], first)
        self.assertAlmostEqual(10.0, states[7]["duration"])
        self.assertEqual(30.0, states[7]["threshold_seconds"])
        self.assertEqual(1, len(final))
        self.assertEqual("loitering", final[0]["type"])
        self.assertEqual("porch", final[0]["zone_id"])

    def test_behavior_engine_suppresses_stranger_alert_inside_intrusion_zone(self) -> None:
        engine = BehaviorEngine(
            {
                "behavior_learning": {"enabled": False},
                "identity": {"enabled": False},
            }
        )

        alerts = engine.analyze(
            [person(3)],
            {
                "camera_id": "cam",
                "name": "RTSP Camera",
                "zones": [
                    {
                        "id": "home",
                        "name": "Home",
                        "type": "intrusion",
                        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    }
                ],
                "lines": [],
            },
            (100, 100, 3),
        )

        self.assertEqual(["intrusion"], [alert["type"] for alert in alerts])


if __name__ == "__main__":
    unittest.main()
