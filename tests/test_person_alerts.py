import asyncio
from datetime import datetime
import unittest
from unittest.mock import MagicMock, patch

from analytics.identity_status import PENDING_PERSON_KIND
from analytics.behavior_engine import BehaviorEngine
from analytics.loitering import LoiteringDetector
from analytics.suspicious_stranger import SuspiciousStrangerDetector
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


class FakeAlertSender:
    async def send_alert(self, _alert):
        return True


class FakeSiren:
    async def trigger(self, _alert):
        return False


def person(
    track_id: int,
    identity_kind: str = "stranger",
    class_confirmed: bool = True,
) -> TrackedObject:
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
        class_confirmed=class_confirmed,
    )


def stationary_stranger(track_id: int) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox_xyxy=(20.0, 20.0, 80.0, 80.0),
        class_id=0,
        class_name="person",
        confidence=0.9,
        center_history=[(50.0, 50.0)] * 5,
        identity_label="Stranger",
        identity_kind="stranger",
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
        self.assertEqual("__global_stranger_watch__", first[0]["zone_id"])
        self.assertEqual("Full Frame", first[0]["zone_name"])
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

    def test_unconfirmed_person_class_does_not_trigger_stranger_alert(self) -> None:
        detector = UnknownPersonDetector()

        alerts = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(1, class_confirmed=False)],
            datetime.now(),
        )

        self.assertEqual([], alerts)

    def test_full_frame_stranger_alert_cooldown_is_per_camera(self) -> None:
        first = AlertManager._cooldown_key(
            {
                "camera_id": "cam",
                "type": "stranger_detected",
                "track_id": 1,
                "zone_id": "__global_stranger_watch__",
                "zone_name": "Full Frame",
            }
        )
        second = AlertManager._cooldown_key(
            {
                "camera_id": "cam",
                "type": "stranger_detected",
                "track_id": 2,
                "zone_id": "__global_stranger_watch__",
                "zone_name": "Full Frame",
            }
        )

        self.assertEqual(first, second)

    def test_line_intrusion_cooldown_is_per_track(self) -> None:
        first = AlertManager._cooldown_key(
            {
                "camera_id": "cam",
                "type": "intrusion",
                "track_id": 1,
                "line_id": "door",
                "line_name": "Door",
            }
        )
        second = AlertManager._cooldown_key(
            {
                "camera_id": "cam",
                "type": "intrusion",
                "track_id": 2,
                "line_id": "door",
                "line_name": "Door",
            }
        )

        self.assertNotEqual(first, second)

    def test_stranger_detected_uses_dedicated_cooldown_override(self) -> None:
        async def run_case() -> None:
            manager = AlertManager(
                {
                    "telegram": {
                        "cooldown_seconds": 5,
                        "cooldown_overrides": {"stranger_detected": 12},
                    },
                }
            )
            manager.bot = FakeAlertSender()
            manager.discord = FakeAlertSender()
            manager.siren = FakeSiren()

            loop = asyncio.get_running_loop()
            stranger_alert = {
                "camera_id": "cam",
                "camera_name": "Camera",
                "type": "stranger_detected",
                "track_id": 1,
                "zone_id": "__global_stranger_watch__",
                "zone_name": "Full Frame",
                "notification_channels": ["telegram"],
            }
            manager._last_sent_at[manager._cooldown_key(stranger_alert)] = loop.time() - 6

            await manager._handle_alert(stranger_alert)

            intrusion_alert = {
                "camera_id": "cam",
                "camera_name": "Camera",
                "type": "intrusion",
                "track_id": 2,
                "zone_id": "front",
                "zone_name": "Front",
                "notification_channels": ["telegram"],
            }
            manager._last_sent_at[manager._cooldown_key(intrusion_alert)] = loop.time() - 6

            await manager._handle_alert(intrusion_alert)

            recent = manager.get_recent("cam")
            self.assertEqual(1, len(recent))
            self.assertFalse(recent[0]["suppressed"])

        asyncio.run(run_case())

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

    def test_behavior_engine_skips_theft_without_asset_watch_zone(self) -> None:
        engine = BehaviorEngine(
            {
                "behavior_learning": {"enabled": False},
                "identity": {"enabled": False},
            }
        )
        engine.asset_watch.analyze = MagicMock(return_value=[{"type": "asset_missing"}])
        engine.theft_behavior.analyze = MagicMock(return_value=[{"type": "suspicious_theft_behavior"}])

        alerts = engine.analyze(
            [person(3)],
            {"camera_id": "cam", "name": "RTSP Camera", "zones": [], "lines": []},
            (100, 100, 3),
        )

        engine.asset_watch.analyze.assert_not_called()
        engine.theft_behavior.analyze.assert_not_called()
        self.assertNotIn("asset_missing", [alert["type"] for alert in alerts])
        self.assertNotIn("suspicious_theft_behavior", [alert["type"] for alert in alerts])

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
            {"intrusion", "stranger_watch"},
            set(generated),
        )
        self.assertTrue(all(zone.name == "Full Frame" for zone in generated.values()))
        self.assertNotIn("asset_watch", generated)
        self.assertNotIn("loitering", generated)

    def test_auto_global_zone_can_be_disabled_per_camera(self) -> None:
        zones = BehaviorEngine._load_zones(
            {
                "zones": [],
                "auto_global_zone": False,
            }
        )

        self.assertEqual([], zones)

    def test_behavior_engine_loitering_requires_configured_roi(self) -> None:
        engine = BehaviorEngine(
            {
                "behavior": {"loitering_threshold_seconds": 20},
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

        self.assertNotIn("loitering", [alert["type"] for alert in alerts])
        self.assertEqual({}, engine.get_person_timer_states("cam"))

    def test_bbox_timer_runs_only_for_loitering_roi(self) -> None:
        engine = BehaviorEngine(
            {
                "behavior": {"loitering_threshold_seconds": 20},
                "behavior_learning": {"enabled": False},
                "identity": {"enabled": False},
            }
        )
        config = {
            "camera_id": "cam",
            "name": "Camera",
            "zones": [
                {
                    "id": "porch",
                    "name": "Porch",
                    "type": "loitering",
                    "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                }
            ],
            "lines": [],
        }
        now = [100.0]

        with patch("analytics.loitering.time.monotonic", side_effect=lambda: now[0]):
            engine.analyze(
                [
                    person(1, identity_kind="stranger"),
                ],
                config,
                (100, 100, 3),
            )
            now[0] = 112.0
            engine.analyze(
                [
                    person(1, identity_kind="stranger"),
                ],
                config,
                (100, 100, 3),
            )

        states = engine.get_person_timer_states("cam")

        self.assertIn(1, states)
        self.assertGreaterEqual(states[1]["duration"], 12.0)
        self.assertEqual(20.0, states[1]["threshold_seconds"])
        self.assertEqual("porch", states[1]["zone_id"])

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

    @patch("analytics.loitering.time.monotonic", side_effect=[100.0, 101.0, 102.1])
    def test_loitering_timer_survives_short_missing_gap(self, _monotonic) -> None:
        detector = LoiteringDetector(
            default_threshold_seconds=2,
            state_grace_seconds=3,
        )
        now = datetime.now()
        zone = Zone(
            id="porch",
            name="Porch",
            zone_type="loitering",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )

        detector.analyze("cam", "RTSP Camera", [person(7)], [zone], (100, 100, 3), now)
        detector.analyze("cam", "RTSP Camera", [], [zone], (100, 100, 3), now)
        final = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(7)],
            [zone],
            (100, 100, 3),
            now,
        )

        self.assertEqual(1, len(final))
        self.assertEqual("loitering", final[0]["type"])
        self.assertGreaterEqual(final[0]["duration"], 2.0)

    @patch("analytics.suspicious_stranger.time.monotonic", side_effect=[100.0, 101.0, 102.1])
    def test_stranger_watch_timer_survives_short_missing_gap(self, _monotonic) -> None:
        detector = SuspiciousStrangerDetector(
            default_threshold_seconds=2,
            settings={"min_history_points": 3},
            state_grace_seconds=3,
        )
        now = datetime.now()
        zone = Zone(
            id="yard",
            name="Yard",
            zone_type="stranger_watch",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )

        detector.analyze(
            "cam",
            "RTSP Camera",
            [stationary_stranger(9)],
            [zone],
            (100, 100, 3),
            now,
        )
        detector.analyze("cam", "RTSP Camera", [], [zone], (100, 100, 3), now)
        final = detector.analyze(
            "cam",
            "RTSP Camera",
            [stationary_stranger(9)],
            [zone],
            (100, 100, 3),
            now,
        )

        self.assertEqual(1, len(final))
        self.assertEqual("suspicious_stranger", final[0]["type"])
        self.assertGreaterEqual(final[0]["duration"], 2.0)

    def test_zone_dwell_policy_parses_identity_and_time_scaling(self) -> None:
        zone = Zone.from_config(
            {
                "id": "porch",
                "type": "loitering",
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "threshold_seconds": 20,
                "identity_multipliers": {
                    "known_person": None,
                    "stranger": 0.5,
                },
                "time_of_day_multipliers": [
                    {"start": "22:00", "end": "06:00", "multiplier": 0.25}
                ],
            }
        )

        self.assertIsNotNone(zone.dwell_policy)
        assert zone.dwell_policy is not None
        self.assertTrue(zone.dwell_policy.is_excluded("known_person"))
        self.assertEqual(
            2.5,
            zone.dwell_policy.effective_threshold(
                "stranger",
                datetime(2026, 1, 1, 23, 0),
            ),
        )

    @patch("analytics.loitering.time.monotonic", side_effect=[100.0, 106.0])
    def test_loitering_uses_time_of_day_multiplier(self, _monotonic) -> None:
        detector = LoiteringDetector(default_threshold_seconds=20)
        zone = Zone.from_config(
            {
                "id": "porch",
                "name": "Porch",
                "type": "loitering",
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "threshold_seconds": 20,
                "time_of_day_multipliers": [
                    {"start": "22:00", "end": "06:00", "multiplier": 0.25}
                ],
            }
        )
        timestamp = datetime(2026, 1, 1, 23, 0)

        detector.analyze("cam", "RTSP Camera", [person(7)], [zone], (100, 100, 3), timestamp)
        alerts = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(7)],
            [zone],
            (100, 100, 3),
            timestamp,
        )

        self.assertEqual(1, len(alerts))
        self.assertEqual(5.0, alerts[0]["threshold_seconds"])

    @patch("analytics.loitering.time.monotonic", side_effect=[100.0, 103.0])
    def test_loitering_skips_excluded_identity(self, _monotonic) -> None:
        detector = LoiteringDetector(default_threshold_seconds=2)
        zone = Zone.from_config(
            {
                "id": "porch",
                "name": "Porch",
                "type": "loitering",
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "identity_multipliers": {"known_person": None},
            }
        )
        now = datetime.now()

        detector.analyze(
            "cam",
            "RTSP Camera",
            [person(7, identity_kind="known_person")],
            [zone],
            (100, 100, 3),
            now,
        )
        alerts = detector.analyze(
            "cam",
            "RTSP Camera",
            [person(7, identity_kind="known_person")],
            [zone],
            (100, 100, 3),
            now,
        )

        self.assertEqual([], alerts)

    @patch("analytics.loitering.time.monotonic", side_effect=[100.0, 103.0, 105.0])
    def test_loitering_escalation_tiers_fire_once_in_order(self, _monotonic) -> None:
        detector = LoiteringDetector(default_threshold_seconds=20)
        zone = Zone.from_config(
            {
                "id": "porch",
                "name": "Porch",
                "type": "loitering",
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "escalation_tiers": [
                    {
                        "after_seconds": 2,
                        "alert_type": "loitering_notice",
                        "channels": ["telegram"],
                    },
                    {
                        "after_seconds": 4,
                        "alert_type": "loitering_critical",
                        "siren": True,
                        "channels": ["telegram", "discord"],
                    },
                ],
            }
        )
        now = datetime.now()

        detector.analyze("cam", "RTSP Camera", [person(7)], [zone], (100, 100, 3), now)
        first = detector.analyze("cam", "RTSP Camera", [person(7)], [zone], (100, 100, 3), now)
        second = detector.analyze("cam", "RTSP Camera", [person(7)], [zone], (100, 100, 3), now)

        self.assertEqual(["loitering_notice"], [alert["type"] for alert in first])
        self.assertEqual(["loitering_critical"], [alert["type"] for alert in second])
        self.assertEqual(["telegram", "discord"], second[0]["notification_channels"])
        self.assertTrue(second[0]["siren"])

    @patch(
        "analytics.loitering.time.monotonic",
        side_effect=[100.0, 102.0, 106.0, 130.0, 132.0],
    )
    def test_loitering_session_gap_accumulates_return_visits(self, _monotonic) -> None:
        detector = LoiteringDetector(default_threshold_seconds=4, state_grace_seconds=3)
        zone = Zone.from_config(
            {
                "id": "porch",
                "name": "Porch",
                "type": "loitering",
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "threshold_seconds": 4,
                "session_gap_seconds": 60,
            }
        )
        now = datetime.now()

        detector.analyze("cam", "RTSP Camera", [person(7)], [zone], (100, 100, 3), now)
        detector.analyze("cam", "RTSP Camera", [person(7)], [zone], (100, 100, 3), now)
        detector.analyze("cam", "RTSP Camera", [], [zone], (100, 100, 3), now)
        detector.analyze("cam", "RTSP Camera", [person(7)], [zone], (100, 100, 3), now)
        alerts = detector.analyze("cam", "RTSP Camera", [person(7)], [zone], (100, 100, 3), now)

        self.assertEqual(1, len(alerts))
        self.assertEqual("loitering", alerts[0]["type"])
        self.assertGreaterEqual(alerts[0]["duration"], 4.0)

    @patch("analytics.suspicious_stranger.time.monotonic", side_effect=[100.0, 106.0])
    def test_stranger_watch_uses_identity_multiplier(self, _monotonic) -> None:
        detector = SuspiciousStrangerDetector(
            default_threshold_seconds=10,
            settings={"min_history_points": 3},
        )
        zone = Zone.from_config(
            {
                "id": "yard",
                "name": "Yard",
                "type": "stranger_watch",
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "threshold_seconds": 10,
                "identity_multipliers": {"stranger": 0.5},
            }
        )
        now = datetime.now()

        detector.analyze(
            "cam",
            "RTSP Camera",
            [stationary_stranger(9)],
            [zone],
            (100, 100, 3),
            now,
        )
        alerts = detector.analyze(
            "cam",
            "RTSP Camera",
            [stationary_stranger(9)],
            [zone],
            (100, 100, 3),
            now,
        )

        self.assertEqual(1, len(alerts))
        self.assertEqual("suspicious_stranger", alerts[0]["type"])
        self.assertEqual(5.0, alerts[0]["threshold_seconds"])

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

        self.assertEqual([], alerts)


if __name__ == "__main__":
    unittest.main()
