from datetime import datetime
import unittest

from analytics.intrusion import CountingLine, IntrusionDetector
from core.tracker import TrackedObject


def tracked_object(
    track_id: int,
    bbox_xyxy: tuple[float, float, float, float],
    center_history: list[tuple[float, float]],
    identity_kind: str | None = "stranger",
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox_xyxy=bbox_xyxy,
        class_id=0,
        class_name="person",
        confidence=0.9,
        center_history=center_history,
        identity_label=(
            "Stranger"
            if identity_kind == "stranger"
            else "Identifying"
            if identity_kind == "pending_person"
            else "Known person"
        ),
        identity_kind=identity_kind,
    )


class IntrusionLineCrossingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.line = CountingLine(
            id="door",
            name="Door",
            point1=(0.0, 0.5),
            point2=(1.0, 0.5),
        )
        self.frame_shape = (100, 100, 3)
        self.now = datetime.now()

    def test_counts_out_crossing_without_alert(self) -> None:
        counter = IntrusionDetector()
        counter.analyze(
            "cam",
            "Camera",
            [tracked_object(1, (40.0, 55.0, 60.0, 85.0), [(50.0, 70.0), (50.0, 70.0)])],
            [],
            [self.line],
            self.frame_shape,
            self.now,
        )

        alerts = counter.analyze(
            "cam",
            "Camera",
            [tracked_object(1, (40.0, 45.0, 60.0, 75.0), [(50.0, 70.0), (50.0, 60.0)])],
            [],
            [self.line],
            self.frame_shape,
            self.now,
        )

        self.assertEqual([], alerts)
        self.assertEqual({"door": {"in": 0, "out": 1}}, counter.get_counters("cam"))

    def test_touching_line_does_not_repeat_until_track_leaves_line(self) -> None:
        counter = IntrusionDetector()
        counter.analyze(
            "cam",
            "Camera",
            [tracked_object(1, (40.0, 55.0, 60.0, 85.0), [(50.0, 70.0), (50.0, 70.0)])],
            [],
            [self.line],
            self.frame_shape,
            self.now,
        )
        first = counter.analyze(
            "cam",
            "Camera",
            [tracked_object(1, (40.0, 45.0, 60.0, 75.0), [(50.0, 70.0), (50.0, 60.0)])],
            [],
            [self.line],
            self.frame_shape,
            self.now,
        )
        repeated = counter.analyze(
            "cam",
            "Camera",
            [tracked_object(1, (40.0, 44.0, 60.0, 74.0), [(50.0, 60.0), (50.0, 59.0)])],
            [],
            [self.line],
            self.frame_shape,
            self.now,
        )

        self.assertEqual([], first)
        self.assertEqual([], repeated)
        self.assertEqual({"door": {"in": 0, "out": 1}}, counter.get_counters("cam"))

    def test_in_crossing_emits_intrusion_alert_immediately(self) -> None:
        counter = IntrusionDetector()

        alerts = counter.analyze(
            "cam",
            "Camera",
            [tracked_object(2, (40.0, 50.0, 60.0, 70.0), [(50.0, 40.0), (50.0, 60.0)])],
            [],
            [self.line],
            self.frame_shape,
            self.now,
        )

        self.assertEqual(1, len(alerts))
        self.assertEqual("intrusion", alerts[0]["type"])
        self.assertEqual("door", alerts[0]["line_id"])
        self.assertEqual("IN", alerts[0]["direction"])
        self.assertEqual({"door": {"in": 1, "out": 0}}, counter.get_counters("cam"))

    def test_known_person_in_crossing_also_emits_intrusion_alert(self) -> None:
        counter = IntrusionDetector()

        alerts = counter.analyze(
            "cam",
            "Camera",
            [
                tracked_object(
                    3,
                    (40.0, 50.0, 60.0, 70.0),
                    [(50.0, 40.0), (50.0, 60.0)],
                    identity_kind="known_person",
                )
            ],
            [],
            [self.line],
            self.frame_shape,
            self.now,
        )

        self.assertEqual(1, len(alerts))
        self.assertEqual("intrusion", alerts[0]["type"])
        self.assertEqual({"door": {"in": 1, "out": 0}}, counter.get_counters("cam"))

    def test_pending_in_crossing_alerts_immediately(self) -> None:
        counter = IntrusionDetector()

        alerts = counter.analyze(
            "cam",
            "Camera",
            [
                tracked_object(
                    4,
                    (40.0, 50.0, 60.0, 70.0),
                    [(50.0, 40.0), (50.0, 60.0)],
                    identity_kind="pending_person",
                )
            ],
            [],
            [self.line],
            self.frame_shape,
            self.now,
        )

        self.assertEqual(1, len(alerts))
        self.assertEqual("intrusion", alerts[0]["type"])
        self.assertEqual("IN", alerts[0]["direction"])


if __name__ == "__main__":
    unittest.main()
