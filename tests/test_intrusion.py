from datetime import datetime
import unittest

from analytics.intrusion import IntrusionDetector
from analytics.zone import Zone
from core.tracker import TrackedObject


def tracked_object(track_id: int, class_name: str, class_id: int) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox_xyxy=(20.0, 20.0, 80.0, 80.0),
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        center_history=[(50.0, 50.0)],
    )


class IntrusionDetectorTest(unittest.TestCase):
    def test_alerts_once_per_person_occupancy_episode(self) -> None:
        detector = IntrusionDetector(reset_frames=2)
        zone = Zone(
            id="room",
            name="Room",
            zone_type="all",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
        frame_shape = (100, 100, 3)
        now = datetime.now()

        first_alerts = detector.analyze(
            "cam",
            "Camera",
            [
                tracked_object(1, "person", 0),
                tracked_object(2, "bicycle", 1),
            ],
            [zone],
            frame_shape,
            now,
        )
        self.assertEqual(["person"], [alert["class_name"] for alert in first_alerts])

        replacement_track_alerts = detector.analyze(
            "cam",
            "Camera",
            [tracked_object(3, "person", 0)],
            [zone],
            frame_shape,
            now,
        )
        self.assertEqual([], replacement_track_alerts)

        for _ in range(3):
            detector.analyze("cam", "Camera", [], [zone], frame_shape, now)

        reentry_alerts = detector.analyze(
            "cam",
            "Camera",
            [tracked_object(4, "person", 0)],
            [zone],
            frame_shape,
            now,
        )
        self.assertEqual(1, len(reentry_alerts))


if __name__ == "__main__":
    unittest.main()
