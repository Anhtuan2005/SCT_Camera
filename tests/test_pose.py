import unittest

from core.pose import PoseEstimator
from core.tracker import TrackedObject


class PoseMatchingTests(unittest.TestCase):
    def test_best_pose_index_falls_back_to_center_when_iou_is_low(self) -> None:
        estimator = PoseEstimator.__new__(PoseEstimator)
        estimator.match_iou = 0.35
        obj = TrackedObject(
            track_id=1,
            bbox_xyxy=(0.0, 0.0, 100.0, 300.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[],
        )

        matched = estimator._best_pose_index(
            obj,
            [(-20.0, 100.0, 120.0, 200.0)],
            set(),
        )

        self.assertEqual(0, matched)

    def test_best_pose_index_rejects_far_center_fallback(self) -> None:
        estimator = PoseEstimator.__new__(PoseEstimator)
        estimator.match_iou = 0.35
        obj = TrackedObject(
            track_id=1,
            bbox_xyxy=(0.0, 0.0, 100.0, 300.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[],
        )

        matched = estimator._best_pose_index(
            obj,
            [(300.0, 100.0, 440.0, 200.0)],
            set(),
        )

        self.assertIsNone(matched)


if __name__ == "__main__":
    unittest.main()
