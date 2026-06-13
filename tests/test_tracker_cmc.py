"""Regression tests for ByteTrack camera motion compensation integration."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from core.detector import Detection
from core.tracker import ByteTrackTracker, _SafeGMC


class _FakeDetector:
    confidence = 0.25
    class_ids = [0, 16]
    iou = 0.5
    imgsz = 640

    def __init__(self) -> None:
        self.detections: list[Detection] = []

    def detect(self, _frame: np.ndarray) -> list[Detection]:
        return self.detections

    @staticmethod
    def class_name(class_id: int) -> str:
        return {0: "person", 16: "dog"}.get(class_id, str(class_id))


class ByteTrackCameraMotionTests(unittest.TestCase):
    def test_cmc_failure_falls_back_to_identity_transform(self) -> None:
        gmc = _SafeGMC("sparseOptFlow", 2)
        gmc._gmc.apply = MagicMock(side_effect=RuntimeError("no features"))
        gmc._gmc.reset_params = MagicMock()

        transform = gmc.apply(np.zeros((100, 100, 3), dtype=np.uint8))

        np.testing.assert_array_equal(
            transform,
            np.eye(2, 3, dtype=np.float32),
        )
        gmc._gmc.reset_params.assert_called_once()

    def test_cmc_is_attached_to_bytetrack_when_enabled(self) -> None:
        tracker = ByteTrackTracker(
            _FakeDetector(),
            {
                "tracking": {
                    "tracker": "bytetrack.yaml",
                    "camera_motion_compensation": {
                        "enabled": True,
                        "method": "sparseOptFlow",
                        "downscale": 2,
                    },
                }
            },
        )

        self.assertTrue(tracker.cmc_enabled)
        self.assertTrue(hasattr(tracker._tracker, "gmc"))

    def test_tracker_threshold_overrides_are_applied(self) -> None:
        tracker = ByteTrackTracker(
            _FakeDetector(),
            {
                "tracking": {
                    "tracker": "bytetrack.yaml",
                    "track_high_thresh": 0.12,
                    "track_low_thresh": 0.05,
                    "new_track_thresh": 0.12,
                }
            },
        )

        self.assertEqual(0.12, tracker._tracker.args.track_high_thresh)
        self.assertEqual(0.05, tracker._tracker.args.track_low_thresh)
        self.assertEqual(0.12, tracker._tracker.args.new_track_thresh)

    def test_tracker_threshold_overrides_update_existing_args(self) -> None:
        tracker = ByteTrackTracker(
            _FakeDetector(),
            {"tracking": {"tracker": "bytetrack.yaml"}},
        )

        tracker.update_settings(
            {
                "tracking": {
                    "track_high_thresh": 0.12,
                    "track_low_thresh": 0.05,
                    "new_track_thresh": 0.12,
                }
            }
        )

        self.assertEqual(0.12, tracker._tracker.args.track_high_thresh)
        self.assertEqual(0.05, tracker._tracker.args.track_low_thresh)
        self.assertEqual(0.12, tracker._tracker.args.new_track_thresh)

    def test_tracker_output_is_converted_to_tracked_object(self) -> None:
        detector = _FakeDetector()
        detector.detections = [
            Detection(
                bbox_xyxy=(10.0, 20.0, 30.0, 60.0),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )
        ]
        tracker = ByteTrackTracker(
            detector,
            {"tracking": {"tracker": "bytetrack.yaml"}},
        )
        tracker._tracker = MagicMock()
        tracker._tracker.update.return_value = np.asarray(
            [[10.0, 20.0, 30.0, 60.0, 7.0, 0.9, 0.0, 0.0]],
            dtype=np.float32,
        )

        objects = tracker.track(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].track_id, 7)
        self.assertEqual(objects[0].class_name, "person")
        self.assertEqual(objects[0].center_history, [(20.0, 40.0)])
        tracker._tracker.update.assert_called_once()

    def test_track_ids_are_unique_when_bytetrack_reuses_id_across_classes(self) -> None:
        tracker = ByteTrackTracker(
            _FakeDetector(),
            {"tracking": {"tracker": "bytetrack.yaml"}},
        )
        tracker._tracker = MagicMock()
        tracker._tracker.update.return_value = np.asarray(
            [
                [10.0, 20.0, 40.0, 80.0, 8.0, 0.9, 0.0, 0.0],
                [50.0, 60.0, 90.0, 110.0, 8.0, 0.8, 16.0, 0.0],
            ],
            dtype=np.float32,
        )

        objects = tracker.track(np.zeros((120, 120, 3), dtype=np.uint8))

        self.assertEqual(2, len(objects))
        self.assertEqual({"person", "dog"}, {obj.class_name for obj in objects})
        self.assertEqual(2, len({obj.track_id for obj in objects}))

    def test_nested_same_class_boxes_are_deduped_even_when_iou_is_low(self) -> None:
        tracker = ByteTrackTracker(
            _FakeDetector(),
            {"tracking": {"tracker": "bytetrack.yaml"}},
        )
        tracker._tracker = MagicMock()
        tracker._tracker.update.return_value = np.asarray(
            [
                [0.0, 0.0, 100.0, 100.0, 1.0, 0.75, 0.0, 0.0],
                [10.0, 10.0, 60.0, 60.0, 2.0, 0.95, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        objects = tracker.track(np.zeros((120, 120, 3), dtype=np.uint8))

        self.assertEqual(1, len(objects))

    def test_stale_track_expires_after_grace_frames(self) -> None:
        tracker = ByteTrackTracker(
            _FakeDetector(),
            {"tracking": {"tracker": "bytetrack.yaml", "track_grace_frames": 1}},
        )
        tracker._tracker = MagicMock()
        tracker._tracker.update.return_value = np.asarray(
            [[10.0, 20.0, 30.0, 60.0, 7.0, 0.9, 0.0, 0.0]],
            dtype=np.float32,
        )
        tracker.track(np.zeros((100, 100, 3), dtype=np.uint8))

        tracker._tracker.update.return_value = np.empty((0, 8), dtype=np.float32)

        self.assertEqual(1, len(tracker.track(np.zeros((100, 100, 3), dtype=np.uint8))))
        self.assertEqual([], tracker.track(np.zeros((100, 100, 3), dtype=np.uint8)))

    def test_reset_clears_local_history_and_gmc_state(self) -> None:
        tracker = ByteTrackTracker(
            _FakeDetector(),
            {
                "tracking": {
                    "tracker": "bytetrack.yaml",
                    "camera_motion_compensation": {"enabled": True},
                }
            },
        )
        tracker._history[3].append((10.0, 10.0))
        tracker._missing_frames[3] = 1
        tracker._tracker.reset = MagicMock()
        tracker._tracker.gmc.reset_params = MagicMock()

        tracker.reset()

        self.assertEqual(dict(tracker._history), {})
        self.assertEqual(tracker._missing_frames, {})
        tracker._tracker.reset.assert_called_once()
        tracker._tracker.gmc.reset_params.assert_called_once()


if __name__ == "__main__":
    unittest.main()
