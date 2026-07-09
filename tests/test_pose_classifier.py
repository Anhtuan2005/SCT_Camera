"""Regression tests for pose classification."""

from __future__ import annotations

import unittest

from analytics.pose_classifier import PoseClassifier
from core.tracker import TrackedObject


def _keypoints(
    shoulder_x: float = 50.0,
    shoulder_y: float = 20.0,
    hip_x: float = 50.0,
    hip_y: float = 60.0,
    knee_x: float = 50.0,
    knee_y: float = 120.0,
    ankle_x: float = 50.0,
    ankle_y: float = 160.0,
    confidence: float = 0.9,
) -> list[tuple[float, float, float]]:
    points = [(0.0, 0.0, 0.0)] * 17
    for index in (5, 6):
        points[index] = (shoulder_x, shoulder_y, confidence)
    for index in (11, 12):
        points[index] = (hip_x, hip_y, confidence)
    for index in (13, 14):
        points[index] = (knee_x, knee_y, confidence)
    for index in (15, 16):
        points[index] = (ankle_x, ankle_y, confidence)
    return points


def _person(
    bbox: tuple[float, float, float, float],
    history: list[tuple[float, float]] | None = None,
    keypoints: list[tuple[float, float, float]] | None = None,
    track_id: int = 1,
) -> TrackedObject:
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    return TrackedObject(
        track_id=track_id,
        bbox_xyxy=bbox,
        class_id=0,
        class_name="person",
        confidence=0.9,
        center_history=history or [center],
        pose_keypoints=keypoints,
    )


class PoseClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = PoseClassifier(
            {
                "pose_classification": {
                    "smoothing_window": 1,
                    "motion_window": 4,
                    "motion_switch_confirm_frames": 1,
                    "motion_smoothing_window": 1,
                    "speed_hysteresis_margin": 0.0,
                    "speed_noise_floor": 0.0,
                }
            }
        )
        self.frame_shape = (100, 100, 3)

    def test_labels_standing_still_from_upright_static_track(self) -> None:
        objects = self.classifier.label_objects(
            "cam",
            [_person((40.0, 0.0, 80.0, 100.0), keypoints=_keypoints())],
            self.frame_shape,
        )

        self.assertEqual("standing_still", objects[0].pose_label)

    def test_labels_walking_and_running_from_motion_speed(self) -> None:
        # Walking: moderate displacement across several frames
        walking_history = [(60.0 + i * 8.0, 50.0) for i in range(6)]
        walking = self.classifier.label_objects(
            "cam",
            [
                _person(
                    (40.0, 0.0, 80.0, 100.0),
                    history=walking_history,
                    keypoints=_keypoints(),
                    track_id=10,
                )
            ],
            self.frame_shape,
        )
        # Running: large displacement across several frames
        running_history = [(60.0 + i * 25.0, 50.0) for i in range(6)]
        running = self.classifier.label_objects(
            "cam",
            [
                _person(
                    (40.0, 0.0, 80.0, 100.0),
                    history=running_history,
                    keypoints=_keypoints(),
                    track_id=20,
                )
            ],
            self.frame_shape,
        )

        self.assertEqual("walking_slow", walking[0].pose_label)
        self.assertEqual("running", running[0].pose_label)

    def test_labels_sitting_from_knee_hip_ratio(self) -> None:
        objects = self.classifier.label_objects(
            "cam",
            [
                _person(
                    (25.0, 0.0, 85.0, 100.0),
                    keypoints=_keypoints(shoulder_y=20.0, hip_y=60.0, knee_y=70.0),
                )
            ],
            self.frame_shape,
        )

        self.assertEqual("sitting", objects[0].pose_label)

    def test_labels_lying_from_body_axis_angle(self) -> None:
        objects = self.classifier.label_objects(
            "cam",
            [
                _person(
                    (0.0, 20.0, 160.0, 100.0),
                    keypoints=_keypoints(
                        shoulder_x=20.0,
                        shoulder_y=50.0,
                        hip_x=100.0,
                        hip_y=60.0,
                        knee_x=130.0,
                        knee_y=70.0,
                    ),
                )
            ],
            self.frame_shape,
        )

        self.assertEqual("lying", objects[0].pose_label)

    def test_labels_lying_from_relaxed_angle_when_ankle_near_hip(self) -> None:
        objects = self.classifier.label_objects(
            "cam",
            [
                _person(
                    (0.0, 20.0, 160.0, 100.0),
                    keypoints=_keypoints(
                        shoulder_x=50.0,
                        shoulder_y=20.0,
                        hip_x=80.0,
                        hip_y=60.0,
                        ankle_x=120.0,
                        ankle_y=80.0,
                    ),
                )
            ],
            self.frame_shape,
        )

        self.assertEqual("lying", objects[0].pose_label)

    def test_labels_unknown_without_required_keypoints(self) -> None:
        objects = self.classifier.label_objects(
            "cam",
            [_person((40.0, 0.0, 80.0, 100.0), keypoints=None)],
            self.frame_shape,
        )

        self.assertEqual("unknown", objects[0].pose_label)


if __name__ == "__main__":
    unittest.main()
