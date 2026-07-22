"""Regression tests for pose classification."""

from __future__ import annotations

import unittest
from unittest.mock import patch

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

    def test_keeps_compact_upper_body_only_pose_as_sitting(self) -> None:
        keypoints = _keypoints(
            shoulder_x=20.0,
            shoulder_y=20.0,
            hip_x=33.0,
            hip_y=60.0,
        )
        for index in (13, 14, 15, 16):
            keypoints[index] = (0.0, 0.0, 0.0)

        objects = self.classifier.label_objects(
            "cam",
            [_person((0.0, 0.0, 100.0, 110.0), keypoints=keypoints)],
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

    def test_labels_lying_from_floor_view_compact_straight_pose(self) -> None:
        objects = self.classifier.label_objects(
            "cam",
            [
                _person(
                    (0.0, 0.0, 100.0, 119.0),
                    keypoints=_keypoints(
                        shoulder_x=20.0,
                        shoulder_y=20.0,
                        hip_x=33.0,
                        hip_y=60.0,
                        knee_x=45.0,
                        knee_y=90.0,
                        ankle_x=57.0,
                        ankle_y=118.0,
                    ),
                )
            ],
            self.frame_shape,
        )

        self.assertEqual("lying", objects[0].pose_label)

    def test_labels_lying_from_floor_view_reclined_pose(self) -> None:
        objects = self.classifier.label_objects(
            "cam",
            [
                _person(
                    (0.0, 0.0, 100.0, 115.0),
                    keypoints=_keypoints(
                        shoulder_x=20.0,
                        shoulder_y=20.0,
                        hip_x=40.0,
                        hip_y=60.0,
                        knee_x=55.0,
                        knee_y=85.0,
                        ankle_x=55.0,
                        ankle_y=115.0,
                    ),
                )
            ],
            self.frame_shape,
        )

        self.assertEqual("lying", objects[0].pose_label)

    def test_labels_lying_from_wide_bbox_when_torso_keypoints_are_vertical(self) -> None:
        objects = self.classifier.label_objects(
            "cam",
            [
                _person(
                    (0.0, 20.0, 200.0, 164.0),
                    keypoints=_keypoints(
                        shoulder_x=80.0,
                        shoulder_y=30.0,
                        hip_x=85.0,
                        hip_y=70.0,
                        knee_x=120.0,
                        knee_y=110.0,
                        ankle_x=100.0,
                        ankle_y=150.0,
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

    def test_inserts_named_states_between_posture_families(self) -> None:
        sitting_person = _person(
            (25.0, 0.0, 85.0, 100.0),
            keypoints=_keypoints(shoulder_y=20.0, hip_y=60.0, knee_y=70.0),
        )
        upright_person = _person(
            (40.0, 0.0, 80.0, 100.0),
            keypoints=_keypoints(),
        )

        with patch("analytics.pose_classifier.time.monotonic", return_value=0.0):
            sitting = self.classifier.label_objects(
                "getting-up", [sitting_person], self.frame_shape
            )
        with patch("analytics.pose_classifier.time.monotonic", return_value=0.1):
            getting_up = self.classifier.label_objects(
                "getting-up", [upright_person], self.frame_shape
            )
        with patch("analytics.pose_classifier.time.monotonic", return_value=0.9):
            standing = self.classifier.label_objects(
                "getting-up", [upright_person], self.frame_shape
            )

        with patch("analytics.pose_classifier.time.monotonic", return_value=0.0):
            self.classifier.label_objects(
                "sitting-down", [upright_person], self.frame_shape
            )
        with patch("analytics.pose_classifier.time.monotonic", return_value=0.1):
            sitting_down = self.classifier.label_objects(
                "sitting-down", [sitting_person], self.frame_shape
            )

        lying_person = _person(
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
        with patch("analytics.pose_classifier.time.monotonic", return_value=0.0):
            self.classifier.label_objects(
                "changing-posture", [upright_person], self.frame_shape
            )
        with patch("analytics.pose_classifier.time.monotonic", return_value=0.1):
            changing_to_lying = self.classifier.label_objects(
                "changing-posture", [lying_person], self.frame_shape
            )

        self.assertEqual("sitting", sitting[0].pose_label)
        self.assertEqual("getting_up", getting_up[0].pose_label)
        self.assertEqual("standing_still", standing[0].pose_label)
        self.assertEqual("sitting_down", sitting_down[0].pose_label)
        self.assertEqual("changing_to_lying", changing_to_lying[0].pose_label)

    def test_motion_change_within_upright_family_has_no_posture_transition(self) -> None:
        standing = _person(
            (40.0, 0.0, 80.0, 100.0),
            keypoints=_keypoints(),
            track_id=30,
        )
        walking = _person(
            (40.0, 0.0, 80.0, 100.0),
            history=[(60.0 + i * 8.0, 50.0) for i in range(6)],
            keypoints=_keypoints(),
            track_id=30,
        )

        with patch("analytics.pose_classifier.time.monotonic", return_value=0.0):
            self.classifier.label_objects("cam", [standing], self.frame_shape)
        for now in (0.1, 0.2, 0.3):
            with patch("analytics.pose_classifier.time.monotonic", return_value=now):
                result = self.classifier.label_objects(
                    "cam", [walking], self.frame_shape
                )

        self.assertEqual("walking_slow", result[0].pose_label)


if __name__ == "__main__":
    unittest.main()
