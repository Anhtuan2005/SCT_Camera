import unittest

import numpy as np

from analytics.identity_status import PENDING_PERSON_KIND, STRANGER_KIND
from analytics.person_identity import PersonIdentityResolver, _Reference
from core.tracker import TrackedObject


class _FakeFaceAnalyzer:
    def __init__(self, faces):
        self.faces = faces
        self.calls = 0

    def get(self, _image):
        self.calls += 1
        return self.faces


class _ShapeAwareFaceAnalyzer:
    def __init__(self, faces_by_shape):
        self.faces_by_shape = faces_by_shape
        self.calls = []

    def get(self, image):
        self.calls.append(image.shape[:2])
        return self.faces_by_shape.get(image.shape[:2], [])


def _person(track_id: int, bbox=(0.0, 0.0, 100.0, 200.0)) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox_xyxy=bbox,
        class_id=0,
        class_name="person",
        confidence=0.9,
        center_history=[(50.0, 100.0)],
    )


def _resolver(face_embedding, threshold=0.45):
    resolver = PersonIdentityResolver(
        {
            "identity": {
                "enabled": True,
                "known_persons": [],
                "similarity_threshold": threshold,
                "recognition_interval_frames": 5,
                "unknown_confirmation_attempts": 1,
            }
        }
    )
    analyzer = _FakeFaceAnalyzer(
        [
            {
                "bbox": np.array([20.0, 10.0, 60.0, 50.0]),
                "det_score": 0.95,
                "normed_embedding": np.asarray(face_embedding, dtype=np.float32),
            }
        ]
    )
    resolver._face_app = analyzer
    resolver._ready = True
    resolver.references = [
        _Reference(
            name="Alice",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
    ]
    return resolver, analyzer


class PersonIdentityTests(unittest.TestCase):
    def test_matches_insightface_embedding_inside_person_box(self) -> None:
        resolver, _analyzer = _resolver([0.98, 0.1])

        labeled = resolver.label_objects(
            "cam",
            [_person(7)],
            np.zeros((240, 160, 3), dtype=np.uint8),
        )

        self.assertEqual("Alice", labeled[0].identity_label)
        self.assertEqual("known_person", labeled[0].identity_kind)
        self.assertGreater(labeled[0].identity_score, 0.9)

    def test_rejects_embedding_below_similarity_threshold(self) -> None:
        resolver, _analyzer = _resolver([0.0, 1.0])

        labeled = resolver.label_objects(
            "cam",
            [_person(8)],
            np.zeros((240, 160, 3), dtype=np.uint8),
        )

        self.assertEqual("Stranger", labeled[0].identity_label)
        self.assertEqual("stranger", labeled[0].identity_kind)
        self.assertIsNone(labeled[0].identity_score)

    def test_does_not_assign_face_outside_person_box(self) -> None:
        resolver, _analyzer = _resolver([1.0, 0.0])
        resolver._face_app = _ShapeAwareFaceAnalyzer(
            {
                (240, 160): [
                    {
                        "bbox": np.array([20.0, 10.0, 60.0, 50.0]),
                        "det_score": 0.95,
                        "normed_embedding": np.array([1.0, 0.0], dtype=np.float32),
                    }
                ]
            }
        )

        labeled = resolver.label_objects(
            "cam",
            [_person(9, bbox=(80.0, 80.0, 150.0, 150.0))],
            np.zeros((240, 160, 3), dtype=np.uint8),
        )

        self.assertEqual("stranger", labeled[0].identity_kind)

    def test_known_track_uses_cache_without_repeated_inference(self) -> None:
        resolver, analyzer = _resolver([1.0, 0.0])
        frame = np.zeros((240, 160, 3), dtype=np.uint8)

        resolver.label_objects("cam", [_person(10)], frame)
        second = resolver.label_objects("cam", [_person(10)], frame)

        self.assertEqual(1, analyzer.calls)
        self.assertEqual("Alice", second[0].identity_label)

    def test_rotated_face_bbox_maps_back_inside_person_box(self) -> None:
        resolver = PersonIdentityResolver(
            {
                "identity": {
                    "enabled": True,
                    "known_persons": [],
                    "similarity_threshold": 0.45,
                    "recognition_interval_frames": 5,
                }
            }
        )
        analyzer = _ShapeAwareFaceAnalyzer(
            {
                (160, 240): [
                    {
                        "bbox": np.array([20.0, 10.0, 60.0, 50.0]),
                        "det_score": 0.95,
                        "normed_embedding": np.array([1.0, 0.0], dtype=np.float32),
                    }
                ]
            }
        )
        resolver._face_app = analyzer
        resolver._ready = True
        resolver.references = [
            _Reference(
                name="Alice",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]

        labeled = resolver.label_objects(
            "cam",
            [_person(11, bbox=(8.0, 178.0, 55.0, 225.0))],
            np.zeros((240, 160, 3), dtype=np.uint8),
        )

        self.assertEqual("Alice", labeled[0].identity_label)
        self.assertEqual([(240, 160), (160, 240)], analyzer.calls)

    def test_keeps_searching_rotations_when_first_faces_do_not_match(self) -> None:
        resolver = PersonIdentityResolver(
            {
                "identity": {
                    "enabled": True,
                    "known_persons": [],
                    "similarity_threshold": 0.45,
                    "recognition_interval_frames": 5,
                    "unknown_confirmation_attempts": 1,
                }
            }
        )
        analyzer = _ShapeAwareFaceAnalyzer(
            {
                (240, 160): [
                    {
                        "bbox": np.array([20.0, 10.0, 60.0, 50.0]),
                        "det_score": 0.95,
                        "normed_embedding": np.array([0.0, 1.0], dtype=np.float32),
                    }
                ],
                (160, 240): [
                    {
                        "bbox": np.array([20.0, 10.0, 60.0, 50.0]),
                        "det_score": 0.95,
                        "normed_embedding": np.array([1.0, 0.0], dtype=np.float32),
                    }
                ],
            }
        )
        resolver._face_app = analyzer
        resolver._ready = True
        resolver.references = [
            _Reference(
                name="Alice",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]

        labeled = resolver.label_objects(
            "cam",
            [_person(12, bbox=(8.0, 178.0, 55.0, 225.0))],
            np.zeros((240, 160, 3), dtype=np.uint8),
        )

        self.assertEqual("Alice", labeled[0].identity_label)
        self.assertEqual([(240, 160), (160, 240)], analyzer.calls)

    def test_unmatched_person_stays_pending_before_confirmed_stranger(self) -> None:
        resolver = PersonIdentityResolver(
            {
                "identity": {
                    "enabled": True,
                    "known_persons": [],
                    "recognition_interval_frames": 1,
                    "unknown_confirmation_attempts": 2,
                }
            }
        )
        resolver._face_app = _FakeFaceAnalyzer([])
        resolver._ready = True
        resolver.references = [
            _Reference(
                name="Alice",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]
        frame = np.zeros((240, 160, 3), dtype=np.uint8)

        first = resolver.label_objects("cam", [_person(13)], frame)
        second = resolver.label_objects("cam", [_person(13)], frame)

        self.assertEqual(PENDING_PERSON_KIND, first[0].identity_kind)
        self.assertEqual("Identifying", first[0].identity_label)
        self.assertEqual(STRANGER_KIND, second[0].identity_kind)
        self.assertEqual("Stranger", second[0].identity_label)

    def test_pending_person_retries_without_waiting_recognition_interval(self) -> None:
        resolver = PersonIdentityResolver(
            {
                "identity": {
                    "enabled": True,
                    "known_persons": [],
                    "recognition_interval_frames": 99,
                    "unknown_confirmation_attempts": 2,
                    "orientations_per_attempt": 1,
                }
            }
        )
        analyzer = _FakeFaceAnalyzer([])
        resolver._face_app = analyzer
        resolver._ready = True
        resolver.references = [
            _Reference(
                name="Alice",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]
        frame = np.zeros((240, 160, 3), dtype=np.uint8)

        first = resolver.label_objects("cam", [_person(14)], frame)
        second = resolver.label_objects("cam", [_person(14)], frame)

        self.assertEqual(PENDING_PERSON_KIND, first[0].identity_kind)
        self.assertEqual(STRANGER_KIND, second[0].identity_kind)
        self.assertEqual(2, analyzer.calls)


if __name__ == "__main__":
    unittest.main()
