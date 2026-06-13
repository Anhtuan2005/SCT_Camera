import unittest

from core.detector import (
    _parse_class_confidences,
    _passes_class_confidence,
    _predict_confidence_threshold,
    _valid_detection_shape,
)


class DetectorShapeFilterTests(unittest.TestCase):
    def test_rejects_extremely_thin_person_boxes(self) -> None:
        self.assertFalse(
            _valid_detection_shape(
                (460.0, 320.0, 494.0, 720.0),
                "person",
                4.0,
            )
        )

    def test_allows_wide_sitting_or_lying_person_boxes(self) -> None:
        self.assertTrue(
            _valid_detection_shape(
                (610.0, 360.0, 1030.0, 540.0),
                "person",
                4.0,
            )
        )

    def test_does_not_filter_non_person_classes(self) -> None:
        self.assertTrue(
            _valid_detection_shape(
                (460.0, 320.0, 494.0, 720.0),
                "curtain",
                4.0,
            )
        )

    def test_class_confidence_can_lower_dog_threshold_only(self) -> None:
        thresholds = _parse_class_confidences({"dog": 0.12, "person": 0.25})

        self.assertEqual(0.12, _predict_confidence_threshold(0.25, thresholds))
        self.assertTrue(_passes_class_confidence(0.13, "dog", 0.25, thresholds))
        self.assertFalse(_passes_class_confidence(0.13, "person", 0.25, thresholds))
        self.assertFalse(_passes_class_confidence(0.13, "suitcase", 0.25, thresholds))

    def test_class_confidence_can_lower_bicycle_threshold(self) -> None:
        thresholds = _parse_class_confidences({"bicycle": 0.10, "person": 0.10})

        self.assertEqual(0.10, _predict_confidence_threshold(0.25, thresholds))
        self.assertTrue(_passes_class_confidence(0.217, "bicycle", 0.25, thresholds))
        self.assertFalse(_passes_class_confidence(0.09, "bicycle", 0.25, thresholds))

    def test_person_confidence_defaults_to_lower_threshold_only(self) -> None:
        thresholds = _parse_class_confidences({})

        self.assertEqual(0.20, thresholds["person"])
        self.assertEqual(0.20, _predict_confidence_threshold(0.4, thresholds))
        self.assertTrue(_passes_class_confidence(0.21, "person", 0.4, thresholds))
        self.assertFalse(_passes_class_confidence(0.21, "dog", 0.4, thresholds))


if __name__ == "__main__":
    unittest.main()
