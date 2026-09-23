import unittest

from scripts.evaluate_mot_tracking import _processed_bbox_to_source


class MotEvaluationMappingTests(unittest.TestCase):
    def test_maps_cw90_processed_box_back_to_source_coordinates(self) -> None:
        source_box = _processed_bbox_to_source(
            (50.0, 100.0, 150.0, 300.0),
            "cw90",
            0.5,
            source_width=1000,
            source_height=600,
        )

        self.assertEqual((200.0, 300.0, 600.0, 500.0), source_box)

    def test_clips_box_to_source_frame(self) -> None:
        source_box = _processed_bbox_to_source(
            (-10.0, -20.0, 120.0, 80.0),
            "none",
            1.0,
            source_width=100,
            source_height=60,
        )

        self.assertEqual((0.0, 0.0, 100.0, 60.0), source_box)


if __name__ == "__main__":
    unittest.main()
