import unittest

import numpy as np

from utils.drawing import ALERT_COLOR, ALERT_FLASH_INTERVAL_SECONDS, draw_annotations
from utils.drawing import _object_label
from core.tracker import TrackedObject


class DrawingTests(unittest.TestCase):
    def test_alert_visual_constants_match_project_alert_style(self) -> None:
        self.assertEqual((48, 59, 255), ALERT_COLOR)
        self.assertEqual(0.25, ALERT_FLASH_INTERVAL_SECONDS)

    def test_roi_overlay_keeps_zone_interior_clear(self) -> None:
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        annotated = draw_annotations(
            frame,
            [],
            {
                "camera_id": "cam",
                "name": "Camera",
                "zones": [
                    {
                        "id": "gate",
                        "name": "Gate",
                        "type": "intrusion",
                        "polygon": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
                    }
                ],
                "lines": [],
            },
        )

        self.assertEqual([0, 0, 0], annotated[70, 70].tolist())

    def test_person_label_includes_known_pose(self) -> None:
        obj = TrackedObject(
            track_id=4,
            bbox_xyxy=(10.0, 10.0, 40.0, 80.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[(25.0, 45.0)],
            identity_label="Stranger",
            identity_kind="stranger",
            pose_label="lying",
        )

        self.assertEqual("Stranger #4 - lying", _object_label(obj))

    def test_person_label_omits_unknown_pose(self) -> None:
        obj = TrackedObject(
            track_id=4,
            bbox_xyxy=(10.0, 10.0, 40.0, 80.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[(25.0, 45.0)],
            identity_label="Stranger",
            identity_kind="stranger",
            pose_label="unknown",
        )

        self.assertEqual("Stranger #4", _object_label(obj))


if __name__ == "__main__":
    unittest.main()
