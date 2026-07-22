import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np

from analytics.identity_status import KNOWN_PERSON_KIND
from utils.drawing import (
    ALERT_COLOR,
    ALERT_FLASH_INTERVAL_SECONDS,
    EMERGENCY_COLOR,
    EMERGENCY_FLASH_INTERVAL_SECONDS,
    draw_annotations,
)
from utils.drawing import _draw_frame_hud, _draw_object, _draw_theft_overlay, _object_label, _visual_scale
from core.tracker import TrackedObject


class DrawingTests(unittest.TestCase):
    def test_alert_visual_constants_match_project_alert_style(self) -> None:
        self.assertEqual((48, 59, 255), ALERT_COLOR)
        self.assertEqual(0.25, ALERT_FLASH_INTERVAL_SECONDS)
        self.assertEqual((32, 48, 255), EMERGENCY_COLOR)
        self.assertEqual(0.5, EMERGENCY_FLASH_INTERVAL_SECONDS)

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

    def test_frame_hud_is_centered_at_top(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        text = "motorcycle | Objects: 0"

        with patch("utils.drawing._draw_label") as draw_label:
            _draw_frame_hud(frame, {"name": "motorcycle"}, [])

        origin = draw_label.call_args.args[2]
        visual_scale = _visual_scale(frame)
        scale = 0.56 * visual_scale
        thickness = max(1, int(round(1.15 * visual_scale)))
        (text_width, _), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )
        self.assertAlmostEqual(frame.shape[1] / 2, origin[0] + text_width / 2, delta=1)
        self.assertEqual(34, origin[1])

    def test_theft_overlay_respects_camera_toggle(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        states = [{"score": 1, "score_threshold": 2}]

        with patch("utils.drawing._draw_theft_overlay") as draw_overlay:
            draw_annotations(
                frame,
                [],
                {"camera_id": "cam", "show_theft_overlay": False},
                theft_states=states,
            )
            draw_overlay.assert_not_called()

            draw_annotations(
                frame,
                [],
                {"camera_id": "cam", "show_theft_overlay": True},
                theft_states=states,
            )
            draw_overlay.assert_called_once()
            self.assertEqual(states, draw_overlay.call_args.args[1])

    def test_alerted_theft_state_flashes_its_zone_without_a_new_notification(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        config = {
            "camera_id": "cam",
            "zones": [
                {
                    "id": "bike-zone",
                    "name": "Bike Zone",
                    "type": "asset_watch",
                    "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                }
            ],
        }
        states = [{"zone_id": "bike-zone", "alerted": True}]

        with patch("utils.drawing.time.monotonic", return_value=0.5):
            with patch("utils.drawing._draw_zone") as draw_zone:
                draw_annotations(frame, [], config, theft_states=states)

        self.assertTrue(draw_zone.call_args.kwargs["alerting"])
        self.assertTrue(draw_zone.call_args.kwargs["flash_on"])

        with patch("utils.drawing.time.monotonic", return_value=0.75):
            with patch("utils.drawing._draw_zone") as draw_zone:
                draw_annotations(frame, [], config, theft_states=states)

        self.assertTrue(draw_zone.call_args.kwargs["alerting"])
        self.assertFalse(draw_zone.call_args.kwargs["flash_on"])

    def test_alerted_theft_overlay_uses_red_border(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        state = {
            "score": 4,
            "score_threshold": 2,
            "alerted": True,
            "person_track_id": 4,
            "vehicle_track_id": 1,
        }

        with patch("utils.drawing.cv2.rectangle") as rectangle:
            _draw_theft_overlay(frame, [state])

        self.assertTrue(any(call.args[3] == ALERT_COLOR for call in rectangle.call_args_list))

    def test_fall_alert_uses_distinct_emergency_visual_for_known_person(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        person = TrackedObject(
            track_id=4,
            bbox_xyxy=(10.0, 10.0, 40.0, 80.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[(25.0, 45.0)],
            identity_label="Tuan",
            identity_kind=KNOWN_PERSON_KIND,
            pose_label="lying",
        )
        alerts = [
            {
                "type": "possible_fall",
                "track_id": 4,
                "started_at": 100.0,
                "expires_at": 106.0,
            }
        ]

        with patch("utils.drawing.time.monotonic", return_value=100.0):
            with patch("utils.drawing._draw_object") as draw_object:
                with patch(
                    "utils.drawing._draw_emergency_overlay", create=True
                ) as draw_emergency:
                    draw_annotations(frame, [person], {"camera_id": "cam"}, active_alerts=alerts)

        self.assertFalse(draw_object.call_args.kwargs["alerting"])
        self.assertEqual("possible fall", draw_object.call_args.kwargs["emergency_label"])
        self.assertTrue(draw_object.call_args.kwargs["emergency_flash_on"])
        draw_emergency.assert_called_once()
        self.assertEqual("SOS | POSSIBLE FALL", draw_emergency.call_args.args[1])

    def test_fall_recovery_does_not_flash_as_an_emergency(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        person = TrackedObject(
            track_id=4,
            bbox_xyxy=(10.0, 10.0, 40.0, 80.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[(25.0, 45.0)],
            identity_label="Tuan",
            identity_kind=KNOWN_PERSON_KIND,
            pose_label="sitting",
        )
        alerts = [
            {
                "type": "fall_recovery",
                "track_id": 4,
                "started_at": 100.0,
                "expires_at": 106.0,
            }
        ]

        with patch("utils.drawing.time.monotonic", return_value=100.0):
            with patch("utils.drawing._draw_object") as draw_object:
                with patch(
                    "utils.drawing._draw_emergency_overlay", create=True
                ) as draw_emergency:
                    draw_annotations(frame, [person], {"camera_id": "cam"}, active_alerts=alerts)

        self.assertIsNone(draw_object.call_args.kwargs["emergency_label"])
        draw_emergency.assert_not_called()

    def test_unresponsive_alert_uses_the_stronger_sos_label(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        alerts = [
            {
                "type": "possible_fall",
                "track_id": 4,
                "started_at": 100.0,
                "expires_at": 106.0,
            },
            {
                "type": "possible_unresponsive",
                "track_id": 4,
                "started_at": 100.0,
                "expires_at": 106.0,
            },
        ]

        with patch("utils.drawing.time.monotonic", return_value=100.0):
            with patch("utils.drawing._draw_emergency_overlay") as draw_emergency:
                draw_annotations(frame, [], {"camera_id": "cam"}, active_alerts=alerts)

        draw_emergency.assert_called_once()
        self.assertEqual("SOS | POSSIBLE EMERGENCY", draw_emergency.call_args.args[1])

    def test_emergency_object_label_replaces_lying_pose(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        person = TrackedObject(
            track_id=4,
            bbox_xyxy=(10.0, 10.0, 140.0, 180.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[(75.0, 95.0)],
            identity_label="Tuan",
            identity_kind=KNOWN_PERSON_KIND,
            pose_label="lying",
        )

        with patch("utils.drawing._draw_label") as draw_label:
            _draw_object(
                frame,
                person,
                emergency_label="possible emergency",
                emergency_flash_on=True,
            )

        self.assertEqual("SOS Tuan - possible emergency", draw_label.call_args.args[1])

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

    def test_person_transition_pose_uses_readable_label(self) -> None:
        obj = TrackedObject(
            track_id=4,
            bbox_xyxy=(10.0, 10.0, 40.0, 80.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[(25.0, 45.0)],
            identity_label="Stranger",
            identity_kind="stranger",
            pose_label="getting_up",
        )

        self.assertEqual("Stranger #4 - getting up", _object_label(obj))
        self.assertEqual(
            "Stranger #4 - changing posture",
            _object_label(replace(obj, pose_label="changing_to_lying")),
        )


if __name__ == "__main__":
    unittest.main()
