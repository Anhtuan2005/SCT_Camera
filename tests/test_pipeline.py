import unittest
from threading import RLock

import numpy as np

from core.pipeline import CameraPipeline, _AnalysisSnapshot


class PipelineCadenceTests(unittest.TestCase):
    def test_analysis_due_respects_frame_skip_and_ai_max_fps(self) -> None:
        self.assertTrue(CameraPipeline._analysis_due(1, 1, 10, 100.0, 0.0))
        self.assertFalse(CameraPipeline._analysis_due(2, 3, 10, 100.2, 100.0))
        self.assertFalse(CameraPipeline._analysis_due(3, 3, 10, 100.05, 100.0))
        self.assertTrue(CameraPipeline._analysis_due(3, 3, 10, 100.11, 100.0))

    def test_analysis_due_allows_unlimited_ai_rate(self) -> None:
        self.assertTrue(CameraPipeline._analysis_due(5, 1, 0, 100.01, 100.0))

    def test_run_analysis_sync_completes_before_returning(self) -> None:
        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline._analysis_state_lock = RLock()
        pipeline._analysis_inflight = False
        pipeline._last_analysis_started_at = 0.0
        pipeline._analysis_token = 0
        pipeline._analysis_generation = 0
        calls = []

        def analyze(frame, config, camera_id, token, generation):
            calls.append((frame.shape, config["camera_id"], camera_id, token, generation))
            with pipeline._analysis_state_lock:
                if pipeline._analysis_token == token:
                    pipeline._analysis_inflight = False

        pipeline._analyze_frame = analyze

        completed = pipeline._run_analysis_sync(
            np.zeros((4, 5, 3), dtype=np.uint8),
            {"camera_id": "cam"},
            "cam",
            42.0,
        )

        self.assertTrue(completed)
        self.assertEqual([((4, 5, 3), "cam", "cam", 1, 0)], calls)
        self.assertFalse(pipeline._analysis_inflight)
        self.assertEqual(42.0, pipeline._last_analysis_started_at)

    def test_display_frame_uses_analyzed_frame_to_keep_boxes_aligned(self) -> None:
        current_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        analyzed_frame = np.full((8, 8, 3), 200, dtype=np.uint8)
        snapshot = _AnalysisSnapshot(
            objects=[],
            counters={},
            person_timer_states={},
            new_alert_count=0,
            annotated_frame=analyzed_frame,
            result_id=1,
        )

        displayed = CameraPipeline._display_frame_for_snapshot(
            current_frame,
            {"camera_id": "cam", "name": "Camera", "zones": [], "lines": []},
            snapshot,
        )

        np.testing.assert_array_equal(analyzed_frame, displayed)

    def test_publish_snapshot_only_when_analysis_result_changes(self) -> None:
        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline._published_analysis_result_id = 2

        self.assertFalse(
            pipeline._should_publish_snapshot(
                _AnalysisSnapshot([], {}, {}, 0, np.zeros((1, 1, 3), dtype=np.uint8), 2)
            )
        )
        self.assertTrue(
            pipeline._should_publish_snapshot(
                _AnalysisSnapshot([], {}, {}, 0, np.zeros((1, 1, 3), dtype=np.uint8), 3)
            )
        )
        self.assertTrue(
            pipeline._should_publish_snapshot(
                _AnalysisSnapshot([], {}, {}, 0, None, 0)
            )
        )

    def test_apply_frame_rotation_uses_camera_config(self) -> None:
        frame = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

        self.assertEqual(
            [[4, 1], [5, 2], [6, 3]],
            CameraPipeline._apply_frame_rotation(
                frame,
                {"frame_rotation": "cw90"},
            ).tolist(),
        )
        self.assertEqual(
            [[3, 6], [2, 5], [1, 4]],
            CameraPipeline._apply_frame_rotation(
                frame,
                {"frame_rotation": "ccw90"},
            ).tolist(),
        )
        self.assertEqual(
            [[6, 5, 4], [3, 2, 1]],
            CameraPipeline._apply_frame_rotation(
                frame,
                {"frame_rotation": "180"},
            ).tolist(),
        )
        self.assertEqual(
            frame.tolist(),
            CameraPipeline._apply_frame_rotation(
                frame,
                {"frame_rotation": "none"},
            ).tolist(),
        )

    def test_pose_needed_only_for_enabled_theft_asset_zones(self) -> None:
        settings = {
            "pose": {"enabled": True},
            "behavior": {"theft": {"enabled": True}},
        }
        asset_zone = {
            "type": "asset_watch",
            "polygon": [[0, 0], [1, 0], [1, 1]],
        }

        self.assertFalse(CameraPipeline._pose_needed({"zones": []}, settings))
        self.assertTrue(CameraPipeline._pose_needed({"zones": [asset_zone]}, settings))
        self.assertFalse(
            CameraPipeline._pose_needed(
                {"zones": [asset_zone]},
                {"pose": {"enabled": False}, "behavior": {"theft": {"enabled": True}}},
            )
        )
        self.assertFalse(
            CameraPipeline._pose_needed(
                {"zones": [asset_zone]},
                {"pose": {"enabled": True}, "behavior": {"theft": {"enabled": False}}},
            )
        )


if __name__ == "__main__":
    unittest.main()
