"""Regression tests for required runtime settings."""

from __future__ import annotations

import unittest
from pathlib import Path
from threading import RLock
from unittest.mock import MagicMock, patch
from tempfile import TemporaryDirectory

from analytics.behavior_engine import BehaviorEngine
from core.pose import PoseEstimator
from web.app import (
    MAX_QUALITY_RUNTIME_SETTINGS,
    RuntimeState,
    _enforce_required_runtime_settings,
)


class RuntimeSettingsTests(unittest.TestCase):
    def test_pose_can_be_disabled(self) -> None:
        settings = _enforce_required_runtime_settings(
            {"pose": {"enabled": False, "allow_download": False}}
        )

        self.assertFalse(settings["pose"]["enabled"])
        self.assertFalse(settings["pose"]["allow_download"])

    def test_pose_model_change_resets_loaded_model(self) -> None:
        estimator = PoseEstimator(
            {"pose": {"enabled": False, "model": "first.pt"}},
            RLock(),
            "cpu",
            False,
        )
        estimator.model = object()

        estimator.update_settings(
            {"pose": {"enabled": False, "model": "second.pt"}},
        )

        self.assertFalse(estimator.enabled)
        self.assertIsNone(estimator.model)
        self.assertEqual(estimator.model_path, "second.pt")

    def test_disabling_pose_releases_loaded_model(self) -> None:
        estimator = PoseEstimator(
            {"pose": {"enabled": True, "model": "first.pt"}},
            RLock(),
            "cpu",
            False,
        )
        estimator.model = object()

        estimator.update_settings({"pose": {"enabled": False, "model": "first.pt"}})

        self.assertFalse(estimator.enabled)
        self.assertIsNone(estimator.model)

    def test_new_camera_automatically_applies_max_quality_profile(self) -> None:
        with TemporaryDirectory() as directory:
            runtime = RuntimeState.__new__(RuntimeState)
            runtime._lock = RLock()
            runtime.settings = {
                "pipeline": {"stream_max_height": 720},
                "behavior": {},
            }
            runtime.settings_path = Path(directory) / "settings.yaml"
            runtime.cameras_dir = Path(directory) / "cameras"
            runtime.cameras = {}
            runtime.frame_buffers = {}
            runtime.pipelines = {}
            runtime.behavior_engine = MagicMock()
            runtime.behavior_engine.get_counters.return_value = {}
            runtime.update_settings = MagicMock(return_value={})
            runtime._save_camera_config = MagicMock()
            runtime._restart_pipeline = MagicMock()

            saved = runtime.upsert_camera(
                {
                    "name": "Living Room",
                    "source": "rtsp://camera",
                    "enabled": True,
                }
            )

        runtime.update_settings.assert_called_once_with(MAX_QUALITY_RUNTIME_SETTINGS)
        runtime._restart_pipeline.assert_called_once()
        self.assertEqual(saved["vision_profile"], "max_quality_realtime")

    def test_camera_update_preserves_frame_rotation(self) -> None:
        with TemporaryDirectory() as directory:
            runtime = RuntimeState.__new__(RuntimeState)
            runtime._lock = RLock()
            runtime.settings = {"pipeline": {"stream_max_height": 720}}
            runtime.cameras_dir = Path(directory) / "cameras"
            runtime.cameras = {
                "imou_camera": {
                    "camera_id": "imou_camera",
                    "name": "IMOU Camera",
                    "source": "rtsp://camera",
                    "enabled": True,
                    "frame_rotation": "cw90",
                    "unknown_person_policy": "assume_stranger",
                    "notification_channels": ["telegram"],
                    "zones": [],
                    "lines": [],
                }
            }
            runtime.frame_buffers = {}
            runtime.pipelines = {}
            runtime.behavior_engine = MagicMock()
            runtime.behavior_engine.get_counters.return_value = {}
            runtime.update_settings = MagicMock(return_value={})
            runtime._save_camera_config = MagicMock()
            runtime._restart_pipeline = MagicMock()

            saved = runtime.upsert_camera(
                {
                    "camera_id": "imou_camera",
                    "name": "IMOU Camera",
                    "source": "rtsp://camera",
                    "enabled": True,
                    "notification_channels": ["discord"],
                }
            )

        runtime.update_settings.assert_not_called()
        self.assertEqual("cw90", saved["frame_rotation"])
        self.assertEqual("assume_stranger", saved["unknown_person_policy"])
        runtime._save_camera_config.assert_called_once()
        runtime._restart_pipeline.assert_called_once()

    @patch("web.app.CameraPipeline")
    def test_restart_pipeline_uses_camera_scoped_behavior_engines(self, pipeline_cls) -> None:
        first_pipeline = MagicMock()
        second_pipeline = MagicMock()
        pipeline_cls.side_effect = [first_pipeline, second_pipeline]

        runtime = RuntimeState.__new__(RuntimeState)
        runtime._lock = RLock()
        runtime.settings = {
            "pipeline": {"stream_max_height": 720},
            "behavior": {},
            "behavior_learning": {"enabled": False},
            "identity": {"enabled": False},
        }
        runtime.frame_buffers = {}
        runtime.pipelines = {}
        runtime.detector = MagicMock()
        runtime.pose_estimator = MagicMock()
        runtime.alert_manager = MagicMock()
        runtime.behavior_engine = BehaviorEngine(runtime.settings)
        runtime.identity_resolver = runtime.behavior_engine.identity_resolver

        runtime._restart_pipeline(
            {
                "camera_id": "front",
                "name": "Front",
                "enabled": True,
            }
        )
        runtime._restart_pipeline(
            {
                "camera_id": "garage",
                "name": "Garage",
                "enabled": True,
            }
        )

        engines = [
            call.kwargs["behavior_engine"]
            for call in pipeline_cls.call_args_list
        ]
        self.assertIsNot(engines[0], engines[1])
        self.assertIs(engines[0].identity_resolver, runtime.identity_resolver)
        self.assertIs(engines[1].identity_resolver, runtime.identity_resolver)
        first_pipeline.start.assert_called_once()
        second_pipeline.start.assert_called_once()

    def test_default_profile_is_realtime_optimized(self) -> None:
        self.assertEqual("yolo11n.pt", MAX_QUALITY_RUNTIME_SETTINGS["detection"]["model"])
        self.assertEqual(640, MAX_QUALITY_RUNTIME_SETTINGS["detection"]["imgsz"])
        self.assertEqual("yolo11n-pose.pt", MAX_QUALITY_RUNTIME_SETTINGS["pose"]["model"])
        self.assertEqual(640, MAX_QUALITY_RUNTIME_SETTINGS["pose"]["imgsz"])
        self.assertEqual(2, MAX_QUALITY_RUNTIME_SETTINGS["pipeline"]["frame_skip"])
        self.assertEqual(10, MAX_QUALITY_RUNTIME_SETTINGS["pipeline"]["ai_max_fps"])
        self.assertEqual(500, MAX_QUALITY_RUNTIME_SETTINGS["pipeline"]["analysis_stale_after_ms"])
        self.assertEqual(5.0, MAX_QUALITY_RUNTIME_SETTINGS["pipeline"]["analysis_timeout_min_seconds"])
        self.assertEqual(720, MAX_QUALITY_RUNTIME_SETTINGS["pipeline"]["processing_max_height"])
        self.assertEqual(
            0.12,
            MAX_QUALITY_RUNTIME_SETTINGS["detection"]["class_confidences"]["dog"],
        )
        self.assertEqual(
            0.10,
            MAX_QUALITY_RUNTIME_SETTINGS["detection"]["class_confidences"]["bicycle"],
        )
        self.assertEqual(
            0.20,
            MAX_QUALITY_RUNTIME_SETTINGS["detection"]["class_confidences"]["person"],
        )
        self.assertEqual(0.10, MAX_QUALITY_RUNTIME_SETTINGS["tracking"]["track_high_thresh"])
        self.assertEqual(0.05, MAX_QUALITY_RUNTIME_SETTINGS["tracking"]["track_low_thresh"])
        self.assertEqual(0.10, MAX_QUALITY_RUNTIME_SETTINGS["tracking"]["new_track_thresh"])
        self.assertEqual(90, MAX_QUALITY_RUNTIME_SETTINGS["tracking"]["track_buffer"])
        self.assertEqual(3, MAX_QUALITY_RUNTIME_SETTINGS["tracking"]["track_grace_frames"])
        self.assertFalse(
            MAX_QUALITY_RUNTIME_SETTINGS["tracking"]["camera_motion_compensation"]["enabled"]
        )
        self.assertEqual(
            0.7,
            MAX_QUALITY_RUNTIME_SETTINGS["tracking"]["duplicate_containment_threshold"],
        )
        self.assertEqual(0.45, MAX_QUALITY_RUNTIME_SETTINGS["identity"]["similarity_threshold"])
        self.assertEqual("buffalo_sc", MAX_QUALITY_RUNTIME_SETTINGS["identity"]["model"])
        self.assertEqual("cuda:0", MAX_QUALITY_RUNTIME_SETTINGS["identity"]["device"])
        self.assertEqual(320, MAX_QUALITY_RUNTIME_SETTINGS["identity"]["detection_size"])
        self.assertEqual(30, MAX_QUALITY_RUNTIME_SETTINGS["identity"]["min_face_size"])
        self.assertEqual(10, MAX_QUALITY_RUNTIME_SETTINGS["identity"]["recognition_interval_frames"])
        self.assertEqual(5, MAX_QUALITY_RUNTIME_SETTINGS["identity"]["unknown_confirmation_attempts"])
        self.assertEqual(90, MAX_QUALITY_RUNTIME_SETTINGS["identity"]["known_memory_frames"])
        self.assertEqual(
            0.25,
            MAX_QUALITY_RUNTIME_SETTINGS["identity"]["known_memory_min_area_ratio"],
        )


if __name__ == "__main__":
    unittest.main()
