"""Regression tests for required runtime settings."""

from __future__ import annotations

import unittest
from pathlib import Path
from threading import RLock
from unittest.mock import MagicMock
from tempfile import TemporaryDirectory

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

    def test_default_profile_is_realtime_optimized(self) -> None:
        self.assertEqual(800, MAX_QUALITY_RUNTIME_SETTINGS["detection"]["imgsz"])
        self.assertEqual("yolo11n-pose.pt", MAX_QUALITY_RUNTIME_SETTINGS["pose"]["model"])
        self.assertEqual(640, MAX_QUALITY_RUNTIME_SETTINGS["pose"]["imgsz"])
        self.assertEqual(10, MAX_QUALITY_RUNTIME_SETTINGS["pipeline"]["ai_max_fps"])
        self.assertEqual(720, MAX_QUALITY_RUNTIME_SETTINGS["pipeline"]["processing_max_height"])


if __name__ == "__main__":
    unittest.main()
