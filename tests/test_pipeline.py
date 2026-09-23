import unittest
from dataclasses import replace
from queue import Queue
from threading import Condition, Event, RLock, get_ident
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import core.pipeline as pipeline_module
from core.pipeline import (
    CameraPipeline,
    _AnalysisSnapshot,
    _filter_false_person_detections,
)
from core.tracker import TrackedObject


class PipelineCadenceTests(unittest.TestCase):
    def test_analysis_worker_warmup_uses_live_frame_shape(self) -> None:
        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline.detector = MagicMock()
        pipeline.pose_estimator = MagicMock()
        pipeline._get_pipeline_params = lambda: {"processing_max_height": 720}
        pipeline._get_config = lambda: {"frame_rotation": "cw90"}

        pipeline._warmup_analysis_models()

        warmed_frame = pipeline.detector.detect.call_args_list[0].args[0]
        self.assertEqual((720, 405, 3), warmed_frame.shape)
        self.assertEqual(2, pipeline.detector.detect.call_count)
        self.assertEqual(2, pipeline.pose_estimator.warmup.call_count)

    def test_live_capture_keeps_only_the_latest_frame(self) -> None:
        class Capture:
            def __init__(self) -> None:
                self.frames: Queue[np.ndarray | None] = Queue()
                self.read_count = 0

            def read(self):
                frame = self.frames.get(timeout=1.0)
                if frame is None:
                    return False, None
                self.read_count += 1
                return True, frame

            def release(self) -> None:
                self.frames.put(None)

        capture = Capture()
        reader = pipeline_module._LatestFrameCapture(capture, Event())
        reader.start()
        try:
            for value in (1, 2, 3):
                capture.frames.put(np.full((1, 1, 3), value, dtype=np.uint8))

            deadline = monotonic() + 1.0
            while capture.read_count < 3 and monotonic() < deadline:
                sleep(0.005)

            latest = reader.read_latest(last_sequence=0, timeout=0.1)

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(3, latest.sequence)
            self.assertEqual(3, int(latest.frame[0, 0, 0]))
            self.assertEqual(2, latest.dropped_frames)
        finally:
            capture.frames.put(None)
            reader.stop()

    def test_run_redacts_rtsp_credentials_from_logs(self) -> None:
        class ClosedCapture:
            def isOpened(self) -> bool:
                return False

        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline._stop_event = Event()
        pipeline.frame_buffer = SimpleNamespace(set_status=lambda *_args, **_kwargs: None)
        pipeline._get_config = lambda: {
            "camera_id": "cam",
            "name": "Camera",
            "source": "rtsp://alice:top-secret@192.0.2.10/live",
        }
        pipeline._get_pipeline_params = lambda: {
            "max_reconnect_attempts": 1,
            "reconnect_delay": 0.0,
        }
        pipeline._open_capture = lambda _source: ClosedCapture()
        pipeline._sleep_interruptible = lambda _seconds: pipeline._stop_event.set()

        with patch("core.pipeline.logger.info") as log_info:
            pipeline._run()

        rendered_calls = " ".join(str(call) for call in log_info.call_args_list)
        self.assertNotIn("alice:top-secret", rendered_calls)
        self.assertIn("rtsp://***@192.0.2.10/live", rendered_calls)

    def test_analysis_due_respects_frame_skip_and_ai_max_fps(self) -> None:
        self.assertTrue(CameraPipeline._analysis_due(1, 1, 10, 100.0, 0.0))
        self.assertFalse(CameraPipeline._analysis_due(2, 3, 10, 100.2, 100.0))
        self.assertFalse(CameraPipeline._analysis_due(3, 3, 10, 100.05, 100.0))
        self.assertTrue(CameraPipeline._analysis_due(3, 3, 10, 100.11, 100.0))

    def test_analysis_due_allows_unlimited_ai_rate(self) -> None:
        self.assertTrue(CameraPipeline._analysis_due(5, 1, 0, 100.01, 100.0))

    def test_first_frame_uses_async_analysis_and_publishes_immediately(self) -> None:
        class Capture:
            def __init__(self, stop_event: Event) -> None:
                self.stop_event = stop_event
                self.read_count = 0

            def isOpened(self) -> bool:
                return True

            def read(self):
                self.read_count += 1
                if self.read_count == 1:
                    return True, np.zeros((4, 5, 3), dtype=np.uint8)
                self.stop_event.set()
                return False, None

            def release(self) -> None:
                return None

        class Buffer:
            def __init__(self) -> None:
                self.frames = []

            def set_status(self, *_args, **_kwargs) -> None:
                return None

            def update(self, frame, **_kwargs) -> None:
                self.frames.append(frame)

        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline._stop_event = Event()
        pipeline.frame_buffer = Buffer()
        pipeline.tracker = SimpleNamespace(reset=lambda: None)
        pipeline.behavior_engine = SimpleNamespace(reset_camera=lambda _camera_id: None)
        pipeline._published_analysis_result_id = 0
        pipeline._last_stale_warning_result_id = 0
        pipeline._get_config = lambda: {"camera_id": "cam", "name": "Camera", "source": 0}
        pipeline._get_pipeline_params = lambda: {
            "frame_skip": 2,
            "ai_max_fps": 10.0,
            "analysis_timeout": 5.0,
            "analysis_stale_after_ms": 500.0,
            "reconnect_delay": 0.0,
            "max_reconnect_attempts": 1,
            "processing_max_height": 720,
            "realtime_video_playback": True,
            "loop_video_files": False,
            "drop_late_video_frames": True,
        }
        pipeline._open_capture = lambda _source: Capture(pipeline._stop_event)
        pipeline._reset_analysis_state = lambda: None
        pipeline._run_analysis_sync = lambda *_args, **_kwargs: self.fail(
            "first-frame AI must not block capture"
        )
        submitted_frames = []
        pipeline._submit_analysis_if_due = (
            lambda _frame, _config, _camera_id, frame_index, _now, _params: submitted_frames.append(frame_index)
        )
        pipeline._analysis_snapshot = lambda: SimpleNamespace(
            new_alert_count=0,
            result_id=0,
            frame_index=0,
        )
        pipeline._display_frame_for_snapshot = lambda frame, *_args: SimpleNamespace(
            frame=frame,
            object_count=0,
            staleness_ms=0.0,
            stale_warning=False,
        )
        pipeline._monitor_fps_health = lambda *_args: None

        pipeline._run()

        self.assertEqual([1], submitted_frames)
        self.assertEqual(1, len(pipeline.frame_buffer.frames))

    def test_source_reset_clears_tracker_behavior_and_analysis_state(self) -> None:
        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline.tracker = MagicMock()
        pipeline.behavior_engine = MagicMock()
        pipeline._reset_analysis_state = MagicMock()

        pipeline._reset_source_state("cam")

        pipeline.tracker.reset.assert_called_once_with()
        pipeline.behavior_engine.reset_camera.assert_called_once_with("cam")
        pipeline._reset_analysis_state.assert_called_once_with()

    def test_inflight_analysis_timeout_allows_new_analysis(self) -> None:
        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline._analysis_state_lock = RLock()
        pipeline._analysis_inflight = True
        pipeline._last_analysis_started_at = 100.0
        pipeline._analysis_token = 0
        pipeline._analysis_generation = 0
        pipeline._analysis_condition = Condition(pipeline._analysis_state_lock)
        pipeline._analysis_thread = None
        pipeline._analysis_job = None
        pipeline._stop_event = Event()
        pipeline.frame_skip = 1
        pipeline.ai_max_fps = 10
        pipeline.analysis_timeout = 0.2
        calls = []
        done = Event()

        def analyze(frame, config, camera_id, token, generation, frame_capture_time, frame_index):
            calls.append((token, frame_capture_time, frame_index))
            with pipeline._analysis_state_lock:
                if pipeline._analysis_token == token:
                    pipeline._analysis_inflight = False
            done.set()

        pipeline._analyze_frame = analyze

        submitted = pipeline._submit_analysis_if_due(
            np.zeros((4, 5, 3), dtype=np.uint8),
            {"camera_id": "cam"},
            "cam",
            2,
            100.21,
            {"frame_skip": 1, "ai_max_fps": 10, "analysis_timeout": 0.2},
        )
        self.assertTrue(done.wait(timeout=1.0))
        pipeline._stop_event.set()
        with pipeline._analysis_condition:
            pipeline._analysis_condition.notify_all()
        pipeline._analysis_thread.join(timeout=1.0)

        self.assertTrue(submitted)
        self.assertEqual([(2, 100.21, 2)], calls)
        self.assertFalse(pipeline._analysis_inflight)

    def test_async_analysis_reuses_persistent_worker_thread(self) -> None:
        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline._analysis_state_lock = RLock()
        pipeline._analysis_condition = Condition(pipeline._analysis_state_lock)
        pipeline._analysis_thread = None
        pipeline._analysis_job = None
        pipeline._analysis_inflight = False
        pipeline._last_analysis_started_at = 0.0
        pipeline._analysis_token = 0
        pipeline._analysis_generation = 0
        pipeline._stop_event = Event()
        pipeline.frame_skip = 1
        pipeline.ai_max_fps = 0
        pipeline.analysis_timeout = 1.0
        thread_ids = []
        done = [Event(), Event()]

        def analyze(frame, config, camera_id, token, generation, frame_capture_time, frame_index):
            thread_ids.append(get_ident())
            with pipeline._analysis_state_lock:
                if pipeline._analysis_token == token:
                    pipeline._analysis_inflight = False
            done[len(thread_ids) - 1].set()

        pipeline._analyze_frame = analyze

        for index in range(2):
            submitted = pipeline._submit_analysis_if_due(
                np.zeros((4, 5, 3), dtype=np.uint8),
                {"camera_id": "cam"},
                "cam",
                index + 1,
                100.0 + index,
                {"frame_skip": 1, "ai_max_fps": 0, "analysis_timeout": 1.0},
            )
            self.assertTrue(submitted)
            self.assertTrue(done[index].wait(timeout=1.0))

        pipeline._stop_event.set()
        with pipeline._analysis_condition:
            pipeline._analysis_condition.notify_all()
        pipeline._analysis_thread.join(timeout=1.0)

        self.assertEqual(2, len(thread_ids))
        self.assertEqual(1, len(set(thread_ids)))

    def test_inflight_analysis_timeout_does_not_duplicate_live_thread(self) -> None:
        class LiveThread:
            def is_alive(self) -> bool:
                return True

        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline._analysis_state_lock = RLock()
        pipeline._analysis_inflight = True
        pipeline._analysis_thread = LiveThread()
        pipeline._last_analysis_started_at = 100.0
        pipeline._analysis_token = 0
        pipeline._analysis_generation = 0
        pipeline.frame_skip = 1
        pipeline.ai_max_fps = 10
        pipeline.analysis_timeout = 0.2
        pipeline._analyze_frame = lambda *args: None

        with patch("core.pipeline.logger.warning") as warning:
            submitted = pipeline._submit_analysis_if_due(
                np.zeros((4, 5, 3), dtype=np.uint8),
                {"camera_id": "cam"},
                "cam",
                2,
                100.21,
                {"frame_skip": 1, "ai_max_fps": 10, "analysis_timeout": 0.2},
            )
            pipeline._submit_analysis_if_due(
                np.zeros((4, 5, 3), dtype=np.uint8),
                {"camera_id": "cam"},
                "cam",
                2,
                100.21,
                {"frame_skip": 1, "ai_max_fps": 10, "analysis_timeout": 0.2},
            )

        self.assertFalse(submitted)
        self.assertTrue(pipeline._analysis_inflight)
        self.assertEqual(0, pipeline._analysis_token)
        self.assertEqual(1, warning.call_count)

    def test_display_frame_suppresses_stale_analysis_objects(self) -> None:
        current_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        analyzed_frame = np.full((8, 8, 3), 200, dtype=np.uint8)
        tracked = TrackedObject(
            track_id=7,
            bbox_xyxy=(1.0, 1.0, 6.0, 6.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[],
        )
        snapshot = _AnalysisSnapshot(
            objects=[tracked],
            counters={},
            person_timer_states={},
            new_alert_count=0,
            annotated_frame=analyzed_frame,
            result_id=1,
            frame_capture_time=100.0,
            frame_index=4,
        )

        display = CameraPipeline._display_frame_for_snapshot(
            current_frame,
            {"camera_id": "cam", "name": "Camera", "zones": [], "lines": []},
            snapshot,
            100.6,
            500.0,
        )

        self.assertTrue(display.used_analysis)
        self.assertTrue(display.stale_warning)
        self.assertEqual(1, display.object_count)
        self.assertAlmostEqual(600.0, display.staleness_ms)

    def test_display_frame_uses_fresh_analysis_objects(self) -> None:
        tracked = TrackedObject(
            track_id=7,
            bbox_xyxy=(1.0, 1.0, 6.0, 6.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[],
        )
        snapshot = _AnalysisSnapshot(
            objects=[tracked],
            counters={},
            person_timer_states={},
            new_alert_count=0,
            annotated_frame=np.zeros((8, 8, 3), dtype=np.uint8),
            result_id=1,
            frame_capture_time=100.0,
            frame_index=4,
        )

        display = CameraPipeline._display_frame_for_snapshot(
            np.zeros((8, 8, 3), dtype=np.uint8),
            {"camera_id": "cam", "name": "Camera", "zones": [], "lines": []},
            snapshot,
            100.2,
            500.0,
        )

        self.assertTrue(display.used_analysis)
        self.assertFalse(display.stale_warning)
        self.assertEqual(1, display.object_count)
        self.assertAlmostEqual(200.0, display.staleness_ms)

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

    def test_visual_alert_targets_expire_after_ttl(self) -> None:
        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline._active_visual_alerts = []

        visual_alerts = CameraPipeline._visual_alerts_for_alerts(
            [
                {
                    "type": "intrusion",
                    "track_id": 7,
                    "zone_id": "gate",
                }
            ],
            100.0,
        )
        active = pipeline._merge_visual_alerts_locked(visual_alerts, 100.0)

        self.assertEqual(1, len(active))
        self.assertEqual(7, active[0]["track_id"])
        self.assertEqual("gate", active[0]["zone_id"])

        expired = pipeline._merge_visual_alerts_locked([], 107.0)

        self.assertEqual([], expired)

    def test_visual_alert_keeps_occluded_fall_location(self) -> None:
        visual_alerts = CameraPipeline._visual_alerts_for_alerts(
            [
                {
                    "type": "possible_fall",
                    "track_id": 7,
                    "occluded": True,
                    "last_known_bbox": [10.0, 20.0, 30.0, 40.0],
                    "visual_hold_seconds": 60,
                }
            ],
            100.0,
        )

        self.assertTrue(visual_alerts[0]["occluded"])
        self.assertEqual([10.0, 20.0, 30.0, 40.0], visual_alerts[0]["last_known_bbox"])
        self.assertEqual(160.0, visual_alerts[0]["expires_at"])

    def test_recovery_clears_persistent_fall_visual(self) -> None:
        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline._active_visual_alerts = [
            {
                "type": "possible_fall",
                "track_id": 7,
                "started_at": 100.0,
                "expires_at": 160.0,
            }
        ]
        recovery = CameraPipeline._visual_alerts_for_alerts(
            [{"type": "fall_recovery", "track_id": 7}],
            110.0,
        )

        active = pipeline._merge_visual_alerts_locked(recovery, 110.0)

        self.assertEqual([], active)

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

    def test_pose_needed_follows_pose_enabled_setting(self) -> None:
        settings = {
            "pose": {"enabled": True},
            "behavior": {"theft": {"enabled": False}},
        }

        self.assertTrue(CameraPipeline._pose_needed({"zones": []}, settings))
        self.assertFalse(
            CameraPipeline._pose_needed(
                {"zones": []},
                {"pose": {"enabled": False}, "behavior": {"theft": {"enabled": True}}},
            )
        )

    def test_pose_filter_hysteresis_keeps_confirmed_person_briefly(self) -> None:
        confirmed = TrackedObject(
            track_id=7,
            bbox_xyxy=(1.0, 1.0, 6.0, 6.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[],
            pose_keypoints=[(1.0, 1.0, 0.9), (2.0, 2.0, 0.8), (3.0, 3.0, 0.7)],
        )
        pose_fail_streak: dict[int, int] = {}
        pose_confirmed_person: set[int] = set()

        kept = _filter_false_person_detections(
            [confirmed],
            3,
            0.25,
            pose_fail_streak,
            pose_confirmed_person,
            2,
        )
        first_dropout = _filter_false_person_detections(
            [replace(confirmed, pose_keypoints=None)],
            3,
            0.25,
            pose_fail_streak,
            pose_confirmed_person,
            2,
        )
        second_dropout = _filter_false_person_detections(
            [replace(confirmed, pose_keypoints=[])],
            3,
            0.25,
            pose_fail_streak,
            pose_confirmed_person,
            2,
        )
        third_dropout = _filter_false_person_detections(
            [replace(confirmed, pose_keypoints=None)],
            3,
            0.25,
            pose_fail_streak,
            pose_confirmed_person,
            2,
        )

        self.assertEqual([7], [obj.track_id for obj in kept])
        self.assertEqual([7], [obj.track_id for obj in first_dropout])
        self.assertEqual([7], [obj.track_id for obj in second_dropout])
        self.assertEqual([], third_dropout)

    def test_pose_filter_still_drops_unconfirmed_person_without_keypoints(self) -> None:
        unconfirmed = TrackedObject(
            track_id=8,
            bbox_xyxy=(1.0, 1.0, 6.0, 6.0),
            class_id=0,
            class_name="person",
            confidence=0.9,
            center_history=[],
            pose_keypoints=None,
        )

        filtered = _filter_false_person_detections(
            [unconfirmed],
            3,
            0.25,
            {},
            set(),
            8,
        )

        self.assertEqual([], filtered)


if __name__ == "__main__":
    unittest.main()
