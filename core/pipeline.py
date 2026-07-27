"""Camera pipeline orchestrator."""

from __future__ import annotations

import copy
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp",
)

import cv2
import numpy as np

from analytics.behavior_engine import BehaviorEngine
from core.detector import YOLOv11Detector
from core.frame_buffer import FrameBuffer, FrameSnapshot
from core.pose import PoseEstimator
from core.tracker import ByteTrackTracker, TrackedObject
from utils.drawing import draw_annotations
from utils.logger import get_logger

logger = get_logger(__name__)

STALE_FRAME_WARNING_MS = 500.0
FPS_MONITOR_INTERVAL_SECONDS = 5.0
LOW_FPS_THRESHOLD = 5.0
LOW_FPS_SOURCE_RATIO = 0.8
LOW_FPS_WARNING_SECONDS = 10.0
FPS_SPIKE_THRESHOLD = 500.0
MAX_RECONNECT_BACKOFF_SECONDS = 60.0
VISUAL_ALERT_SECONDS = 6.0
MAX_ACTIVE_VISUAL_ALERTS = 32
POSE_FILTER_FAIL_GRACE_FRAMES = 8


def _redact_source_for_log(source: int | str) -> int | str:
    if not isinstance(source, str):
        return source
    return re.sub(r"(?i)\b(rtsps?://)[^/\s]*@", r"\1***@", source)


def _filter_false_person_detections(
    objects: list[TrackedObject],
    min_valid_keypoints: int = 3,
    min_keypoint_confidence: float = 0.25,
    pose_fail_streak: dict[int, int] | None = None,
    pose_confirmed_person: set[int] | None = None,
    fail_grace_frames: int = 0,
) -> list[TrackedObject]:
    """Remove 'person' detections that have too few valid pose keypoints.

    Real people always produce several confident keypoints (shoulders, hips,
    knees, etc.).  False positives — motorcycles, chairs, picture frames —
    almost never produce valid pose output.
    """
    use_hysteresis = pose_fail_streak is not None and pose_confirmed_person is not None
    if use_hysteresis and pose_fail_streak is not None and pose_confirmed_person is not None:
        return _filter_false_person_detections_with_hysteresis(
            objects,
            min_valid_keypoints,
            min_keypoint_confidence,
            pose_fail_streak,
            pose_confirmed_person,
            fail_grace_frames,
        )
    filtered: list[TrackedObject] = []
    for obj in objects:
        if obj.class_name != "person":
            filtered.append(obj)
            continue
        # No keypoints at all means pose model didn't find a matching person
        # — almost certainly a false positive.
        kps = obj.pose_keypoints
        keypoint_confidences = (
            [] if not kps else [round(float(conf), 3) for _x, _y, conf in kps]
        )
        if kps is None or len(kps) == 0:
            logger.debug(
                (
                    "Dropping false person detection track=%d valid_keypoints=%d "
                    "keypoint_confidences=%s"
                ),
                obj.track_id,
                0,
                keypoint_confidences,
            )
            continue
        valid_count = sum(
            1 for x, y, conf in kps if conf >= min_keypoint_confidence
        )
        if valid_count < min_valid_keypoints:
            logger.debug(
                "Dropping false person detection track=%d "
                "valid_keypoints=%d min_valid_keypoints=%d "
                "keypoint_confidences=%s",
                obj.track_id,
                valid_count,
                min_valid_keypoints,
                keypoint_confidences,
            )
            continue
        logger.debug(
            (
                "Keeping person detection track=%d valid_keypoints=%d "
                "min_valid_keypoints=%d keypoint_confidences=%s"
            ),
            obj.track_id,
            valid_count,
            min_valid_keypoints,
            keypoint_confidences,
        )
        filtered.append(obj)
    return filtered


def _filter_false_person_detections_with_hysteresis(
    objects: list[TrackedObject],
    min_valid_keypoints: int,
    min_keypoint_confidence: float,
    pose_fail_streak: dict[int, int],
    pose_confirmed_person: set[int],
    fail_grace_frames: int,
) -> list[TrackedObject]:
    active_person_ids = {obj.track_id for obj in objects if obj.class_name == "person"}
    filtered: list[TrackedObject] = []
    for obj in objects:
        if obj.class_name != "person":
            filtered.append(obj)
            continue

        kps = obj.pose_keypoints
        keypoint_confidences = (
            [] if not kps else [round(float(conf), 3) for _x, _y, conf in kps]
        )
        valid_count = (
            0
            if not kps
            else sum(
                1
                for _x, _y, conf in kps
                if conf >= min_keypoint_confidence
            )
        )
        if valid_count >= min_valid_keypoints:
            pose_confirmed_person.add(obj.track_id)
            pose_fail_streak.pop(obj.track_id, None)
            logger.debug(
                (
                    "Keeping person detection track=%d valid_keypoints=%d "
                    "min_valid_keypoints=%d keypoint_confidences=%s"
                ),
                obj.track_id,
                valid_count,
                min_valid_keypoints,
                keypoint_confidences,
            )
            filtered.append(obj)
            continue

        if obj.track_id in pose_confirmed_person:
            fail_count = pose_fail_streak.get(obj.track_id, 0) + 1
            pose_fail_streak[obj.track_id] = fail_count
            if fail_count <= max(0, int(fail_grace_frames)):
                logger.debug(
                    (
                        "Keeping confirmed person track=%d despite weak pose "
                        "valid_keypoints=%d fail_streak=%d grace_frames=%d "
                        "keypoint_confidences=%s"
                    ),
                    obj.track_id,
                    valid_count,
                    fail_count,
                    fail_grace_frames,
                    keypoint_confidences,
                )
                filtered.append(obj)
                continue

        if not kps:
            logger.debug(
                (
                    "Dropping false person detection track=%d valid_keypoints=%d "
                    "keypoint_confidences=%s"
                ),
                obj.track_id,
                valid_count,
                keypoint_confidences,
            )
        else:
            logger.debug(
                "Dropping false person detection track=%d "
                "valid_keypoints=%d min_valid_keypoints=%d "
                "keypoint_confidences=%s",
                obj.track_id,
                valid_count,
                min_valid_keypoints,
                keypoint_confidences,
            )

    for track_id in list(pose_fail_streak):
        if track_id not in active_person_ids:
            pose_fail_streak.pop(track_id, None)
    pose_confirmed_person.intersection_update(active_person_ids)
    return filtered


@dataclass(frozen=True)
class _AnalysisSnapshot:
    objects: list[TrackedObject]
    counters: dict[str, Any]
    person_timer_states: dict[int, dict[str, Any]]
    new_alert_count: int
    annotated_frame: Any | None
    result_id: int
    frame_capture_time: float = 0.0
    frame_index: int = 0
    theft_states: list[dict[str, Any]] = field(default_factory=list)
    visual_alerts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _DisplayFrame:
    frame: Any
    object_count: int
    staleness_ms: float
    used_analysis: bool
    stale_warning: bool


@dataclass(frozen=True)
class _AnalysisJob:
    frame: Any
    config: dict[str, Any]
    camera_id: str
    token: int
    generation: int
    frame_capture_time: float
    frame_index: int


@dataclass(frozen=True)
class _CapturedFrame:
    frame: Any
    sequence: int
    captured_at: float
    capture_fps: float
    dropped_frames: int


class _LatestFrameCapture:
    """Continuously drain a live capture and retain only its newest frame."""

    def __init__(self, capture: Any, parent_stop_event: threading.Event) -> None:
        self._capture = capture
        self._parent_stop_event = parent_stop_event
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._frame: Any | None = None
        self._sequence = 0
        self._consumed_sequence = 0
        self._captured_at = 0.0
        self._last_frame_at = 0.0
        self._capture_fps = 0.0
        self._dropped_frames = 0
        self._failed = False

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="latest-frame-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.5)

    def read_latest(
        self,
        last_sequence: int,
        timeout: float = 2.0,
    ) -> _CapturedFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._sequence > last_sequence
                    or self._failed
                    or self._stop_event.is_set()
                    or self._parent_stop_event.is_set()
                ),
                timeout=timeout,
            )
            if self._sequence <= last_sequence or self._frame is None:
                return None
            self._consumed_sequence = self._sequence
            return _CapturedFrame(
                frame=self._frame,
                sequence=self._sequence,
                captured_at=self._captured_at,
                capture_fps=self._capture_fps,
                dropped_frames=self._dropped_frames,
            )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set() and not self._parent_stop_event.is_set():
                ok, frame = self._capture.read()
                captured_at = time.monotonic()
                with self._condition:
                    if not ok or frame is None:
                        self._failed = True
                        self._condition.notify_all()
                        return
                    if self._sequence > self._consumed_sequence:
                        self._dropped_frames += 1
                    if self._last_frame_at > 0:
                        delta = max(captured_at - self._last_frame_at, 1e-6)
                        instant_fps = 1.0 / delta
                        self._capture_fps = (
                            instant_fps
                            if self._capture_fps <= 0
                            else (self._capture_fps * 0.85) + (instant_fps * 0.15)
                        )
                    self._last_frame_at = captured_at
                    self._frame = frame
                    self._sequence += 1
                    self._captured_at = captured_at
                    self._condition.notify_all()
        finally:
            self._capture.release()
            with self._condition:
                self._condition.notify_all()


class CameraPipeline:
    """Run capture, tracking, analytics, annotation, and alerting for one camera."""

    def __init__(
        self,
        camera_config: dict[str, Any],
        settings: dict[str, Any],
        detector: YOLOv11Detector,
        pose_estimator: PoseEstimator,
        behavior_engine: BehaviorEngine,
        frame_buffer: FrameBuffer,
        alert_manager: Any,
    ) -> None:
        self._config_lock = threading.RLock()
        self.camera_config = copy.deepcopy(camera_config)
        self._settings = copy.deepcopy(settings)
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.behavior_engine = behavior_engine
        self.frame_buffer = frame_buffer
        self.alert_manager = alert_manager
        self.tracker = ByteTrackTracker(detector, settings)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._analysis_state_lock = threading.RLock()
        self._analysis_condition = threading.Condition(self._analysis_state_lock)
        self._analysis_thread: threading.Thread | None = None
        self._analysis_ready = threading.Event()
        self._analysis_job: _AnalysisJob | None = None
        self._analysis_inflight = False
        self._analysis_token = 0
        self._analysis_generation = 0
        self._last_analysis_started_at = 0.0
        self._analysis_timeout_warning_token: int | None = None
        self._latest_objects: list[TrackedObject] = []
        self._latest_counters: dict[str, Any] = {}
        self._latest_person_timer_states: dict[int, dict[str, Any]] = {}
        self._latest_theft_states: list[dict[str, Any]] = []
        self._latest_annotated_frame: Any | None = None
        self._latest_analysis_result_id = 0
        self._latest_analysis_capture_time = 0.0
        self._latest_analysis_frame_index = 0
        self._active_visual_alerts: list[dict[str, Any]] = []
        self._pose_fail_streak: dict[int, int] = {}
        self._pose_confirmed_person: set[int] = set()
        self._published_analysis_result_id = 0
        self._pending_alert_count = 0
        self._previous_track_ids: set[int] = set()
        self._last_fps_sample_at = 0.0
        self._low_fps_started_at: float | None = None
        self._last_stale_warning_result_id = 0

        pipeline_settings = settings.get("pipeline", {})
        self._frame_skip = max(1, int(pipeline_settings.get("frame_skip", 2)))
        self._ai_max_fps = max(0.0, float(pipeline_settings.get("ai_max_fps", 10)))
        self._analysis_timeout = max(
            1.0,
            float(pipeline_settings.get("analysis_timeout_min_seconds", 5.0)),
        )
        self._analysis_stale_after_ms = max(
            0.0,
            float(pipeline_settings.get("analysis_stale_after_ms", STALE_FRAME_WARNING_MS)),
        )
        self._reconnect_delay = float(pipeline_settings.get("reconnect_delay", 5))
        self._max_reconnect_attempts = int(pipeline_settings.get("max_reconnect_attempts", 10))
        self._processing_max_height = int(
            pipeline_settings.get(
                "processing_max_height",
                pipeline_settings.get("stream_max_height", 720),
            )
        )
        self._realtime_video_playback = bool(pipeline_settings.get("realtime_video_playback", True))
        self._loop_video_files = bool(pipeline_settings.get("loop_video_files", True))
        self._drop_late_video_frames = bool(pipeline_settings.get("drop_late_video_frames", True))

    # -- Thread-safe pipeline settings access ----------------------------------

    def _get_pipeline_params(self) -> dict[str, Any]:
        """Return a snapshot of mutable pipeline parameters under lock."""
        with self._config_lock:
            return {
                "frame_skip": self._frame_skip,
                "ai_max_fps": self._ai_max_fps,
                "analysis_timeout": self._analysis_timeout,
                "analysis_stale_after_ms": self._analysis_stale_after_ms,
                "reconnect_delay": self._reconnect_delay,
                "max_reconnect_attempts": self._max_reconnect_attempts,
                "processing_max_height": self._processing_max_height,
                "realtime_video_playback": self._realtime_video_playback,
                "loop_video_files": self._loop_video_files,
                "drop_late_video_frames": self._drop_late_video_frames,
            }

    def update_pipeline_settings(self, pipeline_settings: dict[str, Any]) -> None:
        """Atomically update mutable pipeline parameters from the web thread."""
        with self._config_lock:
            self._frame_skip = max(1, int(pipeline_settings.get("frame_skip", self._frame_skip)))
            self._ai_max_fps = max(0.0, float(pipeline_settings.get("ai_max_fps", self._ai_max_fps)))
            self._analysis_stale_after_ms = max(
                0.0,
                float(pipeline_settings.get("analysis_stale_after_ms", self._analysis_stale_after_ms)),
            )
            self._analysis_timeout = max(
                1.0,
                float(pipeline_settings.get("analysis_timeout_min_seconds", self._analysis_timeout)),
            )
            self._reconnect_delay = float(pipeline_settings.get("reconnect_delay", self._reconnect_delay))
            self._max_reconnect_attempts = int(pipeline_settings.get("max_reconnect_attempts", self._max_reconnect_attempts))
            self._processing_max_height = int(
                pipeline_settings.get(
                    "processing_max_height",
                    pipeline_settings.get("stream_max_height", self._processing_max_height),
                )
            )
            self._realtime_video_playback = bool(pipeline_settings.get("realtime_video_playback", self._realtime_video_playback))
            self._loop_video_files = bool(pipeline_settings.get("loop_video_files", self._loop_video_files))
            self._drop_late_video_frames = bool(pipeline_settings.get("drop_late_video_frames", self._drop_late_video_frames))

    @property
    def camera_id(self) -> str:
        """Return the camera id."""
        with self._config_lock:
            return str(self.camera_config.get("camera_id", "unknown"))

    def start(self) -> None:
        """Start the pipeline thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._analysis_ready.clear()
        with self._analysis_condition:
            self._ensure_analysis_worker_locked(self.camera_id)
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-pipeline-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the pipeline thread and wait for capture release."""
        self._stop_event.set()
        with self._analysis_condition:
            self._analysis_condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=min(1.0, timeout))
        self.frame_buffer.set_status("offline", "Pipeline stopped")

    def update_config(self, camera_config: dict[str, Any]) -> None:
        """Replace the camera configuration used by the next frame loop."""
        with self._config_lock:
            self.camera_config = copy.deepcopy(camera_config)

    def status(self) -> FrameSnapshot:
        """Return current frame buffer status."""
        return self.frame_buffer.snapshot()

    def _run(self) -> None:
        reconnect_attempts = 0
        frame_index = 0

        analysis_ready = getattr(self, "_analysis_ready", None)
        if analysis_ready is not None:
            self.frame_buffer.set_status("connecting", "Warming AI models")
            while not self._stop_event.is_set() and not analysis_ready.wait(timeout=0.1):
                pass

        while not self._stop_event.is_set():
            params = self._get_pipeline_params()
            config = self._get_config()
            camera_id = str(config.get("camera_id", "unknown"))
            camera_name = str(config.get("name", camera_id))
            source = self._parse_source(config.get("source", 0))

            self.frame_buffer.set_status("connecting")
            logger.info(
                "Opening camera %s from source %s",
                camera_id,
                _redact_source_for_log(source),
            )
            capture = self._open_capture(source)

            if not capture.isOpened():
                reconnect_attempts += 1
                max_attempts = params["max_reconnect_attempts"]
                error = f"Cannot open source (attempt {reconnect_attempts}/{max_attempts})"
                logger.warning("Camera %s: %s", camera_id, error)
                self.frame_buffer.set_status("offline", error)
                if reconnect_attempts >= max_attempts:
                    # Exponential backoff then reset counter for permanent retry.
                    backoff = min(
                        MAX_RECONNECT_BACKOFF_SECONDS,
                        params["reconnect_delay"] * (2 ** min(reconnect_attempts - max_attempts, 5)),
                    )
                    logger.warning(
                        "Camera %s exceeded reconnect limit (%d); backing off %.1fs before retrying",
                        camera_id,
                        max_attempts,
                        backoff,
                    )
                    self._sleep_interruptible(backoff)
                    reconnect_attempts = 0
                    continue
                self._sleep_interruptible(params["reconnect_delay"])
                continue

            reconnect_attempts = 0
            self.frame_buffer.set_status("connecting", "Waiting for first frame")
            self._reset_source_state(camera_id)
            frame_index = 0

            latest_capture: _LatestFrameCapture | None = None
            try:
                is_video_file = self._is_video_file_source(source)
                frame_interval = self._frame_interval(capture) if is_video_file and params["realtime_video_playback"] else 0.0
                next_frame_at = time.monotonic()
                last_capture_sequence = 0
                if self._is_rtsp_source(source):
                    latest_capture = _LatestFrameCapture(capture, self._stop_event)
                    latest_capture.start()
                while not self._stop_event.is_set():
                    captured_at = 0.0
                    capture_fps = 0.0
                    dropped_capture_frames = 0
                    if latest_capture is not None:
                        captured = latest_capture.read_latest(last_capture_sequence)
                        ok = captured is not None
                        frame = captured.frame if captured is not None else None
                        if captured is not None:
                            last_capture_sequence = captured.sequence
                            captured_at = captured.captured_at
                            capture_fps = captured.capture_fps
                            dropped_capture_frames = captured.dropped_frames
                    else:
                        ok, frame = capture.read()
                    if not ok or frame is None:
                        if is_video_file and params["loop_video_files"]:
                            self._reset_looping_video_capture(capture)
                            self._reset_source_state(camera_id)
                            frame_index = 0
                            next_frame_at = time.monotonic()
                            params = self._get_pipeline_params()
                            continue
                        logger.warning("Camera %s disconnected", camera_id)
                        self.frame_buffer.set_status("offline", "Camera disconnected")
                        break

                    config = self._get_config()
                    params = self._get_pipeline_params()
                    frame = self._apply_frame_rotation(frame, config)
                    frame = self._resize_for_processing(frame, params["processing_max_height"])
                    frame_index += 1
                    now = captured_at or time.monotonic()
                    self._submit_analysis_if_due(frame, config, camera_id, frame_index, now, params)
                    snapshot = self._analysis_snapshot()
                    display = self._display_frame_for_snapshot(
                        frame,
                        config,
                        snapshot,
                        now,
                        params["analysis_stale_after_ms"],
                    )
                    if (
                        display.stale_warning
                        and snapshot.result_id != self._last_stale_warning_result_id
                    ):
                        logger.warning(
                            (
                                "Stale analysis reused: camera_id=%s timestamp=%s "
                                "staleness_ms=%.0f analysis_frame_index=%d current_frame_index=%d"
                            ),
                            camera_id,
                            self._log_timestamp(),
                            display.staleness_ms,
                            snapshot.frame_index,
                            frame_index,
                        )
                        self._last_stale_warning_result_id = snapshot.result_id
                    self.frame_buffer.update(
                        display.frame,
                        object_count=display.object_count,
                        new_alert_count=snapshot.new_alert_count,
                        status="online",
                        staleness_ms=display.staleness_ms,
                        captured_at_monotonic=captured_at,
                        capture_fps=capture_fps,
                        dropped_capture_frames=dropped_capture_frames,
                    )
                    self._published_analysis_result_id = snapshot.result_id
                    self._monitor_fps_health(camera_id, now)

                    if frame_interval > 0:
                        next_frame_at += frame_interval
                        delay = next_frame_at - time.monotonic()
                        if delay > 0:
                            self._sleep_interruptible(delay)
                        elif delay < -frame_interval and params["drop_late_video_frames"]:
                            skipped = self._drop_late_frames(capture, -delay, frame_interval, params["loop_video_files"])
                            frame_index += skipped
                            next_frame_at += skipped * frame_interval
                            if next_frame_at < time.monotonic() - frame_interval:
                                next_frame_at = time.monotonic()
                        elif delay < -frame_interval:
                            next_frame_at = time.monotonic()
            finally:
                if latest_capture is not None:
                    latest_capture.stop()
                else:
                    capture.release()
                if not self._stop_event.is_set():
                    params = self._get_pipeline_params()
                    logger.info("Reconnecting camera %s after %.1fs", camera_name, params["reconnect_delay"])
                    self._sleep_interruptible(params["reconnect_delay"])

        self.frame_buffer.set_status("offline", "Pipeline exited")

    def _reset_source_state(self, camera_id: str) -> None:
        self.tracker.reset()
        self.behavior_engine.reset_camera(camera_id)
        self._reset_analysis_state()

    def _reset_analysis_state(self) -> None:
        with self._analysis_state_lock:
            self._analysis_generation += 1
            self._analysis_token += 1
            self._analysis_inflight = False
            self._last_analysis_started_at = 0.0
            self._analysis_timeout_warning_token = None
            self._latest_objects = []
            self._latest_counters = {}
            self._latest_person_timer_states = {}
            self._latest_theft_states = []
            self._latest_annotated_frame = None
            self._latest_analysis_result_id = 0
            self._latest_analysis_capture_time = 0.0
            self._latest_analysis_frame_index = 0
            self._active_visual_alerts = []
            self._pose_fail_streak = {}
            self._pose_confirmed_person = set()
            self._published_analysis_result_id = 0
            self._pending_alert_count = 0
            self._previous_track_ids = set()
            self._low_fps_started_at = None
            self._last_stale_warning_result_id = 0
            self._analysis_job = None
            self._analysis_condition.notify_all()

    def _submit_analysis_if_due(
        self,
        frame: Any,
        config: dict[str, Any],
        camera_id: str,
        frame_index: int,
        now: float,
        params: dict[str, Any] | None = None,
    ) -> bool:
        if params is None:
            params = self._get_pipeline_params()
        with self._analysis_state_lock:
            if self._analysis_inflight:
                elapsed = now - self._last_analysis_started_at
                if elapsed < params["analysis_timeout"]:
                    return False
                analysis_thread = getattr(self, "_analysis_thread", None)
                if analysis_thread is not None and analysis_thread.is_alive():
                    if getattr(self, "_analysis_timeout_warning_token", None) != self._analysis_token:
                        logger.warning(
                            (
                                "Camera %s: analysis thread still running after %.1fs; "
                                "keeping current analysis slot to avoid duplicate AI work"
                            ),
                            camera_id,
                            elapsed,
                        )
                        self._analysis_timeout_warning_token = self._analysis_token
                    return False
                logger.warning(
                    "Camera %s: analysis thread timed out after %.1fs - forcing reset",
                    camera_id,
                    elapsed,
                )
                self._analysis_inflight = False
                self._analysis_token += 1
            if not self._analysis_due(
                frame_index,
                params["frame_skip"],
                params["ai_max_fps"],
                now,
                self._last_analysis_started_at,
            ):
                return False
            self._ensure_analysis_worker_locked(camera_id)
            self._analysis_inflight = True
            self._last_analysis_started_at = now
            self._analysis_token += 1
            token = self._analysis_token
            generation = self._analysis_generation
            self._analysis_job = _AnalysisJob(
                frame.copy(),
                copy.deepcopy(config),
                camera_id,
                token,
                generation,
                now,
                frame_index,
            )
            self._analysis_condition.notify()
        return True

    def _ensure_analysis_worker_locked(self, camera_id: str) -> None:
        analysis_thread = getattr(self, "_analysis_thread", None)
        if analysis_thread is not None and analysis_thread.is_alive():
            return
        self._analysis_thread = threading.Thread(
            target=self._analysis_worker,
            name=f"camera-analysis-{camera_id}",
            daemon=True,
        )
        self._analysis_thread.start()

    def _analysis_worker(self) -> None:
        analysis_ready = getattr(self, "_analysis_ready", None)
        if analysis_ready is not None:
            try:
                self._warmup_analysis_models()
            finally:
                analysis_ready.set()
        while not self._stop_event.is_set():
            with self._analysis_condition:
                while self._analysis_job is None and not self._stop_event.is_set():
                    self._analysis_condition.wait(timeout=0.2)
                if self._analysis_job is None:
                    continue
                job = self._analysis_job
                self._analysis_job = None

            self._analyze_frame(
                job.frame,
                job.config,
                job.camera_id,
                job.token,
                job.generation,
                job.frame_capture_time,
                job.frame_index,
            )

    def _warmup_analysis_models(self) -> None:
        params = self._get_pipeline_params()
        height = int(params.get("processing_max_height", 720))
        height = height if height > 0 else 720
        rotation = str(self._get_config().get("frame_rotation", "none")).lower()
        width = round(height * (9 / 16 if rotation in {"cw90", "ccw90"} else 16 / 9))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        started = time.monotonic()
        for _ in range(2):
            try:
                self.detector.detect(frame)
            except Exception as exc:
                logger.warning("Detector warm-up failed; first live inference will retry: %s", exc)
            try:
                self.pose_estimator.warmup(frame)
            except Exception as exc:
                logger.warning("Pose warm-up failed; first live inference will retry: %s", exc)
        logger.info(
            "Analysis worker warmed for shape=%dx%d in %.1fs",
            width,
            height,
            time.monotonic() - started,
        )

    @staticmethod
    def _analysis_due(
        frame_index: int,
        frame_skip: int,
        ai_max_fps: float,
        now: float,
        last_analysis_started_at: float,
    ) -> bool:
        if frame_index % max(1, int(frame_skip)) != 0:
            return False
        if ai_max_fps <= 0 or last_analysis_started_at <= 0:
            return True
        return now - last_analysis_started_at >= 1.0 / ai_max_fps

    def _analysis_snapshot(self) -> _AnalysisSnapshot:
        with self._analysis_state_lock:
            self._prune_visual_alerts_locked(time.monotonic())
            pending_alert_count = self._pending_alert_count
            self._pending_alert_count = 0
            return _AnalysisSnapshot(
                objects=list(self._latest_objects),
                counters=copy.deepcopy(self._latest_counters),
                person_timer_states=copy.deepcopy(self._latest_person_timer_states),
                theft_states=copy.deepcopy(self._latest_theft_states),
                new_alert_count=pending_alert_count,
                annotated_frame=self._latest_annotated_frame,
                result_id=self._latest_analysis_result_id,
                frame_capture_time=self._latest_analysis_capture_time,
                frame_index=self._latest_analysis_frame_index,
                visual_alerts=copy.deepcopy(self._active_visual_alerts),
            )

    def _should_publish_snapshot(self, snapshot: _AnalysisSnapshot) -> bool:
        return (
            snapshot.annotated_frame is None
            or snapshot.result_id != self._published_analysis_result_id
        )

    @staticmethod
    def _display_frame_for_snapshot(
        frame: Any,
        config: dict[str, Any],
        snapshot: _AnalysisSnapshot,
        current_frame_capture_time: float,
        stale_after_ms: float,
    ) -> _DisplayFrame:
        has_analysis = snapshot.result_id > 0 and snapshot.frame_capture_time > 0
        staleness_ms = 0.0
        if has_analysis:
            staleness_ms = max(
                0.0,
                (current_frame_capture_time - snapshot.frame_capture_time) * 1000.0,
            )
        use_analysis = has_analysis
        objects = snapshot.objects if has_analysis else []
        annotated = draw_annotations(
            frame,
            objects,
            config,
            snapshot.counters,
            snapshot.person_timer_states,
            snapshot.visual_alerts,
            snapshot.theft_states,
        )
        return _DisplayFrame(
            frame=annotated,
            object_count=len(objects),
            staleness_ms=staleness_ms,
            used_analysis=use_analysis,
            stale_warning=has_analysis and staleness_ms > stale_after_ms,
        )

    def _analyze_frame(
        self,
        frame: Any,
        config: dict[str, Any],
        camera_id: str,
        token: int,
        generation: int,
        frame_capture_time: float,
        frame_index: int,
    ) -> None:
        try:
            behavior_engine = self.behavior_engine
            settings = self._get_settings()
            t0 = time.monotonic()
            objects = self.tracker.track(frame)
            t1 = time.monotonic()
            if not self._monitor_track_stability(
                camera_id,
                frame_index,
                objects,
                generation,
                token,
            ):
                return
            if self._pose_needed(config, settings):
                objects = self.pose_estimator.attach(frame, objects)
                # Filter out false-positive "person" detections.  Real people
                # always produce multiple confident keypoints (shoulders, hips).
                # Objects like motorcycles, chairs, and pictures produce zero.
                min_valid_kps = int(
                    settings.get("pose", {}).get("min_valid_keypoints", 3)
                )
                kp_conf = float(
                    settings.get("pose_classification", {}).get(
                        "min_keypoint_confidence", 0.25
                    )
                )
                pose_fail_grace_frames = int(
                    settings.get("pose", {}).get(
                        "false_person_filter_grace_frames",
                        POSE_FILTER_FAIL_GRACE_FRAMES,
                    )
                )
                objects = _filter_false_person_detections(
                    objects,
                    min_valid_kps,
                    kp_conf,
                    self._pose_fail_streak,
                    self._pose_confirmed_person,
                    pose_fail_grace_frames,
                )
                objects = self.tracker.suppress_ghost_animal_over_person(objects)
            t2 = time.monotonic()
            objects = behavior_engine.label_objects(objects, config, frame)
            t3 = time.monotonic()
            alerts = behavior_engine.analyze(objects, config, frame.shape)
            t4 = time.monotonic()
            track_ms = (t1 - t0) * 1000
            pose_ms = (t2 - t1) * 1000
            identity_ms = (t3 - t2) * 1000
            behavior_ms = (t4 - t3) * 1000
            total_ms = (t4 - t0) * 1000
            counters = behavior_engine.get_counters(camera_id)
            person_timer_states = behavior_engine.get_person_timer_states(camera_id)
            theft_states = behavior_engine.get_theft_states(camera_id)
            visual_alerts = self._visual_alerts_for_alerts(alerts, t4)
            logger.debug(
                (
                    "Analysis %s: track_ms=%.0f pose_ms=%.0f identity_ms=%.0f "
                    "behavior_ms=%.0f total_ms=%.0f objs=%d alerts=%d"
                ),
                camera_id,
                track_ms,
                pose_ms,
                identity_ms,
                behavior_ms,
                total_ms,
                len(objects),
                len(alerts),
            )

            with self._analysis_state_lock:
                if generation != self._analysis_generation or token != self._analysis_token:
                    return
                active_visual_alerts = self._merge_visual_alerts_locked(visual_alerts, t4)

            annotated = draw_annotations(
                frame,
                objects,
                config,
                counters,
                person_timer_states,
                active_visual_alerts,
                theft_states,
            )
            for alert in alerts:
                alert["notification_channels"] = list(config.get("notification_channels", ["telegram"]))
                alert["frame"] = annotated.copy()
                self.alert_manager.enqueue_threadsafe(alert)

            self.frame_buffer.set_ai_latency(total_ms)
            with self._analysis_state_lock:
                if generation != self._analysis_generation or token != self._analysis_token:
                    return
                self._latest_objects = objects
                self._latest_counters = counters
                self._latest_person_timer_states = person_timer_states
                self._latest_theft_states = theft_states
                self._latest_annotated_frame = annotated
                self._latest_analysis_capture_time = frame_capture_time
                self._latest_analysis_frame_index = frame_index
                self._latest_analysis_result_id += 1
                self._pending_alert_count += len(alerts)
            self._warn_if_analysis_slow(
                camera_id,
                total_ms,
                track_ms,
                pose_ms,
                identity_ms,
                behavior_ms,
            )
        except Exception as exc:
            logger.exception("Camera %s processing error: %s", camera_id, exc)
            self.frame_buffer.set_status("degraded", str(exc))
        finally:
            with self._analysis_state_lock:
                if self._analysis_token == token:
                    self._analysis_inflight = False

    @staticmethod
    def _visual_alerts_for_alerts(
        alerts: list[dict[str, Any]],
        now: float,
    ) -> list[dict[str, Any]]:
        visual_alerts: list[dict[str, Any]] = []
        for alert in alerts:
            item: dict[str, Any] = {
                "type": str(alert.get("type", "unknown")),
                "started_at": now,
                "expires_at": now + VISUAL_ALERT_SECONDS,
            }
            for key in ("track_id", "zone_id", "line_id"):
                value = alert.get(key)
                if value is not None:
                    item[key] = value
            if any(key in item for key in ("track_id", "zone_id", "line_id")):
                visual_alerts.append(item)
        return visual_alerts

    def _merge_visual_alerts_locked(
        self,
        alerts: list[dict[str, Any]],
        now: float,
    ) -> list[dict[str, Any]]:
        self._prune_visual_alerts_locked(now)
        self._active_visual_alerts.extend(alerts)
        if len(self._active_visual_alerts) > MAX_ACTIVE_VISUAL_ALERTS:
            self._active_visual_alerts = self._active_visual_alerts[-MAX_ACTIVE_VISUAL_ALERTS:]
        return copy.deepcopy(self._active_visual_alerts)

    def _prune_visual_alerts_locked(self, now: float) -> None:
        if not hasattr(self, "_active_visual_alerts"):
            self._active_visual_alerts = []
        self._active_visual_alerts = [
            alert
            for alert in self._active_visual_alerts
            if float(alert.get("expires_at", 0.0)) > now
        ]

    def _get_config(self) -> dict[str, Any]:
        with self._config_lock:
            return copy.deepcopy(self.camera_config)

    def _get_settings(self) -> dict[str, Any]:
        with self._config_lock:
            return copy.deepcopy(self._settings)

    @staticmethod
    def _parse_source(source: Any) -> int | str:
        if isinstance(source, int):
            return source
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return str(source)

    def _open_capture(self, source: int | str) -> cv2.VideoCapture:
        """Open a capture source with Windows-friendly backend fallbacks."""
        if not isinstance(source, int):
            if self._is_rtsp_source(source):
                capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    logger.info("Opened RTSP source with backend %s", self._backend_name(capture))
                return capture
            return cv2.VideoCapture(source)

        settings = self._get_settings()
        backend_name = str(
            settings.get("pipeline", {}).get("camera_backend", "msmf")
        ).lower()
        backend_map = {
            "any": cv2.CAP_ANY,
            "msmf": cv2.CAP_MSMF,
            "dshow": cv2.CAP_DSHOW,
        }
        preferred = backend_map.get(backend_name, cv2.CAP_MSMF)
        candidates = []
        for backend in (preferred, cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            if backend not in candidates:
                candidates.append(backend)

        for backend in candidates:
            capture = cv2.VideoCapture(source, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info(
                    "Opened camera source %s with backend %s",
                    source,
                    self._backend_name(capture),
                )
                return capture
            capture.release()
        return cv2.VideoCapture(source)

    def _resize_for_processing(self, frame: Any, processing_max_height: int = 0) -> Any:
        """Downscale oversized frames before detection, drawing, and streaming."""
        if processing_max_height <= 0:
            processing_max_height = self._get_pipeline_params()["processing_max_height"]
        if processing_max_height <= 0:
            return frame
        height, width = frame.shape[:2]
        if height <= processing_max_height:
            return frame
        scale = processing_max_height / float(height)
        size = (max(1, int(width * scale)), processing_max_height)
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _apply_frame_rotation(frame: Any, config: dict[str, Any]) -> Any:
        """Rotate a camera frame before AI and drawing when a camera is mounted sideways."""
        rotation = str(config.get("frame_rotation", "none")).strip().lower()
        if rotation == "cw90":
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == "ccw90":
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if rotation == "180":
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    @staticmethod
    def _is_video_file_source(source: int | str) -> bool:
        if isinstance(source, int):
            return False
        text = str(source)
        if text.lower().startswith(("rtsp://", "http://", "https://")):
            return False
        return Path(text).exists()

    @staticmethod
    def _is_rtsp_source(source: int | str) -> bool:
        return isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://"))

    @staticmethod
    def _pose_needed(config: dict[str, Any], settings: dict[str, Any]) -> bool:
        return bool(settings.get("pose", {}).get("enabled", True))

    @staticmethod
    def _frame_interval(capture: cv2.VideoCapture) -> float:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0 or fps > 120:
            return 0.0
        return 1.0 / fps

    def _drop_late_frames(
        self,
        capture: cv2.VideoCapture,
        late_by: float,
        frame_interval: float,
        loop_video_files: bool | None = None,
    ) -> int:
        if frame_interval <= 0:
            return 0
        if loop_video_files is None:
            loop_video_files = self._get_pipeline_params()["loop_video_files"]
        frames_to_drop = min(8, max(1, int(late_by / frame_interval)))
        dropped = 0
        for _ in range(frames_to_drop):
            if capture.grab():
                dropped += 1
                continue
            if loop_video_files:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        return dropped

    def _reset_looping_video_capture(self, capture: cv2.VideoCapture) -> None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        drained = self._drain_capture_buffer(capture, max_frames=1)
        if drained:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    @staticmethod
    def _drain_capture_buffer(capture: cv2.VideoCapture, max_frames: int = 1) -> int:
        drained = 0
        for _ in range(max(0, int(max_frames))):
            if not capture.grab():
                break
            drained += 1
        return drained

    def _monitor_fps_health(self, camera_id: str, now: float) -> None:
        if now - self._last_fps_sample_at < FPS_MONITOR_INTERVAL_SECONDS:
            return
        self._last_fps_sample_at = now
        snapshot = self.frame_buffer.snapshot()
        fps = float(snapshot.fps)
        capture_fps = float(snapshot.capture_fps)
        low_fps_threshold = max(
            LOW_FPS_THRESHOLD,
            capture_fps * LOW_FPS_SOURCE_RATIO if capture_fps > 0 else 0.0,
        )
        timestamp = self._log_timestamp()

        if fps > FPS_SPIKE_THRESHOLD:
            logger.warning(
                "FPS spike detected: camera_id=%s timestamp=%s fps=%.1f",
                camera_id,
                timestamp,
                fps,
            )

        if fps >= low_fps_threshold:
            self._low_fps_started_at = None
            return
        if self._low_fps_started_at is None:
            self._low_fps_started_at = now
            return
        if now - self._low_fps_started_at >= LOW_FPS_WARNING_SECONDS:
            logger.warning(
                (
                    "Low FPS detected: camera_id=%s timestamp=%s fps=%.1f "
                    "capture_fps=%.1f threshold_fps=%.1f dropped_frames=%d "
                    "duration_seconds=%.0f"
                ),
                camera_id,
                timestamp,
                fps,
                capture_fps,
                low_fps_threshold,
                snapshot.dropped_capture_frames,
                now - self._low_fps_started_at,
            )

    def _monitor_track_stability(
        self,
        camera_id: str,
        frame_index: int,
        objects: list[TrackedObject],
        generation: int,
        token: int,
    ) -> bool:
        current_track_ids = {obj.track_id for obj in objects}
        warning: tuple[list[int], int, int] | None = None
        with self._analysis_state_lock:
            if generation != self._analysis_generation or token != self._analysis_token:
                return False
            previous_track_ids = set(self._previous_track_ids)
            if previous_track_ids:
                lost_track_ids = sorted(previous_track_ids - current_track_ids)
                if len(lost_track_ids) > len(previous_track_ids) * 0.5:
                    warning = (
                        lost_track_ids,
                        len(previous_track_ids),
                        len(current_track_ids),
                    )
            self._previous_track_ids = current_track_ids

        if warning is not None:
            lost_track_ids, previous_count, current_count = warning
            logger.warning(
                (
                    "Track stability drop: camera_id=%s timestamp=%s "
                    "frame_index=%d lost_track_ids=%s previous_count=%d current_count=%d"
                ),
                camera_id,
                self._log_timestamp(),
                frame_index,
                lost_track_ids,
                previous_count,
                current_count,
            )
        return True

    def _warn_if_analysis_slow(
        self,
        camera_id: str,
        latency_ms: float,
        track_ms: float,
        pose_ms: float,
        identity_ms: float,
        behavior_ms: float,
    ) -> None:
        params = self._get_pipeline_params()
        ai_max_fps = params["ai_max_fps"]
        if ai_max_fps <= 0:
            return
        threshold_ms = 2.0 * (1000.0 / ai_max_fps)
        if latency_ms <= threshold_ms:
            return
        logger.warning(
            (
                "Analysis latency high: camera_id=%s timestamp=%s "
                "track_ms=%.0f pose_ms=%.0f identity_ms=%.0f behavior_ms=%.0f "
                "total_ms=%.0f threshold_ms=%.0f ai_max_fps=%.1f"
            ),
            camera_id,
            self._log_timestamp(),
            track_ms,
            pose_ms,
            identity_ms,
            behavior_ms,
            latency_ms,
            threshold_ms,
            ai_max_fps,
        )

    @staticmethod
    def _backend_name(capture: cv2.VideoCapture) -> str:
        try:
            return capture.getBackendName()
        except cv2.error:
            return "unknown"

    @staticmethod
    def _log_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _sleep_interruptible(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self._stop_event.is_set() and time.monotonic() < end:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.1, remaining))
