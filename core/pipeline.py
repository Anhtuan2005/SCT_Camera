"""Camera pipeline orchestrator."""

from __future__ import annotations

import copy
import os
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp",
)

import cv2

from analytics.behavior_engine import BehaviorEngine
from core.detector import YOLOv11Detector
from core.frame_buffer import FrameBuffer, FrameSnapshot
from core.pose import PoseEstimator
from core.tracker import ByteTrackTracker, TrackedObject
from utils.drawing import draw_annotations
from utils.logger import get_logger

logger = get_logger(__name__)


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
        self.settings = settings
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.behavior_engine = behavior_engine
        self.frame_buffer = frame_buffer
        self.alert_manager = alert_manager
        self.tracker = ByteTrackTracker(detector, settings)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        pipeline_settings = settings.get("pipeline", {})
        self.frame_skip = max(1, int(pipeline_settings.get("frame_skip", 2)))
        self.reconnect_delay = float(pipeline_settings.get("reconnect_delay", 5))
        self.max_reconnect_attempts = int(pipeline_settings.get("max_reconnect_attempts", 10))
        self.processing_max_height = int(
            pipeline_settings.get(
                "processing_max_height",
                pipeline_settings.get("stream_max_height", 720),
            )
        )
        self.realtime_video_playback = bool(pipeline_settings.get("realtime_video_playback", True))
        self.loop_video_files = bool(pipeline_settings.get("loop_video_files", True))
        self.drop_late_video_frames = bool(pipeline_settings.get("drop_late_video_frames", True))

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
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-pipeline-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the pipeline thread and wait for capture release."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
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
        last_objects: list[TrackedObject] = []
        frame_index = 0

        while not self._stop_event.is_set():
            config = self._get_config()
            camera_id = str(config.get("camera_id", "unknown"))
            camera_name = str(config.get("name", camera_id))
            source = self._parse_source(config.get("source", 0))

            self.frame_buffer.set_status("connecting")
            logger.info("Opening camera %s from source %s", camera_id, source)
            capture = self._open_capture(source)

            if not capture.isOpened():
                reconnect_attempts += 1
                error = f"Cannot open source (attempt {reconnect_attempts}/{self.max_reconnect_attempts})"
                logger.warning("Camera %s: %s", camera_id, error)
                self.frame_buffer.set_status("offline", error)
                if reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error("Camera %s exceeded reconnect limit", camera_id)
                    break
                self._sleep_interruptible(self.reconnect_delay)
                continue

            reconnect_attempts = 0
            self.frame_buffer.set_status("connecting", "Waiting for first frame")

            try:
                is_video_file = self._is_video_file_source(source)
                frame_interval = self._frame_interval(capture) if is_video_file and self.realtime_video_playback else 0.0
                next_frame_at = time.monotonic()
                while not self._stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        if is_video_file and self.loop_video_files:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            last_objects = []
                            frame_index = 0
                            next_frame_at = time.monotonic()
                            continue
                        logger.warning("Camera %s disconnected", camera_id)
                        self.frame_buffer.set_status("offline", "Camera disconnected")
                        break

                    frame = self._resize_for_processing(frame)
                    frame_index += 1
                    config = self._get_config()
                    process_now = frame_index % self.frame_skip == 0
                    alerts: list[dict[str, Any]] = []

                    if process_now:
                        try:
                            last_objects = self.tracker.track(frame)
                            last_objects = self.pose_estimator.attach(frame, last_objects)
                            last_objects = self.behavior_engine.label_objects(
                                last_objects,
                                config,
                                frame,
                            )
                            alerts = self.behavior_engine.analyze(
                                last_objects,
                                config,
                                frame.shape,
                            )
                        except Exception as exc:
                            logger.exception("Camera %s processing error: %s", camera_id, exc)
                            self.frame_buffer.set_status("degraded", str(exc))

                    counters = self.behavior_engine.get_counters(camera_id)
                    stranger_watch_states = self.behavior_engine.get_stranger_watch_states(camera_id)
                    annotated = draw_annotations(
                        frame,
                        last_objects,
                        config,
                        counters,
                        stranger_watch_states,
                    )
                    for alert in alerts:
                        alert["notification_channels"] = list(config.get("notification_channels", ["telegram"]))
                        alert["frame"] = annotated.copy()
                        self.alert_manager.enqueue_threadsafe(alert)

                    self.frame_buffer.update(
                        annotated,
                        object_count=len(last_objects),
                        new_alert_count=len(alerts),
                        status="online",
                    )

                    if frame_interval > 0:
                        next_frame_at += frame_interval
                        delay = next_frame_at - time.monotonic()
                        if delay > 0:
                            self._sleep_interruptible(delay)
                        elif delay < -frame_interval and self.drop_late_video_frames:
                            skipped = self._drop_late_frames(capture, -delay, frame_interval)
                            frame_index += skipped
                            next_frame_at += skipped * frame_interval
                            if next_frame_at < time.monotonic() - frame_interval:
                                next_frame_at = time.monotonic()
                        elif delay < -frame_interval:
                            next_frame_at = time.monotonic()
            finally:
                capture.release()
                if not self._stop_event.is_set():
                    logger.info("Reconnecting camera %s after %.1fs", camera_name, self.reconnect_delay)
                    self._sleep_interruptible(self.reconnect_delay)

        self.frame_buffer.set_status("offline", "Pipeline exited")

    def _get_config(self) -> dict[str, Any]:
        with self._config_lock:
            return copy.deepcopy(self.camera_config)

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
            if str(source).lower().startswith("rtsp://"):
                capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    logger.info("Opened RTSP source with backend %s", self._backend_name(capture))
                return capture
            return cv2.VideoCapture(source)

        backend_name = str(
            self.settings.get("pipeline", {}).get("camera_backend", "msmf")
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

    def _resize_for_processing(self, frame: Any) -> Any:
        """Downscale oversized frames before detection, drawing, and streaming."""
        if self.processing_max_height <= 0:
            return frame
        height, width = frame.shape[:2]
        if height <= self.processing_max_height:
            return frame
        scale = self.processing_max_height / float(height)
        size = (max(1, int(width * scale)), self.processing_max_height)
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _is_video_file_source(source: int | str) -> bool:
        if isinstance(source, int):
            return False
        text = str(source)
        if text.lower().startswith(("rtsp://", "http://", "https://")):
            return False
        return Path(text).exists()

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
    ) -> int:
        if frame_interval <= 0:
            return 0
        frames_to_drop = min(8, max(1, int(late_by / frame_interval)))
        dropped = 0
        for _ in range(frames_to_drop):
            if capture.grab():
                dropped += 1
                continue
            if self.loop_video_files:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        return dropped

    @staticmethod
    def _backend_name(capture: cv2.VideoCapture) -> str:
        try:
            return capture.getBackendName()
        except cv2.error:
            return "unknown"

    def _sleep_interruptible(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self._stop_event.is_set() and time.monotonic() < end:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.1, remaining))
