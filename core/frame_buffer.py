"""Thread-safe frame buffer for MJPEG streaming."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition
from time import time

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameSnapshot:
    """A lightweight status snapshot for a camera stream."""

    status: str
    object_count: int
    alert_count: int
    updated_at: float
    error: str | None
    version: int
    fps: float
    staleness_ms: float = 0.0
    ai_latency_ms: float = 0.0


class FrameBuffer:
    """Store the latest annotated frame and expose JPEG snapshots safely."""

    def __init__(self, max_height: int = 720, jpeg_quality: int = 82) -> None:
        self.max_height = max_height
        self.jpeg_quality = jpeg_quality
        self._condition = Condition()
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._version = 0
        self._status = "offline"
        self._object_count = 0
        self._alert_count = 0
        self._updated_at = 0.0
        self._error: str | None = None
        self._last_frame_at = 0.0
        self._fps = 0.0
        self._staleness_ms = 0.0
        self._ai_latency_ms = 0.0

    def update(
        self,
        frame: np.ndarray,
        object_count: int,
        new_alert_count: int = 0,
        status: str = "online",
        error: str | None = None,
        staleness_ms: float = 0.0,
    ) -> None:
        """Store a new frame and notify stream consumers."""
        jpeg = self._encode_jpeg(frame)
        now = time()
        with self._condition:
            self._frame = frame.copy()
            self._jpeg = jpeg
            self._version += 1
            self._status = status
            self._object_count = int(object_count)
            self._alert_count += int(new_alert_count)
            if self._last_frame_at > 0:
                delta = max(now - self._last_frame_at, 1e-6)
                instant_fps = 1.0 / delta
                self._fps = instant_fps if self._fps <= 0 else (self._fps * 0.85) + (instant_fps * 0.15)
            self._last_frame_at = now
            self._updated_at = now
            self._error = error
            self._staleness_ms = max(0.0, float(staleness_ms))
            self._condition.notify_all()

    def set_ai_latency(self, ms: float) -> None:
        """Store the latest end-to-end AI analysis latency in milliseconds."""
        with self._condition:
            self._ai_latency_ms = max(0.0, float(ms))

    def set_status(self, status: str, error: str | None = None) -> None:
        """Update stream status without changing the current frame."""
        with self._condition:
            self._status = status
            self._error = error
            self._staleness_ms = 0.0
            self._ai_latency_ms = 0.0
            self._updated_at = time()
            self._version += 1
            self._condition.notify_all()

    def snapshot(self) -> FrameSnapshot:
        """Return a thread-safe status snapshot."""
        with self._condition:
            return FrameSnapshot(
                status=self._status,
                object_count=self._object_count,
                alert_count=self._alert_count,
                updated_at=self._updated_at,
                error=self._error,
                version=self._version,
                fps=self._fps,
                staleness_ms=self._staleness_ms,
                ai_latency_ms=self._ai_latency_ms,
            )

    def wait_for_jpeg(
        self,
        last_version: int,
        timeout: float = 2.0,
        placeholder_text: str = "Waiting for camera",
    ) -> tuple[bytes, int]:
        """Wait for a new frame and return it as JPEG bytes."""
        with self._condition:
            if self._version == last_version:
                self._condition.wait(timeout=timeout)
            jpeg = self._jpeg
            version = self._version
            status = self._status
            error = self._error

        if jpeg is None:
            return self._placeholder_jpeg(placeholder_text, status, error), version
        return jpeg, version

    def latest_jpeg(self, placeholder_text: str = "Waiting for camera") -> bytes:
        """Return the latest frame as JPEG, or a generated placeholder image."""
        with self._condition:
            jpeg = self._jpeg
            status = self._status
            error = self._error
        if jpeg is None:
            return self._placeholder_jpeg(placeholder_text, status, error)
        return jpeg

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        resized = self._resize(frame)
        ok, encoded = cv2.imencode(
            ".jpg",
            resized,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return self._placeholder_jpeg("JPEG encode failed", "error", None)
        return encoded.tobytes()

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height <= self.max_height:
            return frame
        scale = self.max_height / float(height)
        new_size = (int(width * scale), self.max_height)
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def _placeholder_jpeg(self, text: str, status: str, error: str | None) -> bytes:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:] = (22, 24, 28)
        cv2.putText(image, text, (48, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (230, 235, 240), 2)
        cv2.putText(image, f"Status: {status}", (48, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 170, 190), 2)
        if error:
            cv2.putText(image, error[:90], (48, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 170, 255), 2)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return encoded.tobytes() if ok else b""
