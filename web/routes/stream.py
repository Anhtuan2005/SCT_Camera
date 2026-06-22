"""MJPEG streaming endpoints."""

from __future__ import annotations

import threading
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from utils.logger import get_logger

router = APIRouter(prefix="/api")
logger = get_logger(__name__)

_MAX_CONCURRENT_STREAMS = 10
_active_streams = 0
_streams_lock = threading.Lock()


def _acquire_stream_slot() -> bool:
    """Try to acquire a stream slot. Returns False when at capacity."""
    global _active_streams
    with _streams_lock:
        if _active_streams >= _MAX_CONCURRENT_STREAMS:
            return False
        _active_streams += 1
        return True


def _release_stream_slot() -> None:
    """Release a stream slot."""
    global _active_streams
    with _streams_lock:
        _active_streams = max(0, _active_streams - 1)


@router.get("/stream/{cam_id}")
def camera_stream(request: Request, cam_id: str) -> StreamingResponse:
    """Return an MJPEG stream for a camera."""
    runtime = request.app.state.runtime
    buffer = runtime.frame_buffer(cam_id)
    camera = runtime.get_camera(cam_id)
    if buffer is None or camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not _acquire_stream_slot():
        raise HTTPException(
            status_code=503,
            detail=f"Maximum concurrent streams reached ({_MAX_CONCURRENT_STREAMS})",
        )

    def frames() -> Iterator[bytes]:
        try:
            last_version = -1
            camera_name = str(camera.get("name", cam_id))
            while True:
                jpeg, last_version = buffer.wait_for_jpeg(
                    last_version=last_version,
                    timeout=2.0,
                    placeholder_text=f"{camera_name} stream offline",
                )
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        except GeneratorExit:
            pass
        finally:
            _release_stream_slot()
            logger.debug("MJPEG stream closed for camera %s", cam_id)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

