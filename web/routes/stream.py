"""MJPEG streaming endpoints."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api")


@router.get("/stream/{cam_id}")
def camera_stream(request: Request, cam_id: str) -> StreamingResponse:
    """Return an MJPEG stream for a camera."""
    runtime = request.app.state.runtime
    buffer = runtime.frame_buffer(cam_id)
    camera = runtime.get_camera(cam_id)
    if buffer is None or camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    def frames() -> Iterator[bytes]:
        last_version = -1
        camera_name = str(camera.get("name", cam_id))
        while True:
            jpeg, last_version = buffer.wait_for_jpeg(
                last_version=last_version,
                timeout=2.0,
                placeholder_text=f"{camera_name} stream offline",
            )
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
