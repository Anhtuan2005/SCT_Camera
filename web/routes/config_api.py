"""REST configuration and alert API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api")


class CameraPayload(BaseModel):
    """Request payload for camera create/update."""

    camera_id: str | None = None
    name: str
    source: str | int
    enabled: bool = True
    frame_rotation: str | None = Field(default=None, pattern="^(none|cw90|ccw90|180)$")
    unknown_person_policy: str | None = Field(
        default=None,
        pattern="^(face_match|assume_stranger|unknown_by_default|all_unknown)$",
    )
    notification_channels: list[str] | None = None


class CameraEnabledPayload(BaseModel):
    """Request payload for toggling camera availability."""

    enabled: bool = True


class ZonePayload(BaseModel):
    """Request payload for polygon zone create/update."""

    id: str | None = None
    name: str
    type: str = Field(pattern="^(all|intrusion|loitering|counting|stranger_watch|asset_watch)$")
    polygon: list[list[float]]
    threshold_seconds: float | None = None


class LinePayload(BaseModel):
    """Request payload for counting line create/update."""

    id: str | None = None
    name: str
    point1: list[float]
    point2: list[float]
    direction: str = "forward"


class BehaviorEventLabelPayload(BaseModel):
    """Request payload for labeling a behavior-learning event."""

    label: str
    notes: str = ""


@router.get("/cameras")
async def list_cameras(request: Request) -> list[dict[str, Any]]:
    """List all camera configs with runtime status."""
    return request.app.state.runtime.list_cameras()


@router.post("/cameras")
async def upsert_camera(request: Request, payload: CameraPayload) -> dict[str, Any]:
    """Create or update a camera."""
    return request.app.state.runtime.upsert_camera(payload.model_dump(exclude_none=True))


@router.delete("/cameras/{cam_id}")
async def delete_camera(request: Request, cam_id: str) -> dict[str, bool]:
    """Delete a camera."""
    deleted = request.app.state.runtime.delete_camera(cam_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"deleted": True}


@router.post("/cameras/{cam_id}/enabled")
async def set_camera_enabled(request: Request, cam_id: str, payload: CameraEnabledPayload) -> dict[str, Any]:
    """Enable or disable a camera and start/stop its pipeline."""
    camera = request.app.state.runtime.set_camera_enabled(cam_id, payload.enabled)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.get("/cameras/{cam_id}/zones")
async def get_zones(request: Request, cam_id: str) -> list[dict[str, Any]]:
    """Return all zones for a camera."""
    camera = request.app.state.runtime.get_raw_camera(cam_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera.get("zones", [])


@router.post("/cameras/{cam_id}/zones")
async def upsert_zone(request: Request, cam_id: str, payload: ZonePayload) -> dict[str, Any]:
    """Create or update a zone."""
    zone = request.app.state.runtime.upsert_zone(cam_id, payload.model_dump(exclude_none=True))
    if zone is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return zone


@router.delete("/cameras/{cam_id}/zones/{zone_id}")
async def delete_zone(request: Request, cam_id: str, zone_id: str) -> dict[str, bool]:
    """Delete a zone."""
    deleted = request.app.state.runtime.delete_zone(cam_id, zone_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"deleted": True}


@router.get("/cameras/{cam_id}/lines")
async def get_lines(request: Request, cam_id: str) -> list[dict[str, Any]]:
    """Return all counting lines for a camera."""
    camera = request.app.state.runtime.get_raw_camera(cam_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera.get("lines", [])


@router.post("/cameras/{cam_id}/lines")
async def upsert_line(request: Request, cam_id: str, payload: LinePayload) -> dict[str, Any]:
    """Create or update a counting line."""
    line = request.app.state.runtime.upsert_line(cam_id, payload.model_dump(exclude_none=True))
    if line is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return line


@router.delete("/cameras/{cam_id}/lines/{line_id}")
async def delete_line(request: Request, cam_id: str, line_id: str) -> dict[str, bool]:
    """Delete a counting line."""
    deleted = request.app.state.runtime.delete_line(cam_id, line_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Line not found")
    return {"deleted": True}


@router.post("/settings/telegram/test")
async def test_telegram(request: Request) -> dict[str, bool]:
    """Send a Telegram test message."""
    sent = await request.app.state.runtime.alert_manager.send_test_message()
    return {"sent": sent}


@router.post("/settings/discord/test")
async def test_discord(request: Request) -> dict[str, bool]:
    """Send a Discord test message."""
    sent = await request.app.state.runtime.alert_manager.send_discord_test_message()
    return {"sent": sent}


@router.put("/settings")
async def update_settings(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    """Update global settings."""
    return request.app.state.runtime.update_settings(payload)


@router.post("/detection/toggle/{cam_id}")
async def toggle_detection(request: Request, cam_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Toggle detection on/off for a single camera."""
    active = bool(payload.get("active", True))
    result = request.app.state.runtime.toggle_detection(cam_id, active)
    if result is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return result


@router.post("/detection/toggle-all")
async def toggle_all_detection(request: Request, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Toggle detection on/off for all enabled cameras."""
    active = bool(payload.get("active", True))
    return request.app.state.runtime.toggle_all_detection(active)


@router.get("/alerts/{cam_id}")
async def get_alerts(
    request: Request,
    cam_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    """Return recent camera alerts."""
    if request.app.state.runtime.get_camera(cam_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return request.app.state.runtime.get_alerts(cam_id, limit=limit, offset=offset)


@router.get("/behavior-events")
async def list_behavior_events(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    unlabeled_only: bool = False,
) -> list[dict[str, Any]]:
    """Return recent behavior-learning candidate events."""
    return request.app.state.runtime.list_behavior_events(limit=limit, unlabeled_only=unlabeled_only)


@router.post("/behavior-events/{event_id}/label")
async def label_behavior_event(
    request: Request,
    event_id: str,
    payload: BehaviorEventLabelPayload,
) -> dict[str, Any]:
    """Save a supervised label for a behavior-learning event."""
    try:
        return request.app.state.runtime.label_behavior_event(event_id, payload.label, payload.notes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
