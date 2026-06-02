"""HTML dashboard routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    """Render the multi-camera dashboard."""
    runtime = request.app.state.runtime
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "page_id": "dashboard", "cameras": runtime.list_cameras()},
    )


@router.get("/camera/{cam_id}", response_class=HTMLResponse)
async def camera_detail_page(request: Request, cam_id: str) -> HTMLResponse:
    """Render a single camera detail page with editors."""
    runtime = request.app.state.runtime
    camera = runtime.get_camera(cam_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return templates.TemplateResponse(
        "camera_detail.html",
        {"request": request, "page_id": "camera", "camera": camera, "cam_id": cam_id},
    )


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    """Render global settings and camera management."""
    runtime = request.app.state.runtime
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "page_id": "settings",
            "settings": runtime.settings,
            "cameras": runtime.list_cameras(),
        },
    )
