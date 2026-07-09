"""Authentication routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from web.auth import (
    AUTH_COOKIE_NAME,
    get_auth_settings,
    make_session_token,
    safe_next_url,
    verify_credentials,
)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str | None = None) -> HTMLResponse:
    """Render the login page."""
    next_url = safe_next_url(next)
    if getattr(request.state, "authenticated", False):
        return RedirectResponse(next_url, status_code=303)
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "page_id": "login",
            "next_url": next_url,
            "error": None,
        },
    )


@router.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request) -> HTMLResponse:
    """Validate credentials and issue a signed session cookie."""
    settings = get_auth_settings(request)
    form = await request.form()
    username = str(form.get("username") or "")
    password = str(form.get("password") or "")
    next_url = safe_next_url(str(form.get("next") or "/"))
    if not verify_credentials(username, password, settings):
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "page_id": "login",
                "next_url": next_url,
                "error": "Sai username hoặc password",
            },
            status_code=401,
        )

    response = RedirectResponse(next_url, status_code=303)
    response.set_cookie(
        AUTH_COOKIE_NAME,
        make_session_token(settings.username, settings.session_secret, settings.max_age_seconds),
        max_age=settings.max_age_seconds,
        httponly=True,
        secure=settings.secure_cookie,
        samesite="lax",
    )
    return response


@router.post("/logout")
async def logout() -> RedirectResponse:
    """Clear the session cookie and return to login."""
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(AUTH_COOKIE_NAME)
    return response
