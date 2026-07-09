"""Session authentication helpers for the dashboard."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse, Response

AUTH_COOKIE_NAME = "sct_camera_session"
DEFAULT_AUTH_MAX_AGE_SECONDS = 8 * 60 * 60
DEFAULT_PASSWORD_ENV = "SCT_CAMERA_PASSWORD"
DEFAULT_USERNAME_ENV = "SCT_CAMERA_USERNAME"
DEFAULT_SECRET_ENV = "SCT_CAMERA_SESSION_SECRET"
PUBLIC_PATHS = {"/login", "/logout", "/api/health", "/favicon.ico"}
PUBLIC_PREFIXES = ("/static/",)


@dataclass(frozen=True)
class AuthSettings:
    enabled: bool
    username: str
    password: str | None
    password_sha256: str | None
    session_secret: str
    max_age_seconds: int
    secure_cookie: bool


class AuthMiddleware(BaseHTTPMiddleware):
    """Require a valid signed session cookie before dashboard/API access."""

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_auth_settings(request)
        username = current_username(request, settings)
        request.state.authenticated = username is not None
        request.state.auth_user = username

        if not settings.enabled or _is_public_path(request.url.path):
            return await call_next(request)
        if username is not None:
            return await call_next(request)
        if _expects_html(request):
            next_url = quote(
                request.url.path + (f"?{request.url.query}" if request.url.query else ""),
                safe="",
            )
            return RedirectResponse(f"/login?next={next_url}", status_code=303)
        return JSONResponse({"detail": "Authentication required"}, status_code=401)


def auth_secret_from_settings(settings: dict[str, Any]) -> str:
    """Return configured auth secret or generate a per-process fallback."""
    auth = _auth_section(settings)
    env_name = str(auth.get("session_secret_env") or DEFAULT_SECRET_ENV)
    secret = os.getenv(env_name) or auth.get("session_secret")
    if secret:
        return str(secret)
    return secrets.token_urlsafe(32)


def get_auth_settings(request: Request) -> AuthSettings:
    runtime_settings = getattr(request.app.state.runtime, "settings", {})
    auth = _auth_section(runtime_settings)
    password_env = str(auth.get("password_env") or DEFAULT_PASSWORD_ENV)
    username_env = str(auth.get("username_env") or DEFAULT_USERNAME_ENV)
    password = os.getenv(password_env) or auth.get("password")
    return AuthSettings(
        enabled=bool(auth.get("enabled", False)),
        username=str(os.getenv(username_env) or auth.get("username") or "admin"),
        password=str(password) if password is not None else None,
        password_sha256=(
            str(auth.get("password_sha256"))
            if auth.get("password_sha256") is not None
            else None
        ),
        session_secret=str(request.app.state.auth_secret),
        max_age_seconds=max(
            60,
            int(auth.get("session_max_age_seconds", DEFAULT_AUTH_MAX_AGE_SECONDS)),
        ),
        secure_cookie=bool(auth.get("secure_cookie", False)),
    )


def current_username(request: Request, settings: AuthSettings | None = None) -> str | None:
    settings = settings or get_auth_settings(request)
    token = request.cookies.get(AUTH_COOKIE_NAME)
    if not token:
        return None
    return verify_session_token(token, settings.session_secret)


def verify_credentials(username: str, password: str, settings: AuthSettings) -> bool:
    if not hmac.compare_digest(username, settings.username):
        return False
    if settings.password_sha256:
        digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, settings.password_sha256)
    if settings.password is None:
        return False
    return hmac.compare_digest(password, settings.password)


def make_session_token(username: str, secret: str, max_age_seconds: int) -> str:
    payload = {
        "sub": username,
        "exp": int(time.time()) + max_age_seconds,
        "nonce": secrets.token_urlsafe(12),
    }
    encoded_payload = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _sign(encoded_payload, secret)
    return f"{encoded_payload}.{signature}"


def verify_session_token(token: str, secret: str) -> str | None:
    try:
        encoded_payload, signature = token.split(".", 1)
    except ValueError:
        return None
    expected = _sign(encoded_payload, secret)
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_unb64(encoded_payload))
    except (ValueError, json.JSONDecodeError):
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    username = payload.get("sub")
    return str(username) if username else None


def safe_next_url(value: str | None, default: str = "/") -> str:
    if not value:
        return default
    if not value.startswith("/") or value.startswith("//"):
        return default
    return value


def _auth_section(settings: dict[str, Any]) -> dict[str, Any]:
    web = settings.get("web", {})
    auth = web.get("auth", {}) if isinstance(web, dict) else {}
    return auth if isinstance(auth, dict) else {"enabled": bool(auth)}


def _is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS or any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)


def _expects_html(request: Request) -> bool:
    if request.method == "GET" and not request.url.path.startswith("/api/"):
        return True
    accept = request.headers.get("accept", "")
    return "text/html" in accept


def _sign(encoded_payload: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256).digest()
    return _b64(digest)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))
