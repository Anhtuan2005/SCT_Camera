"""Async Discord webhook sender for SCT alerts."""

from __future__ import annotations

import asyncio
from typing import Any

import cv2
import httpx
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class DiscordBot:
    """Send alert messages and optional images to a Discord webhook."""

    def __init__(self, settings: dict[str, Any]) -> None:
        discord = settings.get("discord", {})
        self.webhook_url = str(discord.get("webhook_url", "")).strip()
        self.enabled = bool(discord.get("enabled", False))
        self.username = str(discord.get("username", "SCT Camera")).strip() or "SCT Camera"
        self.max_retries = int(discord.get("max_retries", 3))
        self.timeout = httpx.Timeout(20.0, connect=10.0)

    async def send_alert(self, alert: dict[str, Any]) -> bool:
        """Send an alert to Discord with a snapshot when available."""
        content = self._build_message(alert)
        frame = alert.get("frame")
        if isinstance(frame, np.ndarray):
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if ok:
                return await self._request_with_retry(
                    data={"content": content, "username": self.username},
                    files={"file": ("alert.jpg", encoded.tobytes(), "image/jpeg")},
                )
        return await self.send_text(content)

    async def send_text(self, text: str) -> bool:
        """Send a plain Discord webhook message."""
        return await self._request_with_retry(
            data={"content": text, "username": self.username},
            files=None,
        )

    async def _request_with_retry(
        self,
        data: dict[str, Any],
        files: dict[str, tuple[str, bytes, str]] | None,
    ) -> bool:
        if not self.enabled:
            logger.info("Discord disabled; alert not sent")
            return False
        if not self.webhook_url:
            logger.warning("Discord webhook URL missing; alert not sent")
            return False

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.webhook_url, data=data, files=files)
                    response.raise_for_status()
                    return True
            except Exception as exc:
                logger.warning(
                    "Discord webhook failed on attempt %s/%s: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            await asyncio.sleep(min(2 ** attempt, 10))
        return False

    def _build_message(self, alert: dict[str, Any]) -> str:
        alert_type = str(alert.get("type", "alert")).replace("_", " ").upper()
        camera_name = str(alert.get("camera_name", alert.get("camera_id", "Camera")))
        class_name = str(alert.get("class_name", "object"))
        identity_label = str(alert.get("identity_label", "")).strip()
        object_label = identity_label if identity_label and class_name == "person" else class_name
        track_id = alert.get("track_id", "-")
        timestamp = str(alert.get("timestamp", ""))
        place = str(
            alert.get("zone_name")
            or alert.get("line_name")
            or alert.get("zone_id")
            or alert.get("line_id")
            or "-"
        )
        details = str(alert.get("details", ""))

        return (
            f"**[{alert_type}]** - {camera_name}\n"
            f"Time: {timestamp}\n"
            f"Object: {object_label} (Track #{track_id})\n"
            f"Zone/Line: {place}\n"
            f"Details: {details}"
        )
