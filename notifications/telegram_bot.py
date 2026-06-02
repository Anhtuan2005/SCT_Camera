"""Async Telegram sender using raw Bot API HTTP calls."""

from __future__ import annotations

import asyncio
from typing import Any

import cv2
import httpx
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class TelegramBot:
    """Send alert photos and test messages to Telegram asynchronously."""

    def __init__(self, settings: dict[str, Any]) -> None:
        telegram = settings.get("telegram", {})
        self.bot_token = str(telegram.get("bot_token", "")).strip()
        self.chat_id = str(telegram.get("chat_id", "")).strip()
        self.enabled = bool(telegram.get("enabled", True))
        self.max_retries = int(telegram.get("max_retries", 3))
        self.timeout = httpx.Timeout(20.0, connect=10.0)

    async def send_alert(self, alert: dict[str, Any]) -> bool:
        """Send an annotated alert image with a detailed caption."""
        caption = self._build_caption(alert)
        frame = alert.get("frame")
        if isinstance(frame, np.ndarray):
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if ok:
                return await self._request_with_retry(
                    "sendPhoto",
                    data={
                        "chat_id": self.chat_id,
                        "caption": caption,
                        "parse_mode": "Markdown",
                    },
                    files={"photo": ("alert.jpg", encoded.tobytes(), "image/jpeg")},
                )
        return await self.send_text(caption)

    async def send_text(self, text: str) -> bool:
        """Send a plain Telegram text message."""
        return await self._request_with_retry(
            "sendMessage",
            data={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            },
            files=None,
        )

    async def _request_with_retry(
        self,
        method: str,
        data: dict[str, Any],
        files: dict[str, tuple[str, bytes, str]] | None,
    ) -> bool:
        if not self.enabled:
            logger.info("Telegram disabled; alert not sent")
            return False
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot token or chat id missing; alert not sent")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, data=data, files=files)
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("ok") is True:
                        return True
                    logger.warning("Telegram API returned non-ok payload: %s", payload)
            except Exception as exc:
                logger.warning(
                    "Telegram %s failed on attempt %s/%s: %s",
                    method,
                    attempt,
                    self.max_retries,
                    exc,
                )
            await asyncio.sleep(min(2 ** attempt, 10))
        return False

    def _build_caption(self, alert: dict[str, Any]) -> str:
        alert_type = str(alert.get("type", "alert")).replace("_", " ").upper()
        camera_name = self._escape(str(alert.get("camera_name", alert.get("camera_id", "Camera"))))
        class_name = str(alert.get("class_name", "object"))
        identity_label = str(alert.get("identity_label", "")).strip()
        object_label = identity_label if identity_label and class_name == "person" else class_name
        class_name = self._escape(object_label)
        track_id = alert.get("track_id", "-")
        timestamp = self._escape(str(alert.get("timestamp", "")))
        place = self._escape(
            str(
                alert.get("zone_name")
                or alert.get("line_name")
                or alert.get("zone_id")
                or alert.get("line_id")
                or "-"
            )
        )
        details = self._escape(str(alert.get("details", "")))

        return (
            "──────────────────\n"
            f"🚨 **[{alert_type}]** — {camera_name}\n"
            f"📅 Time: {timestamp}\n"
            f"🎯 Object: {class_name} (Track #{track_id})\n"
            f"📍 Zone/Line: {place}\n"
            f"📊 Details: {details}\n"
            "──────────────────"
        )

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("_", "\\_").replace("*", "\\*").replace("[", "\\[").replace("]", "\\]")
