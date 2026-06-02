"""Alert cooldown, deduplication, queueing, and history management."""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

from notifications.siren import SirenController
from notifications.telegram_bot import TelegramBot
from utils.logger import get_logger

logger = get_logger(__name__)


class AlertManager:
    """Manage alert cooldowns and async Telegram delivery."""

    def __init__(self, settings: dict[str, Any]) -> None:
        telegram = settings.get("telegram", {})
        self.cooldown_seconds = float(telegram.get("cooldown_seconds", 10))
        self.bot = TelegramBot(settings)
        self.siren = SirenController(settings)
        self.queue: asyncio.Queue[dict[str, Any]] | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_sent_at: dict[tuple[str, str, str], float] = {}
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=200))
        self._history_lock = threading.Lock()

    async def start(self) -> None:
        """Start the async alert worker."""
        if self._running:
            return
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue(maxsize=1000)
        self._running = True
        self._task = asyncio.create_task(self._worker(), name="telegram-alert-worker")
        logger.info("Alert manager started")

    async def stop(self) -> None:
        """Stop the async alert worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Alert manager stopped")

    def enqueue_threadsafe(self, alert: dict[str, Any]) -> None:
        """Enqueue an alert from a pipeline thread."""
        if self.loop is None or self.queue is None:
            logger.warning("Alert manager not running; dropping alert %s", alert.get("type"))
            return

        def put_alert() -> None:
            if self.queue is None:
                return
            try:
                self.queue.put_nowait(alert)
            except asyncio.QueueFull:
                logger.error("Alert queue full; dropping alert %s", alert.get("type"))

        self.loop.call_soon_threadsafe(put_alert)

    async def enqueue(self, alert: dict[str, Any]) -> None:
        """Enqueue an alert from async code."""
        if self.queue is None:
            logger.warning("Alert manager not running; dropping alert %s", alert.get("type"))
            return
        await self.queue.put(alert)

    async def send_test_message(self) -> bool:
        """Send a Telegram test message using current settings."""
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return await self.bot.send_text(f"✅ SCT Camera test alert\nTime: {now}")

    def get_recent(self, camera_id: str, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Return recent alert history for one camera."""
        with self._history_lock:
            items = list(self._history.get(camera_id, deque()))
        items.reverse()
        return items[offset : offset + limit]

    def update_settings(self, settings: dict[str, Any]) -> None:
        """Apply new Telegram settings without restarting FastAPI."""
        telegram = settings.get("telegram", {})
        self.cooldown_seconds = float(telegram.get("cooldown_seconds", self.cooldown_seconds))
        self.bot = TelegramBot(settings)
        self.siren = SirenController(settings)

    async def _worker(self) -> None:
        assert self.queue is not None
        while self._running:
            alert = await self.queue.get()
            try:
                await self._handle_alert(alert)
            except Exception as exc:
                logger.exception("Alert worker failed: %s", exc)
            finally:
                self.queue.task_done()

    async def _handle_alert(self, alert: dict[str, Any]) -> None:
        record = self._history_record(alert)
        cooldown_key = self._cooldown_key(alert)
        now = asyncio.get_running_loop().time()
        last_sent = self._last_sent_at.get(cooldown_key, 0.0)
        cooldown_remaining = self.cooldown_seconds - (now - last_sent)

        if cooldown_remaining > 0:
            record["suppressed"] = True
            record["sent"] = False
            record["message"] = f"Cooldown active ({cooldown_remaining:.1f}s remaining)"
            logger.info(
                "Alert suppressed by cooldown: camera=%s type=%s target=%s",
                cooldown_key[0],
                cooldown_key[1],
                cooldown_key[2],
            )
        else:
            sent = await self.bot.send_alert(alert)
            siren_triggered = await self.siren.trigger(alert) if bool(alert.get("siren")) else False
            record["suppressed"] = False
            record["sent"] = sent
            record["siren_triggered"] = siren_triggered
            record["message"] = "Sent to Telegram" if sent else "Telegram send skipped or failed"
            self._last_sent_at[cooldown_key] = now
            logger.info(
                "Alert processed: camera=%s type=%s target=%s sent=%s",
                cooldown_key[0],
                cooldown_key[1],
                cooldown_key[2],
                sent,
            )

        self._append_history(record)

    def _append_history(self, record: dict[str, Any]) -> None:
        camera_id = str(record.get("camera_id", "unknown"))
        with self._history_lock:
            self._history[camera_id].append(record)

    @staticmethod
    def _cooldown_key(alert: dict[str, Any]) -> tuple[str, str, str]:
        target = str(
            alert.get("zone_id")
            or alert.get("line_id")
            or alert.get("zone_name")
            or alert.get("line_name")
            or "global"
        )
        return str(alert.get("camera_id", "unknown")), str(alert.get("type", "alert")), target

    @staticmethod
    def _history_record(alert: dict[str, Any]) -> dict[str, Any]:
        record = {
            key: value
            for key, value in alert.items()
            if key not in {"frame"} and not key.startswith("_")
        }
        record["received_at"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return record
