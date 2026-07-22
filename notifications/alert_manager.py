"""Alert cooldown, deduplication, queueing, and history management."""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

from core.database import DatabaseManager
from notifications.discord_bot import DiscordBot
from notifications.siren import SirenController
from notifications.telegram_bot import TelegramBot
from utils.logger import get_logger

logger = get_logger(__name__)

_QUEUE_HIGH_WATER = 800
_QUEUE_MAX_SIZE = 1000


class AlertManager:
    """Manage alert cooldowns, delivery, and history."""

    def __init__(
        self,
        settings: dict[str, Any],
        database: DatabaseManager | None = None,
    ) -> None:
        telegram = settings.get("telegram", {})
        self.cooldown_seconds = float(telegram.get("cooldown_seconds", 10))
        self.cooldown_overrides = self._parse_cooldown_overrides(telegram)
        self.bot = TelegramBot(settings)
        self.discord = DiscordBot(settings)
        self.siren = SirenController(settings)
        self.queue: asyncio.Queue[dict[str, Any]] | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_sent_at: dict[tuple[str, str, str], float] = {}
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=200))
        self._history_lock = threading.Lock()
        self.database = database

    async def start(self) -> None:
        """Start the async alert worker."""
        if self._running:
            return
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
        self._running = True
        self._task = asyncio.create_task(self._worker(), name="alert-worker")
        logger.info("Alert manager started")

    async def stop(self) -> None:
        """Stop the async alert worker and close HTTP clients."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Drain any remaining alerts so they are recorded in history.
        if self.queue is not None:
            while not self.queue.empty():
                try:
                    alert = self.queue.get_nowait()
                    await self._handle_alert(alert)
                except asyncio.QueueEmpty:
                    break
                except Exception as exc:
                    logger.warning("Error draining alert queue: %s", exc)
        # Close persistent HTTP clients.
        await self.bot.close()
        await self.discord.close()
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
                if self.queue.qsize() >= _QUEUE_HIGH_WATER:
                    logger.warning(
                        "Alert queue high water mark reached: %d/%d",
                        self.queue.qsize(),
                        _QUEUE_MAX_SIZE,
                    )
            except asyncio.QueueFull:
                logger.error(
                    "Alert queue full (%d); dropping %s alert for camera %s",
                    _QUEUE_MAX_SIZE,
                    alert.get("type"),
                    alert.get("camera_id"),
                )

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

    async def send_discord_test_message(self) -> bool:
        """Send a Discord test message using current settings."""
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return await self.discord.send_text(f"SCT Camera test alert\nTime: {now}")

    def get_recent(self, camera_id: str, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Return recent alert history for one camera."""
        if self.database is not None:
            try:
                return self.database.get_recent_alerts(camera_id, limit=limit, offset=offset)
            except Exception as exc:
                logger.warning("Could not read alert history from SQLite: %s", exc)
        with self._history_lock:
            items = list(self._history.get(camera_id, deque()))
        items.reverse()
        return items[offset : offset + limit]

    def update_settings(self, settings: dict[str, Any]) -> None:
        """Apply new alert settings without restarting FastAPI."""
        telegram = settings.get("telegram", {})
        self.cooldown_seconds = float(telegram.get("cooldown_seconds", self.cooldown_seconds))
        self.cooldown_overrides = self._parse_cooldown_overrides(telegram, fallback=self.cooldown_overrides)
        self.bot = TelegramBot(settings)
        self.discord = DiscordBot(settings)
        self.siren = SirenController(settings)

    async def _worker(self) -> None:
        assert self.queue is not None
        while self._running:
            try:
                alert = await self.queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self._handle_alert(alert)
            except Exception as exc:
                logger.exception("Alert worker failed: %s", exc)
            finally:
                self.queue.task_done()

    async def _handle_alert(self, alert: dict[str, Any]) -> None:
        record = self._history_record(alert)
        cooldown_key = self._cooldown_key(alert)
        alert_type = str(alert.get("type", "alert"))
        cooldown_seconds = self.cooldown_overrides.get(alert_type, self.cooldown_seconds)
        now = asyncio.get_running_loop().time()
        last_sent = self._last_sent_at.get(cooldown_key, 0.0)
        cooldown_remaining = cooldown_seconds - (now - last_sent)
        channels = self._channels_for_alert(alert)

        if cooldown_remaining > 0:
            record["suppressed"] = True
            record["sent"] = False
            record["telegram_sent"] = False
            record["discord_sent"] = False
            record["siren_triggered"] = False
            record["message"] = f"Cooldown active ({cooldown_remaining:.1f}s remaining)"
            logger.info(
                "Alert suppressed by cooldown: camera=%s type=%s target=%s",
                cooldown_key[0],
                cooldown_key[1],
                cooldown_key[2],
            )
        else:
            telegram_sent, discord_sent = await asyncio.gather(
                self.bot.send_alert(alert) if "telegram" in channels else asyncio.sleep(0, result=False),
                self.discord.send_alert(alert) if "discord" in channels else asyncio.sleep(0, result=False),
            )
            siren_triggered = await self.siren.trigger(alert) if bool(alert.get("siren")) else False
            record["suppressed"] = False
            record["sent"] = telegram_sent or discord_sent
            record["telegram_sent"] = telegram_sent
            record["discord_sent"] = discord_sent
            record["siren_triggered"] = siren_triggered
            sent_channels = [
                name
                for name, delivered in (
                    ("Telegram", telegram_sent),
                    ("Discord", discord_sent),
                )
                if delivered
            ]
            record["message"] = (
                f"Sent to {', '.join(sent_channels)}"
                if sent_channels
                else "Alert send skipped or failed"
            )
            self._last_sent_at[cooldown_key] = now
            logger.info(
                "Alert processed: camera=%s type=%s target=%s telegram=%s discord=%s",
                cooldown_key[0],
                cooldown_key[1],
                cooldown_key[2],
                telegram_sent,
                discord_sent,
            )

        self._append_history(record)
        await self._persist_record(record, channels)

    def _append_history(self, record: dict[str, Any]) -> None:
        camera_id = str(record.get("camera_id", "unknown"))
        with self._history_lock:
            self._history[camera_id].append(record)

    async def _persist_record(
        self,
        record: dict[str, Any],
        channels: set[str],
    ) -> None:
        if self.database is None:
            return
        deliveries = []
        for channel in sorted(channels):
            if record.get("suppressed"):
                status = "suppressed"
            elif record.get(f"{channel}_sent"):
                status = "sent"
            else:
                status = "failed"
            deliveries.append(
                {
                    "channel": channel,
                    "status": status,
                    "error": "Delivery returned false" if status == "failed" else None,
                }
            )
        try:
            await asyncio.to_thread(self.database.persist_alert, record, deliveries)
        except Exception as exc:
            logger.exception("Could not persist alert %s to SQLite: %s", record.get("event_id"), exc)

    @staticmethod
    def _channels_for_alert(alert: dict[str, Any]) -> set[str]:
        raw_channels = alert.get("notification_channels", ["telegram"])
        if isinstance(raw_channels, str):
            items = [raw_channels]
        elif isinstance(raw_channels, list):
            items = raw_channels
        else:
            items = ["telegram"]
        channels = {str(item).strip().lower() for item in items}
        channels &= {"telegram", "discord"}
        return channels or {"telegram"}

    @staticmethod
    def _cooldown_key(alert: dict[str, Any]) -> tuple[str, str, str]:
        if alert.get("type") == "intrusion" and alert.get("line_id") and alert.get("track_id") is not None:
            return (
                str(alert.get("camera_id", "unknown")),
                "intrusion",
                f"{alert.get('line_id')}:track:{alert.get('track_id')}",
            )
        target = str(
            alert.get("zone_id")
            or alert.get("line_id")
            or alert.get("zone_name")
            or alert.get("line_name")
            or (
                f"track:{alert['track_id']}"
                if alert.get("track_id") is not None
                else None
            )
            or "global"
        )
        return str(alert.get("camera_id", "unknown")), str(alert.get("type", "alert")), target

    @staticmethod
    def _parse_cooldown_overrides(
        telegram: dict[str, Any],
        fallback: dict[str, float] | None = None,
    ) -> dict[str, float]:
        raw = telegram.get("cooldown_overrides", fallback or {})
        if not isinstance(raw, dict):
            return dict(fallback or {})
        parsed: dict[str, float] = {}
        for alert_type, seconds in raw.items():
            try:
                parsed[str(alert_type)] = max(0.0, float(seconds))
            except (TypeError, ValueError):
                continue
        return parsed

    @staticmethod
    def _history_record(alert: dict[str, Any]) -> dict[str, Any]:
        record = {
            key: value
            for key, value in alert.items()
            if key not in {"frame"} and not key.startswith("_")
        }
        record["event_id"] = str(
            record.get("event_id")
            or record.get("behavior_event_id")
            or uuid.uuid4()
        )
        record["received_at"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return record
