"""Optional local siren trigger for high-severity alerts."""

from __future__ import annotations

import asyncio
import platform
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


class SirenController:
    """Trigger a local beep or configured command when an alert requires a siren."""

    def __init__(self, settings: dict[str, Any]) -> None:
        siren = settings.get("siren", {})
        self.enabled = bool(siren.get("enabled", False))
        self.mode = str(siren.get("mode", "beep")).lower()
        self.command = str(siren.get("command", "")).strip()
        self.duration_seconds = float(siren.get("duration_seconds", 5))
        self.frequency_hz = int(siren.get("frequency_hz", 1600))
        self.cooldown_seconds = float(siren.get("cooldown_seconds", 30))
        self.timeout_seconds = float(siren.get("timeout_seconds", 10))
        self._last_trigger_at = 0.0

    async def trigger(self, alert: dict[str, Any]) -> bool:
        """Trigger the configured siren once, respecting cooldown."""
        if not self.enabled:
            return False

        loop = asyncio.get_running_loop()
        now = loop.time()
        if now - self._last_trigger_at < self.cooldown_seconds:
            logger.info("Siren suppressed by cooldown for alert %s", alert.get("type"))
            return False

        self._last_trigger_at = now
        if self.mode == "command" and self.command:
            return await self._run_command()
        return await asyncio.to_thread(self._beep)

    async def _run_command(self) -> bool:
        try:
            process = await asyncio.create_subprocess_shell(
                self.command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=self.timeout_seconds)
            if process.returncode == 0:
                logger.info("Siren command completed")
                return True
            logger.warning("Siren command exited with code %s", process.returncode)
        except Exception as exc:
            logger.warning("Siren command failed: %s", exc)
        return False

    def _beep(self) -> bool:
        try:
            if platform.system().lower() == "windows":
                import winsound

                remaining_ms = max(1, int(self.duration_seconds * 1000))
                chunk_ms = min(700, remaining_ms)
                while remaining_ms > 0:
                    winsound.Beep(self.frequency_hz, min(chunk_ms, remaining_ms))
                    remaining_ms -= chunk_ms
                return True
            print("\a", end="", flush=True)
            return True
        except Exception as exc:
            logger.warning("Siren beep failed: %s", exc)
            return False
