"""Shared dwell-threshold policy and session timing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Any, Hashable


@dataclass(frozen=True)
class EscalationTier:
    after_seconds: float
    alert_type: str
    siren: bool = False
    channels: list[str] = field(default_factory=lambda: ["telegram"])


@dataclass(frozen=True)
class TimeWindow:
    start: dt_time
    end: dt_time
    multiplier: float

    def contains(self, now: dt_time) -> bool:
        if self.start <= self.end:
            return self.start <= now < self.end
        return now >= self.start or now < self.end


@dataclass(frozen=True)
class DwellPolicy:
    default_threshold_seconds: float
    escalation_tiers: list[EscalationTier] = field(default_factory=list)
    identity_multipliers: dict[str, float | None] = field(default_factory=dict)
    time_windows: list[TimeWindow] = field(default_factory=list)
    session_gap_seconds: float = 0.0

    @classmethod
    def from_zone_config(
        cls,
        data: dict[str, Any],
        default_threshold: float,
    ) -> "DwellPolicy":
        tiers = [
            EscalationTier(
                after_seconds=float(tier["after_seconds"]),
                alert_type=str(tier.get("alert_type", "loitering")),
                siren=bool(tier.get("siren", False)),
                channels=list(tier.get("channels", ["telegram"])),
            )
            for tier in sorted(
                data.get("escalation_tiers", []) or [],
                key=lambda item: item["after_seconds"],
            )
        ]
        identity_multipliers = {
            str(key): (None if value is None else float(value))
            for key, value in (data.get("identity_multipliers", {}) or {}).items()
        }
        windows = [
            TimeWindow(
                start=_parse_hhmm(window["start"]),
                end=_parse_hhmm(window["end"]),
                multiplier=float(window["multiplier"]),
            )
            for window in data.get("time_of_day_multipliers", []) or []
        ]
        return cls(
            default_threshold_seconds=float(
                data.get("threshold_seconds") or default_threshold
            ),
            escalation_tiers=tiers,
            identity_multipliers=identity_multipliers,
            time_windows=windows,
            session_gap_seconds=max(0.0, float(data.get("session_gap_seconds", 0.0))),
        )

    def is_excluded(self, identity_kind: str | None) -> bool:
        return (
            identity_kind in self.identity_multipliers
            and self.identity_multipliers[identity_kind] is None
        )

    def scale_factor(
        self,
        identity_kind: str | None,
        now: datetime | None = None,
    ) -> float:
        factor = 1.0
        identity_multiplier = self.identity_multipliers.get(identity_kind)
        if identity_multiplier is not None:
            factor *= identity_multiplier

        current_time = (now or datetime.now().astimezone()).time()
        for window in self.time_windows:
            if window.contains(current_time):
                factor *= window.multiplier
                break
        return factor

    def effective_threshold(
        self,
        identity_kind: str | None,
        now: datetime | None = None,
    ) -> float:
        return self.default_threshold_seconds * self.scale_factor(identity_kind, now)

    def effective_tiers(
        self,
        identity_kind: str | None,
        now: datetime | None = None,
    ) -> list[EscalationTier]:
        factor = self.scale_factor(identity_kind, now)
        return [
            EscalationTier(
                after_seconds=tier.after_seconds * factor,
                alert_type=tier.alert_type,
                siren=tier.siren,
                channels=tier.channels,
            )
            for tier in self.escalation_tiers
        ]


@dataclass
class DwellSession:
    total_duration: float
    last_seen: float
    active: bool = True


def update_dwell_session(
    sessions: dict[Hashable, DwellSession],
    key: Hashable,
    now: float,
    state_grace_seconds: float,
    session_gap_seconds: float,
) -> float:
    state = sessions.get(key)
    if state is None:
        sessions[key] = DwellSession(total_duration=0.0, last_seen=now)
        return 0.0

    gap = max(0.0, now - state.last_seen)
    if state.active or gap <= state_grace_seconds:
        state.total_duration += gap
    elif session_gap_seconds <= 0 or gap > session_gap_seconds:
        state.total_duration = 0.0

    state.last_seen = now
    state.active = True
    return state.total_duration


def current_dwell_duration(
    state: DwellSession,
    now: float,
    state_grace_seconds: float,
) -> float:
    gap = max(0.0, now - state.last_seen)
    if state.active or gap <= state_grace_seconds:
        return state.total_duration + gap
    return state.total_duration


def should_purge_missing_session(
    state: DwellSession,
    now: float,
    state_grace_seconds: float,
    session_gap_seconds: float,
) -> bool:
    state.active = False
    gap = max(0.0, now - state.last_seen)
    if gap <= state_grace_seconds:
        return False
    return session_gap_seconds <= 0 or gap > session_gap_seconds


def _parse_hhmm(value: str) -> dt_time:
    hour, minute = value.split(":")
    return dt_time(int(hour), int(minute))
