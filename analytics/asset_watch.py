"""Asset watch rule for possible property removal."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from math import hypot
from typing import Any

from analytics.identity_status import is_confirmed_stranger
from analytics.zone import Zone
from core.tracker import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _AssetState:
    class_name: str
    track_id: int
    first_seen: float
    last_seen_inside: float
    last_center: tuple[float, float]
    last_bbox: tuple[float, float, float, float]
    confidence: float
    last_person_near: float = 0.0
    person_track_id: int | None = None
    person_label: str = ""
    outside_since: float | None = None
    inside_streak_start: float | None = None
    alerted: bool = False


class AssetWatchDetector:
    """Alert when a watched asset disappears or leaves a guarded zone near a person."""

    def __init__(
        self,
        default_missing_seconds: float = 6.0,
        settings: dict[str, Any] | None = None,
    ) -> None:
        settings = settings or {}
        self.default_missing_seconds = default_missing_seconds
        self.asset_classes = {
            str(item)
            for item in settings.get(
                "asset_classes",
                ["bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase"],
            )
        }
        self.person_window_seconds = float(settings.get("person_window_seconds", 12))
        self.min_presence_seconds = float(settings.get("min_presence_seconds", 2))
        self.interaction_distance_ratio = float(settings.get("interaction_distance_ratio", 0.22))
        self.reappear_confirm_seconds = max(
            0.0,
            float(settings.get("reappear_confirm_seconds", 1.0)),
        )
        self.alert_repeat_seconds = max(
            0.0,
            float(settings.get("alert_repeat_seconds", settings.get("repeat_window_seconds", 300))),
        )
        self.cleanup_seconds = float(settings.get("cleanup_seconds", 90))
        self._assets: dict[tuple[str, str, int], _AssetState] = {}
        self._alerted_assets: dict[tuple[str, str, str, int], float] = {}

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        zones: list[Zone],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return possible asset-removal alerts for asset-watch zones."""
        watch_zones = [zone for zone in zones if zone.applies_to("asset_watch")]
        if not watch_zones:
            watch_zones = [_global_asset_watch_zone()]

        now = time.monotonic()
        alerts: list[dict[str, Any]] = []
        asset_objects = [obj for obj in objects if obj.class_name in self.asset_classes]
        person_objects = [obj for obj in objects if self._is_unknown_person(obj)]
        objects_by_track = {obj.track_id: obj for obj in asset_objects}

        for zone in watch_zones:
            zone_key = (camera_id, zone.id)
            missing_seconds = zone.threshold_seconds or self.default_missing_seconds
            people_in_zone = [
                obj
                for obj in person_objects
                if zone.contains_point(obj.center[0], obj.center[1], frame_shape)
            ]
            inside_asset_ids = self._update_inside_assets(
                camera_id,
                zone,
                asset_objects,
                people_in_zone,
                frame_shape,
                now,
            )
            self._update_recent_people(zone_key, people_in_zone, frame_shape, now)
            alerts.extend(
                self._moved_out_alerts(
                    camera_id,
                    camera_name,
                    zone,
                    objects_by_track,
                    inside_asset_ids,
                    frame_shape,
                    timestamp,
                    now,
                    missing_seconds,
                )
            )
            alerts.extend(
                self._missing_alerts(
                    camera_id,
                    camera_name,
                    zone,
                    inside_asset_ids,
                    asset_objects,
                    frame_shape,
                    timestamp,
                    now,
                    missing_seconds,
                )
            )
            self._cleanup_zone(camera_id, zone.id, now)

        return alerts

    def _update_inside_assets(
        self,
        camera_id: str,
        zone: Zone,
        asset_objects: list[TrackedObject],
        people_in_zone: list[TrackedObject],
        frame_shape: tuple[int, int, int],
        now: float,
    ) -> set[int]:
        inside_asset_ids: set[int] = set()
        for asset in asset_objects:
            if not zone.contains_point(asset.center[0], asset.center[1], frame_shape):
                continue
            inside_asset_ids.add(asset.track_id)
            key = (camera_id, zone.id, asset.track_id)
            state = self._assets.get(key)
            if state is None:
                state = _AssetState(
                    class_name=asset.class_name,
                    track_id=asset.track_id,
                    first_seen=now,
                    last_seen_inside=now,
                    last_center=asset.center,
                    last_bbox=asset.bbox_xyxy,
                    confidence=asset.confidence,
                    inside_streak_start=now,
                )
                self._assets[key] = state
            else:
                state.class_name = asset.class_name
                state.confidence = asset.confidence
                if state.inside_streak_start is None:
                    state.inside_streak_start = now
                if now - state.inside_streak_start >= self.reappear_confirm_seconds:
                    state.last_seen_inside = now
                    state.last_center = asset.center
                    state.last_bbox = asset.bbox_xyxy
                    state.outside_since = None

            person = self._nearest_interacting_person(asset, people_in_zone, frame_shape)
            if person is not None:
                self._mark_person_near(state, person, now)
        for key, state in self._assets.items():
            if key[:2] == (camera_id, zone.id) and key[2] not in inside_asset_ids:
                state.inside_streak_start = None
        return inside_asset_ids

    def _update_recent_people(
        self,
        zone_key: tuple[str, str],
        people_in_zone: list[TrackedObject],
        frame_shape: tuple[int, int, int],
        now: float,
    ) -> None:
        if not people_in_zone:
            return
        for key, state in self._assets.items():
            if key[:2] != zone_key:
                continue
            if now - state.last_seen_inside > self.person_window_seconds:
                continue
            person = self._nearest_person_to_bbox(state.last_bbox, people_in_zone, frame_shape)
            if person is not None:
                self._mark_person_near(state, person, now)

    def _moved_out_alerts(
        self,
        camera_id: str,
        camera_name: str,
        zone: Zone,
        objects_by_track: dict[int, TrackedObject],
        inside_asset_ids: set[int],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
        now: float,
        missing_seconds: float,
    ) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        for key, state in list(self._assets.items()):
            if key[:2] != (camera_id, zone.id):
                self._log_gate("moved_out_wrong_zone", now, state, zone_id=zone.id)
                continue
            if key[2] in inside_asset_ids:
                self._log_gate("moved_out_asset_inside", now, state, zone_id=zone.id)
                continue
            asset = objects_by_track.get(state.track_id)
            if asset is None:
                self._log_gate("moved_out_asset_not_tracked", now, state, zone_id=zone.id)
                continue
            if zone.contains_point(asset.center[0], asset.center[1], frame_shape):
                self._log_gate("moved_out_asset_still_in_zone", now, state, zone_id=zone.id)
                continue
            if state.outside_since is None:
                state.outside_since = now
            outside_seconds = now - state.outside_since
            if outside_seconds < missing_seconds:
                self._log_gate(
                    "moved_out_waiting_threshold",
                    now,
                    state,
                    zone_id=zone.id,
                    elapsed_seconds=round(outside_seconds, 3),
                    threshold_seconds=missing_seconds,
                )
                continue
            alert = self._alert_if_ready(
                camera_id,
                camera_name,
                zone,
                state,
                timestamp,
                now,
                "asset_removed",
                f"{state.class_name} moved out of watched zone",
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _missing_alerts(
        self,
        camera_id: str,
        camera_name: str,
        zone: Zone,
        inside_asset_ids: set[int],
        asset_objects: list[TrackedObject],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
        now: float,
        missing_seconds: float,
    ) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        for key, state in list(self._assets.items()):
            if key[:2] != (camera_id, zone.id):
                self._log_gate("missing_wrong_zone", now, state, zone_id=zone.id)
                continue
            if state.track_id in inside_asset_ids:
                self._log_gate("missing_asset_inside", now, state, zone_id=zone.id)
                continue
            missing_elapsed = now - state.last_seen_inside
            if missing_elapsed < missing_seconds:
                self._log_gate(
                    "missing_waiting_threshold",
                    now,
                    state,
                    zone_id=zone.id,
                    elapsed_seconds=round(missing_elapsed, 3),
                    threshold_seconds=missing_seconds,
                )
                continue
            if self._same_asset_still_present(state, asset_objects, zone, frame_shape, now):
                self._log_gate("missing_same_asset_still_present", now, state, zone_id=zone.id)
                continue
            alert = self._alert_if_ready(
                camera_id,
                camera_name,
                zone,
                state,
                timestamp,
                now,
                "asset_missing",
                f"{state.class_name} disappeared from watched zone",
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _alert_if_ready(
        self,
        camera_id: str,
        camera_name: str,
        zone: Zone,
        state: _AssetState,
        timestamp: datetime,
        now: float,
        alert_type: str,
        message: str,
    ) -> dict[str, Any] | None:
        if state.alerted:
            self._log_gate("alert_already_sent", now, state, zone_id=zone.id)
            return None
        self._cleanup_alerted_assets(now)
        if self._recent_asset_alert_exists(
            camera_id,
            zone.id,
            state.class_name,
            state.track_id,
            now,
        ):
            self._log_gate("alert_recent_duplicate", now, state, zone_id=zone.id)
            return None
        presence_seconds = now - state.first_seen
        if presence_seconds < self.min_presence_seconds:
            self._log_gate(
                "alert_presence_too_short",
                now,
                state,
                zone_id=zone.id,
                elapsed_seconds=round(presence_seconds, 3),
                threshold_seconds=self.min_presence_seconds,
            )
            return None
        person_elapsed = now - state.last_person_near
        if person_elapsed > self.person_window_seconds:
            self._log_gate(
                "alert_person_window_expired",
                now,
                state,
                zone_id=zone.id,
                elapsed_seconds=round(person_elapsed, 3),
                threshold_seconds=self.person_window_seconds,
            )
            return None

        state.alerted = True
        asset_signature = (camera_id, zone.id, state.class_name, state.track_id)
        self._alerted_assets[asset_signature] = now
        actor = state.person_label or "unknown person"
        return {
            "type": alert_type,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "track_id": state.track_id,
            "class_name": state.class_name,
            "zone_id": zone.id,
            "zone_name": zone.name,
            "actor_track_id": state.person_track_id,
            "actor_label": actor,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "siren": True,
            "details": f"{message} after {actor} was nearby",
        }

    def _same_asset_still_present(
        self,
        state: _AssetState,
        asset_objects: list[TrackedObject],
        zone: Zone,
        frame_shape: tuple[int, int, int],
        now: float,
    ) -> bool:
        max_distance = self._frame_diagonal(frame_shape) * 0.12
        for asset in asset_objects:
            if asset.class_name != state.class_name:
                self._log_gate(
                    "same_asset_class_mismatch",
                    now,
                    state,
                    zone_id=zone.id,
                    candidate_track_id=asset.track_id,
                    candidate_class=asset.class_name,
                )
                continue
            if not zone.contains_point(asset.center[0], asset.center[1], frame_shape):
                self._log_gate(
                    "same_asset_outside_zone",
                    now,
                    state,
                    zone_id=zone.id,
                    candidate_track_id=asset.track_id,
                )
                continue
            distance = hypot(
                asset.center[0] - state.last_center[0],
                asset.center[1] - state.last_center[1],
            )
            if distance > max_distance:
                self._log_gate(
                    "same_asset_distance_exceeded",
                    now,
                    state,
                    zone_id=zone.id,
                    candidate_track_id=asset.track_id,
                    distance=round(distance, 3),
                    max_distance=round(max_distance, 3),
                )
                continue
            streak_seconds = (
                now - state.inside_streak_start
                if state.inside_streak_start is not None
                else 0.0
            )
            if (
                state.inside_streak_start is not None
                and streak_seconds >= self.reappear_confirm_seconds
            ):
                self._log_gate(
                    "same_asset_reappear_confirmed",
                    now,
                    state,
                    zone_id=zone.id,
                    candidate_track_id=asset.track_id,
                    elapsed_seconds=round(streak_seconds, 3),
                    threshold_seconds=self.reappear_confirm_seconds,
                )
                return True
            self._log_gate(
                "same_asset_reappear_unconfirmed",
                now,
                state,
                zone_id=zone.id,
                candidate_track_id=asset.track_id,
                elapsed_seconds=round(streak_seconds, 3),
                threshold_seconds=self.reappear_confirm_seconds,
            )
        self._log_gate("same_asset_absent", now, state, zone_id=zone.id)
        return False

    def _nearest_interacting_person(
        self,
        asset: TrackedObject,
        people: list[TrackedObject],
        frame_shape: tuple[int, int, int],
    ) -> TrackedObject | None:
        return self._nearest_person_to_bbox(asset.bbox_xyxy, people, frame_shape)

    def _nearest_person_to_bbox(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        people: list[TrackedObject],
        frame_shape: tuple[int, int, int],
    ) -> TrackedObject | None:
        max_distance = self._frame_diagonal(frame_shape) * self.interaction_distance_ratio
        nearest: TrackedObject | None = None
        nearest_distance = max_distance
        for person in people:
            distance = self._point_to_bbox_distance(person.center, bbox_xyxy)
            if distance <= nearest_distance:
                nearest = person
                nearest_distance = distance
        return nearest

    @staticmethod
    def _point_to_bbox_distance(
        point: tuple[float, float],
        bbox_xyxy: tuple[float, float, float, float],
    ) -> float:
        x, y = point
        x1, y1, x2, y2 = bbox_xyxy
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        return hypot(dx, dy)

    @staticmethod
    def _mark_person_near(state: _AssetState, person: TrackedObject, now: float) -> None:
        state.last_person_near = now
        state.person_track_id = person.track_id
        state.person_label = person.identity_label or person.class_name

    @staticmethod
    def _is_unknown_person(obj: TrackedObject) -> bool:
        return is_confirmed_stranger(obj)

    @staticmethod
    def _frame_diagonal(frame_shape: tuple[int, int, int]) -> float:
        height, width = frame_shape[:2]
        return max(hypot(width, height), 1.0)

    @staticmethod
    def _log_gate(gate: str, now: float, state: _AssetState, **fields: Any) -> None:
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.debug(
            "asset_watch gate=%s now=%.3fs track_id=%s class=%s %s",
            gate,
            now,
            state.track_id,
            state.class_name,
            details,
        )

    def _cleanup_zone(self, camera_id: str, zone_id: str, now: float) -> None:
        for key, state in list(self._assets.items()):
            if key[:2] != (camera_id, zone_id):
                continue
            if now - state.last_seen_inside > self.cleanup_seconds:
                self._assets.pop(key, None)
        self._cleanup_alerted_assets(now)

    def _recent_asset_alert_exists(
        self,
        camera_id: str,
        zone_id: str,
        class_name: str,
        track_id: int,
        now: float,
    ) -> bool:
        for key, alerted_at in self._alerted_assets.items():
            alerted_camera, alerted_zone, alerted_class, alerted_track = key
            if (alerted_camera, alerted_zone, alerted_class) != (camera_id, zone_id, class_name):
                continue
            if alerted_track == track_id:
                return True
            if self.alert_repeat_seconds > 0 and now - alerted_at < self.alert_repeat_seconds:
                return True
        return False

    def _cleanup_alerted_assets(self, now: float) -> None:
        if self.alert_repeat_seconds <= 0:
            self._alerted_assets.clear()
            return
        self._alerted_assets = {
            key: alerted_at
            for key, alerted_at in self._alerted_assets.items()
            if now - alerted_at < self.alert_repeat_seconds
        }

    def _clear_camera(self, camera_id: str) -> None:
        for key in list(self._assets):
            if key[0] == camera_id:
                self._assets.pop(key, None)
        self._alerted_assets = {
            key: alerted_at
            for key, alerted_at in self._alerted_assets.items()
            if key[0] != camera_id
        }


def _global_asset_watch_zone() -> Zone:
    return Zone(
        id="global-asset-watch",
        name="Full Frame",
        zone_type="asset_watch",
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        threshold_seconds=None,
    )
