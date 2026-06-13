"""Suspicious theft behavior rule for watched vehicles and assets."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from math import hypot
from typing import Any

from analytics.identity_status import is_confirmed_stranger
from analytics.zone import Zone
from core.tracker import TrackedObject


@dataclass
class _PairState:
    first_near: float
    last_seen: float
    initial_vehicle_center: tuple[float, float]
    last_person_side: int = 0
    last_side_change_at: float = 0.0
    pass_count: int = 0
    alerted: bool = False
    latest_behaviors: set[str] = field(default_factory=set)


class SuspiciousTheftDetector:
    """Score suspicious behavior near watched vehicles or movable assets."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        settings = settings or {}
        self.enabled = bool(settings.get("enabled", True))
        self.vehicle_classes = {
            str(item)
            for item in settings.get("vehicle_classes", ["bicycle", "car", "motorcycle", "bus", "truck"])
        }
        self.proximity_distance_meters = float(settings.get("proximity_distance_meters", 1.5))
        self.meters_per_frame_diagonal = max(0.1, float(settings.get("meters_per_frame_diagonal", 12)))
        self.proximity_seconds = float(settings.get("proximity_seconds", 10))
        self.vehicle_move_min_ratio = float(settings.get("vehicle_move_min_ratio", 0.025))
        self.same_direction_min_cosine = float(settings.get("same_direction_min_cosine", 0.65))
        self.pacing_min_passes = int(settings.get("pacing_min_passes", 2))
        self.score_threshold = int(settings.get("score_threshold", 2))
        self.require_near_duration = bool(settings.get("require_near_duration", True))
        self.require_vehicle_signal = bool(settings.get("require_vehicle_signal", True))
        self.pose_wrist_distance_ratio = float(settings.get("pose_wrist_distance_ratio", 0.055))
        self.near_gap_reset_seconds = float(settings.get("near_gap_reset_seconds", 2))
        self.stale_seconds = float(settings.get("stale_seconds", 30))
        self._pairs: dict[tuple[str, str, int, int], _PairState] = {}
        self._alerted_vehicle_signatures: set[tuple[str, str, str]] = set()

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        zones: list[Zone],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return suspicious theft behavior alerts for asset-watch zones."""
        if not self.enabled:
            return []

        watch_zones = [zone for zone in zones if zone.applies_to("asset_watch")]
        if not watch_zones:
            watch_zones = [_global_asset_watch_zone()]

        now = time.monotonic()
        alerts: list[dict[str, Any]] = []
        persons = [obj for obj in objects if self._is_unknown_person(obj)]
        vehicles = [obj for obj in objects if obj.class_name in self.vehicle_classes]

        for zone in watch_zones:
            seen_pair_keys: set[tuple[str, str, int, int]] = set()
            zone_persons = [obj for obj in persons if zone.contains_point(obj.center[0], obj.center[1], frame_shape)]
            zone_vehicles = [obj for obj in vehicles if zone.contains_point(obj.center[0], obj.center[1], frame_shape)]
            for person in zone_persons:
                for vehicle in zone_vehicles:
                    if not self._is_near_vehicle(person, vehicle, frame_shape):
                        continue
                    state = self._state_for(camera_id, zone.id, person, vehicle, now)
                    seen_pair_keys.add(self._pair_key(camera_id, zone.id, person, vehicle))
                    self._update_pacing(state, person, vehicle, frame_shape, now)
                    alert = self._evaluate_pair(
                        camera_id,
                        camera_name,
                        zone,
                        person,
                        vehicle,
                        state,
                        frame_shape,
                        timestamp,
                        now,
                    )
                    if alert is not None:
                        alerts.append(alert)
            self._cleanup_zone(camera_id, zone.id, now, seen_pair_keys)

        return alerts

    def _evaluate_pair(
        self,
        camera_id: str,
        camera_name: str,
        zone: Zone,
        person: TrackedObject,
        vehicle: TrackedObject,
        state: _PairState,
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
        now: float,
    ) -> dict[str, Any] | None:
        near_seconds = now - state.first_near
        vehicle_vector = _motion_vector(vehicle.center_history)
        person_vector = _motion_vector(person.center_history)
        vehicle_started_moving = self._vehicle_started_moving(vehicle, state, frame_shape)
        same_direction = _same_direction(person_vector, vehicle_vector, self.same_direction_min_cosine)
        pose_push = self._pose_push_signal(person, vehicle, frame_shape)

        behaviors: set[str] = set()
        if near_seconds >= self.proximity_seconds:
            behaviors.add("near_vehicle_duration")
        if state.pass_count >= self.pacing_min_passes:
            behaviors.add("pacing_near_vehicle")
        if vehicle_started_moving:
            behaviors.add("vehicle_started_moving")
        if same_direction:
            behaviors.add("moving_same_direction")
        if pose_push:
            behaviors.add("pose_push_contact")
        state.latest_behaviors = behaviors

        vehicle_signal = bool({"vehicle_started_moving", "moving_same_direction", "pose_push_contact"} & behaviors)
        if state.alerted or len(behaviors) < self.score_threshold:
            return None
        if self.require_near_duration and "near_vehicle_duration" not in behaviors:
            return None
        if self.require_vehicle_signal and not vehicle_signal:
            return None
        vehicle_signature = (camera_id, zone.id, vehicle.class_name)
        if vehicle_signature in self._alerted_vehicle_signatures:
            return None

        state.alerted = True
        self._alerted_vehicle_signatures.add(vehicle_signature)
        details = (
            f"Unknown person stayed near {vehicle.class_name} for {near_seconds:.1f}s; "
            f"passes={state.pass_count}; behaviors={', '.join(sorted(behaviors))}"
        )
        return {
            "type": "suspicious_theft_behavior",
            "camera_id": camera_id,
            "camera_name": camera_name,
            "track_id": person.track_id,
            "class_name": person.class_name,
            "identity_label": person.identity_label or "Stranger",
            "identity_kind": person.identity_kind or "stranger",
            "vehicle_track_id": vehicle.track_id,
            "vehicle_class_name": vehicle.class_name,
            "zone_id": zone.id,
            "zone_name": zone.name,
            "score": len(behaviors),
            "score_threshold": self.score_threshold,
            "behaviors": sorted(behaviors),
            "near_seconds": round(near_seconds, 1),
            "pacing_passes": state.pass_count,
            "vehicle_started_moving": vehicle_started_moving,
            "moving_same_direction": same_direction,
            "pose_push_contact": pose_push,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "siren": True,
            "details": details,
        }

    def _state_for(
        self,
        camera_id: str,
        zone_id: str,
        person: TrackedObject,
        vehicle: TrackedObject,
        now: float,
    ) -> _PairState:
        key = self._pair_key(camera_id, zone_id, person, vehicle)
        state = self._pairs.get(key)
        if state is None:
            state = _PairState(
                first_near=now,
                last_seen=now,
                initial_vehicle_center=vehicle.center,
                last_side_change_at=now,
            )
            self._pairs[key] = state
        else:
            state.last_seen = now
        return state

    @staticmethod
    def _pair_key(
        camera_id: str,
        zone_id: str,
        person: TrackedObject,
        vehicle: TrackedObject,
    ) -> tuple[str, str, int, int]:
        return camera_id, zone_id, person.track_id, vehicle.track_id

    def _update_pacing(
        self,
        state: _PairState,
        person: TrackedObject,
        vehicle: TrackedObject,
        frame_shape: tuple[int, int, int],
        now: float,
    ) -> None:
        side_deadband = self._frame_diagonal(frame_shape) * 0.025
        offset_x = person.center[0] - vehicle.center[0]
        if abs(offset_x) < side_deadband:
            return
        side = 1 if offset_x > 0 else -1
        if state.last_person_side == 0:
            state.last_person_side = side
            return
        if side == state.last_person_side:
            return
        if now - state.last_side_change_at < 0.8:
            return
        state.pass_count += 1
        state.last_person_side = side
        state.last_side_change_at = now

    def _is_near_vehicle(
        self,
        person: TrackedObject,
        vehicle: TrackedObject,
        frame_shape: tuple[int, int, int],
    ) -> bool:
        max_distance_px = self._frame_diagonal(frame_shape) * (
            self.proximity_distance_meters / self.meters_per_frame_diagonal
        )
        return _point_to_bbox_distance(person.center, vehicle.bbox_xyxy) <= max_distance_px

    def _vehicle_started_moving(
        self,
        vehicle: TrackedObject,
        state: _PairState,
        frame_shape: tuple[int, int, int],
    ) -> bool:
        min_distance = self._frame_diagonal(frame_shape) * self.vehicle_move_min_ratio
        from_initial = hypot(
            vehicle.center[0] - state.initial_vehicle_center[0],
            vehicle.center[1] - state.initial_vehicle_center[1],
        )
        from_history = _vector_length(_motion_vector(vehicle.center_history))
        return max(from_initial, from_history) >= min_distance

    def _pose_push_signal(
        self,
        person: TrackedObject,
        vehicle: TrackedObject,
        frame_shape: tuple[int, int, int],
    ) -> bool:
        if not person.pose_keypoints:
            return False
        max_distance = self._frame_diagonal(frame_shape) * self.pose_wrist_distance_ratio
        for index in (9, 10):
            if index >= len(person.pose_keypoints):
                continue
            x, y, confidence = person.pose_keypoints[index]
            if confidence < 0.25:
                continue
            if _point_to_bbox_distance((x, y), vehicle.bbox_xyxy) <= max_distance:
                return True
        return False

    @staticmethod
    def _is_unknown_person(obj: TrackedObject) -> bool:
        return is_confirmed_stranger(obj)

    @staticmethod
    def _frame_diagonal(frame_shape: tuple[int, int, int]) -> float:
        height, width = frame_shape[:2]
        return max(hypot(width, height), 1.0)

    def _cleanup_zone(
        self,
        camera_id: str,
        zone_id: str,
        now: float,
        seen_pair_keys: set[tuple[str, str, int, int]],
    ) -> None:
        for key, state in list(self._pairs.items()):
            if key[:2] != (camera_id, zone_id):
                continue
            if key not in seen_pair_keys and now - state.last_seen > self.near_gap_reset_seconds:
                self._pairs.pop(key, None)
                continue
            if now - state.last_seen > self.stale_seconds:
                self._pairs.pop(key, None)

    def _clear_camera(self, camera_id: str) -> None:
        for key in list(self._pairs):
            if key[0] == camera_id:
                self._pairs.pop(key, None)
        self._alerted_vehicle_signatures = {
            key for key in self._alerted_vehicle_signatures if key[0] != camera_id
        }


def _global_asset_watch_zone() -> Zone:
    return Zone(
        id="global-asset-watch",
        name="Full Frame",
        zone_type="asset_watch",
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        threshold_seconds=None,
    )


def _motion_vector(history: list[tuple[float, float]], window: int = 8) -> tuple[float, float]:
    if len(history) < 2:
        return 0.0, 0.0
    points = history[-window:]
    return points[-1][0] - points[0][0], points[-1][1] - points[0][1]


def _same_direction(
    first: tuple[float, float],
    second: tuple[float, float],
    min_cosine: float,
) -> bool:
    first_length = _vector_length(first)
    second_length = _vector_length(second)
    if first_length <= 0 or second_length <= 0:
        return False
    cosine = ((first[0] * second[0]) + (first[1] * second[1])) / (first_length * second_length)
    return cosine >= min_cosine


def _vector_length(vector: tuple[float, float]) -> float:
    return hypot(vector[0], vector[1])


def _point_to_bbox_distance(
    point: tuple[float, float],
    bbox_xyxy: tuple[float, float, float, float],
) -> float:
    x, y = point
    x1, y1, x2, y2 = bbox_xyxy
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return hypot(dx, dy)
