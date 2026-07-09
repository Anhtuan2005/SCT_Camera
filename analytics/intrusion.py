"""Intrusion detection rule for restricted zones."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from analytics.zone import Point, Zone, to_pixel_point
from core.tracker import TrackedObject


@dataclass(frozen=True)
class CountingLine:
    """A counting line with a normal direction used to classify in/out."""

    id: str
    name: str
    point1: Point
    point2: Point
    direction: str = "forward"

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "CountingLine":
        """Build a counting line from YAML or API data."""
        return cls(
            id=str(data.get("id", data.get("name", "line"))),
            name=str(data.get("name", data.get("id", "Line"))),
            point1=tuple(map(float, data.get("point1", (0.0, 0.0)))),
            point2=tuple(map(float, data.get("point2", (1.0, 1.0)))),
            direction=str(data.get("direction", "forward")),
        )

    def pixel_points(
        self,
        frame_shape: tuple[int, int, int] | tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return pixel endpoints for this line."""
        height, width = int(frame_shape[0]), int(frame_shape[1])
        return (
            to_pixel_point(self.point1, width, height),
            to_pixel_point(self.point2, width, height),
        )


class IntrusionDetector:
    """Handle intrusion zones and line-crossing alerts."""

    def __init__(
        self,
        reset_frames: int = 30,
        allowed_classes: Iterable[str] | str | None = None,
    ) -> None:
        self.reset_frames = reset_frames
        if isinstance(allowed_classes, str):
            allowed_classes = [allowed_classes]
        self.allowed_classes = {
            str(class_name)
            for class_name in (allowed_classes or ["person"])
        }
        self._occupied_zones: set[tuple[str, str]] = set()
        self._empty_frames: dict[tuple[str, str], int] = defaultdict(int)
        self._counters: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"in": 0, "out": 0}
        )
        self._last_side: dict[tuple[str, str, int], float] = {}
        self._last_touching_line: dict[tuple[str, str, int], bool] = {}

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        zones: list[Zone],
        lines: list[CountingLine],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return intrusion alerts for inbound person line crossings."""
        alerts: list[dict[str, Any]] = []
        alerts.extend(
            self._analyze_lines(
                camera_id,
                camera_name,
                objects,
                lines,
                frame_shape,
                timestamp,
            )
        )
        return alerts

    def get_counters(self, camera_id: str) -> dict[str, dict[str, int]]:
        """Return counters for a camera indexed by line id."""
        return {
            line_id: dict(counter)
            for (cam_id, line_id), counter in self._counters.items()
            if cam_id == camera_id
        }

    def _analyze_zones(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        zones: list[Zone],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        intrusion_zones = [zone for zone in zones if zone.applies_to("intrusion")]
        active_zone_keys = {(camera_id, zone.id) for zone in intrusion_zones}

        for zone in intrusion_zones:
            zone_key = (camera_id, zone.id)
            people_inside = [
                obj
                for obj in objects
                if obj.class_name in self.allowed_classes
                and zone.contains_point(obj.center[0], obj.center[1], frame_shape)
            ]

            if people_inside:
                self._empty_frames[zone_key] = 0
                if zone_key in self._occupied_zones:
                    continue
                self._occupied_zones.add(zone_key)
                actor = max(people_inside, key=lambda obj: obj.confidence)
                alerts.append(
                    {
                        "type": "intrusion",
                        "camera_id": camera_id,
                        "camera_name": camera_name,
                        "track_id": actor.track_id,
                        "class_id": actor.class_id,
                        "class_name": actor.class_name,
                        "zone_id": zone.id,
                        "zone_name": zone.name,
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "details": "Person entered restricted zone",
                    }
                )
                continue

            if zone_key not in self._occupied_zones:
                continue
            self._empty_frames[zone_key] += 1
            if self._empty_frames[zone_key] > self.reset_frames:
                self._occupied_zones.discard(zone_key)
                self._empty_frames.pop(zone_key, None)

        for zone_key in list(self._occupied_zones):
            if zone_key[0] == camera_id and zone_key not in active_zone_keys:
                self._occupied_zones.discard(zone_key)
                self._empty_frames.pop(zone_key, None)

        return alerts

    def _analyze_lines(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        lines: list[CountingLine],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []

        for line in lines:
            p1, p2 = line.pixel_points(frame_shape)
            for obj in objects:
                if len(obj.center_history) < 2:
                    continue
                if obj.class_name not in self.allowed_classes or not obj.class_confirmed:
                    continue
                prev_point = obj.center_history[-2]
                curr_point = obj.center_history[-1]
                key = (camera_id, line.id, obj.track_id)
                prev_side = self._signed_side(prev_point, p1, p2)
                curr_side = self._signed_side(curr_point, p1, p2)
                touching_line = self._segment_intersects_bbox(p1, p2, obj.bbox_xyxy)
                was_touching_line = self._last_touching_line.get(key, False)
                center_crossed = (
                    self._segments_intersect(prev_point, curr_point, p1, p2)
                    or (prev_side < 0 <= curr_side)
                    or (prev_side > 0 >= curr_side)
                )
                crossed = center_crossed or (touching_line and not was_touching_line)

                if not crossed:
                    self._last_side[key] = curr_side
                    self._last_touching_line[key] = touching_line
                    continue

                direction = self._direction_for(line.direction, prev_side, curr_side)
                counter = self._counters[(camera_id, line.id)]
                counter[direction] += 1
                self._last_side[key] = curr_side
                self._last_touching_line[key] = touching_line
                if direction != "in":
                    continue
                alerts.append(
                    self._line_intrusion_alert(
                        camera_id,
                        camera_name,
                        obj,
                        line,
                        timestamp,
                    )
                )

        active_track_ids = {obj.track_id for obj in objects}
        stale_keys = [
            key
            for key in self._last_side
            if key[0] == camera_id and key[2] not in active_track_ids
        ]
        for key in stale_keys:
            self._last_side.pop(key, None)
            self._last_touching_line.pop(key, None)

        return alerts

    def _line_intrusion_alert(
        self,
        camera_id: str,
        camera_name: str,
        obj: TrackedObject,
        line: CountingLine,
        timestamp: datetime,
    ) -> dict[str, Any]:
        counter = self._counters[(camera_id, line.id)]
        return {
            "type": "intrusion",
            "camera_id": camera_id,
            "camera_name": camera_name,
            "track_id": obj.track_id,
            "class_id": obj.class_id,
            "class_name": obj.class_name,
            "identity_label": obj.identity_label,
            "identity_kind": obj.identity_kind,
            "line_id": line.id,
            "line_name": line.name,
            "direction": "IN",
            "total_count": dict(counter),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "details": (
                f"Person crossed line IN | "
                f"Total count: IN={counter['in']}, OUT={counter['out']}"
            ),
        }

    @staticmethod
    def _signed_side(
        point: tuple[float, float],
        p1: tuple[int, int],
        p2: tuple[int, int],
    ) -> float:
        return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (
            point[0] - p1[0]
        )

    @staticmethod
    def _direction_for(
        config_direction: str,
        prev_side: float,
        curr_side: float,
    ) -> str:
        forward = "in" if prev_side < curr_side else "out"
        if config_direction.lower() in {"reverse", "backward", "out"}:
            return "out" if forward == "in" else "in"
        return forward

    @staticmethod
    def _orientation(
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
    ) -> float:
        return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (
            c[1] - b[1]
        )

    @classmethod
    def _segments_intersect(
        cls,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        d: tuple[float, float],
    ) -> bool:
        o1 = cls._orientation(a, b, c)
        o2 = cls._orientation(a, b, d)
        o3 = cls._orientation(c, d, a)
        o4 = cls._orientation(c, d, b)
        return (o1 * o2 < 0) and (o3 * o4 < 0)

    @staticmethod
    def _segment_intersects_bbox(
        p1: tuple[float, float],
        p2: tuple[float, float],
        bbox_xyxy: tuple[float, float, float, float],
    ) -> bool:
        x1, y1, x2, y2 = bbox_xyxy
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        t0, t1 = 0.0, 1.0
        for edge, distance in (
            (-dx, p1[0] - left),
            (dx, right - p1[0]),
            (-dy, p1[1] - top),
            (dy, bottom - p1[1]),
        ):
            if edge == 0:
                if distance < 0:
                    return False
                continue
            t = distance / edge
            if edge < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
        return True
