"""Line crossing counter for tracked objects."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from analytics.zone import Point, to_pixel_point
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

    def pixel_points(self, frame_shape: tuple[int, int, int] | tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return pixel endpoints for this line."""
        height, width = int(frame_shape[0]), int(frame_shape[1])
        return to_pixel_point(self.point1, width, height), to_pixel_point(self.point2, width, height)


class LineCounter:
    """Count tracked objects crossing configured lines."""

    def __init__(self) -> None:
        self._counters: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"in": 0, "out": 0}
        )
        self._last_side: dict[tuple[str, str, int], float] = {}

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        objects: list[TrackedObject],
        lines: list[CountingLine],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Return line crossing alerts and update counters."""
        alerts: list[dict[str, Any]] = []

        for line in lines:
            p1, p2 = line.pixel_points(frame_shape)
            for obj in objects:
                if len(obj.center_history) < 2:
                    continue
                prev_point = obj.center_history[-2]
                curr_point = obj.center_history[-1]
                key = (camera_id, line.id, obj.track_id)
                prev_side = self._signed_side(prev_point, p1, p2)
                curr_side = self._signed_side(curr_point, p1, p2)

                if key not in self._last_side:
                    self._last_side[key] = curr_side
                    continue

                crossed = (
                    self._segments_intersect(prev_point, curr_point, p1, p2)
                    or (prev_side < 0 <= curr_side)
                    or (prev_side > 0 >= curr_side)
                )
                if not crossed or prev_side == curr_side:
                    self._last_side[key] = curr_side
                    continue

                direction = self._direction_for(line.direction, prev_side, curr_side)
                counter = self._counters[(camera_id, line.id)]
                counter[direction] += 1
                self._last_side[key] = curr_side
                alerts.append(
                    {
                        "type": "line_crossing",
                        "camera_id": camera_id,
                        "camera_name": camera_name,
                        "track_id": obj.track_id,
                        "class_id": obj.class_id,
                        "class_name": obj.class_name,
                        "line_id": line.id,
                        "line_name": line.name,
                        "direction": direction.upper(),
                        "total_count": dict(counter),
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "details": (
                            f"Direction: {direction.upper()} | "
                            f"Total count: IN={counter['in']}, OUT={counter['out']}"
                        ),
                    }
                )

        active_track_ids = {obj.track_id for obj in objects}
        stale_keys = [key for key in self._last_side if key[0] == camera_id and key[2] not in active_track_ids]
        for key in stale_keys:
            self._last_side.pop(key, None)

        return alerts

    def get_counters(self, camera_id: str) -> dict[str, dict[str, int]]:
        """Return counters for a camera indexed by line id."""
        return {
            line_id: dict(counter)
            for (cam_id, line_id), counter in self._counters.items()
            if cam_id == camera_id
        }

    @staticmethod
    def _signed_side(point: tuple[float, float], p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])

    @staticmethod
    def _direction_for(config_direction: str, prev_side: float, curr_side: float) -> str:
        forward = "in" if prev_side < curr_side else "out"
        if config_direction.lower() in {"reverse", "backward", "out"}:
            return "out" if forward == "in" else "in"
        return forward

    @staticmethod
    def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

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
