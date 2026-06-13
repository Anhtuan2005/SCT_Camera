"""Polygon zone management for behavior analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class Zone:
    """A polygon zone using normalized or pixel coordinates."""

    id: str
    name: str
    zone_type: str
    polygon: list[Point]
    threshold_seconds: float | None = None
    auto_generated: bool = False

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "Zone":
        """Build a zone from YAML or API data."""
        polygon = [tuple(map(float, point)) for point in data.get("polygon", [])]
        return cls(
            id=str(data.get("id", data.get("name", "zone"))),
            name=str(data.get("name", data.get("id", "Zone"))),
            zone_type=str(data.get("type", data.get("zone_type", "intrusion"))),
            polygon=polygon,
            threshold_seconds=(
                float(data["threshold_seconds"])
                if data.get("threshold_seconds") is not None
                else None
            ),
            auto_generated=bool(data.get("auto_generated", False)),
        )

    def pixel_polygon(self, frame_shape: tuple[int, int, int] | tuple[int, int]) -> np.ndarray:
        """Return this polygon as an OpenCV pixel-coordinate array."""
        height, width = int(frame_shape[0]), int(frame_shape[1])
        points = [to_pixel_point(point, width, height) for point in self.polygon]
        return np.array(points, dtype=np.int32)

    def contains_point(
        self,
        cx: float,
        cy: float,
        frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    ) -> bool:
        """Return True when the point is inside or on the zone polygon."""
        if len(self.polygon) < 3:
            return False

        if frame_shape is None:
            polygon = np.array(self.polygon, dtype=np.float32)
            test_point = (float(cx), float(cy))
        else:
            polygon = self.pixel_polygon(frame_shape).astype(np.float32)
            test_point = (float(cx), float(cy))

        return cv2.pointPolygonTest(polygon, test_point, False) >= 0

    def applies_to(self, behavior_type: str) -> bool:
        """Return True when this zone should run a behavior rule."""
        return self.zone_type == "all" or self.zone_type == behavior_type


def to_pixel_point(point: Point, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized or pixel point to integer pixel coordinates."""
    x, y = point
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return int(round(x * width)), int(round(y * height))
    return int(round(x)), int(round(y))


def to_normalized_point(point: Point, width: int, height: int) -> tuple[float, float]:
    """Convert a pixel point to normalized coordinates."""
    x, y = point
    return max(0.0, min(1.0, x / width)), max(0.0, min(1.0, y / height))
