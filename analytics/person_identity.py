"""Person and animal labeling for tracked objects."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.tracker import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class _Reference:
    name: str
    histogram: np.ndarray


class PersonIdentityResolver:
    """Assign stable demo labels to people and animal classes."""

    def __init__(self, settings: dict[str, Any]) -> None:
        identity = settings.get("identity", {})
        self.enabled = bool(identity.get("enabled", True))
        self.unknown_label = str(identity.get("unknown_person_label", "Stranger"))
        self.threshold = float(identity.get("histogram_similarity_threshold", 0.78))
        self.min_crop_height = int(identity.get("min_person_crop_height", 80))
        self.references = self._load_references(identity.get("known_persons", []))
        self._track_cache: dict[tuple[str, int], tuple[str, str, float | None]] = {}

    def label_objects(
        self,
        camera_id: str,
        objects: list[TrackedObject],
        frame_bgr: np.ndarray,
    ) -> list[TrackedObject]:
        """Return objects with identity fields filled in."""
        active_keys = {(camera_id, obj.track_id) for obj in objects}
        stale_keys = [key for key in self._track_cache if key[0] == camera_id and key not in active_keys]
        for key in stale_keys:
            self._track_cache.pop(key, None)

        labeled: list[TrackedObject] = []
        for obj in objects:
            if obj.class_name in {"cat", "dog"}:
                labeled.append(
                    replace(
                        obj,
                        identity_label=obj.class_name,
                        identity_kind="animal",
                        identity_score=None,
                    )
                )
                continue
            if obj.class_name != "person":
                labeled.append(obj)
                continue
            labeled.append(self._label_person(camera_id, obj, frame_bgr))
        return labeled

    def _label_person(
        self,
        camera_id: str,
        obj: TrackedObject,
        frame_bgr: np.ndarray,
    ) -> TrackedObject:
        key = (camera_id, obj.track_id)
        cached = self._track_cache.get(key)
        if cached and cached[1] == "known_person":
            label, kind, score = cached
            return replace(obj, identity_label=label, identity_kind=kind, identity_score=score)

        label = self.unknown_label
        kind = "stranger"
        score: float | None = None
        if self.enabled and self.references:
            match = self._match_person(obj, frame_bgr)
            if match is not None:
                label, score = match
                kind = "known_person"

        self._track_cache[key] = (label, kind, score)
        return replace(obj, identity_label=label, identity_kind=kind, identity_score=score)

    def _match_person(
        self,
        obj: TrackedObject,
        frame_bgr: np.ndarray,
    ) -> tuple[str, float] | None:
        histogram = self._histogram_for_box(frame_bgr, obj.bbox_xyxy)
        if histogram is None:
            return None

        best_name = ""
        best_score = -1.0
        for reference in self.references:
            score = float(cv2.compareHist(histogram, reference.histogram, cv2.HISTCMP_CORREL))
            if score > best_score:
                best_name = reference.name
                best_score = score

        if best_score >= self.threshold:
            return best_name, best_score
        return None

    def _load_references(self, known_people: Any) -> list[_Reference]:
        references: list[_Reference] = []
        if not isinstance(known_people, list):
            return references

        for person in known_people:
            if not isinstance(person, dict):
                continue
            name = str(person.get("name", "")).strip()
            if not name:
                continue
            for image_path in self._reference_paths(person):
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning("Known person reference image not readable: %s", image_path)
                    continue
                histogram = self._histogram(image)
                if histogram is not None:
                    references.append(_Reference(name=name, histogram=histogram))
        if references:
            logger.info("Loaded %s known-person reference image(s)", len(references))
        return references

    @staticmethod
    def _reference_paths(person: dict[str, Any]) -> list[Path]:
        raw_paths = person.get("reference_images", person.get("reference_image", []))
        if isinstance(raw_paths, (str, Path)):
            raw_paths = [raw_paths]
        if not isinstance(raw_paths, list):
            return []

        paths: list[Path] = []
        project_root = Path(__file__).resolve().parents[1]
        for raw_path in raw_paths:
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = project_root / path
            if path.is_dir():
                for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    paths.extend(sorted(path.glob(pattern)))
            else:
                paths.append(path)
        return paths

    def _histogram_for_box(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> np.ndarray | None:
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        left = max(0, min(width - 1, int(round(x1))))
        top = max(0, min(height - 1, int(round(y1))))
        right = max(left + 1, min(width, int(round(x2))))
        bottom = max(top + 1, min(height, int(round(y2))))
        if bottom - top < self.min_crop_height:
            return None
        return self._histogram(frame_bgr[top:bottom, left:right])

    @staticmethod
    def _histogram(image_bgr: np.ndarray) -> np.ndarray | None:
        if image_bgr.size == 0:
            return None
        resized = cv2.resize(image_bgr, (96, 192), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [12, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(histogram, histogram)
        return histogram.flatten().astype(np.float32)
