"""Person and animal labeling for tracked objects."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analytics.identity_status import (
    KNOWN_PERSON_KIND,
    PENDING_PERSON_KIND,
    STRANGER_KIND,
)
from core.tracker import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class _Reference:
    name: str
    embedding: np.ndarray


@dataclass(frozen=True)
class _KnownIdentityMemory:
    label: str
    score: float | None
    bbox_xyxy: tuple[float, float, float, float]
    frame_number: int


class PersonIdentityResolver:
    """Assign person names with InsightFace and stable labels to animal classes."""

    def __init__(self, settings: dict[str, Any]) -> None:
        identity = settings.get("identity", {})
        self.enabled = bool(identity.get("enabled", True))
        self.unknown_label = str(identity.get("unknown_person_label", "Stranger"))
        self.pending_label = str(identity.get("pending_person_label", "Identifying"))
        self.unknown_confirmation_attempts = max(
            0,
            int(identity.get("unknown_confirmation_attempts", 2)),
        )
        self.threshold = float(identity.get("similarity_threshold", 0.45))
        self.min_detection_score = float(
            identity.get("min_detection_score", 0.5)
        )
        self.min_face_size = max(1, int(identity.get("min_face_size", 24)))
        self.recognition_interval = max(
            1,
            int(identity.get("recognition_interval_frames", 5)),
        )
        self.known_memory_frames = max(
            0,
            int(identity.get("known_memory_frames", 90)),
        )
        self.known_memory_distance_ratio = float(
            identity.get("known_memory_distance_ratio", 0.18)
        )
        self.known_memory_min_area_ratio = max(
            0.0,
            min(1.0, float(identity.get("known_memory_min_area_ratio", 0.25))),
        )
        self.model_name = str(identity.get("model", "buffalo_l"))
        self.device = str(identity.get("device", "auto")).lower()
        self.det_size = self._parse_detection_size(
            identity.get("detection_size", 640)
        )
        self.orientations = self._parse_orientations(
            identity.get("orientations", ["none", "cw90", "ccw90", "180"])
        )
        self.reference_orientations = self._parse_orientations(
            identity.get("reference_orientations", ["none", "cw90", "ccw90", "180"])
        )
        self.orientations_per_attempt = max(
            1,
            int(identity.get("orientations_per_attempt", 2)),
        )
        self.model_root = self._resolve_path(
            identity.get("model_root", "models/insightface")
        )
        raw_providers = identity.get("providers", [])
        self.configured_providers = (
            [str(provider) for provider in raw_providers]
            if isinstance(raw_providers, list)
            else []
        )
        self.known_people = (
            identity.get("known_persons", [])
            if isinstance(identity.get("known_persons", []), list)
            else []
        )

        self.references: list[_Reference] = []
        self._face_app: Any | None = None
        self._ready = False
        self._initialization_attempted = False
        self._inference_lock = threading.RLock()
        self._track_cache: dict[
            tuple[str, int],
            tuple[str, str, float | None],
        ] = {}
        self._failed_attempt_counts: dict[tuple[str, int], int] = {}
        self._camera_frame_counts: dict[str, int] = {}
        self._last_attempt_frame: dict[tuple[str, int], int] = {}
        self._camera_orientation_offsets: dict[str, int] = {}
        self._known_identity_memory: dict[
            tuple[str, str],
            _KnownIdentityMemory,
        ] = {}

        if self.enabled and self.known_people:
            self._ensure_ready()

    def label_objects(
        self,
        camera_id: str,
        objects: list[TrackedObject],
        frame_bgr: np.ndarray,
        assume_unknown_persons: bool = False,
    ) -> list[TrackedObject]:
        """Return objects with identity fields filled in."""
        active_keys = {
            (camera_id, obj.track_id)
            for obj in objects
            if obj.class_name == "person"
        }
        self._remove_stale_tracks(camera_id, active_keys)

        frame_number = self._camera_frame_counts.get(camera_id, 0) + 1
        self._camera_frame_counts[camera_id] = frame_number
        self._remove_stale_identity_memory(camera_id, frame_number)
        unresolved = [
            obj
            for obj in objects
            if obj.class_name == "person"
            and self._track_cache.get((camera_id, obj.track_id), ("", "", None))[1]
            != KNOWN_PERSON_KIND
        ]

        matches: dict[int, tuple[str, float]] = {}
        recognition_attempted = False
        if (
            unresolved
            and not assume_unknown_persons
            and self.enabled
            and self._recognition_due(camera_id, unresolved, frame_number)
            and self._ensure_ready()
        ):
            recognition_attempted = True
            matches = self._match_people(camera_id, unresolved, frame_bgr)
            for obj in unresolved:
                key = (camera_id, obj.track_id)
                self._last_attempt_frame[key] = frame_number
                if obj.track_id in matches:
                    self._failed_attempt_counts.pop(key, None)
                else:
                    self._failed_attempt_counts[key] = (
                        self._failed_attempt_counts.get(key, 0) + 1
                    )

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

            key = (camera_id, obj.track_id)
            cached = self._track_cache.get(key)
            if assume_unknown_persons:
                label = self.unknown_label
                kind = STRANGER_KIND
                score = None
                self._track_cache[key] = (label, kind, score)
                self._failed_attempt_counts.pop(key, None)
            elif cached and cached[1] == KNOWN_PERSON_KIND:
                label, kind, score = cached
            elif obj.track_id in matches:
                label, score = matches[obj.track_id]
                kind = KNOWN_PERSON_KIND
                self._track_cache[key] = (label, kind, score)
            elif (
                memory := self._known_memory_match(
                    camera_id,
                    obj,
                    frame_bgr.shape,
                    frame_number,
                )
            ) is not None:
                label = memory.label
                kind = KNOWN_PERSON_KIND
                score = memory.score
                self._track_cache[key] = (label, kind, score)
            elif self._identity_still_pending(key):
                label = self.pending_label
                kind = PENDING_PERSON_KIND
                score = None
                self._track_cache[key] = (label, kind, score)
            else:
                label = self.unknown_label
                kind = STRANGER_KIND
                score = None
                if cached is None or cached[1] != STRANGER_KIND:
                    logger.info(
                        "Track %d confirmed as Stranger after %d failed attempt(s)",
                        obj.track_id,
                        self._failed_attempt_counts.get(key, 0),
                    )
                self._track_cache[key] = (label, kind, score)
            labeled_obj = replace(
                obj,
                identity_label=label,
                identity_kind=kind,
                identity_score=score,
            )
            if kind == KNOWN_PERSON_KIND:
                self._remember_known_identity(camera_id, labeled_obj, frame_number)
            labeled.append(labeled_obj)
        return labeled

    def _recognition_due(
        self,
        camera_id: str,
        objects: list[TrackedObject],
        frame_number: int,
    ) -> bool:
        return any(
            self._track_cache.get((camera_id, obj.track_id), ("", "", None))[1]
            == PENDING_PERSON_KIND
            or (
                frame_number
                - self._last_attempt_frame.get(
                    (camera_id, obj.track_id),
                    -self.recognition_interval,
                )
                >= self.recognition_interval
            )
            for obj in objects
        )

    def _remove_stale_tracks(
        self,
        camera_id: str,
        active_keys: set[tuple[str, int]],
    ) -> None:
        stale_keys = [
            key
            for key in self._track_cache
            if key[0] == camera_id and key not in active_keys
        ]
        for key in stale_keys:
            self._track_cache.pop(key, None)
            self._last_attempt_frame.pop(key, None)
            self._failed_attempt_counts.pop(key, None)

    def _remember_known_identity(
        self,
        camera_id: str,
        obj: TrackedObject,
        frame_number: int,
    ) -> None:
        if not obj.identity_label:
            return
        self._known_identity_memory[(camera_id, obj.identity_label)] = (
            _KnownIdentityMemory(
                label=obj.identity_label,
                score=obj.identity_score,
                bbox_xyxy=obj.bbox_xyxy,
                frame_number=frame_number,
            )
        )

    def _known_memory_match(
        self,
        camera_id: str,
        obj: TrackedObject,
        frame_shape: tuple[int, ...],
        frame_number: int,
    ) -> _KnownIdentityMemory | None:
        if self.known_memory_frames <= 0:
            return None
        best_memory: _KnownIdentityMemory | None = None
        best_rank = -1.0
        for key, memory in self._known_identity_memory.items():
            if key[0] != camera_id:
                continue
            if frame_number - memory.frame_number > self.known_memory_frames:
                continue
            overlap = _bbox_iou(obj.bbox_xyxy, memory.bbox_xyxy)
            area_ratio = _bbox_area_ratio(obj.bbox_xyxy, memory.bbox_xyxy)
            distance = self._bbox_center_distance_ratio(
                obj.bbox_xyxy,
                memory.bbox_xyxy,
                frame_shape,
            )
            if area_ratio < self.known_memory_min_area_ratio and overlap < 0.25:
                continue
            if overlap <= 0.03 and distance > self.known_memory_distance_ratio:
                continue
            rank = max(overlap, 1.0 - distance)
            if rank > best_rank:
                best_rank = rank
                best_memory = memory
        return best_memory

    def _remove_stale_identity_memory(
        self,
        camera_id: str,
        frame_number: int,
    ) -> None:
        stale_keys = [
            key
            for key, memory in self._known_identity_memory.items()
            if key[0] == camera_id
            and frame_number - memory.frame_number > self.known_memory_frames
        ]
        for key in stale_keys:
            self._known_identity_memory.pop(key, None)

    def _identity_still_pending(self, key: tuple[str, int]) -> bool:
        if not self.enabled or not self.references:
            return False
        return (
            self._failed_attempt_counts.get(key, 0)
            < self.unknown_confirmation_attempts
        )

    @staticmethod
    def _bbox_center_distance_ratio(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
        frame_shape: tuple[int, ...],
    ) -> float:
        height, width = frame_shape[:2]
        diagonal = max((float(width) ** 2 + float(height) ** 2) ** 0.5, 1.0)
        first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
        second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
        return (
            (first_center[0] - second_center[0]) ** 2
            + (first_center[1] - second_center[1]) ** 2
        ) ** 0.5 / diagonal

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        if (
            not self.enabled
            or not self.known_people
            or self._initialization_attempted
        ):
            return False

        with self._inference_lock:
            if self._ready:
                return True
            if self._initialization_attempted:
                return False
            self._initialization_attempted = True
            try:
                import onnxruntime as ort
                from insightface.app import FaceAnalysis

                providers = self._providers(ort.get_available_providers())
                self._face_app = FaceAnalysis(
                    name=self.model_name,
                    root=str(self.model_root),
                    allowed_modules=["detection", "recognition"],
                    providers=providers,
                )
                self._face_app.prepare(
                    ctx_id=self._context_id(providers),
                    det_size=self.det_size,
                )
                self.references = self._load_references(self.known_people)
                self._ready = bool(self.references)
                if self._ready:
                    logger.info(
                        "InsightFace ready: model=%s providers=%s references=%d",
                        self.model_name,
                        providers,
                        len(self.references),
                    )
                else:
                    logger.warning(
                        "InsightFace found no usable known-person reference faces"
                    )
            except Exception as exc:
                self._face_app = None
                logger.exception("InsightFace initialization failed: %s", exc)
        return self._ready

    def _match_people(
        self,
        camera_id: str,
        people: list[TrackedObject],
        frame_bgr: np.ndarray,
    ) -> dict[int, tuple[str, float]]:
        matches: dict[int, tuple[str, float]] = {}
        matched_orientation: str | None = None
        for orientation, faces in self._analyze_face_sets(
            frame_bgr,
            self._orientation_batch(camera_id),
        ):
            if not faces:
                logger.debug(
                    "No faces detected in orientation %s for camera %s",
                    orientation,
                    camera_id,
                )
                continue
            for face in sorted(faces, key=self._face_quality, reverse=True):
                face_bbox = self._face_bbox(face)
                embedding = self._face_embedding(face)
                if face_bbox is None or embedding is None:
                    continue
                detection_score = self._face_detection_score(face)
                if detection_score < self.min_detection_score:
                    continue
                face_width = face_bbox[2] - face_bbox[0]
                face_height = face_bbox[3] - face_bbox[1]
                if min(face_width, face_height) < self.min_face_size:
                    logger.debug(
                        "Face too small (%.0fx%.0f < %d min) for track matching",
                        face_width,
                        face_height,
                        self.min_face_size,
                    )
                    continue

                person = self._person_for_face(face_bbox, people, set(matches))
                if person is None:
                    continue
                best_name, best_score = self._best_reference_score(embedding)
                if best_score >= self.threshold:
                    matches[person.track_id] = (best_name, best_score)
                    matched_orientation = orientation
                    logger.debug(
                        "Face match result for track %d: %s (score=%.3f, threshold=%.3f)",
                        person.track_id,
                        best_name,
                        best_score,
                        self.threshold,
                    )
                else:
                    logger.debug(
                        "Best match for track %d: %s score=%.3f (threshold=%.3f) - rejected",
                        person.track_id,
                        best_name,
                        best_score,
                        self.threshold,
                    )
            if len(matches) >= len(people):
                break
        self._update_orientation_offset(camera_id, matched_orientation)
        return matches

    def _analyze_faces(self, image_bgr: np.ndarray) -> list[Any]:
        for _orientation, faces in self._analyze_face_sets(
            image_bgr,
            self.reference_orientations,
        ):
            if faces:
                return faces
        return []

    def _analyze_face_sets(
        self,
        image_bgr: np.ndarray,
        orientations: list[str],
    ) -> Iterator[tuple[str, list[Any]]]:
        if self._face_app is None or image_bgr.size == 0:
            return

        for orientation in orientations:
            rotated = self._rotate_image(image_bgr, orientation)
            yield orientation, self._analyze_faces_once(
                rotated,
                orientation,
                image_bgr.shape,
            )

    def _orientation_batch(self, camera_id: str) -> list[str]:
        if not self.orientations:
            return ["none"]
        batch_size = min(self.orientations_per_attempt, len(self.orientations))
        start = self._camera_orientation_offsets.get(camera_id, 0) % len(
            self.orientations
        )
        return [
            self.orientations[(start + offset) % len(self.orientations)]
            for offset in range(batch_size)
        ]

    def _update_orientation_offset(
        self,
        camera_id: str,
        matched_orientation: str | None,
    ) -> None:
        if not self.orientations:
            return
        if matched_orientation in self.orientations:
            self._camera_orientation_offsets[camera_id] = self.orientations.index(
                matched_orientation
            )
            return
        start = self._camera_orientation_offsets.get(camera_id, 0)
        self._camera_orientation_offsets[camera_id] = (
            start + self.orientations_per_attempt
        ) % len(self.orientations)

    def _analyze_faces_once(
        self,
        image_bgr: np.ndarray,
        orientation: str,
        original_shape: tuple[int, ...],
    ) -> list[Any]:
        try:
            with self._inference_lock:
                faces = list(self._face_app.get(image_bgr))
        except Exception as exc:
            logger.warning("InsightFace inference failed: %s", exc)
            return []
        if orientation == "none":
            return faces
        return [
            self._map_rotated_face_to_original(face, orientation, original_shape)
            for face in faces
        ]

    @staticmethod
    def _rotate_image(image_bgr: np.ndarray, orientation: str) -> np.ndarray:
        if orientation == "cw90":
            return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        if orientation == "ccw90":
            return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if orientation == "180":
            return cv2.rotate(image_bgr, cv2.ROTATE_180)
        return image_bgr

    @classmethod
    def _map_rotated_face_to_original(
        cls,
        face: Any,
        orientation: str,
        original_shape: tuple[int, ...],
    ) -> Any:
        bbox = cls._face_bbox(face)
        if bbox is None:
            return face

        mapped_face = {
            "bbox": np.asarray(
                cls._map_bbox_to_original(bbox, orientation, original_shape),
                dtype=np.float32,
            )
        }
        for key in ("det_score", "normed_embedding", "embedding"):
            value = cls._face_value(face, key)
            if value is not None:
                mapped_face[key] = value
        return mapped_face

    @staticmethod
    def _map_bbox_to_original(
        bbox: tuple[float, float, float, float],
        orientation: str,
        original_shape: tuple[int, ...],
    ) -> tuple[float, float, float, float]:
        height, width = original_shape[:2]
        x1, y1, x2, y2 = bbox
        points = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        mapped_points: list[tuple[float, float]] = []
        for x, y in points:
            if orientation == "cw90":
                mapped_points.append((y, height - x))
            elif orientation == "ccw90":
                mapped_points.append((width - y, x))
            elif orientation == "180":
                mapped_points.append((width - x, height - y))
            else:
                mapped_points.append((x, y))

        xs = [max(0.0, min(float(width), x)) for x, _y in mapped_points]
        ys = [max(0.0, min(float(height), y)) for _x, y in mapped_points]
        return min(xs), min(ys), max(xs), max(ys)

    def _best_reference_match(
        self,
        embedding: np.ndarray,
    ) -> tuple[str, float] | None:
        best_name, best_score = self._best_reference_score(embedding)
        if best_score >= self.threshold:
            return best_name, best_score
        return None

    def _best_reference_score(self, embedding: np.ndarray) -> tuple[str, float]:
        best_name = ""
        best_score = -1.0
        for reference in self.references:
            score = float(np.dot(embedding, reference.embedding))
            if score > best_score:
                best_name = reference.name
                best_score = score
        return best_name, best_score

    def _load_references(self, known_people: list[Any]) -> list[_Reference]:
        references: list[_Reference] = []
        for person in known_people:
            if not isinstance(person, dict):
                continue
            name = str(person.get("name", "")).strip()
            if not name:
                continue
            for image_path in self._reference_paths(person):
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(
                        "Known person reference image not readable: %s",
                        image_path,
                    )
                    continue
                faces = self._analyze_faces(image)
                face = max(faces, key=self._face_quality, default=None)
                embedding = self._face_embedding(face)
                if embedding is None:
                    logger.warning(
                        "No recognizable face in known-person image: %s",
                        image_path,
                    )
                    continue
                references.append(_Reference(name=name, embedding=embedding))
        if references:
            logger.info(
                "Loaded %d InsightFace known-person reference image(s)",
                len(references),
            )
        return references

    @staticmethod
    def _person_for_face(
        face_bbox: tuple[float, float, float, float],
        people: list[TrackedObject],
        matched_track_ids: set[int],
    ) -> TrackedObject | None:
        center_x = (face_bbox[0] + face_bbox[2]) / 2.0
        center_y = (face_bbox[1] + face_bbox[3]) / 2.0
        candidates = [
            person
            for person in people
            if person.track_id not in matched_track_ids
            and person.bbox_xyxy[0] <= center_x <= person.bbox_xyxy[2]
            and person.bbox_xyxy[1] <= center_y <= person.bbox_xyxy[3]
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda person: (
                person.bbox_xyxy[2] - person.bbox_xyxy[0]
            )
            * (person.bbox_xyxy[3] - person.bbox_xyxy[1]),
        )

    @classmethod
    def _face_quality(cls, face: Any) -> float:
        bbox = cls._face_bbox(face)
        if bbox is None:
            return -1.0
        area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        return area * max(0.0, cls._face_detection_score(face))

    @classmethod
    def _face_bbox(
        cls,
        face: Any,
    ) -> tuple[float, float, float, float] | None:
        raw_bbox = cls._face_value(face, "bbox")
        if raw_bbox is None:
            return None
        bbox = np.asarray(raw_bbox, dtype=np.float32).reshape(-1)
        if bbox.size < 4 or not np.isfinite(bbox[:4]).all():
            return None
        return tuple(float(value) for value in bbox[:4])

    @classmethod
    def _face_embedding(cls, face: Any) -> np.ndarray | None:
        if face is None:
            return None
        raw_embedding = cls._face_value(face, "normed_embedding")
        if raw_embedding is None:
            raw_embedding = cls._face_value(face, "embedding")
        if raw_embedding is None:
            return None
        embedding = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.size == 0 or not np.isfinite(embedding).all() or norm <= 0:
            return None
        return embedding / norm

    @classmethod
    def _face_detection_score(cls, face: Any) -> float:
        raw_score = cls._face_value(face, "det_score")
        try:
            return float(raw_score)
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _face_value(face: Any, key: str) -> Any:
        if face is None:
            return None
        value = getattr(face, key, None)
        if value is not None:
            return value
        if isinstance(face, dict):
            return face.get(key)
        return None

    def _providers(self, available: list[str]) -> list[str]:
        if self.configured_providers:
            providers = [
                provider
                for provider in self.configured_providers
                if provider in available
            ]
            if providers:
                return providers

        use_cuda = self.device != "cpu" and "CUDAExecutionProvider" in available
        if use_cuda:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _context_id(self, providers: list[str]) -> int:
        if "CUDAExecutionProvider" not in providers:
            return -1
        if ":" not in self.device:
            return 0
        try:
            return int(self.device.rsplit(":", 1)[1])
        except ValueError:
            return 0

    @staticmethod
    def _parse_detection_size(value: Any) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return max(32, int(value[0])), max(32, int(value[1]))
        size = max(32, int(value))
        return size, size

    @staticmethod
    def _parse_orientations(value: Any) -> list[str]:
        valid = {"none", "cw90", "ccw90", "180"}
        if isinstance(value, str):
            raw_items = [item.strip() for item in value.split(",")]
        elif isinstance(value, (list, tuple)):
            raw_items = [str(item).strip() for item in value]
        else:
            raw_items = []

        orientations: list[str] = []
        for item in raw_items:
            if item in valid and item not in orientations:
                orientations.append(item)
        return orientations or ["none", "cw90", "ccw90", "180"]

    @staticmethod
    def _resolve_path(value: Any) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[1] / path

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
                for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                    paths.extend(sorted(path.glob(pattern)))
            else:
                paths.append(path)
        return paths


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    intersection = width * height
    if intersection <= 0:
        return 0.0

    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _bbox_area_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    larger = max(first_area, second_area)
    if larger <= 0:
        return 0.0
    return min(first_area, second_area) / larger
