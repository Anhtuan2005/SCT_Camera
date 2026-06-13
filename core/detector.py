"""YOLOv11 detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Detection:
    """A single YOLO detection."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class YOLOv11Detector:
    """Load and reuse a YOLOv11 model for detection and tracking."""

    def __init__(self, settings: dict[str, Any]) -> None:
        detection_settings = settings.get("detection", {})
        self.model_path = str(detection_settings.get("model", "yolo11n.pt"))
        self.confidence = float(detection_settings.get("confidence", 0.4))
        self.class_confidences = _parse_class_confidences(
            detection_settings.get("class_confidences", {})
        )
        self.class_ids = [int(item) for item in detection_settings.get("classes", [0, 15, 16, 2, 3, 5, 7])]
        self.iou = float(detection_settings.get("iou", 0.5))
        self.person_max_aspect_ratio = float(
            detection_settings.get("person_max_aspect_ratio", 4.0)
        )
        self.device = str(detection_settings.get("device", "cuda:0"))
        self.use_half = bool(detection_settings.get("half", True))
        self.imgsz = int(detection_settings.get("imgsz", 640))
        self.inference_lock = RLock()

        self._configure_device()
        logger.info("Loading YOLO model %s on %s", self.model_path, self.device)

        from ultralytics import YOLO

        self.model = YOLO(self.model_path)
        try:
            self.model.to(self.device)
        except Exception as exc:
            logger.warning("Could not move YOLO model to %s: %s", self.device, exc)
            self.device = "cpu"
            self.use_half = False

        self.names = self._extract_names()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect configured classes in a BGR frame."""
        with self.inference_lock:
            results = self.model.predict(
                frame_bgr,
                conf=_predict_confidence_threshold(
                    self.confidence,
                    self.class_confidences,
                ),
                iou=self.iou,
                classes=self.class_ids,
                device=self.device,
                half=self.use_half and self.device.startswith("cuda"),
                imgsz=self.imgsz,
                verbose=False,
            )

        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []
        for xyxy, conf, cls_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
            class_id = int(cls_id)
            class_name = self.class_name(class_id)
            bbox = tuple(float(value) for value in xyxy)
            confidence = float(conf)
            if not _passes_class_confidence(
                confidence,
                class_name,
                self.confidence,
                self.class_confidences,
            ):
                continue
            if not _valid_detection_shape(
                bbox,
                class_name,
                self.person_max_aspect_ratio,
            ):
                continue
            detections.append(
                Detection(
                    bbox_xyxy=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections

    def update_settings(self, settings: dict[str, Any]) -> None:
        """Apply detection settings and reload the model when required."""
        detection_settings = settings.get("detection", {})
        model_path = str(detection_settings.get("model", self.model_path))
        requested_device = str(detection_settings.get("device", self.device))
        requested_half = bool(detection_settings.get("half", self.use_half))

        previous_device = self.device
        previous_half = self.use_half
        self.device = requested_device
        self.use_half = requested_half
        self._configure_device()

        if model_path != self.model_path or self.device != previous_device:
            logger.info("Reloading YOLO model %s on %s", model_path, self.device)
            try:
                from ultralytics import YOLO

                model = YOLO(model_path)
                model.to(self.device)
            except Exception:
                self.device = previous_device
                self.use_half = previous_half
                raise
            with self.inference_lock:
                self.model = model
                self.model_path = model_path
                self.names = self._extract_names()

        self.confidence = float(detection_settings.get("confidence", self.confidence))
        self.class_confidences = _parse_class_confidences(
            detection_settings.get("class_confidences", self.class_confidences)
        )
        self.class_ids = [
            int(item) for item in detection_settings.get("classes", self.class_ids)
        ]
        self.iou = float(detection_settings.get("iou", self.iou))
        self.person_max_aspect_ratio = float(
            detection_settings.get(
                "person_max_aspect_ratio",
                self.person_max_aspect_ratio,
            )
        )
        self.imgsz = int(detection_settings.get("imgsz", self.imgsz))

    def class_name(self, class_id: int) -> str:
        """Return a human-readable class name for an integer class id."""
        return self.names.get(class_id, str(class_id))

    def _configure_device(self) -> None:
        if not self.device.startswith("cuda"):
            self.use_half = False
            return

        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable; falling back to CPU")
                self.device = "cpu"
                self.use_half = False
        except Exception as exc:
            logger.warning("Could not inspect CUDA availability: %s", exc)
            self.device = "cpu"
            self.use_half = False

    def _extract_names(self) -> dict[int, str]:
        raw_names = getattr(self.model, "names", {})
        if isinstance(raw_names, dict):
            return {int(key): str(value) for key, value in raw_names.items()}
        return {index: str(value) for index, value in enumerate(raw_names)}


def _parse_class_confidences(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, float] = {}
    for class_name, threshold in value.items():
        try:
            parsed[str(class_name)] = max(0.0, min(1.0, float(threshold)))
        except (TypeError, ValueError):
            continue
    return parsed


def _predict_confidence_threshold(
    default_confidence: float,
    class_confidences: dict[str, float],
) -> float:
    thresholds = [default_confidence, *class_confidences.values()]
    return max(0.0, min(thresholds))


def _passes_class_confidence(
    confidence: float,
    class_name: str,
    default_confidence: float,
    class_confidences: dict[str, float],
) -> bool:
    threshold = class_confidences.get(class_name, default_confidence)
    return confidence >= threshold


def _valid_detection_shape(
    bbox_xyxy: tuple[float, float, float, float],
    class_name: str,
    person_max_aspect_ratio: float,
) -> bool:
    if class_name != "person" or person_max_aspect_ratio <= 0:
        return True

    x1, y1, x2, y2 = bbox_xyxy
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0 or height <= 0:
        return False
    aspect_ratio = max(width / height, height / width)
    return aspect_ratio <= person_max_aspect_ratio
