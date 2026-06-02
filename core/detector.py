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
        self.class_ids = [int(item) for item in detection_settings.get("classes", [0, 15, 16, 2, 3, 5, 7])]
        self.iou = float(detection_settings.get("iou", 0.5))
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
                conf=self.confidence,
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
            detections.append(
                Detection(
                    bbox_xyxy=tuple(float(value) for value in xyxy),
                    confidence=float(conf),
                    class_id=class_id,
                    class_name=self.class_name(class_id),
                )
            )
        return detections

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
