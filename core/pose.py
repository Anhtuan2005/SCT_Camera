"""Optional person pose estimation using an Ultralytics pose model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from core.tracker import TrackedObject, _bbox_iou
from utils.logger import get_logger

logger = get_logger(__name__)


class PoseEstimator:
    """Attach pose keypoints to tracked person objects when a pose model is available."""

    def __init__(
        self,
        settings: dict[str, Any],
        inference_lock: RLock,
        device: str,
        use_half: bool,
    ) -> None:
        self.inference_lock = inference_lock
        self.device = device
        self.use_half = use_half
        self.model: Any | None = None
        self.enabled = False
        self.model_path = ""
        self.confidence = 0.35
        self.imgsz = 640
        self.match_iou = 0.25
        self.allow_download = False
        self.update_settings(settings, device=device, use_half=use_half)

    def update_settings(
        self,
        settings: dict[str, Any],
        device: str | None = None,
        use_half: bool | None = None,
    ) -> None:
        previous_enabled = self.enabled
        previous_model_path = self.model_path
        previous_device = self.device
        previous_use_half = self.use_half
        if device is not None:
            self.device = device
        if use_half is not None:
            self.use_half = use_half

        pose_settings = settings.get("pose", {})
        self.enabled = bool(pose_settings.get("enabled", self.enabled))
        self.model_path = str(pose_settings.get("model", "yolo11n-pose.pt"))
        self.confidence = float(pose_settings.get("confidence", 0.35))
        self.imgsz = int(pose_settings.get("imgsz", settings.get("detection", {}).get("imgsz", 640)))
        self.match_iou = float(pose_settings.get("match_iou", 0.25))
        self.allow_download = bool(pose_settings.get("allow_download", False))
        if (
            self.enabled != previous_enabled
            or self.model_path != previous_model_path
            or self.device != previous_device
            or self.use_half != previous_use_half
        ):
            self.model = None

    def attach(self, frame_bgr: np.ndarray, objects: list[TrackedObject]) -> list[TrackedObject]:
        if not self.enabled or not any(obj.class_name == "person" for obj in objects):
            return objects
        if not self._ensure_model():
            return objects

        assert self.model is not None
        with self.inference_lock:
            results = self.model.predict(
                frame_bgr,
                conf=self.confidence,
                device=self.device,
                half=self.use_half and self.device.startswith("cuda"),
                imgsz=self.imgsz,
                verbose=False,
            )
        if not results:
            return objects

        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None or len(boxes) == 0:
            return objects

        pose_boxes = [tuple(float(value) for value in xyxy) for xyxy in boxes.xyxy.cpu().tolist()]
        raw_keypoints = keypoints.data.cpu().tolist()
        attached: list[TrackedObject] = []
        used_pose_indexes: set[int] = set()
        for obj in objects:
            if obj.class_name != "person":
                attached.append(obj)
                continue
            pose_index = self._best_pose_index(obj, pose_boxes, used_pose_indexes)
            if pose_index is None:
                attached.append(obj)
                continue
            used_pose_indexes.add(pose_index)
            attached.append(replace(obj, pose_keypoints=self._clean_keypoints(raw_keypoints[pose_index])))
        return attached

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if not self.enabled:
            return False

        path = Path(self.model_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() and not self.allow_download:
            logger.warning(
                "Pose model %s not found; set pose.allow_download=true or place the file locally",
                self.model_path,
            )
            self.enabled = False
            return False

        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            try:
                self.model.to(self.device)
            except Exception as exc:
                logger.warning("Could not move pose model to %s: %s", self.device, exc)
            return True
        except Exception as exc:
            logger.warning("Pose model disabled: %s", exc)
            self.enabled = False
            self.model = None
            return False

    def _best_pose_index(
        self,
        obj: TrackedObject,
        pose_boxes: list[tuple[float, float, float, float]],
        used_pose_indexes: set[int],
    ) -> int | None:
        best_index: int | None = None
        best_iou = self.match_iou
        for index, pose_box in enumerate(pose_boxes):
            if index in used_pose_indexes:
                continue
            iou = _bbox_iou(obj.bbox_xyxy, pose_box)
            if iou > best_iou:
                best_index = index
                best_iou = iou
        return best_index

    @staticmethod
    def _clean_keypoints(raw_keypoints: list[list[float]]) -> list[tuple[float, float, float]]:
        cleaned: list[tuple[float, float, float]] = []
        for point in raw_keypoints:
            if len(point) >= 3:
                cleaned.append((float(point[0]), float(point[1]), float(point[2])))
            elif len(point) >= 2:
                cleaned.append((float(point[0]), float(point[1]), 1.0))
        return cleaned
