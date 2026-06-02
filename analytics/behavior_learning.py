"""Behavior event logging and lightweight supervised risk scoring."""

from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.tracker import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


FEATURE_NAMES = [
    "duration",
    "threshold_seconds",
    "duration_ratio",
    "near_seconds",
    "score",
    "score_ratio",
    "pacing_passes",
    "object_confidence",
    "bbox_area_ratio",
    "path_length_ratio",
    "net_distance_ratio",
    "displacement_ratio",
    "speed_ratio",
    "has_vehicle_signal",
    "vehicle_started_moving",
    "moving_same_direction",
    "pose_push_contact",
    "object_count",
    "zone_configured",
    "is_person",
    "is_vehicle",
    "is_asset",
]

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
ASSET_CLASSES = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "handbag",
    "suitcase",
}


@dataclass(frozen=True)
class BehaviorRiskModel:
    """Small logistic model trained by scripts/train_behavior_classifier.py."""

    feature_names: list[str]
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "BehaviorRiskModel":
        data = np.load(path, allow_pickle=False)
        feature_names = [str(item) for item in data["feature_names"].tolist()]
        return cls(
            feature_names=feature_names,
            weights=data["weights"].astype(float),
            bias=float(data["bias"]),
            mean=data["mean"].astype(float),
            scale=data["scale"].astype(float),
        )

    def score(self, features: dict[str, float]) -> float:
        vector = np.array([float(features.get(name, 0.0)) for name in self.feature_names], dtype=float)
        vector = (vector - self.mean) / self.scale
        logit = float(vector @ self.weights + self.bias)
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_value = math.exp(logit)
        return exp_value / (1.0 + exp_value)


class BehaviorLearningService:
    """Append candidate behavior events and optionally score them with a model."""

    def __init__(self, settings: dict[str, Any]) -> None:
        learning = settings.get("behavior_learning", {})
        self.enabled = bool(learning.get("enabled", True))
        self.log_candidates = bool(learning.get("log_candidates", True))
        self.gate_alerts = bool(learning.get("gate_alerts", False))
        self.min_risk_score = float(learning.get("min_risk_score", 0.65))
        self.event_log_path = Path(str(learning.get("event_log_path", "data/behavior_events.jsonl")))
        self.model_path = Path(str(learning.get("model_path", "models/behavior_classifier.npz")))
        self._lock = threading.Lock()
        self._model_mtime: float | None = None
        self._model: BehaviorRiskModel | None = None
        self._load_model_if_available(force=True)

    def update_settings(self, settings: dict[str, Any]) -> None:
        self.__init__(settings)

    def enrich_alerts(
        self,
        alerts: list[dict[str, Any]],
        objects: list[TrackedObject],
        camera_config: dict[str, Any],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Log alerts as training candidates and add risk metadata."""
        if not self.enabled or not alerts:
            return alerts

        self._load_model_if_available()
        enriched: list[dict[str, Any]] = []
        for alert in alerts:
            record = self._event_record(alert, objects, camera_config, frame_shape, timestamp)
            risk_score = self._score(record["features"])
            alert["behavior_event_id"] = record["event_id"]
            alert["behavior_features"] = record["features"]
            if risk_score is not None:
                alert["behavior_risk_score"] = round(risk_score, 4)
                alert["behavior_model_path"] = str(self.model_path)
                if self.gate_alerts and risk_score < self.min_risk_score:
                    alert["behavior_suppressed"] = True
                    alert["behavior_suppression_reason"] = (
                        f"risk_score {risk_score:.3f} below {self.min_risk_score:.3f}"
                    )
            record["risk_score"] = risk_score
            record["gate_alerts"] = self.gate_alerts
            record["min_risk_score"] = self.min_risk_score
            if self.log_candidates:
                self._append_record(record)
            if not alert.get("behavior_suppressed"):
                enriched.append(alert)
        return enriched

    def _score(self, features: dict[str, float]) -> float | None:
        if self._model is None:
            return None
        try:
            return self._model.score(features)
        except Exception as exc:
            logger.warning("Behavior model scoring failed: %s", exc)
            return None

    def _load_model_if_available(self, force: bool = False) -> None:
        if not self.model_path.exists():
            self._model = None
            self._model_mtime = None
            return
        mtime = self.model_path.stat().st_mtime
        if not force and self._model is not None and self._model_mtime == mtime:
            return
        try:
            self._model = BehaviorRiskModel.load(self.model_path)
            self._model_mtime = mtime
            logger.info("Loaded behavior risk model: %s", self.model_path)
        except Exception as exc:
            self._model = None
            self._model_mtime = None
            logger.warning("Could not load behavior risk model %s: %s", self.model_path, exc)

    def _append_record(self, record: dict[str, Any]) -> None:
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.event_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _event_record(
        self,
        alert: dict[str, Any],
        objects: list[TrackedObject],
        camera_config: dict[str, Any],
        frame_shape: tuple[int, int, int],
        timestamp: datetime,
    ) -> dict[str, Any]:
        features = extract_behavior_features(alert, objects, camera_config, frame_shape)
        return {
            "event_id": str(uuid.uuid4()),
            "timestamp": timestamp.isoformat(),
            "camera_id": str(camera_config.get("camera_id", alert.get("camera_id", "unknown"))),
            "camera_name": str(camera_config.get("name", alert.get("camera_name", "unknown"))),
            "alert_type": str(alert.get("type", "unknown")),
            "track_id": alert.get("track_id"),
            "zone_id": alert.get("zone_id"),
            "zone_name": alert.get("zone_name"),
            "line_id": alert.get("line_id"),
            "line_name": alert.get("line_name"),
            "features": features,
            "alert": {
                key: value
                for key, value in alert.items()
                if key not in {"frame", "behavior_features"} and not key.startswith("_")
            },
            "label": None,
            "label_notes": "",
        }


def extract_behavior_features(
    alert: dict[str, Any],
    objects: list[TrackedObject],
    camera_config: dict[str, Any],
    frame_shape: tuple[int, int, int],
) -> dict[str, float]:
    """Convert a behavior candidate into stable numeric features."""
    height, width = frame_shape[:2]
    frame_area = max(float(width * height), 1.0)
    diagonal = max(math.hypot(width, height), 1.0)
    obj = _find_object(objects, alert.get("track_id"))
    class_name = str(alert.get("class_name") or alert.get("class_name") or (obj.class_name if obj else ""))
    vehicle_signal = bool(
        alert.get("vehicle_started_moving")
        or alert.get("moving_same_direction")
        or alert.get("pose_push_contact")
    )

    duration = _number(alert.get("duration"))
    threshold = _number(alert.get("threshold_seconds"))
    near_seconds = _number(alert.get("near_seconds"))
    score = _number(alert.get("score"))
    score_threshold = max(_number(alert.get("score_threshold")), 1.0)

    bbox_area_ratio = 0.0
    path_length_ratio = 0.0
    net_distance_ratio = 0.0
    displacement_ratio = 0.0
    speed_ratio = 0.0
    confidence = _number(alert.get("identity_score"))
    if obj is not None:
        x1, y1, x2, y2 = obj.bbox_xyxy
        bbox_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / frame_area
        path_length, net_distance, displacement = _motion_metrics(obj.center_history)
        path_length_ratio = path_length / diagonal
        net_distance_ratio = net_distance / diagonal
        displacement_ratio = displacement / diagonal
        speed_ratio = path_length_ratio / max(len(obj.center_history) - 1, 1)
        confidence = obj.confidence
        class_name = obj.class_name

    features = {
        "duration": duration,
        "threshold_seconds": threshold,
        "duration_ratio": duration / max(threshold, 1.0),
        "near_seconds": near_seconds,
        "score": score,
        "score_ratio": score / score_threshold,
        "pacing_passes": _number(alert.get("pacing_passes")),
        "object_confidence": confidence,
        "bbox_area_ratio": bbox_area_ratio,
        "path_length_ratio": path_length_ratio,
        "net_distance_ratio": net_distance_ratio,
        "displacement_ratio": displacement_ratio,
        "speed_ratio": speed_ratio,
        "has_vehicle_signal": 1.0 if vehicle_signal else 0.0,
        "vehicle_started_moving": 1.0 if bool(alert.get("vehicle_started_moving")) else 0.0,
        "moving_same_direction": 1.0 if bool(alert.get("moving_same_direction")) else 0.0,
        "pose_push_contact": 1.0 if bool(alert.get("pose_push_contact")) else 0.0,
        "object_count": float(len(objects)),
        "zone_configured": 1.0 if camera_config.get("zones") else 0.0,
        "is_person": 1.0 if class_name == "person" else 0.0,
        "is_vehicle": 1.0 if class_name in VEHICLE_CLASSES else 0.0,
        "is_asset": 1.0 if class_name in ASSET_CLASSES else 0.0,
    }
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def _find_object(objects: list[TrackedObject], track_id: Any) -> TrackedObject | None:
    try:
        wanted = int(track_id)
    except (TypeError, ValueError):
        return None
    for obj in objects:
        if obj.track_id == wanted:
            return obj
    return None


def _motion_metrics(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    path_length = sum(
        math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
        for index in range(1, len(points))
    )
    net_distance = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    displacement = max(math.hypot(x - points[0][0], y - points[0][1]) for x, y in points)
    return path_length, net_distance, displacement


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
