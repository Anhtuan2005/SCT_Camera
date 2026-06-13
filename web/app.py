"""FastAPI app factory and runtime configuration management."""

from __future__ import annotations

import copy
import csv
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from analytics.behavior_engine import BehaviorEngine
from core.detector import YOLOv11Detector
from core.frame_buffer import FrameBuffer
from core.pipeline import CameraPipeline
from core.pose import PoseEstimator
from notifications.alert_manager import AlertManager
from utils.logger import get_logger
from web.routes import config_api, dashboard, stream

logger = get_logger(__name__)

ALERT_CHANNELS = {"telegram", "discord"}
MAX_QUALITY_RUNTIME_SETTINGS: dict[str, Any] = {
    "detection": {
        "model": "yolo11n.pt",
        "confidence": 0.25,
        "class_confidences": {
            "backpack": 0.12,
            "bicycle": 0.10,
            "bus": 0.15,
            "car": 0.15,
            "cat": 0.12,
            "dog": 0.12,
            "handbag": 0.12,
            "motorcycle": 0.10,
            "person": 0.20,
            "suitcase": 0.12,
            "truck": 0.15,
        },
        "classes": [0, 1, 2, 3, 5, 7, 24, 26, 28, 15, 16],
        "device": "cuda:0",
        "half": True,
        "imgsz": 640,
        "iou": 0.55,
        "person_max_aspect_ratio": 4.0,
    },
    "pose": {
        "enabled": True,
        "model": "yolo11n-pose.pt",
        "allow_download": True,
        "confidence": 0.2,
        "imgsz": 640,
        "match_iou": 0.2,
    },
    "pipeline": {
        "frame_skip": 2,
        "ai_max_fps": 10,
        "analysis_stale_after_ms": 500,
        "analysis_timeout_min_seconds": 5.0,
        "processing_max_height": 720,
    },
    "tracking": {
        "track_high_thresh": 0.10,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.10,
        "track_buffer": 90,
        "track_grace_frames": 3,
        "duplicate_iou_threshold": 0.85,
        "duplicate_containment_threshold": 0.7,
        "camera_motion_compensation": {
            "enabled": False,
        },
    },
    "identity": {
        "model": "buffalo_sc",
        "device": "cuda:0",
        "similarity_threshold": 0.45,
        "detection_size": 320,
        "min_face_size": 30,
        "recognition_interval_frames": 10,
        "unknown_confirmation_attempts": 5,
        "known_memory_frames": 90,
        "known_memory_distance_ratio": 0.18,
        "known_memory_min_area_ratio": 0.25,
        "orientations": ["none"],
        "reference_orientations": ["none", "cw90", "ccw90", "180"],
        "orientations_per_attempt": 1,
    },
}


def _enforce_required_runtime_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Keep required vision features enabled in every configuration path."""
    pose = settings.setdefault("pose", {})
    pose.setdefault("enabled", True)
    pose.setdefault("allow_download", True)
    return settings


def _safe_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return value.strip("_") or f"id_{uuid.uuid4().hex[:8]}"


def _normalize_notification_channels(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = ["telegram"]
    channels = []
    for item in raw_items:
        channel = str(item).strip().lower()
        if channel in ALERT_CHANNELS and channel not in channels:
            channels.append(channel)
    return channels or ["telegram"]


def load_settings(path: Path) -> dict[str, Any]:
    """Load global YAML settings."""
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("settings.yaml must contain a mapping")
    return data


def load_camera_configs(cameras_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all camera YAML files from a directory."""
    cameras_dir.mkdir(parents=True, exist_ok=True)
    cameras: dict[str, dict[str, Any]] = {}
    for path in sorted(cameras_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            logger.warning("Skipping invalid camera config %s", path)
            continue
        camera_id = _safe_id(str(data.get("camera_id") or path.stem))
        if camera_id in cameras:
            logger.warning("Skipping duplicate camera_id %s from %s", camera_id, path)
            continue
        data["camera_id"] = camera_id
        data.setdefault("name", camera_id)
        data.setdefault("enabled", False)
        data["notification_channels"] = _normalize_notification_channels(data.get("notification_channels"))
        if not isinstance(data.get("zones", []), list):
            logger.warning("Camera %s has invalid zones; using an empty list", camera_id)
            data["zones"] = []
        else:
            data.setdefault("zones", [])
        if not isinstance(data.get("lines", []), list):
            logger.warning("Camera %s has invalid lines; using an empty list", camera_id)
            data["lines"] = []
        else:
            data.setdefault("lines", [])
        cameras[camera_id] = data
    return cameras


def create_app(runtime: "RuntimeState") -> FastAPI:
    """Create the FastAPI dashboard application."""
    app = FastAPI(title="SCT Camera", version="1.0.0")
    app.state.runtime = runtime

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    app.include_router(dashboard.router)
    app.include_router(stream.router)
    app.include_router(config_api.router)

    @app.on_event("startup")
    async def startup() -> None:
        await runtime.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await runtime.stop()

    return app


class RuntimeState:
    """In-memory runtime state shared by FastAPI routes and pipelines."""

    def __init__(
        self,
        settings: dict[str, Any],
        cameras: dict[str, dict[str, Any]],
        settings_path: Path,
        cameras_dir: Path,
    ) -> None:
        self.settings = _enforce_required_runtime_settings(copy.deepcopy(settings))
        self.cameras = copy.deepcopy(cameras)
        self.settings_path = settings_path
        self.cameras_dir = cameras_dir
        self._lock = threading.RLock()

        max_height = int(self.settings.get("pipeline", {}).get("stream_max_height", 720))
        self.frame_buffers = {
            camera_id: FrameBuffer(max_height=max_height) for camera_id in self.cameras
        }
        self.detector = YOLOv11Detector(self.settings)
        self.pose_estimator = PoseEstimator(
            self.settings,
            self.detector.inference_lock,
            self.detector.device,
            self.detector.use_half,
        )
        self.behavior_engine = BehaviorEngine(self.settings)
        self.identity_resolver = self.behavior_engine.identity_resolver
        self.alert_manager = AlertManager(self.settings)
        self.pipelines: dict[str, CameraPipeline] = {}

    def behavior_event_log_path(self) -> Path:
        """Return the configured behavior event JSONL path."""
        learning = self.settings.get("behavior_learning", {})
        path = Path(str(learning.get("event_log_path", "data/behavior_events.jsonl")))
        if not path.is_absolute():
            path = self.settings_path.parent.parent / path
        return path

    def list_behavior_events(self, limit: int = 100, unlabeled_only: bool = False) -> list[dict[str, Any]]:
        """Return recent behavior learning events from JSONL."""
        path = self.behavior_event_log_path()
        if not path.exists():
            return []
        labels = self._load_behavior_labels()
        events: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_id = str(event.get("event_id", ""))
                if event_id in labels:
                    event["label"] = labels[event_id]["label"]
                    event["label_notes"] = labels[event_id]["notes"]
                if unlabeled_only and event.get("label"):
                    continue
                event.pop("features", None)
                events.append(event)
        return list(reversed(events))[:limit]

    def label_behavior_event(self, event_id: str, label: str, notes: str = "") -> dict[str, Any]:
        """Upsert an event label in a companion CSV file."""
        event_id = str(event_id).strip()
        label = str(label).strip()
        if not event_id or not label:
            raise ValueError("event_id and label are required")
        labels_path = self.behavior_event_log_path().with_name("behavior_labels.csv")
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str]] = []
        if labels_path.exists():
            with labels_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("event_id"):
                        rows.append(
                            {
                                "event_id": str(row.get("event_id", "")),
                                "label": str(row.get("label", "")),
                                "notes": str(row.get("notes", "")),
                            }
                        )
        rows = [row for row in rows if row["event_id"] != event_id]
        rows.append({"event_id": event_id, "label": label, "notes": notes})
        with labels_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["event_id", "label", "notes"])
            writer.writeheader()
            writer.writerows(rows)
        return {"event_id": event_id, "label": label, "notes": notes}

    def _load_behavior_labels(self) -> dict[str, dict[str, str]]:
        labels_path = self.behavior_event_log_path().with_name("behavior_labels.csv")
        if not labels_path.exists():
            return {}
        labels: dict[str, dict[str, str]] = {}
        with labels_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                event_id = str(row.get("event_id", "")).strip()
                if not event_id:
                    continue
                labels[event_id] = {
                    "label": str(row.get("label", "")).strip(),
                    "notes": str(row.get("notes", "")).strip(),
                }
        return labels

    async def start(self) -> None:
        """Start alert manager and enabled camera pipelines."""
        await self.alert_manager.start()
        with self._lock:
            configs = list(self.cameras.values())
        for camera_config in configs:
            if bool(camera_config.get("enabled", False)):
                self._restart_pipeline(camera_config)

    def toggle_detection(self, camera_id: str, active: bool) -> dict[str, Any] | None:
        """Pause or resume detection for a single camera without changing persisted config.

        When *active* is False the pipeline is stopped and the frame buffer is
        set to ``paused``.  When *active* is True the pipeline is restarted
        using the current in-memory config.
        """
        with self._lock:
            config = self.cameras.get(camera_id)
            if config is None:
                return None
            buffer = self.frame_buffers.get(camera_id)

        if active:
            # Resume: restart the pipeline with current config
            self._restart_pipeline(config)
        else:
            # Pause: stop the pipeline and mark as paused
            with self._lock:
                pipeline = self.pipelines.pop(camera_id, None)
            if pipeline:
                pipeline.stop()
            if buffer:
                buffer.set_status("paused", "Detection paused by user")

        return self._public_camera(copy.deepcopy(config))

    def toggle_all_detection(self, active: bool) -> list[dict[str, Any]]:
        """Pause or resume detection for **all** enabled cameras."""
        with self._lock:
            configs = [
                copy.deepcopy(cfg)
                for cfg in self.cameras.values()
                if bool(cfg.get("enabled", False))
            ]
        results = []
        for config in configs:
            result = self.toggle_detection(str(config["camera_id"]), active)
            if result:
                results.append(result)
        return results

    async def stop(self) -> None:
        """Stop all camera pipelines and alert worker."""
        with self._lock:
            pipelines = list(self.pipelines.values())
            self.pipelines.clear()
        for pipeline in pipelines:
            pipeline.stop()
        await self.alert_manager.stop()

    def list_cameras(self) -> list[dict[str, Any]]:
        """Return public camera config plus runtime status for dashboard/API."""
        with self._lock:
            configs = [copy.deepcopy(config) for config in self.cameras.values()]
        cameras = [self._public_camera(config) for config in sorted(configs, key=lambda item: item["camera_id"])]
        return cameras

    def get_camera(self, camera_id: str) -> dict[str, Any] | None:
        """Return one camera config with runtime status."""
        with self._lock:
            config = self.cameras.get(camera_id)
            if config is None:
                return None
            return self._public_camera(copy.deepcopy(config))

    def get_raw_camera(self, camera_id: str) -> dict[str, Any] | None:
        """Return a deep copy of one camera config."""
        with self._lock:
            config = self.cameras.get(camera_id)
            return copy.deepcopy(config) if config is not None else None

    def upsert_camera(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create or update a camera config, persist it, and restart its pipeline."""
        camera_id = str(payload.get("camera_id") or self._new_camera_id(payload.get("name", "camera")))
        camera_id = self._safe_id(camera_id)
        with self._lock:
            is_new_source = camera_id not in self.cameras
        if is_new_source:
            self.update_settings(copy.deepcopy(MAX_QUALITY_RUNTIME_SETTINGS))

        with self._lock:
            existing = copy.deepcopy(self.cameras.get(camera_id, {}))
            config = {
                "camera_id": camera_id,
                "name": str(payload.get("name") or existing.get("name") or camera_id),
                "source": payload.get("source", existing.get("source", 0)),
                "enabled": bool(payload.get("enabled", existing.get("enabled", True))),
                "frame_rotation": str(
                    payload.get("frame_rotation", existing.get("frame_rotation", "none"))
                ),
                "unknown_person_policy": str(
                    payload.get(
                        "unknown_person_policy",
                        existing.get("unknown_person_policy", "face_match"),
                    )
                ),
                "auto_global_zone": bool(
                    payload.get("auto_global_zone", existing.get("auto_global_zone", True))
                ),
                "vision_profile": str(
                    payload.get(
                        "vision_profile",
                        existing.get("vision_profile", "max_quality_realtime"),
                    )
                ),
                "notification_channels": _normalize_notification_channels(
                    payload.get("notification_channels", existing.get("notification_channels"))
                ),
                "zones": payload.get("zones", existing.get("zones", [])),
                "lines": payload.get("lines", existing.get("lines", [])),
            }
            self.cameras[camera_id] = config
            self.frame_buffers.setdefault(
                camera_id,
                FrameBuffer(max_height=int(self.settings.get("pipeline", {}).get("stream_max_height", 720))),
            )
            self._save_camera_config(config)

        self._restart_pipeline(config)
        return self._public_camera(config)

    def set_camera_enabled(self, camera_id: str, enabled: bool) -> dict[str, Any] | None:
        """Enable or disable a camera, persist the config, and update its pipeline."""
        with self._lock:
            config = self.cameras.get(camera_id)
            if config is None:
                return None
            config["enabled"] = bool(enabled)
            saved = copy.deepcopy(config)
            self._save_camera_config(config)

        self._restart_pipeline(saved)
        return self._public_camera(saved)

    def delete_camera(self, camera_id: str) -> bool:
        """Remove a camera config and stop its pipeline."""
        with self._lock:
            if camera_id not in self.cameras:
                return False
            pipeline = self.pipelines.pop(camera_id, None)
            self.cameras.pop(camera_id, None)
            self.frame_buffers.pop(camera_id, None)
            path = self.cameras_dir / f"{self._safe_id(camera_id)}.yaml"
        if pipeline:
            pipeline.stop()
        if path.exists():
            path.unlink()
        return True

    def upsert_zone(self, camera_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Create or update a camera zone and persist the camera config."""
        with self._lock:
            config = self.cameras.get(camera_id)
            if config is None:
                return None
            zone_id = self._safe_id(str(payload.get("id") or f"zone_{uuid.uuid4().hex[:8]}"))
            zone = {
                "id": zone_id,
                "name": str(payload.get("name") or zone_id),
                "type": str(payload.get("type") or payload.get("zone_type") or "all"),
                "polygon": payload.get("polygon", []),
            }
            if payload.get("threshold_seconds") not in (None, ""):
                zone["threshold_seconds"] = float(payload["threshold_seconds"])
            zones = [item for item in config.get("zones", []) if str(item.get("id")) != zone_id]
            zones.append(zone)
            config["zones"] = zones
            self._save_camera_config(config)
            saved = copy.deepcopy(zone)
        self._sync_pipeline_config(camera_id)
        return saved

    def delete_zone(self, camera_id: str, zone_id: str) -> bool:
        """Delete a zone from a camera config."""
        with self._lock:
            config = self.cameras.get(camera_id)
            if config is None:
                return False
            original_count = len(config.get("zones", []))
            config["zones"] = [item for item in config.get("zones", []) if str(item.get("id")) != zone_id]
            if len(config["zones"]) == original_count:
                return False
            self._save_camera_config(config)
        self._sync_pipeline_config(camera_id)
        return True

    def upsert_line(self, camera_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Create or update a camera counting line and persist the config."""
        with self._lock:
            config = self.cameras.get(camera_id)
            if config is None:
                return None
            line_id = self._safe_id(str(payload.get("id") or f"line_{uuid.uuid4().hex[:8]}"))
            line = {
                "id": line_id,
                "name": str(payload.get("name") or line_id),
                "point1": payload.get("point1", [0.25, 0.5]),
                "point2": payload.get("point2", [0.75, 0.5]),
                "direction": str(payload.get("direction", "forward")),
            }
            lines = [item for item in config.get("lines", []) if str(item.get("id")) != line_id]
            lines.append(line)
            config["lines"] = lines
            self._save_camera_config(config)
            saved = copy.deepcopy(line)
        self._sync_pipeline_config(camera_id)
        return saved

    def delete_line(self, camera_id: str, line_id: str) -> bool:
        """Delete a counting line from a camera config."""
        with self._lock:
            config = self.cameras.get(camera_id)
            if config is None:
                return False
            original_count = len(config.get("lines", []))
            config["lines"] = [item for item in config.get("lines", []) if str(item.get("id")) != line_id]
            if len(config["lines"]) == original_count:
                return False
            self._save_camera_config(config)
        self._sync_pipeline_config(camera_id)
        return True

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge settings, persist them, and apply runtime-safe changes."""
        with self._lock:
            updated_settings = self._deep_merge(copy.deepcopy(self.settings), payload)
            updated_settings = _enforce_required_runtime_settings(updated_settings)
            self.detector.update_settings(updated_settings)
            self.settings = updated_settings
            self.settings_path.write_text(
                yaml.safe_dump(self.settings, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            self.alert_manager.update_settings(self.settings)
            self.behavior_engine = BehaviorEngine(self.settings)
            self.identity_resolver = self.behavior_engine.identity_resolver
            self.pose_estimator.update_settings(
                self.settings,
                device=self.detector.device,
                use_half=self.detector.use_half,
            )
            pipeline_settings = self.settings.get("pipeline", {})
            stream_max_height = int(pipeline_settings.get("stream_max_height", 720))
            for buffer in self.frame_buffers.values():
                buffer.max_height = stream_max_height
            for pipeline in self.pipelines.values():
                pipeline.settings = self.settings
                pipeline.pose_estimator = self.pose_estimator
                pipeline.behavior_engine = self._new_behavior_engine()
                pipeline.frame_skip = max(1, int(pipeline_settings.get("frame_skip", 2)))
                pipeline.ai_max_fps = max(0.0, float(pipeline_settings.get("ai_max_fps", 10)))
                pipeline.analysis_stale_after_ms = max(
                    0.0,
                    float(pipeline_settings.get("analysis_stale_after_ms", 500)),
                )
                pipeline.analysis_timeout = max(
                    1.0,
                    float(pipeline_settings.get("analysis_timeout_min_seconds", 5.0)),
                )
                pipeline.reconnect_delay = float(pipeline_settings.get("reconnect_delay", 5))
                pipeline.max_reconnect_attempts = int(pipeline_settings.get("max_reconnect_attempts", 10))
                pipeline.processing_max_height = int(
                    pipeline_settings.get(
                        "processing_max_height",
                        pipeline_settings.get("stream_max_height", 720),
                    )
                )
                pipeline.realtime_video_playback = bool(pipeline_settings.get("realtime_video_playback", True))
                pipeline.loop_video_files = bool(pipeline_settings.get("loop_video_files", True))
                pipeline.drop_late_video_frames = bool(pipeline_settings.get("drop_late_video_frames", True))
                pipeline.tracker.confidence = self.detector.confidence
                pipeline.tracker.class_ids = list(self.detector.class_ids)
                pipeline.tracker.iou = self.detector.iou
                pipeline.tracker.imgsz = self.detector.imgsz
                pipeline.tracker.grace_frames = int(
                    self.settings.get("tracking", {}).get(
                        "track_grace_frames",
                        pipeline.tracker.grace_frames,
                    )
                )
                pipeline.tracker.duplicate_iou_threshold = float(
                    self.settings.get("tracking", {}).get(
                        "duplicate_iou_threshold",
                        pipeline.tracker.duplicate_iou_threshold,
                    )
                )
                pipeline.tracker.duplicate_containment_threshold = float(
                    self.settings.get("tracking", {}).get(
                        "duplicate_containment_threshold",
                        pipeline.tracker.duplicate_containment_threshold,
                    )
                )
                pipeline.tracker.update_settings(self.settings)
        return copy.deepcopy(self.settings)

    def frame_buffer(self, camera_id: str) -> FrameBuffer | None:
        """Return the frame buffer for a camera."""
        with self._lock:
            return self.frame_buffers.get(camera_id)

    def get_alerts(self, camera_id: str, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Return recent alert history."""
        return self.alert_manager.get_recent(camera_id, limit=limit, offset=offset)

    def _public_camera(self, config: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(config.get("camera_id"))
        buffer = self.frame_buffers.get(camera_id)
        snapshot = buffer.snapshot() if buffer else None
        result = copy.deepcopy(config)
        if snapshot:
            result.update(
                {
                    "status": snapshot.status,
                    "object_count": snapshot.object_count,
                    "alert_count": snapshot.alert_count,
                    "updated_at": snapshot.updated_at,
                    "error": snapshot.error,
                    "fps": round(snapshot.fps, 1),
                    "staleness_ms": round(snapshot.staleness_ms, 0),
                    "ai_latency_ms": round(snapshot.ai_latency_ms, 0),
                    "analysis_stale_after_ms": float(
                        self.settings.get("pipeline", {}).get("analysis_stale_after_ms", 500)
                    ),
                }
            )
        else:
            result.update(
                {
                    "status": "offline",
                    "object_count": 0,
                    "alert_count": 0,
                    "error": None,
                    "fps": 0.0,
                    "staleness_ms": 0.0,
                    "ai_latency_ms": 0.0,
                    "analysis_stale_after_ms": float(
                        self.settings.get("pipeline", {}).get("analysis_stale_after_ms", 500)
                    ),
                }
            )
        result["line_counters"] = self._behavior_engine_for(camera_id).get_counters(camera_id)
        with self._lock:
            result["detection_active"] = camera_id in self.pipelines
        return result

    def _restart_pipeline(self, camera_config: dict[str, Any]) -> None:
        camera_id = str(camera_config.get("camera_id"))
        with self._lock:
            existing = self.pipelines.pop(camera_id, None)
            buffer = self.frame_buffers.setdefault(
                camera_id,
                FrameBuffer(max_height=int(self.settings.get("pipeline", {}).get("stream_max_height", 720))),
            )
        if existing:
            existing.stop()
        if not bool(camera_config.get("enabled", False)):
            buffer.set_status("offline", "Camera disabled")
            return
        pipeline = CameraPipeline(
            camera_config=copy.deepcopy(camera_config),
            settings=self.settings,
            detector=self.detector,
            pose_estimator=self.pose_estimator,
            behavior_engine=self._new_behavior_engine(),
            frame_buffer=buffer,
            alert_manager=self.alert_manager,
        )
        with self._lock:
            self.pipelines[camera_id] = pipeline
        pipeline.start()

    def _sync_pipeline_config(self, camera_id: str) -> None:
        with self._lock:
            config = copy.deepcopy(self.cameras.get(camera_id))
            pipeline = self.pipelines.get(camera_id)
        if config and pipeline:
            pipeline.update_config(config)

    def _save_camera_config(self, config: dict[str, Any]) -> None:
        self.cameras_dir.mkdir(parents=True, exist_ok=True)
        camera_id = self._safe_id(str(config["camera_id"]))
        path = self.cameras_dir / f"{camera_id}.yaml"
        path.write_text(
            yaml.safe_dump(self._camera_config_for_disk(config), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def _new_behavior_engine(self) -> BehaviorEngine:
        return BehaviorEngine(
            self.settings,
            identity_resolver=self.identity_resolver,
        )

    def _behavior_engine_for(self, camera_id: str) -> BehaviorEngine:
        with self._lock:
            pipeline = self.pipelines.get(camera_id)
        return pipeline.behavior_engine if pipeline else self.behavior_engine

    def _camera_config_for_disk(self, config: dict[str, Any]) -> dict[str, Any]:
        persisted = copy.deepcopy(config)
        zones = [self._zone_config_for_disk(zone) for zone in persisted.get("zones", [])]
        if zones:
            persisted["zones"] = zones
        else:
            persisted.pop("zones", None)
        if not persisted.get("lines"):
            persisted.pop("lines", None)
        return persisted

    def _zone_config_for_disk(self, zone: dict[str, Any]) -> dict[str, Any]:
        persisted = copy.deepcopy(zone)
        default_threshold = self._default_zone_threshold(str(persisted.get("type") or persisted.get("zone_type") or ""))
        raw_threshold = persisted.get("threshold_seconds")
        if default_threshold is not None and raw_threshold not in (None, ""):
            try:
                if abs(float(raw_threshold) - default_threshold) < 0.000001:
                    persisted.pop("threshold_seconds", None)
            except (TypeError, ValueError):
                pass
        return persisted

    def _default_zone_threshold(self, zone_type: str) -> float | None:
        behavior = self.settings.get("behavior", {})
        if zone_type == "loitering":
            return float(behavior.get("loitering_threshold_seconds", 30))
        if zone_type == "stranger_watch":
            return float(behavior.get("stranger_watch_seconds", 180))
        return None

    @staticmethod
    def _safe_id(value: str) -> str:
        return _safe_id(value)

    @classmethod
    def _new_camera_id(cls, name: Any) -> str:
        base = cls._safe_id(str(name).lower())[:32] or "camera"
        return f"{base}_{uuid.uuid4().hex[:6]}"

    @classmethod
    def _deep_merge(cls, base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = cls._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
