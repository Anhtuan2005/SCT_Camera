"""Fall detection based on a fast upright-to-lying transition."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.tracker import TrackedObject

_ARMING_LABELS = {"standing_still", "walking_slow", "running"}
_RECOVERY_LABELS = _ARMING_LABELS | {"sitting"}
_LYING_LABELS = {"lying", "changing_to_lying"}


@dataclass
class _FallCandidate:
    pre_lying_label: str
    upright_since: float
    last_upright_at: float
    upright_center_y: float
    upright_height: float
    became_lying_at: float | None = None
    last_lying_at: float | None = None
    vertical_drop_ratio: float = 0.0
    initial_alerted_at: float | None = None
    escalated_at: float | None = None
    next_reminder_at: float | None = None
    recovery_started_at: float | None = None


class FallDetector:
    """Emit staged safety alerts for a tracked person who remains down."""

    def __init__(self, settings: dict[str, Any]) -> None:
        cfg = settings.get("fall_detection", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.min_upright_seconds = max(0.0, float(cfg.get("min_upright_seconds", 1)))
        self.max_transition_seconds = max(
            0.0, float(cfg.get("max_transition_seconds", 3))
        )
        self.min_vertical_drop_ratio = max(
            0.0, float(cfg.get("min_vertical_drop_ratio", 0.2))
        )
        self.lying_alert_seconds = max(
            0.0, float(cfg.get("lying_alert_seconds", 10))
        )
        self.pose_grace_seconds = max(
            0.0, float(cfg.get("pose_grace_seconds", 1))
        )
        self.escalation_seconds = max(
            self.lying_alert_seconds,
            float(cfg.get("escalation_seconds", 45)),
        )
        self.urgent_reminder_seconds = max(
            0.0, float(cfg.get("urgent_reminder_seconds", 60))
        )
        self.recovery_confirm_seconds = max(
            0.0, float(cfg.get("recovery_confirm_seconds", 3))
        )
        self.motion_window = max(2, int(cfg.get("motion_window", 8)))
        self.max_down_motion_ratio = max(
            0.0, float(cfg.get("max_down_motion_ratio", 0.08))
        )
        self.emergency_number = str(cfg.get("emergency_number", "115")).strip()
        self._states: dict[tuple[str, int], _FallCandidate] = {}

    def analyze(
        self,
        camera_id: str,
        camera_name: str,
        tracked_objects: list[TrackedObject],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        active_keys = {
            (camera_id, obj.track_id)
            for obj in tracked_objects
            if obj.class_name == "person"
        }
        for key in [key for key in self._states if key[0] == camera_id and key not in active_keys]:
            self._states.pop(key, None)

        now = time.monotonic()
        alerts: list[dict[str, Any]] = []
        for obj in tracked_objects:
            if obj.class_name != "person" or obj.pose_label is None:
                continue
            key = (camera_id, obj.track_id)

            if obj.pose_label in _RECOVERY_LABELS:
                alerts.extend(
                    self._handle_recovery_pose(
                        key, camera_id, camera_name, obj, timestamp, now
                    )
                )
                continue

            if obj.pose_label in _LYING_LABELS:
                alert = self._handle_lying_pose(
                    key, camera_id, camera_name, obj, timestamp, now
                )
                if alert is not None:
                    alerts.append(alert)
                continue

            state = self._states.get(key)
            if (
                state is not None
                and state.last_lying_at is not None
                and now - state.last_lying_at > self.pose_grace_seconds
            ):
                self._states.pop(key, None)

        return alerts

    def _handle_recovery_pose(
        self,
        key: tuple[str, int],
        camera_id: str,
        camera_name: str,
        obj: TrackedObject,
        timestamp: datetime,
        now: float,
    ) -> list[dict[str, Any]]:
        state = self._states.get(key)
        if state is not None and state.became_lying_at is not None:
            confirmed_recovery_pose = (
                obj.pose_label == "sitting" or self._looks_upright_bbox(obj)
            )
            if not confirmed_recovery_pose:
                state.recovery_started_at = None
                return []
            if state.recovery_started_at is None:
                state.recovery_started_at = now
                return []
            if now - state.recovery_started_at < self.recovery_confirm_seconds:
                return []

            if state.initial_alerted_at is not None:
                alert = self._recovery_alert(
                    camera_id, camera_name, obj, timestamp, state, now
                )
                self._states.pop(key, None)
                if obj.pose_label in _ARMING_LABELS:
                    self._states[key] = self._new_upright_state(obj, now)
                return [alert]

            self._states.pop(key, None)

        if obj.pose_label == "sitting":
            if (
                state is not None
                and state.became_lying_at is None
                and now - state.last_upright_at <= self.max_transition_seconds
            ):
                return []
            self._states.pop(key, None)
            return []

        # Pose classification is smoothed. During the first fall frames the
        # box can already be horizontal while the stable label is still upright.
        if not self._looks_upright_bbox(obj):
            return []

        state = self._states.get(key)
        if state is None:
            self._states[key] = self._new_upright_state(obj, now)
        else:
            state.pre_lying_label = str(obj.pose_label)
            state.last_upright_at = now
            current_height = self._bbox_height(obj)
            # Motion labels can linger while the person is already collapsing.
            # Keep the taller confirmed-upright box as the drop reference.
            if current_height >= state.upright_height:
                state.upright_center_y = obj.center[1]
                state.upright_height = current_height
        return []

    def _handle_lying_pose(
        self,
        key: tuple[str, int],
        camera_id: str,
        camera_name: str,
        obj: TrackedObject,
        timestamp: datetime,
        now: float,
    ) -> dict[str, Any] | None:
        state = self._states.get(key)
        if state is None:
            return None

        if state.became_lying_at is None:
            upright_seconds = now - state.upright_since
            transition_seconds = now - state.last_upright_at
            vertical_drop_ratio = max(
                0.0,
                (obj.center[1] - state.upright_center_y) / state.upright_height,
            )
            if (
                upright_seconds < self.min_upright_seconds
                or transition_seconds > self.max_transition_seconds
                or vertical_drop_ratio < self.min_vertical_drop_ratio
            ):
                self._states.pop(key, None)
                return None

            state.became_lying_at = now
            state.last_lying_at = now
            state.vertical_drop_ratio = vertical_drop_ratio
            return None

        if (
            state.last_lying_at is not None
            and now - state.last_lying_at > self.pose_grace_seconds
            and (
                state.recovery_started_at is None
                or now - state.recovery_started_at >= self.recovery_confirm_seconds
            )
        ):
            self._states.pop(key, None)
            return None

        state.last_lying_at = now
        state.recovery_started_at = None
        down_seconds = now - state.became_lying_at
        motion_ratio = self._recent_motion_ratio(obj)

        if state.initial_alerted_at is None:
            if down_seconds < self.lying_alert_seconds:
                return None
            state.initial_alerted_at = now
            return self._fall_alert(
                camera_id,
                camera_name,
                obj,
                timestamp,
                state,
                down_seconds,
                motion_ratio,
            )

        if state.escalated_at is None:
            if (
                down_seconds < self.escalation_seconds
                or motion_ratio > self.max_down_motion_ratio
            ):
                return None
            state.escalated_at = now
            state.next_reminder_at = (
                now + self.urgent_reminder_seconds
                if self.urgent_reminder_seconds > 0
                else None
            )
            return self._urgent_alert(
                camera_id,
                camera_name,
                obj,
                timestamp,
                state,
                down_seconds,
                motion_ratio,
                reminder=False,
            )

        if state.next_reminder_at is not None and now >= state.next_reminder_at:
            state.next_reminder_at = now + self.urgent_reminder_seconds
            return self._urgent_alert(
                camera_id,
                camera_name,
                obj,
                timestamp,
                state,
                down_seconds,
                motion_ratio,
                reminder=True,
            )
        return None

    @staticmethod
    def _new_upright_state(obj: TrackedObject, now: float) -> _FallCandidate:
        return _FallCandidate(
            pre_lying_label=str(obj.pose_label),
            upright_since=now,
            last_upright_at=now,
            upright_center_y=obj.center[1],
            upright_height=FallDetector._bbox_height(obj),
        )

    def _fall_alert(
        self,
        camera_id: str,
        camera_name: str,
        obj: TrackedObject,
        timestamp: datetime,
        state: _FallCandidate,
        down_seconds: float,
        motion_ratio: float,
    ) -> dict[str, Any]:
        return self._base_alert(
            "possible_fall",
            "critical",
            "CÓ THỂ CÓ NGƯỜI BỊ NGÃ",
            "initial",
            camera_id,
            camera_name,
            obj,
            timestamp,
            state,
            down_seconds,
            motion_ratio,
            (
                f"Người này chuyển nhanh từ {state.pre_lying_label} sang nằm và "
                f"chưa đứng dậy sau {down_seconds:.0f} giây."
            ),
            self._emergency_action(),
            siren=True,
            threshold_seconds=self.lying_alert_seconds,
        )

    def _urgent_alert(
        self,
        camera_id: str,
        camera_name: str,
        obj: TrackedObject,
        timestamp: datetime,
        state: _FallCandidate,
        down_seconds: float,
        motion_ratio: float,
        reminder: bool,
    ) -> dict[str, Any]:
        title = (
            "NHẮC LẠI: NGƯỜI NGÃ VẪN NẰM BẤT ĐỘNG"
            if reminder
            else "KHẨN CẤP: NGƯỜI NGÃ VẪN NẰM BẤT ĐỘNG"
        )
        return self._base_alert(
            "possible_unresponsive",
            "emergency",
            title,
            "reminder" if reminder else "urgent",
            camera_id,
            camera_name,
            obj,
            timestamp,
            state,
            down_seconds,
            motion_ratio,
            (
                "Camera phát hiện một người ngã và nằm gần như bất động trong "
                f"{down_seconds:.0f} giây. Đây có thể là tình trạng y tế khẩn cấp."
            ),
            self._emergency_action(),
            siren=True,
            threshold_seconds=self.escalation_seconds,
        )

    def _recovery_alert(
        self,
        camera_id: str,
        camera_name: str,
        obj: TrackedObject,
        timestamp: datetime,
        state: _FallCandidate,
        now: float,
    ) -> dict[str, Any]:
        assert state.became_lying_at is not None
        return self._base_alert(
            "fall_recovery",
            "info",
            "ĐÃ PHÁT HIỆN THAY ĐỔI TƯ THẾ",
            "recovery",
            camera_id,
            camera_name,
            obj,
            timestamp,
            state,
            now - state.became_lying_at,
            self._recent_motion_ratio(obj),
            f"Người được cảnh báo đã chuyển sang tư thế {obj.pose_label} ổn định.",
            "Vẫn nên liên hệ hoặc đến kiểm tra để chắc chắn người đó an toàn.",
            siren=False,
            threshold_seconds=self.recovery_confirm_seconds,
        )

    def _base_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        alert_stage: str,
        camera_id: str,
        camera_name: str,
        obj: TrackedObject,
        timestamp: datetime,
        state: _FallCandidate,
        down_seconds: float,
        motion_ratio: float,
        details: str,
        recommended_action: str,
        *,
        siren: bool,
        threshold_seconds: float,
    ) -> dict[str, Any]:
        assert state.became_lying_at is not None
        return {
            "type": alert_type,
            "severity": severity,
            "title": title,
            "alert_stage": alert_stage,
            "safety_critical": True,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "track_id": obj.track_id,
            "class_name": obj.class_name,
            "identity_label": obj.identity_label or "Unknown",
            "pre_lying_label": state.pre_lying_label,
            "transition_seconds": round(
                state.became_lying_at - state.last_upright_at, 2
            ),
            "vertical_drop_ratio": round(state.vertical_drop_ratio, 3),
            "motion_ratio": round(motion_ratio, 3),
            "duration": round(down_seconds, 2),
            "threshold_seconds": threshold_seconds,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "emergency_number": self.emergency_number,
            "recommended_action": recommended_action,
            "siren": siren,
            "details": details,
        }

    def _emergency_action(self) -> str:
        emergency = (
            f"gọi cấp cứu {self.emergency_number} ngay"
            if self.emergency_number
            else "gọi số cấp cứu địa phương ngay"
        )
        return (
            "Mở camera và đến kiểm tra ngay. Nếu người đó không phản hồi, khó thở "
            f"hoặc có dấu hiệu đột quỵ, {emergency}."
        )

    def _recent_motion_ratio(self, obj: TrackedObject) -> float:
        points = obj.center_history[-self.motion_window :]
        if len(points) < 2:
            return 0.0
        origin_x, origin_y = points[0]
        displacement = max(
            math.hypot(x - origin_x, y - origin_y) for x, y in points[1:]
        )
        x1, y1, x2, y2 = obj.bbox_xyxy
        diagonal = max(math.hypot(x2 - x1, y2 - y1), 1.0)
        return displacement / diagonal

    @staticmethod
    def _bbox_height(obj: TrackedObject) -> float:
        return max(1.0, obj.bbox_xyxy[3] - obj.bbox_xyxy[1])

    @staticmethod
    def _looks_upright_bbox(obj: TrackedObject) -> bool:
        width = max(1.0, obj.bbox_xyxy[2] - obj.bbox_xyxy[0])
        return FallDetector._bbox_height(obj) >= width
