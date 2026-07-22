"""Pose label classifier combining keypoint geometry and motion speed."""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, replace
from math import atan2, degrees, hypot, pi
from typing import Any

from core.tracker import TrackedObject

logger = logging.getLogger(__name__)

_UPRIGHT_LABELS = {"standing_still", "walking_slow", "running"}


@dataclass
class _PoseTrackState:
    label_history: deque[str]
    # Separate, longer history for motion labels only (standing/walking/running).
    # Geometric labels (sitting, lying) bypass this and update instantly.
    motion_label_history: deque[str]
    # Per-frame speed ratio history for hysteresis decisions.
    speed_history: deque[float]
    # The last confirmed motion label (for hysteresis).
    last_motion_label: str
    confirmed_pose_family: str | None = None
    transition_target_family: str | None = None
    transition_started_at: float | None = None


class PoseClassifier:
    """Assign a stable pose label to each tracked person."""

    def __init__(self, settings: dict[str, Any]) -> None:
        cfg = settings.get("pose_classification", {})
        self.min_keypoint_confidence = float(cfg.get("min_keypoint_confidence", 0.25))
        self.smoothing_window = max(1, int(cfg.get("smoothing_window", 5)))
        self.lying_angle_min_degrees = float(cfg.get("lying_angle_min_degrees", 55))
        self.lying_angle_relaxed_degrees = float(cfg.get("lying_angle_relaxed_degrees", 30))
        self.lying_hip_ankle_max_ratio = float(cfg.get("lying_hip_ankle_max_ratio", 0.6))
        self.lying_bbox_aspect_ratio_max = float(
            cfg.get("lying_bbox_aspect_ratio_max", 0.8)
        )
        self.lying_straight_leg_angle_min = float(
            cfg.get(
                "lying_straight_leg_angle_min",
                cfg.get("sitting_straight_leg_angle_min", 140),
            )
        )
        self.lying_straight_leg_bbox_aspect_ratio_max = float(
            cfg.get("lying_straight_leg_bbox_aspect_ratio_max", 1.6)
        )
        self.sitting_knee_hip_ratio_max = float(cfg.get("sitting_knee_hip_ratio_max", 0.4))
        self.sitting_hip_ankle_max_ratio = float(cfg.get("sitting_hip_ankle_max_ratio", 1.0))
        self.sitting_bbox_aspect_ratio_max = float(
            cfg.get("sitting_bbox_aspect_ratio_max", 1.3)
        )
        self.speed_walk_min_ratio = float(cfg.get("speed_walk_min_ratio", 0.15))
        self.speed_run_min_ratio = float(cfg.get("speed_run_min_ratio", 0.45))
        self.speed_noise_floor = float(cfg.get("speed_noise_floor", 2.0))
        self.motion_window = max(2, int(cfg.get("motion_window", 8)))

        # ── Hysteresis settings ──────────────────────────────────────────
        # To transition UP (e.g. walking → running), the speed must exceed
        # the threshold by this margin.  To transition DOWN (running → walking),
        # the speed must drop below the threshold by this margin.
        # This prevents rapid flickering when speed hovers near a threshold.
        self.speed_hysteresis_margin = float(
            cfg.get("speed_hysteresis_margin", 0.08)
        )
        # Motion-specific smoothing window — longer than the general
        # smoothing_window so that motion labels change more slowly and
        # do not flicker every few frames.
        self.motion_smoothing_window = max(
            1, int(cfg.get("motion_smoothing_window", 10))
        )
        # How many consecutive frames of a new motion label are required
        # before the label actually switches.  Prevents a single fast frame
        # from flipping "walking" to "running".
        self.motion_switch_confirm_frames = max(
            1, int(cfg.get("motion_switch_confirm_frames", 4))
        )
        self.transition_confirm_seconds = max(
            0.0, float(cfg.get("transition_confirm_seconds", 0.75))
        )
        # Speed history length used for hysteresis averaging.
        self._speed_history_maxlen = max(
            self.motion_smoothing_window,
            self.motion_window,
        )

        self._states: dict[tuple[str, int], _PoseTrackState] = {}

    def label_objects(
        self,
        camera_id: str,
        tracked_objects: list[TrackedObject],
        frame_shape: tuple[int, int, int],
    ) -> list[TrackedObject]:
        """Return objects with pose_label filled in for person tracks."""
        active_keys = {
            (camera_id, obj.track_id)
            for obj in tracked_objects
            if obj.class_name == "person"
        }
        for key in [key for key in self._states if key[0] == camera_id and key not in active_keys]:
            self._states.pop(key, None)

        result: list[TrackedObject] = []
        now = time.monotonic()
        for obj in tracked_objects:
            if obj.class_name != "person":
                result.append(obj)
                continue
            key = (camera_id, obj.track_id)
            state = self._states.setdefault(
                key,
                _PoseTrackState(
                    label_history=deque(maxlen=self.smoothing_window),
                    motion_label_history=deque(maxlen=self.motion_smoothing_window),
                    speed_history=deque(maxlen=self._speed_history_maxlen),
                    last_motion_label="standing_still",
                ),
            )
            raw_label = self._classify(obj, frame_shape, state)
            # Geometric labels (sitting, lying, unknown) use Counter vote
            # smoothing to absorb single-frame misclassifications.
            # Motion labels (standing_still, walking_slow, running) are
            # already stabilised by _stable_motion_label's hysteresis +
            # confirmation gate, so adding another Counter vote on top
            # would cause lag and fighting.  Use the hysteresis output
            # directly.
            if raw_label in ("sitting", "lying", "unknown"):
                state.label_history.append(raw_label)
                stable_label = Counter(state.label_history).most_common(1)[0][0]
            else:
                # Motion label — push it into history (so that a future
                # geometric label can still win the vote if the person
                # sits down), but use the hysteresis result directly.
                state.label_history.append(raw_label)
                stable_label = raw_label
            stable_label = self._transition_label(stable_label, state, now)
            result.append(replace(obj, pose_label=stable_label))
        return result

    def _transition_label(
        self,
        candidate_label: str,
        state: _PoseTrackState,
        now: float,
    ) -> str:
        """Insert a short, derived state between confirmed posture families."""
        target_family = self._pose_family(candidate_label)
        if target_family is None:
            state.transition_target_family = None
            state.transition_started_at = None
            return candidate_label

        current_family = state.confirmed_pose_family
        if current_family is None:
            state.confirmed_pose_family = target_family
            return candidate_label

        if target_family == current_family:
            state.transition_target_family = None
            state.transition_started_at = None
            return candidate_label

        if state.transition_target_family != target_family:
            state.transition_target_family = target_family
            state.transition_started_at = now

        started_at = state.transition_started_at
        if started_at is not None and now - started_at >= self.transition_confirm_seconds:
            state.confirmed_pose_family = target_family
            state.transition_target_family = None
            state.transition_started_at = None
            return candidate_label

        if current_family == "upright" and target_family == "sitting":
            return "sitting_down"
        if current_family in {"sitting", "lying"} and target_family == "upright":
            return "getting_up"
        if target_family == "lying":
            return "changing_to_lying"
        return "changing_posture"

    @staticmethod
    def _pose_family(label: str) -> str | None:
        if label in _UPRIGHT_LABELS:
            return "upright"
        if label in {"sitting", "lying"}:
            return label
        return None

    def _classify(
        self,
        obj: TrackedObject,
        frame_shape: tuple[int, int, int],
        state: _PoseTrackState,
    ) -> str:
        geo = self._geometric_label(obj)
        if geo in ("lying", "sitting", "unknown"):
            return geo
        # geo == "upright" — use motion speed to distinguish standing / walking / running
        speed = self._speed_ratio(obj)
        state.speed_history.append(speed)
        raw_motion = self._raw_motion_label(speed)
        return self._stable_motion_label(raw_motion, state)

    def _raw_motion_label(self, speed: float) -> str:
        """Classify a single speed value into a motion label (no smoothing)."""
        if speed < self.speed_walk_min_ratio:
            return "standing_still"
        if speed < self.speed_run_min_ratio:
            return "walking_slow"
        return "running"

    def _stable_motion_label(
        self,
        raw_label: str,
        state: _PoseTrackState,
    ) -> str:
        """Apply hysteresis + multi-frame confirmation to prevent flicker.

        The idea:
        1. Use the *average* speed over recent frames instead of a single
           frame's value so that one noisy spike doesn't flip the label.
        2. Apply *hysteresis*: to go from walking→running the avg speed must
           exceed run_threshold + margin, and to go from running→walking it
           must drop below run_threshold − margin.  This creates a "dead zone"
           that absorbs normal speed fluctuations.
        3. Require the new label to appear in N consecutive raw classifications
           before actually switching (confirmation gate).
        """
        state.motion_label_history.append(raw_label)
        previous_label = state.last_motion_label

        # ── Average speed for hysteresis ─────────────────────────────────
        # Use only the last few speeds (same as confirmation window) so the
        # average responds quickly to real speed changes.  The confirmation
        # gate still prevents single-frame flicker.
        recent_n = min(len(state.speed_history), self.motion_switch_confirm_frames + 2)
        if recent_n > 0:
            recent_speeds = list(state.speed_history)[-recent_n:]
            avg_speed = sum(recent_speeds) / len(recent_speeds)
        else:
            avg_speed = 0.0

        # ── Hysteresis thresholds ────────────────────────────────────────
        margin = self.speed_hysteresis_margin

        # Determine what the hysteresis-adjusted label should be.
        if previous_label == "running":
            # Must drop clearly below run threshold to downgrade.
            if avg_speed < self.speed_run_min_ratio - margin:
                if avg_speed < self.speed_walk_min_ratio - margin:
                    hysteresis_label = "standing_still"
                else:
                    hysteresis_label = "walking_slow"
            else:
                hysteresis_label = "running"
        elif previous_label == "walking_slow":
            # Must clearly exceed run threshold to upgrade.
            if avg_speed >= self.speed_run_min_ratio + margin:
                hysteresis_label = "running"
            elif avg_speed < self.speed_walk_min_ratio - margin:
                hysteresis_label = "standing_still"
            else:
                hysteresis_label = "walking_slow"
        else:  # standing_still
            if avg_speed >= self.speed_run_min_ratio + margin:
                hysteresis_label = "running"
            elif avg_speed >= self.speed_walk_min_ratio + margin:
                hysteresis_label = "walking_slow"
            else:
                hysteresis_label = "standing_still"

        # ── Confirmation gate ────────────────────────────────────────────
        # Only switch if the hysteresis label has been consistent for enough
        # recent frames.
        if hysteresis_label == previous_label:
            return previous_label

        # Count how many of the most recent raw labels match the proposed
        # new hysteresis label.
        recent = list(state.motion_label_history)
        tail = recent[-self.motion_switch_confirm_frames:]
        if len(tail) >= self.motion_switch_confirm_frames and all(
            label == hysteresis_label for label in tail
        ):
            state.last_motion_label = hysteresis_label
            logger.debug(
                "motion label switch: %s → %s (avg_speed=%.3f)",
                previous_label,
                hysteresis_label,
                avg_speed,
            )
            return hysteresis_label

        # Not enough consecutive confirmation frames — keep previous label.
        return previous_label

    # ── Geometric analysis ───────────────────────────────────────────────

    def _geometric_label(self, obj: TrackedObject) -> str:
        keypoints = obj.pose_keypoints
        if not keypoints:
            return "unknown"

        shoulder = self._avg_point(keypoints, 5, 6)
        hip = self._avg_point(keypoints, 11, 12)
        knee = self._avg_point(keypoints, 13, 14)
        ankle = self._avg_point(keypoints, 15, 16)
        if shoulder is None or hip is None:
            return "unknown"

        torso_len = hypot(hip[0] - shoulder[0], hip[1] - shoulder[1])
        torso_angle = self._body_axis_angle_from_vertical(shoulder, hip)

        hip_ankle_ratio = None
        if ankle is not None and torso_len > 0:
            hip_ankle_ratio = abs(ankle[1] - hip[1]) / torso_len

        thigh_ratio = None
        if knee is not None and torso_len > 0:
            thigh_ratio = hypot(knee[0] - hip[0], knee[1] - hip[1]) / torso_len

        # --- Bbox aspect ratio: height / width ---
        x1, y1, x2, y2 = obj.bbox_xyxy
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        bbox_aspect = bbox_h / bbox_w

        # --- Knee angle (hip-knee-ankle) ---
        knee_angle = None
        if knee is not None and hip is not None and ankle is not None:
            knee_angle = self._angle_at_vertex(hip, knee, ankle)

        logger.debug(
            "pose_debug track=%s torso_angle=%.1f hip_ankle_ratio=%s "
            "thigh_ratio=%s bbox_aspect=%.2f knee_angle=%s",
            obj.track_id,
            torso_angle,
            f"{hip_ankle_ratio:.2f}" if hip_ankle_ratio is not None else "NA",
            f"{thigh_ratio:.2f}" if thigh_ratio is not None else "NA",
            bbox_aspect,
            f"{knee_angle:.1f}" if knee_angle is not None else "NA",
        )

        # ── Lying detection ──────────────────────────────────────────────
        if torso_angle >= self.lying_angle_min_degrees:
            return "lying"
        if (
            hip_ankle_ratio is not None
            and hip_ankle_ratio <= self.lying_hip_ankle_max_ratio
            and torso_angle >= self.lying_angle_relaxed_degrees
        ):
            return "lying"
        if bbox_aspect <= self.lying_bbox_aspect_ratio_max:
            return "lying"
        if (
            bbox_aspect <= self.lying_straight_leg_bbox_aspect_ratio_max
            and knee_angle is not None
            and knee_angle >= self.lying_straight_leg_angle_min
        ):
            return "lying"

        # ── Sitting detection ────────────────────────────────────────────
        if thigh_ratio is not None and thigh_ratio <= self.sitting_knee_hip_ratio_max:
            return "sitting"

        if (
            hip_ankle_ratio is not None
            and hip_ankle_ratio <= self.sitting_hip_ankle_max_ratio
            and torso_angle < self.lying_angle_relaxed_degrees
        ):
            return "sitting"

        if bbox_aspect <= self.sitting_bbox_aspect_ratio_max:
            if torso_angle < self.lying_angle_relaxed_degrees:
                return "sitting"

        return "upright"

    # ── Keypoint helpers ─────────────────────────────────────────────────

    def _avg_point(
        self,
        keypoints: list[tuple[float, float, float]],
        left_index: int,
        right_index: int,
    ) -> tuple[float, float] | None:
        points = [
            (keypoints[index][0], keypoints[index][1])
            for index in (left_index, right_index)
            if index < len(keypoints)
            and keypoints[index][2] >= self.min_keypoint_confidence
        ]
        if not points:
            return None
        return (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )

    @staticmethod
    def _body_axis_angle_from_vertical(
        shoulder: tuple[float, float],
        hip: tuple[float, float],
    ) -> float:
        """0 deg = upright torso, 90 deg = horizontal torso."""
        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]
        if dx == 0 and dy == 0:
            return 0.0
        return degrees(atan2(abs(dx), abs(dy)))

    @staticmethod
    def _angle_at_vertex(
        point_a: tuple[float, float],
        vertex: tuple[float, float],
        point_b: tuple[float, float],
    ) -> float:
        """Return the angle (degrees) at *vertex* formed by segments vertex→A and vertex→B."""
        ax, ay = point_a[0] - vertex[0], point_a[1] - vertex[1]
        bx, by = point_b[0] - vertex[0], point_b[1] - vertex[1]
        dot = ax * bx + ay * by
        mag_a = hypot(ax, ay)
        mag_b = hypot(bx, by)
        if mag_a < 1e-6 or mag_b < 1e-6:
            return 180.0
        cos_val = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
        return degrees(atan2((1.0 - cos_val ** 2) ** 0.5, cos_val))

    # ── Speed measurement ────────────────────────────────────────────────

    def _speed_ratio(self, obj: TrackedObject) -> float:
        vector = _motion_vector(
            obj.center_history,
            window=self.motion_window,
            noise_floor=self.speed_noise_floor,
        )
        distance = hypot(*vector)
        x1, y1, x2, y2 = obj.bbox_xyxy
        bbox_diagonal = max(hypot(x2 - x1, y2 - y1), 1.0)
        return distance / bbox_diagonal


# ── Module-level helpers ─────────────────────────────────────────────────

def _motion_vector(
    history: list[tuple[float, float]],
    window: int = 8,
    noise_floor: float = 2.0,
) -> tuple[float, float]:
    if len(history) < 2:
        return 0.0, 0.0
    points = history[-window:]
    if len(points) < 4:
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        if hypot(dx, dy) < noise_floor * len(points):
            return 0.0, 0.0
        return dx, dy

    steps_x = [points[i + 1][0] - points[i][0] for i in range(len(points) - 1)]
    steps_y = [points[i + 1][1] - points[i][1] for i in range(len(points) - 1)]

    # Clamp per-step displacements below noise_floor to zero.
    steps_mag = [hypot(sx, sy) for sx, sy in zip(steps_x, steps_y)]
    for i, mag in enumerate(steps_mag):
        if mag < noise_floor:
            steps_x[i] = 0.0
            steps_y[i] = 0.0

    n_steps = len(steps_x)
    median_dx = _median(steps_x)
    median_dy = _median(steps_y)
    return median_dx * n_steps, median_dy * n_steps


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
