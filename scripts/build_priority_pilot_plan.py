"""Build the frozen single-participant 20-clip pilot and camera YAML files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_VERSION = "priority-pilot-v2-single-participant-2026-08-18"
PARTICIPANT = "P01"
BEHAVIORS = (
    "intrusion",
    "loitering",
    "suspicious_behavior",
    "theft",
    "line_crossing",
)
CONDITIONS = (
    ("daylight", "frontal_oblique"),
    ("daylight", "side_oblique"),
    ("low_light", "frontal_oblique"),
    ("low_light", "side_oblique"),
)
# Ten conditions per repetition: five daylight/low-light and five frontal/side.
# Each case flips lighting/viewpoint; the four joint cells total 6/4/4/6.
REPETITION_CONDITIONS = {
    1: (0, 3, 1, 2, 0, 3, 1, 2, 0, 3),
    2: (3, 0, 2, 1, 3, 0, 2, 1, 3, 0),
}
SCENARIOS = {
    "intrusion": {
        "positive": (25, "Cross the confirmed line once from image-left to image-right (configured IN) near 10 s."),
        "negative": (25, "Cross once from image-right to image-left (OUT); never cross back IN."),
        "positive_direction": "IN",
        "negative_direction": "OUT",
        "threshold": "No dwell threshold; confirmed person, >=2 center-history points, IN only.",
    },
    "loitering": {
        "positive": (35, "Enter the confirmed ROI by 5 s and remain continuously inside for >=22 s."),
        "negative": (25, "Enter the ROI, remain <15 s, then leave and stay outside."),
        "positive_direction": "",
        "negative_direction": "",
        "threshold": "20 s dwell; 3 s state grace.",
    },
    "suspicious_behavior": {
        "positive": (200, "Enter by 5 s and stand nearly still inside stranger_watch until >=190 s; keep full body visible."),
        "negative": (200, "Walk purposefully through the ROI once within 15 s, exit, and stay outside; do not pace or wait."),
        "positive_direction": "",
        "negative_direction": "",
        "threshold": "180 s dwell; >=5 history points; stationary displacement <=0.04 diagonal or pacing path/net rule.",
    },
    "theft": {
        "positive": (35, "Stay near the watched vehicle for >=10 s, then visibly move/push it while remaining in the ROI."),
        "negative": (35, "Stay near the stationary vehicle for >=12 s with hands away; do not touch, pace around, or move it."),
        "positive_direction": "",
        "negative_direction": "",
        "threshold": "Near >=10 s; score >=2; near cue mandatory plus >=1 vehicle/pose cue.",
    },
    "line_crossing": {
        "positive": (25, "Cross the confirmed line once from image-right to image-left (OUT) near 10 s."),
        "negative": (25, "Move parallel to the line or turn early; keep the entire person bbox clear of the line."),
        "positive_direction": "OUT",
        "negative_direction": "NONE",
        "threshold": "No dwell threshold; counter increments on center crossing or new line/bbox contact; IN and OUT counted.",
    },
}
RUNTIME_FILES = (
    "analytics/asset_watch.py",
    "analytics/behavior_engine.py",
    "analytics/behavior_learning.py",
    "analytics/dwell_policy.py",
    "analytics/intrusion.py",
    "analytics/loitering.py",
    "analytics/suspicious_stranger.py",
    "analytics/theft_behavior.py",
    "analytics/zone.py",
    "core/detector.py",
    "core/pose.py",
    "core/tracker.py",
    "config/settings.yaml",
    "scripts/run_behavior_experiment.py",
    "scripts/evaluate_behavior_experiment.py",
    "scripts/derive_risk_labels_from_behavior.py",
    "yolo11s.pt",
    "yolo11n-pose.pt",
)


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repetition in (1, 2):
        split = "development" if repetition == 1 else "test"
        condition_indexes = REPETITION_CONDITIONS[repetition]
        for behavior_index, behavior in enumerate(BEHAVIORS):
            scenario = SCENARIOS[behavior]
            for label_index, label in enumerate(("positive", "negative")):
                position = (behavior_index * 2) + label_index
                lighting, viewpoint = CONDITIONS[condition_indexes[position]]
                duration, action = scenario[label]
                session_id = f"{split}_r{repetition}_{lighting}_{viewpoint}"
                clip_id = f"{PARTICIPANT}_{behavior}_{label}_r{repetition}_{lighting}_{viewpoint}"
                rows.append({
                    "experiment_version": EXPERIMENT_VERSION,
                    "clip_id": clip_id,
                    "filename": f"{clip_id}.mp4",
                    "behavior": behavior,
                    "label": label,
                    "participant_id": PARTICIPANT,
                    "repetition": repetition,
                    "session_id": session_id,
                    "split": split,
                    "group_id": session_id,
                    "lighting": lighting,
                    "viewpoint": viewpoint,
                    "expected_direction": scenario[f"{label}_direction"],
                    "target_duration_seconds": duration,
                    "thresholds_frozen": scenario["threshold"],
                    "action_script": action,
                    "camera_setup": _camera_setup(viewpoint, lighting),
                    "video_path": f"experiments/priority_19_08/manual/videos_pilot/{split}/{session_id}/{clip_id}.mp4",
                    "camera_config_path": f"experiments/priority_19_08/manual/configs_pilot/{clip_id}.yaml",
                    "roi_line_confirmation": "NEEDS_MANUAL_CONFIRMATION",
                    "status": "TO_RECORD",
                })
    return rows


def _camera_setup(viewpoint: str, lighting: str) -> str:
    angle = (
        "actor path 20-35 degrees off the camera optical axis"
        if viewpoint == "frontal_oblique"
        else "actor path 45-70 degrees across the camera optical axis"
    )
    light = "stable daylight, no backlight" if lighting == "daylight" else "stable low light, face/body still detectable"
    return f"Fixed 2.5-3 m high, 20-35 degree downward tilt; {angle}; {light}; full body and ROI/line visible; >=1280x720, 25-30 FPS"


def camera_config(row: dict[str, Any]) -> dict[str, Any]:
    behavior = row["behavior"]
    config: dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "clip_id": row["clip_id"],
        "participant_id": row["participant_id"],
        "repetition": row["repetition"],
        "session_id": row["session_id"],
        "expected_behavior": behavior,
        "expected_label": row["label"],
        "expected_direction": row["expected_direction"],
        "camera_id": f"pilot_{row['clip_id']}",
        "name": f"Pilot {row['clip_id']}",
        "source": row["video_path"],
        "frame_rotation": "none",
        "auto_global_zone": False,
        "unknown_person_policy": "assume_stranger",
        "manual_confirmation": {
            "status": "NEEDS_MANUAL_CONFIRMATION",
            "allowed_edits": "polygon/point1/point2/direction/frame_rotation only",
            "do_not_change": "experiment_version, behavior thresholds, expected behavior/label",
        },
        "zones": [],
        "lines": [],
    }
    if behavior in {"intrusion", "line_crossing"}:
        config["lines"] = [{
            "id": "pilot_boundary",
            "name": "Pilot Boundary",
            "point1": [0.5, 0.85],
            "point2": [0.5, 0.15],
            "direction": "forward",
        }]
        config["manual_confirmation"]["line_note"] = (
            "Placeholder makes image-left to image-right IN. Confirm line placement and direction after a test frame."
        )
    zone_type = {
        "loitering": "loitering",
        "suspicious_behavior": "stranger_watch",
        "theft": "asset_watch",
    }.get(behavior)
    if zone_type:
        zone: dict[str, Any] = {
            "id": f"pilot_{zone_type}",
            "name": f"Pilot {zone_type}",
            "type": zone_type,
            "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        }
        if behavior == "loitering":
            zone["threshold_seconds"] = 20
        elif behavior == "suspicious_behavior":
            zone["threshold_seconds"] = 180
        config["zones"] = [zone]
        config["manual_confirmation"]["roi_note"] = (
            "Replace the placeholder polygon with the visible physical test area; keep its type and threshold unchanged."
        )
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _thresholds() -> dict[str, Any]:
    settings = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    behavior = settings["behavior"]
    theft = behavior["theft"]
    values = {
        "intrusion": {
            "allowed_classes": behavior.get("intrusion_classes", ["person"]),
            "reset_frames_unused_by_current_line_semantics": behavior["intrusion_reset_frames"],
        },
        "loitering": {
            "threshold_seconds": behavior["loitering_threshold_seconds"],
            "state_grace_seconds": behavior["loitering_state_grace_seconds"],
        },
        "suspicious_behavior": {
            "threshold_seconds": behavior["stranger_watch_seconds"],
            "state_grace_seconds": behavior["stranger_watch_state_grace_seconds"],
            **behavior["suspicious"],
        },
        "theft": dict(theft),
        "line_crossing": {
            "directions_counted": ["IN", "OUT"],
            "crossing_rule": "center path crossing OR newly intersecting person bbox",
        },
    }
    expected = (values["loitering"]["threshold_seconds"], values["suspicious_behavior"]["threshold_seconds"], values["theft"]["proximity_seconds"], values["theft"]["score_threshold"])
    if expected != (20, 180, 10, 2):
        raise ValueError(f"Reported behavior thresholds changed; review pilot plan before regenerating: {expected}")
    return values


def write_plan(out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = out_dir / "configs_pilot"
    configs_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    _write_csv(out_dir / "recording_plan_pilot.csv", rows, list(rows[0]))
    gt_fields = [
        "experiment_version", "clip_id", "behavior", "event_id", "start_seconds",
        "eligible_time_seconds", "end_seconds", "expected_direction", "label", "split",
        "group_id", "participant_id", "repetition", "session_id", "annotator", "reviewer", "config_confirmed",
        "annotation_status", "notes",
    ]
    gt_rows = [{
        "experiment_version": EXPERIMENT_VERSION,
        "clip_id": row["clip_id"],
        "behavior": row["behavior"],
        "event_id": f"gt_{row['clip_id']}_01" if row["label"] == "positive" else "",
        "start_seconds": "",
        "eligible_time_seconds": "",
        "end_seconds": "",
        "expected_direction": row["expected_direction"] if row["label"] == "positive" else "",
        "label": row["label"],
        "split": row["split"],
        "group_id": row["group_id"],
        "participant_id": row["participant_id"],
        "repetition": row["repetition"],
        "session_id": row["session_id"],
        "annotator": "",
        "reviewer": "",
        "config_confirmed": "NO",
        "annotation_status": "NEEDS_MANUAL_INPUT",
        "notes": "",
    } for row in rows]
    _write_csv(out_dir / "behavior_ground_truth_pilot.csv", gt_rows, gt_fields)
    expected_configs = {f"{row['clip_id']}.yaml" for row in rows}
    for stale_path in configs_dir.glob("*.yaml"):
        if stale_path.name not in expected_configs:
            stale_path.unlink()
    for row in rows:
        (out_dir / "videos_pilot" / row["split"] / row["session_id"]).mkdir(
            parents=True,
            exist_ok=True,
        )
        path = configs_dir / f"{row['clip_id']}.yaml"
        path.write_text(
            yaml.safe_dump(camera_config(row), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    thresholds = {
        "experiment_version": EXPERIMENT_VERSION,
        "source": "config/settings.yaml plus explicit per-clip zone thresholds",
        "thresholds": _thresholds(),
    }
    (out_dir / "pilot_behavior_thresholds.yaml").write_text(
        yaml.safe_dump(thresholds, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    runtime_hashes = {
        relative: _sha256(ROOT / relative)
        for relative in RUNTIME_FILES
        if (ROOT / relative).is_file()
    }
    freeze = {
        "experiment_version": EXPERIMENT_VERSION,
        "pilot_scope": "single-participant deadline-safe pilot; not full-scale validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_worktree_clean": _git(["status", "--porcelain"]) == "",
        "hashes_authoritative_when_worktree_dirty": True,
        "python_version_at_generation": platform.python_version(),
        "production_behavior_code_modified_for_pilot": False,
        "historical_logs_allowed": False,
        "participant_design": {
            "participants": [PARTICIPANT],
            "participant_generalization_claimed": False,
            "participant_disjoint_split_claimed": False,
            "grouping_unit": "recording session",
            "development_sessions": 4,
            "test_sessions": 4,
            "validation_split": "none; thresholds are frozen and not tuned",
        },
        "semantics": {
            "intrusion": "Production alert type intrusion for confirmed-person line crossing direction IN only.",
            "line_crossing": "Evaluation-only counter-delta instrumentation for both IN and OUT; pilot positives use OUT.",
            "polygon_intrusion": "Inactive: IntrusionDetector.analyze does not call _analyze_zones.",
        },
        "runtime_file_sha256": runtime_hashes,
    }
    freeze_path = out_dir.parent / "PILOT_EXPERIMENT_FREEZE.yaml"
    freeze_path.write_text(
        yaml.safe_dump(freeze, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments/priority_19_08/manual",
    )
    args = parser.parse_args()
    rows = write_plan(args.out_dir)
    total_seconds = sum(int(row["target_duration_seconds"]) for row in rows)
    print(f"clips={len(rows)} total_seconds={total_seconds} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
