"""Freeze the P02 reviewer-expansion run before model inference."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "reviewer_expansion_20260908"
MANIFEST = BASE / "p02_manifest.csv"
GROUND_TRUTH = BASE / "p02_ground_truth.csv"
FREEZE = BASE / "P02_EXPERIMENT_FREEZE.yaml"
RESULTS = BASE / "p02_run"
VERSION = "priority-pilot-v3-two-participant-2026-09-08"

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

# Timings were marked from the frozen geometry overlays before inference.
# Values are in source-video seconds after the configured frame rotation.
POSITIVE_TIMES = {
    "P02_intrusion_positive_r1_daylight_frontal_oblique": (5.0, 7.5, 10.0),
    "P02_loitering_positive_r1_daylight_side_oblique": (0.0, 20.0, 37.0),
    "P02_suspicious_behavior_positive_r1_daylight_frontal_oblique": (0.0, 180.0, 211.4),
    "P02_theft_positive_r1_daylight_side_oblique": (6.5, 21.6, 33.0),
    "P02_line_crossing_positive_r1_daylight_frontal_oblique": (11.0, 13.8, 16.0),
    "P02_intrusion_positive_r2_low_light_side_oblique": (10.5, 13.1, 15.5),
    "P02_loitering_positive_r2_low_light_frontal_oblique": (1.5, 21.5, 30.0),
    "P02_suspicious_behavior_positive_r2_low_light_side_oblique": (0.0, 180.0, 212.5),
    "P02_theft_positive_r2_low_light_frontal_oblique": (6.5, 23.7, 26.5),
    "P02_line_crossing_positive_r2_low_light_side_oblique": (14.0, 17.1, 19.5),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_ground_truth(rows: list[dict[str, str]]) -> None:
    fields = (
        "experiment_version",
        "clip_id",
        "behavior",
        "event_id",
        "start_seconds",
        "eligible_time_seconds",
        "end_seconds",
        "expected_direction",
        "label",
        "split",
        "group_id",
        "participant_id",
        "repetition",
        "session_id",
        "annotator",
        "reviewer",
        "config_confirmed",
        "annotation_status",
        "notes",
    )
    output: list[dict[str, object]] = []
    for row in rows:
        config = yaml.safe_load((ROOT / row["camera_config_path"]).read_text(encoding="utf-8"))
        clip_id = row["clip_id"]
        label = str(config["expected_label"])
        behavior = str(config["expected_behavior"])
        times = POSITIVE_TIMES.get(clip_id)
        if label == "positive" and times is None:
            raise RuntimeError(f"Missing positive timing: {clip_id}")
        start, eligible, end = times if times is not None else ("", "", "")
        output.append(
            {
                "experiment_version": VERSION,
                "clip_id": clip_id,
                "behavior": behavior,
                "event_id": f"gt_{clip_id}_01" if label == "positive" else "",
                "start_seconds": start,
                "eligible_time_seconds": eligible,
                "end_seconds": end,
                "expected_direction": str(config.get("expected_direction") or ""),
                "label": label,
                "split": row["split"],
                "group_id": row["group_id"],
                "participant_id": "P02",
                "repetition": int(config["repetition"]),
                "session_id": str(config["session_id"]),
                "annotator": "AI_visual_QA_pre_inference_2026-09-08",
                "reviewer": "author_scenario_labels_2026-09-08",
                "config_confirmed": "YES",
                "annotation_status": "VERIFIED",
                "notes": (
                    "Pre-inference geometry-overlay timing; author supplied scenario labels. "
                    "Independent human timing review remains required before publication."
                    if label == "positive"
                    else "Author-supplied negative scenario visually checked before inference; no target event."
                ),
            }
        )
    with GROUND_TRUTH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def main() -> None:
    if FREEZE.exists():
        raise RuntimeError(f"Freeze already exists: {FREEZE}")
    if RESULTS.exists() and any(RESULTS.iterdir()):
        raise RuntimeError("Cannot freeze after inference outputs exist")
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20 or any(row["status"] != "READY" for row in rows):
        raise RuntimeError("Expected exactly 20 READY P02 clips")
    if len({row["clip_id"] for row in rows}) != 20:
        raise RuntimeError("P02 clip IDs are not unique")
    if len({row["video_sha256"] for row in rows}) != 20:
        raise RuntimeError("P02 videos are not unique")

    write_ground_truth(rows)
    freeze = {
        "experiment_version": VERSION,
        "scope": "second-participant reviewer expansion using the frozen 20-scenario protocol",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "git_worktree_clean": False,
        "hashes_authoritative_when_worktree_dirty": True,
        "historical_logs_allowed": False,
        "production_behavior_code_modified_for_expansion": False,
        "threshold_tuning_after_p02_inference_allowed": False,
        "participant_design": {
            "participants_in_this_run": ["P02"],
            "participant_disjoint_from_original_pilot": True,
            "original_threshold_development_participant": "P01",
            "grouping_unit": "participant plus recording session",
            "development_sessions": 4,
            "test_sessions": 4,
        },
        "annotation": {
            "performed_before_inference": True,
            "scenario_labels_supplied_by_author": True,
            "timings_marked_from_geometry_overlays": True,
            "independent_human_timing_review_required_before_publication": True,
            "ground_truth_sha256": sha256(GROUND_TRUTH),
        },
        "manifest_sha256": sha256(MANIFEST),
        "video_sha256": {row["clip_id"]: row["video_sha256"] for row in rows},
        "config_sha256": {
            row["clip_id"]: sha256(ROOT / row["camera_config_path"]) for row in rows
        },
        "runtime_file_sha256": {path: sha256(ROOT / path) for path in RUNTIME_FILES},
    }
    FREEZE.write_text(
        yaml.safe_dump(freeze, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"ground_truth={GROUND_TRUTH}")
    print(f"freeze={FREEZE}")


if __name__ == "__main__":
    main()
