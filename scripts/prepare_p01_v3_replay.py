"""Prepare a same-runtime P01 replay for the two-participant reviewer expansion."""

from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "reviewer_expansion_20260908"
P01_BASE = ROOT / "experiments" / "priority_19_08" / "manual"
P01_MANIFEST_SOURCE = P01_BASE / "pilot_manifest.csv"
P01_GROUND_TRUTH_SOURCE = P01_BASE / "behavior_ground_truth_pilot.csv"
P02_MANIFEST = BASE / "p02_manifest.csv"
P02_GROUND_TRUTH = BASE / "p02_ground_truth.csv"
P02_FREEZE = BASE / "P02_EXPERIMENT_FREEZE.yaml"
CONFIG_ROOT = BASE / "configs_p01_v3"
P01_MANIFEST = BASE / "p01_v3_manifest.csv"
P01_GROUND_TRUTH = BASE / "p01_v3_ground_truth.csv"
COMBINED_MANIFEST = BASE / "combined_v3_manifest.csv"
COMBINED_GROUND_TRUTH = BASE / "combined_v3_ground_truth.csv"
FREEZE = BASE / "P01_V3_REPLAY_FREEZE.yaml"
RESULTS = BASE / "p01_v3_run"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def prepare_p01_manifest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        clip_id = row["clip_id"]
        video_path = ROOT / row["video_path"]
        old_config_path = ROOT / row["camera_config_path"]
        if not video_path.is_file() or not old_config_path.is_file():
            raise RuntimeError(f"Missing P01 input for {clip_id}")
        if sha256(video_path) != row["video_sha256"]:
            raise RuntimeError(f"P01 video hash changed: {clip_id}")

        config = yaml.safe_load(old_config_path.read_text(encoding="utf-8")) or {}
        config["experiment_version"] = VERSION
        config["session_id"] = f"P01_{config['session_id']}"
        confirmation = config.setdefault("manual_confirmation", {})
        confirmation["basis"] = (
            "Original P01 geometry and labels; replayed under the frozen v3 runtime "
            "for a same-version two-participant analysis."
        )
        new_config_path = CONFIG_ROOT / f"{clip_id}.yaml"
        new_config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        row["experiment_version"] = VERSION
        row["group_id"] = f"P01_{row['group_id']}"
        row["camera_config_path"] = new_config_path.relative_to(ROOT).as_posix()
        prepared.append(row)
    return prepared


def prepare_p01_ground_truth(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        row["experiment_version"] = VERSION
        row["group_id"] = f"P01_{row['group_id']}"
        row["session_id"] = f"P01_{row['session_id']}"
        row["notes"] = (
            f"{row.get('notes', '').strip()} Reused unchanged for the same-runtime P01 replay; "
            "independent human timing review remains required before publication."
        ).strip()
        prepared.append(row)
    return prepared


def main() -> None:
    required = (
        P01_MANIFEST_SOURCE,
        P01_GROUND_TRUTH_SOURCE,
        P02_MANIFEST,
        P02_GROUND_TRUTH,
        P02_FREEZE,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required inputs: {missing}")
    if FREEZE.exists():
        raise RuntimeError(f"Replay freeze already exists: {FREEZE}")
    if RESULTS.exists() and any(RESULTS.iterdir()):
        raise RuntimeError("Cannot freeze P01 replay after its inference outputs exist")

    p01_source_manifest = read_csv(P01_MANIFEST_SOURCE)
    p01_source_gt = read_csv(P01_GROUND_TRUTH_SOURCE)
    p02_manifest = read_csv(P02_MANIFEST)
    p02_gt = read_csv(P02_GROUND_TRUTH)
    if len(p01_source_manifest) != 20 or len(p01_source_gt) != 20:
        raise RuntimeError("Expected exactly 20 P01 manifest and ground-truth rows")
    if len(p02_manifest) != 20 or len(p02_gt) != 20:
        raise RuntimeError("Expected exactly 20 P02 manifest and ground-truth rows")
    if any(row.get("status") != "READY" for row in p01_source_manifest + p02_manifest):
        raise RuntimeError("All 40 clips must be READY")

    p01_manifest = prepare_p01_manifest(p01_source_manifest)
    p01_gt = prepare_p01_ground_truth(p01_source_gt)
    manifest_fields = list(p01_manifest[0])
    gt_fields = list(p01_gt[0])
    if set(manifest_fields) != set(p02_manifest[0]):
        raise RuntimeError("P01 and P02 manifest schemas differ")
    if set(gt_fields) != set(p02_gt[0]):
        raise RuntimeError("P01 and P02 ground-truth schemas differ")

    write_csv(P01_MANIFEST, p01_manifest, manifest_fields)
    write_csv(P01_GROUND_TRUTH, p01_gt, gt_fields)
    combined_manifest = p01_manifest + p02_manifest
    combined_gt = p01_gt + p02_gt
    write_csv(COMBINED_MANIFEST, combined_manifest, manifest_fields)
    write_csv(COMBINED_GROUND_TRUTH, combined_gt, gt_fields)

    if len({row["clip_id"] for row in combined_manifest}) != 40:
        raise RuntimeError("Combined clip IDs are not unique")
    if len({row["video_sha256"] for row in combined_manifest}) != 40:
        raise RuntimeError("Combined videos are not unique")

    freeze = {
        "experiment_version": VERSION,
        "scope": "P01 same-runtime replay for a harmonized P01+P02 analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "git_worktree_clean": False,
        "hashes_authoritative_when_worktree_dirty": True,
        "historical_logs_allowed": False,
        "production_behavior_code_modified_for_replay": False,
        "threshold_tuning_after_p02_inference_allowed": False,
        "methodology": {
            "p02_component_frozen_before_p02_inference": True,
            "p02_freeze_path": P02_FREEZE.relative_to(ROOT).as_posix(),
            "p02_freeze_sha256": sha256(P02_FREEZE),
            "this_freeze_created_before_p01_v3_replay": True,
            "p01_geometry_and_ground_truth_reused_without_timing_changes": True,
            "same_runtime_required_for_combined_metrics": True,
            "independent_human_timing_review_required_before_publication": True,
        },
        "participant_design": {
            "participants": ["P01", "P02"],
            "clips_per_participant": 20,
            "combined_clips": 40,
            "combined_test_clips": 20,
            "combined_development_clips": 20,
        },
        "p01_manifest_sha256": sha256(P01_MANIFEST),
        "p01_ground_truth_sha256": sha256(P01_GROUND_TRUTH),
        "combined_manifest_sha256": sha256(COMBINED_MANIFEST),
        "combined_ground_truth_sha256": sha256(COMBINED_GROUND_TRUTH),
        "video_sha256": {row["clip_id"]: row["video_sha256"] for row in combined_manifest},
        "config_sha256": {
            row["clip_id"]: sha256(ROOT / row["camera_config_path"])
            for row in combined_manifest
        },
        "runtime_file_sha256": {path: sha256(ROOT / path) for path in RUNTIME_FILES},
    }
    FREEZE.write_text(
        yaml.safe_dump(freeze, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"p01_manifest={P01_MANIFEST}")
    print(f"combined_manifest={COMBINED_MANIFEST}")
    print(f"freeze={FREEZE}")


if __name__ == "__main__":
    main()
