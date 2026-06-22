"""Freeze fall-detection dataset files, split, labels, and checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
REQUIRED_COLUMNS = {
    "video",
    "split",
    "event_id",
    "label",
    "start_seconds",
    "end_seconds",
    "camera_id",
    "person_id",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/fall_dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/fall_dataset_manifest.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ground_truth(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], [f"Missing {path.name}"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            return [], [f"ground_truth.csv missing columns: {', '.join(missing)}"]
        rows = [{key: str(value or "").strip() for key, value in row.items()} for row in reader]
    errors: list[str] = []
    for index, row in enumerate(rows, start=2):
        if row["split"] not in {"train", "test"}:
            errors.append(f"row {index}: split must be train/test")
        if row["label"] not in {"fall", "non_fall"}:
            errors.append(f"row {index}: label must be fall/non_fall")
        try:
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
            if start < 0 or end < start:
                raise ValueError
        except ValueError:
            errors.append(f"row {index}: invalid start_seconds/end_seconds")
    return rows, errors


def main() -> int:
    args = _parse_args()
    dataset = args.dataset.resolve()
    files = sorted(
        path
        for split in ("train", "test")
        for path in (dataset / split).rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ) if dataset.exists() else []
    rows, blockers = _read_ground_truth(dataset / "ground_truth.csv")
    relative_files = {path.relative_to(dataset).as_posix() for path in files}
    referenced_files = {row["video"].replace("\\", "/") for row in rows}
    missing_videos = sorted(referenced_files - relative_files)
    unlabeled_videos = sorted(relative_files - referenced_files)
    if missing_videos:
        blockers.append(f"Ground truth references missing videos: {len(missing_videos)}")
    if unlabeled_videos:
        blockers.append(f"Videos without ground truth: {len(unlabeled_videos)}")

    test_counts = Counter(row["label"] for row in rows if row["split"] == "test")
    if test_counts["fall"] < 30:
        blockers.append(f"Need >=30 test fall events; found {test_counts['fall']}")
    if test_counts["non_fall"] < 50:
        blockers.append(f"Need >=50 test non-fall events; found {test_counts['non_fall']}")
    for field in ("camera_id", "person_id"):
        train_values = {row[field] for row in rows if row["split"] == "train" and row[field]}
        test_values = {row[field] for row in rows if row["split"] == "test" and row[field]}
        overlap = sorted(train_values & test_values)
        if overlap:
            blockers.append(f"{field} overlap between train/test: {', '.join(overlap)}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset),
        "status": "frozen" if not blockers else "blocked",
        "ground_truth_format": sorted(REQUIRED_COLUMNS),
        "acceptance": {"test_fall_min": 30, "test_non_fall_min": 50},
        "counts": {
            "videos": len(files),
            "rows": len(rows),
            "test_fall": test_counts["fall"],
            "test_non_fall": test_counts["non_fall"],
        },
        "blockers": blockers,
        "files": [
            {
                "path": path.relative_to(dataset).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "ground_truth": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest status: {manifest['status']}")
    print(f"Wrote {args.output}")
    for blocker in blockers:
        print(f"BLOCKED: {blocker}")
    return 0 if not blockers or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
