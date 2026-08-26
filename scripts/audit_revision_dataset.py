"""Audit revision data without inventing labels or metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.behavior_learning import FEATURE_NAMES
from scripts.train_behavior_classifier import label_to_target


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=ROOT / "data/behavior_events.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/behavior_labels.csv")
    parser.add_argument("--database", type=Path, default=ROOT / "data/sct_camera.db")
    parser.add_argument("--video", action="append", type=Path, default=[])
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    return parser.parse_args()


def _video_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"path": str(path), "exists": True, "readable": False}
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return {
            "path": str(path.resolve()),
            "exists": True,
            "readable": True,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(fps, 3),
            "frame_count": frame_count,
            "duration_seconds": round(frame_count / fps, 3) if fps > 0 else None,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    finally:
        capture.release()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database_counts(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in ("alerts", "behavior_events", "behavior_labels", "video_clips", "alert_clips")
            if table in tables
        }
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_counts": counts,
    }


def audit(
    events_path: Path,
    labels_path: Path,
    videos: list[Path],
    database_path: Path | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    invalid_json_lines: list[int] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines.append(line_number)
                continue
            if isinstance(item, dict):
                records.append(item)

    ids = [str(item.get("event_id", "")).strip() for item in records]
    id_counts = Counter(event_id for event_id in ids if event_id)
    alert_types = Counter(str(item.get("alert_type", "unknown")) for item in records)
    cameras = Counter(str(item.get("camera_id", "unknown")) for item in records)
    camera_alert_types = Counter(
        (str(item.get("camera_id", "unknown")), str(item.get("alert_type", "unknown")))
        for item in records
    )
    timestamps = sorted(str(item.get("timestamp", "")) for item in records if item.get("timestamp"))
    embedded_labels = Counter(
        str(item.get("label") or "").strip().lower()
        for item in records
        if str(item.get("label") or "").strip()
    )

    feature_rows = 0
    complete_feature_rows = 0
    missing_features: Counter[str] = Counter()
    observed_features: set[str] = set()
    feature_stats: dict[str, dict[str, float | int | None]] = {
        name: {"count": 0, "nonzero": 0, "min": None, "max": None, "sum": 0.0}
        for name in FEATURE_NAMES
    }
    feature_vectors: Counter[tuple[float, ...]] = Counter()
    for item in records:
        features = item.get("features")
        if not isinstance(features, dict):
            continue
        feature_rows += 1
        observed_features.update(str(name) for name in features)
        missing = [name for name in FEATURE_NAMES if name not in features]
        if not missing:
            complete_feature_rows += 1
        missing_features.update(missing)
        vector: list[float] = []
        for name in FEATURE_NAMES:
            value = float(features.get(name, 0.0))
            vector.append(value)
            stats = feature_stats[name]
            stats["count"] = int(stats["count"]) + 1
            stats["nonzero"] = int(stats["nonzero"]) + int(value != 0.0)
            stats["min"] = value if stats["min"] is None else min(float(stats["min"]), value)
            stats["max"] = value if stats["max"] is None else max(float(stats["max"]), value)
            stats["sum"] = float(stats["sum"]) + value
        feature_vectors[tuple(vector)] += 1

    for stats in feature_stats.values():
        count = int(stats["count"])
        stats["mean"] = round(float(stats.pop("sum")) / count, 8) if count else None

    label_rows: list[dict[str, str]] = []
    if labels_path.is_file():
        with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
            label_rows = [dict(row) for row in csv.DictReader(handle)]
    external_by_id = {
        str(row.get("event_id", "")).strip(): str(row.get("label", "")).strip()
        for row in label_rows
        if str(row.get("event_id", "")).strip()
    }
    external_labels = Counter(label.lower() for label in external_by_id.values() if label)
    record_ids = set(id_counts)
    positive = negative = unrecognized = 0
    for item in records:
        event_id = str(item.get("event_id", "")).strip()
        label = external_by_id.get(event_id) or str(item.get("label") or "").strip()
        target = label_to_target(label)
        if target == 1:
            positive += 1
        elif target == 0:
            negative += 1
        elif label:
            unrecognized += 1

    return {
        "schema_version": 1,
        "events": {
            "path": str(events_path.resolve()),
            "size_bytes": events_path.stat().st_size,
            "sha256": _sha256(events_path),
            "records": len(records),
            "unique_event_ids": len(id_counts),
            "missing_event_ids": sum(not event_id for event_id in ids),
            "duplicate_event_ids": sum(count > 1 for count in id_counts.values()),
            "duplicate_records_beyond_first": sum(count - 1 for count in id_counts.values()),
            "invalid_json_lines": invalid_json_lines,
            "timestamp_min": timestamps[0] if timestamps else None,
            "timestamp_max": timestamps[-1] if timestamps else None,
            "alert_type_counts": dict(alert_types.most_common()),
            "camera_counts": dict(cameras.most_common()),
            "camera_alert_type_counts": {
                camera_id: {
                    alert_type: camera_alert_types[(camera_id, alert_type)]
                    for alert_type in sorted({kind for camera, kind in camera_alert_types if camera == camera_id})
                }
                for camera_id in sorted(cameras)
            },
            "embedded_label_counts": dict(embedded_labels.most_common()),
        },
        "external_labels": {
            "path": str(labels_path.resolve()),
            "size_bytes": labels_path.stat().st_size if labels_path.is_file() else None,
            "sha256": _sha256(labels_path) if labels_path.is_file() else None,
            "rows": len(label_rows),
            "unique_event_ids": len(external_by_id),
            "blank_rows": sum(not label for label in external_by_id.values()),
            "nonblank_rows": sum(bool(label) for label in external_by_id.values()),
            "label_counts": dict(external_labels.most_common()),
            "orphan_event_ids": len(set(external_by_id) - record_ids),
            "events_without_label_row": len(record_ids - set(external_by_id)),
        },
        "risk_dataset": {
            "recognized_labeled_samples": positive + negative,
            "positive_samples": positive,
            "negative_samples": negative,
            "unrecognized_labeled_samples": unrecognized,
            "model_file_exists": (ROOT / "models/behavior_classifier.npz").is_file(),
            "evaluation_feasible": positive > 0 and negative > 0,
        },
        "features": {
            "expected_count": len(FEATURE_NAMES),
            "expected_names": FEATURE_NAMES,
            "observed_names": sorted(observed_features),
            "rows_with_feature_mapping": feature_rows,
            "rows_complete_22": complete_feature_rows,
            "missing_feature_counts": dict(missing_features),
            "value_statistics": feature_stats,
            "distinct_feature_vectors": len(feature_vectors),
            "records_in_repeated_feature_vectors": sum(
                count for count in feature_vectors.values() if count > 1
            ),
            "duplicate_feature_records_beyond_first": sum(
                count - 1 for count in feature_vectors.values() if count > 1
            ),
        },
        "videos": [_video_metadata(path) for path in videos],
        "database": _database_counts(database_path) if database_path is not None else None,
    }


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for field in ("records", "unique_event_ids", "duplicate_event_ids", "duplicate_records_beyond_first"):
        rows.append({"section": "events", "item": field, "value": result["events"][field]})
    for item, count in result["events"]["alert_type_counts"].items():
        rows.append({"section": "alert_type", "item": item, "value": count})
    for item, count in result["events"]["camera_counts"].items():
        rows.append({"section": "camera", "item": item, "value": count})
    for field, value in result["risk_dataset"].items():
        rows.append({"section": "risk_dataset", "item": field, "value": value})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "item", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _args()
    result = audit(args.events, args.labels, args.video, args.database)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(args.csv_out, result)
    risk = result["risk_dataset"]
    print(
        f"events={result['events']['records']} unique={result['events']['unique_event_ids']} "
        f"labeled={risk['recognized_labeled_samples']} positive={risk['positive_samples']} "
        f"negative={risk['negative_samples']}"
    )
    print(f"wrote={args.json_out}")
    print(f"wrote={args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
