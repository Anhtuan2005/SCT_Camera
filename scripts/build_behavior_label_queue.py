"""Build a deterministic, alert-type-stratified queue for human event review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/behavior_events.jsonl"))
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _even_sample(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    items = sorted(items, key=lambda item: str(item.get("timestamp", "")))
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def main() -> int:
    args = _args()
    if args.per_type < 1:
        raise ValueError("--per-type must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.events.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            grouped[str(item.get("alert_type", "unknown"))].append(item)

    rows: list[dict[str, Any]] = []
    for alert_type, items in sorted(grouped.items()):
        for item in _even_sample(items, args.per_type):
            alert = item.get("alert") if isinstance(item.get("alert"), dict) else {}
            rows.append(
                {
                    "event_id": item.get("event_id", ""),
                    "timestamp": item.get("timestamp", ""),
                    "camera_id": item.get("camera_id", ""),
                    "camera_name": item.get("camera_name", ""),
                    "alert_type": alert_type,
                    "track_id": item.get("track_id", ""),
                    "zone_id": item.get("zone_id", ""),
                    "line_id": item.get("line_id", ""),
                    "details": alert.get("details", ""),
                    "label": "",
                    "behavior_correct": "",
                    "video_evidence_path": "",
                    "evidence_start_seconds": "",
                    "evidence_end_seconds": "",
                    "source_video": "",
                    "scene_id": "",
                    "participant_id": "",
                    "group_id": "",
                    "annotator": "",
                    "notes": "",
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} alert_types={len(grouped)} wrote={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
